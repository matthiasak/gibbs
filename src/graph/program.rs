//! Execution of placed computational graphs: one primary device per program, plus (on Apple
//! hosts) Neural Engine regions split off it when they pass the measured gate ([`gate`]).

use crate::graph::passes::fusion::FuseWeights;
use crate::graph::placement::{
    device_supports_op, place_graph, placement_rows, PlacementPlan, PlacementRow, Secondary,
};
use crate::graph::{cpu, DType, Graph, NodeId, Op, StateStore, Tensor, ValueId, ValueKind, Weights};
use crate::hardware::{available_devices, Device, PlacementMode};
use std::collections::HashMap;

/// The program's executable form on one device.
pub enum CompiledSegment {
    Cpu {
        graph: Graph,
        weights: Weights,
    },
    #[cfg(target_os = "macos")]
    Metal {
        graph: Graph,
        session: std::sync::Arc<crate::graph::metal::Session>,
        plan: crate::graph::metal::PreparedPlan,
    },
    /// A region lowered to one ANE program; `tables` holds the `inv_freq` weights its host-side
    /// Rope tables read.
    #[cfg(target_os = "macos")]
    Ane {
        graph: Graph,
        program: crate::graph::ane::AneProgram,
        io: crate::graph::ane::IoLayout,
        tables: Weights,
    },
    /// CUDA, ROCm or Intel XPU: the portable executor.
    Portable {
        device: Device,
        graph: Graph,
        session: crate::graph::portable::PortableSession,
    },
    Xla {
        graph: Graph,
        session: crate::graph::xla::XlaSession,
    },
}

impl CompiledSegment {
    pub fn device(&self) -> Device {
        match self {
            Self::Cpu { .. } => Device::Cpu,
            #[cfg(target_os = "macos")]
            Self::Metal { .. } => Device::Metal,
            #[cfg(target_os = "macos")]
            Self::Ane { .. } => Device::Ane,
            Self::Portable { device, .. } => *device,
            Self::Xla { .. } => Device::Xla,
        }
    }

    pub fn graph(&self) -> &Graph {
        match self {
            Self::Cpu { graph, .. } | Self::Portable { graph, .. } | Self::Xla { graph, .. } => {
                graph
            }
            #[cfg(target_os = "macos")]
            Self::Metal { graph, .. } | Self::Ane { graph, .. } => graph,
        }
    }

    /// Runs the segment's graph with `inputs` bound to its inputs inside `state`.
    fn run(&mut self, state: &mut StateStore, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, String> {
        match self {
            Self::Cpu { graph, weights } => cpu::run(graph, weights, state, inputs),
            Self::Portable { session, .. } => session.run(state, inputs),
            Self::Xla { graph, session } => session.run(graph, state, inputs),
            #[cfg(target_os = "macos")]
            Self::Metal {
                graph,
                session,
                plan,
            } => session.run_prepared(graph, plan, Some(state), inputs),
            #[cfg(target_os = "macos")]
            Self::Ane {
                graph,
                program,
                io,
                tables,
            } => crate::graph::ane::run(program, io, graph, tables, &inputs),
        }
    }
}

/// One segment of a heterogeneous program: its compiled fragment and the program values bound,
/// in order, to the fragment's inputs and outputs.
pub struct SegmentExec {
    pub compiled: CompiledSegment,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}

/// Runs `segments` in order: `inputs` bind to `graph_inputs`, each segment reads the values
/// earlier segments (or the inputs) produced, and `graph_outputs` are returned.
fn run_segments(
    segments: &mut [SegmentExec],
    state: &mut StateStore,
    graph_inputs: &[ValueId],
    graph_outputs: &[ValueId],
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>, String> {
    let mut vals: HashMap<ValueId, Tensor> = graph_inputs.iter().copied().zip(inputs).collect();
    let take = |vals: &HashMap<ValueId, Tensor>, v: &ValueId| {
        vals.get(v)
            .cloned()
            .ok_or_else(|| format!("segment value {v:?} was never produced"))
    };
    for seg in segments {
        let ins = seg
            .inputs
            .iter()
            .map(|v| take(&vals, v))
            .collect::<Result<Vec<_>, _>>()?;
        let outs = seg.compiled.run(state, ins)?;
        vals.extend(seg.outputs.iter().copied().zip(outs));
    }
    graph_outputs.iter().map(|v| take(&vals, v)).collect()
}

/// `nodes` of `graph` as a standalone graph: operands produced outside become inputs (first-use
/// order), weights and states keep their names, and its outputs are the values a node outside
/// `nodes` or `graph.outputs` reads. Returns it with the `graph` values bound to its inputs and
/// outputs.
fn fragment(graph: &Graph, nodes: &[NodeId]) -> (Graph, Vec<ValueId>, Vec<ValueId>) {
    let inside: std::collections::HashSet<u32> = nodes.iter().map(|n| n.0).collect();
    let mut read_outside: std::collections::HashSet<ValueId> =
        graph.outputs.iter().copied().collect();
    for (i, n) in graph.nodes.iter().enumerate() {
        if !inside.contains(&(i as u32)) {
            read_outside.extend(n.inputs.iter().copied());
        }
    }
    let mut g = Graph::new();
    let mut map: HashMap<ValueId, ValueId> = HashMap::new();
    let (mut ins, mut outs) = (Vec::new(), Vec::new());
    for &n in nodes {
        let node = graph.node(n);
        let mut operands = Vec::with_capacity(node.inputs.len());
        for &v in &node.inputs {
            let id = *map.entry(v).or_insert_with(|| {
                let id = ValueId(g.values.len() as u32);
                let mut val = graph.value(v).clone();
                if !matches!(val.kind, ValueKind::Weight | ValueKind::State { .. }) {
                    val.kind = ValueKind::Input;
                    g.inputs.push(id);
                    ins.push(v);
                }
                g.values.push(val);
                id
            });
            operands.push(id);
        }
        let nid = NodeId(g.nodes.len() as u32);
        let mut results = Vec::with_capacity(node.outputs.len());
        for &v in &node.outputs {
            let id = ValueId(g.values.len() as u32);
            let mut val = graph.value(v).clone();
            val.kind = ValueKind::Node(nid);
            g.values.push(val);
            map.insert(v, id);
            results.push(id);
            if read_outside.contains(&v) {
                g.outputs.push(id);
                outs.push(v);
            }
        }
        g.nodes.push(crate::graph::Node {
            name: node.name.clone(),
            op: node.op.clone(),
            inputs: operands,
            outputs: results,
            device: node.device,
        });
    }
    (g, ins, outs)
}

/// Whether `graph` is a decode instance (an attention over cached keys for one query row): the
/// ANE's ~0.1 ms per evaluation never pays there.
fn is_decode(graph: &Graph) -> bool {
    graph.nodes.iter().any(|n| {
        matches!(n.op, Op::CachedAttention { .. }) && {
            let s = &graph.value(n.inputs[0]).shape;
            s.len() > 1 && s[..s.len() - 1].iter().product::<usize>() == 1
        }
    })
}

/// The ANE secondary a program compiled under `mode` may offload to: forced by `Pin(Ane)`,
/// gated under `Auto` on a Metal host, otherwise none.
fn secondary_for(mode: PlacementMode, available: &[Device]) -> Option<Secondary> {
    let ane = available.contains(&Device::Ane) && available.contains(&Device::Metal);
    match mode {
        PlacementMode::Pin(Device::Ane) if ane => Some(Secondary {
            device: Device::Ane,
            force: true,
        }),
        PlacementMode::Auto if ane => Some(Secondary {
            device: Device::Ane,
            force: false,
        }),
        _ => None,
    }
}

/// One [`GraphProgram::run_reshaped`] instance shape: the caller's graph canonicalized and fused
/// onto the fused weights the program already binds.
struct ReshapedInstance {
    /// The caller graph's node and output counts (a tapped instance adds outputs, not nodes).
    key: (usize, usize),
    shapes: Vec<Vec<usize>>,
    graph: Graph,
    #[cfg(target_os = "macos")]
    plan: Option<crate::graph::metal::PreparedPlan>,
    /// The instance split into Metal and ANE segments when the gate chose that; empty: `plan`.
    segments: Vec<SegmentExec>,
}

/// An executable graph program with state persistence.
pub struct GraphProgram {
    /// The whole graph on the primary device (its nodes' `device` marks the ANE regions).
    pub segment: CompiledSegment,
    /// The graph split into primary-device and ANE segments when the gate chose that; empty:
    /// [`Self::segment`] runs it.
    pub segments: Vec<SegmentExec>,
    pub plan: PlacementPlan,
    pub primary_device: Device,
    pub state: StateStore,
    /// The pages of the program's paged KV states ([`crate::kv`]); every run attaches a view of it
    /// to [`Self::state`].
    pub kv_pool: Option<std::sync::Arc<crate::kv::KvPool>>,
    pub graph_inputs: Vec<ValueId>,
    pub graph_outputs: Vec<ValueId>,
    /// [`Self::stream_next`]'s argmax graph and plan, with the `valid` they were built for.
    #[cfg(target_os = "macos")]
    argmax: Option<(usize, Graph, crate::graph::metal::PreparedPlan)>,
    /// [`Self::run_reshaped`]'s instances (Metal: with their plans), least recently used first.
    reshaped_graphs: Vec<ReshapedInstance>,
    /// The one-element f32 weights: the constants the canonical reductions of a reshaped
    /// instance test.
    scalars: Weights,
    /// The device reshaped instances may offload regions to.
    secondary: Option<Secondary>,
    /// Host copies of the weights the secondary lowers (it compiles them into its programs), for
    /// reshaped instances; `None` for decode programs under `Auto`.
    // ponytail: duplicates the offloadable weights in host memory for the program's lifetime;
    // read them back from the Metal buffers if that RSS matters.
    ane_weights: Option<Weights>,
    /// Reshaped instance keys whose gate kept every region on Metal: re-planned instances skip
    /// the gate (an evicted instance is rebuilt often).
    ane_metal_only: std::collections::HashSet<((usize, usize), Vec<Vec<usize>>)>,
}

/// The positions a fed stream from `pos` may write without regrowing contiguous state buffers:
/// up to the next 4096-row boundary (their lazy growth step), within their declared rows. Paged
/// states (`kv`) cap nothing: a stream reserves its pages up front.
#[cfg(target_os = "macos")]
fn stream_cap(
    g: &Graph,
    plan: &crate::graph::metal::PreparedPlan,
    kv: Option<&crate::kv::KvPool>,
    pos: u32,
) -> usize {
    let declared = plan
        .state_vars
        .iter()
        .filter(|(_, name)| kv.is_none_or(|p| p.locate(name).is_none()))
        .filter_map(|&(v, _)| {
            let val = g.value(v);
            let rec = matches!(
                val.kind,
                crate::graph::ValueKind::State {
                    recurrent: true,
                    ..
                }
            );
            (!rec && val.shape.len() == 2).then_some(val.shape[0])
        })
        .max();
    let next = (pos as usize + 1).next_multiple_of(4096);
    declared.map_or(next, |d| d.min(next))
}

/// `graph` with output 0 (one f32 row) replaced by the `[.., 1]` u32 index of the first largest of
/// its first `valid` entries (0 = all), NaN ranking lowest: a prefix `Slice` then `TopK { k: 1 }`.
/// The other outputs stay. `None` when output 0 is not one f32 row.
pub(crate) fn argmax_head(graph: &Graph, valid: usize) -> Option<Graph> {
    let &logits = graph.outputs.first()?;
    let v = graph.value(logits);
    let cols = v.shape.last().copied().filter(|&c| c > 0)?;
    if v.dtype != DType::F32 || v.shape.iter().product::<usize>() != cols {
        return None;
    }
    let valid = if valid == 0 { cols } else { valid.min(cols) };
    let mut shape = v.shape.clone();
    let mut g = graph.clone();
    let row = if valid < cols {
        *shape.last_mut()? = valid;
        let slice = Op::Slice {
            axis: -1,
            start: 0,
            end: valid as i64,
        };
        g.add("argmax_valid", slice, &[logits], shape.clone(), DType::F32)
    } else {
        logits
    };
    *shape.last_mut()? = 1;
    g.outputs[0] = g.add("argmax", Op::TopK { k: 1 }, &[row], shape, DType::U32);
    Some(g)
}

/// The error a `GIBBS_PLAN_DUMP_CPU` Metal compile ends with once the plan is printed.
pub const PLAN_DUMPED: &str =
    "GIBBS_PLAN_DUMP_CPU: plan printed; compile stopped before any device work";

/// Programs being compiled right now: placement, weight upload, kernel build and tuning.
static COMPILING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Whether any [`GraphProgram::compile`] is running (a model load shows it as its compile stage).
pub fn compiling() -> bool {
    COMPILING.load(std::sync::atomic::Ordering::Relaxed) > 0
}

/// Total CPU-segment nodes compiled across all non-`Pin(Cpu)` [`GraphProgram`]s in this process.
static COMPILED_CPU_NODES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn compiled_cpu_nodes() -> usize {
    COMPILED_CPU_NODES.load(std::sync::atomic::Ordering::Relaxed)
}

struct CompileGuard;

impl CompileGuard {
    fn enter() -> Self {
        COMPILING.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self
    }
}

impl Drop for CompileGuard {
    fn drop(&mut self) {
        COMPILING.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

impl GraphProgram {
    /// Compiles a Graph and Weights into a single-device GraphProgram under `mode`.
    ///
    /// Fails with the avenue report when `mode` targets the GPU and a node has no GPU lowering.
    /// `GIBBS_PLAN_DUMP_CPU=<ctx>`: a Metal compile prints [`crate::graph::metal::plan_dump`] at
    /// context length `ctx` (`1` when not a number) to stdout and fails with [`PLAN_DUMPED`]
    /// before any device, weight upload or tuning.
    pub fn compile(graph: Graph, weights: Weights, mode: PlacementMode) -> Result<Self, String> {
        Self::compile_with(graph, weights, mode, false)
    }

    /// [`Self::compile`] for a network that also runs other instance shapes through
    /// [`Self::run_reshaped`]: weights the fusion passes fold into fused tensors (q/k/v, gate/up
    /// concatenations) stay bound beside them, so instances whose fusions are rejected still find
    /// them.
    // ponytail: folded originals stay resident next to their fused copy (only unpacked f32/f16
    // tensors fold) for instances that fuse differently; drop them if every instance fuses alike.
    pub fn compile_reshapable(
        graph: Graph,
        weights: Weights,
        mode: PlacementMode,
    ) -> Result<Self, String> {
        Self::compile_with(graph, weights, mode, true)
    }

    fn compile_with(
        mut graph: Graph,
        mut weights: Weights,
        mode: PlacementMode,
        keep_folded: bool,
    ) -> Result<Self, String> {
        let _compiling = CompileGuard::enter();
        #[cfg(test)]
        crate::graph::shape::note(&graph);
        crate::graph::passes::prefuse(&mut graph);
        let mut scalars = Weights::new();
        for (name, t) in weights.iter() {
            if matches!(t, Tensor::F32(v) if v.len() == 1) {
                scalars.insert(name, t.clone());
            }
        }
        let unfused = keep_folded.then(|| weights.clone());
        crate::graph::passes::fuse(&mut graph, FuseWeights::Materialize(&mut weights))?;
        crate::graph::passes::canonical::run(&mut graph, &scalars);
        let dump: Option<usize> = std::env::var("GIBBS_PLAN_DUMP_CPU")
            .ok()
            .map(|v| v.parse().unwrap_or(1));
        let available = if dump.is_some() {
            vec![Device::Cpu, Device::Metal]
        } else {
            available_devices()
        };
        let secondary = secondary_for(mode, &available).filter(|_| dump.is_none());
        // `--backend ane`: Metal runs what the ANE does not.
        let primary_mode = match secondary {
            Some(s) if s.force => PlacementMode::Pin(Device::Metal),
            _ => mode,
        };
        let offload = secondary.filter(|_| !is_decode(&graph));
        #[allow(unused_mut)]
        let mut plan = place_and_adapt(&mut graph, primary_mode, &available, offload)?;
        let ane_weights = secondary
            .filter(|s| keep_folded && (s.force || offload.is_some()))
            .map(|s| {
                let mut w = Weights::new();
                for node in &graph.nodes {
                    if !crate::graph::placement::secondary_supports(s.device, &graph, node) {
                        continue;
                    }
                    for &v in &node.inputs {
                        let val = graph.value(v);
                        if let (ValueKind::Weight, Some(t)) = (&val.kind, weights.get(&val.name)) {
                            w.insert(val.name.clone(), t.clone());
                        }
                    }
                }
                w
            });
        if let Some(unfused) = &unfused {
            for (name, t) in unfused.iter() {
                if weights.get(name).is_none() {
                    weights.insert(name, t.clone());
                }
            }
        }
        let primary_device = plan.primary_device();
        #[cfg(target_os = "macos")]
        if let (Some(ctx), Device::Metal) = (dump, primary_device) {
            print!("{}", crate::graph::metal::plan_dump(&graph, ctx)?);
            return Err(PLAN_DUMPED.into());
        }
        eprintln!(
            "Compiled GraphProgram: {} segments, primary device: {:?}",
            plan.segments.len(),
            primary_device
        );

        let graph_inputs = graph.inputs.clone();
        let graph_outputs = graph.outputs.clone();
        let paged = crate::graph::paged_states(&graph);
        let mut kv_pool = None;
        #[allow(unused_mut)]
        let mut segments = Vec::new();
        let segment = match primary_device {
            #[cfg(target_os = "macos")]
            Device::Metal if !plan.segments.is_empty() => {
                let rt = crate::graph::metal::Runtime::shared()?;
                let active = (!keep_folded).then_some(&graph);
                let mut session =
                    crate::graph::metal::Session::new_for_graph(rt, &weights, active)?;
                if !paged.is_empty() {
                    kv_pool = Some(session.new_kv_pool(&paged)?);
                }
                let session = std::sync::Arc::new(session);
                let prepared_plan = session.prepare_plan(&graph)?;
                if let Some(s) = offload {
                    segments = gate(&mut graph, &mut plan, &session, &weights, s)?;
                }
                CompiledSegment::Metal {
                    graph,
                    session,
                    plan: prepared_plan,
                }
            }
            d if crate::graph::portable::is_portable(d) => {
                let api = crate::graph::portable::device_api(d)
                    .ok_or_else(|| format!("--backend {d}: not available on this host"))?;
                let session =
                    crate::graph::portable::PortableSession::compile(api, &graph, weights)?;
                if !paged.is_empty() {
                    kv_pool = Some(session.new_kv_pool(&paged));
                }
                CompiledSegment::Portable {
                    device: d,
                    graph,
                    session,
                }
            }
            Device::Xla => {
                let session = crate::graph::xla::XlaSession::compile(&graph, weights)?;
                CompiledSegment::Xla { graph, session }
            }
            Device::Cpu => CompiledSegment::Cpu { graph, weights },
            d => return Err(format!("--backend {d}: no executor for it on this host")),
        };

        let kv_pool =
            kv_pool.or_else(|| (!paged.is_empty()).then(|| crate::kv::KvPool::host(&paged)));
        let prog = Self {
            segment,
            segments,
            plan,
            primary_device,
            state: StateStore::new(),
            kv_pool,
            graph_inputs,
            graph_outputs,
            #[cfg(target_os = "macos")]
            argmax: None,
            reshaped_graphs: Vec::new(),
            scalars,
            secondary,
            ane_weights,
            ane_metal_only: Default::default(),
        };
        if mode != PlacementMode::Pin(Device::Cpu) {
            COMPILED_CPU_NODES
                .fetch_add(prog.cpu_node_count(), std::sync::atomic::Ordering::Relaxed);
        }
        Ok(prog)
    }

    /// Number of graph nodes the CPU executor runs: 0 for a Metal program.
    pub fn cpu_node_count(&self) -> usize {
        match &self.segment {
            CompiledSegment::Cpu { graph, .. } => graph.nodes.len(),
            _ => 0,
        }
    }

    /// Gives [`Self::state`] a view of the program's KV pool when it has none (a fresh state);
    /// a view of another program's pool is kept when that pool pages this program's states alike
    /// (separately compiled instances of one network sharing a state), else this errs.
    fn attach_kv(&mut self) -> Result<(), String> {
        let Some(pool) = &self.kv_pool else {
            return Ok(());
        };
        match &self.state.kv {
            None => self.state.kv = Some(crate::kv::KvView::new(pool.clone())),
            Some(v) if !std::sync::Arc::ptr_eq(v.pool(), pool) && !v.pool().covers(pool) => {
                return Err("this state holds another program's KV pages".into())
            }
            Some(_) => {}
        }
        Ok(())
    }

    /// Names of the program's recurrent states (`Graph::state(.., recurrent: true)`): those
    /// [`StateStore::truncate`] refuses to rewind.
    pub fn recurrent_states(&self) -> Vec<String> {
        self.segment.graph().recurrent_states()
    }

    /// Names of the program's states its KV pool does not page (recurrent conv/ssm, `StateWrite`
    /// caches, ...), sorted: what a prefix-cache boundary snapshots.
    pub fn contiguous_states(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .segment
            .graph()
            .values
            .iter()
            .filter_map(|v| match &v.kind {
                crate::graph::ValueKind::State { name, .. }
                    if self.kv_pool.as_ref().is_none_or(|p| p.locate(name).is_none()) =>
                {
                    Some(name.clone())
                }
                _ => None,
            })
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// The program's graph nodes grouped by op kind, device and avenue, most numerous first.
    pub fn placement_summary(&self) -> Vec<PlacementRow> {
        let mut rows: Vec<PlacementRow> = Vec::new();
        for r in placement_rows(self.segment.graph()) {
            match rows
                .iter_mut()
                .find(|x| (&x.op, x.device, &x.avenue) == (&r.op, r.device, &r.avenue))
            {
                Some(x) => x.count += r.count,
                None => rows.push(r),
            }
        }
        rows.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.op.cmp(&b.op)));
        rows
    }

    /// Primary device running this program.
    pub fn primary_device(&self) -> Device {
        self.primary_device
    }

    /// Label describing backend execution.
    pub fn backend_label(&self) -> String {
        match self.primary_device {
            #[cfg(target_os = "macos")]
            Device::Metal => format!(
                "Metal ({})",
                crate::backend::metal::system_device_name()
                    .unwrap_or_else(|_| "Apple GPU".to_string())
            ),
            Device::Cuda => crate::graph::cuda::device_name(),
            Device::Rocm => crate::graph::rocm::device_name(),
            Device::Xla => crate::graph::xla::device_name(),
            Device::IntelXpu => crate::graph::intel_xpu::device_name(),
            other => format!("{other}"),
        }
    }

    /// Computes the exact device memory bytes read per token at absolute context position `pos`.
    pub fn bytes_per_token(&self, pos: usize) -> u64 {
        let mut total = 0u64;
        #[cfg(target_os = "macos")]
        if let CompiledSegment::Metal { plan, .. } = &self.segment {
            total += plan.weight_bytes;
            for sr in &plan.state_reads {
                let visible = (pos + 1).min(sr.window.unwrap_or(usize::MAX)) as u64;
                total += visible * sr.bytes_per_pos + sr.fixed_bytes;
            }
        }
        total
    }

    /// Calculates roofline metrics at context position `pos` given the host ceiling bandwidth in GB/s.
    pub fn roofline(&self, pos: usize, ceiling_gbps: f64) -> (u64, f64, f64) {
        let bytes = self.bytes_per_token(pos);
        let roofline_tok_s = if bytes > 0 && ceiling_gbps > 0.0 {
            ceiling_gbps / (bytes as f64 / 1e9)
        } else {
            0.0
        };
        (bytes, ceiling_gbps, roofline_tok_s)
    }

    /// Whether the program is exactly one Metal segment.
    pub fn is_single_metal(&self) -> bool {
        #[cfg(target_os = "macos")]
        return self.segments.is_empty() && matches!(self.segment, CompiledSegment::Metal { .. });
        #[cfg(not(target_os = "macos"))]
        false
    }

    /// The ANE segments of a reshaped instance `g` (on `session`) when the program has a
    /// secondary, `allow` (or `--backend ane`) admits it and the gate keeps regions; empty
    /// otherwise.
    #[cfg(target_os = "macos")]
    fn instance_segments(
        &self,
        g: &mut Graph,
        session: &std::sync::Arc<crate::graph::metal::Session>,
        allow: bool,
    ) -> Result<Vec<SegmentExec>, String> {
        let (Some(sec), Some(weights)) = (self.secondary, &self.ane_weights) else {
            return Ok(Vec::new());
        };
        if !allow || is_decode(g) {
            return Ok(Vec::new());
        }
        // A secondary exists only on a host with Metal and the ANE: no device probing per instance.
        let (mode, devices) = (PlacementMode::Pin(Device::Metal), [Device::Metal, sec.device]);
        let Ok(mut placed) = place_graph(g, mode, &devices, &device_supports_op, Some(sec))
        else {
            return Ok(Vec::new());
        };
        gate(g, &mut placed, session, weights, sec)
    }

    /// Runs `graph` with the program's weights inside the program's state: another instance shape of
    /// the compiled network (same weight and state names), e.g. a batched prefill pass. Instances
    /// are cached by the graph's node and output counts and input shapes, which identify such a
    /// graph (prefill chunks repeat): on a miss the graph is canonicalized and fused onto the fused
    /// weights the program binds ([`FuseWeights::ReuseOnly`]), and Metal plans it.
    /// `allow_secondary`: the instance may offload ANE regions (gated under `Auto`, forced under
    /// `--backend ane`); callers pass `false` for latency-bound passes such as speculative verify.
    pub fn run_reshaped(
        &mut self,
        graph: &Graph,
        inputs: Vec<Tensor>,
        allow_secondary: bool,
    ) -> Result<Vec<Tensor>, String> {
        #[cfg(test)]
        crate::graph::shape::note(graph);
        self.attach_kv()?;
        let shapes: Vec<Vec<usize>> = graph
            .inputs
            .iter()
            .map(|&v| graph.value(v).shape.clone())
            .collect();
        let key = (graph.nodes.len(), graph.outputs.len());
        let hit = self
            .reshaped_graphs
            .iter()
            .position(|r| r.key == key && r.shapes == shapes);
        // Least recently used first: a varying tail chunk must not evict the full chunk's instance.
        let entry = match hit {
            Some(i) => self.reshaped_graphs.remove(i),
            None => {
                // ponytail: 4 shapes (full chunk, tails, image blocks); more if callers vary
                // their row counts more.
                if self.reshaped_graphs.len() == 4 {
                    self.reshaped_graphs.remove(0);
                }
                let mut g = graph.clone();
                let program = self.segment.graph();
                crate::graph::passes::prefuse(&mut g);
                crate::graph::passes::fuse(&mut g, FuseWeights::ReuseOnly(program))?;
                crate::graph::passes::canonical::run(&mut g, &self.scalars);
                #[cfg(target_os = "macos")]
                let (plan, segments, metal_only) = match &self.segment {
                    CompiledSegment::Metal { session, .. } => {
                        let plan = session.prepare_plan(&g)?;
                        let gated = allow_secondary && !self.ane_metal_only.contains(&(key, shapes.clone()));
                        let segs = self.instance_segments(&mut g, session, gated)?;
                        let metal_only = gated && segs.is_empty();
                        (Some(plan), segs, metal_only)
                    }
                    _ => (None, Vec::new(), false),
                };
                #[cfg(target_os = "macos")]
                if metal_only {
                    self.ane_metal_only.insert((key, shapes.clone()));
                }
                #[cfg(not(target_os = "macos"))]
                let (segments, _) = (Vec::new(), allow_secondary);
                ReshapedInstance {
                    key,
                    shapes,
                    #[cfg(target_os = "macos")]
                    plan,
                    segments,
                    graph: g,
                }
            }
        };
        self.reshaped_graphs.push(entry);
        let last = self.reshaped_graphs.len() - 1;
        let r = &mut self.reshaped_graphs[last];
        if !r.segments.is_empty() {
            let g = &r.graph;
            return run_segments(&mut r.segments, &mut self.state, &g.inputs, &g.outputs, inputs);
        }
        let graph = &r.graph;
        match &mut self.segment {
            CompiledSegment::Cpu { weights, .. } => {
                cpu::run(graph, weights, &mut self.state, inputs)
            }
            CompiledSegment::Portable { session, .. } => {
                session.run_reshaped(graph, &mut self.state, inputs)
            }
            CompiledSegment::Xla { session, .. } => session.run(graph, &mut self.state, inputs),
            #[cfg(target_os = "macos")]
            CompiledSegment::Metal { session, .. } => {
                let plan = r
                    .plan
                    .as_ref()
                    .expect("a Metal program plans its instances");
                session.run_prepared(graph, plan, Some(&mut self.state), inputs)
            }
            #[cfg(target_os = "macos")]
            CompiledSegment::Ane { .. } => unreachable!("the primary segment is never the ANE"),
        }
    }

    /// Executes the graph program with given inputs.
    pub fn run(&mut self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, String> {
        if inputs.len() != self.graph_inputs.len() {
            return Err(format!(
                "program expects {} inputs, got {}",
                self.graph_inputs.len(),
                inputs.len()
            ));
        }
        self.attach_kv()?;
        if !self.segments.is_empty() {
            let (ins, outs) = (&self.graph_inputs, &self.graph_outputs);
            return run_segments(&mut self.segments, &mut self.state, ins, outs, inputs);
        }
        self.segment.run(&mut self.state, inputs)
    }

    /// The next token of a greedy decode stream: `last` is the input token at position `pos` (the
    /// previous call's result after the first), at most `max_new` tokens are queued, and the pick
    /// is the first largest of the first `valid` logits. Each call keeps the next token's command
    /// buffers queued on the device (the picked id feeds the next token there), so the host never
    /// idles the GPU between tokens. `None` when the program cannot stream (then use
    /// [`Self::run`]); a pending one-shot kernel profile ends the stream with `None`.
    /// A stream leaves one speculative token written past the last returned position.
    #[cfg(target_os = "macos")]
    pub fn stream_next(
        &mut self,
        last: u32,
        pos: u32,
        max_new: usize,
        valid: usize,
    ) -> Result<Option<u32>, String> {
        use crate::graph::metal::Session;
        if crate::graph::metal::profile_requested() && self.recurrent_states().is_empty() {
            self.stream_end()?;
            return Ok(None);
        }
        self.attach_kv()?;
        let (true, CompiledSegment::Metal { graph, session, .. }) =
            (self.segments.is_empty(), &self.segment)
        else {
            return Ok(None);
        };
        if let Some(s) = self.state.stream.0.as_deref() {
            if pos != s.next_pos || last != s.last {
                return Err("device stream out of sync".into());
            }
            if s.submitted == s.budget && s.in_flight() == 0 {
                self.state.stream.0 = None;
            }
        }
        if self.state.stream.0.is_none() {
            if graph.outputs.len() != 1
                || std::env::var_os("GIBBS_PROFILE_KERNELS").is_some()
                || crate::graph::disabled("pipeline")
            {
                return Ok(None);
            }
            if self.argmax.as_ref().map(|a| a.0) != Some(valid) {
                self.argmax = match argmax_head(graph, valid) {
                    Some(g) => {
                        let plan = session.prepare_plan(&g)?;
                        Some((valid, g, plan))
                    }
                    None => None,
                };
            }
            let Some((_, g, plan)) = &self.argmax else {
                return Ok(None);
            };
            if !Session::fed_stream_ok(g, plan) {
                return Ok(None);
            }
            let cap = stream_cap(g, plan, self.kv_pool.as_deref(), pos);
            let budget = max_new.min(cap.saturating_sub(pos as usize));
            if budget == 0 {
                return Ok(None);
            }
            let s = session.new_fed_stream(g, plan, &mut self.state, last, pos, budget)?;
            self.state.stream.0 = Some(Box::new(s));
        }
        let (Some((_, g, plan)), Some(s)) = (&self.argmax, self.state.stream.0.as_deref_mut())
        else {
            unreachable!("a stream exists only with its argmax plan");
        };
        while s.in_flight() < 2 && s.submitted < s.budget {
            session.submit_fed(g, plan, s)?;
        }
        let tok = session.wait_fed(s)?;
        s.next_pos += 1;
        s.last = tok;
        Ok(Some(tok))
    }

    /// Like [`Self::stream_next`], supporting both greedy (`temp <= 1e-4`) and device-side
    /// stochastic (`temp > 1e-4`, `top_p`) decoding via the device top-k sampler.
    #[cfg(target_os = "macos")]
    pub fn stream_next_sampled(
        &mut self,
        last: u32,
        pos: u32,
        max_new: usize,
        valid: usize,
        temp: f32,
        top_p: f32,
    ) -> Result<Option<u32>, String> {
        if temp <= 1e-4 {
            return self.stream_next(last, pos, max_new, valid);
        }
        use crate::graph::metal::{SampleGpuSpec, Session};
        if crate::graph::metal::profile_requested() && self.recurrent_states().is_empty() {
            self.stream_end()?;
            return Ok(None);
        }
        self.attach_kv()?;
        if let Some(s) = self.state.stream.0.as_deref() {
            if pos != s.next_pos || last != s.last || s.sample_spec.is_none() {
                self.stream_end()?;
            } else if s.submitted == s.budget && s.in_flight() == 0 {
                self.state.stream.0 = None;
            }
        }
        let (
            true,
            CompiledSegment::Metal {
                graph,
                session,
                plan,
            },
        ) = (self.segments.is_empty(), &self.segment)
        else {
            return Ok(None);
        };
        if self.state.stream.0.is_none() {
            if graph.outputs.len() != 1
                || std::env::var_os("GIBBS_PROFILE_KERNELS").is_some()
                || crate::graph::disabled("pipeline")
                || !Session::fed_stream_ok(graph, plan)
            {
                return Ok(None);
            }
            let Some(&logits) = graph.outputs.first() else {
                return Ok(None);
            };
            let v = graph.value(logits);
            let cols = v.shape.last().copied().unwrap_or(0);
            if v.dtype != DType::F32 || cols == 0 || v.shape.iter().product::<usize>() != cols {
                return Ok(None);
            }
            let vocab = if valid == 0 { cols } else { valid.min(cols) } as u32;
            let cap = stream_cap(graph, plan, self.kv_pool.as_deref(), pos);
            let budget = max_new.min(cap.saturating_sub(pos as usize));
            if budget == 0 {
                return Ok(None);
            }
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0x517cc1b727220a95, |d| d.as_nanos() as u64)
                ^ ((pos as u64) << 32)
                ^ (last as u64);
            let sp = SampleGpuSpec {
                vocab,
                top_k: crate::scheduler::SAMPLE_TOP_K as u32,
                inv_temp: 1.0 / temp.max(1e-4),
                top_p: top_p.clamp(1e-4, 1.0),
                rng_seed: seed,
            };
            let s = session.new_fed_stream_with_sample(
                graph,
                plan,
                &mut self.state,
                last,
                pos,
                budget,
                Some(sp),
            )?;
            self.state.stream.0 = Some(Box::new(s));
        }
        let CompiledSegment::Metal {
            graph,
            session,
            plan,
        } = &self.segment
        else {
            return Ok(None);
        };
        let s = self.state.stream.0.as_deref_mut().expect("allocated above");
        while s.in_flight() < 2 && s.submitted < s.budget {
            session.submit_fed(graph, plan, s)?;
        }
        let tok = session.wait_fed(s)?;
        s.next_pos += 1;
        s.last = tok;
        Ok(Some(tok))
    }

    /// Ends this state's greedy decode stream: waits for its in-flight tokens and drops it.
    /// `true` when draining discarded a submitted token that [`Self::stream_next`] never returned.
    #[cfg(target_os = "macos")]
    pub fn stream_end(&mut self) -> Result<bool, String> {
        let mut discarded = false;
        if let Some(mut s) = self.state.stream.0.take() {
            discarded = s.in_flight() > 0;
            if let CompiledSegment::Metal { session, .. } = &self.segment {
                session.drain_fed(&mut s);
            }
        }
        Ok(discarded)
    }

    #[cfg(not(target_os = "macos"))]
    pub fn stream_next(
        &mut self,
        _last: u32,
        _pos: u32,
        _max_new: usize,
        _valid: usize,
    ) -> Result<Option<u32>, String> {
        Ok(None)
    }

    #[cfg(not(target_os = "macos"))]
    pub fn stream_next_sampled(
        &mut self,
        _last: u32,
        _pos: u32,
        _max_new: usize,
        _valid: usize,
        _temp: f32,
        _top_p: f32,
    ) -> Result<Option<u32>, String> {
        Ok(None)
    }

    #[cfg(not(target_os = "macos"))]
    pub fn stream_end(&mut self) -> Result<bool, String> {
        Ok(false)
    }
}

/// Places `graph` (offloading regions to `secondary` when given) and, on Metal, adapts its dtypes
/// to the Metal kernels.
fn place_and_adapt(
    graph: &mut Graph,
    mode: PlacementMode,
    available: &[Device],
    secondary: Option<Secondary>,
) -> Result<PlacementPlan, String> {
    let plan = place_graph(graph, mode, available, &device_supports_op, secondary)?;
    let device = plan.primary_device();
    if device == Device::Metal || crate::graph::portable::is_portable(device) {
        crate::graph::passes::dtype_adapt::run(graph);
        crate::graph::passes::dtype_adapt::retype_uncoded_weights(graph, &|f| {
            device_supports_op(device, &Op::Linear, DType::Quant(f))
        })?;
    }
    Ok(plan)
}

/// Relative RMS error every output of an ANE region may show against Metal's.
const ANE_REL_ERR: f32 = 2e-2;
/// A region's ANE median must be at most this fraction of Metal's to be kept (`Auto`).
const ANE_SPEEDUP: f64 = 0.95;

/// A measured ANE offload decision for one region signature ([`gate`]).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AneDecision {
    /// Median of 3 timed runs of the region on Metal and on the ANE (host boundary copies
    /// included).
    pub metal_ms: f64,
    pub ane_ms: f64,
    /// Largest relative RMS error of an ANE output against Metal's (non-finite stored as
    /// `f32::MAX`).
    pub rel_err: f32,
    /// `ane_ms <= 0.95 × metal_ms` and `rel_err <= 2e-2`: what `Auto` does.
    pub chosen: bool,
}

/// One gated ANE region: its signature, extent and decision.
#[derive(Debug, Clone)]
pub struct AneReport {
    pub signature: u64,
    /// `{n} nodes: {first} … {last}`.
    pub region: String,
    pub decision: AneDecision,
    /// This process measured it (else the host cache held it).
    pub measured: bool,
    /// The program runs the region on the ANE.
    pub offloaded: bool,
}

impl std::fmt::Display for AneReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let d = &self.decision;
        write!(
            f,
            "ANE region {:016x} ({}): Metal {:.3} ms, ANE {:.3} ms, rel err {:.2e}{} -> {}",
            self.signature,
            self.region,
            d.metal_ms,
            d.ane_ms,
            d.rel_err,
            if self.measured { "" } else { " (cached)" },
            if self.offloaded { "ANE" } else { "Metal" }
        )
    }
}

static ANE_REPORTS: std::sync::Mutex<Vec<AneReport>> = std::sync::Mutex::new(Vec::new());

/// Every ANE gate decision this process made or reused, in order.
pub fn ane_reports() -> Vec<AneReport> {
    ANE_REPORTS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Decisions by signature (hex) from `<cache>/host/ane-decisions-<os build>.json`.
static ANE_DECISIONS: std::sync::Mutex<Option<std::collections::BTreeMap<String, AneDecision>>> =
    std::sync::Mutex::new(None);

fn ane_decisions_path() -> Option<std::path::PathBuf> {
    let os = crate::hardware::sysctl::string(c"kern.osversion").unwrap_or_default();
    let dir = crate::hardware::default_cache_dir().ok()?.join("host");
    Some(dir.join(format!("ane-decisions-{os}.json")))
}

/// Runs `f` over the persisted decisions (loaded on first use); `store` writes them back.
fn with_ane_decisions<R>(
    store: bool,
    f: impl FnOnce(&mut std::collections::BTreeMap<String, AneDecision>) -> R,
) -> R {
    let mut slot = ANE_DECISIONS.lock().unwrap_or_else(|e| e.into_inner());
    let map = slot.get_or_insert_with(|| {
        ane_decisions_path()
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    });
    let r = f(map);
    if let (true, Some(path)) = (store, ane_decisions_path()) {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(json) = serde_json::to_vec_pretty(map) {
            let _ = std::fs::write(path, json);
        }
    }
    r
}

/// FNV-64 over the node ops, devices, operand shapes and dtypes and weight names of `graph`, and
/// the ANE lowering source (an edited lowering re-measures).
fn signature(graph: &Graph) -> u64 {
    use std::fmt::Write;
    let mut s = String::new();
    for n in &graph.nodes {
        let _ = write!(s, "{:?}|{:?}|", n.op, n.device);
        for &v in n.inputs.iter().chain(&n.outputs) {
            let val = graph.value(v);
            let _ = write!(s, "{:?}{:?}", val.shape, val.dtype);
            if val.kind == ValueKind::Weight {
                s += &val.name;
            }
            s.push(';');
        }
    }
    let lowering = include_bytes!("ane/mil.rs");
    crate::graph::ane::fnv64(&[s.as_bytes(), lowering])
}

/// Deterministic inputs for `graph`: standard-normal f32, and u32 `iota` clamped below the rows
/// of every embedding table the input indexes.
fn synth_inputs(graph: &Graph) -> Vec<Tensor> {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut uniform = move || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    graph
        .inputs
        .iter()
        .map(|&v| {
            let val = graph.value(v);
            let n: usize = val.shape.iter().product();
            if val.dtype == DType::U32 {
                let rows = graph
                    .nodes
                    .iter()
                    .filter(|nd| matches!(nd.op, Op::Embedding) && nd.inputs.get(1) == Some(&v))
                    .filter_map(|nd| graph.value(nd.inputs[0]).shape.first().copied())
                    .min()
                    .unwrap_or(usize::MAX);
                Tensor::U32((0..n).map(|i| i.min(rows.saturating_sub(1)) as u32).collect())
            } else {
                Tensor::F32(
                    (0..n)
                        .map(|_| {
                            let (a, b) = (uniform().max(1e-12), uniform());
                            ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32
                        })
                        .collect(),
                )
            }
        })
        .collect()
}

/// One warm-up run of `f`, then the median of 3 timed runs (ms) and the warm-up's outputs.
fn time_ms(
    mut f: impl FnMut() -> Result<Vec<Tensor>, String>,
) -> Result<(f64, Vec<Tensor>), String> {
    let out = f()?;
    let mut t = [0.0f64; 3];
    for x in &mut t {
        let start = std::time::Instant::now();
        f()?;
        *x = start.elapsed().as_secs_f64() * 1e3;
    }
    t.sort_by(f64::total_cmp);
    Ok((t[1], out))
}

/// Puts every node of `graph` back on the primary device: one segment.
fn demote(graph: &mut Graph, placed: &mut PlacementPlan) {
    for n in &mut graph.nodes {
        n.device = placed.primary;
    }
    *placed = crate::graph::placement::segments_of(graph, placed.primary);
}

/// Bytes of the weights the ANE regions of `graph` compile, in the encoding [`crate::graph::ane::lower`]
/// picks: int4 `Q4_0`, int8 other quantized formats, fp16 floats.
#[cfg(target_os = "macos")]
fn ane_weight_bytes(graph: &Graph, info: &crate::graph::ane::AneInfo) -> usize {
    let mut seen = std::collections::HashSet::new();
    let mut total = 0;
    for n in graph.nodes.iter().filter(|n| n.device == Device::Ane) {
        for &v in &n.inputs {
            let val = graph.value(v);
            if val.kind != ValueKind::Weight || !seen.insert(v) {
                continue;
            }
            let e: usize = val.shape.iter().product();
            total += match val.dtype {
                DType::Quant(crate::hardware::QuantFormat::GGUF_Q4_0) if info.int4.is_ok() => {
                    e * 9 / 16
                }
                DType::Quant(_) if info.int8.is_ok() || info.int4.is_ok() => e * 17 / 16,
                _ => e * 2,
            };
        }
    }
    total
}

/// The ANE region `g` lowered over `weights` and compiled.
#[cfg(target_os = "macos")]
fn compile_ane(
    g: Graph,
    weights: &Weights,
    info: &crate::graph::ane::AneInfo,
) -> Result<CompiledSegment, String> {
    use crate::graph::ane;
    let (mil, blob, io) = ane::lower(&g, weights, info)?;
    let (n_in, n_out) = (io.inputs.len(), io.outputs.len());
    let program = ane::AneProgram::compile(&mil, &blob, n_in, io.in_bytes(), n_out, io.out_bytes())?;
    let mut tables = Weights::new();
    for b in &io.inputs {
        if let ane::IoSource::RopeTable(n) = b.source {
            let name = &g.value(g.nodes[n].inputs[2]).name;
            let t = weights
                .get(name)
                .ok_or_else(|| format!("weight '{name}' is not bound"))?;
            tables.insert(name.clone(), t.clone());
        }
    }
    Ok(CompiledSegment::Ane {
        graph: g,
        program,
        io,
        tables,
    })
}

/// Times region `g` as the Metal plan `metal` and as the ANE program `ane` on [`synth_inputs`]
/// (1 warm-up, median of 3) and compares their outputs.
#[cfg(target_os = "macos")]
fn measure_region(
    g: &Graph,
    metal: &crate::graph::metal::PreparedPlan,
    ane: &mut CompiledSegment,
    session: &crate::graph::metal::Session,
) -> Result<AneDecision, String> {
    let inputs = synth_inputs(g);
    let mut st = StateStore::new();
    let (metal_ms, want) = time_ms(|| session.run_prepared(g, metal, Some(&mut st), inputs.clone()))?;
    let (ane_ms, got) = time_ms(|| ane.run(&mut st, inputs.clone()))?;
    let rel = want
        .iter()
        .zip(&got)
        .map(|(w, o)| crate::graph::ane::rel_rms(&o.to_f32(), &w.to_f32()))
        .fold(0.0f32, f32::max);
    let rel_err = if rel.is_finite() { rel } else { f32::MAX };
    Ok(AneDecision {
        metal_ms,
        ane_ms,
        rel_err,
        chosen: rel_err <= ANE_REL_ERR && ane_ms <= ANE_SPEEDUP * metal_ms,
    })
}

/// The measured gate for `graph` placed with ANE regions (`placed`). Each region is lowered and
/// compiled, then measured against its own Metal plan ([`measure_region`]) unless the host cache
/// holds its decision (keyed by the region's [`signature`]). A region stays on the ANE when its
/// outputs are within [`ANE_REL_ERR`] and (`Auto`) its median is at most [`ANE_SPEEDUP`] of
/// Metal's (`--backend ane` skips the speed test); one that cannot be built (budget, memory,
/// lowering) or loses goes back to Metal. `--backend ane` fails when every measured region
/// fails the error test. Returns the program's segments (empty: `graph` runs on Metal alone),
/// with `graph` and `placed` updated to the final placement.
#[cfg(target_os = "macos")]
fn gate(
    graph: &mut Graph,
    placed: &mut PlacementPlan,
    session: &std::sync::Arc<crate::graph::metal::Session>,
    weights: &Weights,
    sec: Secondary,
) -> Result<Vec<SegmentExec>, String> {
    use crate::graph::ane;
    let regions: Vec<Vec<NodeId>> = placed
        .segments
        .iter()
        .filter(|s| s.device == Device::Ane)
        .map(|s| s.nodes.clone())
        .collect();
    if regions.is_empty() {
        return Ok(Vec::new());
    }
    let info = ane::probe()?;
    let memsize = crate::hardware::sysctl::int(c"hw.memsize").unwrap_or(u64::MAX) as f64;
    let need = ane_weight_bytes(graph, info) + crate::hardware::current_resident_memory_bytes();
    let refused = if ane::programs_compiled() + regions.len() > ane::ANE_MAX_PROGRAMS {
        Some("ANE program budget reached".to_string())
    } else if need as f64 > 0.75 * memsize {
        Some(format!("{need} B of ANE weights and RSS exceed 75% of memory"))
    } else {
        None
    };
    if let Some(why) = refused {
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| eprintln!("ANE: regions stay on Metal from here on: {why}"));
        demote(graph, placed);
        return Ok(Vec::new());
    }
    let mut kept: HashMap<NodeId, SegmentExec> = HashMap::new();
    let mut best_err: Option<f32> = None;
    for nodes in &regions {
        let (g, inputs, outputs) = fragment(graph, nodes);
        let sig = signature(&g);
        let key = format!("{sig:016x}");
        let name = |i: usize| graph.node(nodes[i]).name.as_str();
        let label = format!("{} nodes: {} … {}", nodes.len(), name(0), name(nodes.len() - 1));
        let known = with_ane_decisions(false, |m| m.get(&key).copied());
        // A cached region that cannot run on the ANE (lost, failed the error test or failed to
        // build) is not rebuilt.
        let usable = |d: &AneDecision| d.rel_err <= ANE_REL_ERR && (sec.force || d.chosen);
        if let Some(d) = known.filter(|d| !usable(d)) {
            record(sig, label, d, false, false);
            continue;
        }
        let built = session
            .prepare_plan(&g)
            .and_then(|p| Ok((p, compile_ane(g.clone(), weights, info)?)));
        let measured = built.and_then(|(plan, mut seg)| {
            let d = match known {
                Some(d) => (d, false),
                None => (measure_region(&g, &plan, &mut seg, session)?, true),
            };
            Ok((d, seg))
        });
        let ((d, fresh), seg) = match measured {
            Ok(x) => x,
            Err(e) => {
                eprintln!("ANE region {label} stays on Metal: {e}");
                let failed = AneDecision {
                    metal_ms: 0.0,
                    ane_ms: 0.0,
                    rel_err: f32::MAX,
                    chosen: false,
                };
                with_ane_decisions(true, |m| m.insert(key, failed));
                continue;
            }
        };
        if fresh {
            with_ane_decisions(true, |m| m.insert(key, d));
        }
        best_err = Some(best_err.map_or(d.rel_err, |b| b.min(d.rel_err)));
        let offloaded = d.rel_err <= ANE_REL_ERR && (sec.force || d.chosen);
        record(sig, label, d, fresh, offloaded);
        if offloaded {
            let compiled = seg;
            kept.insert(nodes[0], SegmentExec { compiled, inputs, outputs });
        }
    }
    if kept.is_empty() {
        demote(graph, placed);
        return match best_err {
            Some(e) if sec.force => Err(format!(
                "--backend ane: every ANE region's outputs differ from Metal's (best relative RMS {e:.3e}, limit {ANE_REL_ERR:e})"
            )),
            _ => Ok(Vec::new()),
        };
    }
    for nodes in regions.iter().filter(|r| !kept.contains_key(&r[0])) {
        for n in nodes {
            graph.nodes[n.0 as usize].device = placed.primary;
        }
    }
    *placed = crate::graph::placement::segments_of(graph, placed.primary);
    placed
        .segments
        .iter()
        .map(|s| {
            if s.device == Device::Ane {
                return kept
                    .remove(&s.nodes[0])
                    .ok_or_else(|| "an ANE segment lost its program".to_string());
            }
            let (g, inputs, outputs) = fragment(graph, &s.nodes);
            let plan = session.prepare_plan(&g)?;
            let compiled = CompiledSegment::Metal {
                graph: g,
                session: session.clone(),
                plan,
            };
            Ok(SegmentExec { compiled, inputs, outputs })
        })
        .collect()
}

/// Logs a region's gate decision and keeps it for [`ane_reports`].
#[cfg(target_os = "macos")]
fn record(sig: u64, region: String, decision: AneDecision, measured: bool, offloaded: bool) {
    let r = AneReport {
        signature: sig,
        region,
        decision,
        measured,
        offloaded,
    };
    eprintln!("{r}");
    ANE_REPORTS.lock().unwrap_or_else(|e| e.into_inner()).push(r);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, Op};

    fn two_linears() -> (Graph, Weights) {
        let mut g = Graph::new();
        let x = g.input("x", vec![1, 4], DType::F32);
        let w = g.weight("w", vec![4, 4], DType::F32);
        let a = g.add("lin0", Op::Linear, &[x, w], vec![1, 4], DType::F32);
        let b = g.add("lin1", Op::Linear, &[a, w], vec![1, 4], DType::F32);
        g.output(b);
        let mut weights = Weights::new();
        weights.insert("w", Tensor::F32(vec![0.1; 16]));
        (g, weights)
    }

    /// `[rows, 128]` → two ANE-lowerable MLP regions around a `Concat` (Metal only).
    fn ane_sandwich(rows: usize) -> Graph {
        let (mut g, f) = (Graph::new(), DType::F32);
        let silu = Op::Unary {
            kind: crate::graph::UnaryKind::Silu,
        };
        let x = g.input("x", vec![rows, 128], f);
        let w = |g: &mut Graph, n: &str, o: usize, i: usize| g.weight(n, vec![o, i], f);
        let (w1, w2) = (w(&mut g, "mlp0.up", 256, 128), w(&mut g, "mlp0.down", 128, 256));
        let (w3, w4) = (w(&mut g, "mlp1.up", 128, 256), w(&mut g, "mlp1.down", 128, 128));
        let a = g.add("up0", Op::Linear, &[x, w1], vec![rows, 256], f);
        let a = g.add("act0", silu.clone(), &[a], vec![rows, 256], f);
        let a = g.add("down0", Op::Linear, &[a, w2], vec![rows, 128], f);
        let add = Op::Binary {
            kind: crate::graph::BinaryKind::Add,
        };
        let a = g.add("res0", add.clone(), &[a, x], vec![rows, 128], f);
        let c = g.add("cat", Op::Concat { axis: -1 }, &[a, a], vec![rows, 256], f);
        let b = g.add("up1", Op::Linear, &[c, w3], vec![rows, 128], f);
        let b = g.add("act1", silu, &[b], vec![rows, 128], f);
        let b = g.add("down1", Op::Linear, &[b, w4], vec![rows, 128], f);
        // Residuals keep each region's output well above fp16 accumulation noise.
        let b = g.add("res1", add, &[b, a], vec![rows, 128], f);
        g.output(b);
        g
    }

    /// `--backend ane`: both MLP regions run on the ANE between Metal's `Concat`, for the compiled
    /// shape and a reshaped instance, and every output matches the single-device Metal program.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "ANE: cargo test --release --lib segmented_program_matches_single_device -- --ignored"]
    fn segmented_program_matches_single_device() {
        // Uniform in [-0.5, 0.5) from a multiplicative hash (periodic data cancels in fp16 sums).
        let hashed = |j: usize| {
            let h = (j as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            (h >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let mut weights = Weights::new();
        let names = [("mlp0.up", 256 * 128), ("mlp0.down", 128 * 256), ("mlp1.up", 128 * 256), ("mlp1.down", 128 * 128)];
        for (i, (name, n)) in names
            .into_iter()
            .enumerate()
        {
            let v = (0..n).map(|j| hashed(j + (i << 20)) * 0.1).collect();
            weights.insert(name, Tensor::F32(v));
        }
        let x = |rows: usize| vec![Tensor::F32((0..rows * 128).map(|j| hashed(j + (9 << 20))).collect())];
        let ane = PlacementMode::Pin(Device::Ane);
        let mut mixed =
            GraphProgram::compile_reshapable(ane_sandwich(64), weights.clone(), ane).unwrap();
        let devices: Vec<Device> = mixed.segments.iter().map(|s| s.compiled.device()).collect();
        assert_eq!(devices, [Device::Ane, Device::Metal, Device::Ane]);
        let metal = PlacementMode::Pin(Device::Metal);
        let mut single = GraphProgram::compile_reshapable(ane_sandwich(64), weights, metal).unwrap();
        assert!(single.segments.is_empty());
        let rel = |a: &Tensor, b: &Tensor| crate::graph::ane::rel_rms(&a.to_f32(), &b.to_f32());
        let (got, want) = (mixed.run(x(64)).unwrap(), single.run(x(64)).unwrap());
        assert!(rel(&got[0], &want[0]) <= ANE_REL_ERR, "{}", rel(&got[0], &want[0]));
        let g = ane_sandwich(40);
        let got = mixed.run_reshaped(&g, x(40), true).unwrap();
        let want = single.run_reshaped(&g, x(40), true).unwrap();
        assert_eq!(mixed.reshaped_graphs[0].segments.len(), 3);
        assert!(rel(&got[0], &want[0]) <= ANE_REL_ERR, "{}", rel(&got[0], &want[0]));
    }

    /// GPU: `run_reshaped` plans a row count once and reuses it; another row count gets its own plan.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "GPU: cargo test --lib run_reshaped_caches -- --ignored --test-threads=1"]
    fn run_reshaped_caches_one_plan_per_shape() {
        let (g, w) = two_linears();
        let mut prog = GraphProgram::compile(g, w, PlacementMode::Pin(Device::Metal)).unwrap();
        let rows = |m: usize| {
            let mut g = Graph::new();
            let x = g.input("x", vec![m, 4], DType::F32);
            let w = g.weight("w", vec![4, 4], DType::F32);
            let a = g.add("lin0", Op::Linear, &[x, w], vec![m, 4], DType::F32);
            let b = g.add("lin1", Op::Linear, &[a, w], vec![m, 4], DType::F32);
            g.output(b);
            g
        };
        let x = |m: usize| vec![Tensor::F32((0..m * 4).map(|i| i as f32 * 0.25).collect())];
        // Callers rebuild the graph per chunk: the second call's graph is a fresh, equal one.
        let a = prog.run_reshaped(&rows(2), x(2), true).unwrap();
        let b = prog.run_reshaped(&rows(2), x(2), true).unwrap();
        assert_eq!(prog.reshaped_graphs.len(), 1);
        assert_eq!(a[0].as_f32().unwrap(), b[0].as_f32().unwrap());
        // 0.25 * i through two all-0.1 4x4 layers: every output of row r is 0.04 * (sum of its x).
        let want: Vec<f32> = (0..2)
            .flat_map(|r| [0.04 * (0..4).map(|c| (r * 4 + c) as f32 * 0.25).sum::<f32>(); 4])
            .collect();
        for (got, want) in a[0].as_f32().unwrap().iter().zip(&want) {
            assert!((got - want).abs() < 1e-4, "{got} vs {want}");
        }
        prog.run_reshaped(&rows(3), x(3), true).unwrap();
        assert_eq!(prog.reshaped_graphs.len(), 2);
    }

    /// One CPU program runs its graph at two instance shapes (1 row, then 3 rows), sharing its weights
    /// and the conv state across them, exactly like running both graphs on the CPU executor in turn.
    #[test]
    fn run_reshaped_serves_cpu_programs() {
        let rows = |n: usize| {
            let mut g = Graph::new();
            let x = g.input("x", vec![n, 4], DType::F32);
            let w = g.weight("w", vec![3, 4], DType::F32);
            let taps = g.weight("taps", vec![3, 2], DType::F32);
            let st = g.state("conv", vec![1, 3], DType::F32, true);
            let a = g.push("lin", Op::Linear, &[x, w]).unwrap();
            let conv = Op::CausalConv1dCached {
                kernel: 2,
                dilation: 1,
            };
            let c = g.push("conv", conv, &[a, taps, st]).unwrap();
            g.output(c);
            g
        };
        let mut weights = Weights::new();
        weights.insert(
            "w",
            Tensor::F32((0..12).map(|i| i as f32 * 0.1 - 0.5).collect()),
        );
        weights.insert("taps", Tensor::F32(vec![0.5, 1.0, -1.0, 2.0, 0.25, 0.75]));
        let x = |n: usize, s: f32| Tensor::F32((0..n * 4).map(|i| (i as f32 + s).sin()).collect());
        let mut prog =
            GraphProgram::compile(rows(1), weights.clone(), PlacementMode::Pin(Device::Cpu))
                .unwrap();
        assert_eq!(prog.primary_device(), Device::Cpu);
        assert_eq!(prog.recurrent_states(), ["conv"]);
        let got1 = prog.run(vec![x(1, 0.0)]).unwrap();
        let got3 = prog.run_reshaped(&rows(3), vec![x(3, 1.0)], true).unwrap();
        let mut st = StateStore::new();
        let want1 = cpu::run(&rows(1), &weights, &mut st, vec![x(1, 0.0)]).unwrap();
        let want3 = cpu::run(&rows(3), &weights, &mut st, vec![x(3, 1.0)]).unwrap();
        assert_eq!(got1, want1);
        assert_eq!(got3, want3);
        assert_eq!(got3[0].len(), 9);
        assert_eq!(prog.state.get("conv"), st.get("conv"));
    }

    #[test]
    fn placement_summary_counts_every_node_once_on_the_pinned_device() {
        let mut devices = vec![Device::Cpu];
        if available_devices().contains(&Device::Metal) {
            devices.push(Device::Metal);
        }
        for device in devices {
            let (g, w) = two_linears();
            let prog = GraphProgram::compile(g, w, PlacementMode::Pin(device)).unwrap();
            let rows = prog.placement_summary();
            let nodes = prog.segment.graph().nodes.len();
            assert!(nodes > 0);
            assert_eq!(
                rows.iter().map(|r| r.count).sum::<usize>(),
                nodes,
                "{rows:?}"
            );
            assert!(rows.iter().all(|r| r.device == device), "{rows:?}");
            assert!(rows.iter().any(|r| r.op == "Linear"), "{rows:?}");
            assert!(rows.iter().all(|r| r.avenue == "kernel"), "{rows:?}");
            // Counts never increase down the table.
            assert!(rows.windows(2).all(|p| p[0].count >= p[1].count));
        }
    }

    /// The device-argmax graph (`argmax_head`) run on the CPU executor picks the host greedy token
    /// (`sample_logits` at temperature 0) over the first `valid` logits, ties to the lower index,
    /// and keeps the other outputs.
    #[test]
    fn argmax_head_picks_the_host_greedy_token() {
        let mut seed = 0x2545_f491_u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 8
        };
        for n in [1usize, 7, 1000, 4099] {
            let mut g = Graph::new();
            let x = g.input("x", vec![1, n], DType::F32);
            let tap = g.input("tap", vec![1, 3], DType::F32);
            g.output(x);
            g.output(tap);
            for round in 0..6 {
                // A few levels, so the maximum ties; round 5 is all -inf.
                let row: Vec<f32> = (0..n)
                    .map(|_| match round {
                        5 => f32::NEG_INFINITY,
                        r => (next() % (4 + r * 1000)) as f32 * 0.25 - 3.0,
                    })
                    .collect();
                for valid in [0, n, n - n / 3, 1, n + 5] {
                    let head = argmax_head(&g, valid).expect("one f32 row");
                    let inputs = vec![Tensor::F32(row.clone()), Tensor::F32(vec![1.0, 2.0, 3.0])];
                    let out = cpu::run_stateless(&head, &Weights::new(), inputs).unwrap();
                    let v = if valid == 0 { n } else { valid.min(n) };
                    let want = crate::runtime::lm::sample_logits(&row[..v], &[], 0.0, 50, 1.0);
                    assert_eq!(
                        out[0],
                        Tensor::U32(vec![want]),
                        "n={n} round={round} valid={valid}"
                    );
                    assert_eq!(out[1], Tensor::F32(vec![1.0, 2.0, 3.0]));
                }
            }
        }
        // Output 0 must be one f32 row.
        let mut g = Graph::new();
        let x = g.input("x", vec![2, 4], DType::F32);
        g.output(x);
        assert!(argmax_head(&g, 0).is_none());
    }

    /// GPU: a device decode stream returns exactly the per-token greedy tokens, also with two
    /// streams interleaved on one program and across a budget restart.
    #[cfg(target_os = "macos")]
    #[test]
    fn stream_decode_matches_per_token_greedy() {
        if !available_devices().contains(&Device::Metal) {
            return;
        }
        // The llama spec's decode instance on a synthetic (random-init) checkpoint.
        let tmp = tempfile::tempdir().unwrap();
        let cfg = serde_json::json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 64,
            "vocab_size": 64, "max_position_embeddings": 48, "rms_norm_eps": 1e-5});
        std::fs::write(tmp.path().join("config.json"), cfg.to_string()).unwrap();
        let a = crate::model::hub::ModelArtifacts {
            model_id: "gibbs/stream-test".into(),
            local_dir: Some(tmp.path().to_path_buf()),
            config_path: Some(tmp.path().join("config.json")),
            ..Default::default()
        };
        let m =
            crate::arch::load::load(&a, crate::arch::registry().unwrap(), Some("llama")).unwrap();
        use crate::arch::expr::Val;
        let vars = [("rows", Val::Num(1.0)), ("head", Val::Str("last".into()))];
        let inst = crate::arch::build::build(&m.spec, "text", &m.env, &m.store, &vars).unwrap();
        let w = m.store.materialize(&inst.weights).unwrap();
        let vocab = 64;
        let mut prog =
            GraphProgram::compile(inst.graph, w, PlacementMode::Pin(Device::Metal)).unwrap();
        assert!(prog.is_single_metal());
        let reference = |prog: &mut GraphProgram, first: u32, n: usize| {
            prog.state = StateStore::new();
            let (mut tok, mut out) = (first, Vec::new());
            for pos in 0..n as u32 {
                let r = prog
                    .run(vec![Tensor::U32(vec![tok]), Tensor::U32(vec![pos])])
                    .unwrap();
                let logits = &r[0].as_f32().unwrap()[..vocab];
                // First largest, as the device argmax picks it.
                tok = (0..vocab).fold(0, |b, i| if logits[i] > logits[b] { i } else { b }) as u32;
                out.push(tok);
            }
            out
        };
        let (want1, want2) = (reference(&mut prog, 1, 20), reference(&mut prog, 2, 20));
        assert_ne!(
            want1, want2,
            "the fixture should not decode both prompts alike"
        );
        let stream = |prog: &mut GraphProgram, last: u32, pos: usize, max_new: usize| {
            prog.stream_next(last, pos as u32, max_new, vocab)
                .unwrap()
                .expect("the decode program streams")
        };
        // (a) One stream.
        prog.state = StateStore::new();
        let mut got = Vec::new();
        for pos in 0..20 {
            let last = if pos == 0 { 1 } else { got[pos - 1] };
            got.push(stream(&mut prog, last, pos, 20 - pos));
        }
        prog.stream_end().unwrap();
        assert_eq!(got, want1);
        // (b) Two streams interleaved call by call on one program.
        let (mut sa, mut sb) = (StateStore::new(), StateStore::new());
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for pos in 0..20 {
            for (st, out, first) in [(&mut sa, &mut a, 1), (&mut sb, &mut b, 2)] {
                std::mem::swap(&mut prog.state, st);
                let last = if pos == 0 { first } else { out[pos - 1] };
                out.push(stream(&mut prog, last, pos, 20 - pos));
                std::mem::swap(&mut prog.state, st);
            }
        }
        for st in [&mut sa, &mut sb] {
            std::mem::swap(&mut prog.state, st);
            prog.stream_end().unwrap();
            std::mem::swap(&mut prog.state, st);
        }
        assert_eq!((a, b), (want1.clone(), want2));
        // (c) A budget of five, then a restart with three more.
        prog.state = StateStore::new();
        let mut got = Vec::new();
        for pos in 0..8 {
            let last = if pos == 0 { 1 } else { got[pos - 1] };
            got.push(stream(
                &mut prog,
                last,
                pos,
                if pos < 5 { 5 - pos } else { 8 - pos },
            ));
        }
        prog.stream_end().unwrap();
        assert_eq!(got, want1[..8]);
        // (d) Stochastic device stream (the device sampler at temp = 0.7, top_p = 0.9).
        prog.state = StateStore::new();
        let mut sampled = Vec::new();
        for pos in 0..10 {
            let last = if pos == 0 { 1 } else { sampled[pos - 1] };
            let tok = prog
                .stream_next_sampled(last, pos as u32, 10 - pos, vocab, 0.7, 0.9)
                .unwrap()
                .expect("the decode program streams stochastically");
            assert!((tok as usize) < vocab);
            sampled.push(tok);
        }
        prog.stream_end().unwrap();
        assert_eq!(sampled.len(), 10);
    }

    /// One [`backends_match_cpu_on_every_op`] case: a graph, its weights, and its runs in order
    /// (`Some(graph)` runs another instance shape through `run_reshaped`), sharing one state.
    struct Case {
        name: String,
        graph: Graph,
        weights: Weights,
        runs: Vec<(Option<Graph>, Vec<Tensor>)>,
    }

    /// Builds a [`Case`] graph with deterministic pseudo-random data.
    struct Fx {
        g: Graph,
        w: Weights,
        ins: Vec<Tensor>,
        seed: u64,
    }

    impl Fx {
        fn new(seed: u64) -> Self {
            Self {
                g: Graph::new(),
                w: Weights::new(),
                ins: Vec::new(),
                seed: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            }
        }
        fn rnd(&mut self, n: usize) -> Vec<f32> {
            (0..n)
                .map(|_| {
                    self.seed = self
                        .seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((self.seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
                })
                .collect()
        }
        fn x(&mut self, name: &str, shape: &[usize]) -> ValueId {
            let v = self.rnd(shape.iter().product());
            self.ins.push(Tensor::F32(v));
            self.g.input(name, shape.to_vec(), DType::F32)
        }
        fn u(&mut self, name: &str, data: &[u32]) -> ValueId {
            self.u2(name, data, &[data.len()])
        }
        fn u2(&mut self, name: &str, data: &[u32], shape: &[usize]) -> ValueId {
            self.ins.push(Tensor::U32(data.to_vec()));
            self.g.input(name, shape.to_vec(), DType::U32)
        }
        fn w(&mut self, name: &str, shape: &[usize]) -> ValueId {
            let v = self.rnd(shape.iter().product());
            self.wt(name, shape, DType::F32, Tensor::F32(v))
        }
        fn wt(&mut self, name: &str, shape: &[usize], dtype: DType, t: Tensor) -> ValueId {
            self.w.insert(name, t);
            self.g.weight(name, shape.to_vec(), dtype)
        }
        fn op(&mut self, op: Op, ins: &[ValueId]) -> ValueId {
            let name = format!("n{}", self.g.nodes.len());
            self.g.push(name, op, ins).unwrap()
        }
        fn case(mut self, name: &str, outs: &[ValueId]) -> Case {
            for &o in outs {
                self.g.output(o);
            }
            Case {
                name: name.into(),
                graph: self.g,
                weights: self.w,
                runs: vec![(None, self.ins)],
            }
        }
    }

    /// `cols` (whole blocks) by `rows` of random `f` blocks whose CPU decode is finite and small.
    fn quant_rows(f: crate::hardware::QuantFormat, rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        let (elems, bytes, dec) = crate::model::layers::gguf_block_codec(f).expect("block codec");
        let mut s = seed | 1;
        let mut out = Vec::new();
        let mut vals = vec![0f32; elems];
        for _ in 0..rows * cols / elems {
            loop {
                let blk: Vec<u8> = (0..bytes)
                    .map(|_| {
                        s = s
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        (s >> 56) as u8
                    })
                    .collect();
                dec(&blk, &mut vals);
                if vals.iter().all(|v| v.is_finite() && v.abs() < 1e3) {
                    out.extend(blk);
                    break;
                }
            }
        }
        out
    }

    /// A CachedAttention layer instance of `m` query rows: q/k/v inputs, a `rows`-row KV state
    /// pair, positions `start..start + m`, optional per-row `visible_end`.
    fn cached_attn(
        op: &Op,
        m: usize,
        start: u32,
        vis: Option<u32>,
        rows: usize,
        seed: u64,
    ) -> (Graph, Vec<Tensor>) {
        let Op::CachedAttention {
            heads,
            kv_heads,
            head_dim,
            v_head_dim,
            ..
        } = *op
        else {
            unreachable!()
        };
        let mut f = Fx::new(seed);
        let q = f.x("q", &[m, heads * head_dim]);
        let k = f.x("k", &[m, kv_heads * head_dim]);
        let v = f.x("v", &[m, kv_heads * v_head_dim]);
        let ks =
            f.g.state("kc", vec![rows, kv_heads * head_dim], DType::F32, false);
        let vs =
            f.g.state("vc", vec![rows, kv_heads * v_head_dim], DType::F32, false);
        let pos: Vec<u32> = (start..start + m as u32).collect();
        let p = f.u("pos", &pos);
        let mut ins = vec![q, k, v, ks, vs, p];
        if let Some(extra) = vis {
            let ends: Vec<u32> = pos.iter().map(|&p| p + 1 + extra).collect();
            ins.push(f.u("vis", &ends));
        }
        let o = f.op(op.clone(), &ins);
        f.g.output(o);
        (f.g, f.ins)
    }

    fn op_cases() -> Vec<Case> {
        use crate::graph::{BinaryKind as B, ReduceKind, UnaryKind as U, UpsampleMode};
        use crate::hardware::QuantFormat as Q;
        let mut cases = Vec::new();
        let mut seed = 0u64;
        let mut fx = || {
            seed += 1;
            Fx::new(seed)
        };
        // Embedding: f32 table and a block-coded one.
        let mut f = fx();
        let t = f.w("table", &[10, 12]);
        let ids = f.u("ids", &[1, 9, 4, 1]);
        let o = f.op(Op::Embedding, &[t, ids]);
        cases.push(f.case("Embedding f32", &[o]));
        let mut f = fx();
        let raw = quant_rows(Q::GGUF_Q8_0, 6, 64, 7);
        let t = f.wt(
            "table",
            &[6, 64],
            DType::Quant(Q::GGUF_Q8_0),
            Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(
                Q::GGUF_Q8_0,
                raw,
                6,
                64,
            )),
        );
        let ids = f.u("ids", &[5, 0, 3]);
        let o = f.op(Op::Embedding, &[t, ids]);
        cases.push(f.case("Embedding q8_0", &[o]));
        // Linear: gemv and gemm paths, bias, a row width that is not a multiple of 8.
        for rows in [1, 3, 12, 40] {
            let mut f = fx();
            let x = f.x("x", &[rows, 20]);
            let w = f.w("w", &[37, 20]);
            let b = f.w("b", &[37]);
            let o = f.op(Op::Linear, &[x, w, b]);
            cases.push(f.case(&format!("Linear f32 rows {rows}"), &[o]));
        }
        // Linear per weight encoding.
        let dense = |rows: usize, cols: usize, f: &mut Fx| f.rnd(rows * cols);
        for (label, dt) in [("F16", DType::F16), ("BF16", DType::BF16)] {
            for rows in [2, 17] {
                let mut f = fx();
                let x = f.x("x", &[rows, 72]);
                let v = dense(9, 72, &mut f);
                let t = if dt == DType::F16 {
                    Tensor::F16(v.iter().map(|&x| crate::model::f32_to_f16(x)).collect())
                } else {
                    Tensor::BF16(v.iter().map(|&x| crate::model::f32_to_bf16(x)).collect())
                };
                let w = f.wt("w", &[9, 72], dt, t);
                let o = f.op(Op::Linear, &[x, w]);
                cases.push(f.case(&format!("Linear {label} rows {rows}"), &[o]));
            }
        }
        let mut f = fx();
        let x = f.x("x", &[3, 48]);
        let v: Vec<u16> = f
            .rnd(5 * 48)
            .iter()
            .map(|&x| crate::model::f32_to_bf16(x))
            .collect();
        let w = f.wt(
            "w",
            &[5, 48],
            DType::BF16,
            Tensor::Mat(crate::model::weights::Mat::from_bf16(v, 5, 48)),
        );
        let o = f.op(Op::Linear, &[x, w]);
        cases.push(f.case("Linear Mat bf16", &[o]));
        for (i, q) in crate::graph::codec::formats().into_iter().enumerate() {
            let elems = crate::graph::codec::codec(q).unwrap().block_elems as usize;
            let cols = (2 * elems).max(64);
            for rows in [3, 13] {
                let mut f = fx();
                let x = f.x("x", &[rows, cols]);
                let raw = quant_rows(q, 6, cols, 100 + i as u64);
                let mat = crate::model::weights::Mat::from_packed_bytes(q, raw, 6, cols);
                let w = f.wt("w", &[6, cols], DType::Quant(q), Tensor::Mat(mat));
                let o = f.op(Op::Linear, &[x, w]);
                cases.push(f.case(&format!("Linear {q:?} rows {rows}"), &[o]));
            }
        }
        // MatMul with batch broadcast and both transposes.
        for (ta, tb, sa, sb) in [
            (false, false, vec![2, 3, 5], vec![5, 4]),
            (true, true, vec![2, 1, 5, 3], vec![3, 4, 5]),
        ] {
            let mut f = fx();
            let a = f.x("a", &sa);
            let b = f.x("b", &sb);
            let o = f.op(
                Op::MatMul {
                    transpose_a: ta,
                    transpose_b: tb,
                },
                &[a, b],
            );
            cases.push(f.case(&format!("MatMul ta={ta} tb={tb}"), &[o]));
        }
        // Norms.
        let mut f = fx();
        let x = f.x("x", &[4, 300]);
        let (g, b) = (f.w("g", &[300]), f.w("b", &[300]));
        let o1 = f.op(Op::LayerNorm { eps: 1e-5 }, &[x, g, b]);
        let o2 = f.op(Op::LayerNorm { eps: 1e-5 }, &[x, g]);
        let o3 = f.op(Op::RmsNorm { eps: 1e-6 }, &[x, g]);
        let o4 = f.op(Op::RmsNorm { eps: 1e-6 }, &[x]);
        cases.push(f.case("norms", &[o1, o2, o3, o4]));
        let mut f = fx();
        let x = f.x("x", &[2, 8, 3, 5]);
        let (g, b) = (f.w("g", &[8]), f.w("b", &[8]));
        let o1 = f.op(
            Op::GroupNorm {
                groups: 4,
                eps: 1e-5,
            },
            &[x, g, b],
        );
        let o2 = f.op(Op::InstanceNorm { eps: 1e-5 }, &[x, g, b]);
        cases.push(f.case("GroupNorm / InstanceNorm", &[o1, o2]));
        // RoPE variants (one partial).
        for (label, variant, rot, pos3) in [
            (
                "interleaved",
                crate::graph::RopeVariant::Interleaved,
                16,
                false,
            ),
            (
                "half split partial",
                crate::graph::RopeVariant::HalfSplit,
                8,
                false,
            ),
            (
                "mrope",
                crate::graph::RopeVariant::MRope {
                    sections: [3, 3, 2],
                    interleaved: false,
                    pairs: crate::graph::PairLayout::Half,
                },
                16,
                true,
            ),
            (
                "mrope interleaved",
                crate::graph::RopeVariant::MRope {
                    sections: [3, 3, 2],
                    interleaved: true,
                    pairs: crate::graph::PairLayout::Adjacent,
                },
                16,
                true,
            ),
        ] {
            let mut f = fx();
            let x = f.x("x", &[4, 32]);
            let pos: Vec<u32> = if pos3 {
                (0..12).map(|i| 7 * i % 23).collect()
            } else {
                vec![0, 5, 9, 1000]
            };
            let p = f.u("pos", &pos);
            let inv: Vec<f32> = (0..rot / 2)
                .map(|j| 10000f32.powf(-((2 * j) as f32) / rot as f32))
                .collect();
            let inv = f.wt("inv", &[rot / 2], DType::F32, Tensor::F32(inv));
            let o = f.op(
                Op::Rope {
                    head_dim: 16,
                    rotary_dim: rot,
                    variant,
                    mscale: 1.0,
                },
                &[x, p, inv],
            );
            cases.push(f.case(&format!("Rope {label}"), &[o]));
        }
        // Attention: plain, GQA with window/soft cap/bias, non-causal window.
        for (label, heads, kvh, causal, window, cap, bias) in [
            ("plain", 2, 2, true, None, None, false),
            ("gqa window cap bias", 4, 2, true, Some(3), Some(5.0), true),
            ("bidirectional window", 2, 1, false, Some(4), None, false),
        ] {
            let mut f = fx();
            let q = f.x("q", &[6, heads * 8]);
            let k = f.x("k", &[6, kvh * 8]);
            let v = f.x("v", &[6, kvh * 8]);
            let mut ins = vec![q, k, v];
            if bias {
                ins.push(f.x("bias", &[6, 6]));
            }
            let o = f.op(
                Op::Attention {
                    heads,
                    kv_heads: kvh,
                    causal,
                    window,
                    soft_cap: cap,
                    scale: None,
                },
                &ins,
            );
            cases.push(f.case(&format!("Attention {label}"), &[o]));
        }
        // GLUs, unary, binary.
        let mut f = fx();
        let (a, b) = (f.x("a", &[3, 10]), f.x("b", &[3, 10]));
        let o1 = f.op(Op::SwiGlu, &[a, b]);
        let o2 = f.op(Op::GeGlu, &[a, b]);
        cases.push(f.case("GLU", &[o1, o2]));
        let mut f = fx();
        let x = f.x("x", &[200]);
        let pos = f.op(Op::Unary { kind: U::Abs }, &[x]);
        let pos = f.op(
            Op::Unary {
                kind: U::Clamp(0.05, 10.0),
            },
            &[pos],
        );
        let mut outs = Vec::new();
        for kind in [
            U::Exp,
            U::Sigmoid,
            U::Tanh,
            U::Softplus,
            U::Neg,
            U::Relu,
            U::Relu2,
            U::Silu,
            U::GeluErf,
            U::GeluTanh,
            U::Scale(1.5),
            U::Clamp(-0.3, 0.4),
            U::SoftCap(0.5),
            U::RoundEven,
            U::Sign,
        ] {
            outs.push(f.op(Op::Unary { kind }, &[x]));
        }
        for kind in [U::Sqrt, U::Rsqrt, U::Ln] {
            outs.push(f.op(Op::Unary { kind }, &[pos]));
        }
        cases.push(f.case("Unary", &outs));
        let mut f = fx();
        let a = f.x("a", &[2, 3, 4]);
        let b = f.x("b", &[2, 3, 4]);
        let c = f.x("c", &[3, 1]);
        let r = f.x("r", &[2, 1]);
        let t = f.x("t", &[6]);
        let mut outs = Vec::new();
        for kind in [B::Add, B::Sub, B::Mul, B::Div, B::Max, B::Min] {
            outs.push(f.op(Op::Binary { kind }, &[a, b]));
            outs.push(f.op(Op::Binary { kind }, &[a, c]));
        }
        let o = f.g.add(
            "row_scale",
            Op::Binary { kind: B::Mul },
            &[a, r],
            vec![2, 3, 4],
            DType::F32,
        );
        outs.push(o);
        let o = f.g.add(
            "tiled",
            Op::Binary { kind: B::Add },
            &[a, t],
            vec![2, 3, 4],
            DType::F32,
        );
        outs.push(o);
        cases.push(f.case("Binary", &outs));
        // CachedAttention prefill then decode on one state, several configurations; the last
        // two span several KV pages (a windowed ring among them).
        for (label, h, kvh, window, cap, vis, m, rows) in [
            ("gqa", 4, 2, None, None, None, 5, 64),
            ("window cap", 2, 1, Some(3), Some(20.0), None, 5, 64),
            ("visible end", 2, 2, None, None, Some(1), 5, 64),
            ("pages", 4, 2, None, None, None, 130, 4096),
            ("pages window", 4, 2, Some(100), None, None, 130, 4096),
        ] {
            let op = Op::CachedAttention {
                heads: h,
                kv_heads: kvh,
                head_dim: 8,
                v_head_dim: 8,
                scale: 0.3,
                causal: true,
                window,
                soft_cap: cap,
                write: true,
            };
            let (g, ins) = cached_attn(&op, m, 0, vis, rows, 40);
            let mut runs = vec![(None, ins)];
            for (i, pos) in [m as u32, m as u32 + 1].into_iter().enumerate() {
                let (gd, ins) = cached_attn(&op, 1, pos, vis.map(|_| 0), rows, 41 + i as u64);
                runs.push((Some(gd), ins));
            }
            cases.push(Case {
                name: format!("CachedAttention {label}"),
                graph: g,
                weights: Weights::new(),
                runs,
            });
        }
        // Recurrent states over two runs.
        let conv = |seed| {
            let mut f = Fx::new(seed);
            let x = f.x("x", &[3, 6]);
            let w = f.wt(
                "w",
                &[6, 4],
                DType::F32,
                Tensor::F32((0..24).map(|i| (i as f32 * 0.37).sin()).collect()),
            );
            let st = f.g.state("conv", vec![3, 6], DType::F32, true);
            let o = f.op(
                Op::CausalConv1dCached {
                    kernel: 4,
                    dilation: 1,
                },
                &[x, w, st],
            );
            f.case("CausalConv1dCached", &[o])
        };
        let (mut c, c2) = (conv(50), conv(51));
        c.runs.extend(c2.runs);
        cases.push(c);
        let delta = |seed| {
            let mut f = Fx::new(seed);
            let (q, k) = (f.x("q", &[3, 16]), f.x("k", &[3, 16]));
            let v = f.x("v", &[3, 32]);
            let beta = f.x("beta", &[3, 4]);
            let ld = f.x("ld", &[3, 4]);
            let st = f.g.state("ssm", vec![4, 8, 8], DType::F32, true);
            let o = f.op(
                Op::DeltaRule {
                    key_heads: 2,
                    val_heads: 4,
                    key_dim: 8,
                    val_dim: 8,
                },
                &[q, k, v, beta, ld, st],
            );
            f.case("DeltaRule", &[o])
        };
        let (mut c, c2) = (delta(52), delta(53));
        c.runs.extend(c2.runs);
        cases.push(c);
        // Indexing and data movement.
        let mut f = fx();
        let x = f.x("x", &[5, 4]);
        let i0 = f.u("i0", &[4, 0, 2]);
        let i1 = f.u("i1", &[3, 1]);
        let upd = f.x("upd", &[2, 4]);
        let o1 = f.op(Op::Gather { axis: 0 }, &[x, i0]);
        let o2 = f.op(Op::Gather { axis: 1 }, &[x, i1]);
        let o3 = f.op(Op::Scatter { axis: 0 }, &[x, i1, upd]);
        cases.push(f.case("Gather / Scatter", &[o1, o2, o3]));
        let mut f = fx();
        let x = f.x("x", &[2, 3, 4]);
        let o1 = f.op(
            Op::Pad {
                padding: vec![1, 2],
                value: 0.5,
            },
            &[x],
        );
        let o2 = f.op(
            Op::Pad {
                padding: vec![0, 1, 2, 0],
                value: -1.0,
            },
            &[x],
        );
        let o3 = f.op(
            Op::Reshape {
                shape: Some(vec![6, 4]),
            },
            &[x],
        );
        let o4 = f.op(
            Op::Transpose {
                perm: vec![2, 0, 1],
            },
            &[x],
        );
        let o5 = f.op(
            Op::Slice {
                axis: 1,
                start: 1,
                end: 3,
            },
            &[x],
        );
        let o6 = f.op(
            Op::Slice {
                axis: -1,
                start: -3,
                end: 100,
            },
            &[x],
        );
        let o7 = f.op(Op::Unsqueeze { axes: vec![0] }, &[x]);
        let o8 = f.op(Op::Squeeze { axes: vec![0] }, &[o7]);
        let o9 = f.op(
            Op::Tile {
                reps: vec![2, 1, 3],
            },
            &[x],
        );
        let y = f.x("y", &[2, 2, 4]);
        let o10 = f.op(Op::Concat { axis: 1 }, &[x, y, x]);
        cases.push(f.case("movement", &[o1, o2, o3, o4, o5, o6, o7, o8, o9, o10]));
        let mut f = fx();
        let x = f.x("x", &[2, 6, 7]);
        let o1 = f.op(
            Op::AvgPool2d {
                kernel: [2, 2],
                stride: [2, 2],
            },
            &[x],
        );
        let o2 = f.op(
            Op::AvgPool2d {
                kernel: [3, 2],
                stride: [2, 1],
            },
            &[x],
        );
        let mut outs = vec![o1, o2];
        for mode in [
            UpsampleMode::Nearest,
            UpsampleMode::Bilinear,
            UpsampleMode::Bicubic,
        ] {
            outs.push(f.op(Op::Upsample { scale: 2, mode }, &[x]));
        }
        cases.push(f.case("AvgPool2d / Upsample", &outs));
        // Convolutions with bias, stride, padding, dilation and groups.
        let mut f = fx();
        let x = f.x("x", &[4, 9]);
        let w = f.w("w", &[6, 2, 3]);
        let b = f.w("b", &[6]);
        let o = f.op(
            Op::Conv1d {
                stride: 2,
                padding: [1, 2],
                dilation: 2,
                groups: 2,
            },
            &[x, w, b],
        );
        cases.push(f.case("Conv1d", &[o]));
        let mut f = fx();
        let x = f.x("x", &[1, 4, 7, 6]);
        let w = f.w("w", &[4, 2, 3, 3]);
        let b = f.w("b", &[4]);
        let o = f.op(
            Op::Conv2d {
                stride: [2, 1],
                padding: [1, 1, 0, 2],
                dilation: [1, 2],
                groups: 2,
            },
            &[x, w, b],
        );
        cases.push(f.case("Conv2d", &[o]));
        let mut f = fx();
        let x = f.x("x", &[2, 3, 4, 5]);
        let w = f.w("w", &[3, 2, 2, 3, 2]);
        let o = f.op(
            Op::Conv3d {
                stride: [1, 2, 1],
                padding: [0, 1, 1, 1, 0, 1],
                dilation: [1, 1, 2],
                groups: 1,
            },
            &[x, w],
        );
        cases.push(f.case("Conv3d", &[o]));
        // Selection.
        let mut f = fx();
        let mut v = f.rnd(3 * 40);
        v[7] = v[3];
        v[50] = f32::NAN;
        f.ins.push(Tensor::F32(v));
        let x = f.g.input("x", vec![3, 40], DType::F32);
        let o1 = f.op(Op::TopK { k: 5 }, &[x]);
        let idx = f.u2("idx", &[3, 0, 39, 7, 1, 2, 3, 4, 9, 9, 0, 1], &[3, 4]);
        let o2 = f.op(Op::GatherCols, &[x, idx]);
        let row = f.x("row", &[1, 40]);
        let o3 = f.op(Op::GatherCols, &[row, idx]);
        let o4 = f.op(Op::Softmax { axis: -1 }, &[x]);
        let o5 = f.op(Op::Softmax { axis: 0 }, &[x]);
        cases.push(f.case("TopK / GatherCols / Softmax", &[o1, o2, o3, o4, o5]));
        let mut f = fx();
        let score = f.x("score", &[4, 5]);
        let pos = f.u("pos", &[3, 9, 15, 19]);
        let o1 = f.op(
            Op::QsaSelect {
                ratio: 4,
                top_blocks: 2,
            },
            &[score, pos],
        );
        let o2 = f.op(
            Op::QsaBias {
                ratio: 4,
                top_blocks: 2,
            },
            &[score, pos],
        );
        cases.push(f.case("QsaSelect / QsaBias", &[o1, o2]));
        // MoE with an f32 and a block-coded bank.
        for coded in [false, true] {
            let mut f = fx();
            let x = f.x("x", &[2, 64]);
            let bank = if coded {
                let raw = quant_rows(Q::GGUF_Q4_0, 24, 64, 9);
                f.wt(
                    "bank",
                    &[24, 64],
                    DType::Quant(Q::GGUF_Q4_0),
                    Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(
                        Q::GGUF_Q4_0,
                        raw,
                        24,
                        64,
                    )),
                )
            } else {
                f.w("bank", &[24, 64])
            };
            let ids = f.u2("ids", &[3, 0, 1, 3], &[2, 2]);
            let o = f.op(
                Op::MoeLinear {
                    experts: 4,
                    top_k: 2,
                },
                &[x, bank, ids],
            );
            cases.push(f.case(&format!("MoeLinear coded={coded}"), &[o]));
        }
        // Positions, states and pooling.
        let mut f = fx();
        let table = f.w("table", &[32, 2]);
        let qp = f.u("qp", &[0, 3, 7, 40]);
        let kp = f.u("kp", &[0, 1, 2, 3, 60, 200]);
        let o = f.op(
            Op::RelPosBias {
                buckets: 32,
                max_distance: 128,
                bidirectional: true,
                heads: 2,
            },
            &[table, qp, kp],
        );
        let o2 = f.op(
            Op::RelPosBias {
                buckets: 32,
                max_distance: 128,
                bidirectional: false,
                heads: 2,
            },
            &[table, qp, kp],
        );
        cases.push(f.case("RelPosBias", &[o, o2]));
        let mut f = fx();
        let st = f.g.state("rows", vec![8, 4], DType::F32, false);
        let rows = f.x("rows_in", &[2, 4]);
        let p0 = f.u("p0", &[3]);
        let tok = f.op(Op::StateWrite, &[st, rows, p0]);
        let parent = f.u("parent", &[1, 0]);
        let small = f.x("small", &[2, 4]);
        let reo = f.op(Op::StateReorder, &[small, parent]);
        let seen = f.op(
            Op::Unary {
                kind: U::Scale(2.0),
            },
            &[st],
        );
        cases.push(f.case("StateWrite / StateReorder", &[tok, reo, seen]));
        let mut f = fx();
        let x = f.x("x", &[5, 6]);
        let mask = f.u("mask", &[1, 0, 1, 1, 0]);
        let o1 = f.op(Op::MeanPool, &[x, mask]);
        let o2 = f.op(Op::L2Normalize, &[x]);
        let mut outs = vec![o1, o2];
        for (kind, axes) in [
            (ReduceKind::Sum, vec![1]),
            (ReduceKind::Mean, vec![0]),
            (ReduceKind::Max, vec![-1]),
            (ReduceKind::Min, vec![0, 1]),
        ] {
            outs.push(f.op(
                Op::Reduce {
                    kind,
                    axes,
                    keep_dims: false,
                },
                &[x],
            ));
        }
        cases.push(f.case("MeanPool / L2Normalize / Reduce", &outs));
        // The fused MLP block.
        for (geglu, post) in [(false, false), (true, true)] {
            let mut f = fx();
            let x = f.x("x", &[2, 16]);
            let norm = f.w("norm", &[16]);
            let gate = f.w("gate", &[24, 16]);
            let up = f.w("up", &[24, 16]);
            let down = f.w("down", &[16, 24]);
            let mut ins = vec![x, norm, gate, up, down];
            if post {
                ins.push(f.w("post", &[16]));
            }
            let kind = crate::graph::MegakernelKind::Mlp {
                hidden_dim: 16,
                intermediate_dim: 24,
                eps: 1e-6,
                is_geglu: geglu,
                has_post_norm: post,
            };
            let o = f.op(
                Op::Megakernel(crate::graph::MegakernelDescriptor {
                    kind,
                    name: "mlp".into(),
                }),
                &ins,
            );
            cases.push(f.case(&format!("Megakernel geglu={geglu} post={post}"), &[o]));
        }
        cases
    }

    /// Relative RMS of `got` against `want`; non-finite entries must match exactly. u32 outputs
    /// are exact (1.0 on any difference).
    fn rel_rms(got: &Tensor, want: &Tensor) -> f64 {
        match (got, want) {
            (Tensor::U32(a), Tensor::U32(b)) => (a != b) as u8 as f64,
            _ => {
                let (a, b) = (got.to_f32(), want.to_f32());
                if a.len() != b.len() {
                    return f64::INFINITY;
                }
                let (mut num, mut den) = (0f64, 0f64);
                for (&x, &y) in a.iter().zip(&b) {
                    if !y.is_finite() {
                        if x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan()) {
                            return f64::INFINITY;
                        }
                        continue;
                    }
                    num += (x as f64 - y as f64).powi(2);
                    den += (y as f64).powi(2);
                }
                (num / den.max(1e-30)).sqrt()
            }
        }
    }

    /// Device-bound: every available non-CPU, non-Metal device runs one graph per op variant, a
    /// Linear per weight codec and CachedAttention prefill/decode pairs like the CPU executor:
    /// relative RMS <= 1e-4 (2e-2 on XLA, which holds bf16 weights). Prints each op's worst error.
    #[test]
    #[ignore = "device: GIBBS_XPU_ANY_OPENCL=1 cargo test --release --lib -- --ignored backends_match_cpu_on_every_op --nocapture"]
    fn backends_match_cpu_on_every_op() {
        let cases = op_cases();
        let covered: std::collections::BTreeSet<&str> = cases
            .iter()
            .flat_map(|c| c.graph.nodes.iter().map(|n| n.op.name()))
            .collect();
        let all = [
            "Embedding",
            "Linear",
            "MatMul",
            "LayerNorm",
            "RmsNorm",
            "GroupNorm",
            "InstanceNorm",
            "Rope",
            "Attention",
            "SwiGlu",
            "GeGlu",
            "Unary",
            "Binary",
            "CachedAttention",
            "CausalConv1dCached",
            "DeltaRule",
            "Gather",
            "Scatter",
            "Pad",
            "AvgPool2d",
            "Upsample",
            "Conv3d",
            "TopK",
            "GatherCols",
            "MoeLinear",
            "Softmax",
            "RelPosBias",
            "StateReorder",
            "StateWrite",
            "QsaSelect",
            "QsaBias",
            "Conv2d",
            "Conv1d",
            "Reshape",
            "Transpose",
            "Concat",
            "Slice",
            "Squeeze",
            "Unsqueeze",
            "Tile",
            "Reduce",
            "MeanPool",
            "L2Normalize",
            "Megakernel",
        ];
        let missing: Vec<_> = all.iter().filter(|n| !covered.contains(*n)).collect();
        assert!(missing.is_empty(), "ops without a case: {missing:?}");
        let devices: Vec<Device> = available_devices()
            .into_iter()
            .filter(|d| !matches!(d, Device::Cpu | Device::Metal))
            .collect();
        assert!(
            !devices.is_empty(),
            "no non-CPU, non-Metal device on this host"
        );
        for dev in devices {
            let tol = if dev == Device::Xla { 2e-2 } else { 1e-4 };
            let mut worst: std::collections::BTreeMap<String, f64> = Default::default();
            let mut failures = Vec::new();
            for case in &cases {
                let wants = cpu_reference(case);
                let mut gpu = GraphProgram::compile(
                    case.graph.clone(),
                    case.weights.clone(),
                    PlacementMode::Pin(dev),
                )
                .unwrap_or_else(|e| panic!("{dev} {}: {e}", case.name));
                assert_eq!(gpu.cpu_node_count(), 0);
                // Pages released in order are reused last-first: a view allocating after this
                // spacer released its pages gets descending page ids.
                if let Some(pool) = &gpu.kv_pool {
                    let mut spacer = crate::kv::KvView::new(pool.clone());
                    spacer.reserve(512, 0..512).unwrap();
                }
                let mut err = 0f64;
                for (step, ((g, ins), want)) in case.runs.iter().zip(&wants).enumerate() {
                    let got = match g {
                        None => gpu.run(ins.clone()),
                        Some(g) => gpu.run_reshaped(g, ins.clone(), true),
                    };
                    let got = got.unwrap_or_else(|e| panic!("{dev} {} run {step}: {e}", case.name));
                    assert_eq!(want.len(), got.len());
                    for (o, (w, g)) in want.iter().zip(&got).enumerate() {
                        let e = rel_rms(g, w);
                        if !(e <= tol) {
                            failures.push(format!(
                                "{} run {step} output {o}: rel RMS {e:.3e}",
                                case.name
                            ));
                        }
                        err = err.max(e);
                    }
                }
                if gpu.kv_pool.is_some() {
                    let kv = gpu.state.kv.as_ref().expect("paged state attached");
                    assert!(
                        kv.tables.iter().any(|t| !t.is_empty()),
                        "{}: no KV pages",
                        case.name
                    );
                }
                for n in &case.graph.nodes {
                    let e = worst.entry(n.op.name().to_string()).or_default();
                    *e = e.max(err);
                }
            }
            eprintln!("{dev} ({}): worst relative RMS per op", gpu_name(dev));
            for (op, e) in &worst {
                eprintln!("  {op:<20} {e:.3e}");
            }
            assert!(failures.is_empty(), "{dev}: {failures:#?}");
        }
    }

    fn gpu_name(d: Device) -> String {
        crate::graph::portable::device_api(d)
            .map_or_else(|| d.to_string(), |a| a.device_name().to_string())
    }

    /// The CPU executor's outputs for every run of `case`, in order, on one state.
    fn cpu_reference(case: &Case) -> Vec<Vec<Tensor>> {
        let _exact = crate::model::qdot::exact();
        let mode = PlacementMode::Pin(Device::Cpu);
        let mut cpu =
            GraphProgram::compile(case.graph.clone(), case.weights.clone(), mode).unwrap();
        case.runs
            .iter()
            .map(|(g, ins)| {
                match g {
                    None => cpu.run(ins.clone()),
                    Some(g) => cpu.run_reshaped(g, ins.clone(), true),
                }
                .unwrap()
            })
            .collect()
    }
}
