//! Execution of placed computational graphs: one device per program.

use crate::graph::placement::{
    device_supports_op, place_graph, placement_rows, PlacementPlan, PlacementRow,
};
use crate::graph::{cpu, DType, Graph, Op, StateStore, Tensor, ValueId, Weights};
use crate::hardware::{available_devices, Device, PlacementMode};

/// The program's executable form on its one device.
pub enum CompiledSegment {
    Cpu {
        graph: Graph,
        weights: Weights,
    },
    #[cfg(target_os = "macos")]
    Metal {
        graph: Graph,
        session: crate::graph::metal::Session,
        plan: crate::graph::metal::PreparedPlan,
    },
    Cuda {
        graph: Graph,
        session: crate::graph::cuda::CudaSession,
    },
    Rocm {
        graph: Graph,
        session: crate::graph::rocm::RocmSession,
    },
    Xla {
        graph: Graph,
        session: crate::graph::xla::XlaSession,
    },
    IntelXpu {
        graph: Graph,
        session: crate::graph::intel_xpu::IntelXpuSession,
    },
    Extropic {
        graph: Graph,
        session: crate::graph::extropic::ExtropicSession,
    },
}

impl CompiledSegment {
    pub fn device(&self) -> Device {
        match self {
            Self::Cpu { .. } => Device::Cpu,
            #[cfg(target_os = "macos")]
            Self::Metal { .. } => Device::Metal,
            Self::Cuda { .. } => Device::Cuda,
            Self::Rocm { .. } => Device::Rocm,
            Self::Xla { .. } => Device::Xla,
            Self::IntelXpu { .. } => Device::IntelXpu,
            Self::Extropic { .. } => Device::Extropic,
        }
    }

    pub fn graph(&self) -> &Graph {
        match self {
            Self::Cpu { graph, .. }
            | Self::Cuda { graph, .. }
            | Self::Rocm { graph, .. }
            | Self::Xla { graph, .. }
            | Self::IntelXpu { graph, .. }
            | Self::Extropic { graph, .. } => graph,
            #[cfg(target_os = "macos")]
            Self::Metal { graph, .. } => graph,
        }
    }
}

/// An executable graph program with state persistence.
pub struct GraphProgram {
    pub segment: CompiledSegment,
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
    /// [`Self::run_reshaped`]'s plans, keyed by the graph's node count and input shapes.
    #[cfg(target_os = "macos")]
    reshaped: Vec<(
        (usize, usize),
        Vec<Vec<usize>>,
        crate::graph::metal::PreparedPlan,
    )>,
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
    /// concatenations) stay bound beside them, so unfused instances still find them.
    // ponytail: folded originals stay resident next to their fused copy (only unpacked f32/f16
    // tensors fold); fuse reshaped instances the same way if that memory matters.
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
        let unfused = keep_folded.then(|| weights.clone());
        crate::graph::passes::fuse(&mut graph, Some(&mut weights))?;
        let dump: Option<usize> = std::env::var("GIBBS_PLAN_DUMP_CPU")
            .ok()
            .map(|v| v.parse().unwrap_or(1));
        let available = if dump.is_some() {
            vec![Device::Cpu, Device::Metal]
        } else {
            available_devices()
        };
        #[allow(unused_mut)]
        let mut plan = place_and_adapt(&mut graph, mode, &available)?;
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
        #[allow(unused_mut)]
        let mut kv_pool = None;
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
                let prepared_plan = session.prepare_plan(&graph)?;
                CompiledSegment::Metal {
                    graph,
                    session,
                    plan: prepared_plan,
                }
            }
            Device::Cuda => {
                let session = crate::graph::cuda::CudaSession::compile(&graph, weights)?;
                CompiledSegment::Cuda { graph, session }
            }
            Device::Rocm => {
                let session = crate::graph::rocm::RocmSession::compile(&graph, weights)?;
                CompiledSegment::Rocm { graph, session }
            }
            Device::Xla => {
                let session = crate::graph::xla::XlaSession::compile(&graph, weights)?;
                CompiledSegment::Xla { graph, session }
            }
            Device::IntelXpu => {
                let session = crate::graph::intel_xpu::IntelXpuSession::compile(&graph, weights)?;
                CompiledSegment::IntelXpu { graph, session }
            }
            Device::Extropic => {
                let session = crate::graph::extropic::ExtropicSession::compile(&graph, weights)?;
                CompiledSegment::Extropic { graph, session }
            }
            _ => CompiledSegment::Cpu { graph, weights },
        };

        let kv_pool =
            kv_pool.or_else(|| (!paged.is_empty()).then(|| crate::kv::KvPool::host(&paged)));
        let prog = Self {
            segment,
            plan,
            primary_device,
            state: StateStore::new(),
            kv_pool,
            graph_inputs,
            graph_outputs,
            #[cfg(target_os = "macos")]
            argmax: None,
            #[cfg(target_os = "macos")]
            reshaped: Vec::new(),
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
            Device::Extropic => crate::graph::extropic::device_name(),
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
        return matches!(self.segment, CompiledSegment::Metal { .. });
        #[cfg(not(target_os = "macos"))]
        false
    }

    /// Runs `graph` with the program's weights inside the program's state: another instance shape of
    /// the compiled network (same weight and state names), e.g. a batched prefill pass. On Metal, plans
    /// are cached by the graph's node count and input shapes, which identify such a graph (prefill
    /// chunks repeat); on the CPU the graph runs directly.
    pub fn run_reshaped(
        &mut self,
        graph: &Graph,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>, String> {
        #[cfg(test)]
        crate::graph::shape::note(graph);
        self.attach_kv()?;
        match &self.segment {
            CompiledSegment::Cpu { weights, .. } => {
                cpu::run(graph, weights, &mut self.state, inputs)
            }
            CompiledSegment::Cuda { session, .. } => session.run(graph, &mut self.state, inputs),
            CompiledSegment::Rocm { session, .. } => session.run(graph, &mut self.state, inputs),
            CompiledSegment::Xla { session, .. } => session.run(graph, &mut self.state, inputs),
            CompiledSegment::IntelXpu { session, .. } => {
                session.run(graph, &mut self.state, inputs)
            }
            CompiledSegment::Extropic { session, .. } => {
                session.run(graph, &mut self.state, inputs)
            }
            #[cfg(target_os = "macos")]
            CompiledSegment::Metal { session, .. } => {
                let shapes: Vec<Vec<usize>> = graph
                    .inputs
                    .iter()
                    .map(|&v| graph.value(v).shape.clone())
                    .collect();
                // Node and output counts: a tapped instance adds outputs, not nodes.
                let n = (graph.nodes.len(), graph.outputs.len());
                let hit = self
                    .reshaped
                    .iter()
                    .position(|(c, s, _)| *c == n && *s == shapes);
                // Least recently used first: a varying tail chunk must not evict the full chunk's plan.
                let entry = match hit {
                    Some(i) => self.reshaped.remove(i),
                    None => {
                        // ponytail: 4 shapes (full chunk, tails, image blocks); more if callers vary
                        // their row counts more.
                        if self.reshaped.len() == 4 {
                            self.reshaped.remove(0);
                        }
                        (n, shapes, session.prepare_plan(graph)?)
                    }
                };
                self.reshaped.push(entry);
                let plan = &self.reshaped[self.reshaped.len() - 1].2;
                session.run_prepared(graph, plan, Some(&mut self.state), inputs)
            }
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
        match &mut self.segment {
            CompiledSegment::Cpu { graph, weights } => {
                cpu::run(graph, weights, &mut self.state, inputs)
            }
            CompiledSegment::Cuda { graph, session } => session.run(graph, &mut self.state, inputs),
            CompiledSegment::Rocm { graph, session } => session.run(graph, &mut self.state, inputs),
            CompiledSegment::Xla { graph, session } => session.run(graph, &mut self.state, inputs),
            CompiledSegment::IntelXpu { graph, session } => {
                session.run(graph, &mut self.state, inputs)
            }
            CompiledSegment::Extropic { graph, session } => {
                session.run(graph, &mut self.state, inputs)
            }
            #[cfg(target_os = "macos")]
            CompiledSegment::Metal {
                graph,
                session,
                plan,
            } => session.run_prepared(graph, plan, Some(&mut self.state), inputs),
        }
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
        let CompiledSegment::Metal { graph, session, .. } = &self.segment else {
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
            // The positions this stream may write without regrowing state buffers: up to the next
            // 4096-row boundary (the lazy KV growth step), within the declared rows.
            let declared = plan
                .state_vars
                .iter()
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
            let cap = declared.map_or(next, |d| d.min(next));
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
    /// stochastic (`temp > 1e-4`, `top_p`, `rep_penalty`) decoding via `sample_logits_gpu`.
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    pub fn stream_next_sampled(
        &mut self,
        last: u32,
        pos: u32,
        max_new: usize,
        valid: usize,
        temp: f32,
        top_p: f32,
        rep_penalty: f32,
        recent: &[u32],
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
        let CompiledSegment::Metal {
            graph,
            session,
            plan,
        } = &self.segment
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
            let declared = plan
                .state_vars
                .iter()
                .filter_map(|&(v, _)| {
                    let val = graph.value(v);
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
            let cap = declared.map_or(next, |d| d.min(next));
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
                top_k: 64,
                inv_temp: 1.0 / temp.max(1e-4),
                top_p: top_p.clamp(1e-4, 1.0),
                rep_penalty,
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
                recent,
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
    #[allow(clippy::too_many_arguments)]
    pub fn stream_next_sampled(
        &mut self,
        _last: u32,
        _pos: u32,
        _max_new: usize,
        _valid: usize,
        _temp: f32,
        _top_p: f32,
        _rep_penalty: f32,
        _recent: &[u32],
    ) -> Result<Option<u32>, String> {
        Ok(None)
    }

    #[cfg(not(target_os = "macos"))]
    pub fn stream_end(&mut self) -> Result<bool, String> {
        Ok(false)
    }
}

/// Places `graph` (at most one segment) and, on Metal, adapts its dtypes to the Metal kernels.
fn place_and_adapt(
    graph: &mut Graph,
    mode: PlacementMode,
    available: &[Device],
) -> Result<PlacementPlan, String> {
    let plan = place_graph(graph, mode, available, &device_supports_op)
        .map_err(|report| report.error())?;
    assert!(
        plan.segments.len() <= 1,
        "placement yields at most one segment"
    );
    if plan.primary_device() == Device::Metal {
        crate::graph::passes::dtype_adapt::run(graph);
        crate::graph::passes::dtype_adapt::retype_uncoded_weights(graph, &|f| {
            device_supports_op(Device::Metal, &Op::Linear, DType::Quant(f))
        })?;
    }
    Ok(plan)
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
        let a = prog.run_reshaped(&rows(2), x(2)).unwrap();
        let b = prog.run_reshaped(&rows(2), x(2)).unwrap();
        assert_eq!(prog.reshaped.len(), 1);
        assert_eq!(a[0].as_f32().unwrap(), b[0].as_f32().unwrap());
        // 0.25 * i through two all-0.1 4x4 layers: every output of row r is 0.04 * (sum of its x).
        let want: Vec<f32> = (0..2)
            .flat_map(|r| [0.04 * (0..4).map(|c| (r * 4 + c) as f32 * 0.25).sum::<f32>(); 4])
            .collect();
        for (got, want) in a[0].as_f32().unwrap().iter().zip(&want) {
            assert!((got - want).abs() < 1e-4, "{got} vs {want}");
        }
        prog.run_reshaped(&rows(3), x(3)).unwrap();
        assert_eq!(prog.reshaped.len(), 2);
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
        let got3 = prog.run_reshaped(&rows(3), vec![x(3, 1.0)]).unwrap();
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
                    let want = crate::runtime::lm::sample_logits(&row[..v], &[], 0.0, 50, 1.0, 1.0);
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
        // (d) Stochastic device stream (`sample_logits_gpu` at temp = 0.7, top_p = 0.9).
        prog.state = StateStore::new();
        let mut sampled = Vec::new();
        for pos in 0..10 {
            let last = if pos == 0 { 1 } else { sampled[pos - 1] };
            let tok = prog
                .stream_next_sampled(last, pos as u32, 10 - pos, vocab, 0.7, 0.9, 1.1, &sampled)
                .unwrap()
                .expect("the decode program streams stochastically");
            assert!((tok as usize) < vocab);
            sampled.push(tok);
        }
        prog.stream_end().unwrap();
        assert_eq!(sampled.len(), 10);
    }

    #[test]
    fn multi_hardware_backends_compile_and_execute_matching_cpu() {
        let (g, w) = two_linears();
        let input = vec![Tensor::F32(
            (0..4).map(|i| (i as f32) * 0.25 - 0.5).collect(),
        )];
        let mut cpu_prog =
            GraphProgram::compile(g.clone(), w.clone(), PlacementMode::Pin(Device::Cpu)).unwrap();
        let want = cpu_prog.run(input.clone()).unwrap();

        for dev in [
            Device::Cuda,
            Device::Rocm,
            Device::Xla,
            Device::IntelXpu,
            Device::Extropic,
        ] {
            let mut prog =
                GraphProgram::compile(g.clone(), w.clone(), PlacementMode::Pin(dev)).unwrap();
            assert_eq!(prog.primary_device(), dev);
            assert_eq!(prog.cpu_node_count(), 0);
            let got = prog.run(input.clone()).unwrap();
            assert_eq!(got, want, "backend {dev:?} output mismatch");
        }
    }
}
