//! The portable executor: plans a [`Graph`] like the Metal executor (one kernel sequence in node
//! order, activations in a [`SharedArenaPlan`] arena), uploads weights once in their stored
//! encoding, keeps states on the device, and runs the plan on any [`GpuApi`].
//!
//! Every activation is 4 bytes per element (f32, or u32 for index values); every kernel tensor
//! argument is a device pointer plus an element offset. Shapes and strides a kernel needs beyond
//! scalars live in one per-plan `meta` buffer, and positions are read from device buffers, so a
//! plan's launches are identical from run to run (CUDA/HIP replay them from a captured graph).

use super::api::{Arg, Buf, GpuApi, Kernel, Module, Replay};
use super::Encoding;
use crate::graph::plan::{SharedArenaPlan, DEFAULT_ARENA_ALIGNMENT};
use crate::graph::{
    BinaryKind, DType, Graph, MegakernelKind, Op, ReduceKind, RopeVariant, StateStore, Tensor,
    UnaryKind, UpsampleMode, ValueId, ValueKind, Weights,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub(crate) fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slice viewed as its bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

pub(crate) fn as_bytes_mut<T: Copy>(v: &mut [T]) -> &mut [u8] {
    // SAFETY: as above; every bit pattern is a valid f32/u32.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast(), std::mem::size_of_val(v)) }
}

/// A device allocation, freed on drop.
pub struct DevBuf {
    api: Arc<dyn GpuApi>,
    pub buf: Buf,
    pub bytes: usize,
}

impl DevBuf {
    pub fn new(api: &Arc<dyn GpuApi>, bytes: usize) -> Result<Self, String> {
        let bytes = bytes.max(4);
        Ok(Self {
            buf: api.alloc(bytes)?,
            api: api.clone(),
            bytes,
        })
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        self.api.free(self.buf);
    }
}

/// A persistent state tensor on a portable device (4-byte elements).
pub struct DevState {
    mem: DevBuf,
    pub elems: usize,
}

impl std::fmt::Debug for DevState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DevState({} elems on {})",
            self.elems,
            self.mem.api.device_name()
        )
    }
}

impl DevState {
    fn zeroed(api: &Arc<dyn GpuApi>, elems: usize) -> Result<Self, String> {
        Ok(Self {
            mem: DevBuf::new(api, elems * 4)?,
            elems,
        })
    }

    /// The state's elements as f32 (u32 states as their bits).
    pub fn read_f32(&self) -> Result<Vec<f32>, String> {
        let mut v = vec![0f32; self.elems];
        self.mem
            .api
            .download(self.mem.buf, 0, as_bytes_mut(&mut v))?;
        Ok(v)
    }

    /// Writes `v` (at most [`Self::elems`] values) at the start of the state.
    pub fn write_f32(&self, v: &[f32]) -> Result<(), String> {
        if v.len() > self.elems {
            return Err(format!(
                "state write of {} values exceeds its {} elements",
                v.len(),
                self.elems
            ));
        }
        self.mem.api.upload(self.mem.buf, 0, as_bytes(v))
    }

    /// An independent copy with `elems` elements (at least the current count).
    pub fn copy_grown(&self, elems: usize) -> Result<Self, String> {
        let out = Self::zeroed(&self.mem.api, elems.max(self.elems))?;
        out.write_f32(&self.read_f32()?)?;
        Ok(out)
    }
}

/// The compiled kernel module of each device, built once per process.
static MODULES: Mutex<Vec<(String, Module)>> = Mutex::new(Vec::new());

fn module(api: &dyn GpuApi) -> Result<Module, String> {
    let key = format!("{:?}/{}", api.dialect(), api.device_name());
    let mut mods = MODULES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, m)) = mods.iter().find(|(k, _)| *k == key) {
        return Ok(*m);
    }
    let m = api.compile(&super::source(api.dialect()))?;
    mods.push((key, m));
    Ok(m)
}

/// Device bandwidth from a timed copy of 256 MiB (read + write bytes over the median of 5 runs).
pub fn measure_copy_bandwidth(api: &dyn GpuApi) -> Result<f64, String> {
    const SRC: &str = "KERNEL gpk_bw_copy(GLOBAL const u32* a, GLOBAL u32* b, u32 n) {\n    u32 i = GID(0);\n    if (i < n) b[i] = a[i];\n}\n";
    let m = api.compile(&format!("{}{SRC}", super::dialect::prelude(api.dialect())))?;
    let k = api.kernel(&m, "gpk_bw_copy")?;
    let n = 64usize << 20;
    let (a, b) = (api.alloc(n * 4)?, api.alloc(n * 4)?);
    let args = [Arg::Buf(a), Arg::Buf(b), Arg::U32(n as u32)];
    let mut times = Vec::new();
    let mut res = Ok(());
    for _ in 0..6 {
        let t = std::time::Instant::now();
        res = api
            .launch(k, [(n / 256) as u32, 1, 1], [256, 1, 1], &args)
            .and_then(|_| api.sync());
        if res.is_err() {
            break;
        }
        times.push(t.elapsed().as_secs_f64());
    }
    api.free(a);
    api.free(b);
    res?;
    times.remove(0);
    times.sort_by(f64::total_cmp);
    Ok((2 * n * 4) as f64 / times[times.len() / 2] / 1e9)
}

/// An uploaded weight. `enc` is the per-encoding kernels' encoding for a matrix operand; plain
/// operands (`enc == None`) are f32 (or u32) elements.
struct DevWeight {
    mem: DevBuf,
    enc: Option<&'static Encoding>,
    row_bytes: usize,
}

/// Where a value's data is during a run.
#[derive(Clone, Copy, Debug)]
enum Loc {
    None,
    /// Element offset in the arena.
    Arena(u32),
    Weight(Buf),
    State(usize),
}

/// A kernel parameter: a value (device pointer + element offset), a weight buffer, a slice of
/// the plan's meta buffer or arena scratch (pointer + offset), a scalar, or the row capacity of a
/// state at the run's allocation (`StateRows(slot, row width)`, a u32).
#[derive(Clone, Copy, Debug)]
enum P {
    Val(ValueId),
    Buf(Buf),
    Meta(u32),
    Scratch(u32),
    U32(u32),
    F32(f32),
    U64(u64),
    StateRows(usize, u32),
}

struct Step {
    kernel: Kernel,
    grid: [u32; 3],
    block: [u32; 3],
    args: Vec<P>,
}

/// A state the plan reads or writes: its device size at first allocation and in full, and,
/// for a KV cache, its row width (KV caches grow by 4096-row steps).
struct StateSlot {
    name: String,
    full: usize,
    row: Option<usize>,
}

/// Rows `pos + m` a KV state must hold, `pos` being element 0 of graph input `pos_input`
/// (unknown when the positions are computed on the device: the state is then allocated in full).
struct Need {
    state: usize,
    pos_input: Option<usize>,
    m: usize,
}

struct Plan {
    graph: Graph,
    loc: Vec<Loc>,
    arena_elems: usize,
    steps: Vec<Step>,
    meta: DevBuf,
    states: Vec<StateSlot>,
    needs: Vec<Need>,
    /// Launch sequence captured for the binding signature it was recorded with.
    replay: Option<(Vec<u64>, Replay)>,
    last_sig: Option<Vec<u64>>,
}

/// Compiled portable program: weights on the device, the compiled graph's plan and up to four
/// plans of other instance shapes.
pub struct PortableSession {
    api: Arc<dyn GpuApi>,
    module: Module,
    kernels: HashMap<String, Kernel>,
    weights: Weights,
    dev: HashMap<(String, bool), DevWeight>,
    arena: Option<DevBuf>,
    main: Option<Plan>,
    reshaped: Vec<((usize, usize), Vec<Vec<usize>>, Plan)>,
}

fn numel(s: &[usize]) -> usize {
    s.iter().product()
}

fn groups(n: usize, b: usize) -> u32 {
    n.div_ceil(b).max(1) as u32
}

fn norm_axis(axis: isize, rank: usize, name: &str) -> Result<usize, String> {
    let a = if axis < 0 { axis + rank as isize } else { axis };
    if a < 0 || a as usize >= rank {
        return Err(format!(
            "node '{name}': axis {axis} out of range for rank {rank}"
        ));
    }
    Ok(a as usize)
}

fn dense_strides(shape: &[usize]) -> Vec<usize> {
    let mut st = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        st[d] = st[d + 1] * shape[d + 1];
    }
    st
}

fn u32s(v: &[usize]) -> Vec<u32> {
    v.iter().map(|&x| x as u32).collect()
}

const W: [u32; 3] = [256, 1, 1];

impl PortableSession {
    pub fn compile(api: Arc<dyn GpuApi>, graph: &Graph, weights: Weights) -> Result<Self, String> {
        let module = module(&*api)?;
        let mut s = Self {
            api,
            module,
            kernels: HashMap::new(),
            weights,
            dev: HashMap::new(),
            arena: None,
            main: None,
            reshaped: Vec::new(),
        };
        s.main = Some(s.plan(graph)?);
        Ok(s)
    }

    /// Runs the compiled graph.
    pub fn run(
        &mut self,
        state: &mut StateStore,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>, String> {
        let mut plan = self.main.take().expect("compiled plan");
        let out = self.execute(&mut plan, state, inputs);
        self.main = Some(plan);
        out
    }

    /// Runs another instance shape of the compiled network (same weight and state names); plans
    /// are cached by node count, output count and input shapes (4 most recently used).
    pub fn run_reshaped(
        &mut self,
        graph: &Graph,
        state: &mut StateStore,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>, String> {
        let shapes: Vec<Vec<usize>> = graph
            .inputs
            .iter()
            .map(|&v| graph.value(v).shape.clone())
            .collect();
        let key = (graph.nodes.len(), graph.outputs.len());
        let entry = match self
            .reshaped
            .iter()
            .position(|(k, s, _)| *k == key && *s == shapes)
        {
            Some(i) => self.reshaped.remove(i),
            None => {
                // ponytail: 4 shapes, like the Metal plan cache.
                if self.reshaped.len() == 4 {
                    let (_, _, old) = self.reshaped.remove(0);
                    self.drop_plan(old);
                }
                (key, shapes, self.plan(graph)?)
            }
        };
        let (k, s, mut plan) = entry;
        let out = self.execute(&mut plan, state, inputs);
        self.reshaped.push((k, s, plan));
        out
    }

    fn drop_plan(&self, plan: Plan) {
        if let Some((_, r)) = plan.replay {
            self.api.replay_free(r);
        }
    }

    fn kernel(&mut self, name: &str) -> Result<Kernel, String> {
        if let Some(&k) = self.kernels.get(name) {
            return Ok(k);
        }
        let k = self.api.kernel(&self.module, name)?;
        self.kernels.insert(name.to_string(), k);
        Ok(k)
    }

    /// Weight `name` as f32 (or u32) elements.
    fn weight_plain(&mut self, name: &str) -> Result<Buf, String> {
        let key = (name.to_string(), false);
        if let Some(w) = self.dev.get(&key) {
            return Ok(w.mem.buf);
        }
        let t = self
            .weights
            .get(name)
            .ok_or_else(|| format!("weight '{name}' is missing"))?;
        let mem = match t {
            Tensor::U32(v) => {
                let m = DevBuf::new(&self.api, v.len() * 4)?;
                self.api.upload(m.buf, 0, as_bytes(v))?;
                m
            }
            Tensor::F32(v) => {
                let m = DevBuf::new(&self.api, v.len() * 4)?;
                self.api.upload(m.buf, 0, as_bytes(v))?;
                m
            }
            other => {
                let v = other.to_f32();
                let m = DevBuf::new(&self.api, v.len() * 4)?;
                self.api.upload(m.buf, 0, as_bytes(&v))?;
                m
            }
        };
        let buf = mem.buf;
        self.dev.insert(
            key,
            DevWeight {
                mem,
                enc: None,
                row_bytes: 0,
            },
        );
        Ok(buf)
    }

    /// Weight `name` (`[rows, cols]` row-major) in its stored encoding: dense rows are padded to a
    /// multiple of 8 elements, block-coded rows are stored as is. Returns the buffer, encoding
    /// and row stride in bytes.
    fn weight_mat(
        &mut self,
        name: &str,
        cols: usize,
    ) -> Result<(Buf, &'static Encoding, usize), String> {
        let key = (name.to_string(), true);
        if let Some(w) = self.dev.get(&key) {
            return Ok((w.mem.buf, w.enc.expect("matrix upload"), w.row_bytes));
        }
        let t = self
            .weights
            .get(name)
            .ok_or_else(|| format!("weight '{name}' is missing"))?;
        if cols == 0 || t.len() % cols != 0 {
            return Err(format!(
                "weight '{name}' of {} elements is not rows of {cols}",
                t.len()
            ));
        }
        let rows = t.len() / cols;
        let dense = |enc: &str, esize: usize, raw: &[u8]| -> (Vec<u8>, &'static Encoding, usize) {
            let enc = super::encodings()
                .iter()
                .find(|e| e.name == enc)
                .expect("dense encoding");
            let padded = cols.next_multiple_of(8);
            if padded == cols {
                return (raw.to_vec(), enc, cols * esize);
            }
            let mut out = vec![0u8; rows * padded * esize];
            for r in 0..rows {
                out[r * padded * esize..][..cols * esize]
                    .copy_from_slice(&raw[r * cols * esize..][..cols * esize]);
            }
            (out, enc, padded * esize)
        };
        let (bytes, enc, row_bytes) = match t {
            Tensor::F32(v) => dense("F32", 4, as_bytes(v)),
            Tensor::F16(v) => dense("F16", 2, as_bytes(v)),
            Tensor::BF16(v) => dense("BF16", 2, as_bytes(v)),
            Tensor::U32(_) => return Err(format!("weight '{name}' is u32, not a matrix")),
            Tensor::Mat(m) => match m.packed() {
                None => dense("BF16", 2, as_bytes(m.data())),
                Some(p) => {
                    let len = p.format.tensor_bytes(rows * cols);
                    let raw = &p.map[p.off..p.off + len];
                    match (super::encoding(p.format), p.format) {
                        (Some(e), crate::hardware::QuantFormat::FP32) => {
                            let _ = e;
                            dense("F32", 4, raw)
                        }
                        (Some(_), crate::hardware::QuantFormat::FP16) => dense("F16", 2, raw),
                        (Some(_), crate::hardware::QuantFormat::BF16) => dense("BF16", 2, raw),
                        (Some(e), f) => {
                            if cols % e.block_elems != 0 {
                                return Err(format!(
                                    "weight '{name}': {cols} columns are not whole {f:?} blocks"
                                ));
                            }
                            (raw.to_vec(), e, f.tensor_bytes(cols))
                        }
                        // A format without a kernel decodes once at load (placement retyped it).
                        (None, _) => {
                            let v = m.to_f32();
                            dense("F32", 4, as_bytes(&v))
                        }
                    }
                }
            },
        };
        let mem = DevBuf::new(&self.api, bytes.len())?;
        self.api.upload(mem.buf, 0, &bytes)?;
        let buf = mem.buf;
        self.dev.insert(
            key,
            DevWeight {
                mem,
                enc: Some(enc),
                row_bytes,
            },
        );
        Ok((buf, enc, row_bytes))
    }

    fn plan(&mut self, graph: &Graph) -> Result<Plan, String> {
        let mut g = graph.clone();
        crate::graph::passes::dtype_adapt::run(&mut g);
        for v in &g.values {
            if v.shape.is_empty() && !matches!(v.kind, ValueKind::Weight) {
                return Err(format!(
                    "value '{}' has no static shape (the portable executor plans static shapes)",
                    v.name
                ));
            }
        }
        let arena = SharedArenaPlan::build(&g, DEFAULT_ARENA_ALIGNMENT);
        let mut loc = vec![Loc::None; g.values.len()];
        for (&v, span) in &arena.spans {
            loc[v.0 as usize] = Loc::Arena((span.offset / 4) as u32);
        }
        let mut top = arena.total_bytes / 4;
        for &v in &g.inputs {
            loc[v.0 as usize] = Loc::Arena(top as u32);
            top += numel(&g.value(v).shape).next_multiple_of(64);
        }
        let mut b = Builder {
            s: self,
            g: &g,
            loc,
            top,
            steps: Vec::new(),
            meta: Vec::new(),
            states: Vec::new(),
            state_ix: HashMap::new(),
            needs: Vec::new(),
        };
        for i in 0..g.nodes.len() {
            b.lower(i)?;
        }
        for &o in &g.outputs {
            b.bind(o)?;
        }
        let Builder {
            loc,
            top,
            steps,
            meta,
            states,
            needs,
            ..
        } = b;
        let meta_buf = DevBuf::new(&self.api, meta.len() * 4)?;
        if !meta.is_empty() {
            self.api.upload(meta_buf.buf, 0, as_bytes(&meta))?;
        }
        Ok(Plan {
            graph: g,
            loc,
            arena_elems: top,
            steps,
            meta: meta_buf,
            states,
            needs,
            replay: None,
            last_sig: None,
        })
    }

    /// Device state `slot` of `state`, created (zeros, or the host copy) or grown to `rows`.
    fn state_buf(
        &self,
        st: &mut StateStore,
        slot: &StateSlot,
        rows: Option<usize>,
    ) -> Result<(Buf, usize), String> {
        let want = match (slot.row, rows) {
            (Some(w), Some(r)) => (r.max(1).next_multiple_of(4096) * w)
                .min(slot.full)
                .max(r * w),
            _ => slot.full,
        };
        if let Some(d) = st.device.get(&slot.name) {
            if d.elems >= want {
                return Ok((d.mem.buf, d.elems));
            }
            let grown = d.copy_grown(want)?;
            let out = (grown.mem.buf, grown.elems);
            st.device.insert(slot.name.clone(), Arc::new(grown));
            return Ok(out);
        }
        let host = st.buffers.remove(&slot.name).map(|t| t.to_f32());
        let elems = want.max(host.as_ref().map_or(0, Vec::len));
        let d = DevState::zeroed(&self.api, elems)?;
        if let Some(h) = host {
            d.write_f32(&h)?;
        }
        let out = (d.mem.buf, d.elems);
        st.device.insert(slot.name.clone(), Arc::new(d));
        Ok(out)
    }

    fn execute(
        &mut self,
        plan: &mut Plan,
        st: &mut StateStore,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>, String> {
        let g = &plan.graph;
        if inputs.len() != g.inputs.len() {
            return Err(format!(
                "graph has {} inputs, got {}",
                g.inputs.len(),
                inputs.len()
            ));
        }
        let need_bytes = plan.arena_elems * 4;
        if self.arena.as_ref().is_none_or(|a| a.bytes < need_bytes) {
            self.arena = None;
            self.arena = Some(DevBuf::new(&self.api, need_bytes)?);
        }
        let arena = self.arena.as_ref().expect("arena").buf;
        // Inputs.
        let mut first_u32 = vec![None; inputs.len()];
        for (i, (&v, t)) in g.inputs.iter().zip(&inputs).enumerate() {
            let val = g.value(v);
            if numel(&val.shape) != t.len() {
                return Err(format!(
                    "input '{}' expects shape {:?}, got {} elements",
                    val.name,
                    val.shape,
                    t.len()
                ));
            }
            let Loc::Arena(off) = plan.loc[v.0 as usize] else {
                unreachable!("inputs live in the arena")
            };
            let off = off as usize * 4;
            match t {
                Tensor::U32(x) => {
                    first_u32[i] = x.first().copied();
                    self.api.upload(arena, off, as_bytes(x))?
                }
                Tensor::F32(x) => self.api.upload(arena, off, as_bytes(x))?,
                other => self.api.upload(arena, off, as_bytes(&other.to_f32()))?,
            }
        }
        // States, grown to the rows this run writes.
        let mut rows: Vec<Option<usize>> = vec![Some(0); plan.states.len()];
        for n in &plan.needs {
            rows[n.state] = match (rows[n.state], n.pos_input.map(|i| first_u32[i])) {
                (Some(r), Some(Some(p))) => Some(r.max(p as usize + n.m)),
                _ => None,
            };
        }
        let mut sbufs = Vec::with_capacity(plan.states.len());
        for (slot, r) in plan.states.iter().zip(&rows) {
            let r = if slot.row.is_some() { *r } else { None };
            sbufs.push(self.state_buf(st, slot, r)?);
        }
        let resolve = |v: ValueId| -> (Buf, u32) {
            match plan.loc[v.0 as usize] {
                Loc::Arena(o) => (arena, o),
                Loc::Weight(b) => (b, 0),
                Loc::State(i) => (sbufs[i].0, 0),
                Loc::None => unreachable!("every read value is bound"),
            }
        };
        let mut sig: Vec<u64> = vec![arena.0, plan.meta.buf.0];
        sig.extend(sbufs.iter().map(|b| b.0 .0));
        let replayed = match &plan.replay {
            Some((s, r)) if *s == sig => {
                self.api.replay(*r)?;
                true
            }
            _ => false,
        };
        if !replayed {
            let capture = plan.last_sig.as_ref() == Some(&sig) && self.api.capture_begin()?;
            for step in &plan.steps {
                let mut args = Vec::with_capacity(step.args.len() + 4);
                for &p in &step.args {
                    match p {
                        P::Val(v) => {
                            let (b, o) = resolve(v);
                            args.push(Arg::Buf(b));
                            args.push(Arg::U32(o));
                        }
                        P::Buf(b) => args.push(Arg::Buf(b)),
                        P::Meta(o) => {
                            args.push(Arg::Buf(plan.meta.buf));
                            args.push(Arg::U32(o));
                        }
                        P::Scratch(o) => {
                            args.push(Arg::Buf(arena));
                            args.push(Arg::U32(o));
                        }
                        P::U32(x) => args.push(Arg::U32(x)),
                        P::F32(x) => args.push(Arg::F32(x)),
                        P::U64(x) => args.push(Arg::U64(x)),
                        P::StateRows(i, w) => {
                            args.push(Arg::U32((sbufs[i].1 / w.max(1) as usize) as u32))
                        }
                    }
                }
                self.api.launch(step.kernel, step.grid, step.block, &args)?;
            }
            if capture {
                let r = self.api.capture_end()?;
                if let Some((_, old)) = plan.replay.replace((sig.clone(), r)) {
                    self.api.replay_free(old);
                }
                self.api.replay(r)?;
            }
            plan.last_sig = Some(sig);
        }
        // Outputs.
        let mut outs = Vec::with_capacity(g.outputs.len());
        for &o in &g.outputs {
            let val = g.value(o);
            if let (ValueKind::Weight, Some(t)) = (&val.kind, self.weights.get(&val.name)) {
                outs.push(t.clone());
                continue;
            }
            let (b, off) = resolve(o);
            let n = match plan.loc[o.0 as usize] {
                Loc::State(i) => st.device[&plan.states[i].name].elems,
                _ => numel(&val.shape),
            };
            let mut v = vec![0u32; n];
            self.api
                .download(b, off as usize * 4, as_bytes_mut(&mut v))?;
            outs.push(if val.dtype == DType::U32 {
                Tensor::U32(v)
            } else {
                Tensor::F32(v.into_iter().map(f32::from_bits).collect())
            });
        }
        Ok(outs)
    }
}

impl Drop for PortableSession {
    fn drop(&mut self) {
        for p in self.main.take().into_iter().chain(
            std::mem::take(&mut self.reshaped)
                .into_iter()
                .map(|(_, _, p)| p),
        ) {
            self.drop_plan(p);
        }
    }
}

struct Builder<'a> {
    s: &'a mut PortableSession,
    g: &'a Graph,
    loc: Vec<Loc>,
    /// Arena high-water mark in elements (inputs and scratch sit past the node-output spans).
    top: usize,
    steps: Vec<Step>,
    meta: Vec<u32>,
    states: Vec<StateSlot>,
    state_ix: HashMap<String, usize>,
    needs: Vec<Need>,
}

impl Builder<'_> {
    fn launch(
        &mut self,
        name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        args: Vec<P>,
    ) -> Result<(), String> {
        let kernel = self.s.kernel(name)?;
        self.steps.push(Step {
            kernel,
            grid,
            block,
            args,
        });
        Ok(())
    }

    fn meta(&mut self, xs: &[u32]) -> P {
        let off = self.meta.len() as u32;
        self.meta.extend_from_slice(xs);
        P::Meta(off)
    }

    fn scratch(&mut self, elems: usize) -> u32 {
        let off = self.top as u32;
        self.top += elems.max(1).next_multiple_of(64);
        off
    }

    fn state_slot(&mut self, v: ValueId) -> Result<usize, String> {
        let val = self.g.value(v);
        let ValueKind::State { name, .. } = &val.kind else {
            return Err(format!("value '{}' is not a state", val.name));
        };
        if let Some(&i) = self.state_ix.get(name) {
            return Ok(i);
        }
        // A 2-D state only CachedAttention reads is a KV cache: it grows with the context.
        let kv = val.shape.len() == 2
            && self.g.nodes.iter().all(|n| {
                !n.inputs.iter().any(|&x| {
                    matches!(&self.g.value(x).kind, ValueKind::State { name: m, .. } if m == name)
                }) || matches!(n.op, Op::CachedAttention { .. })
            });
        let i = self.states.len();
        self.states.push(StateSlot {
            name: name.clone(),
            full: numel(&val.shape),
            row: kv.then_some(val.shape[1]),
        });
        self.state_ix.insert(name.clone(), i);
        Ok(i)
    }

    /// Binds `v` (weights as plain f32 elements, states to their slot) and returns it as a parameter.
    fn bind(&mut self, v: ValueId) -> Result<P, String> {
        if matches!(self.loc[v.0 as usize], Loc::None) {
            let val = self.g.value(v);
            self.loc[v.0 as usize] = match &val.kind {
                ValueKind::Weight => Loc::Weight(self.s.weight_plain(&val.name)?),
                ValueKind::State { .. } => Loc::State(self.state_slot(v)?),
                _ => return Err(format!("value '{}' is never computed", val.name)),
            };
        }
        Ok(P::Val(v))
    }

    fn shape(&self, v: ValueId) -> &[usize] {
        &self.g.value(v).shape
    }

    fn copy(&mut self, src: P, dst: P, n: usize) -> Result<(), String> {
        self.launch(
            "copy_u32",
            [groups(n, 256), 1, 1],
            W,
            vec![src, dst, P::U32(n as u32)],
        )
    }

    /// `y = x · wᵀ (+ bias)` for x `[rows, in_f]`, weight `name` `[out_f, in_f]`.
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &mut self,
        x: P,
        name: &str,
        bias: Option<P>,
        y: P,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<(), String> {
        let (w, enc, row_bytes) = self.s.weight_mat(name, in_f)?;
        let bias = bias.unwrap_or(P::Buf(w));
        let has_bias = !matches!(bias, P::Buf(_));
        let bias = if has_bias {
            vec![bias]
        } else {
            vec![P::Buf(w), P::U32(0)]
        };
        let mut args = vec![x, P::Buf(w), P::U64(row_bytes as u64)];
        args.extend(bias);
        args.extend([
            P::U32(has_bias as u32),
            y,
            P::U32(rows as u32),
            P::U32(in_f as u32),
            P::U32(out_f as u32),
        ]);
        if rows <= 8 {
            self.launch(
                &format!("gemv_{}", enc.name),
                [out_f as u32, 1, 1],
                [64, 1, 1],
                args,
            )
        } else {
            let grid = [groups(out_f, 32), groups(rows, 32), 1];
            self.launch(&format!("gemm_{}", enc.name), grid, W, args)
        }
    }

    fn rmsnorm(
        &mut self,
        x: P,
        g: Option<P>,
        y: P,
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<(), String> {
        let (g, has) = match g {
            Some(g) => (g, 1),
            None => (x, 0),
        };
        let args = vec![x, g, P::U32(has), y, P::U32(d as u32), P::F32(eps)];
        self.launch("rmsnorm", [rows as u32, 1, 1], W, args)
    }

    fn lower(&mut self, i: usize) -> Result<(), String> {
        let g = self.g;
        let node = &g.nodes[i];
        let name = node.name.as_str();
        let ins = &node.inputs;
        let out = node.outputs[0];
        let y = P::Val(out);
        let on = numel(self.shape(out));
        let last = |k: usize| g.value(ins[k]).shape.last().copied().unwrap_or(0);
        let n_of = |k: usize| numel(&g.value(ins[k]).shape);
        let operand = |k: usize| {
            ins.get(k)
                .copied()
                .ok_or_else(|| format!("node '{name}' is missing operand {k}"))
        };
        match &node.op {
            Op::Embedding => {
                let d = last(0);
                let table = &g.value(ins[0]).name;
                let (w, enc, row_bytes) = self.s.weight_mat(table, d)?;
                let ids = self.bind(ins[1])?;
                let n = n_of(1);
                let table_rows = n_of(0) / d.max(1);
                let args = vec![
                    P::Buf(w),
                    P::U64(row_bytes as u64),
                    P::U32(table_rows as u32),
                    ids,
                    y,
                    P::U32(n as u32),
                    P::U32(d as u32),
                ];
                self.launch(
                    &format!("embed_{}", enc.name),
                    [groups(d.div_ceil(8), 64), n as u32, 1],
                    [64, 1, 1],
                    args,
                )?;
            }
            Op::Linear => {
                let [out_f, in_f] = self.shape(ins[1])[..] else {
                    return Err(format!(
                        "node '{name}': weight shape {:?} is not [out, in]",
                        self.shape(ins[1])
                    ));
                };
                let x = self.bind(ins[0])?;
                let rows = n_of(0) / in_f.max(1);
                let bias = match ins.get(2) {
                    Some(&b) => Some(self.bind(b)?),
                    None => None,
                };
                let wname = g.value(ins[1]).name.clone();
                self.linear(x, &wname, bias, y, rows, in_f, out_f)?;
            }
            Op::Megakernel(desc) => {
                let MegakernelKind::Mlp {
                    hidden_dim: d,
                    intermediate_dim: inter,
                    eps,
                    is_geglu,
                    has_post_norm,
                } = desc.kind;
                let x = self.bind(ins[0])?;
                let norm = self.bind(ins[1])?;
                let rows = n_of(0) / d.max(1);
                let normed = P::Scratch(self.scratch(rows * d));
                let gate = P::Scratch(self.scratch(rows * inter));
                let up = P::Scratch(self.scratch(rows * inter));
                let down = P::Scratch(self.scratch(rows * d));
                self.rmsnorm(x, Some(norm), normed, rows, d, eps)?;
                let (gw, uw, dw) = (
                    g.value(operand(2)?).name.clone(),
                    g.value(operand(3)?).name.clone(),
                    g.value(operand(4)?).name.clone(),
                );
                self.linear(normed, &gw, None, gate, rows, d, inter)?;
                self.linear(normed, &uw, None, up, rows, d, inter)?;
                let n = rows * inter;
                self.launch(
                    "glu",
                    [groups(n, 256), 1, 1],
                    W,
                    vec![gate, up, gate, P::U32(n as u32), P::U32(is_geglu as u32)],
                )?;
                self.linear(gate, &dw, None, down, rows, inter, d)?;
                if has_post_norm {
                    let post = self.bind(operand(5)?)?;
                    self.rmsnorm(down, Some(post), normed, rows, d, eps)?;
                    self.binary(
                        BinaryKind::Add,
                        x,
                        normed,
                        y,
                        rows * d,
                        (rows * d, rows * d),
                        None,
                    )?;
                } else {
                    self.binary(
                        BinaryKind::Add,
                        x,
                        down,
                        y,
                        rows * d,
                        (rows * d, rows * d),
                        None,
                    )?;
                }
            }
            Op::MatMul {
                transpose_a,
                transpose_b,
            } => {
                let (sa, sb) = (self.shape(ins[0]).to_vec(), self.shape(ins[1]).to_vec());
                if sa.len() < 2 || sb.len() < 2 {
                    return Err(format!(
                        "node '{name}': MatMul operands {sa:?} x {sb:?} must be rank >= 2"
                    ));
                }
                let (ar, ac, br, bc) = (
                    sa[sa.len() - 2],
                    sa[sa.len() - 1],
                    sb[sb.len() - 2],
                    sb[sb.len() - 1],
                );
                let (m, k) = if *transpose_a { (ac, ar) } else { (ar, ac) };
                let (k2, n) = if *transpose_b { (bc, br) } else { (br, bc) };
                if k != k2 {
                    return Err(format!(
                        "node '{name}': MatMul inner dims {k} != {k2} ({sa:?} x {sb:?})"
                    ));
                }
                let nd = (sa.len() - 2).max(sb.len() - 2);
                let pad = |d: &[usize]| [vec![1; nd - d.len()], d.to_vec()].concat();
                let (pa, pb) = (pad(&sa[..sa.len() - 2]), pad(&sb[..sb.len() - 2]));
                let mut ob = Vec::with_capacity(nd);
                for d in 0..nd {
                    match (pa[d], pb[d]) {
                        (x, y) if x == y || y == 1 => ob.push(x),
                        (1, y) => ob.push(y),
                        _ => return Err(format!(
                            "node '{name}': MatMul batch dims {sa:?} and {sb:?} do not broadcast"
                        )),
                    }
                }
                let off = |p: &[usize], mut flat: usize, sz: usize| {
                    let (mut o, mut stride) = (0, sz);
                    for d in (0..nd).rev() {
                        let idx = flat % ob[d];
                        flat /= ob[d];
                        if p[d] != 1 {
                            o += idx * stride;
                        }
                        stride *= p[d];
                    }
                    o
                };
                let nb = numel(&ob);
                let mut offs = Vec::with_capacity(2 * nb);
                for bi in 0..nb {
                    offs.push(off(&pa, bi, ar * ac) as u32);
                    offs.push(off(&pb, bi, br * bc) as u32);
                }
                let meta = self.meta(&offs);
                let (a, b) = (self.bind(ins[0])?, self.bind(ins[1])?);
                let args = vec![
                    a,
                    b,
                    y,
                    meta,
                    P::U32(m as u32),
                    P::U32(n as u32),
                    P::U32(k as u32),
                    P::U32(*transpose_a as u32),
                    P::U32(*transpose_b as u32),
                    P::U32(nb as u32),
                ];
                self.launch(
                    "matmul",
                    [groups(n, 16), groups(m, 16), nb as u32],
                    [16, 16, 1],
                    args,
                )?;
            }
            Op::RmsNorm { eps } => {
                let x = self.bind(ins[0])?;
                let gamma = match ins.get(1) {
                    Some(&w) => Some(self.bind(w)?),
                    None => None,
                };
                let d = if ins.len() > 1 { n_of(1) } else { last(0) };
                if d == 0 || n_of(0) % d != 0 {
                    return Err(format!("node '{name}': RmsNorm shapes mismatch"));
                }
                self.rmsnorm(x, gamma, y, n_of(0) / d, d, *eps)?;
            }
            Op::LayerNorm { eps } => {
                if ins.len() < 2 {
                    return Err(format!("node '{name}': LayerNorm needs x and gamma"));
                }
                let d = n_of(1);
                if d == 0 || n_of(0) % d != 0 || (ins.len() > 2 && n_of(2) != d) {
                    return Err(format!("node '{name}': x/gamma/beta lengths disagree"));
                }
                let (x, gm) = (self.bind(ins[0])?, self.bind(ins[1])?);
                let (b, has_b) = match ins.get(2) {
                    Some(&b) => (self.bind(b)?, 1),
                    None => (gm, 0),
                };
                let args = vec![x, gm, b, P::U32(has_b), y, P::U32(d as u32), P::F32(*eps)];
                self.launch("layernorm", [(n_of(0) / d) as u32, 1, 1], W, args)?;
            }
            Op::GroupNorm { .. } | Op::InstanceNorm { .. } => {
                let c = n_of(1);
                let (groups_n, eps) = match node.op {
                    Op::GroupNorm { groups, eps } => (groups, eps),
                    Op::InstanceNorm { eps } => (c, eps),
                    _ => unreachable!(),
                };
                if c == 0 || n_of(2) != c || groups_n == 0 || c % groups_n != 0 {
                    return Err(format!(
                        "node '{name}': gamma/beta lengths do not fit {groups_n} groups"
                    ));
                }
                let (n, hw) = crate::graph::cpu::channel_layout(self.shape(ins[0]), c)
                    .map_err(|e| format!("node '{name}': {e}"))?;
                let (x, gm, b) = (self.bind(ins[0])?, self.bind(ins[1])?, self.bind(ins[2])?);
                let args = vec![
                    x,
                    gm,
                    b,
                    y,
                    P::U32(c as u32),
                    P::U32(hw as u32),
                    P::U32(groups_n as u32),
                    P::F32(eps),
                ];
                self.launch("groupnorm", [groups_n as u32, n as u32, 1], W, args)?;
            }
            Op::Rope {
                head_dim,
                rotary_dim,
                variant,
                mscale,
            } => {
                let x = self.bind(ins[0])?;
                let pos = self.bind(ins[1])?;
                let inv = self.bind(ins[2])?;
                let half = rotary_dim / 2;
                if n_of(2) != half {
                    return Err(format!(
                        "node '{name}': inv_freq has {} entries, expected {half}",
                        n_of(2)
                    ));
                }
                let np = n_of(1);
                let (vcode, sections, inter, pairs) = match variant {
                    RopeVariant::Interleaved => (0u32, [0u32; 3], 0u32, 0u32),
                    RopeVariant::HalfSplit => (1, [0; 3], 0, 0),
                    RopeVariant::MRope {
                        sections,
                        interleaved,
                        pairs,
                    } => (
                        2,
                        *sections,
                        *interleaved as u32,
                        matches!(pairs, crate::graph::PairLayout::Half) as u32,
                    ),
                };
                let n = if np == 0 {
                    1
                } else if vcode == 2 {
                    np / 3
                } else {
                    np
                };
                let total = n_of(0) / n.max(1);
                let heads = if *head_dim > 0 { total / head_dim } else { 1 };
                self.copy(x, y, n_of(0))?;
                let meta = self.meta(&[
                    vcode,
                    sections[0],
                    sections[1],
                    sections[2],
                    inter,
                    pairs,
                    np as u32,
                ]);
                let args = vec![
                    y,
                    pos,
                    inv,
                    meta,
                    P::U32(n as u32),
                    P::U32(heads as u32),
                    P::U32(*head_dim as u32),
                    P::U32(total as u32),
                    P::U32(half as u32),
                    P::F32(*mscale),
                    P::U32(n_of(0) as u32),
                ];
                self.launch(
                    "rope",
                    [groups(half, 64), heads as u32, n as u32],
                    [64, 1, 1],
                    args,
                )?;
            }
            Op::Attention {
                heads,
                kv_heads,
                causal,
                window,
                soft_cap,
                scale,
            } => {
                let hd = last(0);
                let (nq_all, nk_all, nv_all) = (n_of(0), n_of(1), n_of(2));
                if *heads == 0
                    || *kv_heads == 0
                    || heads % kv_heads != 0
                    || hd == 0
                    || hd % heads != 0
                    || nq_all % hd != 0
                    || nk_all == 0
                    || nk_all % (kv_heads * (hd / heads)) != 0
                    || nk_all != nv_all
                {
                    return Err(format!("node '{name}': q/k/v shapes do not fit {heads} heads over {kv_heads} kv heads"));
                }
                let d = hd / heads;
                let (nq, nk) = (nq_all / hd, nk_all / (kv_heads * d));
                let bias_mode = match ins.get(3) {
                    None => 0u32,
                    Some(_) => match n_of(3) {
                        l if l == nk => 1,
                        l if l == nq * nk => 2,
                        l if l == heads * nq * nk => 3,
                        l => return Err(format!("node '{name}': bias of {l} elements is not [{heads}, {nq}, {nk}], [{nq}, {nk}] or [{nk}]")),
                    },
                };
                if d > 512 {
                    return Err(format!("node '{name}': head_dim {d} exceeds 512"));
                }
                let (q, k, v) = (self.bind(ins[0])?, self.bind(ins[1])?, self.bind(ins[2])?);
                let bias = match ins.get(3) {
                    Some(&b) => self.bind(b)?,
                    None => q,
                };
                let sc = scale.unwrap_or(1.0 / (d as f32).sqrt());
                let meta = self.meta(&[
                    nq as u32,
                    nk as u32,
                    *heads as u32,
                    *kv_heads as u32,
                    d as u32,
                    *causal as u32,
                    window.is_some() as u32,
                    window.unwrap_or(0) as u32,
                    bias_mode,
                ]);
                let args = vec![
                    q,
                    k,
                    v,
                    bias,
                    y,
                    meta,
                    P::F32(sc),
                    P::F32(soft_cap.unwrap_or(0.0)),
                ];
                self.launch(
                    "attention",
                    [*heads as u32, nq as u32, 1],
                    [128, 1, 1],
                    args,
                )?;
            }
            Op::SwiGlu | Op::GeGlu => {
                if n_of(0) != n_of(1) {
                    return Err(format!(
                        "node '{name}': {} lengths mismatch",
                        node.op.name()
                    ));
                }
                let (a, b) = (self.bind(ins[0])?, self.bind(ins[1])?);
                let geglu = matches!(node.op, Op::GeGlu) as u32;
                self.launch(
                    "glu",
                    [groups(on, 256), 1, 1],
                    W,
                    vec![a, b, y, P::U32(on as u32), P::U32(geglu)],
                )?;
            }
            Op::Unary { kind } => {
                let (code, p0, p1) = match kind {
                    UnaryKind::Exp => (0, 0.0, 0.0),
                    UnaryKind::Sigmoid => (1, 0.0, 0.0),
                    UnaryKind::Tanh => (2, 0.0, 0.0),
                    UnaryKind::Softplus => (3, 0.0, 0.0),
                    UnaryKind::Neg => (4, 0.0, 0.0),
                    UnaryKind::Sqrt => (5, 0.0, 0.0),
                    UnaryKind::Rsqrt => (6, 0.0, 0.0),
                    UnaryKind::Abs => (7, 0.0, 0.0),
                    UnaryKind::Relu => (8, 0.0, 0.0),
                    UnaryKind::Relu2 => (9, 0.0, 0.0),
                    UnaryKind::Silu => (10, 0.0, 0.0),
                    UnaryKind::GeluErf => (11, 0.0, 0.0),
                    UnaryKind::GeluTanh => (12, 0.0, 0.0),
                    UnaryKind::Scale(s) => (13, *s, 0.0),
                    UnaryKind::Clamp(lo, hi) => (14, *lo, *hi),
                    UnaryKind::SoftCap(c) => (15, *c, 0.0),
                    UnaryKind::Ln => (16, 0.0, 0.0),
                    UnaryKind::RoundEven => (17, 0.0, 0.0),
                    UnaryKind::Sign => (18, 0.0, 0.0),
                };
                let x = self.bind(ins[0])?;
                let n = n_of(0);
                self.launch(
                    "unary",
                    [groups(n, 256), 1, 1],
                    W,
                    vec![x, y, P::U32(n as u32), P::U32(code), P::F32(p0), P::F32(p1)],
                )?;
            }
            Op::Binary { kind } => {
                let (sa, sb) = (self.shape(ins[0]).to_vec(), self.shape(ins[1]).to_vec());
                let (la, lb) = (numel(&sa), numel(&sb));
                let (a, b) = (self.bind(ins[0])?, self.bind(ins[1])?);
                // The CPU's order: numpy broadcast, then `[m, ..] * [m, 1]` row scaling, then tiling.
                if let Some((dims, st_a, st_b)) = broadcast(&sa, &sb, on) {
                    self.binary(*kind, a, b, y, on, (0, 0), Some((dims, st_a, st_b)))?;
                } else if let (BinaryKind::Mul, Some(strided)) = (kind, row_scale(&sa, &sb)) {
                    self.binary(*kind, a, b, y, la.max(lb), (0, 0), Some(strided))?;
                } else if la == lb || (lb > 0 && la % lb == 0) || (la > 0 && lb % la == 0) {
                    self.binary(*kind, a, b, y, la.max(lb), (la, lb), None)?;
                } else {
                    return Err(format!("node '{name}': Binary shape mismatch {la} vs {lb}"));
                }
            }
            Op::Softmax { axis } => {
                let shape = self.shape(ins[0]).to_vec();
                let ax = norm_axis(*axis, shape.len(), name)?;
                let (outer, n, inner) = (numel(&shape[..ax]), shape[ax], numel(&shape[ax + 1..]));
                let x = self.bind(ins[0])?;
                self.launch(
                    "softmax",
                    [(outer * inner) as u32, 1, 1],
                    W,
                    vec![x, y, P::U32(n as u32), P::U32(inner as u32)],
                )?;
            }
            Op::TopK { k } => {
                let cols = last(0);
                if *k == 0 || *k > cols || n_of(0) % cols != 0 {
                    return Err(format!("node '{name}': TopK needs 1 <= k <= row width, got k={k} over {cols} columns"));
                }
                let x = self.bind(ins[0])?;
                self.launch(
                    "topk",
                    [(n_of(0) / cols) as u32, 1, 1],
                    W,
                    vec![x, y, P::U32(cols as u32), P::U32(*k as u32)],
                )?;
            }
            Op::GatherCols => {
                let (cols, k) = (last(0), last(1));
                if cols == 0 || k == 0 || n_of(0) % cols != 0 || n_of(1) % k != 0 {
                    return Err(format!(
                        "node '{name}': GatherCols needs x [rows, n] and idx [rows, k]"
                    ));
                }
                let shared = n_of(0) == cols;
                if !shared && n_of(0) / cols != n_of(1) / k {
                    return Err(format!("node '{name}': GatherCols needs x and idx over the same rows, or a single x row"));
                }
                let (x, idx) = (self.bind(ins[0])?, self.bind(ins[1])?);
                let n = n_of(1);
                self.launch(
                    "gather_cols",
                    [groups(n, 256), 1, 1],
                    W,
                    vec![
                        x,
                        idx,
                        y,
                        P::U32(n as u32),
                        P::U32(cols as u32),
                        P::U32(k as u32),
                        P::U32(shared as u32),
                    ],
                )?;
            }
            Op::MoeLinear { experts, top_k } => {
                let [bank_rows, in_f] = self.shape(ins[1])[..] else {
                    return Err(format!(
                        "node '{name}': the expert bank is not [experts * out, in]"
                    ));
                };
                let slots = n_of(2);
                if *experts == 0
                    || *top_k == 0
                    || in_f == 0
                    || bank_rows % experts != 0
                    || slots % top_k != 0
                {
                    return Err(format!("node '{name}': MoeLinear needs experts, top_k >= 1 and a [experts * out, in] bank with [m, top_k] ids"));
                }
                let out_f = bank_rows / experts;
                let x_rows = n_of(0) / in_f;
                if n_of(0) % in_f != 0 || (x_rows != slots && x_rows * top_k != slots) {
                    return Err(format!("node '{name}': x is neither [m, {in_f}] nor [m * {top_k}, {in_f}] for {slots} slots"));
                }
                let bank = g.value(ins[1]).name.clone();
                let (w, enc, row_bytes) = self.s.weight_mat(&bank, in_f)?;
                let (x, ids) = (self.bind(ins[0])?, self.bind(ins[2])?);
                let args = vec![
                    x,
                    P::Buf(w),
                    P::U64(row_bytes as u64),
                    ids,
                    y,
                    P::U32(*top_k as u32),
                    P::U32(in_f as u32),
                    P::U32(out_f as u32),
                    P::U32((x_rows == slots) as u32),
                    P::U32(*experts as u32),
                ];
                self.launch(
                    &format!("moe_{}", enc.name),
                    [out_f as u32, slots as u32, 1],
                    [64, 1, 1],
                    args,
                )?;
            }
            Op::RelPosBias {
                buckets,
                max_distance,
                bidirectional,
                heads,
            } => {
                let nb = if *bidirectional {
                    buckets / 2
                } else {
                    *buckets
                };
                if nb < 2 || *max_distance <= nb / 2 || n_of(0) != buckets * heads {
                    return Err(format!("node '{name}': RelPosBias needs table [{buckets}, {heads}], at least 2 buckets per direction and max_distance > {}", nb / 2));
                }
                let exact = nb / 2;
                let lscale = (*max_distance as f64 / exact as f64).ln() as f32;
                let (t, qp, kp) = (self.bind(ins[0])?, self.bind(ins[1])?, self.bind(ins[2])?);
                let (n, m) = (n_of(1), n_of(2));
                let args = vec![
                    t,
                    qp,
                    kp,
                    y,
                    P::U32(n as u32),
                    P::U32(m as u32),
                    P::U32(*heads as u32),
                    P::U32(nb as u32),
                    P::U32(*bidirectional as u32),
                    P::F32(lscale),
                ];
                self.launch("rel_pos_bias", [groups(m, 256), n as u32, 1], W, args)?;
            }
            Op::StateReorder => {
                let (x, parent) = (self.bind(ins[0])?, self.bind(ins[1])?);
                let batch = n_of(1);
                let row = n_of(0) / batch.max(1);
                self.launch(
                    "state_reorder",
                    [groups(n_of(0), 256), 1, 1],
                    W,
                    vec![
                        x,
                        parent,
                        y,
                        P::U32(batch as u32),
                        P::U32(row as u32),
                        P::U32(n_of(0) as u32),
                    ],
                )?;
            }
            Op::StateWrite => {
                let w = last(0);
                if w == 0 || last(1) != w || n_of(1) % w != 0 {
                    return Err(format!(
                        "node '{name}': StateWrite rows are not [m, {w}] like the state"
                    ));
                }
                let st = self.bind(ins[0])?;
                let slot = self.state_slot(ins[0])?;
                let (rows, pos) = (self.bind(ins[1])?, self.bind(ins[2])?);
                let full = numel(self.shape(ins[0]));
                self.needs_full(slot);
                let n = n_of(1);
                self.launch(
                    "state_write",
                    [groups(n, 256), 1, 1],
                    W,
                    vec![
                        st,
                        rows,
                        pos,
                        y,
                        P::U32(n as u32),
                        P::U32(w as u32),
                        P::U32(full as u32),
                    ],
                )?;
            }
            Op::QsaSelect { ratio, top_blocks } | Op::QsaBias { ratio, top_blocks } => {
                if *ratio == 0 || n_of(0) != n_of(1) * last(0) {
                    return Err(format!(
                        "node '{name}': QSA needs ratio >= 1 and score [n, B]"
                    ));
                }
                let (score, pos) = (self.bind(ins[0])?, self.bind(ins[1])?);
                let bias = matches!(node.op, Op::QsaBias { .. }) as u32;
                let n = n_of(1);
                let args = vec![
                    score,
                    pos,
                    y,
                    P::U32(n as u32),
                    P::U32(last(0) as u32),
                    P::U32(*ratio as u32),
                    P::U32(*top_blocks as u32),
                    P::U32(bias),
                ];
                self.launch("qsa_select", [groups(n, 64), 1, 1], [64, 1, 1], args)?;
            }
            Op::CachedAttention {
                heads,
                kv_heads,
                head_dim,
                v_head_dim,
                scale,
                causal,
                window,
                soft_cap,
                write,
            } => {
                let (h, kvh, hd, vhd) = (*heads, *kv_heads, *head_dim, *v_head_dim);
                let nq = n_of(0) / (h * hd).max(1);
                if vhd > 512 || hd > 512 {
                    return Err(format!("node '{name}': head dims {hd}/{vhd} exceed 512"));
                }
                let (q, kn, vn) = (self.bind(ins[0])?, self.bind(ins[1])?, self.bind(ins[2])?);
                let cached = ins.len() >= 6;
                let (kw, vw) = ((kvh * hd) as u32, (kvh * vhd) as u32);
                let rows_of = |s: Option<usize>, k: usize, w: u32| match s {
                    Some(slot) => P::StateRows(slot, w),
                    None => P::U32((n_of(k) / w.max(1) as usize) as u32),
                };
                let (ks, vs, pos, kcap, vcap) = if cached {
                    let (ks, vs) = (self.bind(ins[3])?, self.bind(ins[4])?);
                    let pos = self.bind(ins[5])?;
                    let (sk, sv) = (self.state_slot_opt(ins[3])?, self.state_slot_opt(ins[4])?);
                    let pos_input = g.inputs.iter().position(|&v| v == ins[5]);
                    for s in [sk, sv].into_iter().flatten() {
                        self.needs.push(Need {
                            state: s,
                            pos_input,
                            m: nq,
                        });
                    }
                    (ks, vs, pos, rows_of(sk, 3, kw), rows_of(sv, 4, vw))
                } else {
                    // Plain k/v rows starting at position 0.
                    (kn, vn, kn, rows_of(None, 1, kw), rows_of(None, 2, vw))
                };
                if cached && *write {
                    let (nk_new, nv_new) = (n_of(1), n_of(2));
                    self.launch(
                        "kv_append",
                        [groups(nk_new, 256), 1, 1],
                        W,
                        vec![kn, ks, pos, P::U32(nk_new as u32), P::U32(kw), kcap],
                    )?;
                    self.launch(
                        "kv_append",
                        [groups(nv_new, 256), 1, 1],
                        W,
                        vec![vn, vs, pos, P::U32(nv_new as u32), P::U32(vw), vcap],
                    )?;
                }
                let vis = match ins.get(6) {
                    Some(&v) => (self.bind(v)?, 1u32),
                    None => (q, 0),
                };
                let meta = self.meta(&[
                    nq as u32,
                    h as u32,
                    kvh as u32,
                    hd as u32,
                    vhd as u32,
                    *causal as u32,
                    window.is_some() as u32,
                    window.unwrap_or(0) as u32,
                    vis.1,
                    cached as u32,
                ]);
                let args = vec![
                    q,
                    ks,
                    vs,
                    pos,
                    vis.0,
                    y,
                    meta,
                    P::F32(*scale),
                    P::F32(soft_cap.unwrap_or(0.0)),
                    kcap,
                    vcap,
                ];
                self.launch(
                    "cached_attention",
                    [h as u32, nq as u32, 1],
                    [128, 1, 1],
                    args,
                )?;
            }
            Op::CausalConv1dCached { kernel, dilation } => {
                let (k, d) = (*kernel, *dilation);
                let channels = n_of(1) / k.max(1);
                let n = n_of(0) / channels.max(1);
                let hist = (k - 1) * d;
                let (x, w) = (self.bind(ins[0])?, self.bind(ins[1])?);
                let (st, has) = match ins.get(2) {
                    Some(&s) if n_of(2) == hist * channels => (self.bind(s)?, 1u32),
                    _ => (x, 0),
                };
                let args = vec![
                    x,
                    w,
                    st,
                    P::U32(has),
                    y,
                    P::U32(n as u32),
                    P::U32(channels as u32),
                    P::U32(k as u32),
                    P::U32(d as u32),
                ];
                self.launch("causal_conv", [groups(n * channels, 256), 1, 1], W, args)?;
                let writes_state =
                    ins.len() >= 3 && matches!(g.value(ins[2]).kind, ValueKind::State { .. });
                if writes_state && hist > 0 {
                    let tmp = P::Scratch(self.scratch(hist * channels));
                    let args = vec![
                        x,
                        st,
                        P::U32(has),
                        tmp,
                        P::U32(n as u32),
                        P::U32(channels as u32),
                        P::U32(hist as u32),
                    ];
                    self.launch(
                        "causal_conv_state",
                        [groups(hist * channels, 256), 1, 1],
                        W,
                        args,
                    )?;
                    let s = self.bind(ins[2])?;
                    self.copy(tmp, s, hist * channels)?;
                }
            }
            Op::DeltaRule {
                key_heads,
                val_heads,
                key_dim,
                val_dim,
            } => {
                let (kh, vh, kd, vd) = (*key_heads, *val_heads, *key_dim, *val_dim);
                let n = n_of(0) / (kh * kd).max(1);
                let ops: Vec<P> = (0..5)
                    .map(|k| self.bind(ins[k]))
                    .collect::<Result<_, _>>()?;
                let len = vh * kd * vd;
                let in_place = ins.len() > 5
                    && matches!(g.value(ins[5]).kind, ValueKind::State { .. })
                    && n_of(5) == len;
                let s = if in_place {
                    self.bind(ins[5])?
                } else {
                    let tmp = P::Scratch(self.scratch(len));
                    match ins.get(5) {
                        Some(&src) if n_of(5) == len => {
                            let src = self.bind(src)?;
                            self.copy(src, tmp, len)?;
                        }
                        _ => self.launch(
                            "fill_u32",
                            [groups(len, 256), 1, 1],
                            W,
                            vec![tmp, P::U32(len as u32), P::U32(0)],
                        )?,
                    }
                    tmp
                };
                let mut args = ops;
                args.extend([
                    s,
                    y,
                    P::U32(n as u32),
                    P::U32(kh as u32),
                    P::U32(vh as u32),
                    P::U32(kd as u32),
                    P::U32(vd as u32),
                    P::U32(n_of(3) as u32),
                    P::U32(n_of(4) as u32),
                ]);
                self.launch("delta_rule", [vh as u32, 1, 1], [64, 1, 1], args)?;
            }
            Op::Gather { axis } => {
                let xs = self.shape(ins[0]).to_vec();
                let ax = if *axis < 0 {
                    (xs.len() as isize + *axis) as usize
                } else {
                    *axis as usize
                };
                let outer = numel(&xs[..ax.min(xs.len())]);
                let dim = xs.get(ax).copied().unwrap_or(n_of(0));
                let inner = if ax + 1 < xs.len() {
                    numel(&xs[ax + 1..])
                } else {
                    1
                };
                let (x, idx) = (self.bind(ins[0])?, self.bind(ins[1])?);
                let nidx = n_of(1);
                let n = outer.max(1) * nidx * inner;
                self.launch(
                    "gather",
                    [groups(n, 256), 1, 1],
                    W,
                    vec![
                        x,
                        idx,
                        y,
                        P::U32(n as u32),
                        P::U32(nidx as u32),
                        P::U32(dim as u32),
                        P::U32(inner as u32),
                    ],
                )?;
            }
            Op::Scatter { axis } => {
                let bs = self.shape(ins[0]).to_vec();
                let ax = if *axis < 0 {
                    (bs.len() as isize + *axis) as usize
                } else {
                    *axis as usize
                };
                let outer = numel(&bs[..ax.min(bs.len())]).max(1);
                let dim = bs.get(ax).copied().unwrap_or(n_of(0));
                let inner = if ax + 1 < bs.len() {
                    numel(&bs[ax + 1..])
                } else {
                    1
                };
                let (base, idx, upd) = (self.bind(ins[0])?, self.bind(ins[1])?, self.bind(ins[2])?);
                self.copy(base, y, n_of(0))?;
                let n = outer * inner;
                self.launch(
                    "scatter",
                    [groups(n, 256), 1, 1],
                    W,
                    vec![
                        idx,
                        upd,
                        y,
                        P::U32(n as u32),
                        P::U32(n_of(1) as u32),
                        P::U32(dim as u32),
                        P::U32(inner as u32),
                        P::U32(n_of(2) as u32),
                        P::U32(n_of(0) as u32),
                    ],
                )?;
            }
            Op::Slice { axis, start, end } => {
                let xs = self.shape(ins[0]).to_vec();
                let ax = norm_axis(*axis, xs.len(), name)?;
                let (outer, dim, inner) = (numel(&xs[..ax]), xs[ax], numel(&xs[ax + 1..]));
                let clampi = |v: i64| {
                    if v < 0 {
                        (dim as i64 + v).max(0) as usize
                    } else {
                        (v as usize).min(dim)
                    }
                };
                let (s, e) = (clampi(*start), clampi(*end));
                let len = e.saturating_sub(s);
                let x = self.bind(ins[0])?;
                let n = outer * len * inner;
                self.launch(
                    "slice",
                    [groups(n, 256), 1, 1],
                    W,
                    vec![
                        x,
                        y,
                        P::U32(n as u32),
                        P::U32((len * inner) as u32),
                        P::U32((dim * inner) as u32),
                        P::U32((s * inner) as u32),
                    ],
                )?;
            }
            Op::Pad { padding, value } => {
                let shape = self.shape(ins[0]).to_vec();
                let (r, k) = (shape.len(), padding.len() / 2);
                if padding.len() % 2 != 0 || k > r {
                    return Err(format!(
                        "node '{name}': Pad {padding:?} does not fit shape {shape:?}"
                    ));
                }
                let before = |d: usize| {
                    if d + k >= r {
                        padding[2 * (d + k - r)]
                    } else {
                        0
                    }
                };
                let after = |d: usize| {
                    if d + k >= r {
                        padding[2 * (d + k - r) + 1]
                    } else {
                        0
                    }
                };
                let odims: Vec<usize> = (0..r).map(|d| shape[d] + before(d) + after(d)).collect();
                let mut m = vec![r as u32];
                m.extend(u32s(&odims));
                m.extend(u32s(&shape));
                m.extend((0..r).map(|d| before(d) as u32));
                let meta = self.meta(&m);
                let x = self.bind(ins[0])?;
                let n = numel(&odims);
                self.launch(
                    "pad",
                    [groups(n, 256), 1, 1],
                    W,
                    vec![x, y, meta, P::U32(n as u32), P::F32(*value)],
                )?;
            }
            Op::AvgPool2d { kernel, stride } => {
                let shape = self.shape(ins[0]).to_vec();
                let (c, h, w) = if shape.len() >= 3 {
                    (
                        numel(&shape[..shape.len() - 2]),
                        shape[shape.len() - 2],
                        shape[shape.len() - 1],
                    )
                } else {
                    (1, 1, n_of(0))
                };
                let oh = if h >= kernel[0] {
                    (h - kernel[0]) / stride[0] + 1
                } else {
                    0
                };
                let ow = if w >= kernel[1] {
                    (w - kernel[1]) / stride[1] + 1
                } else {
                    0
                };
                let x = self.bind(ins[0])?;
                let n = c * oh * ow;
                let args = vec![
                    x,
                    y,
                    P::U32(n as u32),
                    P::U32(h as u32),
                    P::U32(w as u32),
                    P::U32(oh as u32),
                    P::U32(ow as u32),
                    P::U32(kernel[0] as u32),
                    P::U32(kernel[1] as u32),
                    P::U32(stride[0] as u32),
                    P::U32(stride[1] as u32),
                ];
                self.launch("avgpool2d", [groups(n, 256), 1, 1], W, args)?;
            }
            Op::Upsample { scale, mode } => {
                let (inp, outp) = (self.shape(ins[0]).to_vec(), self.shape(out).to_vec());
                let nsp = crate::graph::cpu::upsample_nsp(&inp, &outp, *scale)
                    .map_err(|e| format!("node '{name}': {e}"))?;
                let r = inp.len();
                let (d, h, w) = if nsp == 3 {
                    (inp[r - 3], inp[r - 2], inp[r - 1])
                } else {
                    (1, inp[r - 2], inp[r - 1])
                };
                let sd = if nsp == 3 { *scale } else { 1 };
                let mode = match mode {
                    UpsampleMode::Nearest => 0,
                    UpsampleMode::Bilinear => 1,
                    UpsampleMode::Bicubic => 2,
                };
                let x = self.bind(ins[0])?;
                let args = vec![
                    x,
                    y,
                    P::U32(on as u32),
                    P::U32(d as u32),
                    P::U32(h as u32),
                    P::U32(w as u32),
                    P::U32(sd as u32),
                    P::U32(*scale as u32),
                    P::U32(mode),
                ];
                self.launch("upsample", [groups(on, 256), 1, 1], W, args)?;
            }
            Op::Conv1d {
                stride,
                padding: p,
                dilation,
                groups: gr,
            } => {
                self.conv(
                    i,
                    1,
                    [1, 1, *stride],
                    [0, 0, p[0]],
                    [0, 0, p[1]],
                    [1, 1, *dilation],
                    *gr,
                )?;
            }
            Op::Conv2d {
                stride,
                padding: p,
                dilation,
                groups: gr,
            } => {
                self.conv(
                    i,
                    2,
                    [1, stride[0], stride[1]],
                    [0, p[0], p[1]],
                    [0, p[2], p[3]],
                    [1, dilation[0], dilation[1]],
                    *gr,
                )?;
            }
            Op::Conv3d {
                stride,
                padding: p,
                dilation,
                groups: gr,
            } => {
                self.conv(
                    i,
                    3,
                    *stride,
                    [p[0], p[1], p[2]],
                    [p[3], p[4], p[5]],
                    *dilation,
                    *gr,
                )?;
            }
            Op::Reshape { .. } | Op::Squeeze { .. } | Op::Unsqueeze { .. } => {
                if n_of(0) != on {
                    return Err(format!(
                        "node '{name}': {} changes the element count",
                        node.op.name()
                    ));
                }
                let x = self.bind(ins[0])?;
                self.copy(x, y, on)?;
            }
            Op::Transpose { perm } => {
                let shape = self.shape(ins[0]).to_vec();
                let r = shape.len();
                let mut seen = vec![false; r];
                if perm.len() != r
                    || perm
                        .iter()
                        .any(|&p| p >= r || std::mem::replace(&mut seen[p], true))
                {
                    return Err(format!(
                        "node '{name}': {perm:?} is not a permutation of rank {r}"
                    ));
                }
                let st = dense_strides(&shape);
                let mut m = vec![r as u32];
                m.extend(perm.iter().map(|&p| shape[p] as u32));
                m.extend(perm.iter().map(|&p| st[p] as u32));
                let meta = self.meta(&m);
                let x = self.bind(ins[0])?;
                self.launch(
                    "transpose",
                    [groups(on, 256), 1, 1],
                    W,
                    vec![x, y, meta, P::U32(on as u32)],
                )?;
            }
            Op::Tile { reps } => {
                let shape = self.shape(ins[0]).to_vec();
                let r = shape.len().max(reps.len());
                let pad = |v: &[usize]| -> Vec<usize> {
                    std::iter::repeat_n(1, r - v.len())
                        .chain(v.iter().copied())
                        .collect()
                };
                let (ish, rp) = (pad(&shape), pad(reps));
                let odims: Vec<usize> = ish.iter().zip(&rp).map(|(a, b)| a * b).collect();
                let mut m = vec![r as u32];
                m.extend(u32s(&odims));
                m.extend(u32s(&ish));
                m.extend(u32s(&dense_strides(&ish)));
                let meta = self.meta(&m);
                let x = self.bind(ins[0])?;
                let n = numel(&odims);
                self.launch(
                    "tile",
                    [groups(n, 256), 1, 1],
                    W,
                    vec![x, y, meta, P::U32(n as u32)],
                )?;
            }
            Op::Concat { axis } => {
                let s0 = self.shape(ins[0]).to_vec();
                let rank = s0.len();
                let ax = norm_axis(*axis, rank, name)?;
                for &v in ins {
                    let sh = self.shape(v);
                    if sh.len() != rank || (0..rank).any(|d| d != ax && sh[d] != s0[d]) {
                        return Err(format!(
                            "node '{name}': concat operands disagree off axis {ax}"
                        ));
                    }
                }
                let inner = numel(&s0[ax + 1..]);
                let total: usize = ins.iter().map(|&v| g.value(v).shape[ax]).sum::<usize>() * inner;
                let mut at = 0;
                for &v in ins {
                    let chunk = g.value(v).shape[ax] * inner;
                    let n = numel(&g.value(v).shape);
                    let x = self.bind(v)?;
                    self.launch(
                        "concat_part",
                        [groups(n, 256), 1, 1],
                        W,
                        vec![
                            x,
                            y,
                            P::U32(n as u32),
                            P::U32(chunk as u32),
                            P::U32(total as u32),
                            P::U32(at as u32),
                        ],
                    )?;
                    at += chunk;
                }
            }
            Op::Reduce { kind, axes, .. } => {
                let shape = self.shape(ins[0]).to_vec();
                let rank = shape.len();
                if axes.is_empty() {
                    return Err(format!("node '{name}': Reduce with no axes is ambiguous"));
                }
                let mut red = vec![false; rank];
                for &a in axes {
                    let ax = if a < 0 { a + rank as isize } else { a };
                    if ax < 0 || ax >= rank as isize || red[ax as usize] {
                        return Err(format!(
                            "node '{name}': bad Reduce axis {a} for rank {rank}"
                        ));
                    }
                    red[ax as usize] = true;
                }
                let st = dense_strides(&shape);
                let (kept, gone): (Vec<usize>, Vec<usize>) = (0..rank).partition(|&d| !red[d]);
                let mut m = vec![kept.len() as u32];
                m.extend(kept.iter().map(|&d| shape[d] as u32));
                m.extend(kept.iter().map(|&d| st[d] as u32));
                m.push(gone.len() as u32);
                m.extend(gone.iter().map(|&d| shape[d] as u32));
                m.extend(gone.iter().map(|&d| st[d] as u32));
                let meta = self.meta(&m);
                let n_out: usize = kept.iter().map(|&d| shape[d]).product();
                let count: usize = gone.iter().map(|&d| shape[d]).product();
                let code = match kind {
                    ReduceKind::Sum => 0,
                    ReduceKind::Mean => 1,
                    ReduceKind::Max => 2,
                    ReduceKind::Min => 3,
                };
                let x = self.bind(ins[0])?;
                self.launch(
                    "reduce",
                    [groups(n_out, 256), 1, 1],
                    W,
                    vec![
                        x,
                        y,
                        meta,
                        P::U32(n_out as u32),
                        P::U32(count as u32),
                        P::U32(code),
                    ],
                )?;
            }
            Op::MeanPool => {
                let d = last(0);
                if d == 0 || n_of(0) != n_of(1) * d {
                    return Err(format!("node '{name}': x does not match mask"));
                }
                let (x, mask) = (self.bind(ins[0])?, self.bind(ins[1])?);
                self.launch(
                    "mean_pool",
                    [groups(d, 256), 1, 1],
                    W,
                    vec![x, mask, y, P::U32(n_of(1) as u32), P::U32(d as u32)],
                )?;
            }
            Op::L2Normalize => {
                let x = self.bind(ins[0])?;
                self.launch(
                    "l2_normalize",
                    [1, 1, 1],
                    W,
                    vec![x, y, P::U32(n_of(0) as u32)],
                )?;
            }
        }
        Ok(())
    }

    fn state_slot_opt(&mut self, v: ValueId) -> Result<Option<usize>, String> {
        match self.g.value(v).kind {
            ValueKind::State { .. } => self.state_slot(v).map(Some),
            _ => Ok(None),
        }
    }

    /// A state written at device-computed rows: allocated in full.
    fn needs_full(&mut self, slot: usize) {
        self.needs.push(Need {
            state: slot,
            pos_input: None,
            m: 0,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn binary(
        &mut self,
        kind: BinaryKind,
        a: P,
        b: P,
        y: P,
        n: usize,
        (la, lb): (usize, usize),
        strided: Option<(Vec<usize>, Vec<usize>, Vec<usize>)>,
    ) -> Result<(), String> {
        let code = match kind {
            BinaryKind::Add => 0,
            BinaryKind::Sub => 1,
            BinaryKind::Mul => 2,
            BinaryKind::Div => 3,
            BinaryKind::Max => 4,
            BinaryKind::Min => 5,
        };
        let (mode, meta) = match strided {
            None => (0, self.meta(&[la.max(1) as u32, lb.max(1) as u32])),
            Some((dims, sa, sb)) => {
                let mut m = vec![dims.len() as u32];
                m.extend(u32s(&dims));
                m.extend(u32s(&sa));
                m.extend(u32s(&sb));
                (1, self.meta(&m))
            }
        };
        self.launch(
            "binary",
            [groups(n, 256), 1, 1],
            W,
            vec![a, b, y, meta, P::U32(n as u32), P::U32(code), P::U32(mode)],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        i: usize,
        nd: usize,
        stride: [usize; 3],
        pb: [usize; 3],
        pe: [usize; 3],
        dil: [usize; 3],
        groups_n: usize,
    ) -> Result<(), String> {
        let g = self.g;
        let node = &g.nodes[i];
        let (name, ins) = (&node.name, &node.inputs);
        let (xs, ws) = (self.shape(ins[0]).to_vec(), self.shape(ins[1]).to_vec());
        if xs.len() < nd + 1 || ws.len() != nd + 2 {
            return Err(format!(
                "node '{name}': conv shapes {xs:?} / {ws:?} do not fit {nd}-d conv"
            ));
        }
        let (mut inp, mut k) = ([1usize; 3], [1usize; 3]);
        inp[3 - nd..].copy_from_slice(&xs[xs.len() - nd..]);
        k[3 - nd..].copy_from_slice(&ws[2..]);
        let c_in = xs[xs.len() - nd - 1];
        let batch = numel(&xs[..xs.len() - nd - 1]);
        let (c_out, cpg) = (ws[0], ws[1]);
        if groups_n == 0
            || c_in != cpg * groups_n
            || c_out % groups_n != 0
            || stride.contains(&0)
            || dil.contains(&0)
        {
            return Err(format!("node '{name}': conv c_in {c_in}, weight {ws:?}, groups {groups_n} are inconsistent"));
        }
        let mut od = [1usize; 3];
        for a in 0..3 {
            let span = dil[a] * (k[a] - 1) + 1;
            if inp[a] + pb[a] + pe[a] < span {
                return Err(format!(
                    "node '{name}': conv kernel larger than padded input"
                ));
            }
            od[a] = (inp[a] + pb[a] + pe[a] - span) / stride[a] + 1;
        }
        let mut m = vec![batch, c_in, c_out, cpg, c_out / groups_n];
        m.extend(inp);
        m.extend(k);
        m.extend(od);
        m.extend(stride);
        m.extend(pb);
        m.extend(dil);
        let meta = self.meta(&u32s(&m));
        let (x, w) = (self.bind(ins[0])?, self.bind(ins[1])?);
        let (bias, has) = match ins.get(2) {
            Some(&b) => (self.bind(b)?, 1),
            None => (x, 0),
        };
        let n = batch * c_out * numel(&od);
        let y = P::Val(node.outputs[0]);
        self.launch(
            "conv",
            [groups(n, 256), 1, 1],
            W,
            vec![x, w, bias, P::U32(has), y, meta, P::U32(n as u32)],
        )
    }
}

/// numpy broadcast of `a` and `b` to `out_len` elements as `(dims, a strides, b strides)`, 0
/// strides where an operand is broadcast; `None` where the CPU executor does not broadcast
/// (identical or unknown shapes, incompatible dims, another element count).
fn broadcast(
    a: &[usize],
    b: &[usize],
    out_len: usize,
) -> Option<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    if a == b || a.is_empty() || b.is_empty() {
        return None;
    }
    let r = a.len().max(b.len());
    let dim = |s: &[usize], i: usize| {
        if i + s.len() >= r {
            s[i + s.len() - r]
        } else {
            1
        }
    };
    let mut shape = Vec::with_capacity(r);
    for i in 0..r {
        let (x, y) = (dim(a, i), dim(b, i));
        if x != y && x != 1 && y != 1 {
            return None;
        }
        shape.push(if x == 1 { y } else { x });
    }
    if numel(&shape) != out_len {
        return None;
    }
    let strides = |s: &[usize]| {
        let mut st = vec![0usize; r];
        let mut acc = 1;
        for i in (0..r).rev() {
            let d = dim(s, i);
            st[i] = if d == 1 { 0 } else { acc };
            acc *= d;
        }
        st
    };
    Some((shape.clone(), strides(a), strides(b)))
}

/// The CPU's `[m, ..] * [m, 1]` row scaling (either order) as a strided product over `[m, d]`.
fn row_scale(a: &[usize], b: &[usize]) -> Option<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let rows = |x: &[usize], s: &[usize]| {
        (x.len() >= 2 && s.len() >= 2 && s.last() == Some(&1) && x[0] == s[0] && numel(s) > 0).then(
            || {
                let d = (numel(x) / numel(s)).max(1);
                (vec![numel(s), d], vec![d, 1], vec![1, 0])
            },
        )
    };
    rows(a, b).or_else(|| rows(b, a).map(|(dims, sx, ss)| (dims, ss, sx)))
}
