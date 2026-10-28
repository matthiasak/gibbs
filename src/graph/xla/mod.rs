//! XLA through the PJRT C API ([`XlaSession`]): a graph lowers to one StableHLO MLIR module
//! ([`emit::emit`]), which the plugin compiles and runs on its device.
//!
//! Plugin: `GIBBS_PJRT_PLUGIN`, else `$TPU_LIBRARY_PATH`, else `libtpu.so` beside the `libtpu`
//! Python package; otherwise XLA is unavailable. Which ops the plugin compiles is probed once (one
//! module per [`probe::op_key`], [`probe::probe_graphs`]) and cached in
//! `~/.cache/gibbs/host/pjrt-<fnv64 of the plugin path>.json` with the plugin's API version.
//! [`supports_op`] reports it, and a graph holding a rejected op fails to load with the plugin's
//! compile error.
//!
//! Paged KV (`crate::kv`): the host pages stay the source of truth (copy-on-write, forks, rings
//! and eviction all act on them), so each run binds every paged state as a page window gathered
//! through the view's table, padded to a power-of-4 page count (≥ 16) so one executable serves a
//! bucket. The window is donated to the module, written rows scatter into their slots, rows a
//! slot does not hold are masked, and the written rows go back into their host pages.

pub mod emit;
pub mod pjrt;
pub mod probe;

use crate::graph::{DType, Graph, Op, StateStore, Tensor, ValueKind, Weights};
use emit::Ty;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

/// The loaded plugin and its probed op table (`None` = compiles, `Some(error)` = rejected).
pub struct Plugin {
    pub client: pjrt::Client,
    pub ops: HashMap<String, Option<String>>,
}

static PLUGIN: LazyLock<Option<Plugin>> = LazyLock::new(load_plugin);

fn plugin_path() -> Option<String> {
    for var in ["GIBBS_PJRT_PLUGIN", "TPU_LIBRARY_PATH"] {
        if let Some(p) = std::env::var(var).ok().filter(|p| !p.is_empty()) {
            return Some(p);
        }
    }
    let out = std::process::Command::new("python3")
        .args([
            "-c",
            "import libtpu, os; print(os.path.dirname(libtpu.__file__))",
        ])
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    let dir = String::from_utf8(out.stdout).ok()?;
    let p = std::path::Path::new(dir.trim()).join("libtpu.so");
    (out.status.success() && p.exists()).then(|| p.to_string_lossy().into_owned())
}

fn fnv64(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Emits `p`'s module with its own weights and declared state shapes.
fn probe_module(p: &probe::Probe) -> Result<emit::Module, String> {
    let states = p
        .graph
        .values
        .iter()
        .filter_map(|v| match &v.kind {
            ValueKind::State { name, .. } => Some((name.clone(), v.shape.clone())),
            _ => None,
        })
        .collect();
    emit::emit(
        &p.graph,
        &|name| p.weights.get(name).map(|t| (t.len(), Ty::F32)),
        &states,
        &[],
    )
}

/// Compiles `m` and runs it once on zeros: some plugins (jax-metal) compile lazily at the first
/// execution, so a module that only compiled is not yet known to run.
fn probe_run(client: &pjrt::Client, m: &emit::Module) -> Result<(), String> {
    let exe = client.compile(&m.text, m.outputs.len() + m.written.len())?;
    let mut args = Vec::with_capacity(m.params.len());
    for &(n, ty) in &m.params {
        let (elem, size) = match ty {
            Ty::I32 => (pjrt::ElemType::S32, 4),
            Ty::BF16 => (pjrt::ElemType::BF16, 2),
            _ => (pjrt::ElemType::F32, 4),
        };
        args.push(client.upload(&vec![0u8; n * size], elem, &[n])?);
    }
    let sizes: Vec<usize> = m.outputs.iter().map(|&(n, _)| n * 4).collect();
    let refs: Vec<&pjrt::Buffer> = args.iter().collect();
    let outs = client.execute(&exe, &refs, &sizes)?;
    let mut host = vec![0u8; sizes[0]];
    client.download(&outs[0], &mut host)
}

/// The plugin's op table. Probes run in order and the table is written to the cache before
/// each one with that op marked as having aborted the process, so a plugin that crashes on an
/// op's probe leaves it rejected for the next process; a cache for the same plugin path, API
/// version and probe modules is reused.
fn probe_ops(client: &pjrt::Client) -> HashMap<String, Option<String>> {
    let probes = probe::probe_graphs();
    let modules: Vec<(String, Result<emit::Module, String>)> = probes
        .iter()
        .map(|p| (p.key.clone(), probe_module(p)))
        .collect();
    let digest = format!(
        "{:016x}",
        fnv64(
            &modules
                .iter()
                .map(|(_, m)| m.as_ref().map_or_else(|e| e.clone(), |m| m.text.clone()))
                .collect::<String>(),
        )
    );
    let version = format!("{}.{}", client.version.0, client.version.1);
    let file = crate::hardware::default_cache_dir()
        .ok()
        .map(|d| d.join("host").join(format!("pjrt-{:016x}.json", fnv64(&client.path))));
    let mut ops: HashMap<String, Option<String>> = file
        .as_ref()
        .and_then(|f| std::fs::read(f).ok())
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .filter(|j| {
            j["plugin"] == client.path.as_str()
                && j["api"] == version.as_str()
                && j["probes"] == digest.as_str()
        })
        .and_then(|j| serde_json::from_value(j["ops"].clone()).ok())
        .unwrap_or_default();
    let save = |ops: &HashMap<String, Option<String>>| {
        let Some(f) = &file else { return };
        let j = serde_json::json!({
            "plugin": client.path,
            "api": version,
            "platform": client.platform,
            "probes": digest,
            "ops": ops,
        });
        let _ = std::fs::create_dir_all(f.parent().expect("host dir"));
        if let Err(e) = std::fs::write(f, j.to_string()) {
            eprintln!("warning: cannot cache PJRT probes in {}: {e}", f.display());
        }
    };
    let todo: Vec<_> = modules
        .into_iter()
        .filter(|(key, _)| !ops.contains_key(key))
        .collect();
    if todo.is_empty() {
        return ops;
    }
    eprintln!(
        "XLA: probing {} op kinds on {} ({})",
        todo.len(),
        client.platform,
        client.path
    );
    for (key, m) in todo {
        ops.insert(
            key.clone(),
            Some("the plugin aborted the process while running this op's probe".into()),
        );
        save(&ops);
        let verdict = m
            .and_then(|m| probe_run(client, &m))
            .err()
            .map(|e| e.lines().take(3).collect::<Vec<_>>().join(" | "));
        ops.insert(key, verdict);
    }
    save(&ops);
    ops
}

fn load_plugin() -> Option<Plugin> {
    let path = plugin_path()?;
    match pjrt::Client::open(&path) {
        Ok(client) => {
            let ops = probe_ops(&client);
            Some(Plugin { client, ops })
        }
        Err(e) => {
            eprintln!("XLA unavailable: {e}");
            None
        }
    }
}

/// The process-wide plugin, loaded and probed on first use.
pub fn plugin() -> Option<&'static Plugin> {
    PLUGIN.as_ref()
}

/// Whether a PJRT plugin loads, creates a client on an addressable device and was probed.
pub fn is_available() -> bool {
    plugin().is_some()
}

pub fn device_name() -> String {
    match plugin() {
        Some(p) => format!(
            "XLA PJRT {} (API {}.{}, {})",
            p.client.platform, p.client.version.0, p.client.version.1, p.client.path
        ),
        None => "XLA PJRT (no plugin)".to_string(),
    }
}

/// Whether the plugin compiled `op`'s probe module (every dtype: weights upload decoded).
pub fn supports_op(op: &Op, _dtype: DType) -> bool {
    plugin().is_some_and(|p| p.ops.get(&probe::op_key(op)) == Some(&None))
}

fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as their bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// A weight resident on the device: element count and the type the module reads it as.
struct DeviceWeight {
    buf: pjrt::Buffer,
    len: usize,
    ty: Ty,
}

/// Compiled executables by module text hash, least recently used first.
type ExeCache = Vec<(u64, Arc<pjrt::Executable>, Arc<emit::Module>)>;

/// One page class bound for a run: each state's `(name, row width, window)`, which stored row
/// every window slot holds (`-1` empty), the slot of each written query row, and `kv_base`.
struct PagedBind {
    class: usize,
    pos: usize,
    nq: usize,
    names: Vec<(String, usize, Vec<f32>)>,
    slot_row: Vec<i32>,
    write_slot: Vec<i32>,
    base: usize,
    /// Ring classes: rows the ring holds (slot `j` holds every row `≡ j` mod it); 0 = full.
    ring: usize,
}

pub struct XlaSession {
    plugin: &'static Plugin,
    weights: HashMap<String, DeviceWeight>,
    exes: parking_lot::Mutex<ExeCache>,
}

impl XlaSession {
    /// Uploads every weight (decoded; F32 on `METAL`, BF16 elsewhere) after checking that the
    /// plugin accepts every op of `graph`.
    pub fn compile(graph: &Graph, weights: Weights) -> Result<Self, String> {
        let plugin = plugin().ok_or("--backend xla: no PJRT plugin (set GIBBS_PJRT_PLUGIN)")?;
        for node in &graph.nodes {
            let key = probe::op_key(&node.op);
            match plugin.ops.get(&key) {
                Some(None) => {}
                Some(Some(err)) => {
                    return Err(format!(
                        "--backend xla: node '{}' ({key}) is rejected by the PJRT plugin {} ({}): {err}",
                        node.name, plugin.client.platform, plugin.client.path
                    ))
                }
                None => return Err(format!("--backend xla: no probe for op {key}")),
            }
        }
        let f32_weights = plugin.client.platform.eq_ignore_ascii_case("metal");
        let mut dev = HashMap::new();
        for (name, t) in weights.iter() {
            let w = match t {
                Tensor::U32(v) => DeviceWeight {
                    buf: plugin.client.upload(bytes(v), pjrt::ElemType::S32, &[v.len()])?,
                    len: v.len(),
                    ty: Ty::I32,
                },
                t => {
                    let v = t.to_f32();
                    let buf = if f32_weights {
                        plugin.client.upload(bytes(&v), pjrt::ElemType::F32, &[v.len()])?
                    } else {
                        let h: Vec<u16> = v.iter().map(|&x| crate::model::f32_to_bf16(x)).collect();
                        plugin.client.upload(bytes(&h), pjrt::ElemType::BF16, &[v.len()])?
                    };
                    DeviceWeight {
                        buf,
                        len: v.len(),
                        ty: if f32_weights { Ty::F32 } else { Ty::BF16 },
                    }
                }
            };
            dev.insert(name.to_string(), w);
        }
        Ok(Self {
            plugin,
            weights: dev,
            exes: parking_lot::Mutex::new(Vec::new()),
        })
    }

    /// Bound shapes and host data of the states `graph` reads. A KV cache read only by
    /// `CachedAttention` (`[cap, w]`, `cap > 4096`) starts at 4096 rows and grows in 4096-row
    /// steps to cover this run's rows, like the CPU executor; other states take their declared
    /// size.
    fn bind_states(
        graph: &Graph,
        state: &StateStore,
        inputs: &[Tensor],
    ) -> Result<Vec<(String, Vec<usize>, Vec<f32>)>, String> {
        // Rows each KV state must hold this run (`usize::MAX`: not known on the host).
        let mut need: HashMap<&str, usize> = HashMap::new();
        for node in &graph.nodes {
            let Op::CachedAttention { heads, head_dim, .. } = node.op else {
                continue;
            };
            if node.inputs.len() < 6 {
                continue;
            }
            let host = |k: usize| {
                graph
                    .inputs
                    .iter()
                    .position(|&i| i == node.inputs[k])
                    .and_then(|i| inputs[i].as_u32().ok())
            };
            let nq = graph.value(node.inputs[0]).shape.iter().product::<usize>()
                / (heads * head_dim).max(1);
            let rows = match host(5).and_then(|p| p.first().copied()) {
                Some(p) => {
                    let vis = node.inputs.get(6).map(|_| host(6));
                    match vis {
                        Some(None) => usize::MAX,
                        Some(Some(v)) => (p as usize + nq).max(v.iter().copied().max().unwrap_or(0) as usize),
                        None => p as usize + nq,
                    }
                }
                None => usize::MAX,
            };
            for k in [3, 4] {
                if let ValueKind::State { name, .. } = &graph.value(node.inputs[k]).kind {
                    let e = need.entry(name).or_default();
                    *e = (*e).max(rows);
                }
            }
        }
        let mut out: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        for (i, v) in graph.values.iter().enumerate() {
            let ValueKind::State { name, .. } = &v.kind else {
                continue;
            };
            if out.iter().any(|(n, ..)| n == name)
                || !graph
                    .nodes
                    .iter()
                    .any(|n| n.inputs.contains(&crate::graph::ValueId(i as u32)))
            {
                continue;
            }
            if v.shape.is_empty() {
                return Err(format!("state '{name}' has no static shape"));
            }
            if state.kv.as_ref().is_some_and(|kv| kv.has(name)) {
                continue;
            }
            let mut data = state.get(name).map(Tensor::to_f32).unwrap_or_default();
            let kv_only = graph.nodes.iter().all(|n| {
                !n.inputs.contains(&crate::graph::ValueId(i as u32))
                    || matches!(n.op, Op::CachedAttention { .. })
            });
            let shape = match (kv_only, &v.shape[..]) {
                (true, &[cap, w]) if cap > 4096 => {
                    let held = data.len() / w.max(1);
                    let want = need.get(name.as_str()).copied().unwrap_or(cap);
                    let rows = held.max(want.max(1).div_ceil(4096).saturating_mul(4096).min(cap));
                    vec![rows, w]
                }
                _ => v.shape.clone(),
            };
            let n: usize = shape.iter().product();
            if data.len() > n {
                return Err(format!(
                    "state '{name}' holds {} elements, graph binds {shape:?}",
                    data.len()
                ));
            }
            data.resize(n, 0.0);
            out.push((name.clone(), shape, data));
        }
        Ok(out)
    }

    /// Reserves this run's rows in the paged KV view (as the CPU executor, node by node) and
    /// gathers each paged state's page window from the host pages: the class's stored rows
    /// (rings: its ring slots), padded to a power-of-4 page count of at least 16 so one
    /// executable serves a whole bucket.
    fn bind_paged(
        graph: &Graph,
        state: &mut StateStore,
        inputs: &[Tensor],
    ) -> Result<Vec<PagedBind>, String> {
        let Some(kv) = state.kv.as_mut() else {
            return Ok(Vec::new());
        };
        let mut binds: Vec<PagedBind> = Vec::new();
        for node in &graph.nodes {
            let Op::CachedAttention {
                heads,
                head_dim,
                write,
                ..
            } = node.op
            else {
                continue;
            };
            let name = |k: usize| match node.inputs.get(k).map(|&v| &graph.value(v).kind) {
                Some(ValueKind::State { name, .. }) => Some(name.clone()),
                _ => None,
            };
            let (Some(kn), Some(vn)) = (name(3), name(4)) else {
                continue;
            };
            if node.inputs.len() < 6 || !kv.has(&kn) || !kv.has(&vn) {
                continue;
            }
            let pos = graph
                .inputs
                .iter()
                .position(|&i| i == node.inputs[5])
                .and_then(|i| inputs[i].as_u32().ok()?.first().copied())
                .ok_or_else(|| format!("node '{}': paged KV needs host positions", node.name))?
                as usize;
            let nq = graph.value(node.inputs[0]).shape.iter().product::<usize>()
                / (heads * head_dim).max(1);
            let rows = if write { pos..pos + nq } else { pos..pos };
            kv.reserve(pos + nq, rows)
                .map_err(|e| format!("node '{}': {e}", node.name))?;
            let (class, _) = kv.pool().locate(&kn).expect("paged above");
            match binds.iter_mut().find(|b| b.class == class) {
                Some(b) if (b.pos, b.nq) != (pos, nq) => {
                    return Err(format!(
                        "node '{}': one page class at two positions in one graph",
                        node.name
                    ))
                }
                Some(b) => {
                    for n in [kn, vn] {
                        if !b.names.iter().any(|(m, ..)| *m == n) {
                            b.names.push((n, 0, Vec::new()));
                        }
                    }
                }
                None => binds.push(PagedBind {
                    class,
                    pos,
                    nq,
                    names: vec![(kn, 0, Vec::new()), (vn, 0, Vec::new())],
                    slot_row: Vec::new(),
                    write_slot: Vec::new(),
                    ring: 0,
                    base: 0,
                }),
            }
        }
        let kv = state.kv.as_ref().expect("checked above");
        for b in &mut binds {
            let ring = kv.ring(b.class);
            b.base = kv.stored(b.class, b.pos);
            // Every row of the class's pages, as the CPU and Metal executors address them:
            // window slot `j` is page row `j` (zeros on a fresh page), so keys a query may see
            // past the written rows (`visible_end`) read what the page holds.
            let held = kv.tables[b.class].len() * crate::kv::PAGE_TOKENS;
            b.ring = if ring > 0 { held } else { 0 };
            let pages = held / crate::kv::PAGE_TOKENS;
            let mut bucket = 16;
            while bucket < pages {
                bucket *= 4;
            }
            let win = bucket * crate::kv::PAGE_TOKENS;
            b.slot_row = (0..win)
                .map(|j| if j < held { j as i32 } else { -1 })
                .collect();
            b.write_slot = (0..b.nq)
                .map(|i| match ring {
                    0 => (b.base + i) as i32,
                    _ => ((b.base + i) % held) as i32,
                })
                .collect();
            for (name, w, data) in &mut b.names {
                let rows = kv.host_rows(name).expect("paged above");
                if rows.elem != 4 {
                    return Err(format!("XLA reads f32 KV pages only ('{name}')"));
                }
                let (c, i) = kv.pool().locate(name).expect("paged above");
                *w = kv.pool().classes[c].names[i].row_width;
                *data = vec![0f32; win * *w];
                for (j, out) in data.chunks_exact_mut(*w).enumerate().take(held) {
                    rows.read(j, out);
                }
            }
        }
        Ok(binds)
    }

    pub fn run(
        &self,
        graph: &Graph,
        state: &mut StateStore,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>, String> {
        if inputs.len() != graph.inputs.len() {
            return Err(format!(
                "graph has {} inputs, got {}",
                graph.inputs.len(),
                inputs.len()
            ));
        }
        let client = &self.plugin.client;
        let paged = Self::bind_paged(graph, state, &inputs)?;
        let mut bound = Self::bind_states(graph, state, &inputs)?;
        let classes: Vec<emit::PagedClass> = paged
            .iter()
            .map(|b| emit::PagedClass {
                names: b.names.iter().map(|(n, ..)| n.clone()).collect(),
                rows: b.slot_row.len(),
                nq: b.nq,
                ring: b.ring,
            })
            .collect();
        for b in &paged {
            for (n, w, data) in &b.names {
                bound.push((n.clone(), vec![b.slot_row.len(), *w], data.clone()));
            }
        }
        let shapes: HashMap<String, Vec<usize>> = bound
            .iter()
            .map(|(n, s, _)| (n.clone(), s.clone()))
            .collect();
        let module = emit::emit(
            graph,
            &|name| self.weights.get(name).map(|w| (w.len, w.ty)),
            &shapes,
            &classes,
        )?;
        let key = fnv64(&module.text);
        let (exe, module) = {
            let mut exes = self.exes.lock();
            match exes.iter().position(|(k, ..)| *k == key) {
                Some(i) => {
                    let e = exes.remove(i);
                    let r = (e.1.clone(), e.2.clone());
                    exes.push(e);
                    r
                }
                None => {
                    let n = module.outputs.len() + module.written.len();
                    let exe = Arc::new(client.compile(&module.text, n)?);
                    // ponytail: 8 shapes (decode, prefill chunk and tails); more if callers vary more.
                    if exes.len() == 8 {
                        exes.remove(0);
                    }
                    let module = Arc::new(module);
                    exes.push((key, exe.clone(), module.clone()));
                    (exe, module)
                }
            }
        };
        // ponytail: non-paged states live on the host and cross to the device every run; keep
        // them device-resident if XLA decode of recurrent models matters.
        let mut owned: Vec<pjrt::Buffer> = Vec::with_capacity(inputs.len() + bound.len());
        for (&id, t) in graph.inputs.iter().zip(&inputs) {
            let v = graph.value(id);
            let n: usize = v.shape.iter().product();
            if t.len() != n {
                return Err(format!(
                    "input '{}' expects shape {:?}, got {} elements",
                    v.name,
                    v.shape,
                    t.len()
                ));
            }
            owned.push(match (v.dtype, t) {
                (DType::U32, Tensor::U32(d)) => client.upload(bytes(d), pjrt::ElemType::S32, &[n])?,
                (DType::U32, _) => return Err(format!("input '{}' must be u32", v.name)),
                (_, t) => client.upload(bytes(&t.to_f32()), pjrt::ElemType::F32, &[n])?,
            });
        }
        for name in &module.states {
            let (_, shape, data) = bound
                .iter()
                .find(|(n, ..)| n == name)
                .expect("emitted states are bound");
            owned.push(client.upload(bytes(data), pjrt::ElemType::F32, &[shape.iter().product()])?);
        }
        for b in &paged {
            owned.push(client.upload(bytes(&b.slot_row), pjrt::ElemType::S32, &[b.slot_row.len()])?);
            owned.push(client.upload(bytes(&b.write_slot), pjrt::ElemType::S32, &[b.nq])?);
            owned.push(client.upload(bytes(&[b.base as i32]), pjrt::ElemType::S32, &[1])?);
        }
        let n_in = graph.inputs.len();
        let mut args: Vec<&pjrt::Buffer> = owned[..n_in].iter().collect();
        for name in &module.weights {
            args.push(&self.weights[name].buf);
        }
        args.extend(owned[n_in..].iter());
        let sizes: Vec<usize> = module
            .outputs
            .iter()
            .map(|&(n, _)| n * 4)
            .chain(module.written.iter().map(|name| shapes[name].iter().product::<usize>() * 4))
            .collect();
        let results = client.execute(&exe, &args, &sizes)?;
        drop(owned);
        let mut outs = Vec::with_capacity(module.outputs.len());
        for (buf, &(n, ty)) in results.iter().zip(&module.outputs) {
            outs.push(if ty == Ty::I32 {
                let mut v = vec![0u32; n];
                // SAFETY: u32 storage viewed as bytes for the download.
                client.download(buf, unsafe {
                    std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4)
                })?;
                Tensor::U32(v)
            } else {
                let mut v = vec![0f32; n];
                // SAFETY: as above.
                client.download(buf, unsafe {
                    std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4)
                })?;
                Tensor::F32(v)
            });
        }
        for (buf, name) in results[module.outputs.len()..].iter().zip(&module.written) {
            let mut v = vec![0f32; buf.bytes / 4];
            // SAFETY: as above.
            client.download(buf, unsafe {
                std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, buf.bytes)
            })?;
            match paged.iter().find(|b| b.names.iter().any(|(n, ..)| n == name)) {
                // Copy the rows this run wrote back into their host pages.
                Some(b) => {
                    let kv = state.kv.as_ref().expect("paged");
                    let rows = kv.host_rows(name).expect("paged");
                    let w = b.names.iter().find(|(n, ..)| n == name).expect("found").1;
                    for &j in &b.write_slot {
                        let src = &v[j as usize * w..][..w];
                        // SAFETY: a host page row of `w` f32 (checked at bind).
                        unsafe {
                            std::ptr::copy_nonoverlapping(src.as_ptr(), rows.row(j as usize) as *mut f32, w)
                        };
                    }
                }
                None => state.insert(name, Tensor::F32(v)),
            }
        }
        Ok(outs)
    }
}
