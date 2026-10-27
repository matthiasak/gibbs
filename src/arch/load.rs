//! Checkpoint -> canonical weights for a family spec. [`open`] reads any supported checkpoint
//! (GGUF incl. split sets, ggml, safetensors single/sharded incl. the pre-quantized schemes,
//! PyTorch / Flax / NumPy, ONNX initializers, `.ninfer`) into raw tensors; [`load`] picks the
//! family, renames raw tensors to canonical names (`names`), random-initializes `synthetic` weights
//! for synthetic ids, registers PEFT LoRA pairs, applies `transforms` and generates `constants`.
//! [`WeightStore`] answers shapes and dtypes without reading data ([`Source`]) and materializes
//! `graph::Weights` on demand.
//!
//! Tensor variants, as the executors bind them (`graph/metal.rs` weight upload: `Tensor::Mat`
//! packed blocks map no-copy, a plain `Mat` binds as bf16, `F16`/`F32` as themselves; `graph/cpu.rs`
//! decodes `Mat` rows with `Mat::decode_row_f32`):
//! - rank >= 2 in a block format: `Tensor::Mat` over the mapped blocks (`DType::Quant`), through
//!   `reencode_native`; a format no Metal codec/gather reads is decoded to `Tensor::F16`;
//! - rank >= 2 bf16: `Tensor::Mat` borrowing the mapping (`DType::BF16`);
//! - rank >= 2 f32 / f16: `Tensor::F32` / `Tensor::F16`;
//! - rank <= 1, rank >= 4 (convolution kernels) and `to_f32` outputs: `Tensor::F32`;
//! - derived (transformed) tensors keep f32/f16, everything else decodes to an owned bf16 `Mat`,
//!   except row-granular rewrites of packed blocks (`untile_packed_rows`, `stack_experts`).

use super::expr::{eval_json, interp, walk, Env, Val};
use super::gen::{self, Gen, GenData};
use super::names::{Renamer, Rule};
use super::{
    glob, Component, Constant, Ignore, Lora, Pipeline, Registry, Signals, Source, Spec, Transform,
};
use crate::graph::{self, DType, Tensor};
use crate::hardware::QuantFormat as Q;
use crate::model::hub::ModelArtifacts;
use crate::model::loaders::gguf::GgufData;
use crate::model::tokenizer::Tokenizer;
use crate::model::weight_format::{GateReport, ReencodeConfig, ReencodePolicy};
use crate::model::weights::{gathers_on_device, Mat, Packed, Weights};
use crate::model::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};
use crate::model::{ModelManifest, SourceFormat, UniversalModelLoader};
use parking_lot::{Mutex, RwLock};
use serde_json::{Map, Value as J};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// How a tensor is stored (and so which `graph::Tensor` variant it materializes as).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    F32,
    F16,
    Bf16,
    /// Block-format bytes served as `Tensor::Mat` (`Mat::from_packed`).
    Packed(Q),
    U32,
}

impl Kind {
    fn of_format(f: Q) -> Kind {
        match f {
            Q::FP32 => Kind::F32,
            Q::FP16 => Kind::F16,
            Q::BF16 => Kind::Bf16,
            f => Kind::Packed(f),
        }
    }
    fn dtype(self) -> DType {
        match self {
            Kind::F32 => DType::F32,
            Kind::F16 => DType::F16,
            Kind::Bf16 => DType::BF16,
            Kind::Packed(f) => DType::Quant(f),
            Kind::U32 => DType::U32,
        }
    }
    /// The kind of a value computed from this one: half floats stay, packed blocks decode to bf16.
    fn dense(self) -> Kind {
        match self {
            Kind::Packed(_) => Kind::Bf16,
            Kind::U32 => Kind::F32,
            k => k,
        }
    }
}

fn dtype_of(t: &Tensor) -> DType {
    match t {
        Tensor::F32(_) => DType::F32,
        Tensor::U32(_) => DType::U32,
        Tensor::F16(_) => DType::F16,
        Tensor::BF16(_) => DType::BF16,
        Tensor::Mat(m) => m
            .packed()
            .map_or(DType::BF16, |p| Kind::of_format(p.format).dtype()),
    }
}

#[derive(Clone)]
enum Store {
    /// Memory-mapped safetensors / `.ninfer` (`model::weights::Weights` decodes MLX, GLQ, EXL3).
    St(Arc<Weights>),
    /// A manifest over one byte buffer: GGUF / ggml files (mapped), or the buffer `read_checkpoint`
    /// gathered (PyTorch, Flax, NumPy) or the ONNX initializers.
    Buf {
        data: Arc<GgufData>,
        man: Arc<ModelManifest>,
    },
}

#[derive(Clone)]
struct RawTensor {
    store: Store,
    /// Logical row-major shape (GGUF/ggml dims already reversed).
    shape: Vec<usize>,
    kind: Kind,
    /// From the multimodal projector beside the weights ([`index_sidecar`]).
    sidecar: bool,
    /// Stored as IEEE/bf16 floats (not GGUF blocks or MLX/EXL3/GLQ packed integer tensors).
    raw_float: bool,
}

/// An opened checkpoint: signals for matching, scopes for expressions, raw tensors.
pub struct Checkpoint {
    pub sig: Signals,
    /// `config.json` (the `.ninfer` catalog's equivalent), `Null` when absent.
    pub cfg: J,
    /// GGUF / ggml metadata, every key, values as JSON.
    pub gguf: Option<Map<String, J>>,
    /// `json.<file>` documents the source supplies itself; others are read lazily from `dir`.
    pub json: Map<String, J>,
    pub dir: Option<PathBuf>,
    pub weights_path: Option<PathBuf>,
    /// A synthetic id without weights: the spec's `synthetic` table supplies random weights.
    pub synthetic: bool,
    raw: BTreeMap<String, RawTensor>,
    /// The GGUF / ggml bytes (its tokenizer).
    container: Option<Arc<GgufData>>,
    /// Safetensors written by `peft::prepare_causal_lm` (the adapter beside it is merged already).
    lora_merged: bool,
    /// Canonical tensors read through `values(name)` (folded into the graph as constants, e.g.
    /// clamp bounds): consumed like graph weights.
    values_read: Mutex<HashSet<String>>,
}

impl Checkpoint {
    /// Raw (checkpoint) tensor names, sorted.
    pub fn raw_names(&self) -> impl Iterator<Item = &str> {
        self.raw.keys().map(String::as_str)
    }

    /// Logical shape of raw tensor `name`.
    pub fn raw_shape(&self, name: &str) -> Option<&[usize]> {
        self.raw.get(name).map(|t| t.shape.as_slice())
    }

    fn entry(&self, name: &str) -> Result<&RawTensor, String> {
        self.raw
            .get(name)
            .ok_or_else(|| format!("raw tensor `{name}` missing"))
    }

    fn raw_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self.entry(name)?;
        match &t.store {
            Store::St(w) if w.packed_gguf(name).is_some() => Ok(w.mat(name)?.to_f32()),
            Store::St(w) => w.vec(name),
            Store::Buf { data, man } => {
                UniversalModelLoader::extract_tensor_from_manifest(man, data, name)
            }
        }
    }

    fn raw_packed(&self, name: &str) -> Result<Packed, String> {
        let t = self.entry(name)?;
        match &t.store {
            Store::St(w) => w
                .packed_gguf(name)
                .cloned()
                .ok_or_else(|| format!("tensor `{name}` is not packed")),
            Store::Buf { data, man } => {
                let vt = &man.tensors[name];
                if vt.byte_offset + vt.byte_len > data.len() {
                    return Err(format!("tensor `{name}` lies outside its file"));
                }
                Ok(Packed {
                    map: data.clone(),
                    off: vt.byte_offset,
                    format: vt.quant_format,
                    shape: t.shape.clone(),
                })
            }
        }
    }

    /// A bf16 raw tensor without copying it.
    fn raw_mat(&self, name: &str) -> Result<Mat, String> {
        let t = self.entry(name)?;
        match &t.store {
            Store::St(w) => w.mat(name),
            Store::Buf { .. } => {
                let (rows, cols) = rows_cols(&t.shape);
                Ok(Mat::from_packed(self.raw_packed(name)?, rows, cols))
            }
        }
    }
}

/// `[rows, cols]` view of `shape`: blocks run along the last dim.
fn rows_cols(shape: &[usize]) -> (usize, usize) {
    let cols = shape.last().copied().unwrap_or(1).max(1);
    (shape.iter().product::<usize>() / cols, cols)
}

fn read_json(p: &Path) -> Result<J, String> {
    let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_slice(&b).map_err(|e| format!("{}: {e}", p.display()))
}

/// GGUF metadata values are stored as strings: numbers (and `0`/`1` bools), comma-joined
/// numeric arrays, or text.
fn meta_json(v: &str) -> J {
    let num = |s: &str| -> Option<J> {
        s.parse::<i64>().map(J::from).ok().or_else(|| {
            s.parse::<f64>()
                .ok()
                .and_then(|f| serde_json::Number::from_f64(f).map(J::Number))
        })
    };
    if let Some(n) = num(v) {
        return n;
    }
    if v.contains(',') {
        if let Some(a) = v.split(',').map(num).collect::<Option<Vec<_>>>() {
            return J::Array(a);
        }
    }
    J::String(v.to_string())
}

/// Opens `artifacts`' checkpoint: reads its config and indexes every raw tensor (no data read).
pub fn open(artifacts: &ModelArtifacts) -> Result<Checkpoint, String> {
    let weights = artifacts.weights_path.clone();
    let dir = artifacts
        .local_dir
        .clone()
        .or_else(|| {
            weights
                .as_deref()
                .and_then(Path::parent)
                .map(Path::to_path_buf)
        })
        .or_else(|| {
            artifacts
                .config_path
                .as_deref()
                .and_then(Path::parent)
                .map(Path::to_path_buf)
        });
    let config = artifacts.config_path.as_deref().filter(|p| p.is_file());
    let mut ck = Checkpoint {
        sig: Signals::default(),
        cfg: config.map(read_json).transpose()?.unwrap_or(J::Null),
        gguf: None,
        json: Map::new(),
        dir,
        weights_path: weights.clone(),
        synthetic: false,
        raw: BTreeMap::new(),
        container: None,
        lora_merged: false,
        values_read: Mutex::default(),
    };
    match &weights {
        None => ck.synthetic = crate::model::hub::is_synthetic_model_id(&artifacts.model_id),
        Some(p) => {
            open_weights(&mut ck, p, config)?;
            if ck.container.is_some() {
                if let Some(mm) = crate::model::hub::local_mmproj(p) {
                    index_sidecar(&mut ck, &mm)?;
                }
            }
        }
    }
    let cfg = &ck.cfg;
    let architectures: Vec<String> = cfg["architectures"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|a| a.as_str().map(String::from))
        .collect();
    let gguf_arch = ck
        .gguf
        .as_ref()
        .and_then(|g| g.get("general.architecture")?.as_str().map(String::from));
    let diffusers_class = ck
        .dir
        .as_ref()
        .map(|d| d.join("model_index.json"))
        .filter(|p| p.is_file())
        .and_then(|p| read_json(&p).ok())
        .and_then(|j| j["_class_name"].as_str().map(String::from))
        // A diffusers component directory (vae/, unet/, ...) names its class in config.json.
        .or_else(|| cfg["_class_name"].as_str().map(String::from));
    ck.sig = Signals {
        hub_id: artifacts.model_id.clone(),
        file_name: weights
            .as_deref()
            .and_then(Path::file_name)
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        // A synthetic config's `model_type` only labels its size preset (hub.rs writes it).
        hf_model_type: cfg["model_type"]
            .as_str()
            .filter(|_| !ck.synthetic)
            .map(String::from),
        causal_lm: architectures
            .iter()
            .any(|a| a.ends_with("ForCausalLM") || a.ends_with("LMHeadModel"))
            || (cfg.is_null() && gguf_arch.is_some()),
        hf_architectures: architectures,
        gguf_arch,
        diffusers_class,
    };
    Ok(ck)
}

fn open_weights(ck: &mut Checkpoint, p: &Path, config: Option<&Path>) -> Result<(), String> {
    use std::io::Read;
    let name = p
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut head = [0u8; 8];
    let n = std::fs::File::open(p)
        .and_then(|mut f| f.read(&mut head))
        .map_err(|e| format!("{}: {e}", p.display()))?;
    let format = UniversalModelLoader::detect_format(&head[..n]).ok();
    if format == Some(SourceFormat::Gguf) {
        let (m, data) = crate::model::loaders::gguf::open(p)?;
        return Ok(index_buf(ck, m, data, false));
    }
    if format == Some(SourceFormat::Ggml) {
        let (m, data) = crate::model::loaders::ggml::open(p)?;
        return Ok(index_buf(ck, m, GgufData::Buf(data), false));
    }
    if name.ends_with(".ninfer") {
        let (w, cfg) = crate::model::ninfer::Artifact::open(p)?.checkpoint()?;
        ck.cfg = cfg;
        index_weights(ck, w);
        return Ok(());
    }
    if name.ends_with(".onnx") {
        // The ONNX reader splits fused `qkv_proj` at `heads x head_dim` rows (HF config keys).
        let c = &ck.cfg;
        let heads = c["num_attention_heads"]
            .as_u64()
            .ok_or("ONNX: config.json has no num_attention_heads")?;
        let head_dim = c["head_dim"]
            .as_u64()
            .or_else(|| Some(c["hidden_size"].as_u64()? / heads))
            .ok_or("ONNX: config.json has no head_dim / hidden_size")?;
        let mut buf = crate::model::loaders::Checkpoint::default();
        let tensors =
            crate::model::loaders::onnx::extract_hf_tensors(p, (heads * head_dim) as usize)?;
        for (n, (shape, v)) in tensors {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            buf.push(n, shape, "F32", &bytes)?;
        }
        return Ok(index_buf(ck, buf.manifest, GgufData::Buf(buf.data), false));
    }
    if name.ends_with(".safetensors") || name.contains(".safetensors.index") {
        let dir = p.parent().unwrap_or(Path::new("."));
        crate::model::prequant::check(config, dir)?;
        ck.lora_merged = peft_merged(p);
        index_weights(ck, crate::model::prequant::open_weights(p, config, true)?);
        return Ok(());
    }
    let (m, data) = crate::model::loaders::read_checkpoint(p)?;
    Ok(index_buf(ck, m, GgufData::Buf(data), false))
}

/// Whether the safetensors header at `p` carries `peft::prepare_causal_lm`'s merge marker.
fn peft_merged(p: &Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(p) else {
        return false;
    };
    let mut len = [0u8; 8];
    if f.read_exact(&mut len).is_err() {
        return false;
    }
    let mut h = vec![0u8; (u64::from_le_bytes(len) as usize).min(1 << 26)];
    f.read_exact(&mut h).is_ok()
        && serde_json::from_slice::<J>(&h)
            .is_ok_and(|j| j["__metadata__"]["peft_merged"].is_string())
}

/// The GGUF / ggml / read-buffer source adapter. GGUF and ggml list dims fastest first: this is
/// the one place they are reversed to row-major.
fn index_buf(ck: &mut Checkpoint, m: ModelManifest, data: GgufData, sidecar: bool) {
    let container = matches!(m.format, SourceFormat::Gguf | SourceFormat::Ggml);
    let data = Arc::new(data);
    if container {
        let meta = ck.gguf.get_or_insert_with(Map::new);
        for (k, v) in &m.metadata {
            if !sidecar || !meta.contains_key(k) {
                meta.insert(k.clone(), meta_json(v));
            }
        }
        if !sidecar {
            ck.container = Some(data.clone());
        }
    }
    let man = Arc::new(m);
    for (name, vt) in &man.tensors {
        let shape: Vec<usize> = if container {
            vt.shape.iter().rev().copied().collect()
        } else {
            vt.shape.clone()
        };
        let kind = if container {
            match Kind::of_format(vt.quant_format) {
                Kind::Packed(f) if crate::graph::codec::codec(f).is_none() => Kind::F16,
                k => k,
            }
        } else {
            match vt.dtype.as_str() {
                "BF16" => Kind::Bf16,
                "F16" => Kind::F16,
                _ => Kind::F32,
            }
        };
        ck.raw.insert(
            name.clone(),
            RawTensor {
                store: Store::Buf {
                    data: data.clone(),
                    man: man.clone(),
                },
                shape,
                kind,
                sidecar,
                raw_float: !container,
            },
        );
    }
}

/// The multimodal projector GGUF beside a GGUF's weights (`mmproj*.gguf`): its tensors
/// join the raw tensors, seen only by specs with `runtime.media`, and its metadata keys the
/// `gguf` scope (the weights file's own keys win, `general.architecture` among them).
fn index_sidecar(ck: &mut Checkpoint, p: &Path) -> Result<(), String> {
    let (m, data) = crate::model::loaders::gguf::open(p)?;
    if let Some(dup) = m.tensors.keys().find(|n| ck.raw.contains_key(*n)) {
        return Err(format!(
            "{}: tensor `{dup}` is also in the weights file",
            p.display()
        ));
    }
    index_buf(ck, m, data, true);
    Ok(())
}

fn index_weights(ck: &mut Checkpoint, w: Weights) {
    let w = Arc::new(w);
    let names: Vec<String> = w.names().map(String::from).collect();
    for name in names {
        let raw_float = w.packed_gguf(&name).is_none()
            && matches!(w.stored_dtype(&name), Some("F32" | "F16" | "BF16"));
        let (shape, kind) = match w.packed_gguf(&name) {
            Some(p) if gathers_on_device(p.format) => (p.shape.clone(), Kind::of_format(p.format)),
            Some(p) => (p.shape.clone(), Kind::F16),
            None => {
                let Ok(shape) = w.shape(&name) else { continue };
                let kind = match w.stored_dtype(&name) {
                    Some("F32") => Kind::F32,
                    Some("F16") => Kind::F16,
                    _ if w.decodes_to_bf16(&name) => Kind::Bf16,
                    _ => Kind::F32,
                };
                (shape, kind)
            }
        };
        ck.raw.insert(
            name,
            RawTensor {
                store: Store::St(w.clone()),
                shape,
                kind,
                sidecar: false,
                raw_float,
            },
        );
    }
}

// ---- canonical tensors ------------------------------------------------------------------------

struct Node {
    /// Logical shape (known without reading data).
    shape: Vec<usize>,
    kind: Kind,
    /// A convolution kernel (`to_f32`): always full-shape F32, never re-stored by `--quant`.
    conv: bool,
    src: Src,
}

enum Src {
    Raw(String),
    /// A PEFT adapter tensor.
    Adapter(String),
    Gen(Arc<Gen>),
    /// Random init (synthetic ids), by seed.
    Synth(u64),
    Xf {
        op: String,
        args: J,
        ins: Vec<Arc<Node>>,
        part: usize,
    },
    /// An output of the spec's `precompute` component, and the tensors that component read.
    Pre {
        data: Vec<f32>,
        ins: Vec<Arc<Node>>,
    },
}

type Table = Arc<RwLock<BTreeMap<String, Arc<Node>>>>;

/// Raw leaves `n` reads.
fn leaves(n: &Node, out: &mut HashSet<String>) {
    match &n.src {
        Src::Raw(r) => {
            out.insert(r.clone());
        }
        Src::Xf { ins, .. } | Src::Pre { ins, .. } => ins.iter().for_each(|i| leaves(i, out)),
        _ => {}
    }
}

/// `parent` with extra variables (rename captures, loop vars).
struct Vars<'a> {
    parent: &'a dyn Env,
    vars: &'a [(String, Val)],
}

impl Env for Vars<'_> {
    fn lookup(&self, root: &str, path: &[Val]) -> Option<Val> {
        match self.vars.iter().find(|(k, _)| k == root) {
            Some((_, v)) => walk(v.clone(), path),
            None => self.parent.lookup(root, path),
        }
    }
    fn has(&self, t: &str) -> bool {
        self.parent.has(t)
    }
    fn shape(&self, t: &str) -> Option<Vec<usize>> {
        self.parent.shape(t)
    }
    fn block(&self, t: &str) -> Option<usize> {
        self.parent.block(t)
    }
    fn raw_has(&self, p: &str) -> bool {
        self.parent.raw_has(p)
    }
    fn values(&self, t: &str) -> Option<Vec<f32>> {
        self.parent.values(t)
    }
    fn dtype(&self, t: &str) -> Option<&'static str> {
        self.parent.dtype(t)
    }
    fn token(&self, s: &str) -> Option<u32> {
        self.parent.token(s)
    }
}

/// Follows `path` into `j` without cloning what it skips.
fn walk_json(mut j: &J, path: &[Val]) -> Option<Val> {
    for seg in path {
        j = match (j, seg) {
            (J::Object(o), Val::Str(k)) => o.get(k)?,
            (J::Object(o), Val::Num(n)) => o.get(&(*n as i64).to_string())?,
            (J::Array(a), Val::Num(n)) => {
                let i = if *n < 0.0 {
                    a.len() as i64 + *n as i64
                } else {
                    *n as i64
                };
                a.get(usize::try_from(i).ok()?)?
            }
            _ => return None,
        };
    }
    Some(Val::from_json(j))
}

/// The expression scope of a loaded model (see the module docs of `expr`): spec `config` vars
/// (lazy, memoized; a cycle is an error), `cfg.<path>`, `gguf.<key>` (`<general.architecture>.<key>`
/// first), `json.<file>.<path>`, `runtime.<path>`; `has`/`shape`/`values` see canonical tensors,
/// `token` the tokenizer's vocabulary.
pub struct ModelEnv {
    config: Map<String, J>,
    runtime: RwLock<J>,
    ckpt: Arc<Checkpoint>,
    table: Table,
    tok: Option<Arc<Tokenizer>>,
    /// Successfully evaluated config vars.
    memo: Mutex<HashMap<String, Val>>,
    files: Mutex<HashMap<String, Option<J>>>,
}

/// One evaluation over a [`ModelEnv`]: the vars being evaluated (cycle detection) and the first
/// error hit inside `lookup` (which can only answer `None`), reported by the entry points.
/// Per evaluation, so concurrent evaluations on one model never see each other's state.
struct Eval<'a> {
    m: &'a ModelEnv,
    busy: std::cell::RefCell<Vec<String>>,
    err: std::cell::RefCell<Option<String>>,
}

impl<'a> Eval<'a> {
    fn new(m: &'a ModelEnv) -> Self {
        Self {
            m,
            busy: Default::default(),
            err: Default::default(),
        }
    }

    fn run<R>(m: &ModelEnv, f: impl FnOnce(&Eval) -> Result<R, String>) -> Result<R, String> {
        let e = Eval::new(m);
        let r = f(&e);
        match e.err.into_inner() {
            Some(err) => Err(err),
            None => r,
        }
    }

    fn record(&self, e: String) {
        self.err.borrow_mut().get_or_insert(e);
    }

    fn var(&self, name: &str) -> Option<Val> {
        if let Some(v) = self.m.memo.lock().get(name) {
            return Some(v.clone());
        }
        if self.busy.borrow().iter().any(|b| b == name) {
            let chain = self.busy.borrow().join(" -> ");
            self.record(format!("config var cycle: {chain} -> {name}"));
            return None;
        }
        self.busy.borrow_mut().push(name.to_string());
        let clean = self.err.borrow().is_none();
        let r = eval_json(&self.m.config[name], self);
        self.busy.borrow_mut().pop();
        match r {
            Ok(v) => {
                if clean && self.err.borrow().is_none() {
                    self.m.memo.lock().insert(name.to_string(), v.clone());
                }
                Some(v)
            }
            Err(e) => {
                self.record(format!("config `{name}`: {e}"));
                None
            }
        }
    }
}

impl Env for Eval<'_> {
    fn lookup(&self, root: &str, path: &[Val]) -> Option<Val> {
        let m = self.m;
        if m.config.contains_key(root) {
            return walk(self.var(root)?, path);
        }
        match root {
            "cfg" => walk_json(&m.ckpt.cfg, path),
            "gguf" => {
                let g = m.ckpt.gguf.as_ref()?;
                let Val::Str(k) = path.first()? else {
                    return None;
                };
                let arch = g.get("general.architecture").and_then(J::as_str);
                let j = arch
                    .and_then(|a| g.get(&format!("{a}.{k}")))
                    .or_else(|| g.get(k))?;
                walk_json(j, &path[1..])
            }
            "json" => {
                let Val::Str(f) = path.first()? else {
                    return None;
                };
                walk_json(&m.file(f)?, &path[1..])
            }
            "runtime" => walk_json(&m.runtime.read(), path),
            _ => None,
        }
    }
    fn has(&self, t: &str) -> bool {
        self.m.has(t)
    }
    fn shape(&self, t: &str) -> Option<Vec<usize>> {
        self.m.shape(t)
    }
    fn block(&self, t: &str) -> Option<usize> {
        self.m.block(t)
    }
    fn raw_has(&self, p: &str) -> bool {
        self.m.raw_has(p)
    }
    fn values(&self, t: &str) -> Option<Vec<f32>> {
        self.m.values(t)
    }
    fn dtype(&self, t: &str) -> Option<&'static str> {
        self.m.dtype(t)
    }
    fn token(&self, s: &str) -> Option<u32> {
        self.m.token(s)
    }
}

impl ModelEnv {
    fn new(spec: &Spec, ckpt: Arc<Checkpoint>, table: Table, tok: Option<Arc<Tokenizer>>) -> Self {
        Self {
            config: spec.config.clone(),
            runtime: RwLock::new(if spec.runtime.is_null() {
                J::Object(Map::new())
            } else {
                spec.runtime.clone()
            }),
            ckpt,
            table,
            tok,
            memo: Mutex::new(HashMap::new()),
            files: Mutex::new(HashMap::new()),
        }
    }

    /// Evaluates a spec value (strings are expressions).
    pub fn eval(&self, j: &J) -> Result<Val, String> {
        Eval::run(self, |e| eval_json(j, e))
    }

    /// [`Self::eval`] with extra variables bound.
    pub fn eval_with(&self, vars: &[(String, Val)], j: &J) -> Result<Val, String> {
        Eval::run(self, |e| eval_json(j, &Vars { parent: e, vars }))
    }

    /// `{expr}` interpolation of `s`.
    pub fn interp(&self, s: &str) -> Result<String, String> {
        self.interp_with(&[], s)
    }

    pub fn interp_with(&self, vars: &[(String, Val)], s: &str) -> Result<String, String> {
        Eval::run(self, |e| interp(s, &Vars { parent: e, vars }))
    }

    /// Config variable `name`.
    pub fn var(&self, name: &str) -> Result<Val, String> {
        if !self.config.contains_key(name) {
            return Err(format!("no config var `{name}`"));
        }
        Eval::run(self, |e| Ok(e.var(name).unwrap_or(Val::Null)))
    }

    /// Sets `runtime.ctx` (the context length the caller runs with); memoized vars are recomputed.
    pub fn set_ctx(&self, ctx: J) {
        self.runtime.write()["ctx"] = ctx;
        self.memo.lock().clear();
    }

    /// `json.<file>`: the source's own document, else `<dir>/<file>.json`, else `<dir>/*/<file>.json`.
    fn file(&self, f: &str) -> Option<J> {
        if let Some(j) = self.ckpt.json.get(f) {
            return Some(j.clone());
        }
        if !f.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return None;
        }
        let mut files = self.files.lock();
        if let Some(j) = files.get(f) {
            return j.clone();
        }
        let file = format!("{f}.json");
        let found = self.ckpt.dir.as_ref().and_then(|d| {
            let direct = d.join(&file);
            if direct.is_file() {
                return Some(direct);
            }
            let mut subs: Vec<PathBuf> = std::fs::read_dir(d)
                .ok()?
                .flatten()
                .map(|e| e.path().join(&file))
                .filter(|p| p.is_file())
                .collect();
            subs.sort();
            subs.into_iter().next()
        });
        let j = found.and_then(|p| read_json(&p).ok());
        files.insert(f.to_string(), j.clone());
        j
    }
}

/// As `&dyn Env` (the interpreter's model scope) each lookup is its own evaluation: a failing
/// config var reads as undefined there; [`ModelEnv::var`] / [`ModelEnv::eval`] report why.
impl Env for ModelEnv {
    fn lookup(&self, root: &str, path: &[Val]) -> Option<Val> {
        Eval::new(self).lookup(root, path)
    }
    fn has(&self, t: &str) -> bool {
        self.table.read().contains_key(t)
    }
    fn shape(&self, t: &str) -> Option<Vec<usize>> {
        self.table.read().get(t).map(|n| n.shape.clone())
    }
    fn block(&self, t: &str) -> Option<usize> {
        self.table.read().get(t).map(|n| match n.kind {
            Kind::Packed(f) => crate::model::layers::gguf_block_codec(f).map_or(1, |b| b.0),
            _ => 1,
        })
    }
    fn raw_has(&self, p: &str) -> bool {
        self.ckpt.raw.keys().any(|r| glob(p, r))
    }
    fn values(&self, t: &str) -> Option<Vec<f32>> {
        let n = self.table.read().get(t).cloned()?;
        let v = (n.shape.iter().product::<usize>() <= SMALL_VALUES)
            .then(|| node_values(&self.ckpt, &n))
            .flatten()?;
        self.ckpt.values_read.lock().insert(t.to_string());
        Some(v)
    }
    fn dtype(&self, t: &str) -> Option<&'static str> {
        Some(match self.table.read().get(t)?.kind {
            Kind::F32 => "f32",
            Kind::F16 => "f16",
            Kind::Bf16 => "bf16",
            Kind::U32 => "u32",
            Kind::Packed(_) => "quant",
        })
    }
    fn token(&self, s: &str) -> Option<u32> {
        self.tok.as_ref()?.vocab.get(s).copied()
    }
}

/// The most elements `values(name)` reads.
const SMALL_VALUES: usize = 1 << 16;

/// `n` as f32 values from checkpoint data, generators and transforms (`None` for adapter,
/// synthetic and precomputed tensors, which exist only once the store is built).
fn node_values(ckpt: &Checkpoint, n: &Node) -> Option<Vec<f32>> {
    match &n.src {
        Src::Raw(r) => ckpt.raw_f32(r).ok(),
        Src::Gen(g) => Some(match &g.data {
            GenData::F32(v) => v.clone(),
            GenData::U32(v) => v.iter().map(|&x| x as f32).collect(),
        }),
        Src::Xf {
            op,
            args,
            ins,
            part,
        } => {
            let vals = ins
                .iter()
                .map(|i| node_values(ckpt, i))
                .collect::<Option<Vec<_>>>()?;
            xf_dense(op, args, ins, vals, *part).ok()
        }
        Src::Pre { data, .. } => Some(data.clone()),
        Src::Adapter(_) | Src::Synth(_) => None,
    }
}

/// Canonical weights of a model: checkpoint tensors after renames and transforms, constants,
/// synthetic and LoRA tensors.
pub struct WeightStore {
    ckpt: Arc<Checkpoint>,
    table: Table,
    lora: HashMap<String, Lora>,
    adapter: Option<Arc<(ModelManifest, Vec<u8>)>>,
    reencode: ReencodeConfig,
    reencode_key: Option<String>,
    reencode_cache: Option<Arc<crate::model::reencode::ReencodeCache>>,
    /// Derived tensors (transforms, constants, re-encodes), computed once.
    // ponytail: a derived F32 tensor is held here and copied into each `materialize` result; hold
    // `Arc` data in `graph::Tensor` if large derived f32 tensors appear.
    cache: Mutex<HashMap<String, Tensor>>,
}

impl WeightStore {
    fn node(&self, name: &str) -> Option<Arc<Node>> {
        let name = name
            .strip_suffix(crate::arch::build::LINEAR_COPY)
            .unwrap_or(name);
        self.table.read().get(name).cloned()
    }

    /// Every canonical tensor name, sorted.
    pub fn names(&self) -> Vec<String> {
        self.table.read().keys().cloned().collect()
    }

    /// Kind after the rank rules and `--quant`.
    fn final_kind(&self, n: &Node) -> Kind {
        let r = n.shape.len();
        if n.kind == Kind::U32 {
            return Kind::U32;
        }
        if r <= 1 || n.conv || (r >= 4 && !matches!(n.kind, Kind::Packed(_))) {
            return Kind::F32;
        }
        match self.reencode.force {
            Some(f) if r == 2 => {
                Kind::of_format(crate::model::reencode::fit(f, n.shape[1], Q::FP16))
            }
            _ => n.kind,
        }
    }

    /// The graph tensors of `names` (what `build::Instance::weights` lists).
    pub fn materialize(&self, names: &[String]) -> Result<graph::Weights, String> {
        let mut w = graph::Weights::new();
        for n in names {
            w.insert(n.clone(), self.get(n)?);
        }
        Ok(w)
    }

    fn get(&self, name: &str) -> Result<Tensor, String> {
        if let Some(t) = self.cache.lock().get(name) {
            return Ok(t.clone());
        }
        let n = self
            .node(name)
            .ok_or_else(|| format!("no tensor `{name}`"))?;
        let k = self.final_kind(&n);
        let t = self
            .build(name, &n, k)
            .map_err(|e| format!("tensor `{name}`: {e}"))?;
        let passthrough = matches!(n.src, Src::Raw(_))
            && k == n.kind
            && !matches!(k, Kind::Packed(_))
            && !matches!(&t, Tensor::Mat(m) if m.packed().is_some());
        if !passthrough {
            self.cache.lock().insert(name.to_string(), t.clone());
        }
        Ok(t)
    }

    /// Whether the re-encode policy decides the stored format of `n` (as `k`): matrices of blocks
    /// and float weight matrices (not generated constants or convolution kernels).
    fn reencodable(&self, n: &Node, k: Kind) -> bool {
        n.shape.len() == 2
            && self.reencode.force.is_none()
            && n.kind == k
            && match k {
                Kind::Packed(_) => true,
                Kind::F32 | Kind::F16 | Kind::Bf16 => {
                    !n.conv
                        && matches!(&n.src, Src::Raw(r) if self.ckpt.raw.get(r).is_some_and(|t| t.raw_float))
                }
                Kind::U32 => false,
            }
    }

    fn cached_reencode(
        &self,
        name: &str,
        src: Q,
        rows: usize,
        cols: usize,
        shape: &[usize],
    ) -> Option<Packed> {
        if let Some(cache) = self.reencode_cache.as_ref() {
            let ct = cache.index.get(name)?;
            let valid_fmt = !crate::model::reencode::is_float(ct.format)
                && crate::graph::codec::codec(ct.format).is_some()
                && cols % crate::graph::codec::block_geometry(ct.format).map_or(1, |g| g.0) == 0;
            let valid = valid_fmt
                && ct.rows == rows
                && ct.cols == cols
                && ct.bytes == rows * ct.format.tensor_bytes(cols)
                && ct.offset % 64 == 0
                && ct
                    .offset
                    .checked_add(ct.bytes)
                    .is_some_and(|end| end <= cache.map.len());
            if valid {
                crate::model::reencode::record_cache_hit(
                    &cache.key,
                    name,
                    src,
                    ct.format,
                    cache.map.clone(),
                    ct.offset,
                    ct.bytes,
                    rows,
                    cols,
                );
                return Some(Packed {
                    map: cache.map.clone(),
                    off: ct.offset,
                    format: ct.format,
                    shape: shape.to_vec(),
                });
            }
            crate::model::reencode::record_cache_mismatch();
            return None;
        }
        if let Some(key) = &self.reencode_key {
            if let Some(pt) = crate::model::reencode::reuse_retained_pending(
                key,
                &self.reencode,
                name,
                src,
                rows,
                cols,
            ) {
                return Some(Packed {
                    map: pt.map,
                    off: pt.off,
                    format: pt.format,
                    shape: shape.to_vec(),
                });
            }
        }
        None
    }

    fn build(&self, name: &str, n: &Node, k: Kind) -> Result<Tensor, String> {
        let (rows, cols) = rows_cols(&n.shape);
        let float = match k {
            Kind::F32 => Some(Q::FP32),
            Kind::F16 => Some(Q::FP16),
            Kind::Bf16 => Some(Q::BF16),
            _ => None,
        };
        // Bit-exact keeps float matrices as stored (its rule would block-encode codec-less sources).
        let lossy = self.reencode.policy != ReencodePolicy::BitExact;
        if let (Some(src), true) = (float, lossy && self.reencodable(n, k)) {
            if let Some(p) = self.cached_reencode(name, src, rows, cols, &n.shape) {
                return Ok(Tensor::Mat(Mat::from_packed(p, rows, cols)));
            }
            let r = crate::model::reencode::reencode_values(
                &self.reencode,
                name,
                src,
                rows,
                cols,
                || self.dense(n).ok(),
            );
            if let Some((f, blocks)) = r {
                let len = blocks.len();
                let map = Arc::new(GgufData::Buf(blocks));
                if let Some(key) = &self.reencode_key {
                    crate::model::reencode::record_pending_tensor(
                        key,
                        name,
                        f,
                        map.clone(),
                        0,
                        len,
                        rows,
                        cols,
                    );
                }
                let p = Packed {
                    map,
                    off: 0,
                    format: f,
                    shape: n.shape.clone(),
                };
                return Ok(Tensor::Mat(Mat::from_packed(p, rows, cols)));
            }
        }
        Ok(match k {
            Kind::U32 => match &n.src {
                Src::Gen(g) => match &g.data {
                    GenData::U32(v) => Tensor::U32(v.clone()),
                    GenData::F32(_) => return Err("f32 generator data for a u32 tensor".into()),
                },
                _ => return Err("only generators produce u32 tensors".into()),
            },
            Kind::F32 => Tensor::F32(self.dense(n)?),
            Kind::F16 => Tensor::F16(self.dense(n)?.into_iter().map(f32_to_f16).collect()),
            Kind::Bf16 => match &n.src {
                Src::Raw(r) if n.kind == Kind::Bf16 => Tensor::Mat(self.ckpt.raw_mat(r)?),
                _ => Tensor::Mat(Mat::from_bf16(
                    self.dense(n)?.into_iter().map(f32_to_bf16).collect(),
                    rows,
                    cols,
                )),
            },
            Kind::Packed(f) if n.kind == Kind::Packed(f) => {
                let mut p = self.packed(n)?;
                // Rank-3 banks regrouped to rows (`reshape`) stay as stored, as their rank-3 raws.
                let regrouped = matches!(&n.src, Src::Xf { op, .. } if op == "reshape");
                if n.shape.len() == 2 && self.reencode.force.is_none() && !regrouped {
                    if let Some(pr) = &self.reencode.progress {
                        pr.total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    if let Some(cp) = lossy
                        .then(|| self.cached_reencode(name, f, rows, cols, &p.shape))
                        .flatten()
                    {
                        p = cp;
                    } else {
                        let raw = p
                            .map
                            .get(p.off..p.off + rows * f.tensor_bytes(cols))
                            .ok_or("packed blocks lie outside their buffer")?;
                        if let Some((f2, blocks)) = crate::model::reencode::reencode_native(
                            &self.reencode,
                            name,
                            f,
                            rows,
                            cols,
                            raw,
                        ) {
                            let len = blocks.len();
                            let map = Arc::new(GgufData::Buf(blocks));
                            if let Some(key) = &self.reencode_key {
                                crate::model::reencode::record_pending_tensor(
                                    key,
                                    name,
                                    f2,
                                    map.clone(),
                                    0,
                                    len,
                                    rows,
                                    cols,
                                );
                            }
                            p = Packed {
                                map,
                                off: 0,
                                format: f2,
                                shape: p.shape,
                            };
                        }
                    }
                    if let Some(pr) = &self.reencode.progress {
                        pr.done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                Tensor::Mat(Mat::from_packed(p, rows, cols))
            }
            // `--quant` into a block format.
            Kind::Packed(f) => {
                let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
                let raw =
                    crate::model::reencode::encode_weight(&self.dense(n)?, cols, rows, f, threads);
                Tensor::Mat(Mat::from_packed_bytes(f, raw, rows, cols))
            }
        })
    }

    /// `n` as block bytes (its kind is `Packed`).
    fn packed(&self, n: &Node) -> Result<Packed, String> {
        let Kind::Packed(f) = n.kind else {
            return Err("not a packed tensor".into());
        };
        match &n.src {
            Src::Raw(r) => self.ckpt.raw_packed(r),
            Src::Xf { op, args, ins, .. } if op == "untile_packed_rows" => {
                let p = self.packed(&ins[0])?;
                let (rows, cols) = rows_cols(&n.shape);
                let row_bytes = f.tensor_bytes(cols);
                let data = p
                    .map
                    .get(p.off..p.off + rows * row_bytes)
                    .ok_or("packed blocks lie outside their buffer")?;
                let u = Untile::from(args)?;
                Ok(Packed {
                    map: Arc::new(GgufData::Buf(untile_rows(data, row_bytes, &u))),
                    off: 0,
                    format: f,
                    shape: n.shape.clone(),
                })
            }
            Src::Xf { op, ins, .. } if op == "to_bf16" => self.packed(&ins[0]),
            Src::Xf { op, ins, .. } if op == "reshape" => {
                if ins[0].shape.last() != n.shape.last() {
                    return Err(format!(
                        "reshape {:?} -> {:?} of packed blocks must keep the row width",
                        ins[0].shape, n.shape
                    ));
                }
                Ok(Packed {
                    shape: n.shape.clone(),
                    ..self.packed(&ins[0])?
                })
            }
            Src::Xf { op, ins, .. } if op == "stack_experts" => {
                let mut bytes = Vec::new();
                for i in ins {
                    let p = self.packed(i)?;
                    let len = f.tensor_bytes(i.shape.iter().product());
                    bytes.extend_from_slice(
                        p.map
                            .get(p.off..p.off + len)
                            .ok_or("packed blocks lie outside their buffer")?,
                    );
                }
                Ok(Packed {
                    map: Arc::new(GgufData::Buf(bytes)),
                    off: 0,
                    format: f,
                    shape: n.shape.clone(),
                })
            }
            _ => Err("not a packed tensor".into()),
        }
    }

    /// `n` decoded to row-major f32.
    fn dense(&self, n: &Node) -> Result<Vec<f32>, String> {
        match &n.src {
            Src::Raw(r) => self.ckpt.raw_f32(r),
            Src::Adapter(a) => {
                let (m, d) = &**self.adapter.as_ref().ok_or("no adapter loaded")?;
                UniversalModelLoader::extract_tensor_from_manifest(m, d, a)
            }
            Src::Gen(g) => Ok(match &g.data {
                GenData::F32(v) => v.clone(),
                GenData::U32(v) => v.iter().map(|&x| x as f32).collect(),
            }),
            Src::Synth(seed) => Ok(synth_values(n.shape.iter().product(), *seed)),
            Src::Pre { data, .. } => Ok(data.clone()),
            Src::Xf {
                op,
                args,
                ins,
                part,
            } => {
                let vals = ins
                    .iter()
                    .map(|i| self.dense(i))
                    .collect::<Result<Vec<_>, _>>()?;
                xf_dense(op, args, ins, vals, *part)
            }
        }
    }

    /// Raw checkpoint tensors that feed none of the `used` canonical names, are not companions
    /// of a used `X.weight` (`X.scales`, `X.weight_scale`, ...: quantization side tensors) and
    /// match no `ignore` glob (tested against the raw and the canonical name).
    pub fn unconsumed(&self, used: &HashSet<String>, ignore: &[String]) -> Vec<String> {
        let table = self.table.read();
        let mut fed = HashSet::new();
        for u in used {
            let u = u.strip_suffix(crate::arch::build::LINEAR_COPY).unwrap_or(u);
            if let Some(n) = table.get(u) {
                leaves(n, &mut fed);
            }
        }
        for u in self.ckpt.values_read.lock().iter() {
            if let Some(n) = table.get(u) {
                leaves(n, &mut fed);
            }
        }
        // Raw name → the canonical names it feeds (through transforms too).
        let mut canonical: HashMap<String, Vec<&str>> = HashMap::new();
        for (c, n) in table.iter() {
            let mut raws = HashSet::new();
            leaves(n, &mut raws);
            for r in raws {
                canonical.entry(r).or_default().push(c.as_str());
            }
        }
        // ponytail: any sibling of a consumed `X.weight` but `X.bias` counts as its companion;
        // ask the source for real companion lists if this hides an unused tensor.
        let companion = |r: &str| {
            r.rsplit_once('.').is_some_and(|(stem, s)| {
                s != "weight" && s != "bias" && fed.contains(&format!("{stem}.weight"))
            })
        };
        self.ckpt
            .raw
            .iter()
            // Projector sidecars are optional: a tower that cannot read one leaves its modality off.
            .filter(|(_, t)| !t.sidecar)
            .map(|(r, _)| r)
            .filter(|r| !fed.contains(*r) && !companion(r))
            .filter(|r| {
                !ignore.iter().any(|g| {
                    glob(g, r)
                        || canonical
                            .get(r.as_str())
                            .is_some_and(|cs| cs.iter().any(|c| glob(g, c)))
                })
            })
            .cloned()
            .collect()
    }
}

impl WeightStore {
    /// Registers a low-rank adapter `y += scale · (x·Aᵀ)·Bᵀ (+ bias)` from an external file on
    /// canonical weight `target` (A `[r, in...]`, B `[out, r, (1, 1)]`): a matrix gets the
    /// interpreter's LoRA branch ([`Source::lora`]); any other weight (a convolution kernel) has
    /// `scale · B·A` merged in at load (`lora_merge`), where a bias delta is refused.
    pub fn add_lora(
        &mut self,
        target: &str,
        a: (Vec<usize>, Vec<f32>),
        b: (Vec<usize>, Vec<f32>),
        scale: f32,
        bias: Option<Vec<f32>>,
    ) -> Result<(), String> {
        let base = self
            .node(target)
            .ok_or_else(|| format!("LoRA target `{target}` is not a weight of this model"))?;
        let gen = |(shape, v): (Vec<usize>, Vec<f32>)| Node {
            shape: shape.clone(),
            kind: Kind::F32,
            conv: false,
            src: Src::Gen(Arc::new(Gen {
                shape,
                data: GenData::F32(v),
            })),
        };
        let (na, nb) = (format!("{target}.lora_A"), format!("{target}.lora_B"));
        // Validates the factor shapes against the base.
        xf_shapes("lora_merge", &J::Null, &[&base.shape, &a.0, &b.0])
            .map_err(|e| format!("LoRA on `{target}`: {e}"))?;
        insert_new(&self.table, na.clone(), gen(a))
            .map_err(|e| format!("LoRA on `{target}`: {e}"))?;
        insert_new(&self.table, nb.clone(), gen(b))
            .map_err(|e| format!("LoRA on `{target}`: {e}"))?;
        if base.shape.len() == 2 {
            let bias_delta = match bias {
                Some(v) => {
                    let n = format!("{target}.lora_bias");
                    insert_new(&self.table, n.clone(), gen((vec![v.len()], v)))?;
                    Some(n)
                }
                None => None,
            };
            self.lora.insert(
                target.to_string(),
                Lora {
                    a: na,
                    b: nb,
                    scale,
                    bias_delta,
                },
            );
            return Ok(());
        }
        if bias.is_some() {
            return Err(format!(
                "LoRA on `{target}`: bias deltas on non-matrix weights are not supported"
            ));
        }
        let ins = vec![
            base.clone(),
            self.node(&na).unwrap(),
            self.node(&nb).unwrap(),
        ];
        let kind = xf_kind("lora_merge", &ins);
        let merged = Node {
            shape: base.shape.clone(),
            kind,
            conv: base.conv,
            src: Src::Xf {
                op: "lora_merge".into(),
                args: serde_json::json!({"scale": scale}),
                ins,
                part: 0,
            },
        };
        self.table
            .write()
            .insert(target.to_string(), Arc::new(merged));
        self.cache.lock().remove(target);
        Ok(())
    }
}

impl Source for WeightStore {
    fn tensor(&self, name: &str) -> Option<(Vec<usize>, DType)> {
        let n = self.node(name)?;
        let k = self.final_kind(&n);
        // A re-encode policy picks the stored format from the data: decide it now (cached).
        if self.reencodable(&n, k) && self.reencode.policy != ReencodePolicy::BitExact {
            match self.get(name) {
                Ok(t) => return Some((n.shape.clone(), dtype_of(&t))),
                Err(e) => tracing::warn!("{e}"),
            }
        }
        Some((n.shape.clone(), k.dtype()))
    }

    fn lora(&self, weight: &str) -> Option<Lora> {
        self.lora.get(weight).cloned()
    }
}

fn synth_values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * 0.02
        })
        .collect()
}

fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

// ---- transforms -------------------------------------------------------------------------------

/// `untile_packed_rows` args: row blocks of `v_heads` heads in tiled order
/// (`[v_per_k, k_heads]`) after `prefix_rows` leading rows, rewritten grouped by key head; with
/// `chunk_rows`, independently within every run of that many rows (a reshaped matrix's columns).
struct Untile {
    k_heads: usize,
    v_heads: usize,
    rows_per_block: usize,
    prefix_rows: usize,
    /// 0: one run over all rows.
    chunk_rows: usize,
}

impl Untile {
    fn from(a: &J) -> Result<Self, String> {
        Ok(Self {
            k_heads: arg_u(a, "k_heads")?,
            v_heads: arg_u(a, "v_heads")?,
            rows_per_block: arg_u(a, "rows_per_block")?,
            prefix_rows: arg_u_or(a, "prefix_rows", 0)?,
            chunk_rows: arg_u_or(a, "chunk_rows", 0)?,
        })
    }
}

/// The tiled -> grouped head-block reorder of [`Untile`] over rows of `row_len` elements.
fn untile_rows<T: Copy>(data: &[T], row_len: usize, u: &Untile) -> Vec<T> {
    let chunk = match u.chunk_rows {
        0 => data.len().max(1),
        c => c * row_len,
    };
    // One exact-size copy, reordered in place (a collected iterator over-allocates).
    let mut out = data.to_vec();
    for (src, dst) in data.chunks(chunk).zip(out.chunks_mut(chunk)) {
        untile_run(src, dst, row_len, u);
    }
    out
}

/// Rewrites `out` (a copy of `data`) with `data`'s tiled head blocks grouped.
fn untile_run<T: Copy>(data: &[T], out: &mut [T], row_len: usize, u: &Untile) {
    let per_k = u.v_heads / u.k_heads.max(1);
    let block = u.rows_per_block * row_len;
    let prefix = u.prefix_rows * row_len;
    if prefix >= data.len() {
        return;
    }
    let (tiled, dest) = (&data[prefix..], &mut out[prefix..]);
    for k in 0..u.k_heads {
        for r in 0..per_k {
            let (src, dst) = ((r * u.k_heads + k) * block, (k * per_k + r) * block);
            if src + block <= tiled.len() && dst + block <= dest.len() {
                dest[dst..dst + block].copy_from_slice(&tiled[src..src + block]);
            }
        }
    }
}

/// Value heads of `block` values each in tiled order
/// (`[v_per_k, k_heads]`) rewritten grouped (`[k_heads, v_per_k]`).
fn untile_heads(x: &mut [f32], k_heads: usize, block: usize) {
    let per_k = x.len() / (block * k_heads).max(1);
    let tiled = x.to_vec();
    for k in 0..k_heads {
        for r in 0..per_k {
            x[(k * per_k + r) * block..][..block]
                .copy_from_slice(&tiled[(r * k_heads + k) * block..][..block]);
        }
    }
}

/// BitNet online weight quant: {-1, 0, 1} on 1 / mean |W|, divided back and cast.
pub fn ternarize(w: &mut [f32], cast: impl Fn(f32) -> f32) {
    let mean = (w.iter().map(|v| v.abs() as f64).sum::<f64>() / w.len().max(1) as f64) as f32;
    let scale = 1.0 / mean.max(1e-5);
    let step = cast(1.0 / scale);
    for v in w {
        *v = (*v * scale).round_ties_even().clamp(-1.0, 1.0) * step;
    }
}

/// Row-major `v` of `shape` with its axes permuted (`out` axis `d` = input axis `perm[d]`).
fn permute(v: &[f32], shape: &[usize], perm: &[usize]) -> Vec<f32> {
    let r = shape.len();
    let mut stride = vec![1usize; r];
    for d in (0..r.saturating_sub(1)).rev() {
        stride[d] = stride[d + 1] * shape[d + 1];
    }
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let mut idx = vec![0usize; r];
    let mut out = Vec::with_capacity(v.len());
    for _ in 0..v.len() {
        out.push(
            v[idx
                .iter()
                .zip(perm)
                .map(|(&i, &p)| i * stride[p])
                .sum::<usize>()],
        );
        for d in (0..r).rev() {
            idx[d] += 1;
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

fn arg_f(a: &J, k: &str) -> Result<f64, String> {
    a[k].as_f64()
        .ok_or_else(|| format!("arg `{k}` must be a number"))
}

fn arg_u(a: &J, k: &str) -> Result<usize, String> {
    Val::from_json(&a[k])
        .usize()
        .map_err(|e| format!("arg `{k}`: {e}"))
}

fn arg_u_or(a: &J, k: &str, d: usize) -> Result<usize, String> {
    if a[k].is_null() {
        Ok(d)
    } else {
        arg_u(a, k)
    }
}

fn arg_us(a: &J, k: &str) -> Result<Vec<usize>, String> {
    a[k].as_array()
        .ok_or_else(|| format!("arg `{k}` must be a list"))?
        .iter()
        .map(|v| {
            Val::from_json(v)
                .usize()
                .map_err(|e| format!("arg `{k}`: {e}"))
        })
        .collect()
}

/// `[[name, rows], ...]`.
fn arg_parts(a: &J) -> Result<Vec<(String, usize)>, String> {
    a["parts"]
        .as_array()
        .ok_or("arg `parts` must be a list of [name, rows]")?
        .iter()
        .map(|p| match p.as_array().map(Vec::as_slice) {
            Some([J::String(n), r]) => Ok((n.clone(), Val::from_json(r).usize()?)),
            _ => Err(format!("bad part {p}: expected [name, rows]")),
        })
        .collect()
}

/// Output (new name, shape) of each part of transform `op` on inputs of `ins` shapes.
fn xf_shapes(
    op: &str,
    a: &J,
    ins: &[&[usize]],
) -> Result<Vec<(Option<String>, Vec<usize>)>, String> {
    let s = ins[0];
    let numel: usize = s.iter().product();
    let same = || Ok(vec![(None, s.to_vec())]);
    match op {
        "add" | "scale" | "softplus" | "inv_softplus" | "neg_exp" | "to_f32" | "to_bf16"
        | "round_bf16" | "ternarize" => same(),
        "untile_packed_rows" => {
            let u = Untile::from(a)?;
            let rows = s.first().copied().unwrap_or(1);
            let run = if u.chunk_rows == 0 {
                rows
            } else {
                u.chunk_rows
            };
            if u.k_heads == 0
                || u.v_heads % u.k_heads != 0
                || rows % run.max(1) != 0
                || u.prefix_rows + u.v_heads * u.rows_per_block != run
            {
                return Err(format!(
                    "{} prefix rows + {} heads x {} rows (over {} key heads) do not tile {s:?}",
                    u.prefix_rows, u.v_heads, u.rows_per_block, u.k_heads
                ));
            }
            same()
        }
        "untile_heads" => {
            let (kh, block) = (arg_u(a, "k_heads")?, arg_u(a, "block")?);
            let prefix = arg_u_or(a, "prefix", 0)?;
            let chunk = arg_u_or(a, "chunk", numel)?;
            let ok = kh > 0
                && block > 0
                && chunk > 0
                && numel % chunk == 0
                && prefix <= chunk
                && (chunk - prefix) % (block * kh) == 0;
            if !ok {
                return Err(format!(
                    "k_heads {kh} x block {block} (prefix {prefix}, chunk {chunk}) do not tile {s:?}"
                ));
            }
            same()
        }
        "permute_head_dims" => {
            let hd = arg_us(a, "perm")?.len();
            if hd == 0 || s.first().is_none_or(|r| r % hd != 0) {
                return Err(format!("rows of {s:?} are not whole heads of {hd}"));
            }
            same()
        }
        "transpose" => {
            let perm = if a["perm"].is_null() {
                if s.len() != 2 {
                    return Err(format!("transpose of rank {} needs `perm`", s.len()));
                }
                vec![1, 0]
            } else {
                arg_us(a, "perm")?
            };
            let mut seen = perm.clone();
            seen.sort_unstable();
            if seen != (0..s.len()).collect::<Vec<_>>() {
                return Err(format!("perm {perm:?} does not permute rank {}", s.len()));
            }
            Ok(vec![(None, perm.iter().map(|&p| s[p]).collect())])
        }
        "reshape" => {
            let raw_dims = a["shape"].as_array().ok_or("arg `shape` must be a list")?;
            let mut shape = Vec::with_capacity(raw_dims.len());
            let mut infer_idx = None;
            for (idx, v) in raw_dims.iter().enumerate() {
                let d = Val::from_json(v)
                    .num()
                    .map_err(|e| format!("arg `shape`: {e}"))? as i64;
                if d == 0 {
                    let kept = s.get(idx).copied().ok_or_else(|| {
                        format!("reshape dim 0 at axis {idx} out of bounds for {s:?}")
                    })?;
                    shape.push(kept);
                } else if d == -1 {
                    if infer_idx.replace(idx).is_some() {
                        return Err(format!("reshape {raw_dims:?} has multiple -1 dimensions"));
                    }
                    shape.push(1);
                } else if d > 0 {
                    shape.push(d as usize);
                } else {
                    return Err(format!("cannot reshape {s:?} to {raw_dims:?}"));
                }
            }
            if let Some(ii) = infer_idx {
                let known: usize = shape
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != ii)
                    .map(|(_, &v)| v)
                    .product();
                if known == 0 || numel % known != 0 {
                    return Err(format!("cannot reshape {s:?} to {raw_dims:?}"));
                }
                shape[ii] = numel / known;
            }
            if shape.iter().product::<usize>() != numel {
                return Err(format!("cannot reshape {s:?} to {shape:?}"));
            }
            Ok(vec![(None, shape)])
        }
        "split_rows" | "deinterleave_heads" => {
            let parts = arg_parts(a)?;
            let heads = if op == "split_rows" {
                1
            } else {
                arg_u(a, "heads")?
            };
            let total: usize = parts.iter().map(|p| p.1).sum();
            if s.first() != Some(&(heads * total)) {
                return Err(format!("{heads} x {total} rows do not split {s:?}"));
            }
            Ok(parts
                .into_iter()
                .map(|(n, r)| {
                    let mut o = s.to_vec();
                    o[0] = heads * r;
                    (Some(n), o)
                })
                .collect())
        }
        "conv_to_linear" if s.len() >= 3 => Ok(vec![(None, vec![s[0], numel / s[0].max(1)])]),
        "last_temporal_tap" if s.len() >= 3 => {
            let mut o = s.to_vec();
            o.remove(2);
            Ok(vec![(None, o)])
        }
        // Base `[o, ...]` (a convolution kernel flattens its trailing dims), A `[r, ...]`, B `[o, r, ...]`.
        "lora_merge" => match (s, ins.get(1).copied(), ins.get(2).copied()) {
            ([o, rest @ ..], Some([r, ra @ ..]), Some([o2, r2, rb @ ..]))
                if !rest.is_empty()
                    && rest.iter().product::<usize>() == ra.iter().product::<usize>()
                    && o == o2
                    && r == r2
                    && rb.iter().product::<usize>() == 1 =>
            {
                same()
            }
            _ => Err(format!(
                "LoRA factors {:?} / {:?} do not fit {s:?}",
                ins.get(1),
                ins.get(2)
            )),
        },
        "stack_experts" => {
            if ins.iter().any(|x| *x != s) {
                return Err("experts differ in shape".into());
            }
            let flatten_2d = a.get("flatten_2d").and_then(J::as_bool).unwrap_or(false);
            let o = if flatten_2d && !s.is_empty() {
                let mut o = vec![ins.len() * s[0]];
                o.extend_from_slice(&s[1..]);
                o
            } else {
                let mut o = vec![ins.len()];
                o.extend_from_slice(s);
                o
            };
            Ok(vec![(None, o)])
        }
        "conv_to_linear" | "last_temporal_tap" => Err(format!("`{op}` needs rank >= 3, got {s:?}")),
        _ => Err(format!("unknown transform `{op}`")),
    }
}

fn xf_kind(op: &str, ins: &[Arc<Node>]) -> Kind {
    let k = ins[0].kind;
    match op {
        "to_f32" => Kind::F32,
        // Float tensors become bf16 `Mat`s (what `Weights::from_gguf` made of GGUF float
        // matrices); packed blocks stay as stored.
        "to_bf16" if matches!(k, Kind::Packed(_)) => k,
        "to_bf16" => Kind::Bf16,
        "untile_packed_rows" | "reshape" => k,
        "stack_experts" if matches!(k, Kind::Packed(_)) && ins.iter().all(|i| i.kind == k) => k,
        _ => k.dense(),
    }
}

/// Part `part` of transform `op` on decoded inputs.
fn xf_dense(
    op: &str,
    a: &J,
    ins: &[Arc<Node>],
    mut vals: Vec<Vec<f32>>,
    part: usize,
) -> Result<Vec<f32>, String> {
    let s = ins[0].shape.as_slice();
    let row = s.iter().skip(1).product::<usize>().max(1);
    let mut v = std::mem::take(&mut vals[0]);
    let map = |v: &mut Vec<f32>, f: &dyn Fn(f32) -> f32| v.iter_mut().for_each(|x| *x = f(*x));
    match op {
        "add" => {
            let c = arg_f(a, "value")? as f32;
            map(&mut v, &|x| x + c);
        }
        "scale" => {
            let c = arg_f(a, "value")? as f32;
            map(&mut v, &|x| x * c);
        }
        "softplus" => map(&mut v, &|x| if x > 20.0 { x } else { x.exp().ln_1p() }),
        "inv_softplus" => map(&mut v, &|x| {
            if x > 20.0 {
                x
            } else {
                x + (-(-x).exp()).ln_1p()
            }
        }),
        "neg_exp" => map(&mut v, &|x| -x.exp()),
        "to_f32" | "to_bf16" | "reshape" => {}
        "round_bf16" => map(&mut v, &|x| bf16_to_f32(f32_to_bf16(x))),
        "ternarize" => match ins[0].kind {
            Kind::Bf16 | Kind::Packed(_) => ternarize(&mut v, |x| bf16_to_f32(f32_to_bf16(x))),
            Kind::F16 => ternarize(&mut v, |x| f16_to_f32(f32_to_f16(x))),
            _ => ternarize(&mut v, |x| x),
        },
        "transpose" => {
            let perm = if a["perm"].is_null() {
                vec![1, 0]
            } else {
                arg_us(a, "perm")?
            };
            v = permute(&v, s, &perm);
        }
        "split_rows" | "deinterleave_heads" => {
            let parts = arg_parts(a)?;
            let heads = if op == "split_rows" {
                1
            } else {
                arg_u(a, "heads")?
            };
            let per_head: usize = parts.iter().map(|p| p.1).sum();
            let off: usize = parts[..part].iter().map(|p| p.1).sum();
            let rows = parts[part].1;
            v = (0..heads)
                .flat_map(|h| {
                    let start = (h * per_head + off) * row;
                    v[start..start + rows * row].to_vec()
                })
                .collect();
        }
        "untile_heads" => {
            let (kh, block) = (arg_u(a, "k_heads")?, arg_u(a, "block")?);
            let prefix = arg_u_or(a, "prefix", 0)?;
            let chunk = arg_u_or(a, "chunk", v.len())?.max(1);
            if block == 0 || chunk > v.len() || v.len() % chunk != 0 || prefix > chunk {
                return Err(format!("untile_heads: bad block/chunk/prefix for {s:?}"));
            }
            v.chunks_exact_mut(chunk)
                .for_each(|c| untile_heads(&mut c[prefix..], kh, block));
        }
        "untile_packed_rows" => v = untile_rows(&v, row, &Untile::from(a)?),
        "permute_head_dims" => {
            let perm = arg_us(a, "perm")?;
            let hd = perm.len();
            let src = v.clone();
            for (r, dst) in v.chunks_exact_mut(row).enumerate() {
                let (h, j) = (r / hd, r % hd);
                let from = (h * hd + perm[j]) * row;
                dst.copy_from_slice(&src[from..from + row]);
            }
        }
        "conv_to_linear" => match a["order"].as_str().unwrap_or("in_k") {
            "in_k" => {}
            "k_in" => {
                let k = row / s[1].max(1);
                v = permute(&v, &[s[0], s[1], k], &[0, 2, 1]);
            }
            o => return Err(format!("conv_to_linear: unknown order `{o}` (in_k, k_in)")),
        },
        "last_temporal_tap" => {
            let (t, rest) = (s[2], s[3..].iter().product::<usize>());
            v = v
                .chunks_exact(t * rest)
                .flat_map(|c| c[(t - 1) * rest..].to_vec())
                .collect();
        }
        "lora_merge" => {
            let (o, i, r) = (s[0], s[1..].iter().product::<usize>(), ins[1].shape[0]);
            let scale = arg_f(a, "scale")? as f32;
            let (aa, bb) = (&vals[1], &vals[2]);
            for oi in 0..o {
                for k in 0..r {
                    let b = bb[oi * r + k] * scale;
                    for (w, x) in v[oi * i..(oi + 1) * i]
                        .iter_mut()
                        .zip(&aa[k * i..(k + 1) * i])
                    {
                        *w += b * x;
                    }
                }
            }
        }
        "stack_experts" => {
            for x in &vals[1..] {
                v.extend_from_slice(x);
            }
        }
        _ => return Err(format!("unknown transform `{op}`")),
    }
    Ok(v)
}

/// Args of a transform: names (`parts[*][0]`, `out`, `a`, `b`, `order`) interpolate, the rest
/// are expressions.
fn eval_args(env: &ModelEnv, caps: &[(String, Val)], args: &J) -> Result<J, String> {
    let Some(o) = args.as_object() else {
        return env.eval_with(caps, args).map(|v| v.to_json());
    };
    let mut out = Map::new();
    for (k, v) in o {
        let e = match (k.as_str(), v) {
            ("out" | "a" | "b" | "order", J::String(s)) => J::String(env.interp_with(caps, s)?),
            ("parts", J::Array(ps)) => J::Array(
                ps.iter()
                    .map(|p| match p.as_array().map(Vec::as_slice) {
                        Some([J::String(n), r]) => Ok(J::Array(vec![
                            J::String(env.interp_with(caps, n)?),
                            env.eval_with(caps, r)?.to_json(),
                        ])),
                        _ => Err(format!("bad part {p}: expected [name, rows]")),
                    })
                    .collect::<Result<_, String>>()?,
            ),
            _ => env.eval_with(caps, v)?.to_json(),
        };
        out.insert(k.clone(), e);
    }
    Ok(J::Object(out))
}

fn insert_new(table: &Table, name: String, n: Node) -> Result<(), String> {
    let mut t = table.write();
    if t.contains_key(&name) {
        return Err(format!("`{name}` already exists"));
    }
    t.insert(name, Arc::new(n));
    Ok(())
}

fn apply_transform(t: &Transform, env: &ModelEnv, table: &Table) -> Result<(), String> {
    let rule = Rule::new(&t.tensor, "")?;
    let matches: Vec<(String, Vec<(String, Val)>)> = table
        .read()
        .keys()
        .filter_map(|n| rule.captures(n).map(|c| (n.clone(), c)))
        .collect();
    let when = |caps: &[(String, Val)]| -> Result<bool, String> {
        match &t.when {
            None => Ok(true),
            Some(w) => Ok(env.eval_with(caps, &J::String(w.clone()))?.truthy()),
        }
    };
    if t.op == "stack_experts" {
        // Group the matches by output name; the stacked capture (`var`, default `e`) orders them.
        let mut groups: BTreeMap<String, Vec<(usize, String, Vec<(String, Val)>)>> =
            BTreeMap::new();
        for (name, caps) in matches {
            let var = t.args["var"].as_str().unwrap_or("e");
            let e = caps
                .iter()
                .find(|(k, _)| k == var)
                .ok_or_else(|| format!("pattern has no `{{{var}}}` capture"))?
                .1
                .usize()?;
            let out = env.interp_with(&caps, t.args["out"].as_str().ok_or("needs `out`")?)?;
            groups.entry(out).or_default().push((e, name, caps));
        }
        for (out, mut members) in groups {
            members.sort_by_key(|m| m.0);
            if members.iter().enumerate().any(|(i, m)| m.0 != i) {
                return Err(format!(
                    "`{out}`: experts are not numbered 0..{}",
                    members.len()
                ));
            }
            if !when(&members[0].2)? {
                continue;
            }
            let args = eval_args(env, &members[0].2, &t.args)?;
            if !args["count"].is_null() && arg_u(&args, "count")? != members.len() {
                return Err(format!(
                    "`{out}`: {} experts, `count` says {}",
                    members.len(),
                    args["count"]
                ));
            }
            let ins: Vec<Arc<Node>> = {
                let tb = table.read();
                members.iter().map(|m| tb[&m.1].clone()).collect()
            };
            let shapes: Vec<&[usize]> = ins.iter().map(|n| n.shape.as_slice()).collect();
            let shape = xf_shapes(&t.op, &args, &shapes)?.remove(0).1;
            let kind = xf_kind(&t.op, &ins);
            insert_new(
                table,
                out,
                Node {
                    shape,
                    kind,
                    conv: false,
                    src: Src::Xf {
                        op: t.op.clone(),
                        args,
                        ins,
                        part: 0,
                    },
                },
            )?;
        }
        return Ok(());
    }
    for (name, caps) in matches {
        if !when(&caps)? {
            continue;
        }
        let args = eval_args(env, &caps, &t.args)?;
        let mut ins = vec![table.read()[&name].clone()];
        // Already bf16 (or packed, kept as stored): nothing to round, stays zero-copy.
        if t.op == "to_bf16" && !matches!(ins[0].kind, Kind::F32 | Kind::F16) {
            continue;
        }
        if t.op == "lora_merge" {
            for k in ["a", "b"] {
                let n = args[k].as_str().ok_or_else(|| format!("needs `{k}`"))?;
                ins.push(
                    table
                        .read()
                        .get(n)
                        .cloned()
                        .ok_or_else(|| format!("no tensor `{n}`"))?,
                );
            }
        }
        // Rounding to bf16 only changes f32 values; bf16/f16/packed stay in their stored encoding.
        if t.op == "round_bf16" && ins[0].kind != Kind::F32 {
            continue;
        }
        let shapes: Vec<&[usize]> = ins.iter().map(|n| n.shape.as_slice()).collect();
        let outs = xf_shapes(&t.op, &args, &shapes).map_err(|e| format!("`{name}`: {e}"))?;
        let kinds: Vec<Kind> = outs.iter().map(|_| xf_kind(&t.op, &ins)).collect();
        let conv = t.op == "to_f32" || (ins[0].conv && t.op != "conv_to_linear");
        for (part, (new, shape)) in outs.into_iter().enumerate() {
            let node = Node {
                shape,
                kind: kinds[part],
                conv,
                src: Src::Xf {
                    op: t.op.clone(),
                    args: args.clone(),
                    ins: ins.clone(),
                    part,
                },
            };
            match new {
                Some(n) => insert_new(table, n, node)?,
                None => {
                    table.write().insert(name.clone(), Arc::new(node));
                }
            }
        }
    }
    Ok(())
}

// ---- model ------------------------------------------------------------------------------------

/// A checkpoint loaded under its family spec.
pub struct Model {
    pub spec: Spec,
    pub env: ModelEnv,
    pub store: WeightStore,
    /// Shared with `env` and the runtime that serves the model (one copy of the vocab maps).
    pub tokenizer: Option<Arc<Tokenizer>>,
    pub ckpt: Arc<Checkpoint>,
    /// (raw checkpoint name, canonical name) of every raw tensor, sorted by raw name.
    pub renames: Vec<(String, String)>,
    /// `spec.ignore` globs whose `when` holds (for [`WeightStore::unconsumed`]).
    pub ignore: Vec<String>,
}

/// Opens `artifacts` and loads it as `family` (detected from the checkpoint when `None`), under
/// the effective re-encode configuration.
pub fn load(
    artifacts: &ModelArtifacts,
    reg: &Registry,
    family: Option<&str>,
) -> Result<Model, String> {
    load_with(artifacts, reg, family, ReencodeConfig::effective())
}

/// [`load`] with an explicit re-encode configuration.
pub fn load_with(
    artifacts: &ModelArtifacts,
    reg: &Registry,
    family: Option<&str>,
    reencode: ReencodeConfig,
) -> Result<Model, String> {
    let ckpt = Arc::new(open(artifacts)?);
    let tokenizer = tokenizer(artifacts, &ckpt)?.map(Arc::new);
    load_from(ckpt, reg, family, tokenizer, reencode)
}

/// tokenizer.json, else the GGUF / ggml tokenizer, else none.
fn tokenizer(a: &ModelArtifacts, ck: &Checkpoint) -> Result<Option<Tokenizer>, String> {
    let json = a
        .tokenizer_path
        .clone()
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .or_else(|| ck.dir.as_ref().map(|d| d.join("tokenizer.json")))
        .filter(|p| p.is_file());
    if let Some(p) = json {
        return crate::model::hf_loader::load_json_tokenizer(&p).map(Some);
    }
    Ok(ck
        .container
        .as_ref()
        .and_then(|d| UniversalModelLoader::extract_gguf_tokenizer(d).ok()))
}

fn load_from(
    ckpt: Arc<Checkpoint>,
    reg: &Registry,
    family: Option<&str>,
    tokenizer: Option<Arc<Tokenizer>>,
    mut reencode: ReencodeConfig,
) -> Result<Model, String> {
    let family = match family {
        Some(f) => f.to_string(),
        None => reg
            .detect(&ckpt.sig)
            .ok_or_else(|| format!("no family spec claims this checkpoint ({:?})", ckpt.sig))?,
    };
    let spec = reg.spec(&family, &ckpt.sig)?;
    if spec.pipeline == Pipeline::Unsupported {
        let sig = &ckpt.sig;
        let name = sig
            .hf_model_type
            .as_deref()
            .or(sig.gguf_arch.as_deref())
            .unwrap_or("");
        return Err(format!("{name}: {}", spec.unsupported_reason(name)));
    }
    let fam = |e: String| format!("{family}: {e}");

    let mut pairs = spec.names.rules.clone();
    for set in &spec.names.use_sets {
        pairs.extend(reg.name_set(set)?);
    }
    let ren = Renamer::from_pairs(&pairs)
        .map_err(fam)?
        .with_strip(spec.names.strip.clone());
    let table: Table = Arc::new(RwLock::new(BTreeMap::new()));
    {
        let mut t = table.write();
        let mut from: HashMap<String, &str> = HashMap::new();
        let media = spec.runtime.get("media").is_some_and(|m| !m.is_null());
        for (raw, r) in ckpt.raw.iter().filter(|(_, r)| media || !r.sidecar) {
            let c = ren.canonical(raw);
            if let Some(prev) = from.insert(c.clone(), raw) {
                return Err(fam(format!(
                    "checkpoint tensors `{prev}` and `{raw}` both rename to `{c}`"
                )));
            }
            t.insert(
                c,
                Arc::new(Node {
                    shape: r.shape.clone(),
                    kind: r.kind,
                    conv: false,
                    src: Src::Raw(raw.clone()),
                }),
            );
        }
    }
    let env = ModelEnv::new(&spec, ckpt.clone(), table.clone(), tokenizer.clone());
    if ckpt.synthetic {
        synthesize(&spec, &env, &table).map_err(fam)?;
    }
    let (lora, adapter) = adapters(&ckpt, &ren, &table).map_err(fam)?;
    for t in &spec.transforms {
        apply_transform(t, &env, &table)
            .map_err(|e| fam(format!("transform `{}` on `{}`: {e}", t.op, t.tensor)))?;
    }
    for (name, c) in &spec.constants {
        let c: Constant = serde_json::from_value(c.clone())
            .map_err(|e| fam(format!("constant `{name}`: {e}")))?;
        let cx = |e: String| fam(format!("constant `{name}`: {e}"));
        if let Some(w) = &c.when {
            if !env.eval(&J::String(w.clone())).map_err(cx)?.truthy() {
                continue;
            }
        }
        let args = env.eval(&c.args).map_err(cx)?.to_json();
        let g = gen::generate(&c.gen, &args).map_err(cx)?;
        let kind = match g.data {
            GenData::F32(_) => Kind::F32,
            GenData::U32(_) => Kind::U32,
        };
        insert_new(
            &table,
            name.clone(),
            Node {
                shape: g.shape.clone(),
                kind,
                conv: false,
                src: Src::Gen(Arc::new(g)),
            },
        )
        .map_err(cx)?;
    }
    if spec.pipeline != super::Pipeline::CausalLm {
        reencode.policy = ReencodePolicy::BitExact;
    }
    if let Some(pats) = spec.runtime.get("ffn_weights").and_then(J::as_array) {
        reencode.ffn_weights = Some(Arc::new(
            pats.iter()
                .filter_map(|p| p.as_str().map(String::from))
                .collect(),
        ));
    }
    if let Some(f) = reencode.force {
        if !crate::model::reencode::encodable(f) {
            return Err(format!(
                "--quant {f:?}: weights cannot be encoded in that format"
            ));
        }
    }
    let gate_ctx = crate::model::reencode::gate_context();
    let (req_policy, req_max_kl) = gate_ctx
        .map(|g| (g.requested_policy, g.max_kl))
        .unwrap_or((reencode.policy, reencode.max_kl));
    let reencode_key = ckpt
        .weights_path
        .as_deref()
        .filter(|_| {
            !ckpt.synthetic && req_policy != ReencodePolicy::BitExact && reencode.force.is_none()
        })
        .and_then(|p| {
            crate::model::reencode::choice_key(p, req_policy, req_max_kl, reencode.force)
        });
    let mut reencode_cache = None;
    if let Some(key) = &reencode_key {
        if gate_ctx.is_none_or(|g| g.allow_redirect) {
            if let Some(choice) = crate::model::reencode::read_choice_record(key) {
                let report = GateReport {
                    mean_kl: choice.mean_kl,
                    top1: choice.top1,
                    tokens: 48,
                };
                if crate::model::weight_format::gate_passes(choice.policy, req_max_kl, &report) {
                    if let Some(cache) = crate::model::reencode::ReencodeCache::open(key) {
                        reencode.policy = choice.policy;
                        reencode.ffn_only = choice.ffn_only;
                        crate::model::reencode::set_cached_choice(choice);
                        reencode_cache = Some(cache);
                    }
                }
            } else if !reencode.ffn_only {
                reencode_cache = crate::model::reencode::ReencodeCache::open(key);
            }
        }
    }
    let reencode_key = reencode_key.filter(|_| reencode.policy != ReencodePolicy::BitExact);
    let store = WeightStore {
        ckpt: ckpt.clone(),
        table: table.clone(),
        lora,
        adapter,
        reencode,
        reencode_key,
        reencode_cache,
        cache: Mutex::new(HashMap::new()),
    };
    if spec.components.contains_key("precompute") {
        precompute(&spec, &env, &store, &table).map_err(|e| fam(format!("precompute: {e}")))?;
    }
    let renames = ckpt
        .raw
        .keys()
        .map(|r| (r.clone(), ren.canonical(r)))
        .collect();
    let mut ignore = Vec::new();
    for i in &spec.ignore {
        match i {
            Ignore::Glob(g) => ignore.push(g.clone()),
            Ignore::When { glob, when } => {
                if env
                    .eval(&J::String(when.clone()))
                    .map_err(|e| fam(format!("ignore `{glob}`: {e}")))?
                    .truthy()
                {
                    ignore.push(glob.clone());
                }
            }
        }
    }
    Ok(Model {
        spec,
        env,
        store,
        tokenizer,
        ckpt,
        renames,
        ignore,
    })
}

/// Runs the spec's `precompute` component once on the CPU (no instance vars, no inputs): its
/// declared outputs and every tap it emits become F32 weights under their names.
fn precompute(
    spec: &Spec,
    env: &ModelEnv,
    store: &WeightStore,
    table: &Table,
) -> Result<(), String> {
    let inst = super::build::build(
        spec,
        "precompute",
        env,
        store,
        &[("taps", Val::Str("*".into()))],
    )?;
    if !inst.inputs.is_empty() {
        return Err("`precompute` takes no inputs".into());
    }
    let g = &inst.graph;
    let shapes: Vec<Vec<usize>> = g
        .outputs
        .iter()
        .map(|&v| g.value(v).shape.clone())
        .collect();
    let ins: Vec<Arc<Node>> = {
        let t = table.read();
        inst.weights
            .iter()
            .filter_map(|w| t.get(w).cloned())
            .collect()
    };
    let w = store.materialize(&inst.weights)?;
    // A load-time weight transform like `transforms`, not inference: allowed under the GPU-only
    // tripwire.
    let _load_time = crate::hardware::cpu_compute::allow();
    let _exact = crate::model::qdot::exact();
    let mut p = crate::graph::program::GraphProgram::compile(
        inst.graph,
        w,
        crate::hardware::PlacementMode::Pin(crate::hardware::Device::Cpu),
    )?;
    let outs = p.run(Vec::new())?;
    for ((name, shape), t) in inst.outputs.into_iter().zip(shapes).zip(outs) {
        let Tensor::F32(data) = t else {
            return Err(format!("output `{name}` is not f32"));
        };
        let src = Src::Pre {
            data,
            ins: ins.clone(),
        };
        insert_new(
            table,
            name,
            Node {
                shape,
                kind: Kind::F32,
                conv: false,
                src,
            },
        )?;
    }
    Ok(())
}

/// Random-initialized `spec.synthetic` weights (deterministic per name).
fn synthesize(spec: &Spec, env: &ModelEnv, table: &Table) -> Result<(), String> {
    for (pat, s) in &spec.synthetic {
        let shape_j = s
            .get("shape")
            .ok_or_else(|| format!("synthetic `{pat}` needs `shape`"))?;
        let mut combos: Vec<Vec<(String, Val)>> = vec![Vec::new()];
        for (var, list) in s.get("over").and_then(J::as_object).into_iter().flatten() {
            let items = env.eval(list)?.list()?.to_vec();
            combos = combos
                .into_iter()
                .flat_map(|c| {
                    items.iter().map(move |it| {
                        let mut c = c.clone();
                        c.push((var.clone(), it.clone()));
                        c
                    })
                })
                .collect();
        }
        for caps in combos {
            let name = env.interp_with(&caps, pat)?;
            let shape = env
                .eval_with(&caps, shape_j)?
                .list()?
                .iter()
                .map(Val::usize)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("synthetic `{name}`: {e}"))?;
            let kind = if shape.len() >= 2 {
                Kind::Bf16
            } else {
                Kind::F32
            };
            let seed = fnv1a(&name);
            insert_new(
                table,
                name,
                Node {
                    shape,
                    kind,
                    conv: false,
                    src: Src::Synth(seed),
                },
            )?;
        }
    }
    Ok(())
}

type Adapters = (HashMap<String, Lora>, Option<Arc<(ModelManifest, Vec<u8>)>>);

/// A PEFT LoRA adapter beside the checkpoint (`adapter_config.json` + `adapter_model.*`): each
/// `lora_A`/`lora_B` pair registers on its target's canonical weight; `modules_to_save` tensors
/// replace theirs. Skipped when the checkpoint is not the adapter's base: an adapter repo
/// snapshot holding the weights `peft::prepare_causal_lm` merged (marked since this loader, or
/// named after the adapter repo, not its `base_model_name_or_path`).
fn adapters(ckpt: &Checkpoint, ren: &Renamer, table: &Table) -> Result<Adapters, String> {
    let Some(dir) = &ckpt.dir else {
        return Ok(Default::default());
    };
    let cfg = dir.join("adapter_config.json");
    if !cfg.is_file() || ckpt.lora_merged {
        return Ok(Default::default());
    }
    let file = ["adapter_model.safetensors", "adapter_model.bin"]
        .iter()
        .map(|f| dir.join(f))
        .find(|p| p.is_file())
        .ok_or("adapter_config.json without adapter_model.safetensors / .bin")?;
    let ad = read_json(&cfg)?;
    if let Some(base) = ad["base_model_name_or_path"].as_str() {
        let is_dir = Path::new(base).canonicalize().ok() == dir.canonicalize().ok();
        if base != ckpt.sig.hub_id && !is_dir {
            return Ok(Default::default());
        }
    }
    let scale = crate::model::peft::lora_scale(&ad)?;
    let (am, data) = crate::model::loaders::read_checkpoint(&file)?;
    let adapter = Arc::new((am, data));
    let am = &adapter.0;
    let kind = |n: &str| match am.tensors[n].dtype.as_str() {
        "BF16" => Kind::Bf16,
        "F16" => Kind::F16,
        _ => Kind::F32,
    };
    let (mut lora, mut used) = (HashMap::new(), HashSet::new());
    let mut t = table.write();
    for raw in ckpt.raw.keys() {
        let c = ren.canonical(raw);
        let (saved, a) = crate::model::peft::adapter_names(am, raw);
        let target = t[&c].shape.clone();
        if let Some(s) = saved {
            if am.tensors[&s].shape != target {
                return Err(format!("adapter {s} does not match {c} {target:?}"));
            }
            t.insert(
                c,
                Arc::new(Node {
                    shape: target,
                    kind: kind(&s),
                    conv: false,
                    src: Src::Adapter(s.clone()),
                }),
            );
            used.insert(s);
        } else if let Some(a) = a {
            let b = a.replace(".lora_A.", ".lora_B.");
            let (sa, sb) = (
                am.tensors[&a].shape.clone(),
                am.tensors.get(&b).map(|t| t.shape.clone()),
            );
            let ok = matches!((&target[..], &sa[..], sb.as_deref()),
                ([o, i], [r, i2], Some([o2, r2])) if i == i2 && o == o2 && r == r2);
            if !ok {
                return Err(format!(
                    "LoRA factors {a} {sa:?} / {b} {sb:?} do not fit {c} {target:?}"
                ));
            }
            let (na, nb) = (format!("{c}.lora_A"), format!("{c}.lora_B"));
            for (n, s, src) in [(&na, sa, &a), (&nb, sb.unwrap(), &b)] {
                t.insert(
                    n.clone(),
                    Arc::new(Node {
                        shape: s,
                        kind: kind(src),
                        conv: false,
                        src: Src::Adapter(src.clone()),
                    }),
                );
            }
            lora.insert(
                c,
                Lora {
                    a: na,
                    b: nb,
                    scale,
                    bias_delta: None,
                },
            );
            used.extend([a, b]);
        }
    }
    if let Some(stray) = am.tensors.keys().find(|k| !used.contains(*k)) {
        return Err(format!(
            "adapter tensor {stray} matches no checkpoint tensor"
        ));
    }
    drop(t);
    Ok((lora, Some(adapter)))
}

impl Model {
    /// Component `name` declared as `{family, dir}`: that family's spec over `<dir>` (a checkpoint
    /// subfolder with its own config.json and weights), or over this checkpoint without `dir`.
    pub fn component(&self, reg: &Registry, name: &str) -> Result<Model, String> {
        let Component::Ref { dir, .. } = self.spec.component(name)? else {
            return Err(format!(
                "{}: component `{name}` is a network, not a family",
                self.spec.family
            ));
        };
        let comp_spec = format!("{}.{name}", self.spec.family);
        let Some(sub) = dir else {
            return load_from(
                self.ckpt.clone(),
                reg,
                Some(&comp_spec),
                self.tokenizer.clone(),
                self.store.reencode.clone(),
            );
        };
        let root = self
            .ckpt
            .dir
            .as_ref()
            .ok_or_else(|| format!("component `{name}`: checkpoint has no directory"))?
            .join(&sub);
        let file = |f: &str| Some(root.join(f)).filter(|p| p.is_file());
        let a = ModelArtifacts {
            model_id: format!("{}/{sub}", self.ckpt.sig.hub_id),
            local_dir: Some(root.clone()),
            config_path: file("config.json"),
            tokenizer_path: file("tokenizer.json"),
            weights_path: crate::model::hub::component_weights(&root).ok(),
            is_cached: true,
            ..Default::default()
        };
        load_with(&a, reg, Some(&comp_spec), self.store.reencode.clone())
    }
}

/// Evaluates `spec.checks`: the first whose `expr` is true refuses the load with its `error`.
pub fn check(model: &Model) -> Result<(), String> {
    for c in &model.spec.checks {
        if model.env.eval(&J::String(c.expr.clone()))?.truthy() {
            return Err(model.env.interp(&c.error)?);
        }
    }
    Ok(())
}

// ---- re-encode gate ---------------------------------------------------------------------------

#[cfg(test)]
static TEST_GATED_LOCK: parking_lot::ReentrantMutex<()> = parking_lot::ReentrantMutex::new(());

/// The KL-gated re-encode ladder: loads with `load_once` under each policy the user's effective
/// configuration allows (on Metal: max-speed -> max-speed (FFN only) -> low-bit-native ->
/// bit-exact; elsewhere bit-exact only), keeping the first candidate that re-encoded nothing or
/// whose `kl` against the bit-exact load passes the policy's limits. `load_once` reads the policy
/// through `ReencodeConfig::effective()`; `None` from it means "not this loader".
pub fn gated<T>(
    on_metal: bool,
    load_once: impl Fn() -> Option<Result<T, String>>,
    kl: impl Fn(&T, &T) -> Result<GateReport, String>,
) -> Option<Result<T, String>> {
    #[cfg(test)]
    let _test_guard = TEST_GATED_LOCK.lock();
    use crate::model::reencode::{
        native_stats, reset_native_stats, save_reencode_cache, set_gate_context,
        set_native_applied, take_pending_reencode, ChoiceRecord, GateContext,
    };
    use crate::model::weight_format::{
        gate_passes, reencode_override, set_reencode_override,
        ReencodePolicy::{BitExact, LowBitNative, MaxSpeed, MinSize},
    };
    let user = ReencodeConfig::effective();
    let attempts: &[(&str, ReencodePolicy, bool)] = match user.policy {
        _ if !on_metal => &[("bit-exact", BitExact, false)],
        MinSize => &[
            ("min-size", MinSize, false),
            ("min-size (FFN only)", MinSize, true),
            ("max-speed", MaxSpeed, false),
            ("low-bit-native", LowBitNative, false),
            ("bit-exact", BitExact, false),
        ],
        MaxSpeed => &[
            ("max-speed", MaxSpeed, false),
            ("max-speed (FFN only)", MaxSpeed, true),
            ("low-bit-native", LowBitNative, false),
            ("bit-exact", BitExact, false),
        ],
        LowBitNative => &[
            ("low-bit-native", LowBitNative, false),
            ("bit-exact", BitExact, false),
        ],
        BitExact => &[("bit-exact", BitExact, false)],
    };
    let log = |m: String| {
        tracing::info!("{m}");
        if let Some(f) = &user.log {
            f(&m);
        }
    };
    let previous = reencode_override();
    let load_as = |policy: ReencodePolicy, ffn_only: bool, allow_redirect: bool| {
        if let Some(p) = &user.progress {
            // Every attempt counts its own tensors.
            p.done.store(0, std::sync::atomic::Ordering::Relaxed);
            p.total.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        set_reencode_override(Some(ReencodeConfig {
            policy,
            ffn_only,
            ..user.clone()
        }));
        set_gate_context(Some(GateContext {
            requested_policy: user.policy,
            max_kl: user.max_kl,
            allow_redirect,
        }));
        reset_native_stats();
        load_once()
    };
    let mut reference: Option<T> = None;
    let mut result = None;
    let mut i = 0usize;
    let mut allow_redirect = true;
    while i < attempts.len() {
        let (mut label, mut policy, ffn_only) = attempts[i];
        let Some(loaded) = load_as(policy, ffn_only, allow_redirect && i == 0) else {
            result = None; // not this loader's architecture: the caller's fallback applies its own policy
            break;
        };
        let cand = match loaded {
            Ok(c) => c,
            Err(e) => {
                result = Some(Err(e));
                break;
            }
        };
        let stats = native_stats();
        let pending = take_pending_reencode();
        let saved = stats.bytes_before.saturating_sub(stats.bytes_after);
        if let Some(choice) = stats.cached_choice {
            if stats.cache_mismatch || stats.cache_hits != stats.reencoded || stats.reencoded == 0 {
                // Cached rung was stale or mismatched; retry from rung 0 without redirecting.
                allow_redirect = false;
                continue;
            }
            policy = choice.policy;
            label = attempts
                .iter()
                .find(|&&(_, p, f)| p == choice.policy && f == choice.ffn_only)
                .map_or(policy.name(), |&(l, _, _)| l);
            log(format!(
                "reencode: {label} accepted (cached; mean KL {:.4} nats, top-1 {:.1}%; {} tensors, {:.0} MB saved)",
                choice.mean_kl,
                choice.top1 * 100.0,
                stats.reencoded,
                saved as f64 / 1e6
            ));
            set_native_applied(policy, saved, user.force);
            result = Some(Ok(cand));
            break;
        }
        i += 1;
        if policy == BitExact || stats.reencoded == 0 {
            set_native_applied(
                if stats.reencoded == 0 {
                    BitExact
                } else {
                    policy
                },
                saved,
                user.force,
            );
            if stats.reencoded > 0 || user.policy != BitExact {
                log(format!(
                    "reencode: {label} applied ({} of {} tensors re-encoded)",
                    stats.reencoded, stats.tensors
                ));
            }
            result = Some(Ok(cand));
            break;
        }
        if reference.is_none() {
            match load_as(BitExact, false, false) {
                Some(Ok(r)) => {
                    reference = Some(r);
                    set_reencode_override(Some(ReencodeConfig {
                        policy,
                        ffn_only,
                        ..user.clone()
                    }));
                }
                other => {
                    log(format!(
                        "reencode: no bit-exact reference to gate against ({}); using bit-exact",
                        other.and_then(Result::err).unwrap_or_default()
                    ));
                    result = other_exact(&load_as, user.force);
                    break;
                }
            }
        }
        // `load_as(BitExact)` above reset the stats: the candidate's own were read first.
        let gate = kl(reference.as_ref().expect("loaded above"), &cand);
        match gate {
            Ok(r) if gate_passes(policy, user.max_kl, &r) => {
                if let Some(p) = pending {
                    let choice = ChoiceRecord {
                        policy,
                        ffn_only,
                        mean_kl: r.mean_kl,
                        top1: r.top1,
                    };
                    if let Err(e) = save_reencode_cache(p, choice) {
                        tracing::warn!("reencode cache: {e}");
                    }
                }
                log(format!(
                    "reencode: {label} accepted (mean KL {:.4} nats, top-1 {:.1}%; {} tensors, {:.0} MB saved)",
                    r.mean_kl, r.top1 * 100.0, stats.reencoded, saved as f64 / 1e6
                ));
                set_native_applied(policy, saved, user.force);
                result = Some(Ok(cand));
                break;
            }
            Ok(r) => {
                crate::model::reencode::retain_pending_reencode(policy, pending);
                log(format!(
                    "reencode: {label} rejected (mean KL {:.4} nats, top-1 {:.1}%)",
                    r.mean_kl,
                    r.top1 * 100.0
                ));
            }
            Err(e) => {
                log(format!("reencode: gate unavailable ({e}); using bit-exact"));
                result = other_exact(&load_as, user.force);
                break;
            }
        }
    }
    crate::model::reencode::clear_retained_pending();
    set_gate_context(None);
    set_reencode_override(previous);
    result
}

/// The bit-exact load (still under `--quant`'s `force`), as the last resort when the gate cannot run.
fn other_exact<T>(
    load_as: &dyn Fn(ReencodePolicy, bool, bool) -> Option<Result<T, String>>,
    force: Option<crate::hardware::QuantFormat>,
) -> Option<Result<T, String>> {
    let r = load_as(ReencodePolicy::BitExact, false, false);
    crate::model::reencode::set_native_applied(ReencodePolicy::BitExact, 0, force);
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::loaders::gguf::test_gguf_blocks;
    use serde_json::json;

    fn f32b(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    fn bf16b(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|&x| f32_to_bf16(x).to_le_bytes())
            .collect()
    }
    fn f16b(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|&x| f32_to_f16(x).to_le_bytes())
            .collect()
    }
    fn iota(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    type St<'a> = (&'a str, &'a str, &'a [usize], Vec<u8>);

    /// A safetensors file of `(name, dtype, shape, bytes)`.
    fn write_st(p: &Path, meta: Option<J>, tensors: &[St]) {
        let (mut h, mut data) = (Map::new(), Vec::new());
        if let Some(m) = meta {
            h.insert("__metadata__".into(), m);
        }
        for (n, dt, shape, b) in tensors {
            h.insert(
                n.to_string(),
                json!({"dtype": dt, "shape": shape, "data_offsets": [data.len(), data.len() + b.len()]}),
            );
            data.extend_from_slice(b);
        }
        let mut hb = serde_json::to_vec(&J::Object(h)).unwrap();
        hb.resize(hb.len().next_multiple_of(8), b' ');
        let mut out = (hb.len() as u64).to_le_bytes().to_vec();
        out.extend(hb);
        out.extend(data);
        std::fs::write(p, out).unwrap();
    }

    fn registry(specs: &[&str]) -> Registry {
        let files: Vec<(String, String)> = specs
            .iter()
            .enumerate()
            .map(|(i, s)| (format!("families/f{i}.json5"), s.to_string()))
            .collect();
        Registry::from_files(&files, &[]).unwrap()
    }

    fn exact() -> ReencodeConfig {
        ReencodeConfig {
            policy: ReencodePolicy::BitExact,
            ..Default::default()
        }
    }

    fn artifacts(dir: &Path, weights: Option<&str>, cfg: Option<J>) -> ModelArtifacts {
        if let Some(c) = &cfg {
            std::fs::write(dir.join("config.json"), c.to_string()).unwrap();
        }
        ModelArtifacts {
            model_id: "test/toy".into(),
            local_dir: Some(dir.to_path_buf()),
            config_path: cfg.map(|_| dir.join("config.json")),
            weights_path: weights.map(|w| dir.join(w)),
            ..Default::default()
        }
    }

    /// A toy safetensors checkpoint (`model_type: toy`, hidden 4) loaded under `spec`'s body.
    fn toy(dir: &Path, tensors: &[St], body: &str) -> Result<Model, String> {
        write_st(&dir.join("model.safetensors"), None, tensors);
        let a = artifacts(
            dir,
            Some("model.safetensors"),
            Some(json!({"model_type": "toy", "hidden_size": 4})),
        );
        let spec = format!(
            r#"{{family: "toy", match: {{hf_model_type: ["toy"]}}, pipeline: "causal_lm", {body}}}"#
        );
        load_with(&a, &registry(&[&spec]), None, exact())
    }

    #[test]
    fn projection_copy_reencodes_and_round_trips() {
        let d = tempfile::tempdir().unwrap();
        let (rows, cols) = (64, 256);
        let v: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 40.0)
            .collect();
        write_st(
            &d.path().join("model.safetensors"),
            None,
            &[("embed", "BF16", &[rows, cols], bf16b(&v))],
        );
        let a = artifacts(
            d.path(),
            Some("model.safetensors"),
            Some(json!({"model_type": "toy"})),
        );
        let spec = r#"{family: "toy", match: {hf_model_type: ["toy"]}, pipeline: "causal_lm"}"#;
        let cfg = ReencodeConfig {
            policy: ReencodePolicy::MaxSpeed,
            ..exact()
        };
        let m = load_with(&a, &registry(&[spec]), None, cfg).unwrap();
        let name = format!("embed{}", crate::arch::build::LINEAR_COPY);
        let (_, dtype) = m.store.tensor(&name).unwrap();
        let t = get(&m, &name);
        assert_eq!(dtype_of(&t), dtype);
        let Tensor::Mat(mat) = &t else {
            panic!("not a Mat")
        };
        let p = mat
            .packed()
            .expect("max-speed re-encodes the projection copy");
        let (elems, bytes, dec) = crate::model::layers::gguf_block_codec(p.format).unwrap();
        let mut got = vec![0.0f32; rows * cols];
        for (b, o) in p.map[p.off..]
            .chunks_exact(bytes)
            .zip(got.chunks_exact_mut(elems))
        {
            dec(b, o);
        }
        let ref_v: Vec<f32> = v.iter().map(|&x| bf16_to_f32(f32_to_bf16(x))).collect();
        assert!(crate::model::reencode::relative_rmse(&ref_v, &got) < 0.1);
        // The gather table itself keeps its encoding.
        assert!(matches!(get(&m, "embed"), Tensor::Mat(m) if m.packed().is_none()));
    }

    #[test]
    fn reencoded_blocks_persist_and_mmap_zero_copy_on_second_load() {
        use crate::model::weight_format::{reencode_override, set_reencode_override};
        let _test_guard = TEST_GATED_LOCK.lock();
        let d = tempfile::tempdir().unwrap();
        let (rows, cols) = (64, 256);
        let v: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 40.0)
            .collect();
        let weights_file = d.path().join("model.safetensors");
        write_st(
            &weights_file,
            None,
            &[
                ("attn.w", "BF16", &[rows, cols], bf16b(&v)),
                ("mlp.w", "BF16", &[rows, cols], bf16b(&v)),
            ],
        );
        let a = artifacts(
            d.path(),
            Some("model.safetensors"),
            Some(json!({"model_type": "toy"})),
        );
        let spec = r#"{family: "toy", match: {hf_model_type: ["toy"]}, pipeline: "causal_lm"}"#;
        let reg = registry(&[spec]);
        let prev = reencode_override();
        crate::model::reencode::with_test_cache_dir(d.path(), || {
            set_reencode_override(Some(ReencodeConfig {
                policy: ReencodePolicy::MaxSpeed,
                ..exact()
            }));
            let load_model = || {
                let m = load(&a, &reg, None)?;
                let _ = m.store.tensor("attn.w");
                let _ = m.store.tensor("mlp.w");
                Ok(m)
            };
            // Pass 1: reject rung 0 (max-speed, ffn_only=false), accept rung 1 (max-speed, ffn_only=true).
            let kl_calls = std::cell::Cell::new(0usize);
            let m1 = gated(
                true,
                || Some(load_model()),
                |_, _| {
                    let c = kl_calls.get();
                    kl_calls.set(c + 1);
                    Ok(GateReport {
                        mean_kl: if c == 0 { 0.50 } else { 0.02 },
                        top1: 0.99,
                        tokens: 48,
                    })
                },
            )
            .unwrap()
            .unwrap();
            assert_eq!(kl_calls.get(), 2, "rung 0 rejected, rung 1 accepted");
            let key = crate::model::reencode::choice_key(
                &weights_file,
                ReencodePolicy::MaxSpeed,
                None,
                None,
            )
            .unwrap();
            let cache_dir = d.path().join("reencoded");
            let blocks_path = cache_dir.join(format!("{key}.blocks"));
            let index_path = cache_dir.join(format!("{key}.index.json"));
            let choice_path = cache_dir.join(format!("{key}.json"));
            assert!(blocks_path.is_file(), ".blocks file must be written");
            assert!(index_path.is_file(), ".index.json file must be written");
            assert!(choice_path.is_file(), ".json choice file must be written");

            // Pass 2: jumps directly to accepted rung 1 in ONE load_once call, 0 kl calls, and mmaps .blocks.
            let load_calls = std::cell::Cell::new(0usize);
            let m2 = gated(
                true,
                || {
                    load_calls.set(load_calls.get() + 1);
                    Some(load_model())
                },
                |_, _| panic!("second load must hit reencode cache without running KL gate"),
            )
            .unwrap()
            .unwrap();
            assert_eq!(
                load_calls.get(),
                1,
                "warm load must make exactly one load_once call"
            );
            assert_eq!(crate::model::reencode::native_stats().cache_hits, 1);
            let Tensor::Mat(mat1) = get(&m1, "mlp.w") else {
                panic!("expected Mat")
            };
            let Tensor::Mat(mat2) = get(&m2, "mlp.w") else {
                panic!("expected Mat")
            };
            let p1 = mat1.packed().unwrap();
            let p2 = mat2.packed().unwrap();
            assert!(
                matches!(&*p2.map, GgufData::Map(_)),
                "second load must mmap .blocks"
            );
            assert_eq!(p1.format, p2.format);
            let len = rows * p1.format.tensor_bytes(cols);
            assert_eq!(
                &p1.map[p1.off..p1.off + len],
                &p2.map[p2.off..p2.off + len],
                "mapped blocks must be bit-identical"
            );
            // Non-FFN tensor stayed in its native BF16 representation.
            let Tensor::Mat(attn2) = get(&m2, "attn.w") else {
                panic!("expected Mat")
            };
            assert!(attn2.packed().is_none());

            // Tighter max_kl (0.005 < 0.02) does not blindly accept the 0.02 cached rung.
            set_reencode_override(Some(ReencodeConfig {
                policy: ReencodePolicy::MaxSpeed,
                max_kl: Some(0.005),
                ..exact()
            }));
            let tight_kl_calls = std::cell::Cell::new(0usize);
            let _ = gated(
                true,
                || Some(load_model()),
                |_, _| {
                    tight_kl_calls.set(tight_kl_calls.get() + 1);
                    Ok(GateReport {
                        mean_kl: 0.001,
                        top1: 1.0,
                        tokens: 48,
                    })
                },
            )
            .unwrap()
            .unwrap();
            assert_eq!(tight_kl_calls.get(), 1, "new max_kl must run the KL gate");

            // Size mismatch in .index.json ignores the cache entry and re-encodes.
            set_reencode_override(Some(ReencodeConfig {
                policy: ReencodePolicy::MaxSpeed,
                ..exact()
            }));
            let _ = std::fs::remove_file(&choice_path);
            let mut idx: BTreeMap<String, crate::model::reencode::CachedTensor> =
                serde_json::from_slice(&std::fs::read(&index_path).unwrap()).unwrap();
            idx.get_mut("mlp.w").unwrap().rows = rows + 1;
            std::fs::write(&index_path, serde_json::to_vec(&idx).unwrap()).unwrap();
            let m3 = load_model().unwrap();
            let stats = crate::model::reencode::native_stats();
            assert!(stats.cache_mismatch);
            assert_eq!(stats.cache_hits, 0);
            assert_eq!(stats.reencoded, 2);
            let Tensor::Mat(mat3) = get(&m3, "mlp.w") else {
                panic!("expected Mat")
            };
            let p3 = mat3.packed().unwrap();
            assert_eq!(&p1.map[p1.off..p1.off + len], &p3.map[p3.off..p3.off + len]);
        });
        set_reencode_override(prev);
    }

    fn get(m: &Model, n: &str) -> Tensor {
        m.store
            .materialize(&[n.to_string()])
            .unwrap()
            .get(n)
            .unwrap()
            .clone()
    }

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn safetensors_f32_bf16_f16_open_and_materialize() {
        let d = tempfile::tempdir().unwrap();
        let m = toy(
            d.path(),
            &[
                ("w32", "F32", &[2, 3], f32b(&iota(6))),
                ("w16", "BF16", &[3, 4], bf16b(&iota(12))),
                ("h16", "F16", &[2, 2], f16b(&[0.5, 1.5, -2.0, 4.0])),
                ("norm", "BF16", &[3], bf16b(&[1.0, 2.0, 3.0])),
                ("bank", "BF16", &[2, 1, 3], bf16b(&iota(6))),
                ("conv2d", "BF16", &[1, 1, 1, 2], bf16b(&[1.0, 2.0])),
            ],
            "",
        )
        .unwrap();
        assert_eq!(m.spec.family, "toy");
        let s = &m.store;
        assert_eq!(s.tensor("w32"), Some((vec![2, 3], DType::F32)));
        assert_eq!(s.tensor("w16"), Some((vec![3, 4], DType::BF16)));
        assert_eq!(s.tensor("h16"), Some((vec![2, 2], DType::F16)));
        assert_eq!(s.tensor("norm"), Some((vec![3], DType::F32)));
        assert_eq!(s.tensor("bank"), Some((vec![2, 1, 3], DType::BF16)));
        assert_eq!(s.tensor("conv2d"), Some((vec![1, 1, 1, 2], DType::F32)));
        assert!(matches!(get(&m, "w32"), Tensor::F32(v) if v == iota(6)));
        let Tensor::Mat(w) = get(&m, "w16") else {
            panic!("bf16 matrix must be a Mat")
        };
        assert!(w.packed().is_none() && (w.rows, w.cols) == (3, 4));
        assert_eq!(w.to_f32(), iota(12));
        let Tensor::F16(h) = get(&m, "h16") else {
            panic!("f16 matrix must be F16")
        };
        assert_eq!(
            h,
            vec![
                f32_to_f16(0.5),
                f32_to_f16(1.5),
                f32_to_f16(-2.0),
                f32_to_f16(4.0)
            ]
        );
        assert!(matches!(get(&m, "norm"), Tensor::F32(v) if v == [1.0, 2.0, 3.0]));
        assert!(matches!(get(&m, "conv2d"), Tensor::F32(v) if v == [1.0, 2.0]));
        // Every materialized variant reports the dtype `Source::tensor` promised.
        for n in s.names() {
            assert_eq!(dtype_of(&get(&m, &n)), s.tensor(&n).unwrap().1, "{n}");
        }
    }

    #[test]
    fn precompute_outputs_become_weights() {
        let d = tempfile::tempdir().unwrap();
        let m = toy(
            d.path(),
            &[
                ("w32", "F32", &[2, 3], f32b(&iota(6))),
                ("x", "F32", &[2], f32b(&[5.0, 6.0])),
            ],
            r#"components: {precompute: {body: [
                {repeat: "i", count: 2, body: [
                  {op: "Unary", args: {kind: {Scale: "i + 1"}}, in: ["w:w32"], out: "y"},
                  {tap: "scaled.{i}", in: "y"}]},
                {op: "Transpose", args: {perm: [1, 0]}, in: ["w:w32"], out: "t"}],
              outputs: ["t"]}}"#,
        )
        .unwrap();
        assert!(m.env.has("scaled.0") && m.env.has("scaled.1"));
        assert_eq!(m.env.shape("t"), Some(vec![3, 2]));
        assert_eq!(m.store.tensor("scaled.1"), Some((vec![2, 3], DType::F32)));
        assert!(
            matches!(get(&m, "scaled.1"), Tensor::F32(v) if v == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        );
        assert!(matches!(get(&m, "t"), Tensor::F32(v) if v == [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]));
        // A precomputed weight consumes the checkpoint tensors precompute read.
        assert_eq!(
            m.store.unconsumed(&HashSet::from(["t".to_string()]), &[]),
            ["x"]
        );
    }

    fn kv_str(k: &str, v: &str) -> (String, u32, Vec<u8>) {
        let mut p = (v.len() as u64).to_le_bytes().to_vec();
        p.extend_from_slice(v.as_bytes());
        (k.into(), 8, p)
    }
    fn kv_u32(k: &str, v: u32) -> (String, u32, Vec<u8>) {
        (k.into(), 4, v.to_le_bytes().to_vec())
    }
    /// Q8_0 blocks (`d` = 0.5) whose row `r` holds `r*32 + i - 16`, i in 0..32 (wrapping i8).
    fn q8_rows(rows: usize) -> (Vec<u8>, Vec<f32>) {
        let (mut b, mut v) = (Vec::new(), Vec::new());
        for r in 0..rows {
            b.extend_from_slice(&f32_to_f16(0.5).to_le_bytes());
            for i in 0..32 {
                let q = (r as i32 * 7 + i - 16) as i8;
                b.push(q as u8);
                v.push(q as f32 * 0.5);
            }
        }
        (b, v)
    }

    #[test]
    fn gguf_dims_reverse_blocks_stay_packed_and_metadata_falls_back() {
        let d = tempfile::tempdir().unwrap();
        let (q8, q8v) = q8_rows(2);
        // Q4_0, d = 1: low nibble j -> j - 8, high nibble (15 - j) -> 7 - j.
        let mut q4 = f32_to_f16(1.0).to_le_bytes().to_vec();
        q4.extend((0..16u8).map(|j| j | ((15 - j) << 4)));
        let q4v: Vec<f32> = (0..16)
            .map(|j| j as f32 - 8.0)
            .chain((0..16).map(|j| 7.0 - j as f32))
            .collect();
        let (ssm, ssmv) = q8_rows(4);
        let (e0, e0v) = q8_rows(1);
        let file = test_gguf_blocks(
            &[
                kv_str("general.architecture", "toy"),
                kv_u32("toy.block_count", 2),
                kv_u32("context_length", 64),
            ],
            &[
                ("blk.0.attn_q.weight".into(), vec![32, 2], 8, q8.clone()),
                ("blk.1.attn_q.weight".into(), vec![32, 1], 2, q4),
                ("blk.0.ssm.weight".into(), vec![32, 4], 8, ssm),
                ("blk.0.exp.weight".into(), vec![32, 1], 8, e0.clone()),
                ("blk.1.exp.weight".into(), vec![32, 1], 8, e0),
                ("tok.weight".into(), vec![3, 2], 0, f32b(&iota(6))),
                (
                    "output_norm.weight".into(),
                    vec![3],
                    0,
                    f32b(&[1.0, 2.0, 3.0]),
                ),
            ],
        );
        std::fs::write(d.path().join("toy.gguf"), file).unwrap();
        let spec = r#"{family: "toy", match: {gguf_arch: ["toy"]}, pipeline: "causal_lm",
            config: {layers: "gguf.block_count", ctx: "gguf.context_length"},
            names: {rules: [["blk.{i}.attn_q.weight", "model.layers.{i}.q.weight"]]},
            transforms: [
              {tensor: "blk.0.ssm.weight", op: "untile_packed_rows",
               args: {k_heads: 2, v_heads: 4, rows_per_block: 1}},
              {tensor: "model.layers.0.q.weight", op: "split_rows",
               args: {parts: [["q0.a", 1], ["q0.b", 1]]}},
              {tensor: "blk.{e}.exp.weight", op: "stack_experts", args: {out: "exp.weight", count: 2}},
              {tensor: "exp.weight", op: "reshape", args: {shape: [2, 32]}},
            ]}"#;
        let a = artifacts(d.path(), Some("toy.gguf"), None);
        let m = load_with(&a, &registry(&[spec]), None, exact()).unwrap();
        assert_eq!(m.env.var("layers").unwrap(), Val::Num(2.0), "`toy.` prefix");
        assert_eq!(
            m.env.var("ctx").unwrap(),
            Val::Num(64.0),
            "unprefixed fallback"
        );
        let s = &m.store;
        assert_eq!(
            s.tensor("model.layers.0.q.weight"),
            Some((vec![2, 32], DType::Quant(Q::GGUF_Q8_0)))
        );
        assert_eq!(
            s.tensor("model.layers.1.q.weight"),
            Some((vec![1, 32], DType::Quant(Q::GGUF_Q4_0)))
        );
        assert_eq!(s.tensor("tok.weight"), Some((vec![2, 3], DType::F32)));
        assert_eq!(s.tensor("output_norm.weight"), Some((vec![3], DType::F32)));
        let Tensor::Mat(q) = get(&m, "model.layers.0.q.weight") else {
            panic!()
        };
        let p = q.packed().expect("Q8_0 stays packed");
        assert!(
            Arc::ptr_eq(&p.map, m.ckpt.container.as_ref().unwrap()),
            "zero-copy"
        );
        assert_eq!((q.rows, q.cols, p.format), (2, 32, Q::GGUF_Q8_0));
        assert_eq!(q.to_f32(), q8v);
        assert_eq!(get(&m, "model.layers.1.q.weight").to_f32(), q4v);
        assert!(matches!(get(&m, "tok.weight"), Tensor::F32(v) if v == iota(6)));
        // Row-granular untile keeps the blocks; rows (0, 1, 2, 3) -> (0, 2, 1, 3).
        assert_eq!(
            s.tensor("blk.0.ssm.weight"),
            Some((vec![4, 32], DType::Quant(Q::GGUF_Q8_0)))
        );
        let Tensor::Mat(u) = get(&m, "blk.0.ssm.weight") else {
            panic!()
        };
        assert!(u.packed().is_some());
        let want: Vec<f32> = [0, 2, 1, 3]
            .iter()
            .flat_map(|&r| ssmv[r * 32..(r + 1) * 32].to_vec())
            .collect();
        assert_eq!(u.to_f32(), want);
        // A split of packed rows decodes to bf16.
        assert_eq!(s.tensor("q0.b"), Some((vec![1, 32], DType::BF16)));
        let Tensor::Mat(b) = get(&m, "q0.b") else {
            panic!()
        };
        assert!(b.packed().is_none() && close(&b.to_f32(), &q8v[32..]));
        // Stacked and reshaped ([2, 1, 32] -> [2, 32]) packed experts stay packed.
        assert_eq!(
            s.tensor("exp.weight"),
            Some((vec![2, 32], DType::Quant(Q::GGUF_Q8_0)))
        );
        let Tensor::Mat(exp) = get(&m, "exp.weight") else {
            panic!()
        };
        assert!(exp.packed().is_some());
        assert_eq!(exp.to_f32(), [e0v.clone(), e0v].concat());
    }

    #[test]
    fn diffusers_class_from_model_index_else_config() {
        let d = tempfile::tempdir().unwrap();
        let a = artifacts(
            d.path(),
            None,
            Some(json!({"_class_name": "AutoencoderKL"})),
        );
        assert_eq!(
            open(&a).unwrap().sig.diffusers_class.as_deref(),
            Some("AutoencoderKL")
        );
        std::fs::write(
            d.path().join("model_index.json"),
            r#"{"_class_name": "StableDiffusionPipeline"}"#,
        )
        .unwrap();
        assert_eq!(
            open(&a).unwrap().sig.diffusers_class.as_deref(),
            Some("StableDiffusionPipeline")
        );
    }

    #[test]
    fn renames_map_names_and_refuse_conflicts() {
        let d = tempfile::tempdir().unwrap();
        let t: &[St] = &[
            ("a.w", "F32", &[1], f32b(&[1.0])),
            ("b.w", "F32", &[1], f32b(&[2.0])),
        ];
        let m = toy(
            d.path(),
            t,
            r#"names: {rules: [["{x:*}.w", "layers.{x}.weight"]]}"#,
        )
        .unwrap();
        assert_eq!(m.store.names(), ["layers.a.weight", "layers.b.weight"]);
        let e = toy(
            d.path(),
            t,
            r#"names: {rules: [["a.w", "x"], ["b.w", "x"]]}"#,
        )
        .err()
        .unwrap();
        assert!(
            e.contains("`a.w`") && e.contains("`b.w`") && e.contains("`x`"),
            "{e}"
        );
    }

    #[test]
    fn every_transform() {
        let d = tempfile::tempdir().unwrap();
        let m = toy(
            d.path(),
            &[
                ("add", "F32", &[2, 3], f32b(&iota(6))),
                ("sc", "F32", &[3], f32b(&[1.0, 2.0, 3.0])),
                ("sp", "F32", &[2], f32b(&[0.0, 1.0])),
                ("isp", "F32", &[1], f32b(&[2f32.ln()])),
                ("ne", "F32", &[1], f32b(&[0.0])),
                ("tr", "F32", &[2, 3], f32b(&iota(6))),
                ("rs", "F32", &[2, 3], f32b(&iota(6))),
                ("conv", "BF16", &[2, 1, 3], bf16b(&iota(6))),
                ("rb", "F32", &[1], f32b(&[1.0 + 1.0 / 1024.0])),
                ("rb16", "F16", &[1], f16b(&[1.0 + 1.0 / 1024.0])),
                ("qkv", "F32", &[4, 2], f32b(&iota(8))),
                ("fused", "F32", &[6, 1], f32b(&iota(6))),
                ("ut", "F32", &[6], f32b(&iota(6))),
                ("upr", "F32", &[4, 1], f32b(&iota(4))),
                ("ph", "F32", &[4, 1], f32b(&iota(4))),
                ("c2l", "F32", &[2, 2, 2], f32b(&iota(8))),
                ("ltt", "F32", &[1, 1, 2, 1, 2], f32b(&iota(4))),
                ("tern", "F32", &[4], f32b(&[0.5, -2.0, 0.1, 1.0])),
                ("e.0.w", "F32", &[2, 2], f32b(&[1.0, 2.0, 3.0, 4.0])),
                ("e.1.w", "F32", &[2, 2], f32b(&[5.0, 6.0, 7.0, 8.0])),
                ("lm", "F32", &[2, 2], f32b(&[0.0; 4])),
                ("lma", "F32", &[1, 2], f32b(&[1.0, 2.0])),
                ("lmb", "F32", &[2, 1], f32b(&[3.0, 4.0])),
                ("chain", "F32", &[1], f32b(&[1.0])),
                ("layer.0.n", "F32", &[1], f32b(&[0.0])),
                ("layer.1.n", "F32", &[1], f32b(&[0.0])),
            ],
            r#"config: {bump: "1"},
            transforms: [
              {tensor: "add", op: "add", args: {value: "bump"}},
              {tensor: "sc", op: "scale", args: {value: 2}},
              {tensor: "sc", op: "add", args: {value: 100}, when: "false"},
              {tensor: "sp", op: "softplus"},
              {tensor: "isp", op: "inv_softplus"},
              {tensor: "ne", op: "neg_exp"},
              {tensor: "tr", op: "transpose"},
              {tensor: "rs", op: "reshape", args: {shape: [3, 2]}},
              {tensor: "conv", op: "to_f32"},
              {tensor: "rb", op: "round_bf16"},
              {tensor: "rb16", op: "round_bf16"},
              {tensor: "qkv", op: "split_rows", args: {parts: [["q", 2], ["k", 1], ["v", "bump"]]}},
              {tensor: "fused", op: "deinterleave_heads", args: {heads: 2, parts: [["x", 1], ["y", 2]]}},
              {tensor: "ut", op: "untile_heads", args: {k_heads: 2, block: 1}},
              {tensor: "upr", op: "untile_packed_rows", args: {k_heads: 2, v_heads: 4, rows_per_block: 1}},
              {tensor: "ph", op: "permute_head_dims", args: {perm: [1, 0]}},
              {tensor: "c2l", op: "conv_to_linear", args: {order: "k_in"}},
              {tensor: "ltt", op: "last_temporal_tap"},
              {tensor: "tern", op: "ternarize"},
              {tensor: "e.{e}.w", op: "stack_experts", args: {out: "e.w"}},
              {tensor: "lm", op: "lora_merge", args: {a: "lma", b: "lmb", scale: 0.5}},
              {tensor: "chain", op: "add", args: {value: 1}},
              {tensor: "chain", op: "scale", args: {value: 3}},
              {tensor: "layer.{i}.n", op: "add", args: {value: "i + 1"}},
            ]"#,
        )
        .unwrap();
        let s = &m.store;
        // Shapes are known before any data is read.
        for (n, shape) in [
            ("tr", vec![3, 2]),
            ("rs", vec![3, 2]),
            ("q", vec![2, 2]),
            ("y", vec![4, 1]),
            ("c2l", vec![2, 4]),
            ("ltt", vec![1, 1, 1, 2]),
            ("e.w", vec![2, 2, 2]),
        ] {
            assert_eq!(s.tensor(n).unwrap().0, shape, "{n}");
        }
        assert!(s.cache.lock().is_empty());
        let v = |n: &str| get(&m, n).to_f32();
        assert_eq!(v("add"), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(v("sc"), [2.0, 4.0, 6.0]);
        assert!(close(&v("sp"), &[2f32.ln(), 1f32.exp().ln_1p()]));
        assert!(close(&v("isp"), &[0.0]));
        assert_eq!(v("ne"), [-1.0]);
        assert_eq!(v("tr"), [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert_eq!(v("rs"), iota(6));
        assert_eq!(s.tensor("conv"), Some((vec![2, 1, 3], DType::F32)));
        assert!(matches!(get(&m, "conv"), Tensor::F32(x) if x == iota(6)));
        assert_eq!(v("rb"), [1.0]);
        // round_bf16 leaves non-f32 tensors in their stored encoding (f16 here, packed blocks alike).
        assert_eq!(v("rb16"), [1.0 + 1.0 / 1024.0]);
        assert_eq!(
            (v("q"), v("k"), v("v")),
            (vec![0.0, 1.0, 2.0, 3.0], vec![4.0, 5.0], vec![6.0, 7.0])
        );
        assert_eq!((v("x"), v("y")), (vec![0.0, 3.0], vec![1.0, 2.0, 4.0, 5.0]));
        assert_eq!(v("ut"), [0.0, 2.0, 4.0, 1.0, 3.0, 5.0]);
        assert_eq!(v("upr"), [0.0, 2.0, 1.0, 3.0]);
        assert_eq!(v("ph"), [1.0, 0.0, 3.0, 2.0]);
        assert_eq!(v("c2l"), [0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);
        assert_eq!(v("ltt"), [2.0, 3.0]);
        assert!(close(&v("tern"), &[0.9, -0.9, 0.0, 0.9]));
        assert_eq!(
            v("e.w"),
            iota(8).iter().map(|x| x + 1.0).collect::<Vec<_>>()
        );
        assert_eq!(v("lm"), [1.5, 3.0, 2.0, 4.0]);
        assert_eq!(v("chain"), [6.0]);
        assert_eq!((v("layer.0.n"), v("layer.1.n")), (vec![1.0], vec![2.0]));
        // Derived results are cached; f32 sources stay f32.
        assert_eq!(s.tensor("tr").unwrap().1, DType::F32);
        assert!(s.cache.lock().contains_key("tr"));
        let fresh = tempfile::tempdir().unwrap();
        let e = toy(
            fresh.path(),
            &[("a", "F32", &[3], f32b(&iota(3)))],
            r#"transforms: [{tensor: "a", op: "reshape", args: {shape: [2]}}]"#,
        )
        .err()
        .unwrap();
        assert!(e.contains("reshape") && e.contains("`a`"), "{e}");
        let fresh = tempfile::tempdir().unwrap();
        let e = toy(
            fresh.path(),
            &[("u", "F32", &[4, 1], f32b(&iota(4)))],
            r#"transforms: [{tensor: "u", op: "untile_packed_rows", args: {k_heads: 2, v_heads: 3, rows_per_block: 1}}]"#,
        )
        .err()
        .unwrap();
        assert!(
            e.contains("untile_packed_rows") && e.contains("do not tile"),
            "{e}"
        );
    }

    #[test]
    fn constants_generate_at_load() {
        let d = tempfile::tempdir().unwrap();
        let m = toy(
            d.path(),
            &[],
            r#"constants: {
              ones: {gen: "ones", args: {shape: ["cfg.hidden_size"]}},
              idx: {gen: "iota", args: {n: 3}},
              skipped: {gen: "ones", args: {shape: [1]}, when: "false"},
            }"#,
        )
        .unwrap();
        assert_eq!(m.store.tensor("ones"), Some((vec![4], DType::F32)));
        assert_eq!(m.store.tensor("idx"), Some((vec![3], DType::U32)));
        assert!(m.store.tensor("skipped").is_none() && !m.env.has("skipped"));
        assert!(matches!(get(&m, "ones"), Tensor::F32(v) if v == [1.0; 4]));
        assert!(matches!(get(&m, "idx"), Tensor::U32(v) if v == [0, 1, 2]));
    }

    #[test]
    fn unconsumed_lists_unused_raw_tensors() {
        let d = tempfile::tempdir().unwrap();
        let one = || f32b(&[1.0]);
        let m = toy(
            d.path(),
            &[
                ("used.weight", "F32", &[1], one()),
                ("used.weight_scale", "F32", &[1], one()),
                ("used.bias", "F32", &[1], one()),
                ("unused.weight", "F32", &[1], one()),
                ("skip.me", "F32", &[1], one()),
                ("raw.thing", "F32", &[1], one()),
                ("src.w", "F32", &[1], one()),
                ("maybe.a", "F32", &[1], one()),
                ("maybe.b", "F32", &[1], one()),
            ],
            // `{glob, when}`: ignored only when `when` holds.
            r#"ignore: ["skip.*", "canon.*", {glob: "maybe.a", when: "has('skip.me')"},
                     {glob: "maybe.b", when: "has('nope')"}],
            names: {rules: [["raw.thing", "canon.thing"]]},
            transforms: [{tensor: "src.w", op: "scale", args: {value: 2}}]"#,
        )
        .unwrap();
        let used: HashSet<String> = ["used.weight", "src.w"].map(String::from).into();
        let ignore = m.ignore.clone();
        assert_eq!(
            m.store.unconsumed(&used, &ignore),
            ["maybe.b", "unused.weight", "used.bias"]
        );
        // A `values(name)` read (a build-time constant) consumes the tensor.
        assert_eq!(m.env.values("used.bias"), Some(vec![1.0]));
        assert_eq!(
            m.store.unconsumed(&used, &ignore),
            ["maybe.b", "unused.weight"]
        );
    }

    #[test]
    fn config_vars_are_lazy_memoized_and_refuse_cycles() {
        let d = tempfile::tempdir().unwrap();
        let m = toy(
            d.path(),
            &[],
            r#"config: {a: "b + 1", b: "cfg.hidden_size", c: "d", d: "c", e: "c + 1",
                        f: "first(cfg.missing, 7)", r: "runtime.ctx"},
            runtime: {ffn_weights: ["*.mlp.*"]},
            checks: [{expr: "a > 4", error: "too big: {a}"}]"#,
        )
        .expect("unused cyclic vars do not fail the load");
        assert_eq!(m.env.var("a").unwrap(), Val::Num(5.0));
        assert_eq!(m.env.var("f").unwrap(), Val::Num(7.0));
        let e = m.env.var("c").unwrap_err();
        assert!(e.contains("cycle") && e.contains("c -> d -> c"), "{e}");
        assert!(m.env.var("e").unwrap_err().contains("cycle"));
        assert_eq!(
            m.env.var("a").unwrap(),
            Val::Num(5.0),
            "still fine after an error"
        );
        assert_eq!(m.env.var("r").unwrap(), Val::Null);
        m.env.set_ctx(json!(4096));
        assert_eq!(m.env.var("r").unwrap(), Val::Num(4096.0));
        assert_eq!(check(&m).unwrap_err(), "too big: 5");
        assert_eq!(
            m.store.reencode.ffn_weights.as_deref(),
            Some(&vec!["*.mlp.*".to_string()])
        );
    }

    #[test]
    fn synthetic_ids_random_init_deterministically() {
        let d = tempfile::tempdir().unwrap();
        let mut a = artifacts(
            d.path(),
            None,
            Some(
                json!({"architectures": ["LlamaForCausalLM"], "model_type": "decoy",
                        "hidden_size": 4, "num_hidden_layers": 2}),
            ),
        );
        a.model_id = "gibbs/toy-1m".into();
        let reg = registry(&[
            r#"{family: "decoy", match: {hf_model_type: ["decoy"]}, pipeline: "encoder"}"#,
            r#"{family: "toy", match: {hf_architectures: ["LlamaForCausalLM"]}, pipeline: "causal_lm",
                synthetic: {
                  "model.layers.{i}.w": {shape: ["cfg.hidden_size", 2], over: {i: "range(cfg.num_hidden_layers)"}},
                  "model.norm": {shape: ["cfg.hidden_size"]},
                }}"#,
        ]);
        let load = || load_with(&a, &reg, None, exact()).unwrap();
        let (m1, m2) = (load(), load());
        assert_eq!(
            m1.spec.family, "toy",
            "a synthetic config's model_type is only a label"
        );
        assert_eq!(
            m1.store.tensor("model.layers.1.w"),
            Some((vec![4, 2], DType::BF16))
        );
        assert_eq!(m1.store.tensor("model.norm"), Some((vec![4], DType::F32)));
        let (w0, w1) = (
            get(&m1, "model.layers.0.w").to_f32(),
            get(&m1, "model.layers.1.w").to_f32(),
        );
        assert_eq!(w0, get(&m2, "model.layers.0.w").to_f32(), "deterministic");
        assert_ne!(w0, w1, "per-name seeds");
        assert!(w0.iter().all(|x| x.abs() <= 0.02) && w0.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn peft_adapter_registers_lora_pairs() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join("adapter_config.json"),
            json!({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 2, "lora_alpha": 4})
                .to_string(),
        )
        .unwrap();
        write_st(
            &d.path().join("adapter_model.safetensors"),
            None,
            &[
                (
                    "base_model.model.model.q.lora_A.weight",
                    "F32",
                    &[2, 3],
                    f32b(&iota(6)),
                ),
                (
                    "base_model.model.model.q.lora_B.weight",
                    "F32",
                    &[4, 2],
                    f32b(&iota(8)),
                ),
            ],
        );
        let base_v: Vec<St> = vec![("model.q.weight", "F32", &[4, 3], f32b(&iota(12)))];
        let base: &[St] = &base_v;
        let m = toy(
            d.path(),
            base,
            r#"names: {rules: [["model.q.weight", "q.weight"]]}"#,
        )
        .unwrap();
        let l = m
            .store
            .lora("q.weight")
            .expect("registered on the canonical name");
        assert_eq!(
            (l.a.as_str(), l.b.as_str(), l.scale),
            ("q.weight.lora_A", "q.weight.lora_B", 2.0)
        );
        assert_eq!(m.store.tensor(&l.a), Some((vec![2, 3], DType::F32)));
        assert_eq!(m.store.tensor(&l.b), Some((vec![4, 2], DType::F32)));
        assert_eq!(get(&m, &l.b).to_f32(), iota(8));
        // A snapshot `peft` already merged the adapter into is not adapted twice.
        drop(m);
        write_st(
            &d.path().join("model.safetensors"),
            Some(json!({"peft_merged": "adapter_model"})),
            base,
        );
        let a = artifacts(d.path(), Some("model.safetensors"), None);
        let spec = r#"{family: "toy", match: {file_ext: ["safetensors"]}, pipeline: "causal_lm"}"#;
        let merged = load_with(&a, &registry(&[spec]), None, exact()).unwrap();
        assert!(merged.store.lora("model.q.weight").is_none());
        drop(merged);
        // An adapter naming another base (an adapter repo snapshot holding merged weights from
        // before the marker) is skipped; naming this checkpoint, it registers.
        for (owner, registered) in [("other/base", false), ("test/toy", true)] {
            std::fs::write(
                d.path().join("adapter_config.json"),
                json!({"peft_type": "LORA", "r": 2, "lora_alpha": 4, "base_model_name_or_path": owner})
                    .to_string(),
            )
            .unwrap();
            let m = toy(d.path(), base, "").unwrap();
            assert_eq!(
                m.store.lora("model.q.weight").is_some(),
                registered,
                "{owner}"
            );
        }
    }

    #[test]
    fn quant_force_restores_matrices() {
        let d = tempfile::tempdir().unwrap();
        let w = iota(64).iter().map(|x| x / 64.0).collect::<Vec<_>>();
        write_st(
            &d.path().join("model.safetensors"),
            None,
            &[
                ("w", "F32", &[2, 32], f32b(&w)),
                ("n", "F32", &[2], f32b(&[1.0, 2.0])),
            ],
        );
        let a = artifacts(
            d.path(),
            Some("model.safetensors"),
            Some(json!({"model_type": "toy"})),
        );
        let reg = registry(&[
            r#"{family: "toy", match: {hf_model_type: ["toy"]}, pipeline: "causal_lm"}"#,
        ]);
        for (f, dt) in [
            (Q::FP16, DType::F16),
            (Q::BF16, DType::BF16),
            (Q::GGUF_Q8_0, DType::Quant(Q::GGUF_Q8_0)),
        ] {
            let cfg = ReencodeConfig {
                force: Some(f),
                ..exact()
            };
            let m = load_with(&a, &reg, None, cfg).unwrap();
            assert_eq!(m.store.tensor("w"), Some((vec![2, 32], dt)));
            assert_eq!(
                m.store.tensor("n"),
                Some((vec![2], DType::F32)),
                "vectors stay f32"
            );
            let t = get(&m, "w");
            assert_eq!(dtype_of(&t), dt);
            assert!(
                t.to_f32().iter().zip(&w).all(|(a, b)| (a - b).abs() < 0.01),
                "{f:?}"
            );
        }
    }

    /// The re-encode ladder with fake loads: every lossy candidate "re-encodes" one tensor.
    #[test]
    #[ignore = "sets the process-global re-encode override and native stats; run alone with --ignored"]
    fn gate_ladder_order_and_fallbacks() {
        use crate::model::weight_format::{reencode_override, set_reencode_override};
        use ReencodePolicy::{BitExact as BE, LowBitNative as LB, MaxSpeed as MS, MinSize as MZ};
        let previous = reencode_override();
        set_reencode_override(Some(ReencodeConfig {
            policy: MS,
            ..Default::default()
        }));
        let seen = Mutex::new(Vec::new());
        let load = || {
            let c = ReencodeConfig::effective();
            crate::model::reencode::fake_native_reencoded(usize::from(c.policy != BE));
            let mut s = seen.lock();
            s.push((c.policy, c.ffn_only));
            Some(Ok(s.len()))
        };
        let report = |kl: f64| GateReport {
            mean_kl: kl,
            top1: 1.0,
            tokens: 8,
        };
        let take = || std::mem::take(&mut *seen.lock());
        // Every gate rejects: max-speed, (bit-exact reference), FFN-only, low-bit, bit-exact.
        let r = gated(true, load, |_, _| Ok(report(1e9)));
        assert_eq!(r, Some(Ok(5)));
        assert_eq!(
            take(),
            [
                (MS, false),
                (BE, false),
                (MS, true),
                (LB, false),
                (BE, false)
            ]
        );
        // MinSize ladder: min-size, (bit-exact reference), min-size (FFN only), max-speed, low-bit, bit-exact.
        set_reencode_override(Some(ReencodeConfig {
            policy: MZ,
            ..Default::default()
        }));
        let r = gated(true, load, |_, _| {
            assert!(matches!(ReencodeConfig::effective().policy, MZ | MS | LB));
            Ok(report(1e9))
        });
        assert_eq!(r, Some(Ok(6)));
        assert_eq!(
            take(),
            [
                (MZ, false),
                (BE, false),
                (MZ, true),
                (MS, false),
                (LB, false),
                (BE, false)
            ]
        );
        assert_eq!(
            gated(true, load, |_, _| {
                assert_eq!(
                    ReencodeConfig::effective().policy,
                    MZ,
                    "first candidate's kl runs with candidate's policy active"
                );
                Ok(report(0.0))
            }),
            Some(Ok(1))
        );
        assert_eq!(take(), [(MZ, false), (BE, false)]);
        set_reencode_override(Some(ReencodeConfig {
            policy: MS,
            ..Default::default()
        }));
        // The first candidate passes.
        assert_eq!(gated(true, load, |_, _| Ok(report(0.0))), Some(Ok(1)));
        assert_eq!(take(), [(MS, false), (BE, false)]);
        // The gate cannot run: one more bit-exact load.
        assert_eq!(gated(true, load, |_, _| Err("no gpu".into())), Some(Ok(3)));
        assert_eq!(take(), [(MS, false), (BE, false), (BE, false)]);
        // Off Metal: bit-exact only.
        assert_eq!(gated(false, load, |_, _| unreachable!()), Some(Ok(1)));
        assert_eq!(take(), [(BE, false)]);
        // Not this loader's architecture.
        let none = gated(
            true,
            || None::<Result<usize, String>>,
            |_, _| unreachable!(),
        );
        assert!(none.is_none());
        assert_eq!(
            reencode_override().map(|c| c.policy),
            Some(MS),
            "override restored"
        );
        set_reencode_override(previous);
    }
}
