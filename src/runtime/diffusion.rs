//! The diffusion pipeline. [`Net`] runs one network of a loaded family spec at any instance shape
//! (one compiled program per network; other shapes run reshaped on its weights).

use crate::arch::build::{build, Instance, ShapeCache};
use crate::arch::expr::{eval_json, eval_str, interp, walk, Env, Val};
use crate::arch::gen::{generate, GenData};
use crate::arch::load::Model;
use crate::graph::program::GraphProgram;
use crate::graph::Tensor;
use crate::hardware::PlacementMode;
use crate::model::hub::{HuggingFaceHub, ModelArtifacts};
use crate::model::image::ImageBuffer;
use crate::model::lora::Part as LoraPart;
use crate::model::weight_format::{ReencodeConfig, ReencodePolicy};
use crate::runtime::sched::{
    card_sigmas, default_sigmas, dpm_flow_sigmas, euler_step, linspace_sigmas, DpmSolver2, Mu, Rng,
    Schedule, Scheduler, SchedulerConfig, TRAIN_STEPS,
};
use parking_lot::{Mutex, RwLock};
use serde::Deserialize;
use serde_json::Value as J;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

/// One generation. `None` fields take the pipeline's [`ImageDefaults`].
#[derive(Debug, Clone, Default)]
pub struct ImageRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub steps: Option<usize>,
    /// Classifier-free guidance scale; distilled/turbo models ignore it.
    pub guidance: Option<f32>,
    pub seed: u64,
}

/// What a pipeline runs when the request leaves a field unset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageDefaults {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub guidance: f32,
    /// Pixel sides must be multiples of this.
    pub multiple: usize,
}

/// A text-to-image pipeline, as the server, the TUI and the CLI drive it.
pub trait ImageGenerator: Send + Sync {
    /// Architecture and weights, for logs and the TUI.
    fn summary(&self) -> String;
    /// Repo id it was loaded as, when known (listed by `/v1/models` and the TUI after a hot swap).
    fn model_id(&self) -> Option<String> {
        None
    }
    fn defaults(&self) -> ImageDefaults;
    fn text_to_image(&self, req: &ImageRequest) -> Result<ImageBuffer, String>;
    /// Denoises from `init` noised to `strength` (0..=1).
    fn image_to_image(
        &self,
        _req: &ImageRequest,
        _init: &ImageBuffer,
        _strength: f32,
    ) -> Result<ImageBuffer, String> {
        Err(format!(
            "{} does not support image-to-image",
            self.summary()
        ))
    }
}

/// The active image model, shared by the HTTP server and the TUI (hot-swapped by the picker).
pub type ImageSlot = Arc<RwLock<Option<Arc<dyn ImageGenerator>>>>;

/// Artifacts of a component directory (its `config.json` + weights).
pub fn dir_artifacts(dir: &Path) -> ModelArtifacts {
    let file = |f: &str| Some(dir.join(f)).filter(|p| p.is_file());
    ModelArtifacts {
        model_id: dir.display().to_string(),
        local_dir: Some(dir.to_path_buf()),
        config_path: file("config.json"),
        tokenizer_path: file("tokenizer.json"),
        weights_path: crate::model::hub::component_weights(dir).ok(),
        is_cached: true,
        ..Default::default()
    }
}

/// Step values a `gen` input's `"step:<expr>"` args read.
struct StepEnv<'a>(&'a [(&'a str, f64)]);

impl Env for StepEnv<'_> {
    fn lookup(&self, root: &str, path: &[Val]) -> Option<Val> {
        let v = self.0.iter().find(|(k, _)| *k == root)?.1;
        walk(Val::Num(v), path)
    }
}

/// `args` with every `"step:<expr>"` string evaluated over `step`.
fn step_args(args: &J, step: &[(&str, f64)]) -> Result<J, String> {
    Ok(match args {
        J::String(s) => match s.strip_prefix("step:") {
            Some(e) => eval_str(e, &StepEnv(step))
                .map_err(|e| format!("`{s}`: {e}"))?
                .to_json(),
            None => args.clone(),
        },
        J::Array(a) => J::Array(
            a.iter()
                .map(|x| step_args(x, step))
                .collect::<Result<_, _>>()?,
        ),
        J::Object(o) => J::Object(
            o.iter()
                .map(|(k, v)| Ok((k.clone(), step_args(v, step)?)))
                .collect::<Result<_, String>>()?,
        ),
        other => other.clone(),
    })
}

fn gen_tensor(name: &str, args: &J) -> Result<Tensor, String> {
    Ok(match generate(name, args)?.data {
        GenData::F32(v) => Tensor::F32(v),
        GenData::U32(v) => Tensor::U32(v),
    })
}

fn tensor_len(t: &Tensor) -> usize {
    match t {
        Tensor::F32(v) => v.len(),
        Tensor::U32(v) => v.len(),
        Tensor::F16(v) | Tensor::BF16(v) => v.len(),
        _ => 0,
    }
}

/// One network of a loaded model, run at any instance shape.
pub struct Net {
    pub model: Arc<Model>,
    pub net: String,
    mode: PlacementMode,
    shapes: ShapeCache,
    /// Compiled from the first instance run; every other instance runs reshaped on it.
    program: Mutex<Option<(String, GraphProgram)>>,
    used: Mutex<HashSet<String>>,
}

impl Net {
    pub fn new(model: Arc<Model>, net: &str, mode: PlacementMode) -> Self {
        Self {
            model,
            net: net.to_string(),
            mode,
            shapes: ShapeCache::new(net),
            program: Mutex::new(None),
            used: Mutex::new(HashSet::new()),
        }
    }

    /// Loads component directory `dir` as `family` and runs its network `net` on `mode`.
    pub fn open(dir: &Path, family: &str, net: &str, mode: PlacementMode) -> Result<Self, String> {
        let reencode = ReencodeConfig {
            policy: ReencodePolicy::BitExact,
            force: ReencodeConfig::effective().force,
            ..Default::default()
        };
        let m = crate::arch::load::load_with(
            &dir_artifacts(dir),
            crate::arch::registry()?,
            Some(family),
            reencode,
        )?;
        crate::arch::load::check(&m)?;
        Ok(Self::new(Arc::new(m), net, mode))
    }

    /// The instance of `vars` (built once per var tuple).
    pub fn instance(&self, vars: &[(&str, Val)]) -> Result<Arc<Instance>, String> {
        let m = &self.model;
        let inst = self
            .shapes
            .get(vars, || build(&m.spec, &self.net, &m.env, &m.store, vars))?;
        self.used.lock().extend(inst.weights.iter().cloned());
        Ok(inst)
    }

    /// Runs the instance of `vars`: inputs come from `given` by name; inputs not given that have a
    /// generator are generated (their `"step:<expr>"` args evaluated over `step`). Outputs by name, in
    /// graph output order.
    pub fn run(
        &self,
        vars: &[(&str, Val)],
        given: &[(&str, Tensor)],
        step: &[(&str, f64)],
    ) -> Result<Vec<(String, Tensor)>, String> {
        let inst = self.instance(vars)?;
        let mut ins = Vec::with_capacity(inst.inputs.len());
        for p in &inst.inputs {
            let t = match (given.iter().find(|(n, _)| *n == p.name), &p.gen) {
                (Some((_, t)), _) => Ok(t.clone()),
                (None, Some((g, args))) => step_args(args, step).and_then(|a| gen_tensor(g, &a)),
                (None, None) => Err(format!("{}: input `{}` not given", self.net, p.name)),
            }
            .map_err(|e| format!("{}: input `{}`: {e}", self.net, p.name))?;
            let want: usize = p.shape.iter().product();
            if tensor_len(&t) != want {
                return Err(format!(
                    "{}: input `{}` has {} values, shape {:?} wants {want}",
                    self.net,
                    p.name,
                    tensor_len(&t),
                    p.shape
                ));
            }
            ins.push(t);
        }
        let key = ShapeCache::key(vars);
        let mut slot = self.program.lock();
        let out = match slot.as_mut() {
            Some((k, p)) if *k == key => p.run(ins)?,
            Some((_, p)) => p.run_reshaped(&inst.graph, ins, true)?,
            None => {
                let w = self.model.store.materialize(&inst.weights)?;
                let mut p = GraphProgram::compile_reshapable(inst.graph.clone(), w, self.mode)?;
                let out = p.run(ins)?;
                *slot = Some((key, p));
                out
            }
        };
        Ok(inst.outputs.iter().cloned().zip(out).collect())
    }

    /// Output `name` (f32) of [`Self::run`].
    pub fn run_one(
        &self,
        vars: &[(&str, Val)],
        given: &[(&str, Tensor)],
        step: &[(&str, f64)],
        name: Option<&str>,
    ) -> Result<Vec<f32>, String> {
        let mut out = self.run(vars, given, step)?;
        let i = match name {
            None => 0,
            Some(n) => out
                .iter()
                .position(|(o, _)| o == n)
                .ok_or_else(|| format!("{}: no output `{n}`", self.net))?,
        };
        match out.swap_remove(i).1 {
            Tensor::F32(v) => Ok(v),
            t => Ok(t.as_f32()?.to_vec()),
        }
    }

    /// Weights every instance built so far reads.
    pub fn used(&self) -> HashSet<String> {
        self.used.lock().clone()
    }
}

/// Checkpoint tensors of `model` that none of `nets` (built at least once) reads and its spec
/// does not `ignore`, plus `extra` names read outside any graph.
pub fn unconsumed(model: &Model, nets: &[&Net], extra: &[&str]) -> Vec<String> {
    let mut used: HashSet<String> = nets.iter().flat_map(|n| n.used()).collect();
    used.extend(extra.iter().map(|s| s.to_string()));
    model.store.unconsumed(&used, &model.ignore)
}

// ---- the pipeline ------------------------------------------------------------------------------

/// A network of a spec component (see the module docs).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartRef {
    pub component: String,
    #[serde(default)]
    pub net: Option<String>,
}

/// `runtime.text_encoders[]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextEntry {
    pub name: String,
    pub component: String,
    #[serde(default)]
    pub net: Option<String>,
    #[serde(default)]
    pub optional: bool,
    /// Directory (relative to the root) holding `tokenizer.json`, or `hub:<repo>`.
    pub tokenizer: String,
    /// Directory (relative to the root) whose `special_tokens_map.json` / `tokenizer_config.json`
    /// name the pad token (`pad_token: "config"`); default `tokenizer`.
    #[serde(default)]
    pub tokenizer_config: Option<String>,
    #[serde(default = "prompt_template")]
    pub template: String,
    #[serde(default)]
    pub negative_template: Option<String>,
    #[serde(default)]
    pub prefix: String,
    #[serde(default)]
    pub lowercase: bool,
    #[serde(default)]
    pub add_special: bool,
    #[serde(default)]
    pub bos: J,
    #[serde(default)]
    pub eos: J,
    #[serde(default)]
    pub max_len: J,
    #[serde(default)]
    pub pad_to: J,
    #[serde(default)]
    pub pad_id: J,
    /// Pad token text (`"config"`: read from `tokenizer_config`), looked up in the vocabulary.
    #[serde(default)]
    pub pad_token: Option<String>,
    /// `key_mask`: `true` (default) masks the padding; `false`: every row is a key (all 0);
    /// `"valid_plus1"`: the first pad row is a key too.
    #[serde(default = "yes")]
    pub mask: J,
    #[serde(default = "keep_all")]
    pub keep: String,
    /// Rows dropped from every output; expressions over `prefix_len` (tokens of the template before
    /// `{prompt}`), `prefix_tokens` (tokens of `prefix` alone) and `valid`.
    #[serde(default)]
    pub drop: Option<Range2>,
    #[serde(default)]
    pub vars: serde_json::Map<String, J>,
    pub outputs: serde_json::Map<String, J>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Range2 {
    pub from: J,
    pub to: J,
}

fn prompt_template() -> String {
    "{prompt}".into()
}
fn yes() -> J {
    J::Bool(true)
}
fn keep_all() -> String {
    "all".into()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CtxBlock {
    pub features: Vec<String>,
    #[serde(default)]
    pub width: J,
    #[serde(default)]
    pub rows: J,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Negative {
    #[serde(default)]
    pub default: String,
    #[serde(default)]
    pub zeros_for_empty: J,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedSpec {
    pub kind: String,
    #[serde(default)]
    pub config: J,
    #[serde(default)]
    pub sigmas: Option<String>,
    #[serde(default)]
    pub shift: J,
    #[serde(default)]
    pub mu: J,
    #[serde(default)]
    pub seq: J,
    #[serde(default)]
    pub terminal: J,
    #[serde(default)]
    pub base_seq: J,
    #[serde(default)]
    pub max_seq: J,
    #[serde(default)]
    pub base_shift: J,
    #[serde(default)]
    pub max_shift: J,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuidanceSpec {
    pub kind: String,
    #[serde(default)]
    pub gate: J,
    #[serde(default)]
    pub scale: J,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentSpec {
    pub channels: J,
    pub factor: J,
    #[serde(default = "pack_none")]
    pub pack: String,
    #[serde(default)]
    pub noise_scale: J,
}

fn pack_none() -> String {
    "none".into()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Affine {
    #[serde(default)]
    pub mul: J,
    #[serde(default)]
    pub add: J,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BnDenorm {
    pub mean: String,
    pub var: String,
    #[serde(default)]
    pub eps: J,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VaeSpec {
    pub component: String,
    #[serde(default)]
    pub decode: Option<String>,
    #[serde(default)]
    pub encode: Option<String>,
    #[serde(default)]
    pub decode_affine: Option<Affine>,
    #[serde(default)]
    pub bn_denorm: Option<BnDenorm>,
    #[serde(default = "clamp_pm1")]
    pub output: String,
}

fn clamp_pm1() -> String {
    "clamp_pm1".into()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DefaultsSpec {
    pub width: J,
    pub height: J,
    pub steps: J,
    pub guidance: J,
    pub multiple: J,
    #[serde(default)]
    pub max_side: J,
    #[serde(default)]
    pub max_steps: J,
}

/// The `runtime` keys of `pipeline: diffusion`.
#[derive(Debug, Clone, Deserialize)]
pub struct Runtime {
    #[serde(default)]
    pub text_encoders: Vec<TextEntry>,
    #[serde(default)]
    pub context: Vec<CtxBlock>,
    #[serde(default)]
    pub pooled: Vec<String>,
    #[serde(default)]
    pub negative: Negative,
    #[serde(default)]
    pub context_net: Option<PartRef>,
    pub denoiser: PartRef,
    pub scheduler: SchedSpec,
    #[serde(default = "timestep_t")]
    pub timestep: String,
    pub guidance: GuidanceSpec,
    #[serde(default)]
    pub negate_v: bool,
    pub latent: LatentSpec,
    pub vae: VaeSpec,
    pub defaults: DefaultsSpec,
    #[serde(default)]
    pub summary: Option<String>,
    /// Single-file checkpoints (`match.single_file_tensors`): the diffusers repo (expression)
    /// whose configs, scheduler and tokenizer files the file's components load under.
    #[serde(default)]
    pub reference: Option<J>,
    /// Keys other pipelines read (e.g. the video pipeline's).
    #[serde(flatten)]
    pub other: serde_json::Map<String, J>,
}

fn timestep_t() -> String {
    "t".into()
}

/// Runtime expression scope: request vars, then `<part>.<x>` in that part's model env, then the
/// root model env.
struct REnv<'a> {
    root: &'a Model,
    parts: &'a [(String, Arc<Model>)],
    vars: &'a [(&'a str, Val)],
}

impl Env for REnv<'_> {
    fn lookup(&self, root: &str, path: &[Val]) -> Option<Val> {
        if let Some((_, v)) = self.vars.iter().rev().find(|(k, _)| *k == root) {
            return walk(v.clone(), path);
        }
        if let Some((_, m)) = self.parts.iter().find(|(n, _)| n == root) {
            let (first, rest) = path.split_first()?;
            return m.env.lookup(first.str().ok()?, rest);
        }
        self.root.env.lookup(root, path)
    }
}

/// How a prompt becomes token ids.
struct TextEnc {
    e: TextEntry,
    net: Net,
    tok: crate::model::tokenizer::Tokenizer,
    pad: u32,
}

/// One encoded prompt: every encoder output (`<name>.<key>`) as rows x width.
type Encoded = HashMap<String, (Vec<f32>, usize)>;

enum Sampler {
    Discrete(SchedulerConfig),
    Flow {
        schedule: Schedule,
        base: String,
        card: Option<Vec<f64>>,
        dpm: bool,
        seq: J,
    },
}

/// What a pipeline loads beyond its own snapshot.
#[derive(Debug, Clone, Default)]
pub struct Extras {
    /// LoRA files and their strengths.
    pub loras: Vec<(std::path::PathBuf, f32)>,
    /// (component, weights file) replacing the snapshot's weights of that component (a derived
    /// finetune / quantization, e.g. a GGUF from another repo); config stays the snapshot's.
    pub weights: Vec<(String, std::path::PathBuf)>,
    /// A derived repo's own `scheduler/scheduler_config.json`, replacing the snapshot's.
    pub scheduler: Option<J>,
    /// A derived repo's model card, read by `scheduler.sigmas: "card"` instead of the snapshot's.
    pub card: Option<String>,
}

/// A text-to-image (and image-to-image) pipeline run from its family spec.
pub struct SpecDiffusion {
    id: String,
    rt: Runtime,
    root: Model,
    parts: Vec<(String, Arc<Model>)>,
    text: Vec<Option<TextEnc>>,
    context_net: Option<Net>,
    denoiser: Net,
    decoder: Net,
    encoder: Option<Net>,
    sampler: Sampler,
    channels: usize,
    factor: usize,
    /// Per-channel decode affine `z * mul + add`.
    affine: (Vec<f32>, Vec<f32>),
    /// Packed-feature denorm `x * std + mean`.
    bn: Option<(Vec<f32>, Vec<f32>)>,
    defaults: ImageDefaults,
    max_side: usize,
    max_steps: usize,
    summary: String,
    /// `runtime.video` (video pipelines).
    video: Option<VideoSpec>,
}

fn num(v: Val, what: &str) -> Result<f64, String> {
    v.num().map_err(|e| format!("{what}: {e}"))
}

/// Pixels `[c, h, w]` in [-1, 1] -> RGB (RGBA composed over white).
fn to_image(x: &[f32], c: usize, h: usize, w: usize, over_white: bool) -> ImageBuffer {
    let mut img = ImageBuffer::new(w, h);
    let hw = h * w;
    let px = |v: f32| (((v + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).round() as u8;
    for y in 0..h {
        for xi in 0..w {
            let i = y * w + xi;
            let (r, g, b) = (x[i], x[hw + i], x[2 * hw + i]);
            let (r, g, b) = if over_white && c == 4 {
                let a = (x[3 * hw + i].clamp(-1.0, 1.0) / 2.0 + 0.5) as f32;
                let comp = |v: f32| {
                    let p = v.clamp(-1.0, 1.0) / 2.0 + 0.5;
                    ((a * p + (1.0 - a)) * 255.0).round() as u8
                };
                (comp(r), comp(g), comp(b))
            } else {
                (px(r), px(g), px(b))
            };
            img.set_pixel(xi, y, r, g, b);
        }
    }
    img
}

/// RGB image -> `[3, h, w]` in [-1, 1].
fn from_image(img: &ImageBuffer) -> Vec<f32> {
    let (h, w) = (img.height, img.width);
    let hw = h * w;
    let mut x = vec![0.0; 3 * hw];
    for y in 0..h {
        for xi in 0..w {
            let (r, g, b) = img.get_pixel(xi, y);
            let i = y * w + xi;
            x[i] = r as f32 / 127.5 - 1.0;
            x[hw + i] = g as f32 / 127.5 - 1.0;
            x[2 * hw + i] = b as f32 / 127.5 - 1.0;
        }
    }
    x
}

/// `[c, h, w]` -> 2x2 patch tokens `[(h/2)*(w/2), 4c]`, features `(c, dy, dx)`.
pub fn pack2x2(x: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let wp = w / 2;
    let mut out = vec![0.0; x.len()];
    for ch in 0..c {
        for y in 0..h {
            for xi in 0..w {
                out[((y / 2) * wp + xi / 2) * 4 * c + ch * 4 + (y % 2) * 2 + xi % 2] =
                    x[(ch * h + y) * w + xi];
            }
        }
    }
    out
}

/// Inverse of [`pack2x2`].
pub fn unpack2x2(p: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let wp = w / 2;
    let mut out = vec![0.0; p.len()];
    for ch in 0..c {
        for y in 0..h {
            for xi in 0..w {
                out[(ch * h + y) * w + xi] =
                    p[((y / 2) * wp + xi / 2) * 4 * c + ch * 4 + (y % 2) * 2 + xi % 2];
            }
        }
    }
    out
}

/// `[rows, cols]` transposed.
fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = x[r * cols + c];
        }
    }
    out
}

/// Token ids of `text` (`add_special`: the tokenizer.json post-processor's specials).
fn tokenize(
    tok: &crate::model::tokenizer::Tokenizer,
    text: &str,
    add_special: bool,
) -> Result<Vec<u32>, String> {
    match (&tok.hf, add_special) {
        (Some(hf), true) => Ok(hf
            .encode(text, true)
            .map_err(|e| format!("tokenizer: {e}"))?
            .get_ids()
            .to_vec()),
        _ => Ok(tok.encode(text)),
    }
}

impl SpecDiffusion {
    /// Loads `artifacts` (a diffusers pipeline directory) as `family` (detected when `None`).
    pub fn load(
        artifacts: &ModelArtifacts,
        hub: &HuggingFaceHub,
        mode: PlacementMode,
        family: Option<&str>,
        extra: &Extras,
    ) -> Result<Self, String> {
        Self::load_in(
            crate::arch::registry()?,
            artifacts,
            hub,
            mode,
            family,
            extra,
        )
    }

    /// [`Self::load`] from the specs of `reg`.
    pub fn load_in(
        reg: &crate::arch::Registry,
        artifacts: &ModelArtifacts,
        hub: &HuggingFaceHub,
        mode: PlacementMode,
        family: Option<&str>,
        extra: &Extras,
    ) -> Result<Self, String> {
        let loras = &extra.loras;
        let lora_files = loras
            .iter()
            .map(|(p, s)| Ok((crate::model::lora::LoraFile::open(p)?, *s)))
            .collect::<Result<Vec<_>, String>>()?;
        let mut lora_applied = 0usize;
        // The root is the pipeline directory: its components carry the weights.
        let dir = artifacts
            .local_dir
            .clone()
            .ok_or("diffusion pipeline: no local snapshot directory")?;
        let root_art = ModelArtifacts {
            model_id: artifacts.model_id.clone(),
            local_dir: Some(dir.clone()),
            is_cached: true,
            ..Default::default()
        };
        // The spec's `checks` run on the root and on every component load (a family may
        // reference itself for its networks): checks reading tensors or `cfg` guard on them.
        let reencode = ReencodeConfig {
            policy: ReencodePolicy::BitExact,
            force: ReencodeConfig::effective().force,
            ..Default::default()
        };
        let root = crate::arch::load::load_with(&root_art, reg, family, reencode.clone())?;
        crate::arch::load::check(&root)?;
        let fam = root.spec.family.clone();
        let rt: Runtime = root.spec.runtime_as()?;
        let err = |e: String| format!("{fam}: {e}");

        // Every component once (several nets may share one).
        let mut parts: Vec<(String, Arc<Model>)> = Vec::new();
        let mut by_comp: HashMap<String, Arc<Model>> = HashMap::new();
        let mut comp = |name: &str, part: Option<LoraPart>| -> Result<Arc<Model>, String> {
            if let Some(m) = by_comp.get(name) {
                return Ok(m.clone());
            }
            let mut m = match extra
                .weights
                .iter()
                .find(|(c, _)| c == name)
                .map(|(_, w)| w)
            {
                // Substituted weights under the base's component config.
                Some(w) => {
                    let crate::arch::Component::Ref { dir: Some(d), .. } =
                        root.spec.component(name)?
                    else {
                        return Err(format!(
                            "component `{name}` is not a {{family, dir}} reference"
                        ));
                    };
                    let mut a = dir_artifacts(&dir.join(&d));
                    a.weights_path = Some(w.clone());
                    a.model_id = w.display().to_string();
                    let comp_spec = format!("{}.{name}", root.spec.family);
                    crate::arch::load::load_with(&a, reg, Some(&comp_spec), reencode.clone())?
                }
                _ => root.component(reg, name)?,
            };
            crate::arch::load::check(&m)?;
            if let Some(part) = part {
                for (f, strength) in &lora_files {
                    lora_applied += f.apply(part, &mut m, reg, *strength)?;
                }
            }
            let m = Arc::new(m);
            by_comp.insert(name.to_string(), m.clone());
            Ok(m)
        };
        let comp_dir_exists = |name: &str| -> bool {
            match root.spec.component(name) {
                Ok(crate::arch::Component::Ref { dir: Some(d), .. }) => dir.join(d).is_dir(),
                _ => true,
            }
        };

        let mut text = Vec::new();
        for (ti, e) in rt.text_encoders.iter().enumerate() {
            if e.optional && !comp_dir_exists(&e.component) {
                text.push(None);
                continue;
            }
            let m = comp(&e.component, Some(LoraPart::TextEncoder(ti))).map_err(err)?;
            parts.push((e.name.clone(), m.clone()));
            let tok = match e.tokenizer.strip_prefix("hub:") {
                Some(repo) => {
                    let p = hub.fetch_file(repo, "tokenizer.json")?;
                    crate::model::hf_loader::load_json_tokenizer(&p)?
                }
                None => {
                    let d = dir.join(&e.tokenizer);
                    let j = d.join("tokenizer.json");
                    if j.is_file() {
                        crate::model::hf_loader::load_json_tokenizer(&j)?
                    } else if d.join("vocab.json").is_file() && d.join("merges.txt").is_file() {
                        // A byte-level BPE folder without tokenizer.json.
                        let built = crate::model::tokenizer::byte_bpe_from_vocab_merges(&d)?;
                        let json = built
                            .to_string(false)
                            .map_err(|e| format!("{}: {e}", d.display()))?;
                        crate::model::hf_loader::tokenizer_from_files(&json, &|n| {
                            std::fs::read(d.join(n)).ok()
                        })?
                    } else {
                        crate::model::spm::load(&d.join("spiece.model"))?
                    }
                }
            };
            let pad = match e.pad_token.as_deref() {
                None => 0,
                Some(t) => {
                    let text = if t == "config" {
                        let d = dir.join(e.tokenizer_config.as_deref().unwrap_or(&e.tokenizer));
                        ["special_tokens_map.json", "tokenizer_config.json"]
                            .iter()
                            .filter_map(|f| {
                                let s = std::fs::read_to_string(d.join(f)).ok()?;
                                serde_json::from_str::<J>(&s).ok()
                            })
                            .find_map(|j| {
                                let p = &j["pad_token"];
                                p.as_str()
                                    .or_else(|| p["content"].as_str())
                                    .map(str::to_string)
                            })
                            .ok_or_else(|| err(format!("{}: no pad_token", d.display())))?
                    } else {
                        t.to_string()
                    };
                    *tok.vocab
                        .get(&text)
                        .or_else(|| tok.special_tokens.get(&text))
                        .ok_or_else(|| {
                            err(format!(
                                "{}: pad token {text:?} is not in the vocabulary",
                                e.name
                            ))
                        })?
                }
            };
            let net = Net::new(m, e.net.as_deref().unwrap_or("text"), mode);
            text.push(Some(TextEnc {
                e: e.clone(),
                net,
                tok,
                pad,
            }));
        }
        let den_m = comp(&rt.denoiser.component, Some(LoraPart::Denoiser)).map_err(err)?;
        parts.push(("denoiser".into(), den_m.clone()));
        let denoiser = Net::new(
            den_m,
            rt.denoiser.net.as_deref().unwrap_or("denoiser"),
            mode,
        );
        let context_net = match &rt.context_net {
            Some(p) => Some(Net::new(
                comp(&p.component, None).map_err(err)?,
                p.net.as_deref().unwrap_or("context"),
                mode,
            )),
            None => None,
        };
        let vae_m = comp(&rt.vae.component, None).map_err(err)?;
        drop(comp);
        let lora_total: usize = lora_files
            .iter()
            .map(|(f, _)| {
                (0..rt.text_encoders.len())
                    .map(|i| f.targets(LoraPart::TextEncoder(i)))
                    .sum::<usize>()
                    + f.targets(LoraPart::Denoiser)
            })
            .sum();
        let lora_all: usize = lora_files.iter().map(|(f, _)| f.modules()).sum();
        if lora_applied != lora_total || lora_total != lora_all {
            return Err(err(format!(
                "LoRA patches {lora_all} modules, {lora_applied} of them on networks this pipeline has"
            )));
        }
        parts.push(("vae".into(), vae_m.clone()));
        let decoder = Net::new(
            vae_m.clone(),
            rt.vae.decode.as_deref().unwrap_or("decoder"),
            mode,
        );
        let encoder = rt
            .vae
            .encode
            .as_deref()
            .map(|n| Net::new(vae_m.clone(), n, mode));

        let ev = |j: &J, what: &str| -> Result<Val, String> {
            let env = REnv {
                root: &root,
                parts: &parts,
                vars: &[],
            };
            eval_json(j, &env).map_err(|e| err(format!("runtime.{what}: {e}")))
        };
        let evn = |j: &J, what: &str| -> Result<f64, String> { num(ev(j, what)?, what) };
        let channels = evn(&rt.latent.channels, "latent.channels")? as usize;
        let factor = evn(&rt.latent.factor, "latent.factor")? as usize;
        let per_channel = |j: &J, d: f64, what: &str| -> Result<Vec<f32>, String> {
            if j.is_null() {
                return Ok(vec![d as f32; channels]);
            }
            Ok(match ev(j, what)? {
                Val::List(l) => l
                    .iter()
                    .map(|v| v.num().map(|x| x as f32))
                    .collect::<Result<_, _>>()?,
                v => vec![num(v, what)? as f32; channels],
            })
        };
        let affine = match &rt.vae.decode_affine {
            Some(a) => (
                per_channel(&a.mul, 1.0, "vae.decode_affine.mul")?,
                per_channel(&a.add, 0.0, "vae.decode_affine.add")?,
            ),
            None => (vec![1.0; channels], vec![0.0; channels]),
        };
        let mut extra_used: Vec<String> = Vec::new();
        let bn = match &rt.vae.bn_denorm {
            Some(b) => {
                let eps = if b.eps.is_null() {
                    0.0
                } else {
                    evn(&b.eps, "vae.bn_denorm.eps")?
                };
                let get = |n: &str| -> Result<Vec<f32>, String> {
                    let w = vae_m.store.materialize(&[n.to_string()])?;
                    Ok(w.get(n)
                        .ok_or_else(|| format!("vae tensor `{n}` missing"))?
                        .as_f32()?
                        .to_vec())
                };
                let mean = get(&b.mean)?;
                let std = get(&b.var)?
                    .iter()
                    .map(|v| ((*v as f64 + eps).sqrt()) as f32)
                    .collect();
                extra_used.extend([b.mean.clone(), b.var.clone()]);
                Some((mean, std))
            }
            None => None,
        };
        let d = &rt.defaults;
        let defaults = ImageDefaults {
            width: evn(&d.width, "defaults.width")? as usize,
            height: evn(&d.height, "defaults.height")? as usize,
            steps: evn(&d.steps, "defaults.steps")? as usize,
            guidance: evn(&d.guidance, "defaults.guidance")? as f32,
            multiple: evn(&d.multiple, "defaults.multiple")? as usize,
        };
        let max_side = if d.max_side.is_null() {
            2048
        } else {
            evn(&d.max_side, "defaults.max_side")? as usize
        };
        let max_steps = if d.max_steps.is_null() {
            100
        } else {
            evn(&d.max_steps, "defaults.max_steps")? as usize
        };

        let s = &rt.scheduler;
        let sched_cfg = if let Some(c) = &extra.scheduler {
            c.clone()
        } else if s.config.is_null() {
            ev(
                &J::String("json.scheduler_config".into()),
                "scheduler.config",
            )
            .ok()
            .filter(|v| !v.is_null())
            .map(|v| v.to_json())
            .or_else(|| {
                let p = dir.join("scheduler/scheduler_config.json");
                serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()
            })
            .unwrap_or(J::Null)
        } else {
            ev(&s.config, "scheduler.config")?.to_json()
        };
        let sampler = match s.kind.as_str() {
            "discrete" => Sampler::Discrete(SchedulerConfig::from_json(&sched_cfg).map_err(err)?),
            k @ ("flow_euler" | "flow_dpm2") => {
                let mut sc = Schedule::from_config(&sched_cfg);
                let set = |j: &J, slot: &mut f64, what: &str| -> Result<(), String> {
                    if !j.is_null() {
                        *slot = evn(j, what)?;
                    }
                    Ok(())
                };
                set(&s.shift, &mut sc.shift, "scheduler.shift")?;
                set(&s.base_seq, &mut sc.base_seq, "scheduler.base_seq")?;
                set(&s.max_seq, &mut sc.max_seq, "scheduler.max_seq")?;
                set(&s.base_shift, &mut sc.base_shift, "scheduler.base_shift")?;
                set(&s.max_shift, &mut sc.max_shift, "scheduler.max_shift")?;
                if !s.terminal.is_null() {
                    sc.terminal = match ev(&s.terminal, "scheduler.terminal")? {
                        Val::Null => None,
                        v => Some(num(v, "scheduler.terminal")?).filter(|t| *t > 0.0),
                    };
                }
                if !s.mu.is_null() {
                    sc.mu = match ev(&s.mu, "scheduler.mu")? {
                        Val::Str(k) if k == "static" => Mu::Static,
                        Val::Str(k) if k == "dynamic" => Mu::Dynamic,
                        Val::Str(k) if k == "empirical" => Mu::Empirical,
                        Val::Str(k) if k == "config" => sc.mu,
                        v => Mu::Fixed(num(v, "scheduler.mu")?),
                    };
                }
                let base = s.sigmas.clone().unwrap_or_else(|| "linspace".into());
                let card = (base == "card")
                    .then(|| {
                        extra
                            .card
                            .clone()
                            .or_else(|| std::fs::read_to_string(dir.join("README.md")).ok())
                    })
                    .flatten()
                    .and_then(|c| card_sigmas(&c));
                Sampler::Flow {
                    schedule: sc,
                    base,
                    card,
                    dpm: k == "flow_dpm2",
                    seq: s.seq.clone(),
                }
            }
            k => {
                return Err(err(format!(
                    "runtime.scheduler.kind `{k}` (discrete, flow_euler, flow_dpm2)"
                )))
            }
        };
        let video = match rt.other.get("video") {
            Some(v) => Some(
                serde_json::from_value::<VideoSpec>(v.clone())
                    .map_err(|e| err(format!("runtime.video: {e}")))?,
            ),
            None => None,
        };
        let summary = match &rt.summary {
            Some(s) => ev(&J::String(s.clone()), "summary")?.to_string(),
            None => format!("{fam} ({})", artifacts.model_id),
        };
        let p = Self {
            id: artifacts.model_id.clone(),
            rt,
            root,
            parts,
            text,
            context_net,
            denoiser,
            decoder,
            encoder,
            sampler,
            channels,
            factor,
            affine,
            bn,
            defaults,
            max_side,
            max_steps,
            summary,
            video,
        };
        p.check_consumed(&extra_used)?;
        Ok(p)
    }

    /// Builds every network once at the default shape and refuses checkpoint tensors none reads.
    fn check_consumed(&self, extra: &[String]) -> Result<(), String> {
        let d = self.defaults;
        let (h, w) = (d.height / self.factor, d.width / self.factor);
        for t in self.text.iter().flatten() {
            let rows = self.eval_n(&t.e.pad_to, &[]).unwrap_or(8.0).max(1.0) as usize;
            t.net.instance(&self.text_vars(t, rows)?)?;
        }
        let nt = 8;
        // Video pipelines build at the default clip length (no clean tokens).
        let mut vid: Vec<(&str, Val)> = Vec::new();
        if let Some(v) = &self.video {
            let frames = self.eval_n(&v.frames, &[])?;
            let lf = self.eval_n(&v.latent_frames, &[("frames", Val::Num(frames))])?;
            vid = vec![("frames", Val::Num(lf)), ("clean", Val::Num(0.0))];
        }
        let mut den = self.den_vars(h, w, nt);
        den.extend(vid.iter().cloned());
        self.denoiser.instance(&den)?;
        if let Some(c) = &self.context_net {
            c.instance(&[("nt", Val::Num(nt as f64))])?;
        }
        let hw = [("h", Val::Num(h as f64)), ("w", Val::Num(w as f64))];
        match &self.video {
            Some(v) if v.decode == "clip" => {
                self.decoder
                    .instance(&[hw[0].clone(), hw[1].clone(), vid[0].clone()])?
            }
            _ => self.decoder.instance(&hw)?,
        };
        if let Some(e) = &self.encoder {
            e.instance(&hw)?;
        }
        let mut nets: Vec<&Net> = self.text.iter().flatten().map(|t| &t.net).collect();
        nets.push(&self.denoiser);
        nets.extend(self.context_net.as_ref());
        nets.push(&self.decoder);
        nets.extend(self.encoder.as_ref());
        let extra: Vec<&str> = extra.iter().map(String::as_str).collect();
        let mut seen: Vec<*const Model> = Vec::new();
        for n in &nets {
            let ptr = Arc::as_ptr(&n.model);
            if seen.contains(&ptr) {
                continue;
            }
            seen.push(ptr);
            let same: Vec<&Net> = nets
                .iter()
                .copied()
                .filter(|m| Arc::ptr_eq(&m.model, &n.model))
                .collect();
            let left = unconsumed(&n.model, &same, &extra);
            if !left.is_empty() {
                return Err(format!(
                    "{}: checkpoint tensors no instance reads (list deliberate ones in `ignore`): {}",
                    n.model.spec.family,
                    left.join(", ")
                ));
            }
        }
        Ok(())
    }

    fn env<'a>(&'a self, vars: &'a [(&'a str, Val)]) -> REnv<'a> {
        REnv {
            root: &self.root,
            parts: &self.parts,
            vars,
        }
    }

    fn eval(&self, j: &J, vars: &[(&str, Val)]) -> Result<Val, String> {
        eval_json(j, &self.env(vars))
    }

    fn eval_n(&self, j: &J, vars: &[(&str, Val)]) -> Result<f64, String> {
        if j.is_null() {
            return Err("unset".into());
        }
        self.eval(j, vars)?.num()
    }

    /// Instance vars of a text encoder: `rows`, then the entry's `vars` (expressions over the
    /// runtime scope, e.g. `te.layers` of the encoder's own config).
    fn text_vars<'a>(&self, t: &'a TextEnc, rows: usize) -> Result<Vec<(&'a str, Val)>, String> {
        let mut v = vec![("rows", Val::Num(rows as f64))];
        for (k, j) in &t.e.vars {
            v.push((
                k.as_str(),
                self.eval(j, &[])
                    .map_err(|e| format!("{}: vars.{k}: {e}", t.e.name))?,
            ));
        }
        Ok(v)
    }

    fn den_vars(&self, h: usize, w: usize, nt: usize) -> Vec<(&'static str, Val)> {
        vec![
            ("h", Val::Num(h as f64)),
            ("w", Val::Num(w as f64)),
            ("nt", Val::Num(nt as f64)),
        ]
    }

    /// The encoders' outputs for `prompt`.
    fn encode(&self, prompt: &str, negative: bool) -> Result<Encoded, String> {
        let mut out = Encoded::new();
        for t in self.text.iter().flatten() {
            let e = &t.e;
            let p = if e.lowercase {
                prompt.to_lowercase().trim().to_string()
            } else {
                prompt.to_string()
            };
            let tpl = match (negative, &e.negative_template) {
                (true, Some(n)) => n.as_str(),
                _ => e.template.as_str(),
            };
            let (text, prefix_text) = if tpl == "chat" {
                let msg: crate::model::hf_loader::ChatMessage =
                    serde_json::from_value(serde_json::json!({"role": "user", "content": p}))
                        .map_err(|e| e.to_string())?;
                let text = t
                    .tok
                    .chat_template
                    .as_ref()
                    .ok_or_else(|| format!("{}: the tokenizer has no chat template", e.name))?
                    .render(
                        &[msg],
                        None,
                        true,
                        &serde_json::json!({"enable_thinking": false}),
                    )
                    .map_err(|err| format!("{}: chat template: {err}", e.name))?;
                (text, String::new())
            } else {
                let (before, after) = tpl
                    .split_once("{prompt}")
                    .ok_or_else(|| format!("{}: template has no {{prompt}}", e.name))?;
                let before = before.replace("{prefix}", &e.prefix);
                (
                    format!("{before}{p}{}", after.replace("{prefix}", &e.prefix)),
                    before,
                )
            };
            let prefix_len = if tpl.contains("{prefix}") && !prefix_text.is_empty() {
                tokenize(&t.tok, &prefix_text, e.add_special)?.len()
            } else {
                0
            };
            // Tokens of the `prefix` string alone (a template may put more before `{prompt}`).
            let prefix_tokens = if e.prefix.is_empty() {
                0
            } else {
                tokenize(&t.tok, &e.prefix, e.add_special)?.len()
            };
            let pv = [("prefix_len", Val::Num(prefix_len as f64))];
            let opt_id = |j: &J| -> Result<Option<u32>, String> {
                if j.is_null() {
                    return Ok(None);
                }
                Ok(Some(self.eval_n(j, &pv)? as u32))
            };
            let mut ids = tokenize(&t.tok, &text, e.add_special)?;
            if let Some(b) = opt_id(&e.bos)? {
                if ids.first() != Some(&b) {
                    ids.insert(0, b);
                }
            }
            let eos = opt_id(&e.eos)?;
            if let Some(m) = opt_id(&e.max_len)? {
                let m = m as usize;
                match eos {
                    Some(eos) => {
                        if ids.last() == Some(&eos) {
                            ids.pop();
                        }
                        ids.truncate(m.saturating_sub(1));
                        ids.push(eos);
                    }
                    None => ids.truncate(m),
                }
            } else if let Some(eos) = eos {
                if ids.last() != Some(&eos) {
                    ids.push(eos);
                }
            }
            let valid = ids.len();
            if let Some(n) = opt_id(&e.pad_to)? {
                let pad = match opt_id(&e.pad_id)? {
                    Some(p) => p,
                    None => t.pad,
                };
                ids.resize((n as usize).max(valid), pad);
            }
            let rows = ids.len();
            let key_mask: Vec<f32> = {
                let keys = match &e.mask {
                    J::Bool(true) => valid,
                    J::Bool(false) => rows,
                    J::String(k) if k == "valid_plus1" => valid + 1,
                    m => {
                        return Err(format!(
                            "{}: mask {m} (true, false, \"valid_plus1\")",
                            e.name
                        ))
                    }
                };
                (0..rows)
                    .map(|i| if i < keys { 0.0 } else { f32::NEG_INFINITY })
                    .collect()
            };
            let given = [
                ("token_ids", Tensor::U32(ids.clone())),
                ("positions", Tensor::U32((0..rows as u32).collect())),
                ("key_mask", Tensor::F32(key_mask)),
                (
                    "last_index",
                    Tensor::U32(vec![valid.saturating_sub(1) as u32]),
                ),
            ];
            let res = t.net.run(&self.text_vars(t, rows)?, &given, &[])?;
            let keep = match e.keep.as_str() {
                "all" => rows,
                "valid" => valid,
                "valid_plus1" => (valid + 1).min(rows),
                k => return Err(format!("{}: keep `{k}` (all, valid, valid_plus1)", e.name)),
            };
            let dv = [
                ("prefix_len", Val::Num(prefix_len as f64)),
                ("prefix_tokens", Val::Num(prefix_tokens as f64)),
                ("valid", Val::Num(valid as f64)),
            ];
            let drop = match &e.drop {
                Some(r) => {
                    let (a, b) = (
                        self.eval_n(&r.from, &dv)? as usize,
                        self.eval_n(&r.to, &dv)? as usize,
                    );
                    a..b.max(a)
                }
                None => 0..0,
            };
            for (key, oname) in &e.outputs {
                let oname = interp(
                    oname.as_str().ok_or("text encoder outputs are names")?,
                    &self.env(&[]),
                )?;
                let oname = oname.as_str();
                let v = res
                    .iter()
                    .find(|(n, _)| n == oname)
                    .ok_or_else(|| format!("{}: no output `{oname}`", e.name))?
                    .1
                    .as_f32()?;
                let full = key != "pooled" && v.len() % rows == 0 && v.len() >= rows;
                let (v, n) = if full {
                    let wdt = v.len() / rows;
                    let mut sel = Vec::with_capacity(keep * wdt);
                    for r in (0..keep).filter(|r| !drop.contains(r)) {
                        sel.extend_from_slice(&v[r * wdt..(r + 1) * wdt]);
                    }
                    let n = sel.len() / wdt.max(1);
                    (sel, n)
                } else {
                    (v.to_vec(), 1)
                };
                out.insert(format!("{}.{key}", e.name), (v, n));
            }
        }
        Ok(out)
    }

    /// (context `[nt, width]`, nt, pooled) of an encoded prompt (`zeros`: the same shapes, zero).
    fn assemble(&self, enc: &Encoded, zeros: bool) -> Result<(Vec<f32>, usize, Vec<f32>), String> {
        let mut ctx = Vec::new();
        let (mut nt, mut width) = (0usize, None::<usize>);
        for b in &self.rt.context {
            let parts: Vec<&(Vec<f32>, usize)> =
                b.features.iter().filter_map(|f| enc.get(f)).collect();
            let want_w = if b.width.is_null() {
                None
            } else {
                Some(self.eval_n(&b.width, &[])? as usize)
            };
            let rows = match parts.first() {
                Some(p) => p.1,
                None if !b.rows.is_null() => self.eval_n(&b.rows, &[])? as usize,
                None => continue,
            };
            if parts.iter().any(|p| p.1 != rows) {
                return Err(format!(
                    "context block {:?}: encoders disagree on rows",
                    b.features
                ));
            }
            let natural: usize = parts.iter().map(|p| p.0.len() / rows.max(1)).sum();
            let bw = want_w.unwrap_or(natural).max(natural);
            match width {
                Some(w0) if w0 != bw => {
                    return Err(format!(
                        "context block {:?} is {bw} wide, earlier blocks {w0}",
                        b.features
                    ))
                }
                _ => width = Some(bw),
            }
            for r in 0..rows {
                let start = ctx.len();
                for p in &parts {
                    let pw = p.0.len() / rows;
                    ctx.extend_from_slice(&p.0[r * pw..(r + 1) * pw]);
                }
                ctx.resize(start + bw, 0.0);
            }
            nt += rows;
        }
        let mut pooled = Vec::new();
        for p in &self.rt.pooled {
            pooled.extend_from_slice(
                &enc.get(p)
                    .ok_or_else(|| format!("pooled `{p}`: no such encoder output"))?
                    .0,
            );
        }
        if zeros {
            ctx.iter_mut().for_each(|v| *v = 0.0);
            pooled.iter_mut().for_each(|v| *v = 0.0);
        }
        if let Some(cn) = &self.context_net {
            let out = cn.run_one(
                &[("nt", Val::Num(nt as f64))],
                &[("context", Tensor::F32(ctx))],
                &[],
                None,
            )?;
            let rows = nt;
            ctx = out;
            nt = rows;
        }
        Ok((ctx, nt, pooled))
    }

    fn conditioning(
        &self,
        prompt: &str,
        negative: bool,
    ) -> Result<(Vec<f32>, usize, Vec<f32>), String> {
        let zeros = negative
            && prompt.is_empty()
            && !self.rt.negative.zeros_for_empty.is_null()
            && self.eval(&self.rt.negative.zeros_for_empty, &[])?.truthy();
        let enc = self.encode(prompt, negative)?;
        self.assemble(&enc, zeros)
    }

    /// One generation; `init` = (image, strength) for image-to-image.
    pub fn generate(
        &self,
        req: &ImageRequest,
        init: Option<(&ImageBuffer, f32)>,
    ) -> Result<ImageBuffer, String> {
        let d = self.defaults;
        let (mut width, mut height) =
            (req.width.unwrap_or(d.width), req.height.unwrap_or(d.height));
        if let (Some((img, _)), None, None) = (init, req.width, req.height) {
            // The input's own size, scaled down to the default long side when it is over twice that.
            let long = img.width.max(img.height) as f32;
            let fit = d.width.max(d.height) as f32;
            let s = if long > 2.0 * fit { fit / long } else { 1.0 };
            let side = |v: usize| ((v as f32 * s) as usize) / d.multiple * d.multiple;
            (width, height) = (side(img.width), side(img.height));
        }
        if width == 0
            || height == 0
            || width % d.multiple != 0
            || height % d.multiple != 0
            || width.max(height) > self.max_side
        {
            return Err(format!(
                "width/height must be non-zero multiples of {} up to {}, got {width}x{height}",
                d.multiple, self.max_side
            ));
        }
        let steps = req.steps.unwrap_or(match &self.sampler {
            Sampler::Flow { card: Some(c), .. } => c.len(),
            _ => d.steps,
        });
        if !(1..=self.max_steps).contains(&steps) {
            return Err(format!(
                "steps must be in 1..={}, got {steps}",
                self.max_steps
            ));
        }
        let guidance = req.guidance.unwrap_or(d.guidance);
        let gv = [("guidance", Val::Num(guidance as f64))];
        let g = &self.rt.guidance;
        let cfg = g.kind != "none" && (g.gate.is_null() || self.eval(&g.gate, &gv)?.truthy());
        let scale = if g.scale.is_null() {
            guidance
        } else {
            self.eval_n(&g.scale, &gv)? as f32
        };
        let (c, f) = (self.channels, self.factor);
        let (h, w) = (height / f, width / f);
        let pack = self.rt.latent.pack.as_str();

        let t0 = std::time::Instant::now();
        let (ctx, nt, pooled) = self.conditioning(&req.prompt, false)?;
        let neg = if cfg {
            let n = req
                .negative_prompt
                .clone()
                .unwrap_or_else(|| self.rt.negative.default.clone());
            Some(self.conditioning(&n, true)?)
        } else {
            None
        };
        let t1 = std::time::Instant::now();

        // Schedule.
        let hw = [("h", Val::Num(h as f64)), ("w", Val::Num(w as f64))];
        let default_seq = match pack {
            "pack2x2" => (h / 2) * (w / 2),
            _ => h * w,
        };
        let mut run = self.schedule(steps, &hw, default_seq)?;
        let n_steps = run.steps();

        // Latents in the denoiser's layout.
        let to_layout = |z: Vec<f32>| -> Vec<f32> {
            match pack {
                "pack2x2" => pack2x2(&z, c, h, w),
                "tokens" => transpose(&z, c, h * w),
                _ => z,
            }
        };
        let from_layout = |x: &[f32]| -> Vec<f32> {
            match pack {
                "pack2x2" => unpack2x2(x, c, h, w),
                "tokens" => transpose(x, h * w, c),
                _ => x.to_vec(),
            }
        };
        let mut rng = Rng::new(req.seed);
        let (mut x, start) = match init {
            None => {
                let s = match (&self.rt.latent.noise_scale, &run) {
                    (J::String(k), Run::Discrete(s)) if k == "init_sigma" => s.init_sigma() as f32,
                    (J::Null, _) => 1.0,
                    (J::String(k), _) if k == "init_sigma" => 1.0,
                    (j, _) => self.eval_n(j, &[])? as f32,
                };
                let noise: Vec<f32> = (0..c * h * w).map(|_| rng.gaussian() * s).collect();
                (to_layout(noise), 0)
            }
            Some((img, strength)) => {
                let Run::Discrete(sched) = &mut run else {
                    return Err(format!("{} does not support image-to-image", self.summary));
                };
                let enc = self
                    .encoder
                    .as_ref()
                    .ok_or_else(|| format!("{} has no VAE encoder", self.summary))?;
                let start = sched.start_index(strength)?;
                let img = if (img.width, img.height) == (width, height) {
                    img.clone()
                } else {
                    img.resize_lanczos(width, height)
                };
                let m = enc.run_one(&hw, &[("image", Tensor::F32(from_image(&img)))], &[], None)?;
                let hw_n = h * w;
                let mut z: Vec<f32> = m
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (v - self.affine.1[i / hw_n]) / self.affine.0[i / hw_n])
                    .collect();
                sched.add_noise(&mut z, start, &mut rng);
                (to_layout(z), start)
            }
        };

        let ctx_t = Tensor::F32(ctx);
        let pooled_t = Tensor::F32(pooled);
        let neg_t = neg.map(|(c, n, p)| (Tensor::F32(c), n, Tensor::F32(p)));
        let gt = Tensor::F32(vec![guidance]);
        let t2 = std::time::Instant::now();
        for i in start..n_steps {
            let (t, sigma, k) = run.at(i);
            let sv = [("t", Val::Num(t)), ("sigma", Val::Num(sigma))];
            let ts = self.eval_n(&J::String(self.rt.timestep.clone()), &sv)?;
            let step = [
                ("t", t),
                ("sigma", sigma),
                ("timestep", ts),
                ("guidance", guidance as f64),
            ];
            let xin = Tensor::F32(if k == 1.0 {
                x.clone()
            } else {
                x.iter().map(|v| v * k).collect()
            });
            let tst = Tensor::F32(vec![ts as f32]);
            let eval_once =
                |ctx: &Tensor, nt: usize, pooled: &Tensor| -> Result<Vec<f32>, String> {
                    let vars = self.den_vars(h, w, nt);
                    let given = [
                        ("x", xin.clone()),
                        ("context", ctx.clone()),
                        ("pooled", pooled.clone()),
                        ("timestep", tst.clone()),
                        ("guidance", gt.clone()),
                    ];
                    self.denoiser.run_one(&vars, &given, &step, None)
                };
            let mut out = eval_once(&ctx_t, nt, &pooled_t)?;
            if let Some((uc, un, up)) = &neg_t {
                let u = eval_once(uc, *un, up)?;
                self.guide(&mut out, &u, scale)?;
            }
            if out.len() != x.len() {
                return Err(format!(
                    "denoiser returned {} values, expected {}",
                    out.len(),
                    x.len()
                ));
            }
            if self.rt.negate_v {
                out.iter_mut().for_each(|v| *v = -*v);
            }
            run.advance(i, &out, &mut x, &mut rng);
        }
        let t3 = std::time::Instant::now();
        if let Some((mean, std)) = &self.bn {
            for row in x.chunks_exact_mut(mean.len()) {
                for ((v, m), s) in row.iter_mut().zip(mean).zip(std) {
                    *v = *v * s + m;
                }
            }
        }
        let hw_n = h * w;
        let z: Vec<f32> = from_layout(&x)
            .iter()
            .enumerate()
            .map(|(i, v)| v * self.affine.0[i / hw_n] + self.affine.1[i / hw_n])
            .collect();
        let px = self
            .decoder
            .run_one(&hw, &[("z", Tensor::F32(z))], &[], None)?;
        let (oh, ow) = (h * f, w * f);
        let oc = px.len() / (oh * ow).max(1);
        if oc * oh * ow != px.len() || oc < 3 {
            return Err(format!(
                "VAE decoder returned {} values for {ow}x{oh}",
                px.len()
            ));
        }
        let img = to_image(&px, oc, oh, ow, self.rt.vae.output == "rgba_over_white");
        let n = n_steps - start;
        tracing::info!(
            "{}: {width}x{height} encode {:.2}s, denoise {n} steps x {} evals {:.2}s, vae decode {:.2}s",
            self.root.spec.family,
            (t1 - t0).as_secs_f64(),
            1 + cfg as usize,
            (t3 - t2).as_secs_f64(),
            t3.elapsed().as_secs_f64()
        );
        Ok(img)
    }
}

/// A built schedule: discrete (diffusers schedulers) or flow-matching sigmas.
enum Run {
    Discrete(Scheduler),
    Flow(Vec<f32>, Option<DpmSolver2>),
}

impl Run {
    fn steps(&self) -> usize {
        match self {
            Run::Discrete(s) => s.timesteps.len(),
            Run::Flow(s, _) => s.len() - 1,
        }
    }

    /// (timestep t, sigma, input scale) of step `i`.
    fn at(&self, i: usize) -> (f64, f64, f32) {
        match self {
            Run::Discrete(s) => (
                s.timesteps[i],
                s.sigmas.get(i).copied().unwrap_or(0.0),
                s.input_scale(i) as f32,
            ),
            Run::Flow(s, _) => (s[i] as f64 * TRAIN_STEPS as f64, s[i] as f64, 1.0),
        }
    }

    /// Step `i` of `x` with the model output `out`.
    fn advance(&mut self, i: usize, out: &[f32], x: &mut [f32], rng: &mut Rng) {
        match self {
            Run::Discrete(s) => {
                // ponytail: the discrete step replaces a Vec; one copy per step keeps `x` a slice.
                let mut v = x.to_vec();
                s.step(i, out, &mut v, rng);
                x.copy_from_slice(&v);
            }
            Run::Flow(s, None) => euler_step(x, out, s[i], s[i + 1]),
            Run::Flow(s, Some(dpm)) => dpm.step(x, out, s, i),
        }
    }
}

/// `runtime.video`: the clip-length rule, frame rate, decode mode and image conditioning of a
/// video pipeline (the rest of `runtime` is the diffusion pipeline's).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VideoSpec {
    /// Default output frames (expr).
    pub frames: J,
    /// Latent frames for `frames` output frames (expr over `frames`).
    pub latent_frames: J,
    /// Accepted `frames` (expr over `frames`) and the refusal text.
    #[serde(default)]
    pub frames_ok: J,
    #[serde(default)]
    pub frames_error: Option<String>,
    pub fps: J,
    /// `clip`: the decoder takes the whole clip (vars `frames, h, w`; `z [frames, c, h, w]`; first
    /// output `[frames', 3, H, W]`); `per_frame`: the image decoder runs on every latent frame.
    pub decode: String,
    /// `clean_first_frame`: an input image's encoded latent is latent frame 0, held clean (its
    /// tokens run at timestep 0: denoiser var `clean`) while the rest is denoised.
    #[serde(default)]
    pub image: Option<String>,
}

/// One video generation; `None` fields take the pipeline defaults.
#[derive(Debug, Clone, Default)]
pub struct VideoRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub frames: Option<usize>,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub steps: Option<usize>,
    pub guidance: Option<f32>,
    pub seed: u64,
}

/// Denoised video latents in the denoiser's layout; the first `held` values are the clean
/// (image-conditioned) frame, never stepped.
pub(crate) struct VideoLatents {
    pub x: Vec<f32>,
    pub frames: usize,
    pub h: usize,
    pub w: usize,
    #[cfg_attr(not(test), allow(dead_code))]
    pub held: usize,
}

/// A video pipeline's defaults.
#[derive(Debug, Clone, Copy)]
pub struct VideoDefaults {
    pub frames: usize,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub guidance: f32,
    pub fps: u32,
}

impl SpecDiffusion {
    /// The schedule of `steps` steps; `seq` (latent tokens for a dynamic mu) is `scheduler.seq`
    /// evaluated over `vars`, else `default_seq`.
    fn schedule(
        &self,
        steps: usize,
        vars: &[(&str, Val)],
        default_seq: usize,
    ) -> Result<Run, String> {
        Ok(match &self.sampler {
            Sampler::Discrete(cfg) => Run::Discrete(cfg.build(steps)?),
            Sampler::Flow {
                schedule,
                base,
                card,
                dpm,
                seq,
            } => {
                let seq = if seq.is_null() {
                    default_seq
                } else {
                    self.eval_n(seq, vars)? as usize
                };
                let sig = match base.as_str() {
                    "sd3" => default_sigmas(steps, schedule.shift),
                    "dpm_flow" => dpm_flow_sigmas(steps, schedule.shift),
                    "card" => match card {
                        Some(c) if c.len() == steps => schedule.sigmas(c, seq, steps),
                        _ => schedule.sigmas(&linspace_sigmas(steps), seq, steps),
                    },
                    "linspace" => schedule.sigmas(&linspace_sigmas(steps), seq, steps),
                    b => {
                        return Err(format!(
                            "scheduler.sigmas `{b}` (linspace, sd3, dpm_flow, card)"
                        ))
                    }
                };
                Run::Flow(sig, dpm.then(DpmSolver2::default))
            }
        })
    }

    /// `out = u + scale * (out - u)` per `runtime.guidance.kind`.
    fn guide(&self, out: &mut [f32], u: &[f32], scale: f32) -> Result<(), String> {
        match self.rt.guidance.kind.as_str() {
            "cfg" => out
                .iter_mut()
                .zip(u)
                .for_each(|(o, u)| *o = u + scale * (*o - u)),
            "cfg_rescale" => {
                let rw = match self.rt.latent.pack.as_str() {
                    "pack2x2" => 4 * self.channels,
                    "tokens" => self.channels,
                    _ => out.len(),
                };
                for (o, u) in out.chunks_exact_mut(rw).zip(u.chunks_exact(rw)) {
                    let pn = o.iter().map(|v| v * v).sum::<f32>().sqrt();
                    o.iter_mut()
                        .zip(u)
                        .for_each(|(o, u)| *o = u + scale * (*o - u));
                    let cn = o.iter().map(|v| v * v).sum::<f32>().sqrt();
                    let r = if cn > 0.0 { pn / cn } else { 1.0 };
                    o.iter_mut().for_each(|v| *v *= r);
                }
            }
            k => return Err(format!("guidance.kind `{k}` (none, cfg, cfg_rescale)")),
        }
        Ok(())
    }

    /// `Some` for a video pipeline (`runtime.video`).
    pub fn video_defaults(&self) -> Option<Result<VideoDefaults, String>> {
        let v = self.video.as_ref()?;
        let d = self.defaults;
        Some((|| {
            Ok(VideoDefaults {
                frames: self.eval_n(&v.frames, &[])? as usize,
                width: d.width,
                height: d.height,
                steps: d.steps,
                guidance: d.guidance,
                fps: self.eval_n(&v.fps, &[])? as u32,
            })
        })())
    }

    /// One clip of RGB frames; `image` (pipelines with `video.image`) conditions its first frame.
    pub fn generate_video(
        &self,
        req: &VideoRequest,
        image: Option<&ImageBuffer>,
        log: &dyn Fn(&str),
    ) -> Result<Vec<ImageBuffer>, String> {
        let l = self.video_latents(req, image, log)?;
        self.decode_video(&l, log)
    }

    /// The denoised latents of [`Self::generate_video`].
    pub(crate) fn video_latents(
        &self,
        req: &VideoRequest,
        image: Option<&ImageBuffer>,
        log: &dyn Fn(&str),
    ) -> Result<VideoLatents, String> {
        let v = self
            .video
            .as_ref()
            .ok_or_else(|| format!("{} is not a video pipeline", self.summary))?;
        let vd = self.video_defaults().expect("video pipeline")?;
        let d = self.defaults;
        let (width, height) = (req.width.unwrap_or(d.width), req.height.unwrap_or(d.height));
        if width == 0
            || height == 0
            || width % d.multiple != 0
            || height % d.multiple != 0
            || width.max(height) > self.max_side
        {
            return Err(format!(
                "width/height must be non-zero multiples of {} up to {}, got {width}x{height}",
                d.multiple, self.max_side
            ));
        }
        let frames = req.frames.unwrap_or(vd.frames);
        let fv = [("frames", Val::Num(frames as f64))];
        if !v.frames_ok.is_null() && !self.eval(&v.frames_ok, &fv)?.truthy() {
            return Err(format!(
                "--frames {frames}: {}",
                v.frames_error.as_deref().unwrap_or("not accepted")
            ));
        }
        let lf = self.eval_n(&v.latent_frames, &fv)? as usize;
        let steps = req.steps.unwrap_or(d.steps);
        if !(1..=self.max_steps).contains(&steps) {
            return Err(format!(
                "steps must be in 1..={}, got {steps}",
                self.max_steps
            ));
        }
        let guidance = req.guidance.unwrap_or(d.guidance);
        let gv = [("guidance", Val::Num(guidance as f64))];
        let g = &self.rt.guidance;
        let cfg = g.kind != "none" && (g.gate.is_null() || self.eval(&g.gate, &gv)?.truthy());
        let scale = if g.scale.is_null() {
            guidance
        } else {
            self.eval_n(&g.scale, &gv)? as f32
        };
        let (c, f) = (self.channels, self.factor);
        let (h, w) = (height / f, width / f);
        let hw_n = h * w;
        let tokens = match self.rt.latent.pack.as_str() {
            "tokens" => true,
            "none" => false,
            p => return Err(format!("video latents: pack `{p}` (none, tokens)")),
        };

        let t0 = std::time::Instant::now();
        let (ctx, nt, pooled) = self.conditioning(&req.prompt, false)?;
        let neg = match cfg {
            true => {
                let n = req
                    .negative_prompt
                    .clone()
                    .unwrap_or_else(|| self.rt.negative.default.clone());
                Some(self.conditioning(&n, true)?)
            }
            false => None,
        };
        log(&format!(
            "  prompt encoded: {:.2}s",
            t0.elapsed().as_secs_f64()
        ));

        let vars = [
            ("frames", Val::Num(lf as f64)),
            ("h", Val::Num(h as f64)),
            ("w", Val::Num(w as f64)),
        ];
        let mut run = self.schedule(steps, &vars, lf * hw_n)?;
        // One latent frame `[c, h*w]` <-> the denoiser's per-frame layout (frames stay outermost).
        let to_layout = |z: &[f32]| {
            if tokens {
                transpose(z, c, hw_n)
            } else {
                z.to_vec()
            }
        };
        let mut rng = Rng::new(req.seed);
        let s = match (&self.rt.latent.noise_scale, &run) {
            (J::String(k), Run::Discrete(s)) if k == "init_sigma" => s.init_sigma() as f32,
            (J::Null, _) => 1.0,
            (J::String(k), _) if k == "init_sigma" => 1.0,
            (j, _) => self.eval_n(j, &[])? as f32,
        };
        // Noise drawn directly in the denoiser's layout.
        let mut x: Vec<f32> = (0..lf * c * hw_n).map(|_| rng.gaussian() * s).collect();
        let clean = match image {
            None => 0,
            Some(img) => {
                if v.image.as_deref() != Some("clean_first_frame")
                    || !tokens
                    || !matches!(run, Run::Flow(..))
                {
                    return Err(format!("{} does not take an input image", self.summary));
                }
                let enc = self
                    .encoder
                    .as_ref()
                    .ok_or_else(|| format!("{} has no VAE encoder", self.summary))?;
                let t = std::time::Instant::now();
                let img = if (img.width, img.height) == (width, height) {
                    img.clone()
                } else {
                    img.resize_lanczos(width, height)
                };
                let hw = [vars[1].clone(), vars[2].clone()];
                let m = enc.run_one(&hw, &[("image", Tensor::F32(from_image(&img)))], &[], None)?;
                let z: Vec<f32> = m
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (v - self.affine.1[i / hw_n]) / self.affine.0[i / hw_n])
                    .collect();
                x[..c * hw_n].copy_from_slice(&to_layout(&z));
                log(&format!(
                    "  image encoded: {:.2}s",
                    t.elapsed().as_secs_f64()
                ));
                hw_n
            }
        };
        let held = clean * c;

        let ctx_t = Tensor::F32(ctx);
        let pooled_t = Tensor::F32(pooled);
        let neg_t = neg.map(|(c, n, p)| (Tensor::F32(c), n, Tensor::F32(p)));
        let gt = Tensor::F32(vec![guidance]);
        for i in 0..run.steps() {
            let st = std::time::Instant::now();
            let (t, sigma, k) = run.at(i);
            let sv = [("t", Val::Num(t)), ("sigma", Val::Num(sigma))];
            let ts = self.eval_n(&J::String(self.rt.timestep.clone()), &sv)?;
            let step = [
                ("t", t),
                ("sigma", sigma),
                ("timestep", ts),
                ("guidance", guidance as f64),
            ];
            let xin = Tensor::F32(if k == 1.0 {
                x.clone()
            } else {
                x.iter().map(|v| v * k).collect()
            });
            let tst = Tensor::F32(vec![ts as f32]);
            let eval_once =
                |ctx: &Tensor, nt: usize, pooled: &Tensor| -> Result<Vec<f32>, String> {
                    let mut dv = self.den_vars(h, w, nt);
                    dv.extend([
                        ("frames", Val::Num(lf as f64)),
                        ("clean", Val::Num(clean as f64)),
                    ]);
                    let given = [
                        ("x", xin.clone()),
                        ("context", ctx.clone()),
                        ("pooled", pooled.clone()),
                        ("timestep", tst.clone()),
                        ("guidance", gt.clone()),
                    ];
                    self.denoiser.run_one(&dv, &given, &step, None)
                };
            let mut out = eval_once(&ctx_t, nt, &pooled_t)?;
            if let Some((uc, un, up)) = &neg_t {
                let u = eval_once(uc, *un, up)?;
                self.guide(&mut out, &u, scale)?;
            }
            if out.len() != x.len() {
                return Err(format!(
                    "denoiser returned {} values, expected {}",
                    out.len(),
                    x.len()
                ));
            }
            if self.rt.negate_v {
                out.iter_mut().for_each(|v| *v = -*v);
            }
            run.advance(i, &out[held..], &mut x[held..], &mut rng);
            log(&format!(
                "  step {}/{} (t={t:.1}): {:.2}s",
                i + 1,
                run.steps(),
                st.elapsed().as_secs_f64()
            ));
        }
        Ok(VideoLatents {
            x,
            frames: lf,
            h,
            w,
            held,
        })
    }

    /// RGB frames of denoised latents.
    fn decode_video(
        &self,
        l: &VideoLatents,
        log: &dyn Fn(&str),
    ) -> Result<Vec<ImageBuffer>, String> {
        let v = self
            .video
            .as_ref()
            .ok_or_else(|| format!("{} is not a video pipeline", self.summary))?;
        let (c, f, h, w) = (self.channels, self.factor, l.h, l.w);
        let hw_n = h * w;
        let tokens = self.rt.latent.pack == "tokens";
        let from_layout = |x: &[f32]| {
            if tokens {
                transpose(x, hw_n, c)
            } else {
                x.to_vec()
            }
        };
        let vars = [
            ("frames", Val::Num(l.frames as f64)),
            ("h", Val::Num(h as f64)),
            ("w", Val::Num(w as f64)),
        ];
        let x = &l.x;
        // Latent frames `[c, h*w]`, decode affine applied.
        let t = std::time::Instant::now();
        let fl = c * hw_n;
        let frames_z: Vec<Vec<f32>> = x
            .chunks_exact(fl)
            .map(|fr| {
                from_layout(fr)
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v * self.affine.0[i / hw_n] + self.affine.1[i / hw_n])
                    .collect()
            })
            .collect();
        let over_white = self.rt.vae.output == "rgba_over_white";
        let out = match v.decode.as_str() {
            "clip" => {
                let px = self.decoder.run_one(
                    &vars,
                    &[("z", Tensor::F32(frames_z.concat()))],
                    &[],
                    None,
                )?;
                let (oh, ow) = (h * f, w * f);
                let n = px.len() / (3 * oh * ow).max(1);
                if n * 3 * oh * ow != px.len() {
                    return Err(format!(
                        "VAE decoder returned {} values for {ow}x{oh} RGB frames",
                        px.len()
                    ));
                }
                px.chunks_exact(3 * oh * ow)
                    .map(|p| to_image(p, 3, oh, ow, false))
                    .collect()
            }
            "per_frame" => {
                let hw = [vars[1].clone(), vars[2].clone()];
                let (oh, ow) = (h * f, w * f);
                frames_z
                    .into_iter()
                    .map(|z| {
                        let px = self
                            .decoder
                            .run_one(&hw, &[("z", Tensor::F32(z))], &[], None)?;
                        let oc = px.len() / (oh * ow).max(1);
                        Ok(to_image(&px, oc, oh, ow, over_white))
                    })
                    .collect::<Result<Vec<_>, String>>()?
            }
            m => return Err(format!("video.decode `{m}` (clip, per_frame)")),
        };
        log(&format!("  VAE decode: {:.2}s", t.elapsed().as_secs_f64()));
        Ok(out)
    }
}

impl ImageGenerator for SpecDiffusion {
    fn summary(&self) -> String {
        self.summary.clone()
    }

    fn model_id(&self) -> Option<String> {
        Some(self.id.clone()).filter(|s| !s.is_empty())
    }

    fn defaults(&self) -> ImageDefaults {
        self.defaults
    }

    fn text_to_image(&self, req: &ImageRequest) -> Result<ImageBuffer, String> {
        self.generate(req, None)
    }

    fn image_to_image(
        &self,
        req: &ImageRequest,
        init: &ImageBuffer,
        strength: f32,
    ) -> Result<ImageBuffer, String> {
        self.generate(req, Some((init, strength)))
    }
}

// ---- discovery ---------------------------------------------------------------------------------

/// Tensor names of a GGUF or safetensors file (headers only); `None` for any other file and for a
/// GGUF split set that is not a diffusion model (large LMs: only the first shard's header is read).
pub fn weight_names(path: &Path) -> Option<Vec<String>> {
    Some(match path.extension()?.to_str()? {
        "gguf" => {
            let reg = crate::arch::registry().ok()?;
            if crate::model::loaders::gguf::shard_paths(path).is_some() {
                let arch = crate::model::loaders::gguf::architecture(path)?;
                let fam = reg.detect(&crate::arch::Signals {
                    gguf_arch: Some(arch),
                    ..Default::default()
                })?;
                if !matches!(
                    reg.spec(&fam, &Default::default()).ok()?.pipeline,
                    crate::arch::Pipeline::Diffusion
                ) {
                    return None;
                }
            }
            crate::model::loaders::gguf::open(path)
                .ok()?
                .0
                .tensors
                .into_keys()
                .collect()
        }
        "safetensors" => {
            let file = std::fs::File::open(path).ok()?;
            // SAFETY: read-only mapping of a cache file, dropped before returning.
            let map = unsafe { memmap2::Mmap::map(&file) }.ok()?;
            crate::model::UniversalModelLoader::parse_safetensors(&map)
                .ok()?
                .tensors
                .into_keys()
                .collect()
        }
        _ => return None,
    })
}

fn ext_of(p: &Path) -> String {
    p.extension()
        .map(|e| e.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Whether family `a` is `b` or extends it (directly or through its parents).
fn related(reg: &crate::arch::Registry, a: &str, b: &str) -> bool {
    let mut f = Some(a.to_string());
    for _ in 0..16 {
        match f {
            Some(ref x) if x == b => return true,
            Some(x) => {
                f = reg
                    .spec(&x, &Default::default())
                    .ok()
                    .and_then(|s| s.extends)
            }
            None => return false,
        }
    }
    false
}

/// The pipeline class of a diffusers snapshot (`model_index.json` `_class_name`).
pub fn pipeline_class(dir: &Path) -> Option<String> {
    let s = std::fs::read_to_string(dir.join("model_index.json")).ok()?;
    let v: J = serde_json::from_str(&s).ok()?;
    v["_class_name"].as_str().map(str::to_string)
}

/// The diffusion family of a diffusers pipeline class (`None`: no `pipeline: diffusion` spec).
fn family_of_class(reg: &crate::arch::Registry, class: &str) -> Option<String> {
    let sig = crate::arch::Signals {
        diffusers_class: Some(class.to_string()),
        ..Default::default()
    };
    let f = reg.detect(&sig)?;
    (reg.spec(&f, &sig).ok()?.pipeline == crate::arch::Pipeline::Diffusion).then_some(f)
}

/// Loads `artifacts` as an image pipeline; `None`: not an image model of a ported family (other
/// loaders try). A diffusers snapshot loads by its `model_index.json` class. A weights file that is
/// a diffusion LoRA, or the denoiser of a family (`match.denoiser_tensors`), loads on the base
/// pipeline its model card's `base_model` names (fetched; the denoiser's own files skipped).
pub fn load_image(
    artifacts: &ModelArtifacts,
    hub: &HuggingFaceHub,
    mode: PlacementMode,
) -> Option<Result<Arc<dyn ImageGenerator>, String>> {
    load_image_in(crate::arch::registry().ok()?, artifacts, hub, mode)
}

/// [`load_image`] from the specs of `reg`.
pub fn load_image_in(
    reg: &crate::arch::Registry,
    artifacts: &ModelArtifacts,
    hub: &HuggingFaceHub,
    mode: PlacementMode,
) -> Option<Result<Arc<dyn ImageGenerator>, String>> {
    let arc = |r: Result<SpecDiffusion, String>| r.map(|p| Arc::new(p) as Arc<dyn ImageGenerator>);
    let dir = artifacts.local_dir.as_deref()?;
    if let Some(class) = pipeline_class(dir) {
        if family_of_class(reg, &class).is_some() {
            return Some(arc(SpecDiffusion::load_in(
                reg,
                artifacts,
                hub,
                mode,
                None,
                &Extras::default(),
            )));
        }
        if class.contains("Inpaint") {
            return Some(Err(format!(
                "{class} is an inpainting pipeline: it needs a mask, which no endpoint takes (use a text-to-image repo)"
            )));
        }
        if class.contains("Img2Img") || class.contains("Refiner") {
            return Some(Err(format!(
                "{class} is an image-to-image-only pipeline; load its text-to-image base repo and use /v1/images/edits"
            )));
        }
        // Another kind of pipeline a spec runs (video) loads elsewhere.
        let sig = crate::arch::Signals {
            diffusers_class: Some(class.clone()),
            ..Default::default()
        };
        if reg.detect(&sig).is_some() {
            return None;
        }
        return Some(Err(format!("diffusers pipeline {class} is not supported")));
    }
    let weights = artifacts.weights_path.as_deref()?;
    let names = weight_names(weights)?;
    if let Some(fam) = reg.detect_single_file(&names, &ext_of(weights)) {
        return Some(single_file(
            reg,
            &artifacts.model_id,
            weights,
            &fam,
            hub,
            mode,
            Vec::new(),
        ));
    }
    let lora = weights.extension().is_some_and(|e| e == "safetensors")
        && crate::model::lora::is_diffusion_lora(names.iter().map(String::as_str));
    let denoiser = if lora {
        None
    } else {
        Some(reg.detect_denoiser(&names)?)
    };
    if let Some(r) = denoiser
        .as_deref()
        .and_then(|d| split_files(reg, &artifacts.model_id, weights, d, hub, mode))
    {
        return Some(r);
    }
    // `--model org/repo/file.gguf` names one file of the repo.
    let repo = crate::model::hub::split_weight_file(&artifacts.model_id).0;
    let card = dir
        .join("README.md")
        .is_file()
        .then(|| dir.join("README.md"))
        .map_or_else(|| hub.fetch_file(repo, "README.md"), Ok)
        .and_then(|p| std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display())));
    let what = if lora {
        "diffusion LoRA"
    } else {
        "denoiser weights"
    };
    let card_text = card.as_ref().ok().cloned();
    let base = match card.map(|c| crate::model::lora::card_base_model(&c)) {
        Ok(Some(b)) => b,
        // A LoRA without a card base may belong to another kind of model.
        Ok(None) if lora => return None,
        Ok(None) => {
            return Some(Err(format!(
                "{repo}: {what}, but the model card names no base_model"
            )))
        }
        Err(_) if lora => return None,
        Err(e) => return Some(Err(format!("{repo}: {what}, model card unreadable: {e}"))),
    };
    // diffusers' official mirror of the SD 1.5 repo that LoRA cards still name.
    let base = match base.as_str() {
        "runwayml/stable-diffusion-v1-5" => {
            "stable-diffusion-v1-5/stable-diffusion-v1-5".to_string()
        }
        _ => base,
    };
    let gated = |e: String| {
        format!(
            "base model {base} could not be fetched ({e}); if it is gated, accept its license at https://huggingface.co/{base} with the HF_TOKEN account"
        )
    };
    let class = match hub.fetch_file(&base, "model_index.json").map_err(gated) {
        Ok(p) => p.parent().and_then(pipeline_class),
        // A LoRA on a single-file checkpoint repo.
        Err(_) if lora => {
            let token = std::env::var("HF_TOKEN").ok();
            let b = hub.auto_download_and_cache(&base, token.as_deref()).ok()?;
            let w = b.weights_path?;
            let fam = reg.detect_single_file(&weight_names(&w)?, &ext_of(&w))?;
            let id = format!("{} on {base}", artifacts.model_id);
            return Some(single_file(
                reg,
                &id,
                &w,
                &fam,
                hub,
                mode,
                vec![(weights.to_path_buf(), 1.0)],
            ));
        }
        Err(e) => return Some(Err(e)),
    };
    let fam = class.as_deref().and_then(|c| family_of_class(reg, c));
    match (&fam, &denoiser) {
        // A base no diffusion spec runs: other loaders may (denoiser tensor names are shared).
        (None, _) => return None,
        (Some(f), Some(d)) if !related(reg, f, d) && !related(reg, d, f) => {
            return Some(Err(format!(
                "{repo}: {d} denoiser weights, but its base_model {base} is a {f} pipeline"
            )))
        }
        _ => {}
    }
    let fam = fam.unwrap();
    Some((|| {
        let spec = reg.spec(&fam, &Default::default())?;
        let rt: Runtime = spec.runtime_as()?;
        let skip = match (&denoiser, spec.component(&rt.denoiser.component)?) {
            (Some(_), crate::arch::Component::Ref { dir: Some(d), .. }) => vec![d],
            _ => Vec::new(),
        };
        let skip: Vec<&str> = skip.iter().map(String::as_str).collect();
        let base_dir = hub.fetch_diffusers(&base, &skip).map_err(gated)?;
        let a = ModelArtifacts {
            model_id: format!("{} on {base}", artifacts.model_id),
            local_dir: Some(base_dir),
            is_cached: true,
            ..Default::default()
        };
        let extra = Extras {
            loras: if lora {
                vec![(weights.to_path_buf(), 1.0)]
            } else {
                Vec::new()
            },
            weights: match denoiser {
                Some(_) => vec![(rt.denoiser.component.clone(), weights.to_path_buf())],
                None => Vec::new(),
            },
            // A distilled finetune ships its own schedule: its `scheduler/` and its card's sigmas.
            scheduler: Some(dir.join("scheduler/scheduler_config.json"))
                .filter(|p| p.is_file())
                .map_or_else(
                    || hub.fetch_file(repo, "scheduler/scheduler_config.json"),
                    Ok,
                )
                .ok()
                .and_then(|p| serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()),
            card: card_text.clone(),
        };
        arc(SpecDiffusion::load_in(
            reg,
            &a,
            hub,
            mode,
            Some(&fam),
            &extra,
        ))
    })())
}

/// `runtime.split`: a split-file distribution (e.g. ComfyUI's `diffusion_models/`, `text_encoders/`,
/// `vae/` folders) whose denoiser file sits in a folder named `dir`.
#[derive(Deserialize)]
struct SplitSpec {
    dir: String,
    /// Component -> its sibling folder (the file there whose name contains `prefer`, else the
    /// first) or, absent, the `repo:path` file fetched.
    parts: HashMap<String, SplitPart>,
    /// Pipeline-root files: a `repo:path` string (fetched, copied) or a JSON value (written).
    #[serde(default)]
    files: HashMap<String, J>,
    /// `model_index.json` booleans: true when the denoiser's file or repo name contains (any
    /// case) one of the substrings.
    #[serde(default)]
    name_flags: HashMap<String, Vec<String>>,
}

#[derive(Deserialize)]
struct SplitPart {
    dir: String,
    #[serde(default)]
    prefer: String,
    fetch: String,
}

/// The `repo:path` file of a split layout, fetched.
fn fetch_ref(hub: &HuggingFaceHub, r: &str) -> Result<std::path::PathBuf, String> {
    let (repo, path) = r
        .split_once(':')
        .ok_or_else(|| format!("split file `{r}` is not repo:path"))?;
    hub.fetch_file(repo, path)
}

/// A denoiser file of `fam` inside its split layout's `dir` (`runtime.split`): the sibling folders'
/// components (fetched when missing) around a pipeline root assembled from the spec's `files`.
/// `None` when `fam` has no split layout or the file is elsewhere.
fn split_files(
    reg: &crate::arch::Registry,
    id: &str,
    weights: &Path,
    fam: &str,
    hub: &HuggingFaceHub,
    mode: PlacementMode,
) -> Option<Result<Arc<dyn ImageGenerator>, String>> {
    let spec = reg.spec(fam, &Default::default()).ok()?;
    let rt: Runtime = spec.runtime_as().ok()?;
    let split: SplitSpec = match rt
        .other
        .get("split")
        .map(|j| serde_json::from_value(j.clone()))
    {
        None => return None,
        Some(Ok(s)) => s,
        Some(Err(e)) => return Some(Err(format!("{fam}: runtime.split: {e}"))),
    };
    let folder = weights.parent()?;
    if folder.file_name()? != split.dir.as_str() {
        return None;
    }
    let base = folder.parent()?;
    Some((|| {
        let (root, parts) = split_root(&split, &rt.denoiser.component, base, weights, id, hub)?;
        let a = ModelArtifacts {
            model_id: id.to_string(),
            local_dir: Some(root.clone()),
            is_cached: true,
            ..Default::default()
        };
        let extra = Extras {
            weights: parts,
            ..Default::default()
        };
        SpecDiffusion::load_in(reg, &a, hub, mode, Some(fam), &extra)
            .map(|p| Arc::new(p) as Arc<dyn ImageGenerator>)
    })())
}

/// The pipeline root (`files`, `model_index.json` with `name_flags`) and component weights of a
/// split layout around denoiser file `weights` (`component`) whose folders sit in `base`.
fn split_root(
    split: &SplitSpec,
    component: &str,
    base: &Path,
    weights: &Path,
    id: &str,
    hub: &HuggingFaceHub,
) -> Result<(std::path::PathBuf, Vec<(String, std::path::PathBuf)>), String> {
    let mut parts = vec![(component.to_string(), weights.to_path_buf())];
    for (comp, p) in &split.parts {
        let mut local: Vec<_> = std::fs::read_dir(base.join(&p.dir))
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|f| f.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        local.sort();
        let pick = local
            .iter()
            .find(|f| f.to_string_lossy().contains(&p.prefer))
            .or(local.first())
            .cloned();
        parts.push((
            comp.clone(),
            match pick {
                Some(f) => f,
                None => fetch_ref(hub, &p.fetch)?,
            },
        ));
    }
    let root = std::env::temp_dir()
        .join("gibbs-split")
        // One root per denoiser file (its name flags differ from its siblings').
        .join(weights.to_string_lossy().replace(['/', '\\', ':'], "_"));
    let write = |rel: &str, bytes: &[u8]| -> Result<(), String> {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).map_err(|e| e.to_string())?;
        std::fs::write(&p, bytes).map_err(|e| format!("{}: {e}", p.display()))
    };
    let mut index = J::Object(Default::default());
    for (rel, v) in &split.files {
        match v {
            J::String(r) => {
                let f = fetch_ref(hub, r)?;
                write(
                    rel,
                    &std::fs::read(&f).map_err(|e| format!("{}: {e}", f.display()))?,
                )?;
            }
            v if rel == "model_index.json" => index = v.clone(),
            v => write(rel, v.to_string().as_bytes())?,
        }
    }
    let name = format!(
        "{id} {}",
        weights.file_name().unwrap_or_default().to_string_lossy()
    )
    .to_lowercase();
    for (k, subs) in &split.name_flags {
        index[k] = J::Bool(subs.iter().any(|s| name.contains(&s.to_lowercase())));
    }
    write("model_index.json", index.to_string().as_bytes())?;
    Ok((root, parts))
}

/// A single-file checkpoint of `fam`: the configs of its `runtime.reference` repo (fetched, no
/// weights), every `{family, dir}` component's weights read from `weights`.
fn single_file(
    reg: &crate::arch::Registry,
    id: &str,
    weights: &Path,
    fam: &str,
    hub: &HuggingFaceHub,
    mode: PlacementMode,
    loras: Vec<(std::path::PathBuf, f32)>,
) -> Result<Arc<dyn ImageGenerator>, String> {
    let sig = crate::arch::Signals {
        hub_id: id.to_string(),
        file_name: weights
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        ..Default::default()
    };
    let spec = reg.spec(fam, &sig)?;
    let rt: Runtime = spec.runtime_as()?;
    let reference = match &rt.reference {
        Some(J::String(r)) => r.trim_matches('\'').to_string(),
        _ => {
            return Err(format!(
                "{fam}: single-file checkpoints need `runtime.reference`"
            ))
        }
    };
    let fetch = |f: &str| {
        hub.fetch_file(&reference, f)
            .map_err(|e| format!("{fam} reference {reference}: {e}"))
    };
    let root = fetch("model_index.json")?
        .parent()
        .ok_or("reference snapshot has no directory")?
        .to_path_buf();
    fetch("scheduler/scheduler_config.json")?;
    let mut parts = Vec::new();
    for (name, c) in &spec.components {
        if let Ok(crate::arch::Component::Ref { dir: Some(d), .. }) =
            serde_json::from_value(c.clone())
        {
            fetch(&format!("{d}/config.json"))?;
            parts.push((name.clone(), weights.to_path_buf()));
        }
    }
    // Only the files each text entry reads (missing ones would be network round trips).
    for e in rt
        .text_encoders
        .iter()
        .filter(|e| !e.tokenizer.starts_with("hub:"))
    {
        let d = &e.tokenizer;
        if hub
            .fetch_file(&reference, &format!("{d}/tokenizer.json"))
            .is_err()
        {
            fetch(&format!("{d}/spiece.model"))?;
        }
        if e.pad_token.as_deref() == Some("config") {
            let d = e.tokenizer_config.as_ref().unwrap_or(d);
            for f in ["special_tokens_map.json", "tokenizer_config.json"] {
                let _ = hub.fetch_file(&reference, &format!("{d}/{f}"));
            }
        }
    }
    let a = ModelArtifacts {
        model_id: id.to_string(),
        local_dir: Some(root),
        is_cached: true,
        ..Default::default()
    };
    let extra = Extras {
        loras,
        weights: parts,
        ..Default::default()
    };
    SpecDiffusion::load_in(reg, &a, hub, mode, Some(fam), &extra)
        .map(|p| Arc::new(p) as Arc<dyn ImageGenerator>)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{Registry, Signals, Source};
    use crate::graph::DType;

    struct NoWeights;
    impl Source for NoWeights {
        fn tensor(&self, _: &str) -> Option<(Vec<usize>, DType)> {
            None
        }
    }
    struct NoEnv;
    impl Env for NoEnv {
        fn lookup(&self, _: &str, _: &[Val]) -> Option<Val> {
            None
        }
    }

    /// A gen input's `step:` arg survives the build verbatim (quoted or not) and evaluates per step.
    #[test]
    fn step_args_evaluate_per_step() {
        let spec = r#"{family: "toy", pipeline: "diffusion", components: {n: {
          vars: ["d"],
          inputs: [{name: "t_emb", dtype: "f32", shape: ["d"], gen: "timestep_embedding",
                    args: {t: "step:sigma * 1000", dim: "d", scale: "'step:guidance'"}}],
          body: [{op: "Unary", args: {kind: "Neg"}, in: ["i:t_emb"], out: "y"}],
          outputs: ["y"]}}}"#;
        let r = Registry::from_files(&[("families/toy.json5".into(), spec.into())], &[]).unwrap();
        let s = r.spec("toy", &Signals::default()).unwrap();
        let inst = build(&s, "n", &NoEnv, &NoWeights, &[("d", Val::Num(8.0))]).unwrap();
        let (g, args) = inst.inputs[0].gen.clone().unwrap();
        assert_eq!(args["t"], "step:sigma * 1000");
        assert_eq!(args["scale"], "step:guidance");
        let at = |sigma: f64| match gen_tensor(
            &g,
            &step_args(&args, &[("sigma", sigma), ("guidance", 2.0)]).unwrap(),
        )
        .unwrap()
        {
            Tensor::F32(v) => v,
            _ => unreachable!(),
        };
        let want = |t: f64| {
            let a = serde_json::json!({"t": t, "dim": 8, "scale": 2.0});
            match gen_tensor("timestep_embedding", &a).unwrap() {
                Tensor::F32(v) => v,
                _ => unreachable!(),
            }
        };
        assert_eq!(at(0.5), want(500.0));
        assert_eq!(at(0.25), want(250.0));
        assert_ne!(at(0.5), at(0.25));
    }

    /// A split layout's root: preferred sibling files per component, written configs, name flags.
    #[test]
    fn split_root_assembles_the_pipeline() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().join("snap");
        for f in [
            "dm/toy_turbo_bf16.safetensors",
            "te/a_fp8.safetensors",
            "te/b_bf16.safetensors",
            "vae/v.safetensors",
        ] {
            std::fs::create_dir_all(base.join(f).parent().unwrap()).unwrap();
            std::fs::write(base.join(f), b"").unwrap();
        }
        let split: SplitSpec = serde_json::from_value(serde_json::json!({
            "dir": "dm",
            "parts": {"text_encoder": {"dir": "te", "prefer": "bf16", "fetch": "x/y:te.safetensors"},
                      "vae": {"dir": "vae", "fetch": "x/y:v.safetensors"}},
            "files": {"model_index.json": {"_class_name": "ToyPipeline"}, "text_encoder/config.json": {"n": 2}},
            "name_flags": {"is_distilled": ["TURBO"], "is_raw": ["raw"]}
        }))
        .unwrap();
        let w = base.join("dm/toy_turbo_bf16.safetensors");
        let hub = HuggingFaceHub::with_cache_dir(tmp.path().join("hub"));
        let (root, mut parts) =
            split_root(&split, "transformer", &base, &w, "org/toy", &hub).unwrap();
        parts.sort();
        assert_eq!(
            parts,
            [
                (
                    "text_encoder".to_string(),
                    base.join("te/b_bf16.safetensors")
                ),
                ("transformer".to_string(), w.clone()),
                ("vae".to_string(), base.join("vae/v.safetensors")),
            ]
        );
        let read = |f: &str| -> J {
            serde_json::from_str(&std::fs::read_to_string(root.join(f)).unwrap()).unwrap()
        };
        assert_eq!(
            read("model_index.json"),
            serde_json::json!({"_class_name": "ToyPipeline", "is_distilled": true, "is_raw": false})
        );
        assert_eq!(
            read("text_encoder/config.json"),
            serde_json::json!({"n": 2})
        );
    }
}
