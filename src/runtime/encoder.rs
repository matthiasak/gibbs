//! Encoder pipeline: a family spec's `encoder` component run once per (unpadded) sequence. The
//! first instance (one row) compiles one `GraphProgram`; every sequence length runs through
//! `GraphProgram::run_reshaped` on its weights. One `head` per encoder (`classify` logits, a pooled
//! `embedding`, or the `hidden` states), fixed at load.
//!
//! Inputs the runtime supplies by name: `token_ids`, `positions` (per the spec's
//! `runtime.position_ids` expression: `absolute` 0.., or `pad_offset` from the config var
//! `pad_id` + 1), `token_types` (segment ids) and `attention_mask` (all ones); any other input is
//! a spec generator.
//!
//! On top of it, [`Classifier`] scores text (pairs) with a sequence-classification checkpoint
//! (Hugging Face `*ForSequenceClassification`, a PEFT LoRA `SEQ_CLS` adapter merged into its base,
//! a sentence-transformers body with a scikit-learn logistic head, or a GGUF classifier export)
//! and [`system1`] answers zero-shot questions from sentence embeddings. Everything
//! family-specific (which conversion rebuilds a missing tokenizer.json, the position scheme, the
//! sentence-embedding head) is spec `runtime` data.

pub mod convert;
mod logistic;
pub mod system1;

use crate::arch::build::{build, InputPlan, Instance, ShapeCache};
use crate::arch::expr::Val;
use crate::arch::gen::{generate, GenData};
use crate::arch::load::{self, Model};
use crate::arch::{Pipeline, Signals};
use crate::graph::program::GraphProgram;
use crate::graph::Tensor;
use crate::hardware::{Device, PlacementMode};
use crate::model::hub::{HuggingFaceHub, ModelArtifacts};
use crate::model::weight_format::{ReencodeConfig, ReencodePolicy};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

const COMPONENT: &str = "encoder";

/// An encoder family's `runtime` data.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Runtime {
    /// Expression evaluating to `absolute` | `pad_offset`.
    #[serde(default)]
    position_ids: Option<String>,
    /// Conversions rebuilding a missing tokenizer.json, tried in order ([`convert`]).
    #[serde(default)]
    tokenizer: Vec<convert::Converter>,
    /// The pooling head System 1 embeds sentences with; absent = not a sentence encoder.
    #[serde(default)]
    system1: Option<String>,
}

impl Runtime {
    /// The head System 1 embeds sentences with.
    pub fn system1(&self) -> Option<&str> {
        self.system1.as_deref()
    }
}

/// A loaded encoder spec compiled on one device.
pub struct SpecEncoder {
    pub model: Model,
    rt: Runtime,
    head: String,
    /// `Some(pad)`: positions count from `pad + 1`; pad tokens sit at `pad`.
    pad_offset: Option<u32>,
    shapes: ShapeCache,
    program: Mutex<GraphProgram>,
    device: Device,
    params: usize,
}

impl SpecEncoder {
    /// Loads `artifacts` as `family` (detected when `None`) with its weights as stored.
    pub fn load(
        artifacts: &ModelArtifacts,
        family: Option<&str>,
        mode: impl Into<PlacementMode>,
        head: &str,
    ) -> Result<Self, String> {
        let reencode = ReencodeConfig {
            policy: ReencodePolicy::BitExact,
            ..Default::default()
        };
        let model = load::load_with(artifacts, crate::arch::registry()?, family, reencode)?;
        Self::new(model, mode.into(), head)
    }

    /// Compiles `model`'s `head` instance under `mode`; refuses checkpoint tensors it never reads.
    pub fn new(model: Model, mode: PlacementMode, head: &str) -> Result<Self, String> {
        let device = crate::graph::avenues::resolved_backend(mode);
        load::check(&model)?;
        let fam = model.spec.family.clone();
        let rt: Runtime = model.spec.runtime_as()?;
        let pad_offset = match &rt.position_ids {
            None => None,
            Some(e) => match model.env.eval(&e.as_str().into())? {
                Val::Str(s) if s == "absolute" => None,
                Val::Str(s) if s == "pad_offset" => Some(model.env.var("pad_id")?.usize()? as u32),
                v => return Err(format!("{fam}: runtime.position_ids is {v}, not absolute|pad_offset")),
            },
        };
        let shapes = ShapeCache::new(COMPONENT);
        let v = vars(1, head);
        let first = shapes.get(&v, || build(&model.spec, COMPONENT, &model.env, &model.store, &v))?;
        let used: HashSet<String> = first.weights.iter().cloned().collect();
        let unused = model.store.unconsumed(&used, &model.ignore);
        if !unused.is_empty() {
            return Err(format!(
                "{fam}: checkpoint tensors the `{head}` encoder never reads (list deliberate ones in `ignore`): {}",
                unused.join(", ")
            ));
        }
        let weights = model.store.materialize(&first.weights)?;
        let params = weights.parameter_count();
        let program = GraphProgram::compile_reshapable(first.graph.clone(), weights, mode)?;
        if program.primary_device() != device || (device != Device::Cpu && program.cpu_node_count() > 0) {
            return Err(format!(
                "gpu-only: {fam} compiled for {} with {} CPU nodes, wanted every node on {device}",
                program.primary_device(),
                program.cpu_node_count()
            ));
        }
        Ok(Self {
            model,
            rt,
            head: head.to_string(),
            pad_offset,
            shapes,
            program: Mutex::new(program),
            device,
            params,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Weight count of the compiled instance.
    pub fn parameter_count(&self) -> usize {
        self.params
    }

    /// A config var as a number (`hidden`, `layers`, `heads`, `max_position_embeddings`...).
    pub fn var(&self, name: &str) -> Result<usize, String> {
        self.model.env.var(name)?.usize()
    }

    /// The most tokens a sequence may hold (pad-offset positions lose `pad + 1` table rows).
    pub fn capacity(&self) -> Result<usize, String> {
        let max = self.var("max_position_embeddings")?;
        Ok(max - self.pad_offset.map_or(0, |p| p as usize + 1))
    }

    /// `repo_id`'s tokenizer.json, else the first `runtime.tokenizer` conversion whose `when`
    /// holds, rebuilt from its files. Truncates to `tokenizer_config.json`'s `model_max_length`,
    /// at most `capacity` tokens; never pads.
    pub fn tokenizer(
        &self,
        hub: &HuggingFaceHub,
        repo_id: &str,
        tok_cfg: &Value,
        capacity: usize,
    ) -> Result<tokenizers::Tokenizer, String> {
        let mut tok = match hub.fetch_file(repo_id, "tokenizer.json") {
            Ok(p) => tokenizers::Tokenizer::from_file(&p).map_err(|e| format!("load {}: {e}", p.display()))?,
            Err(json_err) => {
                let mut conv = None;
                for c in &self.rt.tokenizer {
                    let on = match &c.when {
                        None => true,
                        Some(w) => self.model.env.eval(&w.as_str().into())?.truthy(),
                    };
                    if on {
                        conv = Some(&c.kind);
                        break;
                    }
                }
                let kind = conv.ok_or_else(|| format!("{repo_id}: no tokenizer.json ({json_err})"))?;
                let files = kind
                    .files()
                    .into_iter()
                    .map(|f| {
                        hub.fetch_file(repo_id, f)
                            .map_err(|e| format!("{repo_id}: no tokenizer.json ({json_err}) or {f} ({e})"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let paths: Vec<&Path> = files.iter().map(|p| p.as_path()).collect();
                kind.build(&paths, tok_cfg)?
            }
        };
        let max_length = tok_cfg["model_max_length"]
            .as_f64()
            .map_or(capacity, |m| (m as usize).min(capacity));
        tok.with_truncation(Some(tokenizers::TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|e| format!("tokenizer truncation: {e}"))?;
        tok.with_padding(None);
        Ok(tok)
    }

    /// The encoder head's output for one sequence (`types`: segment ids, one per id).
    pub fn run(&self, ids: &[u32], types: &[u32]) -> Result<Vec<f32>, String> {
        let n = ids.len();
        if n == 0 {
            return Err("tokenizer produced no tokens".into());
        }
        let cap = self.capacity()?;
        if n > cap {
            return Err(format!("{n} tokens exceed {cap} positions"));
        }
        let inst = self.instance(n)?;
        let inputs = inst
            .inputs
            .iter()
            .map(|d| self.input(d, ids, types))
            .collect::<Result<Vec<_>, _>>()?;
        let mut out = self.program.lock().run_reshaped(&inst.graph, inputs, true)?;
        if out.is_empty() {
            return Err("encoder graph produced no output".into());
        }
        out.swap_remove(0).into_f32()
    }

    fn instance(&self, rows: usize) -> Result<Arc<Instance>, String> {
        let v = vars(rows, &self.head);
        let m = &self.model;
        self.shapes
            .get(&v, || build(&m.spec, COMPONENT, &m.env, &m.store, &v))
    }

    fn input(&self, d: &InputPlan, ids: &[u32], types: &[u32]) -> Result<Tensor, String> {
        let n = ids.len();
        Ok(match (&d.gen, d.name.as_str()) {
            (Some((g, args)), _) => match generate(g, args)
                .map_err(|e| format!("input `{}`: {e}", d.name))?
                .data
            {
                GenData::F32(v) => Tensor::F32(v),
                GenData::U32(v) => Tensor::U32(v),
            },
            (None, "token_ids") => Tensor::U32(ids.to_vec()),
            (None, "positions") => Tensor::U32(positions(ids, self.pad_offset)),
            (None, "token_types") if types.len() == n => Tensor::U32(types.to_vec()),
            (None, "token_types") => Tensor::U32(vec![0; n]),
            (None, "attention_mask") => Tensor::U32(vec![1; n]),
            (None, other) => {
                return Err(format!(
                    "input `{other}` has no generator and is none the encoder runtime supplies (token_ids, positions, token_types, attention_mask)"
                ))
            }
        })
    }
}

fn vars(rows: usize, head: &str) -> Vec<(&'static str, Val)> {
    vec![
        ("rows", Val::Num(rows as f64)),
        ("head", Val::Str(head.to_string())),
    ]
}

/// Position ids: 0.. , or (`pad`) non-pad tokens from `pad + 1` with pad tokens at `pad`.
fn positions(ids: &[u32], pad: Option<u32>) -> Vec<u32> {
    match pad {
        None => (0..ids.len() as u32).collect(),
        Some(pad) => {
            let mut seen = 0;
            ids.iter()
                .map(|&t| {
                    if t == pad {
                        pad
                    } else {
                        seen += 1;
                        pad + seen
                    }
                })
                .collect()
        }
    }
}

// ---- family lookup ---------------------------------------------------------------------------

/// The encoder family claiming a `config.json` whose class is `arch`, with its `runtime`.
pub fn family(cfg: &Value, arch: Option<&str>) -> Option<(String, Runtime)> {
    let reg = crate::arch::registry().ok()?;
    let sig = Signals {
        hf_model_type: cfg["model_type"].as_str().map(str::to_string),
        hf_architectures: arch.map(str::to_string).into_iter().collect(),
        ..Default::default()
    };
    encoder_family(reg, &sig)
}

/// The encoder family claiming GGUF architecture `arch`.
pub fn gguf_family(arch: &str) -> Option<String> {
    let reg = crate::arch::registry().ok()?;
    let sig = Signals {
        gguf_arch: Some(arch.to_string()),
        ..Default::default()
    };
    encoder_family(reg, &sig).map(|(f, _)| f)
}

fn encoder_family(reg: &crate::arch::Registry, sig: &Signals) -> Option<(String, Runtime)> {
    let f = reg.detect(sig)?;
    let spec = reg.spec(&f, sig).ok()?;
    if spec.pipeline != Pipeline::Encoder {
        return None;
    }
    let rt = spec.runtime_as().ok()?;
    Some((f, rt))
}

/// Every `*ForSequenceClassification` class an encoder family claims (the registry is static,
/// so computed once).
pub fn classifier_architectures() -> &'static [String] {
    static ARCHS: std::sync::LazyLock<Vec<String>> = std::sync::LazyLock::new(|| {
        let Ok(reg) = crate::arch::registry() else {
            return Vec::new();
        };
        reg.matchers()
            .into_iter()
            .filter(|(_, _, p)| *p == Pipeline::Encoder)
            .flat_map(|(_, m, _)| m.hf_architectures)
            .filter(|a| a.ends_with("ForSequenceClassification"))
            .collect()
    });
    &ARCHS
}

/// Text generators: `*ForCausalLM`, `*ForConditionalGeneration`, `*LMHeadModel`.
pub fn is_generator(arch: &str) -> bool {
    ["ForCausalLM", "ForConditionalGeneration", "LMHeadModel"]
        .iter()
        .any(|s| arch.ends_with(s))
}

// ---- classification ----------------------------------------------------------------------------

/// How logits become scores (the HF `problem_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// single_label_classification: probabilities over the labels.
    Softmax,
    /// multi_label_classification: an independent probability per label.
    Sigmoid,
    /// regression, or one label without a problem_type (rerankers, reward models): raw logits,
    /// or scores a head already normalized.
    Identity,
}

impl Activation {
    /// How `config.json` (`problem_type`, label count `n`) turns logits into scores.
    pub fn of(cfg: &Value, n: usize) -> Self {
        match cfg["problem_type"].as_str() {
            Some("multi_label_classification") => Self::Sigmoid,
            Some("regression") => Self::Identity,
            Some("single_label_classification") => Self::Softmax,
            _ if n == 1 => Self::Identity,
            _ => Self::Softmax,
        }
    }

    pub fn apply(self, logits: Vec<f32>) -> Vec<f32> {
        match self {
            Self::Softmax => {
                let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
                let s: f32 = e.iter().sum();
                e.into_iter().map(|v| v / s).collect()
            }
            Self::Sigmoid => logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
            Self::Identity => logits,
        }
    }
}

/// One label and its score.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelScore {
    pub label: String,
    pub score: f32,
}

/// `labels` with `activation` of `logits`, by descending score.
pub fn label_scores(labels: &[String], activation: Activation, logits: Vec<f32>) -> Vec<LabelScore> {
    let mut out: Vec<LabelScore> = labels
        .iter()
        .zip(activation.apply(logits))
        .map(|(l, score)| LabelScore { label: l.clone(), score })
        .collect();
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out
}

/// What a loaded classifier reports about itself.
pub struct Meta {
    pub architecture: String,
    pub labels: Vec<String>,
    pub activation: Activation,
}

/// A sequence classifier on an encoder spec.
pub struct Classifier {
    tok: tokenizers::Tokenizer,
    head: Head,
}

enum Head {
    /// The spec's `classify` logits.
    Logits(SpecEncoder),
    /// A pooled `embedding` scored by a scikit-learn LogisticRegression (`predict_proba`).
    Logistic(SpecEncoder, logistic::Logistic),
}

/// A sequence-classification `config.json`: its class, and for a PEFT LoRA `SEQ_CLS` adapter the
/// base repo and the adapter config.
pub struct Resolved {
    pub cfg: Value,
    pub arch: String,
    pub peft: Option<(String, Value)>,
}

/// config.json and the `*ForSequenceClassification` architecture to build. A PEFT LoRA adapter
/// repo (`adapter_config.json`, task `SEQ_CLS`) resolves to its base model's config with the
/// adapter repo's labels.
pub fn resolve_config(hub: &HuggingFaceHub, repo_id: &str) -> Result<Resolved, String> {
    let own = hub.fetch_file(repo_id, "config.json").and_then(|p| read_json(&p));
    let own_arch = own
        .as_ref()
        .ok()
        .and_then(|c| c["architectures"][0].as_str())
        .map(str::to_string);
    if let (Ok(cfg), Some(a)) = (&own, &own_arch) {
        if a.ends_with("ForSequenceClassification") {
            return Ok(Resolved { cfg: cfg.clone(), arch: a.clone(), peft: None });
        }
    }
    let Ok(ad) = hub
        .fetch_file(repo_id, "adapter_config.json")
        .and_then(|p| read_json(&p))
    else {
        return match own {
            Err(e) => Err(e),
            Ok(cfg) => Err(match (&own_arch, cfg["auto_map"].is_null()) {
                (Some(a), false) => format!("{repo_id}: custom architecture '{a}' (remote code)"),
                (Some(a), true) => format!("{repo_id}: {a} has no sequence-classification head"),
                (None, _) => format!("{repo_id}: config.json names no architecture"),
            }),
        };
    };
    let task = ad["task_type"].as_str().unwrap_or("none");
    let kind = ad["peft_type"].as_str().unwrap_or("none");
    if kind != "LORA" || task != "SEQ_CLS" {
        return Err(format!(
            "{repo_id}: PEFT {kind} adapter for task {task}; only LORA SEQ_CLS adapters classify"
        ));
    }
    crate::model::peft::lora_scale(&ad).map_err(|e| format!("{repo_id}: {e}"))?;
    let base = ad["base_model_name_or_path"]
        .as_str()
        .filter(|b| b.split('/').count() == 2 && !b.starts_with('.'))
        .ok_or_else(|| {
            format!(
                "{repo_id}: adapter base_model_name_or_path {} is not a Hub repo",
                ad["base_model_name_or_path"]
            )
        })?
        .to_string();
    let mut cfg = read_json(&hub.fetch_file(&base, "config.json")?)?;
    let base_arch = cfg["architectures"][0]
        .as_str()
        .ok_or_else(|| format!("{base}: config.json names no architecture"))?;
    let stem = ["ForSequenceClassification", "ForMaskedLM", "ForPreTraining", "ForCausalLM", "Model"]
        .iter()
        .find_map(|s| base_arch.strip_suffix(s))
        .unwrap_or(base_arch);
    let arch = format!("{stem}ForSequenceClassification");
    // Labels and scoring of the fine-tune live in the adapter repo's config.json, when it has one.
    if let Ok(own) = &own {
        for k in ["id2label", "label2id", "problem_type"] {
            if !own[k].is_null() {
                cfg[k] = own[k].clone();
            }
        }
    }
    Ok(Resolved { cfg, arch, peft: Some((base, ad)) })
}

impl Classifier {
    /// A Hugging Face `*ForSequenceClassification` checkpoint `r` of `repo_id` on its encoder
    /// family. A PEFT adapter is merged into its base once (`peft::prepare_seq_cls`), and the
    /// resolved config (classification class, labels) is written beside the merged weights.
    pub fn from_hf(hub: &HuggingFaceHub, repo_id: &str, r: Resolved, mode: PlacementMode) -> Result<(Meta, Self), String> {
        let token = std::env::var("HF_TOKEN").ok();
        // A PEFT adapter reads its base model's tokenizer unless it ships one.
        let tok_repo = match &r.peft {
            Some((base, _)) if hub.fetch_file(repo_id, "tokenizer.json").is_err() => base.as_str(),
            _ => repo_id,
        };
        if r.peft.is_some() {
            crate::model::peft::prepare_seq_cls(hub, repo_id, token.as_deref())?;
        }
        let mut a = hub.auto_download_and_cache(repo_id, token.as_deref())?;
        if r.peft.is_some() {
            let dir = a
                .weights_path
                .as_deref()
                .and_then(Path::parent)
                .ok_or_else(|| format!("{repo_id}: no merged snapshot"))?;
            let path = dir.join("config.json");
            let mut cfg = r.cfg.clone();
            // The base names its own class (`...Model`, `...ForMaskedLM`); the spec matches the
            // classification architecture.
            cfg["architectures"] = serde_json::json!([r.arch]);
            std::fs::write(&path, cfg.to_string()).map_err(|e| format!("{}: {e}", path.display()))?;
            a.config_path = Some(path);
        }
        let enc = SpecEncoder::load(&a, None, mode, "classify").map_err(|e| format!("{repo_id}: {e}"))?;
        let tok_cfg = hub
            .fetch_file(tok_repo, "tokenizer_config.json")
            .ok()
            .map(|p| read_json(&p))
            .transpose()?
            .unwrap_or(Value::Null);
        let tok = enc.tokenizer(hub, tok_repo, &tok_cfg, enc.capacity()?)?;
        let n = enc.var("labels")?;
        let labels = (0..n)
            .map(|i| {
                r.cfg["id2label"][i.to_string()]
                    .as_str()
                    .map_or_else(|| format!("LABEL_{i}"), str::to_string)
            })
            .collect();
        let meta = Meta {
            architecture: r.arch,
            labels,
            activation: Activation::of(&r.cfg, n),
        };
        Ok((meta, Self { tok, head: Head::Logits(enc) }))
    }

    /// A sentence-transformers repo with a scikit-learn head (`config_setfit.json`): the body
    /// pooled per `1_Pooling/config.json` (and normalized) in its encoder spec, then the joblib
    /// `model_head.pkl` LogisticRegression; scores are its `predict_proba`.
    pub fn from_sentence_transformers(hub: &HuggingFaceHub, repo_id: &str, mode: PlacementMode) -> Result<(Meta, Self), String> {
        let err = |e: String| format!("{repo_id}: {e}");
        let st = read_json(&hub.fetch_file(repo_id, "config_setfit.json")?)?;
        let cfg = read_json(&hub.fetch_file(repo_id, "config.json")?)?;
        let arch = cfg["architectures"][0].as_str().unwrap_or("").to_string();
        // modules.json: Transformer, Pooling, optional Normalize; nothing else.
        let modules = read_json(&hub.fetch_file(repo_id, "modules.json")?)?;
        let mut pool_dir = None;
        let mut normalize = st["normalize_embeddings"].as_bool().unwrap_or(false);
        for (i, m) in modules.as_array().ok_or("modules.json is not a list")?.iter().enumerate() {
            let ty = m["type"].as_str().unwrap_or("");
            match (i, ty.rsplit('.').next().unwrap_or("")) {
                (0, "Transformer") if m["path"].as_str().unwrap_or("").is_empty() => {}
                (1, "Pooling") => pool_dir = m["path"].as_str().map(str::to_string),
                (2, "Normalize") => normalize = true,
                _ => return Err(err(format!("sentence-transformers module {i} '{ty}' is not supported"))),
            }
        }
        let pool_dir = pool_dir.ok_or_else(|| err("modules.json has no Pooling module".into()))?;
        let pc = read_json(&hub.fetch_file(repo_id, &format!("{pool_dir}/config.json"))?)?;
        // The encoder spec's pooling head: `mean` | `cls` | `max`, `_l2` when normalized.
        let pool = match pc["pooling_mode"].as_str() {
            Some(m @ ("mean" | "cls" | "max")) => m,
            Some(other) => return Err(err(format!("pooling_mode '{other}' is not supported"))),
            None => {
                let on = |k: &str| pc[k].as_bool() == Some(true);
                let modes = [
                    "pooling_mode_cls_token",
                    "pooling_mode_mean_tokens",
                    "pooling_mode_max_tokens",
                    "pooling_mode_mean_sqrt_len_tokens",
                    "pooling_mode_weightedmean_tokens",
                    "pooling_mode_lasttoken",
                ];
                match modes.map(on) {
                    [true, false, false, false, false, false] => "cls",
                    [false, true, false, false, false, false] => "mean",
                    [false, false, true, false, false, false] => "max",
                    _ => return Err(err(format!("pooling config {pc} is not supported"))),
                }
            }
        };
        let head_kind = if normalize { format!("{pool}_l2") } else { pool.to_string() };
        if family(&cfg, Some(&arch)).is_none() {
            return Err(err(format!("sentence-transformers body '{arch}' has no encoder family")));
        }
        let token = std::env::var("HF_TOKEN").ok();
        let art = hub.auto_download_and_cache(repo_id, token.as_deref())?;
        let enc = SpecEncoder::load(&art, None, mode, &head_kind).map_err(err)?;
        let hidden = enc.var("hidden")?;
        let sbert = hub
            .fetch_file(repo_id, "sentence_bert_config.json")
            .ok()
            .map(|p| read_json(&p))
            .transpose()?
            .unwrap_or(Value::Null);
        let max = enc.capacity()?;
        let capacity = sbert["max_seq_length"].as_u64().map_or(max, |m| (m as usize).min(max));
        let tok_cfg = hub
            .fetch_file(repo_id, "tokenizer_config.json")
            .ok()
            .map(|p| read_json(&p))
            .transpose()?
            .unwrap_or(Value::Null);
        let tok = enc.tokenizer(hub, repo_id, &tok_cfg, capacity)?;
        let pkl = std::fs::read(hub.fetch_file(repo_id, "model_head.pkl")?)
            .map_err(|e| err(format!("model_head.pkl: {e}")))?;
        let (head, classes) = logistic::read_head(&pkl, hidden).map_err(err)?;
        let names: Vec<String> = st["labels"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_string)).collect())
            .unwrap_or_default();
        let labels = classes
            .into_iter()
            .map(|c| match c {
                logistic::Class::Int(i) if !names.is_empty() => names
                    .get(i as usize)
                    .cloned()
                    .ok_or_else(|| err(format!("class {i} has no entry in config_setfit labels"))),
                logistic::Class::Int(i) => Ok(i.to_string()),
                logistic::Class::Str(s) => Ok(s),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let meta = Meta {
            architecture: format!("SetFit ({arch} + LogisticRegression)"),
            labels,
            activation: Activation::Identity,
        };
        Ok((meta, Self { tok, head: Head::Logistic(enc, head) }))
    }

    /// A GGUF classifier or reranker export at `path` on the encoder family of its
    /// architecture: labels from `<arch>.classifier.output_labels`; one logit is a raw score, more
    /// are softmaxed.
    pub fn from_gguf(path: &Path, mode: PlacementMode) -> Result<(Meta, Self), String> {
        let meta = crate::model::gguf_meta::GgufMeta::open(path)?;
        let arch = meta.arch().unwrap_or_default();
        let fam = gguf_family(&arch)
            .ok_or_else(|| format!("GGUF classifier architecture {arch} has no encoder family"))?;
        let artifacts = ModelArtifacts {
            model_id: path.display().to_string(),
            local_dir: path.parent().map(Into::into),
            weights_path: Some(path.to_path_buf()),
            is_cached: true,
            ..Default::default()
        };
        let enc = SpecEncoder::load(&artifacts, Some(&fam), mode, "classify")?;
        let n = enc.var("labels")?;
        let labels = match meta.strings(&format!("{arch}.classifier.output_labels")) {
            Some(l) if l.len() == n => l,
            _ => (0..n).map(|i| format!("LABEL_{i}")).collect(),
        };
        let tok = meta.tokenizer(enc.capacity()?)?;
        let m = Meta {
            architecture: format!("GGUF {arch} classifier"),
            activation: if n == 1 { Activation::Identity } else { Activation::Softmax },
            labels,
        };
        Ok((m, Self { tok, head: Head::Logits(enc) }))
    }

    /// Where the encoder runs.
    pub fn device(&self) -> Device {
        match &self.head {
            Head::Logits(e) | Head::Logistic(e, _) => e.device(),
        }
    }

    /// Scores before the activation (`predict_proba` for a logistic head) of `text` (with `pair`
    /// as the second segment), and the token count.
    pub fn logits(&self, text: &str, pair: Option<&str>) -> Result<(Vec<f32>, usize), String> {
        let e = encode(&self.tok, text, pair)?;
        let n = e.get_ids().len();
        Ok(match &self.head {
            Head::Logits(enc) => (enc.run(e.get_ids(), e.get_type_ids())?, n),
            Head::Logistic(_, _) if pair.is_some() => {
                return Err("SetFit classifiers take a single text, not a pair".into())
            }
            Head::Logistic(enc, head) => (head.proba(&enc.run(e.get_ids(), e.get_type_ids())?), n),
        })
    }
}

fn encode(tok: &tokenizers::Tokenizer, text: &str, pair: Option<&str>) -> Result<tokenizers::Encoding, String> {
    let e = match pair {
        Some(p) => tok.encode((text, p), true),
        None => tok.encode(text, true),
    }
    .map_err(|e| format!("tokenize: {e}"))?;
    if e.get_ids().is_empty() {
        return Err("tokenizer produced no tokens".into());
    }
    Ok(e)
}

pub(crate) fn read_json(p: &Path) -> Result<Value, String> {
    let bytes = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", p.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_offset_positions_skip_pads() {
        assert_eq!(positions(&[5, 6, 7], None), [0, 1, 2]);
        assert_eq!(positions(&[0, 9, 1, 8], Some(1)), [2, 3, 1, 4]);
    }

    #[test]
    fn encoder_families_claim_classifier_classes() {
        let a = classifier_architectures();
        for c in ["RobertaForSequenceClassification", "ModernBertForSequenceClassification", "DebertaV2ForSequenceClassification"] {
            assert!(a.iter().any(|x| x == c), "{c} unclaimed: {a:?}");
        }
        assert!(!a.iter().any(|x| x.starts_with("GPT2")));
        assert!(family(&serde_json::json!({"model_type": "gpt2"}), Some("GPT2ForSequenceClassification")).is_none());
    }
}
