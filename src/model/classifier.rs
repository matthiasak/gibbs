//! Hugging Face sequence classifiers (`text-classification` / `text-ranking` repos), one type for
//! the server and the TUI. Encoders (every family whose spec has `pipeline: encoder`, SetFit and
//! GGUF classifier exports) run on `runtime::encoder`; fastText models on `runtime::bag`;
//! decoder `*ForSequenceClassification` bodies are scored on their last non-pad token
//! (`runtime::seqcls` over the family spec).

#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
pub(crate) mod synth;

use crate::hardware::{Device, PlacementMode};
use crate::model::hf_loader::ChatMessage;
use crate::model::hub::{is_synthetic_model_id, split_weight_file, HuggingFaceHub};
use crate::model::loaders::{read_checkpoint, SafetensorsWeightReader, WeightReader};
use crate::model::nn;
use crate::model::UniversalModelLoader;
use crate::runtime::encoder::{self, is_generator, read_json, Activation, LabelScore};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

/// Decoder `architectures[0]` values loaded as seq-cls bodies.
fn decoder_architectures() -> Vec<String> {
    crate::arch::registry().map_or_else(|_| Vec::new(), |r| r.decoder_seqcls_architectures())
}

/// Whether `arch` (`architectures[0]`) loads as a sequence classifier.
pub fn runs_architecture(arch: &str) -> bool {
    decoder_architectures().iter().any(|a| a == arch)
        || encoder::classifier_architectures().iter().any(|a| a == arch)
}

/// The classifier the engine serves, shared by the HTTP server and the TUI. `Some` makes it the
/// active model: text generation answers 400 until an LLM is loaded again.
pub type ClassifierSlot = Arc<parking_lot::RwLock<Option<Arc<SequenceClassifier>>>>;

/// Separates (text, text_pair) in TUI prompts and `--prompt`: `query || passage`.
pub const PAIR_SEPARATOR: &str = "||";

/// `a || b` → (`a`, Some(`b`)), trimmed; text without the separator is a single input.
pub fn split_pair(s: &str) -> (&str, Option<&str>) {
    match s.split_once(PAIR_SEPARATOR) {
        Some((a, b)) => (a.trim(), Some(b.trim())),
        None => (s.trim(), None),
    }
}

/// A loaded sequence classifier.
pub struct SequenceClassifier {
    pub repo_id: String,
    pub architecture: String,
    pub labels: Vec<String>,
    pub activation: Activation,
    body: Body,
}

enum Body {
    /// An encoder family spec (`runtime::encoder`).
    Encoder(encoder::Classifier),
    /// Causal LM scored by `score` on the final hidden state of its last non-pad token.
    Decoder {
        llm: Box<dyn DecoderBackbone>,
        score: Lin,
        pad: Option<u32>,
        /// sentence-transformers CrossEncoder: a pair is `tokenizer(text, pair)` (the two id
        /// sequences concatenated), not a chat.
        concat_pair: bool,
        /// Metal: `llm` scores on the GPU (`score_gpu`); `score` stays for the label count.
        gpu: bool,
    },
    /// fastText supervised model on the bag runtime: `predict` already returns probabilities.
    FastText(crate::runtime::bag::BagClassifier),
}

/// A decoder body for `*ForSequenceClassification`: HF scores the final-norm hidden state of the
/// last non-pad token with a bias-free `score` layer.
pub trait DecoderBackbone: Send + Sync {
    fn tokenizer(&self) -> &crate::model::tokenizer::Tokenizer;
    fn context_len(&self) -> usize;
    fn hidden(&self) -> usize;
    /// Final-norm hidden state `[hidden]` of the last of `ids` (non-empty; one uncached pass).
    fn last_hidden(&self, ids: &[u32]) -> Result<Vec<f32>, String>;
    /// Metal: compile the score head into a GPU program (called once at load when the backend is
    /// Metal). `score` is `[labels, hidden]` row-major, bias-free.
    fn attach_score_gpu(&mut self, _score: &[f32], _labels: usize) -> Result<(), String> {
        Err(format!(
            "gpu-only: {} has no Metal seq-cls program; run with --backend cpu",
            std::any::type_name_of_val(self)
        ))
    }
    /// Logits `[labels]` of `ids` (non-empty, last non-pad token last) computed entirely on the
    /// GPU. Only valid after a successful `attach_score_gpu`.
    fn score_gpu(&self, _ids: &[u32]) -> Result<Vec<f32>, String> {
        Err("no GPU score program attached".into())
    }
    /// Token ids of `text` (with `pair`, the second segment): `Tokenizer::encode`, and
    /// [`encode_pair`] for a pair.
    fn encode(&self, text: &str, pair: Option<&str>) -> Result<Vec<u32>, String> {
        let tok = self.tokenizer();
        match pair {
            None => Ok(tok.encode(text)),
            Some(answer) => encode_pair(tok, text, answer),
        }
    }
}

/// A (text, answer) pair for a decoder: with no chat template, HF `tokenizer(text, pair)` (how
/// such reward models' cards score (prompt, response)); else the finished user/assistant chat
/// reward models score (HF `apply_chat_template(conv, tokenize=False)`: no generation prompt).
pub(crate) fn encode_pair(
    tok: &crate::model::tokenizer::Tokenizer,
    text: &str,
    answer: &str,
) -> Result<Vec<u32>, String> {
    let Some(spec) = &tok.chat_template else {
        let hf = tok
            .hf
            .as_ref()
            .ok_or("the model has no chat template or tokenizer.json")?;
        return Ok(hf
            .encode((text, answer), true)
            .map_err(|e| format!("tokenize: {e}"))?
            .get_ids()
            .to_vec());
    };
    let conv = [
        ChatMessage::new("user", text),
        ChatMessage::new("assistant", answer),
    ];
    let rendered = spec
        .render(&conv, None, false, &Value::Null)
        .map_err(|e| format!("chat template: {e}"))?;
    Ok(tok.encode(&rendered))
}

/// `architecture` of a fastText hierarchical-softmax re-export (`config.json` + `weights.pt`).
const FASTTEXT_EXPORT: &str = "fasttext_hierarchical_softmax";

/// `architectures[0]` of `repo_id` when it is a `*ForSequenceClassification` checkpoint (its
/// config.json from the cache, else fetched), or a PEFT LoRA `SEQ_CLS` adapter; `None` for
/// anything else.
pub fn detect(hub: &HuggingFaceHub, repo_id: &str) -> Option<String> {
    if is_synthetic_model_id(repo_id)
        || split_weight_file(repo_id).1.is_some()
        || Path::new(repo_id).is_dir()
    {
        return None;
    }
    let config = hub
        .locate_model(repo_id)
        .config_path
        .or_else(|| hub.fetch_file(repo_id, "config.json").ok());
    let cfg = config.as_deref().and_then(|p| read_json(p).ok());
    let arch = cfg.as_ref().and_then(|c| c["architectures"][0].as_str());
    if let Some(a) = arch.filter(|a| a.ends_with("ForSequenceClassification")) {
        return Some(a.to_string());
    }
    // Generators never carry a classifier; skip the adapter probe for them.
    if arch.is_some_and(is_generator) {
        return None;
    }
    // No config.json: a GGUF classifier/reranker export an encoder family claims, or fastText.
    if let Some(a) = config
        .is_none()
        .then(|| crate::model::gguf_meta::hub_arch(hub, repo_id))
        .flatten()
        .filter(|a| encoder::gguf_family(a).is_some())
    {
        return Some(format!("GGUF {a} classifier"));
    }
    if cfg
        .as_ref()
        .is_some_and(|c| c["architecture"] == FASTTEXT_EXPORT)
        || (config.is_none() && crate::model::fasttext::hub_file(hub, repo_id).is_some())
    {
        return Some("fastText".into());
    }
    if hub.fetch_file(repo_id, "config_setfit.json").is_ok() {
        return Some("SetFit".into());
    }
    let adapter = read_json(&hub.fetch_file(repo_id, "adapter_config.json").ok()?).ok()?;
    (adapter["task_type"] == "SEQ_CLS").then(|| "PEFT LoRA (SEQ_CLS)".to_string())
}

/// transformers `model_type`s whose base classes (`<Family>Model`, `...ForMaskedLM`) are plain
/// backbones; another class name over them is the repo's own code.
fn transformers_types() -> Vec<&'static str> {
    crate::arch::compat()["classifier"]["transformers_types"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect()
}

/// Picker verdict for a Hub search result `m` (`/api/models` JSON with config, library_name,
/// tags, siblings, gguf) that a classifier or a decoder tagged for classification decides:
/// `Some(Ok)` loads, `Some(Err(reason))` is refused for that reason, `None` leaves the repo to
/// the generic rules (a text-generation repo, an unrelated task).
pub fn hub_verdict(m: &Value) -> Option<Result<(), String>> {
    let cfg = &m["config"];
    let tag = m["pipeline_tag"].as_str().unwrap_or("");
    let lib = m["library_name"].as_str().unwrap_or("");
    let files: Vec<&str> = m["siblings"]
        .as_array()
        .map(|a| a.iter().filter_map(|s| s["rfilename"].as_str()).collect())
        .unwrap_or_default();
    let tags: Vec<&str> = m["tags"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let classifier_task = matches!(
        tag,
        "text-classification" | "text-ranking" | "zero-shot-classification" | "sentence-similarity"
    );
    let model_type = cfg["model_type"].as_str();
    let arch = cfg["architectures"][0].as_str();
    let custom_file = files.iter().find(|f| {
        let name = f.rsplit('/').next().unwrap_or(f);
        (name.starts_with("modeling_") && name.ends_with(".py"))
            || [".pt", ".pth", ".pkl", ".safetensors", ".json"]
                .iter()
                .any(|ext| {
                    name.strip_suffix(ext).is_some_and(|stem| {
                        stem.ends_with("head") || stem.ends_with("heads") || stem == "readout"
                    })
                })
    });

    // PEFT adapters: the API inlines adapter_config.json as `config.peft`.
    let peft = &cfg["peft"];
    if peft.is_object() || files.contains(&"adapter_config.json") {
        let base = peft["base_model_name_or_path"].as_str().unwrap_or("?");
        return Some(match peft["task_type"].as_str() {
            _ if custom_file.is_some() => Err(format!(
                "PEFT adapter on {base} with a custom head ({})",
                custom_file.unwrap()
            )),
            _ if base.starts_with('/') || base.starts_with('.') => {
                Err(format!("PEFT adapter whose base is a local path ({base})"))
            }
            // SEQ_CLS classifies; CAUSAL_LM generates from the merged base (crate::model::peft).
            Some("SEQ_CLS" | "CAUSAL_LM") | None => Ok(()),
            Some(t) => Err(format!("PEFT {t} adapter on {base} (not a classifier)")),
        });
    }
    if let Some(a) = arch {
        if runs_architecture(a) || files.contains(&"config_setfit.json") {
            return Some(Ok(()));
        }
        if is_generator(a) {
            // The LLM loaders: causal-LM family specs.
            let runs = model_type
                .is_some_and(|t| crate::arch::registry().is_ok_and(|r| r.runs_model_type(t)));
            return (runs && !cfg["auto_map"].is_object()).then_some(Ok(()));
        }
        if !classifier_task {
            return None;
        }
        if cfg["auto_map"].is_object() || custom_file.is_some_and(|f| f.ends_with(".py")) {
            return Some(Err(format!("custom architecture '{a}' (remote code)")));
        }
        if let Some(f) = custom_file {
            return Some(Err(format!(
                "custom architecture '{a}' + head '{f}' (remote code)"
            )));
        }
        // A transformers class is named after its model_type (`BertModel`, `ModernBertForMaskedLM`,
        // `BigBirdForSequenceClassification`); any other class comes from the repo's own code.
        let fold = |s: &str| s.to_lowercase().replace(['-', '_'], "");
        let seq_cls = a.ends_with("ForSequenceClassification");
        let named = model_type.is_some_and(|t| {
            fold(a).starts_with(&fold(t)) && (seq_cls || transformers_types().contains(&t))
        });
        return Some(Err(if !named {
            format!("custom architecture '{a}' (remote code)")
        } else if !seq_cls {
            format!("'{a}' has no sequence-classification head")
        } else {
            format!("architecture '{a}' is not supported")
        }));
    }
    if !classifier_task && lib != "fasttext" {
        return None;
    }
    // No transformers config: decide from the library, the tags and the files.
    if tags.contains(&"pytorch_model_hub_mixin") {
        return Some(Err(
            "custom PyTorch model (PyTorchModelHubMixin, remote code)".into(),
        ));
    }
    let gguf: Vec<&&str> = files.iter().filter(|f| f.ends_with(".gguf")).collect();
    Some(if !gguf.is_empty() {
        if gguf
            .iter()
            .all(|f| f.starts_with("loras/") || f.contains("lora"))
        {
            Err("GGUF LoRA adapter (needs its base model)".into())
        } else if let Some((a, only)) = m["gguf"]["architecture"].as_str().and_then(|a| {
            let enc = crate::arch::registry()
                .ok()?
                .pipeline_gguf_archs(crate::arch::Pipeline::Encoder);
            let only = enc.iter().find(|o| a.contains(o.as_str()) && a != o.as_str())?.clone();
            Some((a, only))
        }) {
            Err(format!(
                "GGUF encoder architecture '{a}' (only '{only}' classifiers load)"
            ))
        } else {
            // A decoder GGUF generates; a supported GGUF classifier loads in the classifier.
            Ok(())
        }
    } else if lib == "fasttext" || tags.contains(&"fasttext") {
        // A `.bin`/`.ftz` model, or the hierarchical-softmax re-export (config + vocab + weights.pt).
        let export = ["config.json", "vocab.json", "weights.pt"]
            .iter()
            .all(|f| files.contains(f));
        if export
            || files
                .iter()
                .any(|f| f.ends_with(".bin") || f.ends_with(".ftz"))
        {
            Ok(())
        } else {
            Err("fastText repo without a .bin/.ftz model file".into())
        }
    } else if lib.is_empty() && model_type.is_none() && files.iter().any(|f| f.ends_with(".py")) {
        Err("custom architecture (remote code, no config.json)".into())
    } else if !lib.is_empty() && lib != "transformers" {
        Err(format!("'{lib}' model (not a transformers checkpoint)"))
    } else if let Some(t) = model_type {
        Err(format!(
            "config.json names no architecture (model_type '{t}')"
        ))
    } else if tags.contains(&"onnx") || files.iter().any(|f| f.ends_with(".onnx")) {
        Err("ONNX export without config.json".into())
    } else {
        Err("no config.json (custom checkpoint)".into())
    })
}

/// `architectures[0]` of the config.json at `path` when it names a sequence classifier.
pub fn architecture_in(path: &Path) -> Option<String> {
    let cfg: Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    let arch = cfg["architectures"][0].as_str()?;
    arch.ends_with("ForSequenceClassification")
        .then(|| arch.to_string())
}

impl SequenceClassifier {
    /// Fetches config.json, the tokenizer files, and the weights of `repo_id`, and builds the
    /// classifier under `mode` (a device pins it).
    pub fn load(hub: &HuggingFaceHub, repo_id: &str, mode: impl Into<PlacementMode>) -> Result<Self, String> {
        let mode = mode.into();
        let device = crate::graph::avenues::resolved_backend(mode);
        let resolved = encoder::resolve_config(hub, repo_id);
        // SetFit: a sentence-transformers body + a scikit-learn head.
        if resolved.is_err() && hub.fetch_file(repo_id, "config_setfit.json").is_ok() {
            return Self::from_setfit(hub, repo_id, mode);
        }
        if resolved.is_err()
            && crate::model::gguf_meta::hub_arch(hub, repo_id).is_some_and(|a| encoder::gguf_family(&a).is_some())
        {
            let token = std::env::var("HF_TOKEN").ok();
            let art = hub.auto_download_and_cache(repo_id, token.as_deref())?;
            let path = art
                .weights_path
                .ok_or_else(|| format!("{repo_id}: no GGUF file"))?;
            return Self::from_gguf(repo_id, &path, mode);
        }
        if resolved.is_err() {
            let cfg = hub
                .fetch_file(repo_id, "config.json")
                .and_then(|p| read_json(&p));
            if cfg.is_ok_and(|c| c["architecture"] == FASTTEXT_EXPORT) {
                hub.fetch_file(repo_id, "vocab.json")?;
                let w = hub.fetch_file(repo_id, "weights.pt")?;
                return Self::from_fasttext(repo_id, w.parent().unwrap_or(Path::new(".")), mode);
            }
            if let Some(file) = crate::model::fasttext::hub_file(hub, repo_id) {
                return Self::from_fasttext(repo_id, &hub.fetch_file(repo_id, &file)?, mode);
            }
        }
        let r = resolved?;
        if encoder::family(&r.cfg, Some(&r.arch)).is_some() {
            let (m, c) = encoder::Classifier::from_hf(hub, repo_id, r, mode)?;
            return Ok(Self::encoder(repo_id, m, c));
        }
        let (cfg, arch, peft) = (r.cfg, r.arch, r.peft);
        if crate::arch::compat()["classifier"]["refused_seqcls_prefix"]
            .as_str()
            .is_some_and(|p| arch.starts_with(p))
        {
            return Err(format!(
                "{repo_id}: {arch} is not supported: mixture-of-experts decoder layers (routed expert MLPs) are not implemented"
            ));
        }
        let dec_archs = decoder_architectures();
        if !dec_archs.iter().any(|a| a == &arch) {
            if !cfg["auto_map"].is_null() {
                return Err(format!(
                    "{repo_id}: custom architecture '{arch}' (remote code)"
                ));
            }
            let mut all = encoder::classifier_architectures().to_vec();
            all.extend(dec_archs);
            return Err(format!("{repo_id}: {arch} is not supported (supported: {})", all.join(", ")));
        }
        let pad_id = cfg["pad_token_id"].as_u64().map(|v| v as u32);

        let body = {
            // No `cpu_only` gate: Metal runs the spec's score instance.
            if peft.is_some() {
                // ponytail: LoRA merges into the f32 encoder checkpoints only; decoder bodies load
                // through their own readers (merge there when a decoder SEQ_CLS adapter matters).
                return Err(format!(
                    "{repo_id}: PEFT LoRA adapters on decoder bases ({arch}) are not supported"
                ));
            }
            // Same token source as `fetch_file` (the encoder path).
            let token = std::env::var("HF_TOKEN").ok();
            let artifacts = hub.auto_download_and_cache(repo_id, token.as_deref())?;
            let weights = artifacts
                .weights_path
                .as_deref()
                .ok_or_else(|| format!("{repo_id}: no weight file"))?;
            let llm: Box<dyn DecoderBackbone> = match arch.as_str() {
                _ => Box::new(crate::runtime::seqcls::SpecBackbone::load(&artifacts, device)?),
            };
            let h = llm.hidden();
            let score = load_score(weights)?;
            if score.is_empty() || score.len() % h != 0 {
                return Err(format!(
                    "{repo_id}: score.weight has {} values, not [num_labels, {h}]",
                    score.len()
                ));
            }
            let score = Lin {
                out_f: score.len() / h,
                w: score,
                b: None,
                in_f: h,
            };
            // Metal: the score head and backbone become a GPU program; a backbone without one
            // fails the load (`gpu-only:`) instead of running on the CPU.
            let mut llm = llm;
            let gpu = device != Device::Cpu;
            if gpu {
                llm.attach_score_gpu(&score.w, score.out_f)?;
            }
            Body::Decoder {
                llm,
                score,
                pad: pad_id,
                concat_pair: cfg["sentence_transformers"].is_object(),
                gpu,
            }
        };
        let n = match &body {
            Body::Decoder { score, .. } => score.out_f,
            Body::Encoder(_) | Body::FastText(_) => unreachable!("encoder and fastText repos load above"),
        };
        let labels = (0..n)
            .map(|i| {
                cfg["id2label"][i.to_string()]
                    .as_str()
                    .map_or_else(|| format!("LABEL_{i}"), str::to_string)
            })
            .collect();
        Ok(Self {
            repo_id: repo_id.to_string(),
            architecture: arch,
            labels,
            activation: Activation::of(&cfg, n),
            body,
        })
    }

    fn encoder(repo_id: &str, m: encoder::Meta, c: encoder::Classifier) -> Self {
        Self {
            repo_id: repo_id.to_string(),
            architecture: m.architecture,
            labels: m.labels,
            activation: m.activation,
            body: Body::Encoder(c),
        }
    }

    /// A SetFit repo: a sentence-transformers encoder body + a scikit-learn LogisticRegression.
    pub fn from_setfit(hub: &HuggingFaceHub, repo_id: &str, mode: impl Into<PlacementMode>) -> Result<Self, String> {
        let (m, c) = encoder::Classifier::from_sentence_transformers(hub, repo_id, mode.into())?;
        Ok(Self::encoder(repo_id, m, c))
    }

    /// A fastText `.bin`/`.ftz` classifier, or (when `path` is a directory) a fastText
    /// hierarchical-softmax re-export (`config.json` + `vocab.json` + `weights.pt`).
    /// Its bag graph compiles under `mode`.
    pub fn from_fasttext(repo_id: &str, path: &Path, mode: impl Into<PlacementMode>) -> Result<Self, String> {
        let m = if path.is_dir() {
            crate::model::fasttext::load_export(path)?
        } else {
            crate::model::fasttext::load(path)?
        };
        let ft = crate::runtime::bag::BagClassifier::new("fasttext", m, mode)?;
        Ok(Self {
            repo_id: repo_id.to_string(),
            architecture: "fastText".into(),
            labels: ft.labels().to_vec(),
            activation: Activation::Identity,
            body: Body::FastText(ft),
        })
    }

    /// A GGUF classifier or reranker export at `path`.
    pub fn from_gguf(repo_id: &str, path: &Path, mode: impl Into<PlacementMode>) -> Result<Self, String> {
        let (m, c) = encoder::Classifier::from_gguf(path, mode.into())?;
        Ok(Self::encoder(repo_id, m, c))
    }

    /// Where the body runs.
    pub fn device(&self) -> Device {
        match &self.body {
            Body::Encoder(c) => c.device(),
            Body::FastText(b) => b.device(),
            Body::Decoder { gpu: true, .. } => Device::Metal,
            Body::Decoder { .. } => Device::Cpu,
        }
    }

    /// One line for logs and the TUI: architecture, labels, scoring, device.
    pub fn summary(&self) -> String {
        let scoring = match self.activation {
            Activation::Softmax => "softmax",
            Activation::Sigmoid => "sigmoid (multi-label)",
            Activation::Identity => "raw score",
        };
        format!(
            "{}, {} label{} ({scoring}), on {}",
            self.architecture,
            self.labels.len(),
            if self.labels.len() == 1 { "" } else { "s" },
            self.device()
        )
    }

    /// Scores `text` (with `pair` as the second segment: cross-encoders and rerankers take
    /// (query, passage); decoder classifiers render it as a user/assistant chat). Returns every
    /// label sorted by descending score, and the number of tokens the model read.
    pub fn classify(
        &self,
        text: &str,
        pair: Option<&str>,
    ) -> Result<(Vec<LabelScore>, usize), String> {
        let (logits, n) = match &self.body {
            Body::Encoder(c) => c.logits(text, pair)?,
            Body::FastText(_) if pair.is_some() => {
                return Err("fastText classifiers take a single text, not a pair".into())
            }
            Body::FastText(ft) => ft.predict(text)?,
            Body::Decoder {
                llm,
                score,
                pad,
                concat_pair,
                gpu,
            } => {
                let ids = match pair {
                    Some(b) if *concat_pair => {
                        let tok = llm.tokenizer();
                        [tok.encode(text), tok.encode(b)].concat()
                    }
                    _ => llm
                        .encode(text, pair)
                        .map_err(|e| format!("{}: {e}", self.repo_id))?,
                };
                if ids.len() > llm.context_len() {
                    return Err(format!(
                        "{} tokens exceed the {}-token context",
                        ids.len(),
                        llm.context_len()
                    ));
                }
                // HF scores the last token that is not the pad token.
                let last = ids
                    .iter()
                    .rposition(|&t| Some(t) != *pad)
                    .ok_or("the input has no non-pad token")?;
                let logits = if *gpu {
                    llm.score_gpu(&ids[..=last])?
                } else {
                    score.apply(&llm.last_hidden(&ids[..=last])?, 1)
                };
                (logits, ids.len())
            }
        };
        Ok((encoder::label_scores(&self.labels, self.activation, logits), n))
    }
}

/// `score.weight` of a decoder checkpoint, read while the f32 body is already resident, so only
/// that tensor is read: safetensors (one file, or the shard of `<file>.index.json` holding it) are
/// memory-mapped and touch just the header and the tensor's pages. Other formats (`.bin`, ...)
/// have no random access and are read whole.
fn load_score(weights: &Path) -> Result<Vec<f32>, String> {
    let name = weights.to_string_lossy();
    let shard = if name.ends_with(".safetensors.index.json") {
        let json = std::fs::read(weights).map_err(|e| format!("{}: {e}", weights.display()))?;
        let index = UniversalModelLoader::parse_safetensors_index(&json)?;
        let file = index
            .get("score.weight")
            .ok_or_else(|| format!("{}: no score.weight in weight_map", weights.display()))?;
        weights.parent().unwrap_or(Path::new(".")).join(file)
    } else if name.ends_with(".safetensors") {
        weights.to_path_buf()
    } else {
        let (manifest, data) = read_checkpoint(weights)?;
        return UniversalModelLoader::extract_tensor_from_manifest(
            &manifest,
            &data,
            "score.weight",
        );
    };
    SafetensorsWeightReader::new(&shard)
        .and_then(|r| r.load_tensor("score.weight"))
        .map_err(|e| format!("{}: {e}", shard.display()))
}

/// A linear layer in PyTorch layout `[out_f, in_f]`.
struct Lin {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
    in_f: usize,
    out_f: usize,
}

impl Lin {
    fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        nn::linear(x, rows, &self.w, self.b.as_deref(), self.in_f, self.out_f)
    }
}
