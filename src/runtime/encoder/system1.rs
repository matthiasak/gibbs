//! "System 1" decision engine: a pretrained sentence encoder (default
//! sentence-transformers/all-MiniLM-L6-v2: any encoder family whose spec names a `runtime.system1`
//! pooling head), or any LLM, answering structured questions zero-shot by cosine similarity
//! between embeddings, in a single forward pass.

use super::SpecEncoder;
#[cfg(test)]
use crate::hardware::Device;
use crate::model::hub::HuggingFaceHub;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An option in a categorical choice question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceOption {
    pub label: String,
    pub description: Option<String>,
}

impl ChoiceOption {
    pub fn new(label: impl Into<String>, description: Option<impl Into<String>>) -> Self {
        Self {
            label: label.into(),
            description: description.map(|d| d.into()),
        }
    }

    /// Text embedded for this option: "label: description", or the bare label.
    fn text(&self) -> String {
        match &self.description {
            Some(d) if !d.trim().is_empty() => format!("{}: {}", self.label, d),
            _ => self.label.clone(),
        }
    }
}

/// Structured questions accepted by the System 1 engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JevQuestion {
    Bool {
        id: String,
        instructions: String,
        criteria_true: Option<String>,
        criteria_false: Option<String>,
    },
    Choice {
        id: String,
        instructions: String,
        options: Vec<ChoiceOption>,
    },
    Score {
        id: String,
        instructions: String,
        criteria: Vec<String>,
    },
}

impl JevQuestion {
    pub fn id(&self) -> &str {
        match self {
            Self::Bool { id, .. } | Self::Choice { id, .. } | Self::Score { id, .. } => id,
        }
    }

    pub fn instructions(&self) -> &str {
        match self {
            Self::Bool { instructions, .. }
            | Self::Choice { instructions, .. }
            | Self::Score { instructions, .. } => instructions,
        }
    }
}

/// Decisions produced by the zero-shot encoder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JevDecision {
    /// P(true) and the resulting decision.
    Bool { bool_prob: f32, decision: bool },
    /// Top option with the full distribution over option labels.
    Choice {
        selected: String,
        probabilities: HashMap<String, f32>,
        confidence: f32,
    },
}

const BOOL_CRITERIA_ERR: &str =
    "bool needs criteria_true and criteria_false for the zero-shot encoder";
const SCORE_ERR: &str =
    "score questions need a trained head; not supported by the zero-shot encoder";

/// Shared, hot-swappable System 1 model slot (sentence encoder or any LLM).
pub type System1Slot = std::sync::Arc<parking_lot::RwLock<Option<std::sync::Arc<JevSystem1Model>>>>;

/// The family and embedding head of a sentence-encoder `config.json` (not a sequence classifier
/// or generator): an encoder family claims its `model_type` and names a System 1 head.
fn sentence_encoder(cfg: &serde_json::Value) -> Option<(String, String)> {
    let arch = cfg["architectures"][0].as_str();
    if arch.is_some_and(|a| a.ends_with("ForSequenceClassification") || super::is_generator(a)) {
        return None;
    }
    // By `model_type` alone: a bare class name (`BertModel`) says nothing about the config keys.
    let (family, rt) = super::family(cfg, None)?;
    Some((family, rt.system1()?.to_string()))
}

/// True when `cfg` is a sentence-encoder `config.json` System 1 runs.
pub fn is_sentence_encoder_config(cfg: &serde_json::Value) -> bool {
    sentence_encoder(cfg).is_some()
}

/// True when `repo_id` is a sentence-encoder checkpoint (`sentence-transformers/all-MiniLM-L6-v2`, etc.).
pub fn detect(hub: &HuggingFaceHub, repo_id: &str) -> bool {
    if crate::model::hub::is_synthetic_model_id(repo_id)
        || crate::model::hub::split_weight_file(repo_id).1.is_some()
        || std::path::Path::new(repo_id).is_dir()
        || repo_id.ends_with(".gguf")
        || repo_id.ends_with(".safetensors")
    {
        return false;
    }
    let Some(config) = hub
        .locate_model(repo_id)
        .config_path
        .or_else(|| hub.fetch_file(repo_id, "config.json").ok())
    else {
        return false;
    };
    let Ok(bytes) = std::fs::read(config) else {
        return false;
    };
    let Ok(cfg) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return false;
    };
    is_sentence_encoder_config(&cfg)
}

const LLM_EMBED_DIM: usize = 384;

/// Deterministic 384-d L2-normalized CountSketch of a centered LLM next-token logit vector.
fn llm_embed_from_logits(logits: &[f32]) -> Vec<f32> {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for &x in logits {
        if x.is_finite() {
            sum += x as f64;
            count += 1;
        }
    }
    let mean = if count > 0 {
        (sum / count as f64) as f32
    } else {
        0.0
    };
    let mut out = vec![0.0f32; LLM_EMBED_DIM];
    for (i, &x) in logits.iter().enumerate() {
        if !x.is_finite() {
            continue;
        }
        let h = (i as u64).wrapping_add(1).wrapping_mul(0x9E3779B97F4A7C15);
        let bucket = ((h >> 32) as usize) % LLM_EMBED_DIM;
        let sign = if (h & 1) == 0 { 1.0f32 } else { -1.0f32 };
        out[bucket] += sign * (x - mean);
    }
    let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
    for v in &mut out {
        *v /= norm;
    }
    out
}

/// Backing model for System 1 embeddings and zero-shot decisions.
pub enum System1Backend {
    /// An encoder spec pooled (and normalized) by its `runtime.system1` head in its graph.
    Encoder {
        tokenizer: tokenizers::Tokenizer,
        encoder: SpecEncoder,
    },
    Llm(crate::scheduler::ExecutableModel),
}

/// Pretrained sentence encoder or LLM used zero-shot for System 1 decisions.
pub struct JevSystem1Model {
    pub repo_id: String,
    /// Softmax temperature over cosine similarities.
    // ponytail: zero-shot similarity, not a trained head; calibrate T or train heads if accuracy matters.
    pub temperature: f32,
    pub backend: System1Backend,
}
impl JevSystem1Model {
    /// Fetches config.json, tokenizer.json and model.safetensors from `repo_id` and builds the
    /// encoder spec under `mode` (a device pins it).
    pub fn load(
        hub: &HuggingFaceHub,
        repo_id: &str,
        mode: impl Into<crate::hardware::PlacementMode>,
    ) -> Result<Self, String> {
        let config_path = hub.fetch_file(repo_id, "config.json")?;
        let tok_path = hub.fetch_file(repo_id, "tokenizer.json")?;
        let st_path = hub.fetch_file(repo_id, "model.safetensors")?;
        let cfg = super::read_json(&config_path)?;
        let (family, head) = sentence_encoder(&cfg)
            .ok_or_else(|| format!("{repo_id}: no encoder family with a System 1 head claims its config.json"))?;
        let artifacts = crate::model::hub::ModelArtifacts {
            model_id: repo_id.to_string(),
            local_dir: st_path.parent().map(Into::into),
            config_path: Some(config_path),
            tokenizer_path: Some(tok_path.clone()),
            weights_path: Some(st_path),
            is_cached: true,
            ..Default::default()
        };
        let encoder = SpecEncoder::load(&artifacts, Some(&family), mode, &head)
            .map_err(|e| format!("{repo_id}: {e}"))?;
        let mut tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| format!("load {}: {e}", tok_path.display()))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: encoder.capacity()?,
                ..Default::default()
            }))
            .map_err(|e| format!("tokenizer truncation: {e}"))?;
        tokenizer.with_padding(None);
        Ok(Self {
            repo_id: repo_id.to_string(),
            temperature: 0.05,
            backend: System1Backend::Encoder { tokenizer, encoder },
        })
    }

    /// Turns any loaded LLM (`ExecutableModel`) into a System 1 embedding and zero-shot decision model.
    pub fn from_llm(repo_id: impl Into<String>, llm: crate::scheduler::ExecutableModel) -> Self {
        Self {
            repo_id: repo_id.into(),
            temperature: 0.05,
            backend: System1Backend::Llm(llm),
        }
    }

    /// The sentence encoder, or `None` when backed by an LLM.
    pub fn encoder(&self) -> Option<&SpecEncoder> {
        match &self.backend {
            System1Backend::Encoder { encoder, .. } => Some(encoder),
            System1Backend::Llm(_) => None,
        }
    }

    /// True when this System 1 model wraps an LLM rather than a dedicated sentence encoder.
    pub fn is_llm(&self) -> bool {
        matches!(self.backend, System1Backend::Llm(_))
    }

    /// Human-readable summary for the TUI and startup logs.
    pub fn summary(&self) -> String {
        match &self.backend {
            System1Backend::Encoder { encoder: e, .. } => format!(
                "System 1 {} encoder ({} layers, hidden {}, {} heads, {:.1}M params) on {:?}; /decide <state> | <opt1>, <opt2> or /embed <text>",
                e.model.spec.family,
                e.var("layers").unwrap_or(0),
                e.var("hidden").unwrap_or(0),
                e.var("heads").unwrap_or(0),
                e.parameter_count() as f64 / 1e6,
                e.device()
            ),
            System1Backend::Llm(llm) => format!(
                "System 1 LLM encoder on {}; /decide <state> | <opt1>, <opt2> or /embed <text> (/system1 off for chat)",
                llm.backend_label()
            ),
        }
    }

    /// Number of tokens the model sees for `text`.
    pub fn token_count(&self, text: &str) -> Result<usize, String> {
        match &self.backend {
            System1Backend::Encoder { tokenizer, .. } => {
                let enc = tokenizer
                    .encode(text, true)
                    .map_err(|e| format!("tokenize: {e}"))?;
                Ok(enc.get_ids().len())
            }
            System1Backend::Llm(llm) => Ok(llm.tokenizer().encode(text).len().max(1)),
        }
    }

    /// Sentence embedding: L2-normalized vector (the encoder's pooled head or 384-d LLM logit sketch).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        match &self.backend {
            System1Backend::Encoder { tokenizer, encoder } => {
                let enc = tokenizer
                    .encode(text, true)
                    .map_err(|e| format!("tokenize: {e}"))?;
                encoder.run(enc.get_ids(), enc.get_type_ids())
            }
            System1Backend::Llm(llm) => {
                let mut ids = llm.tokenizer().encode(text);
                if ids.is_empty() {
                    ids = llm.tokenizer().encode(" ");
                    if ids.is_empty() {
                        ids.push(0);
                    }
                }
                let logits = llm.prompt_logits(ids)?;
                Ok(llm_embed_from_logits(&logits))
            }
        }
    }
    /// Zero-shot choice: softmax(cos(state, option) / T).
    pub fn evaluate_choice_embedded(
        &self,
        state: &[f32],
        options: &[ChoiceOption],
    ) -> Result<JevDecision, String> {
        if options.is_empty() {
            return Err("choice question has no options".into());
        }
        let cos = options
            .iter()
            .map(|o| Ok(dot(state, &self.embed(&o.text())?)))
            .collect::<Result<Vec<f32>, String>>()?;
        let probs = softmax_scaled(&cos, self.temperature);
        let best = (0..probs.len()).fold(0, |b, i| if probs[i] > probs[b] { i } else { b });
        Ok(JevDecision::Choice {
            selected: options[best].label.clone(),
            probabilities: options
                .iter()
                .map(|o| o.label.clone())
                .zip(probs.iter().copied())
                .collect(),
            confidence: probs[best],
        })
    }

    /// Zero-shot bool: softmax over [cos(state, true), cos(state, false)] / T.
    pub fn evaluate_bool_embedded(
        &self,
        state: &[f32],
        criteria_true: Option<&str>,
        criteria_false: Option<&str>,
    ) -> Result<JevDecision, String> {
        let (Some(t), Some(f)) = (
            criteria_true.filter(|s| !s.trim().is_empty()),
            criteria_false.filter(|s| !s.trim().is_empty()),
        ) else {
            return Err(BOOL_CRITERIA_ERR.into());
        };
        let cos = [dot(state, &self.embed(t)?), dot(state, &self.embed(f)?)];
        let p = softmax_scaled(&cos, self.temperature)[0];
        Ok(JevDecision::Bool {
            bool_prob: p,
            decision: p >= 0.5,
        })
    }

    /// Answers one question against a precomputed state embedding.
    pub fn evaluate_embedded(
        &self,
        state: &[f32],
        question: &JevQuestion,
    ) -> Result<JevDecision, String> {
        check_supported(question)?;
        match question {
            JevQuestion::Bool {
                criteria_true,
                criteria_false,
                ..
            } => self.evaluate_bool_embedded(
                state,
                criteria_true.as_deref(),
                criteria_false.as_deref(),
            ),
            JevQuestion::Choice { options, .. } => self.evaluate_choice_embedded(state, options),
            JevQuestion::Score { .. } => Err(SCORE_ERR.into()),
        }
    }

    /// Answers one question about `state`.
    pub fn evaluate(&self, state: &str, question: &JevQuestion) -> Result<JevDecision, String> {
        self.evaluate_embedded(&self.embed(state)?, question)
    }

    /// Answers every question with one shared state embedding. The outer error is a
    /// state-encoding failure; each question carries its own result.
    pub fn batch_evaluate(
        &self,
        state: &str,
        questions: &[JevQuestion],
    ) -> Result<Vec<(String, Result<JevDecision, String>)>, String> {
        let s = self.embed(state)?;
        Ok(questions
            .iter()
            .map(|q| (q.id().to_string(), self.evaluate_embedded(&s, q)))
            .collect())
    }
}

/// Rejects questions the zero-shot encoder cannot answer, before any forward pass.
fn check_supported(q: &JevQuestion) -> Result<(), String> {
    let present = |s: &Option<String>| s.as_deref().is_some_and(|s| !s.trim().is_empty());
    match q {
        JevQuestion::Bool {
            criteria_true,
            criteria_false,
            ..
        } if !(present(criteria_true) && present(criteria_false)) => Err(BOOL_CRITERIA_ERR.into()),
        JevQuestion::Choice { options, .. } if options.is_empty() => {
            Err("choice question has no options".into())
        }
        JevQuestion::Score { .. } => Err(SCORE_ERR.into()),
        _ => Ok(()),
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// softmax(x / t), max-subtracted.
fn softmax_scaled(x: &[f32], t: f32) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|v| ((v - m) / t).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_over_cosines_with_temperature() {
        let p = softmax_scaled(&[0.9, 0.8], 0.05);
        let expect = 1.0 / (1.0 + (-2.0f32).exp());
        assert!((p[0] - expect).abs() < 1e-6);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let flat = softmax_scaled(&[0.9, 0.8], 10.0);
        assert!(flat[0] < 0.51);
    }

    #[test]
    fn unsupported_questions_error() {
        let bool_q = |t: Option<&str>, f: Option<&str>| JevQuestion::Bool {
            id: "b".into(),
            instructions: String::new(),
            criteria_true: t.map(Into::into),
            criteria_false: f.map(Into::into),
        };
        assert_eq!(
            check_supported(&bool_q(Some("yes"), None)),
            Err(BOOL_CRITERIA_ERR.to_string())
        );
        assert_eq!(
            check_supported(&bool_q(None, None)),
            Err(BOOL_CRITERIA_ERR.to_string())
        );
        assert_eq!(check_supported(&bool_q(Some("yes"), Some("no"))), Ok(()));
        let score = JevQuestion::Score {
            id: "s".into(),
            instructions: String::new(),
            criteria: vec!["1".into()],
        };
        assert_eq!(check_supported(&score), Err(SCORE_ERR.to_string()));
    }

    const MINILM: &str = "sentence-transformers/all-MiniLM-L6-v2";

    /// The real all-MiniLM-L6-v2 encoder on `device`, or `None` when its weights are not cached in `disk/`.
    fn cached_minilm(device: Device) -> Option<JevSystem1Model> {
        let dir = std::path::Path::new("disk");
        let cached = dir
            .join(
                "models--sentence-transformers--all-MiniLM-L6-v2/snapshots/main/model.safetensors",
            )
            .exists();
        if !cached {
            eprintln!("SKIP: {MINILM} not cached in disk/");
            return None;
        }
        Some(
            JevSystem1Model::load(&HuggingFaceHub::with_cache_dir(dir.into()), MINILM, device)
                .unwrap(),
        )
    }

    /// n=2 (empty), one word, truncation at max positions, WordPiece [UNK]/non-ASCII, a sentence.
    fn edge_texts() -> [String; 5] {
        [
            String::new(),
            "hello".into(),
            "word ".repeat(600),
            "naïve café 東京 🚀🦀".into(),
            "Delete the user's account and all their personal data.".into(),
        ]
    }

    /// Pins the exact CPU output (f32 bits) of the real encoder, so any change to the CPU path that
    /// moves a bit fails here. Covers the edge texts and both decision kinds. Bits are
    /// host-specific (NEON kernels), hence aarch64 only.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn cpu_output_bits_pinned() {
        let Some(m) = cached_minilm(Device::Cpu) else {
            return;
        };
        let [empty, word, long, unicode, sentence] = edge_texts();
        let texts = [empty.as_str(), &word, &long, &unicode, &sentence];
        assert_eq!(m.token_count("").unwrap(), 2);
        assert_eq!(
            m.token_count(&long).unwrap(),
            m.encoder().unwrap().capacity().unwrap()
        );
        // FNV-1a over every output bit: stable across Rust releases, unlike DefaultHasher.
        let mut h: u64 = 0xcbf29ce484222325;
        let mut mix = |v: f32| {
            for b in v.to_bits().to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x100000001b3);
            }
        };
        let mut state = Vec::new();
        for t in texts {
            state = m.embed(t).unwrap();
            assert_eq!(state.len(), 384);
            state.iter().copied().for_each(&mut mix);
        }
        let JevDecision::Bool {
            bool_prob,
            decision,
        } = m
            .evaluate_bool_embedded(
                &state,
                Some("the request asks to delete data"),
                Some("the request keeps data"),
            )
            .unwrap()
        else {
            unreachable!()
        };
        mix(bool_prob);
        assert!(decision);
        let options: Vec<ChoiceOption> = ["privacy", "billing", "shipping"]
            .iter()
            .map(|l| ChoiceOption {
                label: l.to_string(),
                description: None,
            })
            .collect();
        let JevDecision::Choice {
            selected,
            probabilities,
            ..
        } = m.evaluate_choice_embedded(&state, &options).unwrap()
        else {
            unreachable!()
        };
        options.iter().for_each(|o| mix(probabilities[&o.label]));
        assert_eq!(selected, "privacy");
        assert_eq!(
            h, 7844549035239271037,
            "System 1 CPU output bits changed; re-pin only if the numeric change is intended"
        );
    }

    /// P1 gate: on Metal the real encoder matches the CPU (cosine >= 0.99999 on every edge text)
    /// and every decision on a fixed set picks the same answer.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_matches_cpu_embeddings_and_decisions() {
        let (Some(cpu), Some(gpu)) = (cached_minilm(Device::Cpu), cached_minilm(Device::Metal))
        else {
            return;
        };
        assert_eq!(gpu.encoder().unwrap().device(), Device::Metal);
        for t in edge_texts() {
            let (a, b) = (cpu.embed(&t).unwrap(), gpu.embed(&t).unwrap());
            let cos = dot(&a, &b) / (dot(&a, &a) * dot(&b, &b)).sqrt();
            assert!(cos >= 0.99999, "{t:?}: cosine {cos}");
        }
        let questions = [
            JevQuestion::Bool {
                id: "delete".into(),
                instructions: String::new(),
                criteria_true: Some("the user wants their account and data deleted".into()),
                criteria_false: Some("the user wants to keep their account and data".into()),
            },
            JevQuestion::Choice {
                id: "topic".into(),
                instructions: String::new(),
                options: ["billing", "privacy", "bug", "shipping"]
                    .map(|l| ChoiceOption::new(l, None::<&str>))
                    .to_vec(),
            },
        ];
        for state in [
            "Please remove my account and all my data permanently.",
            "I was charged twice for my subscription this month.",
            "The app crashes when I open the settings page.",
            "Where is my package? It has not arrived yet.",
            "",
        ] {
            let (a, b) = (
                cpu.batch_evaluate(state, &questions).unwrap(),
                gpu.batch_evaluate(state, &questions).unwrap(),
            );
            for ((id, x), (_, y)) in a.iter().zip(&b) {
                match (x.as_ref().unwrap(), y.as_ref().unwrap()) {
                    (
                        JevDecision::Bool {
                            bool_prob: p,
                            decision: d,
                        },
                        JevDecision::Bool {
                            bool_prob: q,
                            decision: e,
                        },
                    ) => {
                        assert_eq!(d, e, "{state:?} {id}");
                        assert!((p - q).abs() < 1e-3, "{state:?} {id}: {p} vs {q}");
                    }
                    (
                        JevDecision::Choice { selected: s, .. },
                        JevDecision::Choice { selected: t, .. },
                    ) => {
                        assert_eq!(s, t, "{state:?} {id}")
                    }
                    other => panic!("{state:?} {id}: decision kinds differ: {other:?}"),
                }
            }
        }
    }

    #[test]
    fn llm_acts_as_system1_model() {
        let dir = tempfile::tempdir().unwrap();
        let llm = crate::arch::testing::synthetic_lm(dir.path(), serde_json::json!({}), crate::hardware::Device::Cpu);
        let s1 = JevSystem1Model::from_llm("foundry/smollm-1m", llm);
        assert!(s1.is_llm());
        assert!(s1.token_count("reactor overheating").unwrap() >= 1);
        assert_eq!(s1.token_count("").unwrap(), 1);
        let emb = s1.embed("reactor overheating").unwrap();
        assert_eq!(emb.len(), LLM_EMBED_DIM);
        let norm = dot(&emb, &emb).sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
        let empty_emb = s1.embed("").unwrap();
        assert_eq!(empty_emb.len(), LLM_EMBED_DIM);
        let JevDecision::Choice {
            selected,
            probabilities,
            ..
        } = s1
            .evaluate_choice_embedded(
                &emb,
                &[
                    ChoiceOption::new("reactor overheating", None::<&str>),
                    ChoiceOption::new("billing invoice", None::<&str>),
                ],
            )
            .unwrap()
        else {
            unreachable!()
        };
        assert_eq!(selected, "reactor overheating");
        assert!(probabilities["reactor overheating"] > probabilities["billing invoice"]);
    }
}
