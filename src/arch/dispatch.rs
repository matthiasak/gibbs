//! The one model-load dispatch: a `--model` id to a loaded model of whatever kind, through its
//! family spec ([`crate::arch`]).

use super::expr::Val;
use super::Pipeline;
use crate::hardware::{Device, HardwareCapability, PlacementMode};
use crate::model::classifier::{self, SequenceClassifier};
use crate::model::hub::{HuggingFaceHub, ModelArtifacts};
use crate::runtime::encoder::system1::JevSystem1Model;
use crate::model::weight_format::{
    reencode_override, set_reencode_override, ReencodeConfig, ReencodePolicy,
};
use crate::runtime::diffusion::{Extras, ImageGenerator, SpecDiffusion};
use crate::scheduler::ExecutableModel;
use serde_json::{json, Map, Value as J};
use std::collections::HashSet;
use std::sync::Arc;

pub enum Loaded {
    /// A causal LM (or encoder-decoder).
    Lm {
        model: ExecutableModel,
        artifacts: ModelArtifacts,
    },
    Classifier(SequenceClassifier),
    System1(JevSystem1Model),
    Image(Arc<dyn ImageGenerator>),
    /// A text-to-video plan, loaded per generation ([`crate::model::video_repo::load`]).
    Video(crate::model::video_repo::Plan),
}

pub struct LoadOpts<'a> {
    pub placement: PlacementMode,
    pub token: Option<&'a str>,
    /// `--quant` verbatim: the GGUF pick of a video repo.
    pub quant_name: Option<&'a str>,
    pub hw: &'a HardwareCapability,
    /// The re-encode configuration every loader runs under.
    pub reencode: ReencodeConfig,
    /// `--arch`: load this family spec, skipping detection.
    pub family: Option<&'a str>,
    pub ctx: Option<usize>,
    /// Receives the stage the load is in, for progress text and failure records.
    pub stage: &'a dyn Fn(&'static str),
    /// Runs before any download of a repo no hub-level route took, with whether it is a PEFT
    /// adapter (merged into a snapshot of its base): an error refuses the load.
    pub admit: &'a dyn Fn(bool) -> Result<(), String>,
    /// The CLI's order: System 1 encoders are detected before
    /// classifiers and run on Metal wherever it exists; the TUI tries classifiers first and keeps
    /// System 1 on its own backend.
    pub system1_first: bool,
}

/// `model_type` is a causal-LM family's own match (not the decoder fallback): the hub picker loads
/// such repos whatever their pipeline tag (`image-text-to-text`, `any-to-any`, ...).
pub fn runs_spec_lm(model_type: &str) -> bool {
    super::registry().is_ok_and(|r| {
        r.matchers().iter().any(|(_, m, p)| {
            *p == Pipeline::CausalLm
                && !m.fallback
                && m.hf_model_type.iter().any(|g| super::glob(g, model_type))
        })
    })
}

/// GGUF `general.architecture` a causal-LM family spec matches: its blocks load as stored.
pub fn runs_spec_lm_gguf(arch: &str) -> bool {
    super::registry().is_ok_and(|r| {
        r.matchers().iter().any(|(_, m, p)| {
            *p == Pipeline::CausalLm && m.gguf_arch.iter().any(|g| super::glob(g, arch))
        })
    })
}

fn device(p: PlacementMode) -> Device {
    crate::graph::avenues::resolved_backend(p)
}

/// Resolves, downloads and loads `id`.
pub fn load_model(hub: &HuggingFaceHub, id: &str, o: &LoadOpts) -> Result<Loaded, String> {
    (o.stage)("resolving");
    if o.family.is_none() {
        // Encoders: System 1 and sequence classifiers run their family specs (`runtime::encoder`).
        let system1 = || {
            let mode = match device(o.placement) {
                // The CLI runs the System 1 encoder on Metal wherever it exists.
                Device::Cpu
                    if o.system1_first
                        && crate::hardware::available_devices().contains(&Device::Metal) =>
                {
                    PlacementMode::Pin(Device::Metal)
                }
                _ => o.placement,
            };
            JevSystem1Model::load(hub, id, mode).map(Loaded::System1)
        };
        if o.system1_first && crate::runtime::encoder::system1::detect(hub, id) {
            return system1();
        }
        if classifier::detect(hub, id).is_some() {
            return SequenceClassifier::load(hub, id, o.placement).map(Loaded::Classifier);
        }
        if !o.system1_first && crate::runtime::encoder::system1::detect(hub, id) {
            return system1();
        }
        if let Some(v) = crate::model::video_repo::detect(hub, id, o.quant_name) {
            return v.map(Loaded::Video);
        }
    }
    (o.stage)(if hub.locate_model(id).weights_path.is_some() {
        "reading cached files"
    } else {
        "downloading"
    });
    // A PEFT LoRA CAUSAL_LM adapter becomes a merged snapshot of its base first.
    let adapter = crate::model::peft::prepare_causal_lm(hub, id, o.token)?;
    (o.admit)(adapter)?;
    let artifacts = hub.auto_download_and_cache(id, o.token)?;
    (o.stage)("reading weights, choosing weight format");
    // Image repos are recognised by their files; `--arch` of a diffusion family loads as that one.
    match o.family {
        Some(f) if super::registry()?.spec(f, &Default::default()).is_ok_and(|s| s.pipeline == Pipeline::Diffusion) => {
            let p = SpecDiffusion::load(&artifacts, hub, o.placement, Some(f), &Extras::default())?;
            return Ok(Loaded::Image(Arc::new(p)));
        }
        Some(_) => {}
        None => {
            if let Some(img) = crate::runtime::diffusion::load_image(&artifacts, hub, o.placement) {
                return img.map(Loaded::Image);
            }
        }
    }
    let model = load_lm(&artifacts, o)?;
    Ok(Loaded::Lm { model, artifacts })
}

/// Loads `artifacts` as a causal LM (or encoder-decoder) under `o.reencode`.
pub fn load_lm(a: &ModelArtifacts, o: &LoadOpts) -> Result<ExecutableModel, String> {
    let previous = reencode_override();
    set_reencode_override(Some(o.reencode.clone()));
    let r = lm(a, o);
    set_reencode_override(previous);
    r
}

fn lm(a: &ModelArtifacts, o: &LoadOpts) -> Result<ExecutableModel, String> {
    let reg = super::registry()?;
    let ck = super::load::open(a)?;
    let family = match o.family {
        Some(f) => f.to_string(),
        None => reg.detect(&ck.sig).ok_or_else(|| {
            format!("{}: no architecture spec matches this checkpoint", a.model_id)
        })?,
    };
    (o.stage)("compiling + tuning kernels");
    let m = match reg.spec(&family, &ck.sig)?.pipeline {
        Pipeline::Encdec => crate::runtime::encdec::load(a, reg, Some(&family), o.placement)?,
        _ => crate::runtime::lm::load_with_ctx(a, reg, Some(&family), o.placement, o.ctx)?,
    };
    Ok(ExecutableModel::new(m))
}

/// `--arch-dump`: the merged spec, the resolved config vars, every raw tensor's canonical name
/// and the raw tensors no component instance reads, as JSON.
pub fn dump(artifacts: &ModelArtifacts, family: Option<&str>) -> Result<String, String> {
    let reg = super::registry()?;
    let exact = ReencodeConfig {
        policy: ReencodePolicy::BitExact,
        ..ReencodeConfig::effective()
    };
    let m = super::load::load_with(artifacts, reg, family, exact)?;
    let config: Map<String, J> = m
        .spec
        .config
        .keys()
        .map(|k| {
            let v = m
                .env
                .var(k)
                .map_or_else(|e| J::String(format!("error: {e}")), |v| v.to_json());
            (k.clone(), v)
        })
        .collect();
    let (mut used, mut unbuilt) = (HashSet::new(), Map::new());
    let vars = [("rows", Val::Num(1.0)), ("head", Val::Str("last".into()))];
    for name in m.spec.components.keys() {
        match super::build::build(&m.spec, name, &m.env, &m.store, &vars) {
            Ok(i) => used.extend(i.weights),
            Err(e) => {
                unbuilt.insert(name.clone(), J::String(e));
            }
        }
    }
    let out = json!({
        "family": m.spec.family,
        "spec": reg.merged(&m.spec.family, &m.ckpt.sig)?,
        "config": config,
        "tensors": m.renames,
        "unconsumed": m.store.unconsumed(&used, &m.ignore),
        "unbuilt_components": unbuilt,
    });
    serde_json::to_string_pretty(&out).map_err(|e| e.to_string())
}
