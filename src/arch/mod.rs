//! Model architectures as data. Each family is a JSON5 spec under `arch/families/` (embedded at
//! build time, see [`EMBEDDED`]); per-checkpoint patches live under `arch/overrides/`, and users
//! add or replace both under `<cache>/arch/{families,overrides}/`. [`load`] turns a matched spec
//! and a checkpoint into weights, [`build`] turns a spec component into a `graph::Graph`.
//!
//! Merge order: family <- its `extends` chain (parent first) <- matching embedded overrides (file
//! name order) <- user family file of the same name (replaces) <- matching user overrides. Merging is
//! RFC 7386 merge patch: objects merge, everything else (arrays included) replaces, `null` deletes.

pub mod build;
pub mod dispatch;
pub mod expr;
pub mod gen;
pub mod load;
pub mod names;
pub mod templates;
#[doc(hidden)]
pub mod testing;
#[cfg(test)]
mod tests_encoders_bert;
#[cfg(test)]
mod tests_encoders_bert_gguf;
#[cfg(test)]
mod tests_encoders_bert_variants;
#[cfg(test)]
mod tests_encoders_deberta;
#[cfg(test)]
mod tests_encoders_factorized;
#[cfg(test)]
mod tests_encoders_modernbert;

pub use dispatch::{load_model, LoadOpts, Loaded};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as J};
use std::path::Path;

/// Every embedded spec file: (path under `arch/`, contents).
pub const EMBEDDED: &[(&str, &str)] = &include!("../../arch/index.rs");

/// Tensor facts the interpreter needs: canonical weights and generated constants.
pub trait Source {
    /// Logical shape and graph dtype of a weight/constant.
    fn tensor(&self, name: &str) -> Option<(Vec<usize>, crate::graph::DType)>;
    /// The low-rank adapter registered for a `Linear` weight.
    fn lora(&self, _weight: &str) -> Option<Lora> {
        None
    }
}

/// A LoRA pair: `y += (x · Aᵀ · Bᵀ) · scale (+ bias_delta)`.
#[derive(Debug, Clone)]
pub struct Lora {
    pub a: String,
    pub b: String,
    pub scale: f32,
    pub bias_delta: Option<String>,
}

/// Which checkpoints a family claims.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Match {
    #[serde(default)]
    pub hf_model_type: Vec<String>,
    #[serde(default)]
    pub hf_architectures: Vec<String>,
    #[serde(default)]
    pub gguf_arch: Vec<String>,
    #[serde(default)]
    pub diffusers_class: Vec<String>,
    #[serde(default)]
    pub file_ext: Vec<String>,
    /// Claims causal LMs no other family matches; tried after every non-fallback family.
    #[serde(default)]
    pub fallback: bool,
    /// Tensor-name groups: a weights file holding every name of any group (names may carry a
    /// `model.diffusion_model.` prefix) is a derived checkpoint of this family's denoiser (a
    /// transformer-only finetune or quantization whose model card names the base pipeline).
    #[serde(default)]
    pub denoiser_tensors: Vec<Vec<String>>,
    /// Tensor-name groups of a single-file pipeline checkpoint (every component's weights in one
    /// file, e.g. LDM / ComfyUI checkpoints): its configs come from the spec's
    /// `runtime.reference` repo.
    #[serde(default)]
    pub single_file_tensors: Vec<Vec<String>>,
    /// File extensions `single_file_tensors` applies to (empty: any).
    #[serde(default)]
    pub single_file_ext: Vec<String>,
    /// Hub repos whose derivatives (finetunes, quantizations, LoRAs whose model card names one as
    /// `base_model`) this family runs; listings only (loading reads the card).
    #[serde(default)]
    pub bases: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Pipeline {
    CausalLm,
    Encoder,
    Encdec,
    Diffusion,
    Video,
    Drafter,
    BagClassifier,
    Unsupported,
}

/// A load refusal: `expr` true means the checkpoint is refused with `error` (interpolated).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Check {
    pub expr: String,
    pub error: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Names {
    /// Shared rule sets from `arch/names/<set>.json5`, applied after `rules`.
    #[serde(default, rename = "use")]
    pub use_sets: Vec<String>,
    #[serde(default)]
    pub rules: Vec<(String, String)>,
    /// Checkpoint-name prefixes removed before any rule (e.g. `model.diffusion_model.`).
    #[serde(default)]
    pub strip: Vec<String>,
}

/// A load-time value transform on a canonical tensor (`tensor` may hold `{i}` captures).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transform {
    pub tensor: String,
    pub op: String,
    #[serde(default)]
    pub args: J,
    #[serde(default)]
    pub when: Option<String>,
}

/// A generated weight tensor.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Constant {
    pub gen: String,
    #[serde(default)]
    pub args: J,
    #[serde(default)]
    pub when: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Macro {
    #[serde(default)]
    pub params: Vec<String>,
    pub body: Vec<J>,
    /// Scope name (or list of names) the macro's `out` binds to; default `out`.
    #[serde(default)]
    pub returns: J,
}

/// A graph input of a network.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputDecl {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<J>,
    #[serde(default)]
    pub when: Option<String>,
    /// Generator the runner fills this input with (see `gen`); absent = caller supplies it.
    #[serde(default)]
    pub gen: Option<String>,
    #[serde(default)]
    pub args: J,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Network {
    #[serde(default)]
    pub vars: Vec<String>,
    #[serde(default)]
    pub inputs: Vec<InputDecl>,
    pub body: Vec<J>,
    /// Names, or `{name, value?, when?}` objects (`value`: the scope value output as `name`).
    #[serde(default)]
    pub outputs: Vec<J>,
}

/// A component: an inline network, or another family's spec run on a checkpoint subfolder.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Component {
    Ref {
        family: String,
        #[serde(default)]
        dir: Option<String>,
        #[serde(default)]
        strip: Vec<String>,
        #[serde(default)]
        ignore: Vec<String>,
        #[serde(default)]
        patch: Option<J>,
    },
    Net(Network),
}

/// A fully merged family spec.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub family: String,
    #[serde(default)]
    pub extends: Option<String>,
    #[serde(default, rename = "match")]
    pub matcher: Match,
    pub pipeline: Pipeline,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub use_lib: Vec<String>,
    /// var -> expression; evaluated lazily on first use (order free, cycles are errors).
    #[serde(default)]
    pub config: Map<String, J>,
    #[serde(default)]
    pub checks: Vec<Check>,
    #[serde(default)]
    pub names: Names,
    #[serde(default)]
    pub transforms: Vec<Transform>,
    #[serde(default)]
    pub constants: Map<String, J>,
    /// Checkpoint tensors deliberately left unread: globs, or `{glob, when}` (only when the `when`
    /// expression holds; resolved into [`load::Model::ignore`] at load).
    #[serde(default)]
    pub ignore: Vec<Ignore>,
    #[serde(default)]
    pub macros: Map<String, J>,
    #[serde(default)]
    pub components: Map<String, J>,
    #[serde(default)]
    pub runtime: J,
    /// Random-init weights for synthetic model ids: name pattern -> `{shape: [expr..], over?:
    /// {var: list-expr}}`; `over` instantiates `{var}` in the pattern for every list item.
    #[serde(default)]
    pub synthetic: Map<String, J>,
}

/// A `Spec::ignore` entry.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Ignore {
    Glob(String),
    When { glob: String, when: String },
}

impl Spec {
    pub fn component(&self, name: &str) -> Result<Component, String> {
        let c = self
            .components
            .get(name)
            .ok_or_else(|| format!("{}: no component `{name}`", self.family))?;
        serde_json::from_value(c.clone())
            .map_err(|e| format!("{}: component `{name}`: {e}", self.family))
    }

    pub fn macro_def(&self, name: &str) -> Result<Macro, String> {
        let m = self
            .macros
            .get(name)
            .ok_or_else(|| format!("{}: no macro `{name}`", self.family))?;
        serde_json::from_value(m.clone())
            .map_err(|e| format!("{}: macro `{name}`: {e}", self.family))
    }

    /// `runtime` deserialized as a pipeline's own type.
    pub fn runtime_as<T: serde::de::DeserializeOwned>(&self) -> Result<T, String> {
        let rt = if self.runtime.is_null() {
            J::Object(Map::new())
        } else {
            self.runtime.clone()
        };
        serde_json::from_value(rt).map_err(|e| format!("{}: runtime: {e}", self.family))
    }

    /// An `unsupported` family's reason for `name`: the first `runtime.reasons` `[glob, reason]`
    /// pair matching it, else `reason`.
    pub fn unsupported_reason(&self, name: &str) -> String {
        self.runtime["reasons"]
            .as_array()
            .into_iter()
            .flatten()
            .find_map(|r| match (r[0].as_str(), r[1].as_str()) {
                (Some(g), Some(why)) if glob(g, name) => Some(why.replace("{name}", name)),
                _ => None,
            })
            .or_else(|| self.reason.clone())
            .unwrap_or_else(|| format!("family `{}` is not supported", self.family))
    }
}

/// An override file: a partial spec applied to checkpoints it names.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Override {
    pub applies_to: AppliesTo,
    pub patch: J,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppliesTo {
    #[serde(default, deserialize_with = "one_or_many")]
    pub hub_id: Vec<String>,
    #[serde(default, deserialize_with = "one_or_many")]
    pub family: Vec<String>,
    #[serde(default, deserialize_with = "one_or_many")]
    pub model_type: Vec<String>,
    #[serde(default, deserialize_with = "one_or_many")]
    pub hf_architectures: Vec<String>,
    #[serde(default, deserialize_with = "one_or_many")]
    pub gguf_arch: Vec<String>,
    #[serde(default, deserialize_with = "one_or_many")]
    pub diffusers_class: Vec<String>,
    #[serde(default, deserialize_with = "one_or_many")]
    pub file_name: Vec<String>,
}

fn one_or_many<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(String),
        Many(Vec<String>),
    }
    match OneOrMany::deserialize(d)? {
        OneOrMany::One(s) => Ok(vec![s]),
        OneOrMany::Many(v) => Ok(v),
    }
}

/// What a checkpoint says about itself, for matching.
#[derive(Debug, Clone, Default)]
pub struct Signals {
    pub hub_id: String,
    pub file_name: String,
    pub hf_model_type: Option<String>,
    pub hf_architectures: Vec<String>,
    pub gguf_arch: Option<String>,
    pub diffusers_class: Option<String>,
    pub causal_lm: bool,
}

impl Signals {
    fn ext(&self) -> &str {
        self.file_name.rsplit_once('.').map_or("", |(_, e)| e)
    }
}

/// RFC 7386 merge patch of `patch` into `target`, plus one extension: a key `k+` whose value is
/// an array appends it to the target's array `k` (so a child spec adds checks / transforms /
/// rules to its parent's instead of repeating them).
pub fn merge_patch(target: &mut J, patch: &J) {
    let J::Object(p) = patch else {
        *target = patch.clone();
        return;
    };
    if !target.is_object() {
        *target = J::Object(Map::new());
    }
    let t = target.as_object_mut().unwrap();
    for (k, v) in p {
        if let (Some(base), J::Array(more)) = (k.strip_suffix('+'), v) {
            match t.entry(base.to_string()).or_insert(J::Array(Vec::new())) {
                J::Array(a) => a.extend(more.iter().cloned()),
                other => *other = v.clone(),
            }
        } else if v.is_null() {
            t.remove(k);
        } else {
            merge_patch(t.entry(k.clone()).or_insert(J::Null), v);
        }
    }
}

/// Parses JSON5 `text` (named `what` in errors).
pub fn parse_json5(what: &str, text: &str) -> Result<J, String> {
    json5::from_str::<J>(text).map_err(|e| format!("{what}: {e}"))
}

/// `*` and `?` glob match (no path semantics).
pub fn glob(pat: &str, s: &str) -> bool {
    fn go(p: &[u8], s: &[u8]) -> bool {
        match p.split_first() {
            None => s.is_empty(),
            Some((b'*', rest)) => (0..=s.len()).any(|i| go(rest, &s[i..])),
            Some((b'?', rest)) => !s.is_empty() && go(rest, &s[1..]),
            Some((c, rest)) => s.first() == Some(c) && go(rest, &s[1..]),
        }
    }
    go(pat.as_bytes(), s.as_bytes())
}

/// Embedded and user spec files.
#[derive(Debug, Clone, Default)]
pub struct Registry {
    /// family name -> raw spec (unmerged), in registry order.
    families: Vec<(String, J)>,
    /// (file name, override), embedded first (file-name order), then user files (file-name order).
    overrides: Vec<(String, Override)>,
    lib: Vec<(String, J)>,
    names: Vec<(String, J)>,
}

impl Registry {
    /// The embedded specs plus the user's `<dir>/{families,overrides}/*.json5` (when `dir` exists).
    pub fn load(user_dir: Option<&Path>) -> Result<Self, String> {
        let files: Vec<(String, String)> = EMBEDDED
            .iter()
            .map(|(p, t)| (p.to_string(), t.to_string()))
            .collect();
        let mut user: Vec<(String, String)> = Vec::new();
        if let Some(dir) = user_dir {
            for sub in ["families", "overrides"] {
                let Ok(rd) = std::fs::read_dir(dir.join(sub)) else {
                    continue;
                };
                let mut paths: Vec<_> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
                paths.sort();
                for p in paths
                    .into_iter()
                    .filter(|p| p.extension().is_some_and(|e| e == "json5"))
                {
                    let rel = format!("{sub}/{}", p.file_name().unwrap().to_string_lossy());
                    match std::fs::read_to_string(&p) {
                        Ok(text) => user.push((rel, text)),
                        Err(err) => tracing::warn!("arch: skipping {}: {err}", p.display()),
                    }
                }
            }
        }
        Self::from_files(&files, &user)
    }

    /// Registry from (path, text) lists: `embedded` then `user` (user families replace embedded
    /// ones of the same `family`; user overrides apply after embedded ones).
    pub fn from_files(
        embedded: &[(String, String)],
        user: &[(String, String)],
    ) -> Result<Self, String> {
        let mut r = Registry::default();
        for (is_user, list) in [(false, embedded), (true, user)] {
            let mut raw_ovs: Vec<(String, String, Override)> = Vec::new();
            let mut user_fams: Vec<(String, String, Option<J>)> = Vec::new();
            for (path, text) in list {
                let j = match json5::from_str::<J>(text) {
                    Ok(j) => j,
                    Err(err) if is_user => {
                        tracing::warn!("arch: skipping {path}: {err}");
                        continue;
                    }
                    Err(err) => return Err(format!("{path}: {err}")),
                };
                let (dir, file) = path.rsplit_once('/').unwrap_or(("", path));
                match dir {
                    "families" => {
                        let Some(name) = j["family"].as_str().map(String::from) else {
                            if is_user {
                                tracing::warn!("arch: skipping {path}: missing `family`");
                                continue;
                            }
                            return Err(format!("{path}: missing `family`"));
                        };
                        if let Some(slot) = r.families.iter_mut().find(|(n, _)| *n == name) {
                            if !is_user {
                                return Err(format!("{path}: family `{name}` defined twice"));
                            }
                            let prev = std::mem::replace(&mut slot.1, j);
                            user_fams.push((path.clone(), name, Some(prev)));
                        } else {
                            r.families.push((name.clone(), j));
                            if is_user {
                                user_fams.push((path.clone(), name, None));
                            }
                        }
                    }
                    "overrides" => {
                        let items: Result<Vec<Override>, _> = match j {
                            J::Array(_) => serde_json::from_value(j),
                            J::Object(mut m) if m.contains_key("overrides") => {
                                serde_json::from_value(m.remove("overrides").unwrap())
                            }
                            other => serde_json::from_value::<Override>(other).map(|o| vec![o]),
                        };
                        match items {
                            Ok(list) => {
                                for (idx, ov) in list.into_iter().enumerate() {
                                    raw_ovs.push((path.clone(), format!("{file}#{idx:02}"), ov));
                                }
                            }
                            Err(err) if is_user => tracing::warn!("arch: skipping {path}: {err}"),
                            Err(err) => return Err(format!("{path}: {err}")),
                        }
                    }
                    "lib" => r.lib.push((file.trim_end_matches(".json5").to_string(), j)),
                    "names" => r
                        .names
                        .push((file.trim_end_matches(".json5").to_string(), j)),
                    _ => {}
                }
            }
            for (path, name, prev) in user_fams {
                if let Err(err) = r
                    .base(&name)
                    .and_then(|b| serde_json::from_value::<Spec>(b).map_err(|e| e.to_string()))
                {
                    tracing::warn!("arch: skipping {path}: {err}");
                    if let Some(old) = prev {
                        if let Some(slot) = r.families.iter_mut().find(|(n, _)| *n == name) {
                            slot.1 = old;
                        }
                    } else {
                        r.families.retain(|(n, _)| *n != name);
                    }
                }
            }
            let mut ovs: Vec<(String, Override)> = Vec::new();
            for (path, file, ov) in raw_ovs {
                if is_user {
                    let bad = ov.applies_to.family.iter().find_map(|f| match r.base(f) {
                        Ok(mut b) => {
                            merge_patch(&mut b, &ov.patch);
                            serde_json::from_value::<Spec>(b)
                                .err()
                                .map(|e| e.to_string())
                        }
                        Err(e) => Some(e),
                    });
                    if let Some(err) = bad {
                        tracing::warn!("arch: skipping {path}: {err}");
                        continue;
                    }
                }
                ovs.push((file, ov));
            }
            ovs.sort_by(|a, b| a.0.cmp(&b.0));
            r.overrides.extend(ovs);
        }
        Ok(r)
    }

    pub fn family_names(&self) -> impl Iterator<Item = &str> {
        self.families.iter().map(|(n, _)| n.as_str())
    }

    fn raw(&self, name: &str) -> Result<&J, String> {
        self.families
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, j)| j)
            .ok_or_else(|| format!("unknown family `{name}`"))
    }

    /// `name` with its `extends` chain merged (parent first), without overrides.
    pub fn base(&self, name: &str) -> Result<J, String> {
        self.merge_layers(name, &[])
    }

    fn merge_layers(&self, name: &str, patches: &[J]) -> Result<J, String> {
        let mut chain = vec![self.raw(name)?.clone()];
        while let Some(parent) = chain.last().unwrap()["extends"].as_str() {
            if chain.len() > 16 {
                return Err(format!("{name}: `extends` cycle"));
            }
            chain.push(self.raw(parent)?.clone());
        }
        let mut pre = J::Object(Map::new());
        for layer in chain.iter().rev() {
            merge_patch(&mut pre, layer);
        }
        for p in patches {
            merge_patch(&mut pre, p);
        }
        let mut merged = J::Object(Map::new());
        let mut macros = Map::new();
        for lib in pre["use_lib"].as_array().cloned().unwrap_or_default() {
            let lib = lib.as_str().ok_or("use_lib: expected names")?;
            let mut l = self
                .lib
                .iter()
                .find(|(n, _)| n == lib)
                .ok_or_else(|| format!("{name}: unknown lib `{lib}`"))?
                .1
                .clone();
            if let Some(o) = l.as_object_mut() {
                if let Some(J::Object(m)) = o.remove("macros") {
                    macros.extend(m);
                }
            }
            merge_patch(&mut merged, &l);
        }
        let root_idx = chain.len() - 1;
        for (idx, layer) in chain.iter().enumerate().rev() {
            if idx == root_idx {
                let mut root = layer.clone();
                if let Some(o) = root.as_object_mut() {
                    for k in ["checks", "ignore", "transforms"] {
                        if merged.get(k).is_some() {
                            if let Some(v) = o.remove(k) {
                                o.insert(format!("{k}+"), v);
                            }
                        }
                    }
                }
                merge_patch(&mut merged, &root);
            } else {
                merge_patch(&mut merged, layer);
            }
        }
        for p in patches {
            merge_patch(&mut merged, p);
        }
        if let Some(m) = merged["macros"].as_object() {
            macros.extend(m.clone());
        }
        merged["macros"] = J::Object(macros);
        // Propagate any `strip` prefixes declared by referenced `names/<set>.json5`.
        let sets: Vec<String> = merged["names"]["use"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        for set in sets {
            let s_strip = self.name_set_strip(&set)?;
            if !s_strip.is_empty() {
                let patch = serde_json::json!({"names": {"strip+": s_strip}});
                merge_patch(&mut merged, &patch);
            }
        }
        // The leaf's own name and match win; a parent's `match` must not claim the child's checkpoints.
        merged["family"] = J::String(name.to_string());
        merged["match"] = chain[0]
            .get("match")
            .cloned()
            .unwrap_or(J::Object(Map::new()));
        Ok(merged)
    }

    /// The merged spec of `family` for checkpoint `sig`: base + every applicable override.
    pub fn spec(&self, family: &str, sig: &Signals) -> Result<Spec, String> {
        serde_json::from_value(self.merged(family, sig)?)
            .map_err(|e| format!("family `{family}`: {e}"))
    }

    /// [`Self::spec`] as JSON (what `--arch-dump` prints). Supports `parent.component` references.
    pub fn merged(&self, family: &str, sig: &Signals) -> Result<J, String> {
        if let Some((parent, comp)) = family.split_once('.') {
            let p_spec = self.spec(parent, sig)?;
            let Component::Ref {
                family: ref_fam,
                strip,
                ignore,
                patch,
                ..
            } = p_spec.component(comp)?
            else {
                return Err(format!(
                    "{parent}: component `{comp}` is not a family reference"
                ));
            };
            let mut patches: Vec<J> = self
                .overrides
                .iter()
                .filter(|(_, ov)| applies(&ov.applies_to, &ref_fam, sig))
                .map(|(_, ov)| ov.patch.clone())
                .collect();
            if let Some(p) = patch {
                patches.push(p);
            }
            let mut j = self.merge_layers(&ref_fam, &patches)?;
            if !strip.is_empty() {
                let mut cur: Vec<J> = strip.into_iter().map(J::String).collect();
                if let Some(existing) = j["names"]["strip"].as_array() {
                    cur.extend(existing.iter().cloned());
                }
                if !j["names"].is_object() {
                    j["names"] = J::Object(Map::new());
                }
                j["names"]["strip"] = J::Array(cur);
            }
            if !ignore.is_empty() {
                let patch = serde_json::json!({"ignore+": ignore});
                merge_patch(&mut j, &patch);
            }
            return Ok(j);
        }
        let patches: Vec<J> = self
            .overrides
            .iter()
            .filter(|(_, ov)| applies(&ov.applies_to, family, sig))
            .map(|(_, ov)| ov.patch.clone())
            .collect();
        self.merge_layers(family, &patches)
    }

    /// Rename rules of a shared set.
    pub fn name_set(&self, set: &str) -> Result<Vec<(String, String)>, String> {
        let s = self
            .names
            .iter()
            .find(|(n, _)| n == set)
            .ok_or_else(|| format!("unknown name set `{set}`"))?;
        serde_json::from_value(s.1["rules"].clone()).map_err(|e| format!("names/{set}: {e}"))
    }

    /// Optional `strip` prefixes of a shared name set.
    pub fn name_set_strip(&self, set: &str) -> Result<Vec<String>, String> {
        let s = self
            .names
            .iter()
            .find(|(n, _)| n == set)
            .ok_or_else(|| format!("unknown name set `{set}`"))?;
        match s.1.get("strip") {
            Some(v) => {
                serde_json::from_value(v.clone()).map_err(|e| format!("names/{set}.strip: {e}"))
            }
            None => Ok(Vec::new()),
        }
    }

    /// The family claiming `sig`: first non-fallback match in registry order, then fallbacks.
    pub fn detect(&self, sig: &Signals) -> Option<String> {
        let matchers: Vec<(&str, Match)> = self
            .families
            .iter()
            .filter_map(|(n, j)| {
                Some((
                    n.as_str(),
                    serde_json::from_value(j.get("match")?.clone()).ok()?,
                ))
            })
            .collect();
        let any = |pats: &[String], s: &str| pats.iter().any(|p| glob(p, s));
        let hit = |m: &Match| {
            sig.hf_model_type
                .as_ref()
                .is_some_and(|t| any(&m.hf_model_type, t))
                || sig
                    .hf_architectures
                    .iter()
                    .any(|a| any(&m.hf_architectures, a))
                || sig.gguf_arch.as_ref().is_some_and(|a| any(&m.gguf_arch, a))
                || sig
                    .diffusers_class
                    .as_ref()
                    .is_some_and(|c| any(&m.diffusers_class, c))
                || (!sig.ext().is_empty() && m.file_ext.iter().any(|e| e == sig.ext()))
        };
        matchers
            .iter()
            .find(|(_, m)| !m.fallback && hit(m))
            .or_else(|| {
                matchers
                    .iter()
                    .find(|(_, m)| m.fallback && (hit(m) || sig.causal_lm))
            })
            .map(|(n, _)| n.to_string())
    }

    /// The family whose `match.denoiser_tensors` claims a weights file with tensor `names`.
    pub fn detect_denoiser(&self, names: &[String]) -> Option<String> {
        self.detect_tensors(names, |m| &m.denoiser_tensors, true)
    }

    /// The family whose `match.single_file_tensors` (and `single_file_ext`) claim a weights file
    /// with extension `ext` and tensor `names`.
    pub fn detect_single_file(&self, names: &[String], ext: &str) -> Option<String> {
        self.detect_tensors(
            names,
            |m| {
                if m.single_file_ext.is_empty() || m.single_file_ext.iter().any(|e| e == ext) {
                    &m.single_file_tensors
                } else {
                    &m.denoiser_tensors[..0]
                }
            },
            false,
        )
    }

    fn detect_tensors(
        &self,
        names: &[String],
        groups: impl Fn(&Match) -> &[Vec<String>],
        unprefixed: bool,
    ) -> Option<String> {
        let has = |n: &str| {
            names.iter().any(|x| {
                x == n || (unprefixed && x.strip_prefix("model.diffusion_model.") == Some(n))
            })
        };
        self.families.iter().find_map(|(f, j)| {
            let m: Match = serde_json::from_value(j.get("match")?.clone()).ok()?;
            groups(&m)
                .iter()
                .any(|g| !g.is_empty() && g.iter().all(|n| has(n)))
                .then(|| f.clone())
        })
    }

    /// Every family's matcher, for listings (hub picker, README).
    pub fn matchers(&self) -> Vec<(String, Match, Pipeline)> {
        self.families
            .iter()
            .filter_map(|(n, j)| {
                let m = serde_json::from_value(
                    j.get("match").cloned().unwrap_or(J::Object(Map::new())),
                )
                .ok()?;
                let p = serde_json::from_value(self.base(n).ok()?["pipeline"].clone()).ok()?;
                Some((n.clone(), m, p))
            })
            .collect()
    }

    /// Why `name` (a `model_type` or GGUF architecture) is refused: the reason of the first
    /// `unsupported` family whose `hf_model_type` / `gguf_arch` globs match it.
    pub fn unsupported(&self, name: &str) -> Option<String> {
        self.families.iter().find_map(|(f, j)| {
            let m: Match = serde_json::from_value(j.get("match")?.clone()).ok()?;
            let hit = m
                .hf_model_type
                .iter()
                .chain(&m.gguf_arch)
                .chain(&m.diffusers_class)
                .any(|p| glob(p, name));
            let spec = self.spec(f, &Signals::default()).ok()?;
            (hit && spec.pipeline == Pipeline::Unsupported).then(|| spec.unsupported_reason(name))
        })
    }

    /// Whether a `causal_lm` family lists `model_type` (glob) in its `hf_model_type`.
    pub fn runs_model_type(&self, model_type: &str) -> bool {
        self.matchers().iter().any(|(_, m, p)| {
            matches!(p, Pipeline::CausalLm | Pipeline::Encdec)
                && m.hf_model_type.iter().any(|g| glob(g, model_type))
        })
    }

    /// The GGUF architectures `causal_lm` families claim.
    pub fn causal_lm_gguf_archs(&self) -> Vec<String> {
        self.pipeline_gguf_archs(Pipeline::CausalLm)
    }

    /// The GGUF architectures the families of `pipeline` claim.
    pub fn pipeline_gguf_archs(&self, pipeline: Pipeline) -> Vec<String> {
        self.matchers()
            .into_iter()
            .filter(|(_, _, p)| *p == pipeline)
            .flat_map(|(_, m, _)| m.gguf_arch)
            .collect()
    }

    /// The `match.bases` of the families of `pipeline`.
    pub fn derived_bases(&self, pipeline: Pipeline) -> Vec<String> {
        self.matchers()
            .into_iter()
            .filter(|(_, _, p)| *p == pipeline)
            .flat_map(|(_, m, _)| m.bases)
            .collect()
    }

    /// Whether any non-unsupported, non-drafter family claims `arch` in `gguf_arch`.
    pub fn runs_gguf_arch(&self, arch: &str) -> bool {
        self.matchers().iter().any(|(_, m, p)| {
            !matches!(p, Pipeline::Unsupported | Pipeline::Drafter)
                && m.gguf_arch.iter().any(|g| glob(g, arch))
        })
    }

    /// The first `gguf_arch` of the fallback causal-LM family.
    pub fn fallback_gguf_arch(&self) -> Option<String> {
        self.matchers()
            .into_iter()
            .find(|(_, m, p)| *p == Pipeline::CausalLm && m.fallback)
            .and_then(|(_, mut m, _)| (!m.gguf_arch.is_empty()).then(|| m.gguf_arch.swap_remove(0)))
    }

    /// `runtime` of the fallback causal-LM family.
    pub fn fallback_runtime(&self) -> Option<J> {
        let (f, _, _) = self
            .matchers()
            .into_iter()
            .find(|(_, m, p)| *p == Pipeline::CausalLm && m.fallback)?;
        Some(self.base(&f).ok()?["runtime"].clone())
    }

    /// `runtime.ninfer` of the family defining it.
    pub fn ninfer_runtime(&self) -> Option<J> {
        self.families.iter().find_map(|(f, _)| {
            let v = &self.base(f).ok()?["runtime"]["ninfer"];
            (!v.is_null()).then(|| v.clone())
        })
    }

    /// Whether a family claiming `gguf_arch == arch` sets `runtime.mmproj: true`.
    pub fn gguf_uses_mmproj(&self, arch: &str) -> bool {
        self.matchers().into_iter().any(|(f, m, _)| {
            m.gguf_arch.iter().any(|g| glob(g, arch))
                && self
                    .base(&f)
                    .ok()
                    .is_some_and(|b| b["runtime"]["mmproj"] == true)
        })
    }

    /// Whether an encoder family with `runtime.system1` claims `model_type`.
    pub fn runs_system1(&self, model_type: &str) -> bool {
        self.matchers().into_iter().any(|(f, m, p)| {
            p == Pipeline::Encoder
                && m.hf_model_type.iter().any(|g| glob(g, model_type))
                && self
                    .base(&f)
                    .ok()
                    .is_some_and(|b| !b["runtime"]["system1"].is_null())
        })
    }

    /// Decoder `*ForSequenceClassification` architectures claimed by `causal_lm` families.
    pub fn decoder_seqcls_architectures(&self) -> Vec<String> {
        self.matchers()
            .into_iter()
            .filter(|(_, _, p)| *p == Pipeline::CausalLm)
            .flat_map(|(_, m, _)| m.hf_architectures)
            .filter(|a| a.ends_with("ForSequenceClassification"))
            .collect()
    }

    /// Shared protocol/tokenizer/picker metadata (`arch/lib/compat.json5`).
    pub fn compat(&self) -> Option<&J> {
        self.lib.iter().find(|(n, _)| n == "compat").map(|(_, j)| j)
    }

    /// Markdown table of supported model families generated from `arch/families/*.json5` matches.
    pub fn supported_models_markdown(&self) -> String {
        let mut rows = vec![
            "| Family | Pipeline | Matches |".to_string(),
            "|:---|:---|:---|".to_string(),
        ];
        for (f, m, p) in self.matchers() {
            if p == Pipeline::Unsupported {
                continue;
            }
            let mut parts = Vec::new();
            let fmt_list = |label: &str, items: &[String]| {
                (!items.is_empty()).then(|| format!("{label}: `{}`", items.join("`, `")))
            };
            parts.extend(fmt_list("model_type", &m.hf_model_type));
            parts.extend(fmt_list("arch", &m.hf_architectures));
            parts.extend(fmt_list("gguf", &m.gguf_arch));
            parts.extend(fmt_list("diffusers", &m.diffusers_class));
            parts.extend(fmt_list("ext", &m.file_ext));
            if !m.single_file_tensors.is_empty() {
                parts.push("single-file".into());
            }
            if m.fallback {
                parts.push("fallback".into());
            }
            if parts.is_empty() {
                continue;
            }
            let pipe = serde_json::to_value(p)
                .ok()
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_default();
            rows.push(format!("| `{f}` | `{pipe}` | {} |", parts.join("; ")));
        }
        rows.join("\n")
    }
}

fn applies(a: &AppliesTo, family: &str, sig: &Signals) -> bool {
    let ci = |g: &str, s: &str| glob(&g.to_ascii_lowercase(), &s.to_ascii_lowercase());
    let opt = |pats: &[String], v: Option<&str>| {
        pats.is_empty() || v.is_some_and(|s| pats.iter().any(|g| ci(g, s)))
    };
    (a.hub_id.is_empty() || a.hub_id.iter().any(|g| ci(g, &sig.hub_id)))
        && (a.family.is_empty() || a.family.iter().any(|g| ci(g, family)))
        && opt(&a.model_type, sig.hf_model_type.as_deref())
        && (a.hf_architectures.is_empty()
            || a.hf_architectures
                .iter()
                .any(|g| sig.hf_architectures.iter().any(|s| ci(g, s))))
        && opt(&a.gguf_arch, sig.gguf_arch.as_deref())
        && opt(&a.diffusers_class, sig.diffusers_class.as_deref())
        && (a.file_name.is_empty() || a.file_name.iter().any(|g| ci(g, &sig.file_name)))
}

/// The embedded specs plus the user's `<cache>/arch/{families,overrides}`, read once per process.
pub fn registry() -> Result<&'static Registry, String> {
    static REG: std::sync::LazyLock<Result<Registry, String>> = std::sync::LazyLock::new(|| {
        let user = crate::hardware::default_cache_dir()
            .ok()
            .map(|d| d.join("arch"));
        Registry::load(user.as_deref())
    });
    REG.as_ref().map_err(Clone::clone)
}

/// [`Registry::unsupported`] over [`registry`].
pub fn unsupported(name: &str) -> Option<String> {
    registry().ok()?.unsupported(name)
}

/// Shared protocol/tokenizer/picker metadata (`arch/lib/compat.json5`).
pub fn compat() -> &'static J {
    static C: std::sync::LazyLock<J> = std::sync::LazyLock::new(|| {
        registry()
            .ok()
            .and_then(|r| r.compat().cloned())
            .unwrap_or(J::Null)
    });
    &C
}

/// Generic GGUF `tokenizer.ggml.model` kind from `arch/lib/compat.json5`.
pub fn gguf_tokenizer_model(model: &str) -> Option<&'static str> {
    compat()["gguf_tokenizers"]["models"][model].as_str()
}

/// `(normalizer, pre_tokenizer)` JSON of GGUF `tokenizer.ggml.pre` from `arch/lib/compat.json5`.
pub fn gguf_pre_tokenizer(pre: &str, byte_level: &J) -> Option<(J, J)> {
    let p = compat()["gguf_tokenizers"]["pre"].get(pre)?;
    let pre_tok = if p["digits_byte_level"] == true {
        serde_json::json!({
            "type": "Sequence",
            "pretokenizers": [{"type": "Digits", "individual_digits": true}, byte_level.clone()],
        })
    } else {
        p.get("pre_tokenizer")?.clone()
    };
    let norm = p.get("normalizer").cloned().unwrap_or(J::Null);
    Some((norm, pre_tok))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(list: &[(&str, &str)]) -> Vec<(String, String)> {
        list.iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect()
    }

    const BASE: &str = r#"{family: "base", match: {hf_model_type: ["base"]}, pipeline: "causal_lm",
        config: {a: "1", b: "2"}, runtime: {prefill_rows: 64, list: [1, 2]}}"#;
    const CHILD: &str = r#"{family: "child", extends: "base", match: {hf_model_type: ["child"]},
        config: {b: "3", c: "4"}, runtime: {list: [9]}}"#;
    const FALLBACK: &str = r#"{family: "fb", match: {fallback: true}, pipeline: "causal_lm"}"#;

    #[test]
    fn denoiser_tensor_groups_claim_weight_files() {
        let toy = r#"{family: "toy", pipeline: "diffusion",
            match: {denoiser_tensors: [["img_in.weight"], ["x.weight", "y.weight"]]}}"#;
        let r = Registry::from_files(
            &files(&[("families/toy.json5", toy), ("families/base.json5", BASE)]),
            &[],
        )
        .unwrap();
        let n = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            r.detect_denoiser(&n(&["model.diffusion_model.img_in.weight"]))
                .as_deref(),
            Some("toy")
        );
        assert_eq!(
            r.detect_denoiser(&n(&["x.weight", "y.weight", "z"]))
                .as_deref(),
            Some("toy")
        );
        assert_eq!(r.detect_denoiser(&n(&["x.weight"])), None);
        let single = r#"{family: "one", pipeline: "diffusion", match: {single_file_tensors: [["a.w", "b.w"]], single_file_ext: ["gguf"]}}"#;
        let r = Registry::from_files(&files(&[("families/one.json5", single)]), &[]).unwrap();
        assert_eq!(
            r.detect_single_file(&n(&["a.w", "b.w"]), "gguf").as_deref(),
            Some("one")
        );
        assert_eq!(
            r.detect_single_file(&n(&["a.w", "b.w"]), "safetensors"),
            None
        );
        assert_eq!(
            r.detect_single_file(&n(&["model.diffusion_model.a.w", "b.w"]), "gguf"),
            None
        );
    }

    /// SD single-file checkpoints claim their pipeline family by tensor names and extension; every
    /// SD pipeline / component family merges; turbo repo ids get 1-step defaults.
    #[test]
    fn sd_single_file_families() {
        let r = Registry::load(None).unwrap();
        let n = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let unet = "model.diffusion_model.input_blocks.0.0.weight";
        let sd1 = n(&[
            "cond_stage_model.transformer.text_model.final_layer_norm.weight",
            unet,
        ]);
        assert_eq!(
            r.detect_single_file(&sd1, "safetensors").as_deref(),
            Some("sd1_file")
        );
        assert_eq!(
            r.detect_single_file(&sd1, "gguf").as_deref(),
            Some("sd2_gguf")
        );
        let sd2 = n(&["cond_stage_model.model.ln_final.weight", unet]);
        assert_eq!(
            r.detect_single_file(&sd2, "safetensors").as_deref(),
            Some("sd2_file")
        );
        let xl = n(&[
            "conditioner.embedders.0.transformer.text_model.final_layer_norm.weight",
            "conditioner.embedders.1.model.ln_final.weight",
            unet,
        ]);
        assert_eq!(
            r.detect_single_file(&xl, "safetensors").as_deref(),
            Some("sdxl_file")
        );
        for f in [
            "sd",
            "sdxl",
            "sd1_file",
            "sd2_file",
            "sdxl_file",
            "sd2_gguf",
            "sd1_file.text_encoder",
            "sd2_file.text_encoder",
            "sdxl_file.text_encoder",
            "sdxl_file.text_encoder_2",
            "sd1_file.unet",
            "sd1_file.vae",
        ] {
            r.spec(f, &Signals::default())
                .unwrap_or_else(|e| panic!("{f}: {e}"));
        }
        let turbo = Signals {
            hub_id: "stabilityai/sdxl-turbo".into(),
            ..Default::default()
        };
        let s = r.spec("sdxl_file", &turbo).unwrap();
        assert_eq!(
            (
                s.runtime["defaults"]["steps"].as_u64(),
                s.runtime["defaults"]["guidance"].as_u64()
            ),
            (Some(1), Some(0))
        );
        assert_eq!(
            s.runtime["reference"],
            "'stabilityai/stable-diffusion-xl-base-1.0'"
        );
    }

    #[test]
    fn ldm_vae_names_map_to_diffusers() {
        let r =
            names::Renamer::from_pairs(&registry().unwrap().name_set("ldm_vae").unwrap()).unwrap();
        let d = |n: &str| r.canonical(&format!("first_stage_model.{n}"));
        assert_eq!(
            d("decoder.up.3.block.0.conv1.weight"),
            "decoder.up_ldm.3.resnets.0.conv1.weight"
        );
        assert_eq!(
            d("decoder.up.0.block.0.nin_shortcut.bias"),
            "decoder.up_ldm.0.resnets.0.conv_shortcut.bias"
        );
        assert_eq!(
            d("decoder.up.2.upsample.conv.weight"),
            "decoder.up_ldm.2.upsamplers.0.conv.weight"
        );
        assert_eq!(
            d("decoder.mid.block_2.norm1.weight"),
            "decoder.mid_block.resnets.1.norm1.weight"
        );
        assert_eq!(
            d("decoder.mid.attn_1.norm.bias"),
            "decoder.mid_block.attentions.0.group_norm.bias"
        );
        assert_eq!(
            d("encoder.mid.attn_1.proj_out.weight"),
            "encoder.mid_block.attentions.0.to_out.0.weight"
        );
        assert_eq!(
            d("decoder.mid.attn_1.q.weight"),
            "decoder.mid_block.attentions.0.to_q.weight"
        );
        assert_eq!(d("decoder.norm_out.weight"), "decoder.conv_norm_out.weight");
        assert_eq!(
            d("encoder.down.1.downsample.conv.bias"),
            "encoder.down_blocks.1.downsamplers.0.conv.bias"
        );
        assert_eq!(
            d("encoder.down.0.block.1.nin_shortcut.weight"),
            "encoder.down_blocks.0.resnets.1.conv_shortcut.weight"
        );
        assert_eq!(d("post_quant_conv.weight"), "post_quant_conv.weight");
        assert_eq!(d("decoder.conv_in.bias"), "decoder.conv_in.bias");
    }

    #[test]
    fn sana_runtime_keys_and_gemma2_encoder_claim() {
        let r = registry().unwrap();
        let s = r.spec("sana", &Signals::default()).unwrap();
        let rt: crate::runtime::diffusion::Runtime = s.runtime_as().unwrap();
        assert!(
            rt.other.is_empty(),
            "unknown runtime keys {:?}",
            rt.other.keys()
        );
        // A Gemma 2 causal LM detects as the encoder family: LM dispatch keeps it off the spec LM path.
        let sig = Signals {
            hf_model_type: Some("gemma2".into()),
            causal_lm: true,
            ..Default::default()
        };
        assert_eq!(r.detect(&sig).as_deref(), Some("gemma2"));
        assert_eq!(r.spec("gemma2", &sig).unwrap().pipeline, Pipeline::Encoder);
    }

    #[test]
    fn flux_families_runtime_keys_and_pipeline_claims() {
        let r = registry().unwrap();
        for (fam, class) in [
            ("flux", "FluxPipeline"),
            ("chroma", "ChromaPipeline"),
            ("flux2", "Flux2KleinPipeline"),
        ] {
            let s = r.spec(fam, &Signals::default()).unwrap();
            let rt: crate::runtime::diffusion::Runtime = s.runtime_as().unwrap();
            assert!(
                rt.other.is_empty(),
                "{fam}: unknown runtime keys {:?}",
                rt.other.keys()
            );
            assert!(
                s.components.get("text_encoder_2").is_none() == (fam != "flux"),
                "{fam}"
            );
            let sig = Signals {
                diffusers_class: Some(class.into()),
                ..Default::default()
            };
            assert_eq!(r.detect(&sig).as_deref(), Some(fam));
        }
        let names = names::Renamer::from_pairs(&r.name_set("bfl").unwrap()).unwrap();
        assert_eq!(
            names.canonical("double_blocks.3.txt_attn.qkv.bias"),
            "transformer_blocks.3.attn.to_added_qkv.bias"
        );
        assert_eq!(
            names.canonical("single_blocks.0.norm.key_norm.scale"),
            "single_transformer_blocks.0.attn.norm_k.weight"
        );
    }

    #[test]
    fn plus_key_appends_to_parent_array() {
        let mut t = serde_json::json!({"checks": [1, 2], "names": {"use": ["s"], "rules": [3]}});
        merge_patch(
            &mut t,
            &serde_json::json!({"checks+": [4], "names": {"rules+": [5]}, "new+": [6]}),
        );
        assert_eq!(
            t,
            serde_json::json!({"checks": [1, 2, 4], "names": {"use": ["s"], "rules": [3, 5]}, "new": [6]})
        );
    }

    #[test]
    fn extends_overrides_and_user_dir_merge_in_order() {
        let embedded = files(&[
            ("families/base.json5", BASE),
            ("families/child.json5", CHILD),
            ("families/fb.json5", FALLBACK),
            (
                "overrides/b.json5",
                r#"{applies_to: {family: "child"}, patch: {runtime: {prefill_rows: 32}, config: {c: "5"}}}"#,
            ),
            (
                "overrides/a.json5",
                r#"{applies_to: {family: "child"}, patch: {runtime: {prefill_rows: 16}}}"#,
            ),
            (
                "overrides/x.json5",
                r#"{applies_to: {hub_id: ["other/*"]}, patch: {config: {a: "99"}}}"#,
            ),
        ]);
        let user = files(&[(
            "overrides/u.json5",
            r#"{applies_to: {hub_id: ["org/*"]}, patch: {config: {a: null, d: "7"}}}"#,
        )]);
        let r = Registry::from_files(&embedded, &user).unwrap();
        let sig = Signals {
            hub_id: "org/m".into(),
            hf_model_type: Some("child".into()),
            ..Default::default()
        };
        assert_eq!(r.detect(&sig).as_deref(), Some("child"));
        let s = r.spec("child", &sig).unwrap();
        assert_eq!(s.pipeline, Pipeline::CausalLm, "inherited");
        assert_eq!(s.config.get("a"), None, "user override deletes");
        assert_eq!(s.config["b"], "3", "child wins over parent");
        assert_eq!(s.config["c"], "5", "override wins over family");
        assert_eq!(s.config["d"], "7");
        // Overrides apply in file-name order: a (16) then b (32).
        assert_eq!(s.runtime["prefill_rows"], 32);
        assert_eq!(s.runtime["list"], serde_json::json!([9]), "arrays replace");
        assert_eq!(s.matcher.hf_model_type, vec!["child".to_string()]);
    }

    #[test]
    fn user_family_replaces_embedded_and_fallback_is_last() {
        let embedded = files(&[
            ("families/base.json5", BASE),
            ("families/fb.json5", FALLBACK),
        ]);
        let user = files(&[(
            "families/base.json5",
            r#"{family: "base", match: {hf_model_type: ["base"]}, pipeline: "encoder"}"#,
        )]);
        let r = Registry::from_files(&embedded, &user).unwrap();
        let sig = Signals {
            hf_model_type: Some("base".into()),
            ..Default::default()
        };
        assert_eq!(r.spec("base", &sig).unwrap().pipeline, Pipeline::Encoder);
        let lm = Signals {
            hf_model_type: Some("unknown".into()),
            causal_lm: true,
            ..Default::default()
        };
        assert_eq!(r.detect(&lm).as_deref(), Some("fb"));
        let other = Signals {
            hf_model_type: Some("unknown".into()),
            ..Default::default()
        };
        assert_eq!(r.detect(&other), None);
    }

    #[test]
    fn globs() {
        assert!(glob("org/*-GGUF", "org/model-GGUF"));
        assert!(glob("*.gguf", "a.Q4_0.gguf"));
        assert!(!glob("org/?", "org/ab"));
    }

    #[test]
    fn lib_blocks_multi_family_and_robust_user_files() {
        let lib =
            r#"{config: {a: "0", from_lib: "1"}, ignore: ["ema.*"], macros: {m: {body: []}}}"#;
        let base = r#"{family: "base", match: {hf_model_type: ["base"]}, pipeline: "causal_lm",
            use_lib: ["shared"], config: {a: "1"}, "ignore+": ["extra.*"]}"#;
        let embedded = files(&[
            ("lib/shared.json5", lib),
            ("families/base.json5", base),
            ("families/child.json5", CHILD),
            (
                "overrides/multi.json5",
                r#"{applies_to: {family: ["base", "child"], hub_id: ["*TURBO*"], file_name: ["*.GGUF"]},
                    patch: {config: {turbo: "1"}}}"#,
            ),
            (
                "overrides/base_only.json5",
                r#"{applies_to: {family: "base"}, patch: {config: {only_base: "1"}}}"#,
            ),
        ]);
        let user = files(&[
            ("families/broken.json5", "{not json5"),
            (
                "families/bad_spec.json5",
                r#"{family: "bad", not_a_field: 1}"#,
            ),
            ("families/base.json5", r#"{family: "base", not_a_field: 1}"#),
            ("overrides/broken.json5", "{not json5"),
            (
                "overrides/bad_patch.json5",
                r#"{applies_to: {family: ["base"]}, patch: {not_a_spec_field: 1}}"#,
            ),
            (
                "overrides/good.json5",
                r#"{applies_to: {family: "child"}, patch: {config: {u: "ok"}}}"#,
            ),
        ]);
        let r = Registry::from_files(&embedded, &user).unwrap();
        let sig = Signals {
            hub_id: "Org/My-Turbo-Model".into(),
            file_name: "weights.gguf".into(),
            ..Default::default()
        };
        let b = r.spec("base", &sig).unwrap();
        assert!(r.family_names().all(|f| f != "bad"));
        assert_eq!(b.config["from_lib"], "1");
        assert_eq!(b.config["a"], "1", "family overrides lib");
        assert_eq!(
            b.config["turbo"], "1",
            "case-insensitive hub_id/file_name + multi-family"
        );
        assert_eq!(b.config["only_base"], "1");
        assert_eq!(b.ignore.len(), 2, "ignore+ appends to lib ignore");
        assert!(b.macros.contains_key("m"));
        let c = r.spec("child", &sig).unwrap();
        assert_eq!(
            c.config["from_lib"], "1",
            "child inherits parent use_lib blocks"
        );
        assert_eq!(c.config["turbo"], "1");
        assert_eq!(
            c.config.get("only_base"),
            None,
            "applies_to.family does not follow extends"
        );
        assert_eq!(c.config["u"], "ok");
    }

    #[test]
    fn all_arch_json5_files_are_embedded() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("arch");
        for dir in ["families", "lib", "names", "overrides"] {
            for e in std::fs::read_dir(root.join(dir)).unwrap() {
                let p = e.unwrap().path();
                if p.extension().is_some_and(|x| x == "json5") {
                    let key = format!("{dir}/{}", p.file_name().unwrap().to_string_lossy());
                    assert!(
                        EMBEDDED.iter().any(|(k, _)| *k == key),
                        "{key} is missing from arch/index.rs"
                    );
                }
            }
        }
    }

    // ---- LM family specs vs the deleted GraphLm / LnDecoder (frozen outputs) (fixtures as in `tests/golden/lm.rs`, `enc.rs`) ----

    use super::testing::{compare, Runner, Weights};
    use crate::hardware::Device;
    use serde_json::{json, Value as J};
    use std::path::Path;

    const V: usize = 256;
    const PROMPT: [u32; 6] = [5, 17, 42, 99, 133, 200];
    const DECODE: [u32; 3] = [7, 64, 250];

    struct Dims {
        h: usize,
        heads: usize,
        kv: usize,
        hd: usize,
        inter: usize,
        layers: usize,
    }

    fn lm_config(model_type: &str, arch: &str, d: &Dims, extra: J) -> J {
        let mut c = json!({
            "model_type": model_type, "architectures": [arch], "vocab_size": V,
            "hidden_size": d.h, "num_hidden_layers": d.layers, "num_attention_heads": d.heads,
            "num_key_value_heads": d.kv, "head_dim": d.hd, "intermediate_size": d.inter,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "hidden_act": "silu",
            "max_position_embeddings": 128, "bos_token_id": 1, "eos_token_id": 2,
            "tie_word_embeddings": true,
        });
        for (k, v) in extra.as_object().unwrap() {
            c[k] = v.clone();
        }
        c
    }

    /// `tests/golden/lm.rs` `llama_weights` (HF llama names).
    fn llama_weights(
        seed: u64,
        d: &Dims,
        qk_norm: bool,
        shared_mlp: bool,
        bitnet: bool,
        untied: bool,
    ) -> Weights {
        let mut w = Weights::new(seed);
        w.s("model.embed_tokens.weight", &[V, d.h], 0.05);
        w.norm("model.norm.weight", d.h, 1.0);
        if untied {
            w.w("lm_head.weight", &[V, d.h]);
        }
        let (a, kvd) = (d.heads * d.hd, d.kv * d.hd);
        let proj = |w: &mut Weights, name: String, shape: [usize; 2]| {
            if !bitnet {
                w.w(name, &shape);
                return;
            }
            let s = 1.0 / (shape[1] as f32).sqrt();
            let v = (0..shape[0] * shape[1])
                .map(|_| s * ((w.rng.next_u64() % 3) as f32 - 1.0))
                .collect();
            w.put(name, &shape, v);
        };
        for l in 0..d.layers {
            let p = format!("model.layers.{l}");
            w.norm(format!("{p}.input_layernorm.weight"), d.h, 1.0);
            w.norm(format!("{p}.post_attention_layernorm.weight"), d.h, 1.0);
            for (n, shape) in [
                ("q_proj", [a, d.h]),
                ("k_proj", [kvd, d.h]),
                ("v_proj", [kvd, d.h]),
                ("o_proj", [d.h, a]),
            ] {
                proj(&mut w, format!("{p}.self_attn.{n}.weight"), shape);
            }
            if qk_norm {
                w.norm(format!("{p}.self_attn.q_norm.weight"), d.hd, 1.0);
                w.norm(format!("{p}.self_attn.k_norm.weight"), d.hd, 1.0);
            }
            if shared_mlp {
                w.w(
                    format!("{p}.shared_mlp.input_linear.weight"),
                    &[2 * d.inter, d.h],
                );
                w.w(
                    format!("{p}.shared_mlp.output_linear.weight"),
                    &[d.h, d.inter],
                );
            } else {
                for (n, shape) in [
                    ("gate_proj", [d.inter, d.h]),
                    ("up_proj", [d.inter, d.h]),
                    ("down_proj", [d.h, d.inter]),
                ] {
                    proj(&mut w, format!("{p}.mlp.{n}.weight"), shape);
                }
            }
            if bitnet {
                w.norm(format!("{p}.self_attn.attn_sub_norm.weight"), a, 1.0);
                w.norm(format!("{p}.mlp.ffn_sub_norm.weight"), d.inter, 1.0);
            }
        }
        w
    }

    fn lfm2_fixture() -> (J, Weights) {
        let d = Dims {
            h: 32,
            heads: 4,
            kv: 2,
            hd: 8,
            inter: 64,
            layers: 3,
        };
        let (conv, k) = ([true, false, true], 3);
        let cfg = lm_config(
            "lfm2",
            "Lfm2ForCausalLM",
            &d,
            json!({
            "norm_eps": 1e-5, "conv_L_cache": k, "conv_bias": false,
            "block_auto_adjust_ff_dim": false, "rope_theta": 1000000.0,
            "layer_types": ["conv", "full_attention", "conv"]}),
        );
        let mut w = Weights::new(13);
        w.s("model.embed_tokens.weight", &[V, d.h], 0.05);
        w.norm("model.embedding_norm.weight", d.h, 1.0);
        for (l, &is_conv) in conv.iter().enumerate() {
            let p = format!("model.layers.{l}");
            w.norm(format!("{p}.operator_norm.weight"), d.h, 1.0);
            w.norm(format!("{p}.ffn_norm.weight"), d.h, 1.0);
            if is_conv {
                w.w(format!("{p}.conv.in_proj.weight"), &[3 * d.h, d.h]);
                w.w(format!("{p}.conv.conv.weight"), &[d.h, 1, k]);
                w.w(format!("{p}.conv.out_proj.weight"), &[d.h, d.h]);
            } else {
                let (a, kvd) = (d.heads * d.hd, d.kv * d.hd);
                w.w(format!("{p}.self_attn.q_proj.weight"), &[a, d.h]);
                w.w(format!("{p}.self_attn.k_proj.weight"), &[kvd, d.h]);
                w.w(format!("{p}.self_attn.v_proj.weight"), &[kvd, d.h]);
                w.w(format!("{p}.self_attn.out_proj.weight"), &[d.h, a]);
                w.norm(format!("{p}.self_attn.q_layernorm.weight"), d.hd, 1.0);
                w.norm(format!("{p}.self_attn.k_layernorm.weight"), d.hd, 1.0);
            }
            w.w(format!("{p}.feed_forward.w1.weight"), &[d.inter, d.h]);
            w.w(format!("{p}.feed_forward.w3.weight"), &[d.inter, d.h]);
            w.w(format!("{p}.feed_forward.w2.weight"), &[d.h, d.inter]);
        }
        (cfg, w)
    }

    /// GPT-NeoX: fused per-head query_key_value, partial RoPE (rotary_pct 0.25), parallel residual.
    fn neox_fixture() -> (J, Weights) {
        let (h, heads, inter, layers) = (32, 2, 64, 2);
        let cfg = json!({"model_type": "gpt_neox", "architectures": ["GPTNeoXForCausalLM"], "vocab_size": V,
            "hidden_size": h, "num_hidden_layers": layers, "num_attention_heads": heads, "intermediate_size": inter,
            "rotary_pct": 0.25, "rope_theta": 10000.0, "layer_norm_eps": 1e-5, "hidden_act": "gelu_new",
            "use_parallel_residual": true, "max_position_embeddings": 128, "tie_word_embeddings": false});
        let mut w = Weights::new(17);
        w.s("gpt_neox.embed_in.weight", &[V, h], 0.05);
        let ln = |w: &mut Weights, n: String| {
            w.norm(format!("{n}.weight"), h, 1.0);
            w.s(format!("{n}.bias"), &[h], 0.1);
        };
        let lin = |w: &mut Weights, n: String, out: usize, inp: usize| {
            w.w(format!("{n}.weight"), &[out, inp]);
            w.s(format!("{n}.bias"), &[out], 0.1);
        };
        for l in 0..layers {
            let p = format!("gpt_neox.layers.{l}");
            ln(&mut w, format!("{p}.input_layernorm"));
            lin(&mut w, format!("{p}.attention.query_key_value"), 3 * h, h);
            lin(&mut w, format!("{p}.attention.dense"), h, h);
            ln(&mut w, format!("{p}.post_attention_layernorm"));
            lin(&mut w, format!("{p}.mlp.dense_h_to_4h"), inter, h);
            lin(&mut w, format!("{p}.mlp.dense_4h_to_h"), h, inter);
        }
        ln(&mut w, "gpt_neox.final_layer_norm".into());
        w.w("embed_out.weight", &[V, h]);
        (cfg, w)
    }

    /// GPT-Neo (`transformer.h.{l}.attn.attention.*`, Linear MLP, learned positions), which GraphLm
    /// loaded. GraphLm scaled scores by 1/sqrt(head_dim); `attention_multiplier: 1.0` made it use
    /// GPT-Neo's unscaled scores, and the local layer's window (256) covers the 9 tokens.
    fn neo_fixture() -> (J, Weights) {
        let (h, inter) = (32, 64);
        let cfg = json!({"model_type": "gpt_neo", "architectures": ["GPTNeoForCausalLM"], "vocab_size": V,
            "hidden_size": h, "num_layers": 2, "num_heads": 2, "intermediate_size": inter,
            "max_position_embeddings": 32, "activation_function": "gelu_new", "layer_norm_epsilon": 1e-5,
            "attention_layers": ["global", "local"], "window_size": 256, "attention_multiplier": 1.0});
        let mut w = Weights::new(18);
        w.s("transformer.wte.weight", &[V, h], 0.05);
        w.s("transformer.wpe.weight", &[32, h], 0.3);
        let ln = |w: &mut Weights, n: String| {
            w.norm(format!("{n}.weight"), h, 1.0);
            w.s(format!("{n}.bias"), &[h], 0.1);
        };
        for l in 0..2 {
            let p = format!("transformer.h.{l}");
            ln(&mut w, format!("{p}.ln_1"));
            for s in ["q_proj", "k_proj", "v_proj"] {
                w.w(format!("{p}.attn.attention.{s}.weight"), &[h, h]);
            }
            w.w(format!("{p}.attn.attention.out_proj.weight"), &[h, h]);
            w.s(format!("{p}.attn.attention.out_proj.bias"), &[h], 0.1);
            ln(&mut w, format!("{p}.ln_2"));
            w.w(format!("{p}.mlp.c_fc.weight"), &[inter, h]);
            w.s(format!("{p}.mlp.c_fc.bias"), &[inter], 0.1);
            w.w(format!("{p}.mlp.c_proj.weight"), &[h, inter]);
            w.s(format!("{p}.mlp.c_proj.bias"), &[h], 0.1);
        }
        ln(&mut w, "transformer.ln_f".into());
        (cfg, w)
    }

    const DEC_VOCAB: usize = 64;

    /// `tests/golden/enc.rs` `gpt2`: pre-LN, tanh GELU, Conv1D weights, `transformer.` prefix.
    fn gpt2_fixture() -> (J, Weights) {
        let (h, ffn, ctx) = (32, 64, 32);
        let mut w = Weights::new(0xEB);
        w.s("transformer.wte.weight", &[DEC_VOCAB, h], 0.5);
        w.s("transformer.wpe.weight", &[ctx, h], 0.3);
        let ln = |w: &mut Weights, n: String| {
            w.norm(format!("{n}.weight"), h, 1.0);
            w.s(format!("{n}.bias"), &[h], 0.1);
        };
        let conv1d = |w: &mut Weights, n: String, inp: usize, out: usize| {
            w.s(
                format!("{n}.weight"),
                &[inp, out],
                1.0 / (inp as f32).sqrt(),
            );
            w.s(format!("{n}.bias"), &[out], 0.1);
        };
        for l in 0..2 {
            let p = format!("transformer.h.{l}");
            ln(&mut w, format!("{p}.ln_1"));
            conv1d(&mut w, format!("{p}.attn.c_attn"), h, 3 * h);
            conv1d(&mut w, format!("{p}.attn.c_proj"), h, h);
            ln(&mut w, format!("{p}.ln_2"));
            conv1d(&mut w, format!("{p}.mlp.c_fc"), h, ffn);
            conv1d(&mut w, format!("{p}.mlp.c_proj"), ffn, h);
        }
        ln(&mut w, "transformer.ln_f".into());
        w.s("score.weight", &[3, h], 0.3);
        let cfg = json!({"architectures": ["GPT2ForSequenceClassification"], "model_type": "gpt2",
            "n_layer": 2, "n_head": 2, "n_embd": h, "n_positions": ctx, "n_ctx": ctx, "vocab_size": DEC_VOCAB,
            "activation_function": "gelu_new", "layer_norm_epsilon": 1e-5, "pad_token_id": 0});
        (cfg, w)
    }

    /// `tests/golden/enc.rs` `opt`: project_in/out (embeddings 16, hidden 32), positions offset 2.
    fn opt_fixture(pre_ln: bool, act: &str, seed: u64) -> (J, Weights) {
        let (h, e, ffn, ctx) = (32, 16, 64, 32);
        let p = "model.decoder";
        let mut w = Weights::new(seed);
        let ln = |w: &mut Weights, n: String| {
            w.norm(format!("{n}.weight"), h, 1.0);
            w.s(format!("{n}.bias"), &[h], 0.1);
        };
        let lin = |w: &mut Weights, n: String, out: usize, inp: usize| {
            w.w(format!("{n}.weight"), &[out, inp]);
            w.s(format!("{n}.bias"), &[out], 0.1);
        };
        w.s(format!("{p}.embed_tokens.weight"), &[DEC_VOCAB, e], 0.5);
        w.s(format!("{p}.embed_positions.weight"), &[ctx + 2, h], 0.3);
        w.w(format!("{p}.project_in.weight"), &[h, e]);
        w.w(format!("{p}.project_out.weight"), &[e, h]);
        for l in 0..2 {
            let b = format!("{p}.layers.{l}");
            ln(&mut w, format!("{b}.self_attn_layer_norm"));
            for s in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                lin(&mut w, format!("{b}.self_attn.{s}"), h, h);
            }
            ln(&mut w, format!("{b}.final_layer_norm"));
            lin(&mut w, format!("{b}.fc1"), ffn, h);
            lin(&mut w, format!("{b}.fc2"), h, ffn);
        }
        if pre_ln {
            ln(&mut w, format!("{p}.final_layer_norm"));
        }
        w.s("score.weight", &[3, e], 1.0);
        let cfg = json!({"architectures": ["OPTForSequenceClassification"], "model_type": "opt",
            "hidden_size": h, "ffn_dim": ffn, "num_hidden_layers": 2, "num_attention_heads": 2,
            "word_embed_proj_dim": e, "max_position_embeddings": ctx, "vocab_size": DEC_VOCAB,
            "do_layer_norm_before": pre_ln, "activation_function": act, "enable_bias": true,
            "layer_norm_elementwise_affine": true, "pad_token_id": 0});
        (cfg, w)
    }

    fn devices() -> Vec<Device> {
        let mut d = vec![Device::Cpu];
        if crate::hardware::available_devices().contains(&Device::Metal) {
            d.push(Device::Metal);
        }
        d
    }

    fn close(bad: &mut Vec<String>, what: &str, got: &[f32], want: &[f32]) {
        let (rel, cos) = compare(got, want);
        eprintln!("{what}: rel {rel:.3e} cos {cos:.6}");
        if !(rel < 1e-3 && cos > 0.9999) {
            bad.push(format!("{what}: rel {rel:.3e} cos {cos:.6}"));
        }
    }

    /// The spec's `out` after a 6-row prefill of `prompt` (head `head`), then after each of 3
    /// one-row decode steps with the KV state carried, and the runner.
    fn spec_steps(
        dir: &Path,
        family: &str,
        dev: Device,
        prompt: &[u32],
        decode: &[u32],
        head: &str,
        out: &str,
    ) -> (Vec<Vec<f32>>, Runner) {
        let mut r = Runner::load(dir, None, dev, 64).unwrap_or_else(|e| panic!("{family}: {e}"));
        assert_eq!(r.model.spec.family, family, "detected family");
        let mut o = vec![r.lm(prompt, 0, head, out, vec![])];
        for (i, &t) in decode.iter().enumerate() {
            o.push(r.lm(&[t], prompt.len() + i, head, out, vec![]));
        }
        (o, r)
    }

    /// Every checkpoint tensor is read by some instance `r` built.
    fn all_read(r: &Runner, case: &str) {
        assert_eq!(
            r.unconsumed(),
            Vec::<String>::new(),
            "{case}: unconsumed checkpoint tensors"
        );
    }

    /// Each GraphLm fixture family against the deleted GraphLm's frozen CPU logits
    /// (`tests/golden/lm_graph_lm_logits.json`); GPT-2 / OPT (which GraphLm did not
    /// load) against the frozen `LnDecoder` hidden states (`tests/golden/lm_ln_decoder_hidden.json`);
    /// BitNet against `tests/golden/lm_bitnet.json`
    /// (GraphLm's BitNet graph is wrong). Prefill 6 rows, then 3 cached decode steps.
    #[test]
    fn llama_family_matches_graph_lm() {
        let tmp = tempfile::tempdir().unwrap();
        let mut bad = Vec::new();
        let seq: Vec<u32> = PROMPT.iter().chain(&DECODE).copied().collect();
        let d = |h, heads, kv, hd, inter, layers| Dims {
            h,
            heads,
            kv,
            hd,
            inter,
            layers,
        };
        let smol = d(64, 4, 2, 16, 128, 2);
        let granite_cfg = lm_config(
            "granitemoehybrid",
            "GraniteMoeHybridForCausalLM",
            &smol,
            json!({
            "attention_multiplier": 0.1, "embedding_multiplier": 3.0, "residual_multiplier": 0.5,
            "logits_scaling": 4.0, "shared_intermediate_size": 128, "num_local_experts": 0,
            "layer_types": ["attention", "attention"], "position_embedding_type": "rope",
            "tie_word_embeddings": false}),
        );
        let (qk, wide) = (d(32, 2, 2, 16, 64, 2), d(32, 4, 2, 16, 64, 2));
        let graph_lm: Vec<(&str, &str, J, Weights)> = vec![
            (
                "smollm",
                "llama",
                lm_config("llama", "LlamaForCausalLM", &smol, json!({})),
                llama_weights(11, &smol, false, false, false, false),
            ),
            (
                "granite",
                "granite",
                granite_cfg,
                llama_weights(12, &smol, false, true, false, true),
            ),
            ("lfm2", "lfm2", lfm2_fixture().0, lfm2_fixture().1),
            (
                "qk_norm",
                "llama",
                lm_config("llama", "LlamaForCausalLM", &qk, json!({})),
                llama_weights(14, &qk, true, false, false, false),
            ),
            (
                "gqa_wide",
                "llama",
                lm_config("llama", "LlamaForCausalLM", &wide, json!({})),
                llama_weights(15, &wide, false, false, false, false),
            ),
            ("gpt_neox", "gpt_neox", neox_fixture().0, neox_fixture().1),
            ("gpt_neo", "gpt2", neo_fixture().0, neo_fixture().1),
        ];
        // The deleted GraphLm's CPU logits on these fixtures (`tests/golden/lm_graph_lm_logits.json`,
        // written by it at c96457c).
        let gl_ref: J = serde_json::from_slice(
            &std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/lm_graph_lm_logits.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for (case, family, cfg, w) in &graph_lm {
            let dir = tmp.path().join(case);
            w.write_checkpoint(&dir, cfg, V);
            for dev in devices() {
                let (got, r) = spec_steps(&dir, family, dev, &PROMPT, &DECODE, "last", "logits");
                all_read(&r, case);
                for (i, g) in got.iter().enumerate() {
                    let want: Vec<f32> = gl_ref["f32"][format!("{case}.step{i}")]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_f64().unwrap() as f32)
                        .collect();
                    close(&mut bad, &format!("{case} {dev:?} step {i}"), g, &want);
                }
            }
        }

        // GPT-2 / OPT: last-row final hidden (after ln_f / project_out) of the 6-row prefill and 3
        // cached decode steps against the deleted `LnDecoder` host forward
        // (`tests/golden/lm_ln_decoder_hidden.json`, written by it at b73c897), and the score
        // instance against `score · hidden`.
        let ln_ref: J = serde_json::from_slice(
            &std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/golden/lm_ln_decoder_hidden.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let dec_seq: Vec<u32> = seq.iter().map(|t| t % DEC_VOCAB as u32).collect();
        let (dp, dd) = dec_seq.split_at(PROMPT.len());
        let decoders: Vec<(&str, &str, (J, Weights))> = vec![
            ("gpt2", "gpt2", gpt2_fixture()),
            ("opt_post_ln", "opt", opt_fixture(false, "relu", 0xEC)),
            ("opt_pre_ln", "opt", opt_fixture(true, "gelu", 0xED)),
        ];
        for (case, family, (cfg, w)) in &decoders {
            let dir = tmp.path().join(case);
            w.write_checkpoint(&dir, cfg, DEC_VOCAB);
            let score = &w.tensors.iter().find(|t| t.0 == "score.weight").unwrap().2;
            let want = |i: usize| -> Vec<f32> {
                ln_ref["f32"][format!("{case}.step{i}")]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as f32)
                    .collect()
            };
            for dev in devices() {
                let (got, mut r) = spec_steps(&dir, family, dev, dp, dd, "hidden", "hidden");
                for (i, g) in got.iter().enumerate() {
                    let w = want(i);
                    close(
                        &mut bad,
                        &format!("{case} {dev:?} step {i}"),
                        &g[g.len() - w.len()..],
                        &w,
                    );
                }
                // Seq-cls: row `last_index` of the 6-row prompt (rewrites the same KV rows).
                let idx = crate::graph::Tensor::U32(vec![PROMPT.len() as u32 - 1]);
                let logits = r.lm(dp, 0, "score", "logits", vec![idx]);
                let h = want(0);
                let want: Vec<f32> = score
                    .chunks(h.len())
                    .map(|row| row.iter().zip(&h).map(|(a, b)| a * b).sum())
                    .collect();
                close(&mut bad, &format!("{case} {dev:?} score"), &logits, &want);
                all_read(&r, case);
            }
        }

        // BitNet: online-ternarized masters, int8 activations, sub-norms, relu².
        let bit = d(64, 4, 2, 16, 128, 2);
        let cfg = lm_config(
            "bitnet",
            "BitNetForCausalLM",
            &bit,
            json!({"hidden_act": "relu2",
            "quantization_config": {"quant_method": "bitnet", "linear_class": "autobitlinear", "quantization_mode": "online"}}),
        );
        let dir = tmp.path().join("bitnet");
        llama_weights(16, &bit, false, false, true, false).write_checkpoint(&dir, &cfg, V);
        let golden: J = serde_json::from_slice(
            &std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/lm_bitnet.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for dev in devices() {
            let (got, r) = spec_steps(&dir, "bitnet", dev, &PROMPT, &DECODE, "last", "logits");
            all_read(&r, "bitnet");
            for (i, g) in got.iter().enumerate() {
                let key = if i == 0 {
                    "prefill6".to_string()
                } else {
                    format!("decode{}", i - 1)
                };
                let want: Vec<f32> = golden["f32"][&key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as f32)
                    .collect();
                close(&mut bad, &format!("bitnet {dev:?} {key}"), g, &want);
            }
        }
        assert!(bad.is_empty(), "mismatches:\n  {}", bad.join("\n  "));
    }

    /// A synthetic `gibbs/*` id (hub.rs writes a `LlamaForCausalLM` config, no weights) random-inits
    /// every weight the llama text instances read.
    #[test]
    fn llama_synthetic_id_runs() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = json!({"architectures": ["LlamaForCausalLM"], "model_type": "qwen2", "hidden_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 256,
            "vocab_size": 512, "max_position_embeddings": 2048, "rms_norm_eps": 1e-6});
        let p = tmp.path().join("config.json");
        std::fs::write(&p, cfg.to_string()).unwrap();
        let a = crate::model::hub::ModelArtifacts {
            model_id: "gibbs/tiny-1m".into(),
            local_dir: Some(tmp.path().to_path_buf()),
            config_path: Some(p),
            ..Default::default()
        };
        let mut r = super::testing::Runner::from_artifacts(&a, None, Device::Cpu, 16).unwrap();
        assert_eq!(r.model.spec.family, "llama");
        for (ids, pos) in [(&[1u32, 2, 3][..], 0), (&[4][..], 3)] {
            let l = r.lm(ids, pos, "last", "logits", vec![]);
            assert_eq!(l.len(), 512);
            assert!(l.iter().all(|x| x.is_finite()) && l.iter().any(|&x| x != 0.0));
        }
        let hw = crate::hardware::HardwareCapability::probe_system();
        let exec = super::dispatch::load_lm(
            &a,
            &super::LoadOpts {
                placement: crate::hardware::PlacementMode::Pin(Device::Cpu),
                token: None,
                quant_name: None,
                hw: &hw,
                reencode: crate::model::weight_format::ReencodeConfig {
                    policy: crate::model::weight_format::ReencodePolicy::BitExact,
                    ..Default::default()
                },
                family: Some("llama"),
                ctx: Some(64),
                stage: &|_| {},
                admit: &|_| Ok(()),
                system1_first: true,
            },
        )
        .unwrap();
        assert_eq!(exec.lm().context_len(), 64);
    }

    /// `match.bases` / `gguf_arch` listings by pipeline.
    #[test]
    fn listings_by_pipeline() {
        let spec = r#"{family: "toy", pipeline: "diffusion", match: {gguf_arch: ["toy_dit"], bases: ["org/toy"]}}"#;
        let r = Registry::from_files(&[("families/toy.json5".into(), spec.into())], &[]).unwrap();
        assert_eq!(r.derived_bases(Pipeline::Diffusion), ["org/toy"]);
        assert_eq!(r.pipeline_gguf_archs(Pipeline::Diffusion), ["toy_dit"]);
        assert!(
            r.derived_bases(Pipeline::CausalLm).is_empty() && r.causal_lm_gguf_archs().is_empty()
        );
    }

    #[test]
    fn supported_models_table_in_readme_matches_registry() {
        let table = Registry::load(None).unwrap().supported_models_markdown();
        let readme = include_str!("../../README.md");
        assert!(
            readme.contains(&table),
            "README.md Supported models table is out of sync with arch/families:\n{table}"
        );
    }

    #[test]
    fn override_adding_use_lib_merges_all_library_sections() {
        let lib = r#"{
            config: {from_lib: "42"},
            checks: [{expr: "from_lib != 42", error: "bad lib config"}],
            ignore: ["lib_ignored.*"],
            macros: {lib_mac: {body: []}}
        }"#;
        let fam = r#"{
            family: "toy",
            match: {hf_model_type: ["toy"]},
            pipeline: "causal_lm",
            config: {own: "1"},
            checks: [{expr: "own != 1", error: "bad own"}]
        }"#;
        let ov = r#"{
            applies_to: {model_type: "TOY*", hf_architectures: ["*ToyForCausalLM*"]},
            patch: {"use_lib+": ["extra"]}
        }"#;
        let r = Registry::from_files(
            &files(&[
                ("lib/extra.json5", lib),
                ("families/toy.json5", fam),
                ("overrides/ov.json5", ov),
            ]),
            &[],
        )
        .unwrap();
        let sig = Signals {
            hf_model_type: Some("toy_v2".into()),
            hf_architectures: vec!["MyToyForCausalLM".into()],
            ..Default::default()
        };
        let s = r.spec("toy", &sig).unwrap();
        assert_eq!(
            s.config.get("from_lib").and_then(|v| v.as_str()),
            Some("42")
        );
        assert_eq!(s.checks.len(), 2, "lib check + family check both present");
        assert_eq!(s.ignore.len(), 1);
        assert!(s.macros.contains_key("lib_mac"));
    }
}
