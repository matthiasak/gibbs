//! Speculative-decoding drafters ([`Drafter`]) for the scheduler.
//!
//! [`Plain`] runs any causal LM that shares the target's tokenizer. Spec drafters run the
//! component a family spec names in `runtime.drafter` (`kind`, `component`):
//!
//! - `kind: "mtp"`, a target's own multi-token-prediction component ([`spec_drafter`], reached
//!   through `NativeLm::drafter`): instance vars `rows` and `src` (`tap`: input `hidden` holds the
//!   target's tap rows; `draft`: a previous `hidden` output), inputs `token_ids`, `hidden`
//!   `[rows, hidden]`, `positions`; outputs `logits` and `hidden` of the last row. Keys `label`,
//!   `taps` (expressions) and `when` (the checkpoint ships the head).
//! - `kind: "block"`, a `pipeline: drafter` family conditioned on target hidden states
//!   ([`load_drafter`]): instance vars `ext` (context rows this pass) and `rows` (config var
//!   `block`); inputs `context` `[ext, taps * target hidden]` with `context_positions` and `pos0`
//!   (its first position), `token_ids` (`[anchor, mask_token, ...]`), `positions` and
//!   `visible_end`; outputs `logits` of the rows from config var `first_row`, plus any other
//!   outputs `step` reads. `step: {component, when}` then runs once per drafted token with input
//!   `prev` (the previous pick) and row i of every block output of the same name; its `logits`
//!   at target ids `ids` (when output) form the pick row. `vocab_map` names a tensor of draft id
//!   -> target id offsets (`d2t`) for heads narrower than the target vocabulary. Config vars
//!   `taps` (minus `tap_shift`), `block`, `mask_token`, `first_row`, `label`, `hidden`. Target
//!   tensors bind by `runtime.shared_from_target` (`name: [target candidates, first present]`).

use crate::arch::build::{build, Instance, ShapeCache};
use crate::arch::expr::Val;
use crate::arch::load::{load_with, Model};
use crate::arch::{glob, registry, Lora, Pipeline, Registry, Signals, Source};
use crate::graph::program::GraphProgram;
use crate::graph::{DType, StateStore, Tensor, Weights};
use crate::hardware::PlacementMode;
use crate::model::hub::ModelArtifacts;
use crate::model::native::{NativeLm, SeqState};
use crate::model::weight_format::{ReencodeConfig, ReencodePolicy};
use crate::scheduler::spec::{DraftLogits, DraftState, Drafter};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::Value as J;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// `runtime.drafter` of a spec.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DrafterSpec {
    kind: String,
    component: String,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    taps: Option<String>,
    #[serde(default)]
    when: Option<String>,
    #[serde(default)]
    step: Option<StepSpec>,
    #[serde(default)]
    vocab_map: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StepSpec {
    component: String,
    when: String,
}

fn drafter_spec(m: &Model) -> Result<Option<DrafterSpec>, String> {
    let Some(j) = m.spec.runtime.get("drafter") else {
        return Ok(None);
    };
    let d: DrafterSpec = serde_json::from_value(j.clone())
        .map_err(|e| format!("{}: runtime.drafter: {e}", m.spec.family))?;
    if let Some(w) = &d.when {
        if !m.env.eval(&J::String(w.clone()))?.truthy() {
            return Ok(None);
        }
    }
    Ok(Some(d))
}

fn usizes(v: Val) -> Result<Vec<usize>, String> {
    match v {
        Val::List(l) => l.iter().map(Val::usize).collect(),
        Val::Num(_) => Ok(vec![v.usize()?]),
        other => Err(format!("expected a list of layers, got {other}")),
    }
}

// ---- spec models ------------------------------------------------------------------------------

/// A spec model plus the target tensors it reads under its own names.
struct Bound {
    model: Arc<Model>,
    /// Drafter name -> (target model, target name).
    shared: HashMap<String, (Arc<Model>, String)>,
}

impl Source for Bound {
    fn tensor(&self, name: &str) -> Option<(Vec<usize>, DType)> {
        match self.shared.get(name) {
            Some((t, n)) => t.store.tensor(n),
            None => self.model.store.tensor(name),
        }
    }
    fn lora(&self, weight: &str) -> Option<Lora> {
        // ponytail: a target adapter on a shared tensor is not applied to the drafter's copy.
        match self.shared.get(weight) {
            Some(_) => None,
            None => self.model.store.lora(weight),
        }
    }
}

impl Bound {
    fn build(&self, comp: &str, vars: &[(&str, Val)]) -> Result<Instance, String> {
        build(&self.model.spec, comp, &self.model.env, self, vars)
    }

    fn materialize(&self, names: &[String]) -> Result<Weights, String> {
        let mut w = Weights::new();
        for n in names {
            let (m, src) = match self.shared.get(n) {
                Some((t, s)) => (t, s),
                None => (&self.model, n),
            };
            let t = m
                .store
                .materialize(std::slice::from_ref(src))?
                .remove(src)
                .ok_or_else(|| format!("no tensor `{src}`"))?;
            w.insert(n.clone(), t);
        }
        Ok(w)
    }
}

/// One component compiled from one instance; other instances run reshaped on its weights, each
/// sequence's state swapped in.
struct Comp {
    name: String,
    shapes: ShapeCache,
    key: String,
    program: Mutex<GraphProgram>,
}

impl Comp {
    fn new(
        b: &Bound,
        name: &str,
        vars: &[(&str, Val)],
        also: &[Vec<(&str, Val)>],
        mode: PlacementMode,
    ) -> Result<(Self, Arc<Instance>), String> {
        let shapes = ShapeCache::new(name);
        let inst = shapes.get(vars, || b.build(name, vars))?;
        // Weights of the instances `also` names are bound too, so they run reshaped.
        let mut names = inst.weights.clone();
        for v in also {
            let other = shapes.get(v, || b.build(name, v))?;
            names.extend(
                other
                    .weights
                    .iter()
                    .filter(|w| !inst.weights.contains(w))
                    .cloned(),
            );
        }
        let w = b.materialize(&names)?;
        let program = GraphProgram::compile_reshapable(inst.graph.clone(), w, mode)
            .map_err(|e| format!("{}/{name}: {e}", b.model.spec.family))?;
        let c = Self {
            name: name.to_string(),
            shapes,
            key: ShapeCache::key(vars),
            program: Mutex::new(program),
        };
        Ok((c, inst))
    }

    /// Runs instance `vars` with its inputs taken by name from `inputs`; outputs by name.
    fn run(
        &self,
        b: &Bound,
        vars: &[(&str, Val)],
        state: &mut StateStore,
        mut inputs: HashMap<&str, Tensor>,
    ) -> Result<HashMap<String, Tensor>, String> {
        let inst = self.shapes.get(vars, || b.build(&self.name, vars))?;
        let ins = inst
            .inputs
            .iter()
            .map(|d| {
                inputs
                    .remove(d.name.as_str())
                    .ok_or_else(|| format!("{}: input `{}` is not supplied", self.name, d.name))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut p = self.program.lock();
        std::mem::swap(&mut p.state, state);
        let out = match ShapeCache::key(vars) == self.key {
            true => p.run(ins),
            false => p.run_reshaped(&inst.graph, ins, true),
        };
        std::mem::swap(&mut p.state, state);
        Ok(inst.outputs.iter().cloned().zip(out?).collect())
    }
}

fn f32_out(out: &HashMap<String, Tensor>, name: &str) -> Result<Vec<f32>, String> {
    match out.get(name) {
        Some(Tensor::F32(v)) => Ok(v.clone()),
        _ => Err(format!("drafter output `{name}` is missing or not f32")),
    }
}

fn positions(from: usize, n: usize) -> Tensor {
    Tensor::U32((from..from + n).map(|p| p as u32).collect())
}

// ---- MTP --------------------------------------------------------------------------------------

/// The weights of `model`'s own drafter component (none without one), for its unused-tensor
/// check.
pub fn drafter_weights(model: &Model) -> Result<Vec<String>, String> {
    match drafter_spec(model)? {
        Some(d) if d.kind == "mtp" => {
            let v = mtp_vars(1, "tap");
            Ok(build(&model.spec, &d.component, &model.env, &model.store, &v)?.weights)
        }
        _ => Ok(Vec::new()),
    }
}

fn mtp_vars(rows: usize, src: &str) -> Vec<(&'static str, Val)> {
    vec![
        ("rows", Val::Num(rows as f64)),
        ("src", Val::Str(src.to_string())),
    ]
}

/// The drafter `model` ships (`runtime.drafter` of kind `mtp`), compiled on `mode`.
pub fn spec_drafter(
    model: &Arc<Model>,
    mode: PlacementMode,
) -> Result<Option<Arc<dyn Drafter>>, String> {
    let Some(d) = drafter_spec(model)? else {
        return Ok(None);
    };
    if d.kind != "mtp" {
        return Err(format!(
            "{}: a causal LM's own drafter must be `kind: mtp`, not `{}`",
            model.spec.family, d.kind
        ));
    }
    let env = &model.env;
    let expr = |e: &Option<String>, what: &str| {
        let e = e
            .as_ref()
            .ok_or_else(|| format!("runtime.drafter needs `{what}`"))?;
        env.eval(&J::String(e.clone()))
    };
    let b = Bound {
        model: model.clone(),
        shared: HashMap::new(),
    };
    // Compiled for the draft step; the catch-up (`tap`) instance also reads the final norm.
    let (comp, _) = Comp::new(
        &b,
        &d.component,
        &mtp_vars(1, "draft"),
        &[mtp_vars(1, "tap")],
        mode,
    )?;
    Ok(Some(Arc::new(Mtp {
        label: expr(&d.label, "label")?.to_string(),
        taps: usizes(expr(&d.taps, "taps")?)?,
        hidden: env.var("hidden")?.usize()?,
        ctx: env.var("ctx")?.usize()?,
        b,
        comp,
    })))
}

/// A multi-token-prediction head drafting autoregressively from the target's last tap.
struct Mtp {
    label: String,
    taps: Vec<usize>,
    hidden: usize,
    ctx: usize,
    b: Bound,
    comp: Comp,
}

/// The head's state and the target positions it holds (all committed).
struct MtpState {
    state: StateStore,
    len: usize,
}

impl Mtp {
    /// Rows `tokens` with `hidden` at positions `pos0..`: the last row's logits and hidden.
    fn step(
        &self,
        st: &mut StateStore,
        src: &str,
        tokens: &[u32],
        hidden: &[f32],
        pos0: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let n = tokens.len();
        let out = self.comp.run(
            &self.b,
            &mtp_vars(n, src),
            st,
            HashMap::from([
                ("token_ids", Tensor::U32(tokens.to_vec())),
                ("hidden", Tensor::F32(hidden.to_vec())),
                ("positions", positions(pos0, n)),
            ]),
        )?;
        Ok((f32_out(&out, "logits")?, f32_out(&out, "hidden")?))
    }
}

impl Drafter for Mtp {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn tap_layers(&self) -> Vec<usize> {
        self.taps.clone()
    }

    fn new_state(&self) -> DraftState {
        Box::new(MtpState {
            state: StateStore::new(),
            len: 0,
        })
    }

    fn propose(
        &self,
        st: &mut DraftState,
        tokens: &[u32],
        hidden: &[f32],
        k: usize,
        pick: &mut dyn FnMut(DraftLogits<'_>) -> u32,
    ) -> Result<Vec<u32>, String> {
        let s = st
            .downcast_mut::<MtpState>()
            .ok_or("draft state was not built by this drafter")?;
        let d = self.hidden;
        let rows = hidden.len() / d;
        if rows == 0 || hidden.len() % d != 0 || s.len + rows + 1 != tokens.len() {
            return Err(format!(
                "{} drafter holds {} positions and got {} hidden values for a {}-token sequence",
                self.label,
                s.len,
                hidden.len(),
                tokens.len()
            ));
        }
        if s.len + rows > self.ctx {
            return Err(format!(
                "{} position {} is past the context length {}",
                self.label,
                s.len + rows,
                self.ctx
            ));
        }
        // Row i is target position s.len + i; it pairs with the token after it.
        let toks = &tokens[s.len + 1..];
        let (mut logits, mut normed) = self.step(&mut s.state, "tap", toks, hidden, s.len)?;
        s.len += rows;
        let k = k.min(self.ctx - s.len + 1);
        let mut drafts = Vec::with_capacity(k);
        while drafts.len() < k {
            let tok = pick(DraftLogits::Dense(&logits));
            drafts.push(tok);
            if drafts.len() < k {
                let pos = s.len + drafts.len() - 1;
                (logits, normed) = self.step(&mut s.state, "draft", &[tok], &normed, pos)?;
            }
        }
        // Draft rows stay in the cache; the next call overwrites them with the target's.
        Ok(drafts)
    }
}

// ---- block drafters ---------------------------------------------------------------------------

/// The `pipeline: drafter` family claiming `sig` (by config architecture or GGUF architecture).
fn drafter_family(reg: &Registry, sig: &Signals) -> Option<String> {
    let any = |pats: &[String], s: &str| pats.iter().any(|p| glob(p, s));
    reg.matchers()
        .into_iter()
        .find(|(_, m, p)| {
            *p == Pipeline::Drafter
                && (sig
                    .hf_architectures
                    .iter()
                    .any(|a| any(&m.hf_architectures, a))
                    || sig.gguf_arch.as_ref().is_some_and(|a| any(&m.gguf_arch, a)))
        })
        .map(|(n, ..)| n)
}

/// The target a drafter repo pairs with (`None` when no drafter family claims its config
/// `architectures` or GGUF architecture `gguf_arch`): the verifier its config names, a
/// `base_model:` hub tag, else the repo name minus a drafter suffix of its family
/// (`runtime.repo_suffixes`; `org/Model-4B-Draft-b16` with suffix `-draft` -> `Model-4B`).
pub fn draft_target(id: &str, cfg: &J, tags: &[&str], gguf_arch: Option<&str>) -> Option<String> {
    let reg = registry().ok()?;
    let sig = Signals {
        hf_architectures: cfg["architectures"]
            .as_array()
            .map(|a| a.iter().filter_map(J::as_str).map(String::from).collect())
            .unwrap_or_default(),
        gguf_arch: gguf_arch.map(String::from),
        ..Default::default()
    };
    let family = drafter_family(reg, &sig)?;
    if let Some(v) = cfg["speculators_config"]["verifier"]["name_or_path"].as_str() {
        return Some(v.to_string());
    }
    let tagged = tags
        .iter()
        .find_map(|t| t.strip_prefix("base_model:").filter(|r| !r.contains(':')));
    if let Some(t) = tagged {
        return Some(t.to_string());
    }
    let name = id.rsplit('/').next().unwrap_or(id);
    let lower = name.to_ascii_lowercase();
    let suffixes = reg.base(&family).ok()?["runtime"]["repo_suffixes"].clone();
    let cut = suffixes
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|s| lower.find(s.as_str()?))
        .min()
        .unwrap_or(name.len());
    Some(name[..cut].to_string())
}

/// Whether drafter target `target` (from [`draft_target`]) is the model `active`: repo names
/// compared without org (verifier paths may be local directories) or a `-GGUF` suffix, and with
/// an explicit weight file (`org/repo/x-Q4_K_M.gguf`) reduced to its repo, case-insensitively.
pub fn is_draft_for(target: &str, active: &str) -> bool {
    let name = |s: &str| {
        let repo = crate::model::hub::split_weight_file(s).0;
        let n = repo
            .trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or(repo);
        let n = n.split('@').next().unwrap_or(n).to_ascii_lowercase();
        n.trim_end_matches("-gguf")
            .trim_end_matches("_gguf")
            .to_string()
    };
    name(target) == name(active)
}

/// Why a GGUF of architecture `arch` cannot attach as a drafter: a drafter family lists it in
/// `runtime.fork_gguf_archs` (globs) but does not claim it (`None` otherwise).
pub fn refused_draft_gguf(arch: &str) -> Option<String> {
    let reg = registry().ok()?;
    reg.matchers().into_iter().find_map(|(f, m, p)| {
        let ours = m.gguf_arch.first()?;
        let forks = reg.base(&f).ok()?["runtime"]["fork_gguf_archs"].clone();
        let fork = p == Pipeline::Drafter
            && !m.gguf_arch.iter().any(|g| glob(g, arch))
            && forks.as_array()?.iter().filter_map(J::as_str).any(|g| glob(g, arch));
        fork.then(|| {
            format!("GGUF draft sidecar with fork architecture '{arch}' (tensor names and metadata differ from '{ours}'; attach a '{ours}' GGUF or the safetensors drafter)")
        })
    })
}

/// Loads `draft` as a drafter for the target loaded from `target`, computing on `mode`: a block
/// drafter when a `pipeline: drafter` family claims it, else a [`Plain`] causal LM.
pub fn load_drafter(
    draft: &ModelArtifacts,
    target: &ModelArtifacts,
    mode: PlacementMode,
) -> Result<Arc<dyn Drafter>, String> {
    let reg = registry()?;
    let ck = crate::arch::load::open(draft)?;
    match drafter_family(reg, &ck.sig) {
        Some(f) => load_block(reg, &f, draft, target, mode),
        None => load_plain(reg, draft, target, mode),
    }
}

fn load_block(
    reg: &Registry,
    family: &str,
    draft: &ModelArtifacts,
    target: &ModelArtifacts,
    mode: PlacementMode,
) -> Result<Arc<dyn Drafter>, String> {
    // Drafter and shared target tensors as stored: they are small next to the target's own
    // (re-encoded) copy.
    // ponytail: the target's embeddings / LM head are held twice (target program and drafter);
    // share the target's bound tensors when that memory matters.
    let exact = ReencodeConfig {
        policy: ReencodePolicy::BitExact,
        ..ReencodeConfig::effective()
    };
    let dm = load_with(draft, reg, Some(family), exact.clone())?;
    crate::arch::load::check(&dm)?;
    let tm = Arc::new(load_with(target, reg, None, exact)?);
    let fam = &dm.spec.family.clone();
    let d = drafter_spec(&dm)?
        .filter(|d| d.kind == "block")
        .ok_or_else(|| format!("{fam}: no `runtime.drafter` of kind `block`"))?;
    let var = |n: &str| dm.env.var(n);
    let block = var("block")?.usize()?;
    // The drafter's cache holds every position the target does, plus one block.
    if let Ok(c) = tm.env.var("ctx").and_then(|v| v.usize()) {
        dm.env.set_ctx((c + block).into());
    }
    let shift = var("tap_shift")?.usize()?;
    let taps = usizes(var("taps")?)?
        .into_iter()
        .map(|t| {
            t.checked_sub(shift)
                .ok_or_else(|| format!("{fam}: tap layer {t} is no decoder layer"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let Ok(n) = tm.env.var("layers").and_then(|v| v.usize()) {
        if let Some(bad) = taps.iter().find(|&&t| t >= n) {
            return Err(format!(
                "drafter taps target layer {bad}; the target has {n} layers"
            ));
        }
    }
    let mut shared = HashMap::new();
    for (name, cands) in dm.spec.runtime["shared_from_target"]
        .as_object()
        .into_iter()
        .flatten()
    {
        let found = cands
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(J::as_str)
            .find(|c| tm.store.tensor(c).is_some())
            .ok_or_else(|| format!("target checkpoint has none of {cands} ({name})"))?;
        shared.insert(name.clone(), (tm.clone(), found.to_string()));
    }
    let hidden = var("hidden")?.usize()?;
    for (t, n) in shared.values() {
        let w = t
            .store
            .tensor(n)
            .map_or(0, |(s, _)| *s.last().unwrap_or(&0));
        if w != hidden {
            return Err(format!(
                "drafter hidden {hidden} != target embedding width {w}"
            ));
        }
    }
    let first_row = var("first_row")?.usize()?;
    let mask = var("mask_token")?.usize()? as u32;
    let b = Bound {
        model: Arc::new(dm),
        shared,
    };
    let dm = &b.model;
    let (comp, inst) = Comp::new(&b, &d.component, &block_vars(1, block), &[], mode)?;
    let width = inst
        .inputs
        .iter()
        .find(|i| i.name == "context")
        .map_or(0, |i| i.shape[1]);
    if let Ok(th) = tm.env.var("hidden").and_then(|v| v.usize()) {
        if width != taps.len() * th {
            return Err(format!(
                "drafter fc takes {width} features; {} taps of the target's hidden {th} give {}",
                taps.len(),
                taps.len() * th
            ));
        }
    }
    let vocab = tm.env.var("vocab")?.usize()?;
    if mask as usize >= vocab {
        return Err(format!(
            "mask token {mask} is outside the target vocabulary"
        ));
    }
    let mut used: HashSet<String> = inst.weights.iter().cloned().collect();
    let step = match &d.step {
        Some(s) if dm.env.eval(&J::String(s.when.clone()))?.truthy() => {
            let (c, si) = Comp::new(&b, &s.component, &[], &[], mode)?;
            used.extend(si.weights.iter().cloned());
            Some(c)
        }
        _ => None,
    };
    // Draft logits narrower than the target vocabulary map to target ids.
    let width_out = block_width(&b, &d.component, block)?;
    let map = match &d.vocab_map {
        Some(n) if width_out < vocab => {
            let off = b
                .materialize(std::slice::from_ref(n))
                .map_err(|_| format!("{fam}: a {width_out}-id draft head ships no `{n}`"))?
                .remove(n)
                .map(|t| t.as_f32().map(<[f32]>::to_vec))
                .transpose()?
                .unwrap_or_default();
            if off.len() != width_out {
                return Err(format!(
                    "{n} has {} entries for {width_out} draft ids",
                    off.len()
                ));
            }
            used.insert(n.clone());
            let ids: Vec<u32> = off
                .iter()
                .enumerate()
                .map(|(i, &o)| i as u32 + o as u32)
                .collect();
            if ids.iter().any(|&t| t as usize >= vocab) {
                return Err(format!("{n} maps outside the target vocabulary"));
            }
            Some(ids)
        }
        _ => None,
    };
    let unused = dm.store.unconsumed(&used, &dm.ignore);
    if !unused.is_empty() {
        return Err(format!(
            "{fam}: drafter tensors the drafter graph does not run: {}",
            unused.join(", ")
        ));
    }
    let label = format!(
        "{} {} (block {block})",
        dm.env.var("label")?,
        draft.model_id
    );
    Ok(Arc::new(Block {
        label,
        taps,
        block,
        mask,
        first_row,
        width,
        vocab,
        comp,
        step,
        map,
        b,
    }))
}

fn block_vars(ext: usize, rows: usize) -> Vec<(&'static str, Val)> {
    vec![
        ("ext", Val::Num(ext as f64)),
        ("rows", Val::Num(rows as f64)),
    ]
}

/// Width of the block component's `logits` rows.
fn block_width(b: &Bound, comp: &str, block: usize) -> Result<usize, String> {
    let inst = b.build(comp, &block_vars(0, block))?;
    let i = inst
        .output_index("logits")
        .ok_or_else(|| format!("{comp}: no `logits` output"))?;
    let v = inst.graph.outputs[i];
    Ok(*inst.graph.value(v).shape.last().unwrap_or(&0))
}

/// A block drafter: one pass over `[anchor, mask, ...]` against the committed context drafts a
/// whole block.
struct Block {
    label: String,
    taps: Vec<usize>,
    block: usize,
    mask: u32,
    first_row: usize,
    /// Features per context row.
    width: usize,
    /// Target vocabulary.
    vocab: usize,
    comp: Comp,
    step: Option<Comp>,
    map: Option<Vec<u32>>,
    b: Bound,
}

/// The drafter's state and the committed positions its context holds.
struct BlockState {
    state: StateStore,
    len: usize,
}

impl Drafter for Block {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn max_k(&self) -> usize {
        // ponytail: verify of <= 16 rows is bit-exact with one-token decode on the targets'
        // multi-row GEMV; larger blocks are capped there until a wider exact path exists.
        (self.block - self.first_row).min(15)
    }

    fn tap_layers(&self) -> Vec<usize> {
        self.taps.clone()
    }

    fn new_state(&self) -> DraftState {
        Box::new(BlockState {
            state: StateStore::new(),
            len: 0,
        })
    }

    fn propose(
        &self,
        st: &mut DraftState,
        tokens: &[u32],
        hidden: &[f32],
        k: usize,
        pick: &mut dyn FnMut(DraftLogits<'_>) -> u32,
    ) -> Result<Vec<u32>, String> {
        let s = st
            .downcast_mut::<BlockState>()
            .ok_or("draft state was not built by this drafter")?;
        if self.width == 0 || hidden.len() % self.width != 0 {
            return Err(format!(
                "tapped hidden states are not whole rows of {}",
                self.width
            ));
        }
        let ext = hidden.len() / self.width;
        let (&anchor, committed) = tokens.split_last().ok_or("empty sequence")?;
        let start = committed.len();
        if s.len + ext != start {
            return Err(format!(
                "drafter context holds {} positions; the target has run {start}",
                s.len + ext
            ));
        }
        let m = k.min(self.block - self.first_row);
        // Every pass appends context rows, so the KV states keep one dtype (on Metal a pass without
        // the context writes would store them as f16); the target runs at least one position
        // between proposes, so this only skips a call that could not draft from new state.
        if ext == 0 {
            return Ok(Vec::new());
        }
        let n = self.block;
        let mut ids = vec![self.mask; n];
        ids[0] = anchor;
        let inputs = HashMap::from([
            ("token_ids", Tensor::U32(ids)),
            ("positions", positions(start, n)),
            ("visible_end", Tensor::U32(vec![(start + n) as u32; n])),
            ("context", Tensor::F32(hidden.to_vec())),
            ("context_positions", positions(s.len, ext)),
            ("pos0", Tensor::U32(vec![s.len as u32])),
        ]);
        let out = self
            .comp
            .run(&self.b, &block_vars(ext, n), &mut s.state, inputs)?;
        s.len += ext;
        let logits = f32_out(&out, "logits")?;
        let w = logits.len() / (n - self.first_row);
        let mut drafts = Vec::with_capacity(m);
        let mut full = vec![f32::NEG_INFINITY; self.vocab];
        let mut prev = anchor;
        for i in 0..m {
            let row = &logits[i * w..][..w];
            let (vals, ids) = match &self.step {
                None => (row.to_vec(), None),
                Some(c) => {
                    let mut ins = HashMap::from([
                        ("prev", Tensor::U32(vec![prev])),
                        ("logits", Tensor::F32(row.to_vec())),
                    ]);
                    for (name, t) in &out {
                        if let (Tensor::F32(v), false) = (t, name == "logits") {
                            let r = v.len() / (n - self.first_row);
                            ins.insert(name.as_str(), Tensor::F32(v[i * r..][..r].to_vec()));
                        }
                    }
                    let so = c.run(&self.b, &[], &mut StateStore::new(), ins)?;
                    let ids = match so.get("ids") {
                        Some(t) => Some(t.as_u32()?.to_vec()),
                        None => None,
                    };
                    (f32_out(&so, "logits")?, ids)
                }
            };
            prev = match ids.as_deref().or(self.map.as_deref()) {
                Some(ids) => {
                    full.fill(f32::NEG_INFINITY);
                    for (&t, &l) in ids.iter().zip(&vals) {
                        full[t as usize] = l;
                    }
                    pick(DraftLogits::Dense(&full))
                }
                None => pick(DraftLogits::Dense(&vals)),
            };
            drafts.push(prev);
        }
        Ok(drafts)
    }
}

// ---- plain drafters ---------------------------------------------------------------------------

/// `vocab_size` of a checkpoint's config.json (nested `text_config` first).
fn config_vocab(a: &ModelArtifacts) -> Option<u64> {
    let j: J = serde_json::from_slice(&std::fs::read(a.config_path.as_ref()?).ok()?).ok()?;
    j["text_config"]["vocab_size"]
        .as_u64()
        .or(j["vocab_size"].as_u64())
}

fn load_plain(
    reg: &Registry,
    draft: &ModelArtifacts,
    target: &ModelArtifacts,
    mode: PlacementMode,
) -> Result<Arc<dyn Drafter>, String> {
    let lm = crate::runtime::lm::load(draft, reg, None, mode)?;
    let ttok = crate::model::hf_loader::load_json_tokenizer(
        target
            .tokenizer_path
            .as_deref()
            .ok_or("target has no tokenizer.json")?,
    )?;
    let probe = "Speculative decoding: 12 × 34 = 408, naïve café 🙂\n<|im_end|>";
    if lm.tokenizer().encode(probe) != ttok.encode(probe) {
        return Err(format!(
            "{} tokenizes differently from {}; plain drafting needs the same tokenizer",
            draft.model_id, target.model_id
        ));
    }
    if let (Some(d), Some(t)) = (config_vocab(draft), config_vocab(target)) {
        if d != t {
            return Err(format!("draft vocab {d} != target vocab {t}"));
        }
    }
    Ok(Arc::new(Plain {
        label: format!("draft {}", draft.model_id),
        lm,
    }))
}

/// A small causal LM with the target's tokenizer, proposing tokens autoregressively.
struct Plain {
    label: String,
    lm: Arc<dyn NativeLm>,
}

/// The draft's sequence (built at the first propose) and the ids it has run.
struct PlainState {
    seq: Option<SeqState>,
    ids: Vec<u32>,
}

impl Drafter for Plain {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn tap_layers(&self) -> Vec<usize> {
        Vec::new()
    }

    fn new_state(&self) -> DraftState {
        Box::new(PlainState {
            seq: None,
            ids: Vec::new(),
        })
    }

    fn propose(
        &self,
        st: &mut DraftState,
        tokens: &[u32],
        _hidden: &[f32],
        k: usize,
        pick: &mut dyn FnMut(DraftLogits<'_>) -> u32,
    ) -> Result<Vec<u32>, String> {
        let st = st
            .downcast_mut::<PlainState>()
            .ok_or("draft state was not built by this drafter")?;
        let m = &self.lm;
        let n = tokens.len();
        let k = k.min(m.context_len().saturating_sub(n));
        if k == 0 || n == 0 {
            return Ok(Vec::new());
        }
        // Rewind to the longest prefix the draft shares with the sequence (its last drafts were
        // rejected or its last call drafted past them), then catch up on committed tokens.
        let keep = st
            .ids
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count()
            .min(n - 1);
        let caught_up = &tokens[keep..n - 1];
        if st.seq.is_none() {
            st.seq = Some(m.prepare(Vec::new(), &[])?.1);
        }
        let s = st.seq.as_mut().expect("built above");
        // Drafts ran as consecutive checkpointing forwards, so any `keep` since the previous
        // call's first draft rewinds exactly.
        m.truncate(s, keep)?;
        if !caught_up.is_empty() {
            m.forward(caught_up, keep, s)?;
        }
        st.ids.truncate(keep);
        st.ids.extend_from_slice(caught_up);
        let mut out = Vec::with_capacity(k);
        let mut last = tokens[n - 1];
        for i in 0..k {
            let logits = m.forward_all(&[last], n - 1 + i, s)?;
            st.ids.push(last);
            last = pick(DraftLogits::Dense(&logits));
            out.push(last);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::testing::Weights as W;
    use crate::hardware::Device;
    use serde_json::json;

    const H: usize = 8;
    const V: usize = 16;

    /// `(llama.cpp name, z-lab name, row-major shape)` of a 1-layer DSpark drafter (hidden 8,
    /// 2 heads x 4 dims, 1 kv head, ffn 8, Markov rank 4, 2 taps).
    fn layout() -> Vec<(&'static str, &'static str, Vec<usize>)> {
        vec![
            ("fc.weight", "fc.weight", vec![H, 2 * H]),
            ("enc.output_norm.weight", "hidden_norm.weight", vec![H]),
            ("output_norm.weight", "norm.weight", vec![H]),
            (
                "markov_w1.weight",
                "markov_head.markov_w1.weight",
                vec![V, 4],
            ),
            (
                "markov_w2.weight",
                "markov_head.markov_w2.weight",
                vec![V, 4],
            ),
            (
                "blk.0.attn_norm.weight",
                "layers.0.input_layernorm.weight",
                vec![H],
            ),
            (
                "blk.0.ffn_norm.weight",
                "layers.0.post_attention_layernorm.weight",
                vec![H],
            ),
            (
                "blk.0.attn_q.weight",
                "layers.0.self_attn.q_proj.weight",
                vec![8, H],
            ),
            (
                "blk.0.attn_k.weight",
                "layers.0.self_attn.k_proj.weight",
                vec![4, H],
            ),
            (
                "blk.0.attn_v.weight",
                "layers.0.self_attn.v_proj.weight",
                vec![4, H],
            ),
            (
                "blk.0.attn_output.weight",
                "layers.0.self_attn.o_proj.weight",
                vec![H, 8],
            ),
            (
                "blk.0.attn_q_norm.weight",
                "layers.0.self_attn.q_norm.weight",
                vec![4],
            ),
            (
                "blk.0.attn_k_norm.weight",
                "layers.0.self_attn.k_norm.weight",
                vec![4],
            ),
            (
                "blk.0.ffn_gate.weight",
                "layers.0.mlp.gate_proj.weight",
                vec![8, H],
            ),
            (
                "blk.0.ffn_up.weight",
                "layers.0.mlp.up_proj.weight",
                vec![8, H],
            ),
            (
                "blk.0.ffn_down.weight",
                "layers.0.mlp.down_proj.weight",
                vec![H, 8],
            ),
        ]
    }

    /// Deterministic values in [-0.5, 0.5); norms near 1.
    fn values(name: &str, n: usize) -> Vec<f32> {
        let mut s = name.bytes().fold(0x9E37_79B9u32, |a, b| {
            a.wrapping_mul(31).wrapping_add(b as u32)
        });
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let v = (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                if name.contains("norm") {
                    1.0 + 0.2 * v
                } else {
                    v
                }
            })
            .collect()
    }

    fn artifacts(dir: &std::path::Path, weights: &str) -> ModelArtifacts {
        let opt = |p: std::path::PathBuf| p.exists().then_some(p);
        ModelArtifacts {
            model_id: dir.display().to_string(),
            local_dir: Some(dir.to_path_buf()),
            config_path: opt(dir.join("config.json")),
            tokenizer_path: opt(dir.join("tokenizer.json")),
            weights_path: opt(dir.join(weights)),
            is_cached: true,
            ..Default::default()
        }
    }

    fn drafts(d: &Arc<dyn Drafter>) -> (Vec<u32>, Vec<Vec<f32>>) {
        let hidden = values("hidden", 2 * 2 * H);
        let mut st = d.new_state();
        let mut rows = Vec::new();
        let out = d
            .propose(&mut st, &[1, 2, 3], &hidden, 3, &mut |l| {
                let DraftLogits::Dense(l) = l else {
                    unreachable!("model drafters pick from dense rows")
                };
                rows.push(l.to_vec());
                l.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as u32
            })
            .unwrap();
        (out, rows)
    }

    /// A llama.cpp `dflash` GGUF (DSpark: Markov head) maps onto the z-lab names and config the
    /// `dflash` spec reads: the drafter built from it drafts exactly like the same weights stored
    /// under the safetensors names. Tensors the drafter graph does not run are refused.
    #[test]
    fn gguf_sidecar_matches_the_safetensors_drafter() {
        let _g = crate::hardware::cpu_compute::allow();
        let dir = tempfile::tempdir().unwrap();
        let (td, sd, gd) = (
            dir.path().join("t"),
            dir.path().join("s"),
            dir.path().join("g"),
        );
        let mut t = W::new(1);
        t.put("model.embed_tokens.weight", &[V, H], values("embed", V * H));
        let tcfg = json!({"model_type": "llama", "architectures": ["LlamaForCausalLM"],
            "hidden_size": H, "num_hidden_layers": 4, "num_attention_heads": 2, "vocab_size": V,
            "max_position_embeddings": 64});
        t.write_checkpoint(&td, &tcfg, V);

        let mut s = W::new(2);
        for (g, z, shape) in layout() {
            let n = shape.iter().product();
            s.put(z, &shape, values(g, n));
        }
        s.write_checkpoint(
            &sd,
            &json!({
            "architectures": ["DSparkDraftModel"], "hidden_size": H, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
            "intermediate_size": 8, "rms_norm_eps": 1e-6, "block_size": 4, "mask_token_id": 3,
            "rope_parameters": {"rope_theta": 10000.0}, "target_layer_ids": [1, 2],
            "sample_from_anchor": true}),
            V,
        );

        let u32_kv = |k: &'static str, v: u32| (k, 4u32, v.to_le_bytes().to_vec());
        let mut name = (b"dflash".len() as u64).to_le_bytes().to_vec();
        name.extend_from_slice(b"dflash");
        let mut taps = 5u32.to_le_bytes().to_vec();
        taps.extend_from_slice(&2u64.to_le_bytes());
        taps.extend([2i32, 3].iter().flat_map(|x| x.to_le_bytes()));
        let kv = [
            ("general.architecture", 8u32, name),
            u32_kv("dflash.block_count", 1),
            u32_kv("dflash.embedding_length", H as u32),
            u32_kv("dflash.feed_forward_length", 8),
            u32_kv("dflash.attention.head_count", 2),
            u32_kv("dflash.attention.head_count_kv", 1),
            u32_kv("dflash.attention.key_length", 4),
            (
                "dflash.attention.layer_norm_rms_epsilon",
                6,
                1e-6f32.to_le_bytes().to_vec(),
            ),
            ("dflash.rope.freq_base", 6, 10000f32.to_le_bytes().to_vec()),
            u32_kv("dflash.block_size", 4),
            ("dflash.target_layers", 9, taps),
            u32_kv("tokenizer.ggml.mask_token_id", 3),
        ];
        let tensors: Vec<(&str, Vec<usize>, Vec<f32>)> = layout()
            .into_iter()
            .map(|(g, _, shape)| {
                (
                    g,
                    shape.iter().rev().copied().collect(),
                    values(g, shape.iter().product()),
                )
            })
            .collect();
        let all: Vec<(&str, &[usize], &[f32])> = tensors
            .iter()
            .map(|(n, d, v)| (*n, &d[..], &v[..]))
            .collect();
        std::fs::create_dir_all(&gd).unwrap();
        let gguf = crate::model::loaders::gguf::test_gguf_file(&kv, &all);
        std::fs::write(gd.join("dflash.gguf"), &gguf).unwrap();

        let cpu = PlacementMode::Pin(Device::Cpu);
        let target = artifacts(&td, "model.safetensors");
        let from_gguf = load_drafter(&artifacts(&gd, "dflash.gguf"), &target, cpu).unwrap();
        let from_st = load_drafter(&artifacts(&sd, "model.safetensors"), &target, cpu).unwrap();
        assert_eq!(from_gguf.tap_layers(), [1, 2]);
        let (a, b) = (drafts(&from_gguf), drafts(&from_st));
        assert_eq!(a.0.len(), 3);
        assert_eq!(a, b);

        // A tensor the drafter graph does not run is refused by name, not skipped.
        let mut extra = all.clone();
        let post = vec![1.0f32; H];
        extra.push(("blk.0.attn_post_norm.weight", &[H], &post));
        std::fs::write(
            gd.join("dflash.gguf"),
            crate::model::loaders::gguf::test_gguf_file(&kv, &extra),
        )
        .unwrap();
        let err = load_drafter(&artifacts(&gd, "dflash.gguf"), &target, cpu)
            .err()
            .unwrap();
        assert!(err.contains("blk.0.attn_post_norm.weight"), "{err}");
    }

    /// A plain drafter that is the target itself drafts the target's greedy continuation, also
    /// after a partial accept (rewind) and a rejected draft.
    #[test]
    fn plain_self_drafter_drafts_the_greedy_continuation() {
        let _g = crate::hardware::cpu_compute::allow();
        let dir = tempfile::tempdir().unwrap();
        let mut w = W::new(7);
        w.w("model.embed_tokens.weight", &[V, H])
            .norm("model.norm.weight", H, 1.0);
        let p = "model.layers.0";
        w.norm(format!("{p}.input_layernorm.weight"), H, 1.0)
            .norm(format!("{p}.post_attention_layernorm.weight"), H, 1.0)
            .w(format!("{p}.self_attn.q_proj.weight"), &[H, H])
            .w(format!("{p}.self_attn.k_proj.weight"), &[4, H])
            .w(format!("{p}.self_attn.v_proj.weight"), &[4, H])
            .w(format!("{p}.self_attn.o_proj.weight"), &[H, H])
            .w(format!("{p}.mlp.gate_proj.weight"), &[16, H])
            .w(format!("{p}.mlp.up_proj.weight"), &[16, H])
            .w(format!("{p}.mlp.down_proj.weight"), &[H, 16])
            .s("lm_head.weight", &[V, H], 2.0);
        w.write_checkpoint(dir.path(), &json!({"model_type": "llama", "architectures": ["LlamaForCausalLM"],
            "hidden_size": H, "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
            "intermediate_size": 16, "vocab_size": V, "max_position_embeddings": 64}), V);
        let a = artifacts(dir.path(), "model.safetensors");
        let lm: Arc<dyn NativeLm> = crate::runtime::lm::load(
            &a,
            registry().unwrap(),
            None,
            PlacementMode::Pin(Device::Cpu),
        )
        .unwrap();
        let argmax = |l: &[f32]| {
            l.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32
        };
        // The model's own greedy continuation of `toks`, `n` tokens.
        let greedy = |toks: &[u32], n: usize| {
            let (_, mut s) = lm.prepare(toks.to_vec(), &[]).unwrap();
            let mut out = vec![argmax(&lm.forward(toks, 0, &mut s).unwrap())];
            while out.len() < n {
                let l = lm
                    .forward(&out[out.len() - 1..], toks.len() + out.len() - 1, &mut s)
                    .unwrap();
                out.push(argmax(&l));
            }
            out
        };
        let d = Plain {
            label: "self".into(),
            lm: lm.clone(),
        };
        let mut st = d.new_state();
        let mut toks = vec![5u32, 9, 14, 3];
        for round in 0..3 {
            let got = d
                .propose(&mut st, &toks, &[], 3, &mut |l| match l {
                    DraftLogits::Dense(l) => argmax(l),
                    DraftLogits::Sparse { .. } => unreachable!("plain drafters pick dense rows"),
                })
                .unwrap();
            assert_eq!(got, greedy(&toks, 3), "round {round}");
            // Accept the first draft, reject the second.
            toks.push(got[0]);
            toks.push((got[1] + 1) % V as u32);
        }
    }
}
