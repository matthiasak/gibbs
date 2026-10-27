//! The encoder-decoder pipeline: [`SpecEncDec`] runs the `encoder` and `decoder` components of
//! any `pipeline: encdec` family spec as a [`NativeLm`].
//!
//! [`NativeLm::prepare`] turns the scheduler's tokens into encoder ids and the decoder prompt
//! (`model::tokenizer::seq2seq`, `runtime.text`), runs the encoder once, and hands its outputs to
//! the decoder generically: every encoder output whose name is a decoder state (the cross K/V
//! `xk/xv.{l}`) is copied into the sequence's [`StateStore`], zero-padded to the state's rows. The
//! decoder then runs one position per pass over that store.
//!
//! The encoder is one program compiled at the first source's bucket (`runtime.encoder_buckets`,
//! then the spec's `src_cap`); other buckets run reshaped on its weights. Inputs filled by name:
//! encoder `token_ids`, `positions`, `key_mask` (0 / -inf past the source); decoder `token_ids`,
//! `positions`, `self_mask` (0 for keys up to the position), `cross_mask` (0 for source keys).
//!
//! Search follows `generation_config.json` (else config.json): greedy configs decode live in
//! `forward` (sampling and streaming work as for any model); `num_beams` > 1 (and texts whose
//! display needs the whole output) run transformers' beam search with its logits processors in
//! `prepare`, and `forward` replays the chosen hypothesis as one-hot logits. A beam forks its
//! sequence by copying the store's self-attention rows and sharing the encoder-handed states.

use crate::arch::build::{build, Instance, ShapeCache};
use crate::arch::expr::Val;
use crate::arch::load::Model;
use crate::arch::Registry;
use crate::graph::program::GraphProgram;
use crate::graph::{StateStore, Tensor, ValueKind};
use crate::hardware::{Device, PlacementMode};
use crate::model::hub::ModelArtifacts;
use crate::model::native::{MediaInput, NativeLm, PrepareCtx, SeqState};
use crate::model::tokenizer::seq2seq::EncDecText;
use crate::model::tokenizer::Tokenizer;
use crate::model::weight_format::{ReencodeConfig, ReencodePolicy};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;

pub const ENCODER: &str = "encoder";
pub const DECODER: &str = "decoder";
/// Mask value of a padded / future key.
const MASKED: f32 = f32::NEG_INFINITY;

/// The `runtime` keys this pipeline reads.
#[derive(Deserialize)]
struct Runtime {
    /// `model::tokenizer::seq2seq::EncDecText` kind.
    text: String,
    #[serde(default = "buckets")]
    encoder_buckets: Vec<usize>,
}

fn buckets() -> Vec<usize> {
    vec![64, 256]
}

/// generation_config.json (falling back to config.json).
struct Gen {
    beams: usize,
    max_len: usize,
    min_len: usize,
    length_penalty: f32,
    no_repeat: usize,
    early_stopping: bool,
    renormalize: bool,
    forced_eos: Option<u32>,
    bad: Vec<u32>,
}

impl Gen {
    fn parse(dir: &Path, cfg: &Value, max_pos: usize) -> Self {
        let gen_cfg: Value = std::fs::read(dir.join("generation_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or(Value::Null);
        let g = |k: &str| -> Value {
            match &gen_cfg[k] {
                Value::Null => cfg[k].clone(),
                v => v.clone(),
            }
        };
        let gu = |k: &str| g(k).as_u64().map(|v| v as u32);
        Self {
            beams: gu("num_beams").unwrap_or(1).max(1) as usize,
            max_len: gu("max_length").map_or(max_pos, |m| m as usize).min(max_pos),
            min_len: gu("min_length").unwrap_or(0) as usize,
            length_penalty: g("length_penalty").as_f64().unwrap_or(1.0) as f32,
            no_repeat: gu("no_repeat_ngram_size").unwrap_or(0) as usize,
            early_stopping: g("early_stopping").as_bool() == Some(true),
            renormalize: g("renormalize_logits").as_bool() == Some(true),
            forced_eos: gu("forced_eos_token_id"),
            bad: g("bad_words_ids")
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|w| match w.as_array()?.as_slice() {
                    [t] => Some(t.as_u64()? as u32),
                    _ => None,
                })
                .collect(),
        }
    }
}

/// One decoding sequence (or beam): its state and decoded length.
struct Beam {
    state: StateStore,
    len: usize,
}

/// A request's decoder side: the beams over one encoded source.
struct Run {
    beams: Vec<Beam>,
    n_src: usize,
}

/// Beam search: the chosen output, replayed by `forward`.
struct Replay {
    prompt: usize,
    out: Vec<u32>,
}

/// Live decoding: `forward` runs the decoder step by step.
struct Live {
    run: Run,
    /// Decoder ids so far (prompt included).
    seq: Vec<u32>,
    prompt: usize,
}

/// A decoder state's declaration: rows and elements.
struct StateDecl {
    rows: usize,
    numel: usize,
}

pub struct SpecEncDec {
    model: Model,
    rt: Runtime,
    text: EncDecText,
    gen: Gen,
    src_cap: usize,
    self_cap: usize,
    vocab: usize,
    enc_shapes: ShapeCache,
    /// The encoder program and the bucket it was compiled for (others run reshaped on it).
    enc: Mutex<Option<(usize, GraphProgram)>>,
    mode: PlacementMode,
    dec: Mutex<GraphProgram>,
    dec_inst: Instance,
    states: HashMap<String, StateDecl>,
    /// Decoder states the encoder fills (shared, never copied, by a fork).
    handoff: HashSet<String>,
    weights: usize,
}

/// Loads `artifacts` as `family` (detected when `None`).
pub fn load(
    artifacts: &ModelArtifacts,
    reg: &Registry,
    family: Option<&str>,
    mode: PlacementMode,
) -> Result<Arc<dyn NativeLm>, String> {
    let reencode = ReencodeConfig {
        policy: ReencodePolicy::BitExact,
        force: ReencodeConfig::effective().force,
        ..Default::default()
    };
    let m = crate::arch::load::load_with(artifacts, reg, family, reencode)?;
    Ok(Arc::new(SpecEncDec::new(m, mode)?))
}

fn read_cfg(dir: &Path) -> Result<Value, String> {
    let p = dir.join("config.json");
    serde_json::from_slice(&std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?)
        .map_err(|e| format!("{}: {e}", p.display()))
}

impl SpecEncDec {
    pub fn new(model: Model, mode: PlacementMode) -> Result<Self, String> {
        crate::arch::load::check(&model)?;
        let fam = model.spec.family.clone();
        let rt: Runtime = model.spec.runtime_as()?;
        let dir = model
            .ckpt
            .dir
            .clone()
            .ok_or_else(|| format!("{fam}: the checkpoint has no directory"))?;
        let cfg = read_cfg(&dir)?;
        let var = |v: &str| -> Result<usize, String> {
            model.env.var(v).and_then(|x| x.usize()).map_err(|e| format!("{fam}: config `{v}`: {e}"))
        };
        let (src_cap, self_cap) = (var("src_cap")?, var("self_cap")?);
        let m = &model;
        let dec_inst = build(&m.spec, DECODER, &m.env, &m.store, &[])?;
        let b0 = rt.encoder_buckets.first().copied().unwrap_or(src_cap).min(src_cap);
        let enc_vars = Self::enc_vars(b0);
        let enc_probe = build(&m.spec, ENCODER, &m.env, &m.store, &enc_vars)?;
        let used: HashSet<String> = dec_inst.weights.iter().chain(&enc_probe.weights).cloned().collect();
        let unused = m.store.unconsumed(&used, &m.ignore);
        if !unused.is_empty() {
            return Err(format!(
                "{fam}: checkpoint tensors no instance reads (list deliberate ones in `ignore`): {}",
                unused.join(", ")
            ));
        }
        let g = &dec_inst.graph;
        let states: HashMap<String, StateDecl> = g
            .values
            .iter()
            .filter_map(|v| match &v.kind {
                ValueKind::State { name, .. } => Some((
                    name.clone(),
                    StateDecl { rows: v.shape[0], numel: v.shape.iter().product() },
                )),
                _ => None,
            })
            .collect();
        let handoff: HashSet<String> =
            enc_probe.outputs.iter().filter(|o| states.contains_key(*o)).cloned().collect();
        let li = dec_inst.output_index("logits").ok_or_else(|| format!("{fam}: the decoder has no `logits` output"))?;
        let vocab = *g.value(g.outputs[li]).shape.last().ok_or("logits have no shape")?;
        let text = EncDecText::load(&rt.text, &dir, &cfg, vocab)?;
        let gen = Gen::parse(&dir, &cfg, src_cap);
        let weights = used.len();
        let dec = GraphProgram::compile(dec_inst.graph.clone(), m.store.materialize(&dec_inst.weights)?, mode)?;
        let enc_shapes = ShapeCache::new(ENCODER);
        enc_shapes.get(&enc_vars, || Ok(enc_probe))?;
        Ok(Self {
            rt,
            text,
            gen,
            src_cap,
            self_cap,
            vocab,
            enc_shapes,
            enc: Mutex::new(None),
            mode,
            dec: Mutex::new(dec),
            dec_inst,
            states,
            handoff,
            weights,
            model,
        })
    }

    /// Nodes of the compiled decoder and encoder programs that run on the CPU.
    pub fn cpu_node_count(&self) -> usize {
        self.dec.lock().cpu_node_count() + self.enc.lock().as_ref().map_or(0, |(_, p)| p.cpu_node_count())
    }

    fn enc_vars(rows: usize) -> Vec<(&'static str, Val)> {
        vec![("rows", Val::Num(rows as f64)), ("cross", Val::Bool(true))]
    }

    /// Runs the encoder over `src` and returns a request with one empty beam.
    fn begin(&self, src: &[u32]) -> Result<Run, String> {
        let n = src.len();
        if n == 0 || n > self.src_cap {
            return Err(format!("source of {n} tokens is outside 1..={}", self.src_cap));
        }
        let bucket = self
            .rt
            .encoder_buckets
            .iter()
            .copied()
            .find(|&b| b >= n && b < self.src_cap)
            .unwrap_or(self.src_cap);
        let vars = Self::enc_vars(bucket);
        let m = &self.model;
        let inst = self
            .enc_shapes
            .get(&vars, || build(&m.spec, ENCODER, &m.env, &m.store, &vars))?;
        let inputs = inst
            .inputs
            .iter()
            .map(|d| {
                Ok(match d.name.as_str() {
                    "token_ids" => {
                        let mut ids = src.to_vec();
                        ids.resize(bucket, 0);
                        Tensor::U32(ids)
                    }
                    "positions" => Tensor::U32((0..bucket as u32).collect()),
                    "key_mask" => Tensor::F32((0..bucket).map(|j| if j < n { 0.0 } else { MASKED }).collect()),
                    other => return Err(format!("encoder input `{other}` is none the encoder-decoder runtime supplies")),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let outs = {
            let mut e = self.enc.lock();
            if e.is_none() {
                let w = m.store.materialize(&inst.weights)?;
                *e = Some((bucket, GraphProgram::compile_reshapable(inst.graph.clone(), w, self.mode)?));
            }
            let (b0, p) = e.as_mut().expect("compiled above");
            // The encoder keeps no state between sources.
            p.state = StateStore::new();
            match *b0 == bucket {
                true => p.run(inputs)?,
                false => p.run_reshaped(&inst.graph, inputs)?,
            }
        };
        let mut state = StateStore::new();
        for (name, t) in inst.outputs.iter().zip(outs) {
            let Some(d) = self.states.get(name) else { continue };
            let mut v = t.into_f32()?;
            v.truncate(n * (d.numel / d.rows));
            v.resize(d.numel, 0.0);
            state.insert(name.clone(), Tensor::F32(v));
        }
        Ok(Run { beams: vec![Beam { state, len: 0 }], n_src: n })
    }

    /// Feeds `tok` at decoder position `b.len`; returns the next-token logits `[vocab]`.
    fn step(&self, b: &mut Beam, n_src: usize, tok: u32) -> Result<Vec<f32>, String> {
        let pos = b.len;
        if pos >= self.self_cap {
            return Err(format!("decoder position {pos} exceeds the compiled {} positions", self.self_cap));
        }
        if tok as usize >= self.vocab {
            return Err(format!("token id {tok} is outside the vocabulary"));
        }
        let inputs = self
            .dec_inst
            .inputs
            .iter()
            .map(|d| {
                let mask = |cap: usize, end: usize| Tensor::F32((0..cap).map(|j| if j < end { 0.0 } else { MASKED }).collect());
                Ok(match d.name.as_str() {
                    "token_ids" => Tensor::U32(vec![tok]),
                    "positions" => Tensor::U32(vec![pos as u32]),
                    "self_mask" => mask(self.self_cap, pos + 1),
                    "cross_mask" => mask(self.src_cap, n_src),
                    other => return Err(format!("decoder input `{other}` is none the encoder-decoder runtime supplies")),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut p = self.dec.lock();
        std::mem::swap(&mut p.state, &mut b.state);
        let out = p.run(inputs);
        std::mem::swap(&mut p.state, &mut b.state);
        let li = self.dec_inst.output_index("logits").expect("checked at load");
        let logits = out?.into_iter().nth(li).ok_or("decoder returned no logits")?.into_f32()?;
        #[cfg(target_os = "macos")]
        {
            // States now in device buffers: the host copies are dead.
            let s = &mut b.state;
            s.buffers.retain(|k, _| !s.metal_buffers.contains_key(k));
        }
        b.len += 1;
        Ok(logits)
    }

    /// A copy of `b` that continues independently: encoder-handed states are shared, the filled
    /// rows of the others copied, paged KV shared by page ([`crate::kv::KvView::fork`]).
    fn fork(&self, b: &Beam) -> Result<Beam, String> {
        // ponytail: on the CPU the host buffers (cross states included) are deep-copied per fork;
        // share them behind an `Arc` if CPU beam search memory matters.
        let mut state = StateStore::new();
        state.buffers = b.state.buffers.clone();
        state.kv = b.state.kv.as_ref().map(|v| v.fork(b.len)).transpose()?;
        #[cfg(target_os = "macos")]
        if !b.state.metal_buffers.is_empty() {
            state.metal_f16 = b.state.metal_f16.clone();
            let device = crate::graph::metal::system_device()?;
            for (name, (buf, n)) in &b.state.metal_buffers {
                if self.handoff.contains(name) {
                    state.metal_buffers.insert(name.clone(), (buf.clone(), *n));
                    continue;
                }
                // ponytail: host memcpy of the filled rows (unified memory) instead of a blit.
                let eb = if b.state.metal_f16.contains(name) { 2 } else { 4 };
                let fresh = crate::graph::metal::new_shared_buffer(&device, *n * eb)?;
                // Row width from the declaration: device buffers may be smaller than declared.
                let filled = self.states.get(name).map_or(*n, |d| (d.numel / d.rows.max(1) * b.len).min(*n));
                // SAFETY: both buffers are shared-storage with at least `filled` elements of `eb`
                // bytes; programs run synchronously, so no GPU work is in flight.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buf.contents() as *const u8,
                        fresh.contents() as *mut u8,
                        filled * eb,
                    );
                }
                state.metal_buffers.insert(name.clone(), (Arc::new(fresh), *n));
            }
        }
        Ok(Beam { state, len: b.len })
    }

    /// One decoder position for every beam (`toks[b]` at `pos`): `[beams, vocab]` logits.
    fn advance(&self, run: &mut Run, toks: &[u32], pos: usize) -> Result<Vec<f32>, String> {
        let mut logits = Vec::with_capacity(toks.len() * self.vocab);
        for (b, &t) in run.beams.iter_mut().zip(toks) {
            debug_assert_eq!(b.len, pos);
            logits.extend(self.step(b, run.n_src, t)?);
        }
        Ok(logits)
    }

    /// Beam `i` continues from beam `parents[i]` (transformers' `reorder_cache`).
    fn reorder(&self, run: &mut Run, parents: Vec<usize>) -> Result<(), String> {
        run.beams = parents.iter().map(|&bi| self.fork(&run.beams[bi])).collect::<Result<_, _>>()?;
        Ok(())
    }

    /// transformers' logits processors on one beam's log-probabilities.
    fn process(&self, lp: &mut [f32], seq: &[u32]) {
        let (g, eos) = (&self.gen, self.text.eos());
        let cur = seq.len();
        for &t in &g.bad {
            if let Some(v) = lp.get_mut(t as usize) {
                *v = f32::NEG_INFINITY;
            }
        }
        if cur < g.min_len {
            lp[eos as usize] = f32::NEG_INFINITY;
        }
        let n = g.no_repeat;
        if n > 0 && cur + 1 >= n {
            let tail = &seq[cur + 1 - n..];
            for w in seq.windows(n) {
                if w[..n - 1] == *tail {
                    lp[w[n - 1] as usize] = f32::NEG_INFINITY;
                }
            }
        }
        if let Some(e) = g.forced_eos.filter(|_| cur + 1 == g.max_len) {
            lp.fill(f32::NEG_INFINITY);
            lp[e as usize] = 0.0;
        }
        if g.renormalize {
            log_softmax(lp);
        }
    }

    /// transformers' beam search: the best hypothesis after `prompt`, ending in EOS unless
    /// `max_length` or the request's `max_new_tokens` cut it; Err once `cancel` is set.
    fn search(&self, src: &[u32], prompt: &[u32], ctx: &PrepareCtx) -> Result<Vec<u32>, String> {
        let eos = self.text.eos();
        let mut run = self.begin(src)?;
        let mut logits = Vec::new();
        for (p, &t) in prompt.iter().enumerate() {
            logits = self.advance(&mut run, &[t], p)?;
        }
        let (k, v, g) = (self.gen.beams.max(1), self.vocab, &self.gen);
        let cap = g
            .max_len
            .min(prompt.len().saturating_add(ctx.max_new_tokens.max(1)));
        let norm = |s: f32, len: usize| s / (len as f32).powf(g.length_penalty);
        let mut beams: Vec<(Vec<u32>, f32)> = vec![(prompt.to_vec(), 0.0)];
        // (normalized score, sequence, ended with EOS)
        let mut done: Vec<(f32, Vec<u32>, bool)> = Vec::new();
        let add = |done: &mut Vec<(f32, Vec<u32>, bool)>, h: (f32, Vec<u32>, bool)| {
            let worst = done.iter().map(|d| d.0).fold(f32::INFINITY, f32::min);
            if done.len() < k || h.0 > worst {
                done.push(h);
                if done.len() > k {
                    let i = (0..done.len())
                        .min_by(|&a, &b| done[a].0.total_cmp(&done[b].0))
                        .unwrap();
                    done.remove(i);
                }
            }
        };
        loop {
            if ctx
                .cancel
                .as_ref()
                .is_some_and(|c| c.load(Ordering::Relaxed))
            {
                // The engine installs no tracing subscriber; stderr reaches the server log.
                eprintln!(
                    "[encdec] beam search cancelled at decoder length {}: client disconnected",
                    beams[0].0.len()
                );
                return Err("cancelled: the client disconnected".into());
            }
            let cur = beams[0].0.len();
            let mut top: Vec<(f32, usize, u32)> = Vec::with_capacity(2 * k + 1);
            for (bi, (seq, score)) in beams.iter().enumerate() {
                let lp = &mut logits[bi * v..(bi + 1) * v];
                log_softmax(lp);
                self.process(lp, seq);
                for (t, &x) in lp.iter().enumerate() {
                    let c = score + x;
                    if c > f32::NEG_INFINITY && (top.len() < 2 * k || c > top[top.len() - 1].0) {
                        let at = top.partition_point(|e| e.0 >= c);
                        top.insert(at, (c, bi, t as u32));
                        top.truncate(2 * k);
                    }
                }
            }
            let mut next: Vec<(usize, u32, f32)> = Vec::with_capacity(k);
            for (rank, &(c, bi, t)) in top.iter().enumerate() {
                if t == eos {
                    if rank < k {
                        add(&mut done, (norm(c, cur), beams[bi].0.clone(), true));
                    }
                } else {
                    next.push((bi, t, c));
                }
                if next.len() == k {
                    break;
                }
            }
            let finished = done.len() >= k
                && (g.early_stopping
                    || top.first().is_none_or(|b| {
                        done.iter().map(|d| d.0).fold(f32::INFINITY, f32::min) >= norm(b.0, cur)
                    }));
            if finished || next.is_empty() {
                break;
            }
            self.reorder(&mut run, next.iter().map(|&(bi, _, _)| bi).collect())?;
            beams = next
                .iter()
                .map(|&(bi, t, c)| {
                    let mut s = beams[bi].0.clone();
                    s.push(t);
                    (s, c)
                })
                .collect();
            if cur + 1 >= cap {
                for (s, c) in beams {
                    let len = s.len() - 1;
                    add(&mut done, (norm(c, len), s, false));
                }
                break;
            }
            let toks: Vec<u32> = beams.iter().map(|b| *b.0.last().unwrap()).collect();
            logits = self.advance(&mut run, &toks, cur)?;
        }
        let (_, seq, ended) = done
            .into_iter()
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .unwrap_or((0.0, prompt.to_vec(), true));
        let mut out = seq[prompt.len()..].to_vec();
        if ended {
            out.push(eos);
        }
        Ok(out)
    }

    fn check_vocab(&self, ids: &[u32]) -> Result<(), String> {
        match ids.iter().find(|&&t| t as usize >= self.vocab) {
            Some(t) => Err(format!("token id {t} is outside the {}-entry vocabulary", self.vocab)),
            None => Ok(()),
        }
    }
}

fn log_softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if m == f32::NEG_INFINITY {
        return;
    }
    let s = x.iter().map(|v| (v - m).exp()).sum::<f32>().ln() + m;
    x.iter_mut().for_each(|v| *v -= s);
}

fn text_only(media: &[MediaInput]) -> Result<(), String> {
    match media.is_empty() {
        true => Ok(()),
        false => Err("this encoder-decoder model takes text only (no image or audio input)".into()),
    }
}

impl NativeLm for SpecEncDec {
    fn backend_label(&self) -> String {
        self.dec.lock().backend_label()
    }

    fn device(&self) -> Device {
        self.dec.lock().primary_device()
    }

    fn kv_pool(&self) -> Option<Arc<crate::kv::KvPool>> {
        self.dec.lock().kv_pool.clone()
    }

    fn summary(&self) -> String {
        let search = match (&self.text, self.gen.beams) {
            (EncDecText::Translation(_), b) if b > 1 => format!("beam search ×{b}"),
            _ => "greedy".into(),
        };
        let langs = match &self.text {
            EncDecText::Translation(t) if t.languages() > 0 => format!(" ({} target languages)", t.languages()),
            _ => String::new(),
        };
        format!("{} spec: {} weights, vocab {}, {search}{langs}", self.model.spec.family, self.weights, self.vocab)
    }

    fn tokenizer(&self) -> &Tokenizer {
        self.text.tokenizer()
    }

    fn context_len(&self) -> usize {
        match &self.text {
            EncDecText::Translation(_) => self.gen.max_len.max(2),
            EncDecText::Prompt { .. } => self.self_cap,
        }
    }

    fn default_max_new_tokens(&self) -> Option<usize> {
        match &self.text {
            EncDecText::Translation(_) => Some(self.gen.max_len),
            EncDecText::Prompt { .. } => None,
        }
    }

    fn prepare(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<(Vec<u32>, SeqState), String> {
        let ctx = PrepareCtx {
            max_new_tokens: usize::MAX,
            cancel: None,
        };
        self.prepare_ctx(ids, media, &ctx)
    }

    fn prepare_ctx(
        &self,
        ids: Vec<u32>,
        media: &[MediaInput],
        ctx: &PrepareCtx,
    ) -> Result<(Vec<u32>, SeqState), String> {
        text_only(media)?;
        let (src, prompt) = self.text.source(&ids)?;
        self.check_vocab(&src)?;
        let n = prompt.len();
        let replay = match &self.text {
            EncDecText::Translation(t) => self.gen.beams > 1 || t.needs_replay(),
            EncDecText::Prompt { .. } => false,
        };
        if !replay {
            let live = Live {
                run: self.begin(&src)?,
                seq: Vec::new(),
                prompt: n,
            };
            return Ok((prompt, Box::new(live)));
        }
        let EncDecText::Translation(t) = &self.text else {
            unreachable!("only translation texts replay")
        };
        let out = t.display(&self.search(&src, &prompt, ctx)?);
        Ok((prompt, Box::new(Replay { prompt: n, out })))
    }

    fn prompt_len(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<usize, String> {
        text_only(media)?;
        Ok(self.text.source(&ids)?.1.len())
    }

    fn forward(
        &self,
        tokens: &[u32],
        start_pos: usize,
        state: &mut SeqState,
    ) -> Result<Vec<f32>, String> {
        let t = match &self.text {
            EncDecText::Translation(t) => Some(t),
            EncDecText::Prompt { .. } => None,
        };
        let vsize = self.tokenizer().vocab_size();
        if let Some(r) = state.downcast_ref::<Replay>() {
            let mut logits = vec![-1e9f32; vsize];
            let end = start_pos + tokens.len();
            if end >= r.prompt {
                let id = r.out.get(end - r.prompt).copied().unwrap_or(self.text.eos());
                logits[id as usize] = 0.0;
            }
            return Ok(logits);
        }
        let live = state
            .downcast_mut::<Live>()
            .ok_or("sequence state is not an encoder-decoder state")?;
        if start_pos > live.seq.len() {
            return Err(format!(
                "decoder position {start_pos} is past the {} decoded",
                live.seq.len()
            ));
        }
        // Rewinding is a length reset: later rows are masked and overwritten.
        live.seq.truncate(start_pos);
        live.run.beams[0].len = start_pos;
        let alias = t.map_or(u32::MAX, |t| t.alias);
        let mut lp = Vec::new();
        for &tok in tokens {
            // The first output token is sampled as its display alias.
            let tok = if tok >= alias { tok - alias } else { tok };
            let n_src = live.run.n_src;
            lp = self.step(&mut live.run.beams[0], n_src, tok)?;
            live.seq.push(tok);
        }
        let Some(t) = t else {
            return Ok(lp);
        };
        let mut logits = vec![-1e9f32; vsize];
        if live.seq.len() < live.prompt {
            return Ok(logits);
        }
        log_softmax(&mut lp);
        self.process(&mut lp, &live.seq);
        let off = if live.seq.len() == live.prompt {
            t.alias as usize
        } else {
            0
        };
        let n = lp.len().min(t.alias as usize);
        for (o, &x) in logits[off..off + n].iter_mut().zip(&lp[..n]) {
            *o = if x.is_finite() { x } else { -1e9 };
        }
        Ok(logits)
    }
}
