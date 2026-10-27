//! Online forward-only `qlabs.sh/noise` block drafter + suffix-automaton lattice ([`OnlineNoiseDrafter`]),
//! with live-tunable [`DraftKnobs`] / [`LiveDraftKnobs`].
//!
//! Combines:
//! 1. [`SuffixLattice`]: zero-parameter n-gram suffix continuation table ($n \in \{1..=4\}$) with
//!    recency decay for high-acceptance token chains (code, JSON, tool calls, repeated identifiers).
//! 2. [`NoiseBlockHead`]: a multi-offset block drafter head over sketches of the target's own token
//!    embeddings ([`EmbedFn`]), trained online from committed target tokens.

use crate::scheduler::spec::{DraftState, Drafter};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::atomic::{AtomicU32, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

/// Bottleneck dimension of [`NoiseBlockHead`] (power of two for exact Walsh-Hadamard transform).
pub const NOISE_BOTTLENECK_DIM: usize = 128;
/// Maximum speculative lookahead offsets maintained by [`NoiseBlockHead`].
pub const NOISE_MAX_K: usize = 15;
/// Shortlist vocabulary capacity tracked online by [`NoiseBlockHead`].
pub const NOISE_SHORTLIST_VOCAB: usize = 256;
/// Committed tokens one update learns from at most (the prompt's tail on the first).
const LEARN_SPAN: usize = 64;

/// Speculative drafting mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum DraftMode {
    /// Use the model's built-in drafter (e.g. MTP) when present, or online noise+suffix when attached.
    #[default]
    Auto,
    /// Disable speculative decoding (`k = 0`).
    Off,
    /// Use only the model's built-in / sidecar drafter (`MTP` / `DFlash`).
    Model,
    /// Use the online `qlabs.sh/noise` neural block head + suffix lattice drafter.
    Noise,
    /// Use only the zero-overhead suffix lattice drafter.
    Suffix,
}

impl DraftMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::Model => "model",
            Self::Noise => "noise",
            Self::Suffix => "suffix",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Auto => Self::Noise,
            Self::Noise => Self::Suffix,
            Self::Suffix => Self::Model,
            Self::Model => Self::Off,
            Self::Off => Self::Auto,
        }
    }

    fn to_u8(self) -> u8 {
        match self {
            Self::Auto => 0,
            Self::Off => 1,
            Self::Model => 2,
            Self::Noise => 3,
            Self::Suffix => 4,
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Off,
            2 => Self::Model,
            3 => Self::Noise,
            4 => Self::Suffix,
            _ => Self::Auto,
        }
    }
}

impl FromStr for DraftMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "off" | "none" | "0" | "false" => Ok(Self::Off),
            "model" | "mtp" | "dflash" => Ok(Self::Model),
            "noise" | "online" | "online-noise" | "qlabs" => Ok(Self::Noise),
            "suffix" | "ngram" | "lookup" => Ok(Self::Suffix),
            other => Err(format!(
                "unknown draft mode '{other}'; valid: auto, noise, suffix, model, off"
            )),
        }
    }
}

/// User-facing tuning knobs for speculative decoding and [`OnlineNoiseDrafter`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DraftKnobs {
    pub mode: DraftMode,
    /// Maximum draft tokens per step (`0..=15`, default `4`).
    pub max_k: usize,
    /// Minimum acceptance rate (`0.30..=0.95`, default `0.65`) before [`KTuner`] penalizes deeper `k`.
    pub target_acceptance: f32,
    /// Weight (`0.0..=1.0`, default `0.25`) on verify compute cost vs raw wall-clock tok/s in [`KTuner`].
    pub cost_weight: f32,
    /// Minimum suffix n-gram match length (`1..=4`, default `2`) before [`SuffixLattice`] proposes.
    pub suffix_min_match: usize,
    /// Online learning rate (`0.0..=0.05`, default `0.005`) for [`NoiseBlockHead`].
    pub noise_lr: f32,
    /// Hadamard perturbation scale $\sigma$ (`1e-4..=0.1`, default `0.01`).
    pub noise_sigma: f32,
}

impl Default for DraftKnobs {
    fn default() -> Self {
        Self {
            mode: DraftMode::Auto,
            max_k: 4,
            target_acceptance: 0.65,
            cost_weight: 0.25,
            suffix_min_match: 2,
            noise_lr: 0.005,
            noise_sigma: 0.01,
        }
    }
}

impl DraftKnobs {
    pub fn sanitized(mut self) -> Self {
        self.max_k = self.max_k.min(NOISE_MAX_K);
        self.target_acceptance = self.target_acceptance.clamp(0.10, 0.98);
        self.cost_weight = self.cost_weight.clamp(0.0, 1.0);
        self.suffix_min_match = self.suffix_min_match.clamp(1, 4);
        self.noise_lr = self.noise_lr.clamp(0.0, 0.10);
        self.noise_sigma = self.noise_sigma.clamp(1e-5, 0.25);
        self
    }
}

/// Lock-free atomic container for [`DraftKnobs`], shared across the scheduler, [`KTuner`], and
/// [`OnlineNoiseDrafter`] so live edits take effect on the very next decode step.
#[derive(Debug)]
pub struct LiveDraftKnobs {
    mode: AtomicU8,
    max_k: AtomicUsize,
    target_acceptance: AtomicU32,
    cost_weight: AtomicU32,
    suffix_min_match: AtomicUsize,
    noise_lr: AtomicU32,
    noise_sigma: AtomicU32,
}

impl Default for LiveDraftKnobs {
    fn default() -> Self {
        Self::new(DraftKnobs::default())
    }
}

impl LiveDraftKnobs {
    pub fn new(knobs: DraftKnobs) -> Self {
        let k = knobs.sanitized();
        Self {
            mode: AtomicU8::new(k.mode.to_u8()),
            max_k: AtomicUsize::new(k.max_k),
            target_acceptance: AtomicU32::new(k.target_acceptance.to_bits()),
            cost_weight: AtomicU32::new(k.cost_weight.to_bits()),
            suffix_min_match: AtomicUsize::new(k.suffix_min_match),
            noise_lr: AtomicU32::new(k.noise_lr.to_bits()),
            noise_sigma: AtomicU32::new(k.noise_sigma.to_bits()),
        }
    }

    pub fn snapshot(&self) -> DraftKnobs {
        DraftKnobs {
            mode: DraftMode::from_u8(self.mode.load(Ordering::Relaxed)),
            max_k: self.max_k.load(Ordering::Relaxed),
            target_acceptance: f32::from_bits(self.target_acceptance.load(Ordering::Relaxed)),
            cost_weight: f32::from_bits(self.cost_weight.load(Ordering::Relaxed)),
            suffix_min_match: self.suffix_min_match.load(Ordering::Relaxed),
            noise_lr: f32::from_bits(self.noise_lr.load(Ordering::Relaxed)),
            noise_sigma: f32::from_bits(self.noise_sigma.load(Ordering::Relaxed)),
        }
    }

    pub fn store(&self, knobs: DraftKnobs) {
        let k = knobs.sanitized();
        self.mode.store(k.mode.to_u8(), Ordering::Relaxed);
        self.max_k.store(k.max_k, Ordering::Relaxed);
        self.target_acceptance
            .store(k.target_acceptance.to_bits(), Ordering::Relaxed);
        self.cost_weight
            .store(k.cost_weight.to_bits(), Ordering::Relaxed);
        self.suffix_min_match
            .store(k.suffix_min_match, Ordering::Relaxed);
        self.noise_lr.store(k.noise_lr.to_bits(), Ordering::Relaxed);
        self.noise_sigma
            .store(k.noise_sigma.to_bits(), Ordering::Relaxed);
    }
}

// ---- Structured Orthogonal Hadamard Perturbations (`qlabs.sh/noise`) --------------------------

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// In-place normalized Fast Walsh-Hadamard Transform ($H^\top H = I$, `buf.len()` power of two).
pub fn fwht_in_place(buf: &mut [f32]) {
    let n = buf.len();
    assert!(
        n.is_power_of_two() && n > 0,
        "FWHT requires power-of-two length"
    );
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..(i + h) {
                let x = buf[j];
                let y = buf[j + h];
                buf[j] = x + y;
                buf[j + h] = x - y;
            }
        }
        h *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for x in buf.iter_mut() {
        *x *= scale;
    }
}

/// Generates `num_dirs` mutually orthogonal perturbation vectors in $\mathbb{R}^{\text{dim}}$
/// (`dim` power of two) with per-coordinate variance $\approx 1$ (`||v||^2 = dim`) using
/// structured Hadamard Rademacher bases ($v_k = \sqrt{d} \, H D_1 e_{\pi(k)}$).
pub fn hadamard_perturbations(dim: usize, num_dirs: usize, seed: u64) -> Vec<Vec<f32>> {
    assert!(dim.is_power_of_two() && num_dirs <= dim);
    let mut signs = vec![1.0f32; dim];
    for (i, s) in signs.iter_mut().enumerate() {
        let h = splitmix64(seed ^ (i as u64).wrapping_mul(0x517C_C1B7_2722_0A95));
        *s = if (h & 1) == 0 { 1.0 } else { -1.0 };
    }
    // Select `num_dirs` distinct basis indices via a bijective power-of-two stride.
    let stride = ((splitmix64(seed ^ 0xA5A5_5A5A_F0F0_0F0F) as usize) | 1) & (dim - 1);
    let offset = (splitmix64(seed ^ 0x1234_5678_9ABC_DEF0) as usize) & (dim - 1);
    let scale = (dim as f32).sqrt();
    (0..num_dirs)
        .map(|k| {
            let idx = (offset + k.wrapping_mul(stride)) & (dim - 1);
            let mut v = vec![0.0f32; dim];
            v[idx] = signs[idx] * scale;
            fwht_in_place(&mut v);
            // Post-multiply by a second deterministic Rademacher diagonal D2: preserves exact
            // pairwise orthogonality: <D2 u, D2 w> = <u, w> = 0.
            for (i, x) in v.iter_mut().enumerate() {
                let d2 = if (splitmix64(seed.wrapping_add(0x7777) ^ (i as u64)) & 2) == 0 {
                    1.0
                } else {
                    -1.0
                };
                *x *= d2;
            }
            v
        })
        .collect()
}

// ---- SuffixLattice ----------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct SuffixKey {
    len: u8,
    toks: [u32; 4],
}

impl SuffixKey {
    fn from_slice(s: &[u32]) -> Self {
        let mut toks = [0u32; 4];
        for (dst, &src) in toks.iter_mut().zip(s) {
            *dst = src;
        }
        Self {
            len: s.len() as u8,
            toks,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct ContSlot {
    tok: u32,
    weight: f32,
}

/// Fast n-gram suffix continuation table ($n \in \{1..=4\}$) with exponential recency decay.
#[derive(Default, Clone)]
pub struct SuffixLattice {
    table: HashMap<SuffixKey, [ContSlot; 4]>,
}

impl SuffixLattice {
    pub fn new() -> Self {
        Self::default()
    }

    /// Indexes all $1..=4$-gram suffixes and their immediate continuation token in `tokens`.
    pub fn observe_tokens(&mut self, tokens: &[u32]) {
        if tokens.len() < 2 {
            return;
        }
        for i in 0..tokens.len() - 1 {
            let next = tokens[i + 1];
            let max_n = (i + 1).min(4);
            for n in 1..=max_n {
                let key = SuffixKey::from_slice(&tokens[i + 1 - n..=i]);
                let slots = self.table.entry(key).or_default();
                for s in slots.iter_mut() {
                    s.weight *= 0.985;
                }
                if let Some(existing) = slots.iter_mut().find(|s| s.weight > 0.0 && s.tok == next) {
                    existing.weight += 1.0;
                } else if let Some(target) = slots
                    .iter_mut()
                    .min_by(|a, b| a.weight.total_cmp(&b.weight))
                {
                    *target = ContSlot {
                        tok: next,
                        weight: 1.0,
                    };
                }
            }
        }
    }

    fn best_next(&self, suffix: &[u32]) -> Option<(u32, f32)> {
        let slots = self.table.get(&SuffixKey::from_slice(suffix))?;
        slots
            .iter()
            .filter(|s| s.weight > 0.05)
            .max_by(|a, b| a.weight.total_cmp(&b.weight))
            .map(|s| (s.tok, s.weight))
    }

    /// Proposes up to `k` tokens by walking suffix matches of length `>= min_match`.
    /// Returns `(draft_tokens, initial_match_len)`.
    pub fn propose(
        &self,
        context: &[u32],
        k: usize,
        min_match: usize,
    ) -> Option<(Vec<u32>, usize)> {
        if context.is_empty() || k == 0 {
            return None;
        }
        let min_n = min_match.clamp(1, 4);
        let mut working = context[context.len().saturating_sub(8)..].to_vec();
        let mut drafts = Vec::with_capacity(k);
        let mut first_match_len = 0;

        for step in 0..k {
            let min_try = if step == 0 { min_n } else { min_n.min(2) };
            let max_try = working.len().min(4);
            if max_try < min_try {
                break;
            }
            let mut picked = None;
            for n in (min_try..=max_try).rev() {
                let suf = &working[working.len() - n..];
                if let Some((tok, _w)) = self.best_next(suf) {
                    picked = Some((tok, n));
                    break;
                }
            }
            let Some((tok, n)) = picked else {
                break;
            };
            if step == 0 {
                first_match_len = n;
            }
            drafts.push(tok);
            working.push(tok);
        }

        if drafts.is_empty() {
            None
        } else {
            Some((drafts, first_match_len))
        }
    }
}

// ---- Token sketches of the target's embedding table ------------------------------------------

/// Writes row `id` of the target's token-embedding table into `out` (the table's width); `false`
/// when the target exposes no table or `id` is out of range.
pub type EmbedFn = Arc<dyn Fn(u32, &mut [f32]) -> bool + Send + Sync>;

/// Width of one token sketch.
const SKETCH_DIM: usize = 256;
/// Token sketches cached (least recently used evicted).
const SKETCH_CACHE: usize = 4096;
/// Context tokens whose sketches form the drafter input.
const CONTEXT_WINDOW: usize = 4;

/// Fixed random features of the target's token embeddings: a seeded sign flip, the normalized fast
/// Walsh-Hadamard transform over `hidden` zero-padded to a power of two, and its first
/// [`SKETCH_DIM`] coordinates, rescaled to unit RMS so features do not depend on the table's scale.
struct Sketcher {
    embed: EmbedFn,
    hidden: usize,
    signs: Vec<f32>,
    buf: Vec<f32>,
    /// Token -> (last use, sketch).
    cache: HashMap<u32, (u64, Box<[f32]>)>,
    clock: u64,
}

impl Sketcher {
    fn new(embed: EmbedFn, hidden: usize) -> Self {
        let n = hidden.max(SKETCH_DIM).next_power_of_two();
        let signs = (0..n as u64)
            .map(|i| {
                if splitmix64(0x5EED_5EED ^ i) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        Self {
            embed,
            hidden,
            signs,
            buf: vec![0.0; n],
            cache: HashMap::new(),
            clock: 0,
        }
    }

    /// The sketch of token `id`; `None` when the target cannot embed it.
    fn sketch(&mut self, id: u32) -> Option<&[f32]> {
        self.clock += 1;
        if !self.cache.contains_key(&id) {
            self.buf.fill(0.0);
            if !(self.embed)(id, &mut self.buf[..self.hidden]) {
                return None;
            }
            for (v, s) in self.buf.iter_mut().zip(&self.signs) {
                *v *= s;
            }
            fwht_in_place(&mut self.buf);
            let mut s: Box<[f32]> = self.buf[..SKETCH_DIM].into();
            rms_norm(&mut s);
            if self.cache.len() >= SKETCH_CACHE {
                // ponytail: O(cache) scan per eviction; an intrusive LRU list if misses dominate.
                if let Some(old) = self.cache.iter().min_by_key(|e| e.1 .0).map(|e| *e.0) {
                    self.cache.remove(&old);
                }
            }
            self.cache.insert(id, (0, s));
        }
        let e = self.cache.get_mut(&id)?;
        e.0 = self.clock;
        Some(&e.1)
    }

    /// The drafter's input after `ctx`: the sketches of its last `window` tokens, newest first
    /// (zeros pad a short context), RMS-normalized together.
    fn context(&mut self, ctx: &[u32], window: usize) -> Vec<f32> {
        let mut x = vec![0.0f32; window * SKETCH_DIM];
        for (i, &t) in ctx.iter().rev().take(window).enumerate() {
            if let Some(s) = self.sketch(t) {
                x[i * SKETCH_DIM..(i + 1) * SKETCH_DIM].copy_from_slice(s);
            }
        }
        rms_norm(&mut x);
        x
    }
}

// ---- NoiseBlockHead ---------------------------------------------------------------------------

const D: usize = NOISE_BOTTLENECK_DIM;
/// Step multiplier of the shortlist heads over `noise_lr`.
const HEAD_LR: f32 = 4.0;

/// Weights updated in ANVIL II moment form, with their first and second moments.
struct Param {
    w: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl Param {
    fn new(w: Vec<f32>) -> Self {
        let n = w.len();
        Self {
            w,
            m: vec![0.0; n],
            v: vec![1e-3; n],
        }
    }

    /// One ANVIL II step on element `i` along gradient `g`.
    #[inline]
    fn step(&mut self, i: usize, g: f32, lr: f32) {
        let m = 0.85 * self.m[i] + 0.15 * g;
        let v = (0.95 * self.v[i] + 0.05 * g * g).max(1e-4);
        self.m[i] = m;
        self.v[i] = v;
        self.w[i] -= lr * m / (v.sqrt() + 1e-3);
    }

    /// Zero weights and fresh moments on `r`.
    fn reset(&mut self, r: std::ops::Range<usize>) {
        self.w[r.clone()].fill(0.0);
        self.m[r.clone()].fill(0.0);
        self.v[r].fill(1e-3);
    }
}

/// Online multi-offset block drafter head: context `c = rmsnorm(W_c x)` over token sketches `x`,
/// and per lookahead offset `j` shortlist logits `H_j c`.
pub struct NoiseBlockHead {
    vocab: usize,
    window: usize,
    cap: usize,
    /// `[D, window * SKETCH_DIM]`.
    w_c: Param,
    /// Per offset `j < NOISE_MAX_K`: `[cap, D]`.
    heads: Vec<Param>,
    /// Active shortlist of token ids (`<= cap`), their decayed counts and slots.
    ids: Vec<u32>,
    counts: Vec<f32>,
    slot_of: HashMap<u32, usize>,
    updates: u64,
}

impl NoiseBlockHead {
    fn new(vocab: usize, window: usize, cap: usize) -> Self {
        let vocab = vocab.max(1);
        let cap = cap.min(vocab);
        let n = window * SKETCH_DIM;
        let scale = 1.0 / (n as f32).sqrt();
        let w_c = (0..(D * n) as u64)
            .map(|i| {
                if splitmix64(0xC0FF_EE00 ^ i) & 1 == 0 {
                    scale
                } else {
                    -scale
                }
            })
            .collect();
        Self {
            vocab,
            window,
            cap,
            w_c: Param::new(w_c),
            heads: (0..NOISE_MAX_K)
                .map(|_| Param::new(vec![0.0; cap * D]))
                .collect(),
            ids: Vec::with_capacity(cap),
            counts: Vec::with_capacity(cap),
            slot_of: HashMap::with_capacity(cap),
            updates: 0,
        }
    }

    /// Registers `tok` in the shortlist (replacing the lowest-count slot when full); its slot.
    fn ensure_shortlist(&mut self, tok: u32) -> usize {
        let tok = tok.min((self.vocab - 1) as u32);
        if let Some(&slot) = self.slot_of.get(&tok) {
            self.counts[slot] += 1.0;
            return slot;
        }
        if self.ids.len() < self.cap {
            let slot = self.ids.len();
            self.ids.push(tok);
            self.counts.push(1.0);
            self.slot_of.insert(tok, slot);
            return slot;
        }
        let (slot, _) = self
            .counts
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap_or((0, &0.0));
        self.slot_of.remove(&self.ids[slot]);
        self.ids[slot] = tok;
        self.counts[slot] = 1.0;
        self.slot_of.insert(tok, slot);
        for h in &mut self.heads {
            h.reset(slot * D..(slot + 1) * D);
        }
        slot
    }

    /// `c = rmsnorm(W_c x)`.
    fn context(&self, x: &[f32]) -> Vec<f32> {
        let mut c: Vec<f32> = self.w_c.w.chunks_exact(x.len()).map(|r| dot(r, x)).collect();
        rms_norm(&mut c);
        c
    }

    /// Shortlist logits `H_j c`, one per active slot.
    fn logits(&self, j: usize, c: &[f32], out: &mut Vec<f32>) {
        out.clear();
        out.extend(
            self.heads[j]
                .w
                .chunks_exact(D)
                .take(self.ids.len())
                .map(|r| dot(r, c)),
        );
    }

    /// Exact local delta rule on `H_j` toward slot `target` from `c`: `ΔH_j ∝ (p - y) ⊗ c`.
    fn delta(&mut self, j: usize, c: &[f32], target: usize, lr: f32) {
        let mut p = Vec::new();
        self.logits(j, c, &mut p);
        softmax(&mut p);
        let h = &mut self.heads[j];
        for (s, &ps) in p.iter().enumerate() {
            let err = ps - if s == target { 1.0 } else { 0.0 };
            if err.abs() < 1e-4 {
                continue;
            }
            for (d, &cd) in c.iter().enumerate() {
                let i = s * D + d;
                h.step(i, err * cd + 1e-4 * h.w[i], lr * HEAD_LR);
            }
        }
    }
}

/// Dot product with 8 independent accumulators (vectorizes).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ca
        .remainder()
        .iter()
        .zip(cb.remainder())
        .map(|(x, y)| x * y)
        .sum();
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

fn softmax(z: &mut [f32]) {
    let max = z.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0;
    for v in z.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in z.iter_mut() {
        *v /= sum;
    }
}

fn rms_norm(x: &mut [f32]) {
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (x.len().max(1) as f32);
    let inv = 1.0 / (mean_sq + 1e-5).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// The head and the sketches it reads, shared by every sequence.
struct Inner {
    head: NoiseBlockHead,
    sk: Sketcher,
}

impl Inner {
    /// Teacher-forced update on committed `tokens`: every (context end `p`, offset `j`) pair whose
    /// target `tokens[p + j]` sits at index `from..` trains once.
    fn learn(&mut self, tokens: &[u32], from: usize, max_k: usize, lr: f32) {
        let Self { head, sk } = self;
        let to = tokens.len();
        if from >= to || lr <= 0.0 {
            return;
        }
        for c in head.counts.iter_mut() {
            *c *= 0.995;
        }
        let slots: Vec<usize> = tokens[from..]
            .iter()
            .map(|&t| head.ensure_shortlist(t))
            .collect();
        if head.ids.len() < 2 {
            return;
        }
        for p in (from + 1).saturating_sub(max_k).max(1)..to {
            let x = sk.context(&tokens[..p], head.window);
            let c = head.context(&x);
            for j in from.saturating_sub(p)..max_k.min(to - p) {
                let slot = slots[p + j - from];
                // A later token of this batch may have taken the slot.
                if head.ids[slot] == tokens[p + j] {
                    head.delta(j, &c, slot, lr);
                }
            }
        }
        head.updates += 1;
    }

    /// Shortlist logits (over `head.ids`) for offsets `0..k` after `ctx`.
    fn draft(&mut self, ctx: &[u32], k: usize) -> Vec<Vec<f32>> {
        let x = self.sk.context(ctx, self.head.window);
        let c = self.head.context(&x);
        (0..k)
            .map(|j| {
                let mut z = Vec::new();
                self.head.logits(j, &c, &mut z);
                z
            })
            .collect()
    }
}

// ---- OnlineNoiseDrafter -----------------------------------------------------------------------

/// Per-sequence state for [`OnlineNoiseDrafter`].
struct OnlineNoiseState {
    /// Committed tokens already ingested into the suffix lattices and the head.
    seen: usize,
    /// Local sequence suffix lattice (seeded immediately from the sequence's own prompt).
    local: SuffixLattice,
}

/// Hybrid online drafter combining [`SuffixLattice`] and [`NoiseBlockHead`].
pub struct OnlineNoiseDrafter {
    vocab: usize,
    /// The target embeds tokens: the head can run (else suffix-only drafting).
    embeds: bool,
    knobs: Arc<LiveDraftKnobs>,
    lattice: RwLock<SuffixLattice>,
    inner: Mutex<Inner>,
}

impl OnlineNoiseDrafter {
    /// A drafter over the target's token embeddings `embed` (`hidden` wide) and `vocab` ids.
    pub fn new(
        embed: EmbedFn,
        hidden: usize,
        vocab: usize,
        knobs: Arc<LiveDraftKnobs>,
    ) -> Arc<Self> {
        let embeds = hidden > 0 && embed(0, &mut vec![0.0; hidden]);
        Arc::new(Self {
            vocab: vocab.max(1),
            embeds,
            knobs,
            lattice: RwLock::new(SuffixLattice::new()),
            inner: Mutex::new(Inner {
                head: NoiseBlockHead::new(vocab, CONTEXT_WINDOW, NOISE_SHORTLIST_VOCAB),
                sk: Sketcher::new(embed, hidden),
            }),
        })
    }

    pub fn knobs(&self) -> &Arc<LiveDraftKnobs> {
        &self.knobs
    }

    pub fn online_updates(&self) -> u64 {
        self.inner.lock().head.updates
    }
}

impl Drafter for OnlineNoiseDrafter {
    fn label(&self) -> String {
        match self.knobs.snapshot().mode {
            DraftMode::Auto => "auto → online-noise+suffix".to_string(),
            DraftMode::Suffix => "suffix-lattice".to_string(),
            _ => "online-noise+suffix".to_string(),
        }
    }

    fn tap_layers(&self) -> Vec<usize> {
        Vec::new()
    }

    fn max_k(&self) -> usize {
        let k = self.knobs.snapshot();
        if k.mode == DraftMode::Off {
            0
        } else {
            k.max_k.min(NOISE_MAX_K)
        }
    }

    fn new_state(&self) -> DraftState {
        Box::new(OnlineNoiseState {
            seen: 0,
            local: SuffixLattice::new(),
        })
    }

    fn propose(
        &self,
        st: &mut DraftState,
        tokens: &[u32],
        _hidden: &[f32],
        k: usize,
        pick: &mut dyn FnMut(&[f32]) -> u32,
    ) -> Result<Vec<u32>, String> {
        let s = st
            .downcast_mut::<OnlineNoiseState>()
            .ok_or("draft state was not built by OnlineNoiseDrafter")?;
        let knobs = self.knobs.snapshot();
        if knobs.mode == DraftMode::Noise && !self.embeds {
            return Err("online drafter needs the target's token embeddings".into());
        }
        let neural = self.embeds && !matches!(knobs.mode, DraftMode::Suffix | DraftMode::Off);
        let k = k.min(NOISE_MAX_K);
        let Some(&anchor) = tokens.last() else {
            return Ok(Vec::new());
        };

        // 1. Ingest newly committed tokens into the local and shared suffix lattices; the head
        // learns from them even on `k == 0` steps.
        if tokens.len() > s.seen {
            let slice = &tokens[s.seen.saturating_sub(4)..];
            s.local.observe_tokens(slice);
            self.lattice.write().observe_tokens(slice);
            if neural {
                let from = s.seen.max(tokens.len().saturating_sub(LEARN_SPAN));
                self.inner.lock().learn(
                    tokens,
                    from,
                    knobs.max_k.clamp(1, NOISE_MAX_K),
                    knobs.noise_lr,
                );
            }
            s.seen = tokens.len();
        }
        if k == 0 || knobs.mode == DraftMode::Off {
            return Ok(Vec::new());
        }

        // 2. Query SuffixLattice (local prompt lattice first, then global cross-sequence lattice).
        let suffix_drafts = s
            .local
            .propose(tokens, k, knobs.suffix_min_match)
            .or_else(|| {
                self.lattice
                    .read()
                    .propose(tokens, k, knobs.suffix_min_match)
            });
        // In Suffix-only mode without a suffix match, return empty drafts (plain decode).
        if knobs.mode == DraftMode::Suffix && suffix_drafts.is_none() {
            return Ok(Vec::new());
        }
        let suf_slice = suffix_drafts
            .as_ref()
            .map(|(v, _)| v.as_slice())
            .unwrap_or(&[]);

        // 3. The head's shortlist logits per offset.
        let (ids, rows, head_updates) = if neural {
            let mut inner = self.inner.lock();
            let rows = inner.draft(tokens, k);
            (inner.head.ids.clone(), rows, inner.head.updates)
        } else {
            (Vec::new(), Vec::new(), 0)
        };

        let mut logits = vec![-20.0f32; self.vocab];
        let mut drafts = Vec::with_capacity(k);
        for offset in 0..k {
            logits.fill(-20.0);
            let mut has_candidate = false;
            let mut top_prob = 0.0f32;
            if let Some(z) = rows.get(offset).filter(|z| !z.is_empty()) {
                let mut p = z.clone();
                softmax(&mut p);
                top_prob = p.iter().fold(0.0f32, |a, &b| a.max(b));
                for (&tok, &l) in ids.iter().zip(z) {
                    if (tok as usize) < self.vocab {
                        logits[tok as usize] = l;
                        has_candidate = true;
                    }
                }
            }

            let has_suffix = if let Some(&suf_tok) = suf_slice.get(offset) {
                if (suf_tok as usize) < self.vocab {
                    // Suffix match boosts the candidate logit strongly while keeping valid distribution support.
                    logits[suf_tok as usize] = logits[suf_tok as usize].max(0.0) + 14.0;
                    has_candidate = true;
                }
                true
            } else {
                false
            };

            if knobs.mode == DraftMode::Suffix && !has_suffix {
                break;
            }
            if knobs.mode == DraftMode::Auto
                && !has_suffix
                && !(head_updates >= 4 && top_prob >= knobs.target_acceptance)
            {
                break;
            }

            if !has_candidate {
                logits[anchor as usize % self.vocab] = 0.0;
            }
            drafts.push(pick(&logits));
        }

        Ok(drafts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic pseudo-random embedding table.
    fn table(hidden: usize) -> EmbedFn {
        Arc::new(move |id, out: &mut [f32]| {
            for (i, o) in out.iter_mut().take(hidden).enumerate() {
                let h = splitmix64(((id as u64) << 32) ^ i as u64);
                *o = (h as u32 as f32) / (u32::MAX as f32) - 0.5;
            }
            true
        })
    }

    #[test]
    fn fwht_and_hadamard_perturbations_are_orthogonal() {
        let dim = NOISE_BOTTLENECK_DIM;
        let dirs = hadamard_perturbations(dim, 8, 42);
        assert_eq!(dirs.len(), 8);
        for (i, u) in dirs.iter().enumerate() {
            let norm_sq: f32 = u.iter().map(|x| x * x).sum();
            assert!(
                (norm_sq - dim as f32).abs() < 1e-3,
                "dir {i} norm_sq = {norm_sq}, expected {dim}"
            );
            for (j, v) in dirs.iter().enumerate().skip(i + 1) {
                let dot: f32 = u.iter().zip(v).map(|(a, b)| a * b).sum();
                assert!(
                    dot.abs() < 1e-3,
                    "dirs ({i}, {j}) not orthogonal: dot = {dot}"
                );
            }
        }
    }

    #[test]
    fn suffix_lattice_recovers_exact_repeated_continuations() {
        let mut lat = SuffixLattice::new();
        // Pattern: [10, 20, 30, 40, 50, 60]
        let seq = [1, 2, 10, 20, 30, 40, 50, 60, 7, 8, 10, 20, 30, 40, 50, 60];
        lat.observe_tokens(&seq);
        let (drafts, match_len) = lat.propose(&[99, 10, 20, 30], 3, 2).expect("suffix match");
        assert_eq!(match_len, 3);
        assert_eq!(drafts, vec![40, 50, 60]);
    }

    /// Committed one token at a time, a repeating sequence teaches every offset its continuation.
    #[test]
    fn noise_block_head_learns_a_repeating_sequence() {
        let mut inner = Inner {
            head: NoiseBlockHead::new(64, CONTEXT_WINDOW, 64),
            sk: Sketcher::new(table(48), 48),
        };
        let cycle = [5u32, 12, 19, 27, 3, 8, 40, 33];
        let seq: Vec<u32> = cycle.iter().cycle().take(400).copied().collect();
        for i in 2..seq.len() {
            inner.learn(&seq[..i], i - 1, 3, 0.005);
        }
        let rows = inner.draft(&seq[..8 * 40 + 1], 3);
        for (j, z) in rows.iter().enumerate() {
            let best = z.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            assert_eq!(inner.head.ids[best], cycle[(j + 1) % 8], "offset {j}");
        }
    }
}
