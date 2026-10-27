//! Online forward-only `qlabs.sh/noise` block drafter + suffix-automaton lattice ([`OnlineNoiseDrafter`]),
//! with live-tunable [`DraftKnobs`] / [`LiveDraftKnobs`].
//!
//! Combines:
//! 1. [`SuffixLattice`]: zero-parameter n-gram suffix continuation table ($n \in \{1..=4\}$) with
//!    recency decay for high-acceptance token chains (code, JSON, tool calls, repeated identifiers).
//! 2. [`NoiseBlockHead`]: a multi-offset block drafter head over sketches of the target's own token
//!    embeddings ([`EmbedFn`]), trained online from committed target tokens.

use crate::scheduler::spec::{DraftLogits, DraftState, Drafter};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;

/// Bottleneck dimension of [`NoiseBlockHead`] (power of two for exact Walsh-Hadamard transform).
pub const NOISE_BOTTLENECK_DIM: usize = 128;
/// Maximum speculative lookahead offsets maintained by [`NoiseBlockHead`].
pub const NOISE_MAX_K: usize = 15;
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
    /// Online learning rate (`0.0..=0.10`, default `0.005`) for [`NoiseBlockHead`].
    pub noise_lr: f32,
    /// Node-perturbation scale $\sigma$ (`1e-5..=0.25`, default `0.01`).
    pub noise_sigma: f32,
    /// Draft refinement passes (`1..=4`, default `2`): pass `r >= 2` conditions each offset on the
    /// previous pass's earlier drafts.
    pub denoise_steps: usize,
    /// Shortlist tokens the head scores (`64..=1024`, default `256`).
    pub shortlist: usize,
    /// Hadamard probe directions per update (`4..=32`, a power of two, default `8`).
    pub probes: usize,
    /// Context tokens the head reads (`1..=8`, default `4`).
    pub window: usize,
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
            denoise_steps: 2,
            shortlist: 256,
            probes: 8,
            window: 4,
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
        self.denoise_steps = self.denoise_steps.clamp(1, 4);
        self.shortlist = self.shortlist.clamp(64, 1024);
        self.probes = self.probes.clamp(4, 32).next_power_of_two();
        self.window = self.window.clamp(1, 8);
        self
    }
}

/// [`DraftKnobs`] shared by the scheduler, [`KTuner`] and [`OnlineNoiseDrafter`], so live edits
/// take effect on the very next decode step.
#[derive(Debug, Default)]
pub struct LiveDraftKnobs(Mutex<DraftKnobs>);

impl LiveDraftKnobs {
    pub fn new(knobs: DraftKnobs) -> Self {
        Self(Mutex::new(knobs.sanitized()))
    }

    pub fn snapshot(&self) -> DraftKnobs {
        *self.0.lock()
    }

    pub fn store(&self, knobs: DraftKnobs) {
        *self.0.lock() = knobs.sanitized();
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

    /// The mean sketch of `toks` (tokens the target cannot embed count as zeros).
    fn mean(&mut self, toks: &[u32]) -> Vec<f32> {
        let mut m = vec![0.0f32; SKETCH_DIM];
        for &t in toks {
            if let Some(s) = self.sketch(t) {
                for (a, b) in m.iter_mut().zip(s) {
                    *a += b;
                }
            }
        }
        let inv = 1.0 / toks.len().max(1) as f32;
        m.iter_mut().for_each(|v| *v *= inv);
        m
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

    /// ANVIL II steps on the outer product `g ⊗ x` (row `i` scaled by `g[i]`, `x.len()` wide).
    fn step_outer(&mut self, g: &[f32], x: &[f32], lr: f32) {
        for (i, &gi) in g.iter().enumerate() {
            if gi.abs() < 1e-7 {
                continue;
            }
            let base = i * x.len();
            for (k, &xk) in x.iter().enumerate() {
                self.step(base + k, gi * xk, lr);
            }
        }
    }
}

/// Online multi-offset block drafter head over token sketches `x`: context `c = rmsnorm(W_c x)`
/// and, per lookahead offset `j`, shortlist logits `H_j c_j`. Refinement passes condition
/// `c_j = rmsnorm(c + U m_j)` on the mean sketch `m_j` of the drafts before `j` (`c_j = c` on the
/// first pass).
pub struct NoiseBlockHead {
    vocab: usize,
    window: usize,
    cap: usize,
    /// `[D, window * SKETCH_DIM]`.
    w_c: Param,
    /// `[D, SKETCH_DIM]`.
    u: Param,
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
        // U starts as the identity on the sketch's first D coordinates: refinement passes see the
        // earlier drafts before U has learned anything.
        let mut u = vec![0.0f32; D * SKETCH_DIM];
        for i in 0..D {
            u[i * SKETCH_DIM + i] = 1.0;
        }
        Self {
            vocab,
            window,
            cap,
            w_c: Param::new(w_c),
            u: Param::new(u),
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

    /// `c_j = rmsnorm(c + U m)`.
    fn refine(&self, c: &[f32], m: &[f32]) -> Vec<f32> {
        let mut cj: Vec<f32> = c
            .iter()
            .zip(self.u.w.chunks_exact(SKETCH_DIM))
            .map(|(&ci, r)| ci + dot(r, m))
            .collect();
        rms_norm(&mut cj);
        cj
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

    /// `H_j ξ_p` for every probe direction, `[probes, active slots]`.
    fn probe_logits(&self, j: usize, dirs: &[Vec<f32>]) -> Vec<f32> {
        let n = self.ids.len();
        dirs.iter()
            .flat_map(|xi| {
                self.heads[j]
                    .w
                    .chunks_exact(D)
                    .take(n)
                    .map(move |r| dot(r, xi))
            })
            .collect()
    }

    /// Trains offset `j` at context `c` toward the target distribution `y` (`(slot, prob)`): the
    /// head takes the exact local delta rule `ΔH_j ∝ (p - y) ⊗ c`. Returns the node-perturbation
    /// (forward-only) credit of `c`, `Σ_p (L(c + σξ_p) - L(c - σξ_p)) / (2σP) · ξ_p` with the
    /// losses through `H_j` (`hx`: [`Self::probe_logits`]).
    #[allow(clippy::too_many_arguments)]
    fn train_offset(
        &mut self,
        j: usize,
        c: &[f32],
        y: &[(usize, f32)],
        dirs: &[Vec<f32>],
        hx: &[f32],
        sigma: f32,
        lr: f32,
    ) -> Vec<f32> {
        let n = self.ids.len();
        let mut z = Vec::with_capacity(n);
        self.logits(j, c, &mut z);
        let mut g = vec![0.0f32; D];
        let mut zs = vec![0.0f32; n];
        let scale = 1.0 / (2.0 * sigma * dirs.len() as f32);
        for (xi, h) in dirs.iter().zip(hx.chunks_exact(n)) {
            for ((o, &zv), &hv) in zs.iter_mut().zip(&z).zip(h) {
                *o = zv + sigma * hv;
            }
            let lp = cross_entropy(&zs, y);
            for ((o, &zv), &hv) in zs.iter_mut().zip(&z).zip(h) {
                *o = zv - sigma * hv;
            }
            let w = (lp - cross_entropy(&zs, y)) * scale;
            for (gd, &x) in g.iter_mut().zip(xi) {
                *gd += w * x;
            }
        }
        softmax(&mut z);
        for &(s, ys) in y {
            z[s] -= ys;
        }
        let h = &mut self.heads[j];
        for (s, &err) in z.iter().enumerate() {
            if err.abs() < 1e-4 {
                continue;
            }
            for (d, &cd) in c.iter().enumerate() {
                let i = s * D + d;
                h.step(i, err * cd + 1e-4 * h.w[i], lr * HEAD_LR);
            }
        }
        g
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

/// Index of the first largest element (0 when empty).
fn argmax(z: &[f32]) -> usize {
    z.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
        .0
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

/// `-Σ y log softmax(z)` over `(index, prob)` targets `y`.
fn cross_entropy(z: &[f32], y: &[(usize, f32)]) -> f32 {
    let max = z.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let lse = max + z.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
    y.iter().map(|&(s, p)| p * (lse - z[s])).sum()
}

fn rms_norm(x: &mut [f32]) {
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (x.len().max(1) as f32);
    let inv = 1.0 / (mean_sq + 1e-5).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Step multiplier of `W_c` and `U` (node-perturbation credit) over `noise_lr`.
const HIDDEN_LR: f32 = 0.1;

/// The head and the sketches it reads, shared by every sequence.
struct Inner {
    head: NoiseBlockHead,
    sk: Sketcher,
}

impl Inner {
    /// Teacher-forced update on committed `tokens`: every (context end `p`, offset `j`) pair whose
    /// target `tokens[p + j]` sits at index `from..` trains once, offset `j`'s refined context
    /// reading the committed tokens before it. A target the verifier scored (`soft`: by token
    /// index) trains toward that distribution over the shortlist, else toward the token.
    fn learn(
        &mut self,
        tokens: &[u32],
        from: usize,
        knobs: &DraftKnobs,
        soft: &[(usize, Vec<(u32, f32)>)],
    ) {
        let Self { head, sk } = self;
        let to = tokens.len();
        let lr = knobs.noise_lr;
        if from >= to || lr <= 0.0 {
            return;
        }
        let max_k = knobs.max_k.clamp(1, NOISE_MAX_K);
        let refine = knobs.denoise_steps >= 2;
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
        let dirs = hadamard_perturbations(D, knobs.probes.min(D), splitmix64(head.updates));
        // ponytail: `H_j ξ_p` is taken once per update, so later examples of a multi-token update
        // see probe losses through slightly stale heads; recompute per example if that matters.
        let mut hx: Vec<Option<Vec<f32>>> = vec![None; max_k];
        for p in (from + 1).saturating_sub(max_k).max(1)..to {
            let x = sk.context(&tokens[..p], head.window);
            let c = head.context(&x);
            let mut g_c = vec![0.0f32; D];
            for j in from.saturating_sub(p)..max_k.min(to - p) {
                let slot = slots[p + j - from];
                // A later token of this batch may have taken the slot.
                if head.ids[slot] != tokens[p + j] {
                    continue;
                }
                let m = (refine && j > 0).then(|| sk.mean(&tokens[p..p + j]));
                let cj = m.as_ref().map_or_else(|| c.clone(), |m| head.refine(&c, m));
                let hxj = hx[j].get_or_insert_with(|| head.probe_logits(j, &dirs));
                let mut y: Vec<(usize, f32)> = soft
                    .iter()
                    .find(|e| e.0 == p + j)
                    .map(|e| {
                        let y = e.1.iter().filter_map(|&(t, q)| Some((*head.slot_of.get(&t)?, q)));
                        y.collect()
                    })
                    .unwrap_or_default();
                let total: f32 = y.iter().map(|e| e.1).sum();
                if total > 0.0 {
                    y.iter_mut().for_each(|e| e.1 /= total);
                } else {
                    y = vec![(slot, 1.0)];
                }
                let g = head.train_offset(j, &cj, &y, &dirs, hxj, knobs.noise_sigma, lr);
                // Identity-path credit: `c_j`'s credit is `c`'s, and `U`'s through `m_j`.
                if let Some(m) = &m {
                    head.u.step_outer(&g, m, lr * HIDDEN_LR);
                }
                for (a, b) in g_c.iter_mut().zip(&g) {
                    *a += b;
                }
            }
            head.w_c.step_outer(&g_c, &x, lr * HIDDEN_LR);
        }
        head.updates += 1;
    }

    /// Shortlist logits (over `head.ids`) for offsets `0..k` after `ctx`, from the last of `steps`
    /// passes: pass `r >= 2` refines offset `j >= 1` on pass `r - 1`'s drafts before `j`
    /// (`fixed[i]` where given, else that pass's argmax).
    fn draft(&mut self, ctx: &[u32], k: usize, steps: usize, fixed: &[u32]) -> Vec<Vec<f32>> {
        let Self { head, sk } = self;
        let x = sk.context(ctx, head.window);
        let c = head.context(&x);
        let mut rows: Vec<Vec<f32>> = (0..k)
            .map(|j| {
                let mut z = Vec::new();
                head.logits(j, &c, &mut z);
                z
            })
            .collect();
        if head.ids.is_empty() {
            return rows;
        }
        for _ in 1..steps {
            let prev: Vec<u32> = rows
                .iter()
                .enumerate()
                .map(|(j, z)| fixed.get(j).copied().unwrap_or(head.ids[argmax(z)]))
                .collect();
            for j in 1..k {
                let cj = head.refine(&c, &sk.mean(&prev[..j]));
                head.logits(j, &cj, &mut rows[j]);
            }
        }
        rows
    }
}

// ---- OnlineNoiseDrafter -----------------------------------------------------------------------

/// Per-sequence state for [`OnlineNoiseDrafter`].
#[derive(Clone)]
struct OnlineNoiseState {
    /// Committed tokens already ingested into the suffix lattices and the head.
    seen: usize,
    /// A fork ingests no further than this on its next propose (its extra tokens are unverified).
    cap: Option<usize>,
    /// The verifier's distributions over upcoming tokens, by token index ([`Drafter::observe`]).
    soft: Vec<(usize, Vec<(u32, f32)>)>,
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
        let k = knobs.snapshot();
        Arc::new(Self {
            vocab: vocab.max(1),
            embeds,
            knobs,
            lattice: RwLock::new(SuffixLattice::new()),
            inner: Mutex::new(Inner {
                head: NoiseBlockHead::new(vocab, k.window, k.shortlist),
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

    /// The shared head, rebuilt when the shortlist or window knob changed.
    fn inner(&self, knobs: &DraftKnobs) -> parking_lot::MutexGuard<'_, Inner> {
        let mut inner = self.inner.lock();
        let h = &inner.head;
        if h.window != knobs.window || h.cap != knobs.shortlist.min(h.vocab) {
            inner.head = NoiseBlockHead::new(self.vocab, knobs.window, knobs.shortlist);
        }
        inner
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
            cap: None,
            soft: Vec::new(),
            local: SuffixLattice::new(),
        })
    }

    fn overlaps(&self) -> bool {
        true
    }

    fn fork_state(&self, st: &DraftState) -> Option<DraftState> {
        let s = st.downcast_ref::<OnlineNoiseState>()?;
        Some(Box::new(OnlineNoiseState {
            cap: Some(s.seen),
            ..s.clone()
        }))
    }

    /// Keeps each verified row as a soft target: the softmax of its logits over the shortlist and
    /// the row's top 8.
    fn observe(&self, st: &mut DraftState, accepted: &[u32], logits: &[f32], width: usize) {
        let Some(s) = st.downcast_mut::<OnlineNoiseState>() else {
            return;
        };
        let mode = self.knobs.snapshot().mode;
        if width == 0 || !self.embeds || matches!(mode, DraftMode::Suffix | DraftMode::Off) {
            return;
        }
        let ids = self.inner.lock().head.ids.clone();
        for (j, row) in logits.chunks_exact(width).take(accepted.len()).enumerate() {
            let mut cand: Vec<u32> = ids.iter().copied().filter(|&t| (t as usize) < width).collect();
            for t in top_n(row, 8) {
                if !cand.contains(&t) {
                    cand.push(t);
                }
            }
            let mut p: Vec<f32> = cand.iter().map(|&t| row[t as usize]).collect();
            softmax(&mut p);
            s.soft.push((s.seen + j, cand.into_iter().zip(p).collect()));
        }
    }

    fn propose(
        &self,
        st: &mut DraftState,
        tokens: &[u32],
        _hidden: &[f32],
        k: usize,
        pick: &mut dyn FnMut(DraftLogits<'_>) -> u32,
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
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Ingest newly committed tokens into the local and shared suffix lattices; the head
        // learns from them even on `k == 0` steps.
        let upto = s.cap.take().map_or(tokens.len(), |c| c.min(tokens.len()));
        if upto > s.seen {
            let slice = &tokens[s.seen.saturating_sub(4)..upto];
            s.local.observe_tokens(slice);
            self.lattice.write().observe_tokens(slice);
            if neural {
                let from = s.seen.max(upto.saturating_sub(LEARN_SPAN));
                self.inner(&knobs)
                    .learn(&tokens[..upto], from, &knobs, &s.soft);
            }
            s.seen = upto;
            s.soft.retain(|e| e.0 >= upto);
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

        // 3. The head's shortlist logits per offset, refined on the suffix drafts where they exist.
        let (ids, rows, head_updates) = if neural {
            let mut inner = self.inner(&knobs);
            let rows = inner.draft(tokens, k, knobs.denoise_steps, suf_slice);
            (inner.head.ids.clone(), rows, inner.head.updates)
        } else {
            (Vec::new(), Vec::new(), 0)
        };

        // 4. Each offset's candidates: the shortlist, with the suffix match's token boosted.
        let (mut cand, mut cand_logits) = (Vec::new(), Vec::new());
        let mut drafts = Vec::with_capacity(k);
        for offset in 0..k {
            cand.clear();
            cand_logits.clear();
            let mut top_prob = 0.0f32;
            if let Some(z) = rows.get(offset).filter(|z| !z.is_empty()) {
                let mut p = z.clone();
                softmax(&mut p);
                top_prob = p.iter().fold(0.0f32, |a, &b| a.max(b));
                cand.extend_from_slice(&ids);
                cand_logits.extend_from_slice(z);
            }
            let has_suffix = suf_slice.get(offset).map(|&t| {
                match cand.iter().position(|&c| c == t) {
                    Some(i) => cand_logits[i] = cand_logits[i].max(0.0) + 14.0,
                    None => {
                        cand.push(t);
                        cand_logits.push(14.0);
                    }
                }
            });
            if knobs.mode == DraftMode::Suffix && has_suffix.is_none() {
                break;
            }
            if knobs.mode == DraftMode::Auto
                && has_suffix.is_none()
                && !(head_updates >= 4 && top_prob >= knobs.target_acceptance)
            {
                break;
            }
            if cand.is_empty() {
                break;
            }
            drafts.push(pick(DraftLogits::Sparse {
                ids: &cand,
                logits: &cand_logits,
            }));
        }

        Ok(drafts)
    }
}

/// Ids of the `n` largest finite entries of `row`, largest first.
fn top_n(row: &[f32], n: usize) -> Vec<u32> {
    let mut best: Vec<(f32, u32)> = Vec::with_capacity(n + 1);
    for (i, &v) in row.iter().enumerate() {
        if v.is_nan() || (best.len() == n && v <= best[n - 1].0) {
            continue;
        }
        let at = best.partition_point(|e| e.0 >= v);
        best.insert(at, (v, i as u32));
        best.truncate(n);
    }
    best.into_iter().map(|e| e.1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::sched::Rng;

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

    /// A 16-state Markov chain (token `100 + s`) with two branch gadgets where the marginal
    /// two-step mode disagrees with the mode after the likelier first step. Trained forward-only
    /// for 2000 committed steps, the first draft is the chain's most likely next token, and
    /// refinement (`denoise_steps = 2`) gets more two-token drafts accepted than one pass.
    #[test]
    fn learns_markov_chain_forward_only() {
        let mut next: Vec<Vec<(u32, f32)>> = (0..16).map(|s| vec![((s + 1) % 16, 1.0)]).collect();
        for base in [0, 8] {
            next[base] = vec![(base as u32 + 1, 0.55), (base as u32 + 2, 0.45)];
            next[base + 1] = vec![(base as u32 + 3, 0.5), (base as u32 + 4, 0.5)];
            next[base + 2] = vec![(base as u32 + 5, 1.0)];
            for s in base + 3..base + 6 {
                next[s] = vec![(base as u32 + 6, 1.0)];
            }
        }
        let prob = |a: u32, b: u32| -> f32 {
            next.get(a.wrapping_sub(100) as usize)
                .and_then(|n| n.iter().find(|e| e.0 + 100 == b))
                .map_or(0.0, |e| e.1)
        };
        let walk = |seed: u64, n: usize| -> Vec<u32> {
            let (mut rng, mut s, mut out) = (Rng::new(seed), 0u32, Vec::with_capacity(n));
            for _ in 0..n {
                out.push(100 + s);
                let (u, mut acc) = (rng.uniform() as f32, 0.0);
                let succ = &next[s as usize];
                s = succ[succ.len() - 1].0;
                for &(t, p) in succ {
                    acc += p;
                    if u <= acc {
                        s = t;
                        break;
                    }
                }
            }
            out
        };
        let (train, test) = (walk(1, 2200), walk(2, 400));
        let run = |steps: usize| -> (f32, f32) {
            let knobs = DraftKnobs {
                max_k: 2,
                denoise_steps: steps,
                ..DraftKnobs::default()
            }
            .sanitized();
            let mut inner = Inner {
                head: NoiseBlockHead::new(256, knobs.window, knobs.shortlist),
                sk: Sketcher::new(table(64), 64),
            };
            let mut i = 1;
            while inner.head.updates < 2000 {
                i += 1;
                inner.learn(&train[..i], i - 1, &knobs, &[]);
            }
            let (mut top1, mut two) = (0.0, 0.0);
            let n = test.len() - 8;
            for p in 8..test.len() {
                let rows = inner.draft(&test[..p], 2, steps, &[]);
                let d: Vec<u32> = rows.iter().map(|z| inner.head.ids[argmax(z)]).collect();
                let last = test[p - 1];
                let best = next[(last - 100) as usize]
                    .iter()
                    .fold(0.0f32, |a, e| a.max(e.1));
                top1 += (prob(last, d[0]) == best) as u32 as f32;
                two += prob(last, d[0]) * prob(d[0], d[1]);
            }
            (top1 / n as f32, two / n as f32)
        };
        let (top1_one, two_one) = run(1);
        let (top1_two, two_two) = run(2);
        assert!(
            top1_one >= 0.9 && top1_two >= 0.9,
            "offset-1 top-1: {top1_one} (1 step), {top1_two} (2 steps)"
        );
        assert!(
            two_two > two_one,
            "two-token acceptance: {two_two} (2 steps) vs {two_one} (1 step)"
        );
    }
}
