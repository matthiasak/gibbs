//! Re-encode policy (sprint-2 S3.5): which format each streamed 2-D weight matrix (Linear weights,
//! LM head) is stored in on a GPU load and how a re-encode is gated per tensor. The whole-model KL
//! gate lives in [`crate::model::weight_format`], the fallback chain in [`crate::arch::load::gated`].
//!
//! Rules, per projection, from its source format `src`:
//! * `MaxSpeed`: Q4_K when `src` is above 4.5 bpw or has no GPU codec, else native.
//! * `LowBitNative`: float sources become Q8_0; a codec-less source becomes the smallest of
//!   Q4_K / Q5_K / Q6_K / Q8_0 whose bpw is at least the source's; else native.
//! * `BitExact`: native when a codec exists; a codec-less source follows the `LowBitNative` rule.
//!
//! "Native" carries the checkpoint's own blocks (GGUF block formats and the repacked source encodings
//! of [`crate::model::repack`]) byte for byte; float sources are re-stored at their own dtype, which
//! is exact. Where blocks cannot be carried (mixed formats inside one fused projection, reordered
//! rows) the f32 values are encoded in the source format when an encoder exists (exact for Q4_0 and
//! Q8_0, one more quantization step for the K-quants, recorded as `lossy`).

use crate::graph::codec::{codec, CODEC_VERSION};
use crate::hardware::QuantFormat as Q;
use crate::model::layers::{dequantize_gguf_blocks, encode_rows, gguf_block_codec};
use crate::model::loaders::gguf::GgufData;
use crate::model::weight_format::ReencodePolicy;
use crate::model::VirtualTensor;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
/// Re-encode formats from smallest to largest.
pub const LADDER: [Q; 4] = [Q::GGUF_Q4_K_M, Q::GGUF_Q5_K_M, Q::GGUF_Q6_K, Q::GGUF_Q8_0];
/// Per-tensor gate: relative RMSE of the re-encoded weights against the source values.
pub const TENSOR_RMSE_LIMIT: f64 = 0.08;
/// Values per row run a native re-encode trial decodes and encodes at a time (8 MB of f32).
const TRIAL_VALUES: usize = 1 << 21;

/// Per-tensor relative RMSE limit for `p`.
pub fn rmse_limit(p: ReencodePolicy) -> f64 {
    match p {
        ReencodePolicy::MinSize => 0.20,
        _ => TENSOR_RMSE_LIMIT,
    }
}

pub fn is_float(f: Q) -> bool {
    matches!(f, Q::FP32 | Q::FP16 | Q::BF16)
}

/// Whether Metal reads `f` directly (float kernels, a registered codec, or a source encoding that
/// [`crate::model::repack`] turns into one).
pub fn has_gpu_codec(f: Q) -> bool {
    is_float(f)
        || codec(f).is_some()
        || matches!(
            f,
            Q::FP8_E4M3 | Q::FP8_E5M2 | Q::BNB_NF4 | Q::BNB_FP4 | Q::GPTQ_W4A16 | Q::AWQ_W4A16
        )
}

/// Formats [`encode_rows`] can produce.
pub fn encodable(f: Q) -> bool {
    is_float(f)
        || matches!(
            f,
            Q::GGUF_Q3_K | Q::GGUF_Q4_0 | Q::GGUF_Q8_0 | Q::GGUF_NVFP4
        )
        || LADDER[..3].contains(&f)
}

fn block_elems(f: Q) -> usize {
    gguf_block_codec(f).map_or(1, |c| c.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Native,
    Encode(Q),
}

/// The smallest ladder format whose bpw is at least `bpw`.
pub fn smallest_ladder_ge(bpw: f32) -> Q {
    LADDER
        .into_iter()
        .find(|f| f.bits_per_weight() >= bpw)
        .unwrap_or(Q::GGUF_Q8_0)
}

/// The policy's decision for a projection stored in `src`; `MaxSpeed` reads this host's measured
/// codec rates (falling back to the static rule where none are measured yet).
pub fn rule(policy: ReencodePolicy, src: Q) -> Target {
    rule_with(policy, src, &crate::model::variant::host_gbps)
}

/// A candidate must need under this fraction of the incumbent's time per weight to replace it:
/// run-to-run noise in the measured rates must not flip a tensor's format (and the re-encode cache).
const TIME_MARGIN: f64 = 0.9;

/// [`rule`] with explicit rates (`GB/s` of a format's GEMV; `None` = not measured).
///
/// `MaxSpeed` picks the target minimizing `bpw / GB/s` (time to stream one weight) among the
/// source format (when a GPU codec reads it), the ladder and Q4_0, restricted to formats the
/// encoder can produce. A source above 4.5 bpw never drops below 4 bpw; a source at or below
/// 4.5 bpw only moves to equal-or-higher precision, so a lossy re-encode never loses bits.
pub fn rule_with(policy: ReencodePolicy, src: Q, rate: &dyn Fn(Q) -> Option<f64>) -> Target {
    let base = static_rule(policy, src);
    if policy != ReencodePolicy::MaxSpeed {
        return base;
    }
    let bpw = src.bits_per_weight();
    let floor = if bpw > 4.5 { 4.0 } else { bpw - 0.01 };
    let time = |f: Q| {
        rate(f)
            .map(|g| f.bits_per_weight() as f64 / g)
            .filter(|t| t.is_finite() && *t > 0.0)
    };
    let incumbent = match base {
        Target::Native => src,
        Target::Encode(f) => f,
    };
    let (mut best, mut best_t) = (incumbent, time(incumbent));
    let readable = has_gpu_codec(src);
    for f in std::iter::once(src).chain(LADDER).chain([Q::GGUF_Q4_0]) {
        let allowed = (f == src && readable) || encodable(f);
        if f == incumbent || !allowed || f.bits_per_weight() < floor {
            continue;
        }
        if let Some(t) = time(f) {
            if best_t.is_none_or(|bt| t < bt * TIME_MARGIN) {
                (best, best_t) = (f, Some(t));
            }
        }
    }
    if best == src && readable {
        Target::Native
    } else {
        Target::Encode(best)
    }
}

/// The rate-independent rule (what runs before the codec benchmark has measured this host).
fn static_rule(policy: ReencodePolicy, src: Q) -> Target {
    let codec_less = !has_gpu_codec(src);
    match policy {
        ReencodePolicy::MaxSpeed if src.bits_per_weight() > 4.5 || codec_less => {
            Target::Encode(Q::GGUF_Q4_K_M)
        }
        ReencodePolicy::MinSize
            if src.bits_per_weight() > Q::GGUF_Q3_K.bits_per_weight() || codec_less =>
        {
            Target::Encode(Q::GGUF_Q3_K)
        }
        ReencodePolicy::LowBitNative if is_float(src) => Target::Encode(Q::GGUF_Q8_0),
        ReencodePolicy::LowBitNative | ReencodePolicy::BitExact if codec_less => {
            Target::Encode(smallest_ladder_ge(src.bits_per_weight()))
        }
        _ => Target::Native,
    }
}

/// The next ladder format above `f`.
pub fn bump(f: Q) -> Option<Q> {
    LADDER
        .into_iter()
        .find(|g| g.bits_per_weight() > f.bits_per_weight())
}

/// `f`, or the next larger format whose blocks tile `cols` (a float dtype when none does).
pub fn fit(mut f: Q, cols: usize, float: Q) -> Q {
    // GGML's convention for rows that hold no whole 256-element block: 4-bit K -> block-32 Q4_0.
    if matches!(f, Q::GGUF_Q3_K | Q::GGUF_Q4_K_M) && cols % 256 != 0 && cols % 32 == 0 {
        return Q::GGUF_Q4_0;
    }
    while cols % block_elems(f) != 0 {
        f = bump(f).unwrap_or(float);
    }
    f
}

/// The format a checkpoint tensor is stored in.
pub fn declared_format(t: &VirtualTensor) -> Q {
    if gguf_block_codec(t.quant_format).is_some() {
        return t.quant_format;
    }
    match t.dtype.as_str() {
        "F32" | "F64" => Q::FP32,
        "F16" => Q::FP16,
        "BF16" => Q::BF16,
        "F8_E4M3" => Q::FP8_E4M3,
        "F8_E5M2" => Q::FP8_E5M2,
        // Packed integers under a float tag: a quantized checkpoint (EXL3, GLQ, ...) no codec reads.
        // TurboQuant is the codec-less low-bit tag: the ladder lifts it to Q4_K_M.
        "U8" | "I8" | "I16" | "U16" | "I32" | "U32" | "I64" | "U64" if is_float(t.quant_format) => {
            Q::TurboQuant
        }
        _ => t.quant_format,
    }
}

/// Counts finished projections for the load-stage text.
#[derive(Default)]
pub struct ReencodeProgress {
    pub done: AtomicUsize,
    pub total: AtomicUsize,
}

impl ReencodeProgress {
    /// `re-encoding for this GPU (max-speed): 12/197 tensors`
    pub fn text(&self, policy: ReencodePolicy) -> String {
        format!(
            "re-encoding for this GPU ({}): {}/{} tensors",
            policy.name(),
            self.done.load(Ordering::Relaxed),
            self.total.load(Ordering::Relaxed)
        )
    }
}

/// `sqrt(Σ(x-y)² / Σx²)`.
pub fn relative_rmse(x: &[f32], y: &[f32]) -> f64 {
    let mut err = (0.0, 0.0);
    add_sq_err(&mut err, x, y);
    rmse_of(err)
}

/// Adds `(Σ(x-y)², Σx²)` of `x`, `y` to `err`.
fn add_sq_err(err: &mut (f64, f64), x: &[f32], y: &[f32]) {
    for (&a, &b) in x.iter().zip(y) {
        err.0 += (a as f64 - b as f64).powi(2);
        err.1 += (a as f64).powi(2);
    }
}

fn rmse_of((num, den): (f64, f64)) -> f64 {
    if den == 0.0 {
        0.0
    } else {
        (num / den).sqrt()
    }
}

/// Encodes row-major `w` as `f` (a float or an encodable block format whose blocks tile `in_f`),
/// splitting rows over `threads` workers at utility QoS; row chunks concatenate.
pub(crate) fn encode_weight(w: &[f32], in_f: usize, out_f: usize, f: Q, threads: usize) -> Vec<u8> {
    let workers = if w.len() < 1 << 16 {
        1
    } else {
        threads.min(out_f)
    };
    let rows = out_f.div_ceil(workers);
    let raw = |c: &[f32]| -> Vec<u8> {
        encode_rows(c, in_f, f).unwrap_or_else(|e| panic!("cannot encode as {f:?}: {e}"))
    };
    if workers == 1 {
        return raw(w);
    }
    std::thread::scope(|s| {
        let handles: Vec<_> = w
            .chunks(rows * in_f)
            .map(|c| {
                s.spawn(move || {
                    crate::tui::qos::set_qos(crate::tui::qos::QosClass::Utility);
                    raw(c)
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("encoder thread panicked"))
            .collect()
    })
}

// ---- native loaders -------------------------------------------------------------------------

/// What the native loaders did to their tensors since [`reset_native_stats`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChoiceRecord {
    pub policy: ReencodePolicy,
    pub ffn_only: bool,
    pub mean_kl: f64,
    pub top1: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NativeStats {
    /// Projection matrices considered.
    pub tensors: usize,
    /// Matrices re-encoded to another format (all lossy: the source blocks differ from the stored ones).
    pub reencoded: usize,
    pub bytes_before: u64,
    pub bytes_after: u64,
    /// Matrices mapped directly from a persisted `.blocks` cache.
    pub cache_hits: usize,
    /// Whether any entry in the `.index.json` cache mismatched the checkpoint tensor's size or format.
    pub cache_mismatch: bool,
    /// Accepted rung loaded from `<choice_key>.json`, if any.
    pub cached_choice: Option<ChoiceRecord>,
}

static NATIVE_STATS: Mutex<NativeStats> = Mutex::new(NativeStats {
    tensors: 0,
    reencoded: 0,
    bytes_before: 0,
    bytes_after: 0,
    cache_hits: 0,
    cache_mismatch: false,
    cached_choice: None,
});

pub fn reset_native_stats() {
    *NATIVE_STATS.lock() = NativeStats::default();
    *PENDING_REENCODE.lock() = None;
}

pub fn native_stats() -> NativeStats {
    *NATIVE_STATS.lock()
}

pub fn set_cached_choice(c: ChoiceRecord) {
    NATIVE_STATS.lock().cached_choice = Some(c);
}
/// Test hook: pretend the current load re-encoded `n` tensors.
#[cfg(test)]
pub(crate) fn fake_native_reencoded(n: usize) {
    let mut s = NATIVE_STATS.lock();
    (s.tensors, s.reencoded) = (n, n);
}

static NATIVE_APPLIED: Mutex<Option<(ReencodePolicy, u64, Option<Q>)>> = Mutex::new(None);

/// Records the policy the last native load ended up with, the bytes per token it saved and the
/// `--quant` format every matrix was forced into (which bypasses the policy).
pub fn set_native_applied(policy: ReencodePolicy, saved_bytes: u64, force: Option<Q>) {
    *NATIVE_APPLIED.lock() = Some((policy, saved_bytes, force));
}

/// `(policy, bytes saved over the source blocks, forced format)` of the last native load that
/// reached the gate.
pub fn native_applied() -> Option<(ReencodePolicy, u64, Option<Q>)> {
    *NATIVE_APPLIED.lock()
}

/// Functional role of a weight tensor for mixed-precision sensitivity gating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    Router,
    LmHead,
    BoundaryAttn,
    AttnOrSsm,
    FfnDown,
    FfnGateUp,
    Other,
}

pub(crate) fn parse_layer_idx(name: &str) -> Option<usize> {
    let mut it = name.split('.');
    while let Some(part) = it.next() {
        if matches!(part, "layers" | "blk" | "h") {
            if let Some(idx) = it.next().and_then(|s| s.parse::<usize>().ok()) {
                return Some(idx);
            }
        }
    }
    None
}

/// Classifies `name` into its [`TensorRole`]. MoE routers (`*.mlp.gate.weight`, `*router*`,
/// `*shared_expert_gate*`, `*ffn_gate_inp*`) are always [`TensorRole::Router`], never FFN.
pub fn classify_tensor(
    name: &str,
    layer: Option<usize>,
    total_layers: Option<usize>,
) -> TensorRole {
    if name.ends_with(".mlp.gate.weight")
        || name.ends_with(".block_sparse_moe.gate.weight")
        || name.contains("router")
        || name.contains("shared_expert_gate")
        || name.contains("ffn_gate_inp")
    {
        return TensorRole::Router;
    }
    if is_lm_head_name(name) {
        return TensorRole::LmHead;
    }
    // `.fc1.` / `.fc2.`: OPT-style feed-forward layers outside an `mlp` scope.
    let ffn_sub = ["mlp", "ffn", "feed_forward", ".fc1.", ".fc2."]
        .iter()
        .any(|s| name.contains(s));
    if ffn_sub {
        if [
            "down",
            "w2",
            "c_proj",
            "fc2",
            "dense_4h_to_h",
            "output_linear",
        ]
        .iter()
        .any(|s| name.contains(s))
        {
            return TensorRole::FfnDown;
        }
        return TensorRole::FfnGateUp;
    }
    if [
        "attn",
        "attention",
        "linear_attn",
        "ssm",
        "mamba",
        "shortconv",
    ]
    .iter()
    .any(|s| name.contains(s))
    {
        let l = layer.or_else(|| parse_layer_idx(name));
        let is_boundary = match (l, total_layers) {
            (Some(0), _) => true,
            (Some(i), Some(n)) if n > 1 && i + 1 == n => true,
            _ => false,
        };
        return if is_boundary {
            TensorRole::BoundaryAttn
        } else {
            TensorRole::AttnOrSsm
        };
    }
    TensorRole::Other
}

/// Whether `name` (an HF-style tensor name) is a feed-forward projection (excludes MoE routers).
pub(crate) fn is_ffn_name(name: &str) -> bool {
    matches!(
        classify_tensor(name, None, None),
        TensorRole::FfnDown | TensorRole::FfnGateUp
    )
}

/// Whether `name` is the LM head (or a tied embedding's projection copy).
pub(crate) fn is_lm_head_name(name: &str) -> bool {
    name.ends_with(crate::arch::build::LINEAR_COPY)
        || name == "output.weight"
        || ["lm_head", "embed_out", "output_projection"]
            .iter()
            .any(|s| name.contains(s))
}

/// Gather tables (token / per-layer embeddings, which a tied head shares) keep their source blocks:
/// a gathered row's precision is not a streaming cost.
fn is_gather_table(name: &str, float_src: bool) -> bool {
    (!float_src || !name.ends_with(crate::arch::build::LINEAR_COPY))
        && ["embed", "token_embd", "ple"]
            .iter()
            .any(|s| name.contains(s))
}

/// Whether `f` still reads faster than the stored `src` by the rule's margin under `rate`; without
/// measured rates, whether it at least holds fewer bits.
fn beats_source(f: Q, src: Q, rate: &dyn Fn(Q) -> Option<f64>) -> bool {
    let time = |q: Q| rate(q).map(|g| q.bits_per_weight() as f64 / g);
    match (time(f), time(src)) {
        (Some(a), Some(b)) => a < b * TIME_MARGIN,
        _ => f64::from(f.bits_per_weight()) < f64::from(src.bits_per_weight()) * TIME_MARGIN,
    }
}

/// A native loader's `rows x cols` projection `name`, stored as `src` blocks in `raw`, under the
/// effective policy `cfg`: `Some((format, blocks))` when the policy re-stores it in a faster format
/// (every such re-encode is lossy and counted in [`native_stats`]), `None` to keep the stored blocks.
pub fn reencode_native(
    cfg: &crate::model::weight_format::ReencodeConfig,
    name: &str,
    src: Q,
    rows: usize,
    cols: usize,
    raw: &[u8],
) -> Option<(Q, Vec<u8>)> {
    let f = target(cfg, name, src, cols, false)?;
    // The source's own decoder, so the values are exactly what the GPU would have read.
    encode_blocks(
        cfg,
        name,
        src,
        f,
        rows,
        cols,
        raw,
        gguf_block_codec(src)?,
    )
}

/// [`reencode_native`] for a projection stored as `src` (any format, floats included) whose
/// values `data` yields on demand (only when the policy re-encodes it).
pub fn reencode_values(
    cfg: &crate::model::weight_format::ReencodeConfig,
    name: &str,
    src: Q,
    rows: usize,
    cols: usize,
    data: impl FnOnce() -> Option<Vec<f32>>,
) -> Option<(Q, Vec<u8>)> {
    let f = target(cfg, name, src, cols, true)?;
    let raw: Vec<u8> = data()?.iter().flat_map(|x| x.to_le_bytes()).collect();
    fn f32_block(b: &[u8], out: &mut [f32]) {
        for (o, c) in out.iter_mut().zip(b.chunks_exact(4)) {
            *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
    }
    encode_blocks(
        cfg,
        name,
        src,
        f,
        rows,
        cols,
        &raw,
        (32, 128, f32_block),
    )
}

/// Pure format selection for `(policy, name, src, cols)` shared by [`target`] and `variant::WeightHeader::after`.
pub fn target_for_policy(
    policy: ReencodePolicy,
    name: &str,
    src: Q,
    cols: usize,
    treat_as_lm_head: bool,
    ffn_override: Option<bool>,
    fallback_float: Q,
) -> Option<Q> {
    let role = classify_tensor(name, None, None);
    if role == TensorRole::Router {
        if matches!(policy, ReencodePolicy::MaxSpeed | ReencodePolicy::MinSize) {
            if has_gpu_codec(src) {
                return None;
            }
            let f = fit(Q::GGUF_Q8_0, cols, fallback_float);
            return (f != src && encodable(f)).then_some(f);
        }
        return None;
    }
    let ffn = ffn_override.unwrap_or_else(|| is_ffn_name(name));
    if policy == ReencodePolicy::MinSize {
        let rule_f = if treat_as_lm_head || role == TensorRole::LmHead {
            Q::GGUF_Q6_K
        } else if ffn {
            if cols % 256 == 0 {
                Q::GGUF_Q3_K
            } else {
                Q::GGUF_Q4_0
            }
        } else if cols % 256 == 0 {
            Q::GGUF_Q4_K_M
        } else {
            Q::GGUF_Q4_0
        };
        if has_gpu_codec(src) && src.bits_per_weight() <= rule_f.bits_per_weight() {
            return None;
        }
        let f = fit(rule_f, cols, fallback_float);
        return (f != src
            && encodable(f)
            && (!has_gpu_codec(src) || f.bits_per_weight() < src.bits_per_weight()))
        .then_some(f);
    }
    let Target::Encode(f) = rule(policy, src) else {
        return None;
    };
    let f = fit(f, cols, fallback_float);
    (f != src && encodable(f)).then_some(f)
}

/// The format the policy re-stores `name` (stored as `src`, `cols` wide) in, if any.
fn target(
    cfg: &crate::model::weight_format::ReencodeConfig,
    name: &str,
    src: Q,
    cols: usize,
    float_src: bool,
) -> Option<Q> {
    let role = classify_tensor(name, None, None);
    let ffn = role != TensorRole::Router
        && match &cfg.ffn_weights {
            Some(pats) => pats.iter().any(|p| crate::arch::glob(p, name)),
            None => is_ffn_name(name),
        };
    if is_gather_table(name, float_src) || (cfg.ffn_only && !ffn) {
        return None;
    }
    let mut stats = NATIVE_STATS.lock();
    stats.tensors += 1;
    drop(stats);
    target_for_policy(
        cfg.policy,
        name,
        src,
        cols,
        is_lm_head_name(name),
        Some(ffn),
        Q::FP32,
    )
}

/// Re-encodes `raw` (`src`'s `rows x cols` weights as `(elems, block_bytes, decode)` blocks) into
/// `f`, bumped until the per-tensor RMSE gate holds; `None` when that no longer beats `src`.
fn encode_blocks(
    cfg: &crate::model::weight_format::ReencodeConfig,
    name: &str,
    src: Q,
    f: Q,
    rows: usize,
    cols: usize,
    raw: &[u8],
    (elems, block_bytes, decode): (usize, usize, crate::model::layers::GgufBlockFn),
) -> Option<(Q, Vec<u8>)> {
    let row_bytes = cols / elems * block_bytes;
    if cols % elems != 0 || raw.len() != rows * row_bytes {
        return None;
    }
    let threads = std::thread::available_parallelism()
        .map_or(2, |n| n.get())
        .saturating_sub(1)
        .max(1);
    // Runs of rows, not the whole tensor, and one trial's blocks alive at a time: the allocator
    // keeps freed large blocks resident, so whole-tensor trial buffers (decoded source, round trip)
    // would stay in the process's resident memory after the load.
    let run = (TRIAL_VALUES / cols).max(1);
    let mut x = vec![0.0f32; run.min(rows) * cols];
    // `f`'s relative RMSE over the tensor and its blocks (`None`: no block format).
    let mut encode = |f: Q| {
        if is_float(f) {
            return (0.0, None);
        }
        let mut err = (0.0, 0.0);
        let mut blocks = Vec::with_capacity(rows * f.tensor_bytes(cols));
        for part in raw.chunks(run * row_bytes) {
            let x = &mut x[..part.len() / block_bytes * elems];
            for (b, out) in part
                .chunks_exact(block_bytes)
                .zip(x.chunks_exact_mut(elems))
            {
                decode(b, out);
            }
            let w = encode_weight(x, cols, x.len() / cols, f, threads);
            add_sq_err(&mut err, x, &dequantize_gguf_blocks(f, &w).unwrap());
            blocks.extend_from_slice(&w);
        }
        (rmse_of(err), Some(blocks))
    };
    let (mut rmse, mut blocks) = encode(f);
    let mut f = f;
    let policy = cfg.policy;
    let role = classify_tensor(name, None, cfg.total_layers);
    let limit = match (policy, role) {
        (ReencodePolicy::MaxSpeed, TensorRole::LmHead) => rmse_limit(policy).min(0.05),
        (ReencodePolicy::MaxSpeed, TensorRole::FfnDown | TensorRole::BoundaryAttn) => {
            rmse_limit(policy).min(0.06)
        }
        _ => rmse_limit(policy),
    };
    let limit = match &cfg.sensitive_weights {
        Some(pats) if pats.iter().any(|p| crate::arch::glob(p, name)) => limit.min(0.05),
        _ => limit,
    };
    while rmse > limit {
        let Some(next) = bump(f).map(|n| fit(n, cols, Q::FP32)) else {
            break;
        };
        // The rejected trial's blocks go before the next trial allocates its own.
        drop(blocks.take());
        f = next;
        (rmse, blocks) = encode(next);
    }
    // The RMSE bump may have walked past the source's own cost (Q4_K -> Q4_0 failing the gate lands on
    // Q5_K, bigger and slower than what was stored): then the stored blocks stay.
    let keep = if policy == ReencodePolicy::MinSize {
        !has_gpu_codec(src) || f.bits_per_weight() < src.bits_per_weight()
    } else {
        beats_source(f, src, &crate::model::variant::host_gbps)
    };
    if !keep {
        return None;
    }
    // A float target stores no blocks.
    let blocks = blocks?;
    let mut stats = NATIVE_STATS.lock();
    stats.reencoded += 1;
    (stats.bytes_before, stats.bytes_after) = (
        stats.bytes_before + (rows * src.tensor_bytes(cols)) as u64,
        stats.bytes_after + blocks.len() as u64,
    );
    tracing::debug!("reencode {name} {rows}x{cols}: {src:?} -> {f:?} (rmse {rmse:.4})");
    Some((f, blocks))
}

// ---- persisted block cache ------------------------------------------------------------------

const TENSOR_ALIGN: usize = 64;
const PAGE_ALIGN: usize = 16384;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedTensor {
    pub format: Q,
    pub offset: usize,
    pub bytes: usize,
    pub rows: usize,
    pub cols: usize,
}

pub struct ReencodeCache {
    pub key: String,
    pub map: Arc<GgufData>,
    pub index: BTreeMap<String, CachedTensor>,
}

#[derive(Clone)]
pub struct PendingTensor {
    pub format: Q,
    pub map: Arc<GgufData>,
    pub off: usize,
    pub bytes: usize,
    pub rows: usize,
    pub cols: usize,
}

pub type PendingReencode = (String, BTreeMap<String, PendingTensor>);

static PENDING_REENCODE: Mutex<Option<PendingReencode>> = Mutex::new(None);

static RETAINED_PENDING: Mutex<Option<(ReencodePolicy, PendingReencode)>> = Mutex::new(None);

/// Retains encoded tensors from a rejected KL-gate rung so subsequent rungs (`ffn_only`) reuse them.
pub fn retain_pending_reencode(policy: ReencodePolicy, pending: Option<PendingReencode>) {
    let Some((key, map)) = pending else {
        return;
    };
    let mut guard = RETAINED_PENDING.lock();
    match guard.as_mut() {
        Some((p, (k, existing))) if *p == policy && *k == key => {
            for (name, pt) in map {
                existing.entry(name).or_insert(pt);
            }
        }
        _ => *guard = Some((policy, (key, map))),
    }
}

pub fn clear_retained_pending() {
    *RETAINED_PENDING.lock() = None;
}

/// Reuses a tensor already encoded on an earlier rung of [`crate::arch::load::gated`] when the
/// current rung's `cfg` selects the same policy for `name`.
pub fn reuse_retained_pending(
    key: &str,
    cfg: &crate::model::weight_format::ReencodeConfig,
    name: &str,
    src: Q,
    rows: usize,
    cols: usize,
) -> Option<PendingTensor> {
    let pt = {
        let guard = RETAINED_PENDING.lock();
        let (policy, (rkey, map)) = guard.as_ref()?;
        if *policy != cfg.policy || rkey != key {
            return None;
        }
        let pt = map.get(name)?.clone();
        if pt.rows != rows || pt.cols != cols {
            return None;
        }
        pt
    };
    // Check whether the current rung's config (`ffn_only`, etc.) actually wants to re-encode `name`.
    let _ = target(cfg, name, src, cols, is_float(src))?;
    let mut stats = NATIVE_STATS.lock();
    stats.reencoded += 1;
    stats.bytes_before += (rows * src.tensor_bytes(cols)) as u64;
    stats.bytes_after += pt.bytes as u64;
    drop(stats);
    record_pending_tensor(
        key,
        name,
        pt.format,
        pt.map.clone(),
        pt.off,
        pt.bytes,
        rows,
        cols,
    );
    Some(pt)
}
#[cfg(test)]
thread_local! {
    static TEST_CACHE_DIR: std::cell::RefCell<Option<PathBuf>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_test_cache_dir<R>(dir: &Path, f: impl FnOnce() -> R) -> R {
    let prev = TEST_CACHE_DIR.with(|c| c.replace(Some(dir.to_path_buf())));
    let r = f();
    TEST_CACHE_DIR.with(|c| *c.borrow_mut() = prev);
    r
}

pub fn reencode_cache_dir() -> Option<PathBuf> {
    #[cfg(test)]
    return TEST_CACHE_DIR.with(|c| c.borrow().as_ref().map(|d| d.join("reencoded")));
    #[cfg(not(test))]
    crate::hardware::default_cache_dir()
        .ok()
        .map(|d| d.join("reencoded"))
}

#[derive(Debug, Clone, Copy)]
pub struct GateContext {
    pub requested_policy: ReencodePolicy,
    pub max_kl: Option<f64>,
    pub allow_redirect: bool,
}

static GATE_CONTEXT: Mutex<Option<GateContext>> = Mutex::new(None);

pub fn set_gate_context(ctx: Option<GateContext>) {
    *GATE_CONTEXT.lock() = ctx;
}

pub fn gate_context() -> Option<GateContext> {
    *GATE_CONTEXT.lock()
}

pub fn choice_key(
    weights: &Path,
    policy: ReencodePolicy,
    max_kl: Option<f64>,
    force: Option<Q>,
) -> Option<String> {
    let m = weights.metadata().ok()?;
    let mtime = m
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let kl_bits = max_kl.map(f64::to_bits);
    let s = format!(
        "{}|{}|{mtime}|{}|{kl_bits:?}|{force:?}|{CODEC_VERSION}",
        weights.display(),
        m.len(),
        policy.name(),
    );
    let mut h = s.bytes().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    });
    if force.is_none()
        && matches!(
            policy,
            ReencodePolicy::MaxSpeed | ReencodePolicy::LowBitNative
        )
    {
        // Every format `rule_with` weighs by rate: the ladder and Q4_0.
        for f in LADDER.into_iter().chain([Q::GGUF_Q4_0]) {
            let bits = crate::model::variant::host_gbps(f).map_or(0, f64::to_bits);
            h = (h ^ bits).wrapping_mul(0x100000001b3);
        }
    }
    Some(format!("{h:016x}"))
}

pub fn read_choice_record(key: &str) -> Option<ChoiceRecord> {
    let dir = reencode_cache_dir()?;
    let bytes = std::fs::read(dir.join(format!("{key}.json"))).ok()?;
    serde_json::from_slice(&bytes).ok()
}

impl ReencodeCache {
    pub fn open(key: &str) -> Option<Arc<Self>> {
        let dir = reencode_cache_dir()?;
        let blocks_path = dir.join(format!("{key}.blocks"));
        let index_path = dir.join(format!("{key}.index.json"));
        let index_bytes = std::fs::read(&index_path).ok()?;
        let index: BTreeMap<String, CachedTensor> = serde_json::from_slice(&index_bytes).ok()?;
        let file = std::fs::File::open(&blocks_path).ok()?;
        // SAFETY: read-only mapping of a cache file replaced only by atomic rename.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.ok()?;
        #[cfg(unix)]
        unsafe {
            libc::madvise(
                mmap.as_ptr() as *mut libc::c_void,
                mmap.len(),
                libc::MADV_WILLNEED,
            );
        }
        Some(Arc::new(Self {
            key: key.to_string(),
            map: Arc::new(GgufData::Map(mmap)),
            index,
        }))
    }
}

/// Records a cache hit for `name` in `NATIVE_STATS` and `PENDING_REENCODE`.

pub fn record_cache_hit(
    key: &str,
    name: &str,
    src: Q,
    format: Q,
    map: Arc<GgufData>,
    off: usize,
    bytes: usize,
    rows: usize,
    cols: usize,
) {
    let mut stats = NATIVE_STATS.lock();
    stats.tensors += 1;
    stats.reencoded += 1;
    stats.cache_hits += 1;
    stats.bytes_before += (rows * src.tensor_bytes(cols)) as u64;
    stats.bytes_after += bytes as u64;
    drop(stats);
    record_pending_tensor(key, name, format, map, off, bytes, rows, cols);
}

pub fn record_cache_mismatch() {
    NATIVE_STATS.lock().cache_mismatch = true;
}

pub fn has_cache_mismatch() -> bool {
    NATIVE_STATS.lock().cache_mismatch
}

pub fn record_pending_tensor(
    key: &str,
    name: &str,
    format: Q,
    map: Arc<GgufData>,
    off: usize,
    bytes: usize,
    rows: usize,
    cols: usize,
) {
    let mut guard = PENDING_REENCODE.lock();
    let entry = guard.get_or_insert_with(|| (key.to_string(), BTreeMap::new()));
    if entry.0 != key {
        *entry = (key.to_string(), BTreeMap::new());
    }
    entry.1.insert(
        name.to_string(),
        PendingTensor {
            format,
            map,
            off,
            bytes,
            rows,
            cols,
        },
    );
}

pub fn take_pending_reencode() -> Option<PendingReencode> {
    PENDING_REENCODE.lock().take()
}

pub fn save_reencode_cache(
    (key, tensors): PendingReencode,
    choice: ChoiceRecord,
) -> Result<(), String> {
    if tensors.is_empty() {
        return Ok(());
    }
    let Some(dir) = reencode_cache_dir() else {
        return Ok(());
    };
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let blocks_path = dir.join(format!("{key}.blocks"));
    let index_path = dir.join(format!("{key}.index.json"));
    let choice_path = dir.join(format!("{key}.json"));
    let pid = std::process::id();
    let blocks_tmp = dir.join(format!("{key}.blocks.tmp.{pid}"));
    let index_tmp = dir.join(format!("{key}.index.json.tmp.{pid}"));
    let choice_tmp = dir.join(format!("{key}.json.tmp.{pid}"));
    let io_err = |e: std::io::Error| e.to_string();
    let mut file = std::io::BufWriter::new(std::fs::File::create(&blocks_tmp).map_err(io_err)?);
    let mut index = BTreeMap::new();
    let mut offset = 0usize;
    {
        use std::io::Write;
        for (name, t) in tensors {
            let pad = (TENSOR_ALIGN - (offset % TENSOR_ALIGN)) % TENSOR_ALIGN;
            if pad > 0 {
                file.write_all(&[0u8; TENSOR_ALIGN][..pad])
                    .map_err(io_err)?;
                offset += pad;
            }
            let slice = t
                .map
                .get(t.off..t.off + t.bytes)
                .ok_or_else(|| format!("tensor `{name}` slice out of bounds"))?;
            file.write_all(slice).map_err(io_err)?;
            index.insert(
                name,
                CachedTensor {
                    format: t.format,
                    offset,
                    bytes: t.bytes,
                    rows: t.rows,
                    cols: t.cols,
                },
            );
            offset += t.bytes;
        }
        let tail_pad = (PAGE_ALIGN - (offset % PAGE_ALIGN)) % PAGE_ALIGN;
        if tail_pad > 0 {
            file.write_all(&vec![0u8; tail_pad]).map_err(io_err)?;
        }
        file.flush().map_err(io_err)?;
    }
    drop(file);
    let json_bytes = serde_json::to_vec_pretty(&index).map_err(|e| e.to_string())?;
    let choice_bytes = serde_json::to_vec_pretty(&choice).map_err(|e| e.to_string())?;
    std::fs::write(&index_tmp, json_bytes).map_err(io_err)?;
    std::fs::write(&choice_tmp, choice_bytes).map_err(io_err)?;
    std::fs::rename(&blocks_tmp, &blocks_path).map_err(io_err)?;
    std::fs::rename(&index_tmp, &index_path).map_err(io_err)?;
    std::fs::rename(&choice_tmp, &choice_path).map_err(io_err)
}
#[cfg(test)]
mod tests {
    use super::*;
    use ReencodePolicy::*;

    /// Rates in GB/s as the codec benchmark measured them on the dev machine (TQ2_0 is the slow one).
    fn rates(f: Q) -> Option<f64> {
        Some(match f {
            Q::GGUF_TQ2_0 => 110.0,
            Q::GGUF_Q4_0 => 300.0,
            Q::GGUF_Q4_K_M => 250.0,
            Q::GGUF_Q5_K_M => 240.0,
            Q::GGUF_Q6_K => 230.0,
            Q::GGUF_Q8_0 => 380.0,
            Q::BF16 | Q::FP16 => 200.0,
            _ => return None,
        })
    }

    /// Smaller is not faster: slow Q4_K (4.5/100) loses to the larger Q5_K (5.5/240); once Q4_K is fast it wins back.
    #[test]
    fn max_speed_picks_min_bytes_over_gbps_not_min_bytes() {
        let slow_q4k = |f: Q| {
            if f == Q::GGUF_Q4_K_M {
                Some(100.0)
            } else {
                rates(f)
            }
        };
        // Q4_K 4.5/100 = .045; Q8_0 8.5/380 = .0224; Q4_0 4.5/300 = .015 is best of all.
        assert_eq!(
            rule_with(MaxSpeed, Q::BF16, &slow_q4k),
            Target::Encode(Q::GGUF_Q4_0)
        );
        // Without Q4_0: Q5_K (5.5/240) beats slow Q4_K; Q8_0 (8.5/380) is inside the noise margin of it.
        let no_q4_0 = |f: Q| if f == Q::GGUF_Q4_0 { None } else { slow_q4k(f) };
        assert_eq!(
            rule_with(MaxSpeed, Q::BF16, &no_q4_0),
            Target::Encode(Q::GGUF_Q5_K_M)
        );
        let fast_q4k = |f: Q| {
            if f == Q::GGUF_Q4_K_M {
                Some(400.0)
            } else {
                no_q4_0(f)
            }
        };
        assert_eq!(
            rule_with(MaxSpeed, Q::BF16, &fast_q4k),
            Target::Encode(Q::GGUF_Q4_K_M)
        );
    }

    #[test]
    fn max_speed_moves_a_slow_low_bit_source_to_the_faster_target() {
        // 2.06 bpw / 110 GB/s is slower per weight than 4.5 bpw / 300 GB/s.
        assert_eq!(
            rule_with(MaxSpeed, Q::GGUF_TQ2_0, &rates),
            Target::Encode(Q::GGUF_Q4_0)
        );
        // Q8_0 (8.5 bpw / 380) takes longer per weight than Q4_0 (4.5 / 300): never "faster by bits".
        assert_eq!(
            rule_with(MaxSpeed, Q::GGUF_Q8_0, &rates),
            Target::Encode(Q::GGUF_Q4_0)
        );
        // The source stays when nothing beats it by the margin (ties keep the source).
        assert_eq!(rule_with(MaxSpeed, Q::GGUF_Q4_0, &rates), Target::Native);
    }

    /// Measured on this host (E2B Q4_K_M, GB/s): a Q4_K tensor the RMSE gate bumps to Q5_K must stay Q4_K.
    #[test]
    fn a_bumped_encode_keeps_the_source_unless_it_is_still_faster() {
        let measured = |f: Q| match f {
            Q::GGUF_Q4_K_M => Some(230.0),
            Q::GGUF_Q5_K_M => Some(171.0),
            Q::GGUF_Q6_K => Some(151.0),
            Q::GGUF_Q4_0 => Some(255.0),
            _ => None,
        };
        assert!(!beats_source(Q::GGUF_Q5_K_M, Q::GGUF_Q4_K_M, &measured));
        assert!(beats_source(Q::GGUF_Q5_K_M, Q::GGUF_Q6_K, &measured));
        assert!(beats_source(Q::GGUF_Q4_0, Q::GGUF_Q6_K, &measured));
        // Unmeasured: only fewer bits count.
        assert!(!beats_source(Q::GGUF_Q5_K_M, Q::GGUF_Q4_K_M, &|_| None));
        assert!(beats_source(Q::GGUF_Q4_0, Q::GGUF_Q6_K, &|_| None));
    }

    #[test]
    fn max_speed_ignores_unmeasured_targets_and_keeps_precision() {
        let only_q8 = |f: Q| match f {
            Q::GGUF_TQ2_0 => Some(20.0),
            Q::GGUF_Q8_0 => Some(380.0),
            _ => None,
        };
        assert_eq!(
            rule_with(MaxSpeed, Q::GGUF_TQ2_0, &only_q8),
            Target::Encode(Q::GGUF_Q8_0)
        );
        for src in [
            Q::GGUF_TQ2_0,
            Q::GGUF_IQ2_M,
            Q::GGUF_Q4_K_M,
            Q::GGUF_IQ4_XS,
            Q::GGUF_Q6_K,
            Q::BF16,
        ] {
            let stored = match rule_with(MaxSpeed, src, &rates) {
                Target::Native => src,
                Target::Encode(f) => f,
            };
            let floor = if src.bits_per_weight() > 4.5 {
                4.0
            } else {
                src.bits_per_weight() - 0.01
            };
            assert!(stored.bits_per_weight() >= floor, "{src:?} -> {stored:?}");
        }
        // No rates at all: the static rule.
        assert_eq!(
            rule_with(MaxSpeed, Q::GGUF_TQ2_0, &|_| None),
            Target::Native
        );
        assert_eq!(
            rule_with(MaxSpeed, Q::BF16, &|_| None),
            Target::Encode(Q::GGUF_Q4_K_M)
        );
    }

    #[test]
    fn tq2_0_to_q4_0_is_bit_exact() {
        use crate::model::layers::{gguf_block_codec, quantize_f32_to_q4_0};
        let (elems, bytes, tq) = gguf_block_codec(Q::GGUF_TQ2_0).unwrap();
        let (_, _, q4) = gguf_block_codec(Q::GGUF_Q4_0).unwrap();
        let mut s = 0x9E3779B97F4A7C15u64;
        for _ in 0..64 {
            let mut blk = vec![0u8; bytes];
            for b in &mut blk[..bytes - 2] {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *b = (s >> 33) as u8;
            }
            blk[bytes - 2..].copy_from_slice(
                &crate::model::f32_to_f16(0.0371 + (s >> 60) as f32 * 0.01).to_le_bytes(),
            );
            let mut want = vec![0.0f32; elems];
            tq(&blk, &mut want);
            let enc = quantize_f32_to_q4_0(&want);
            let mut got = vec![0.0f32; 32];
            for (i, b) in enc.chunks_exact(18).enumerate() {
                q4(b, &mut got);
                for (a, w) in got.iter().zip(&want[i * 32..]) {
                    assert_eq!(a, w, "block {i}");
                }
            }
        }
    }

    #[test]
    fn q3_k_encode_decode_matches_gpu_codec() {
        let (rows, cols) = (64, 1024);
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut u = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as u32) as f32 / 2147483648.0
        };
        let w: Vec<f32> = (0..rows * cols)
            .map(|_| (u() + u() + u() + u() - 2.0) * 0.1)
            .collect();
        let raw = encode_rows(&w, cols, Q::GGUF_Q3_K).unwrap();
        let cpu = dequantize_gguf_blocks(Q::GGUF_Q3_K, &raw).unwrap();
        let rmse = relative_rmse(&w, &cpu);
        assert!(rmse <= 0.20, "Q3_K relative RMSE {rmse} > 0.20");

        #[cfg(target_os = "macos")]
        {
            use crate::graph::codec::{codec, msl_prelude};
            use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};

            let c = codec(Q::GGUF_Q3_K).unwrap();
            let (block_elems, block_bytes) = (c.block_elems as usize, c.block_bytes as usize);
            let subs = block_elems / 8;
            let msl = format!("{}{}", msl_prelude(), c.msl);
            let src = format!(
                "#include <metal_stdlib>\nusing namespace metal;\n{msl}\n\
                 kernel void decode_all(device const uchar* blocks [[buffer(0)]], device float* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n\
                     uint b = gid / {subs}u, sub = gid % {subs}u;\n\
                     float o[8];\n\
                     decode8(blocks + (ulong)b * {block_bytes}ul, sub, o);\n\
                     for (uint i = 0; i < 8; i++) out[gid * 8 + i] = o[i];\n\
                 }}\n"
            );
            let device = Device::system_default().expect("no Metal device");
            let opts = CompileOptions::new();
            opts.set_fast_math_enabled(false);
            let lib = device.new_library_with_source(&src, &opts).unwrap();
            let pso = device
                .new_compute_pipeline_state_with_function(
                    &lib.get_function("decode_all", None).unwrap(),
                )
                .unwrap();
            let threads = raw.len() / block_bytes * subs;
            let inb = device.new_buffer_with_data(
                raw.as_ptr().cast(),
                raw.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let outb = device.new_buffer(
                (threads * 8 * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let q = device.new_command_queue();
            let cb = q.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&inb), 0);
            enc.set_buffer(1, Some(&outb), 0);
            let tg = pso.max_total_threads_per_threadgroup().min(64);
            enc.dispatch_threads(MTLSize::new(threads as u64, 1, 1), MTLSize::new(tg, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let gpu =
                unsafe { std::slice::from_raw_parts(outb.contents() as *const f32, threads * 8) };
            for (i, (&g, &w)) in gpu.iter().zip(&cpu).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "elem {i}: gpu {g:e} != cpu {w:e}");
            }
        }
    }

    #[test]
    fn rules_follow_the_policy_table() {
        crate::model::variant::with_unmeasured_host(|| {
            // MaxSpeed: > 4.5 bpw or codec-less -> Q4_K, else native.
            for src in [
                Q::BF16,
                Q::FP16,
                Q::FP32,
                Q::GGUF_Q8_0,
                Q::GGUF_Q6_K,
                Q::GGUF_Q5_K_M,
                Q::GGUF_Q4_1,
            ] {
                assert_eq!(
                    rule(MaxSpeed, src),
                    Target::Encode(Q::GGUF_Q4_K_M),
                    "{src:?}"
                );
            }
            for src in [
                Q::GGUF_Q4_0,
                Q::GGUF_Q4_K_M,
                Q::GGUF_IQ4_XS,
                Q::GGUF_Q2_K,
                Q::GGUF_IQ2_M,
                Q::GGUF_TQ2_0,
            ] {
                assert_eq!(rule(MaxSpeed, src), Target::Native, "{src:?}");
            }
            assert_eq!(
                rule(MaxSpeed, Q::TurboQuant),
                Target::Encode(Q::GGUF_Q4_K_M)
            );
            // LowBitNative: float -> Q8_0; codec-less -> smallest ladder format >= source bpw.
            for src in [Q::BF16, Q::FP16, Q::FP32] {
                assert_eq!(rule(LowBitNative, src), Target::Encode(Q::GGUF_Q8_0));
            }
            assert_eq!(rule(LowBitNative, Q::GGUF_Q4_K_M), Target::Native);
            assert_eq!(
                rule(LowBitNative, Q::TurboQuant),
                Target::Encode(Q::GGUF_Q4_K_M)
            );
            // BitExact: native whenever a codec exists.
            for src in [Q::BF16, Q::FP32, Q::GGUF_Q8_0, Q::GGUF_Q4_0] {
                assert_eq!(rule(BitExact, src), Target::Native);
            }
            assert_eq!(
                rule(BitExact, Q::TurboQuant),
                Target::Encode(Q::GGUF_Q4_K_M)
            );
            // MinSize: rate-independent, no 4-bpw floor; FFN -> Q3_K (Q4_0 off-256), attn -> Q4_K (Q4_0 off-256),
            // lm_head / #linear -> Q6_K; never adds bits.
            for src in [
                Q::BF16,
                Q::FP16,
                Q::FP32,
                Q::GGUF_Q8_0,
                Q::GGUF_Q6_K,
                Q::GGUF_Q5_K_M,
                Q::GGUF_Q4_K_M,
                Q::GGUF_Q4_0,
            ] {
                assert_eq!(
                    rule_with(MinSize, src, &rates),
                    Target::Encode(Q::GGUF_Q3_K),
                    "{src:?}"
                );
            }
            for src in [Q::GGUF_Q3_K, Q::GGUF_Q2_K, Q::GGUF_IQ2_M, Q::GGUF_TQ2_0] {
                assert_eq!(rule_with(MinSize, src, &rates), Target::Native, "{src:?}");
            }
            let min_cfg = crate::model::weight_format::ReencodeConfig {
                policy: MinSize,
                ..Default::default()
            };
            assert_eq!(
                target(
                    &min_cfg,
                    "model.layers.0.mlp.up_proj.weight",
                    Q::BF16,
                    256,
                    true
                ),
                Some(Q::GGUF_Q3_K)
            );
            assert_eq!(
                target(
                    &min_cfg,
                    "model.layers.0.mlp.up_proj.weight",
                    Q::BF16,
                    576,
                    true
                ),
                Some(Q::GGUF_Q4_0)
            );
            assert_eq!(
                target(
                    &min_cfg,
                    "model.layers.0.self_attn.q_proj.weight",
                    Q::BF16,
                    256,
                    true
                ),
                Some(Q::GGUF_Q4_K_M)
            );
            assert_eq!(
                target(
                    &min_cfg,
                    "model.layers.0.self_attn.q_proj.weight",
                    Q::BF16,
                    576,
                    true
                ),
                Some(Q::GGUF_Q4_0)
            );
            assert_eq!(
                target(&min_cfg, "lm_head.weight", Q::BF16, 256, true),
                Some(Q::GGUF_Q6_K)
            );
            assert_eq!(
                target(
                    &min_cfg,
                    "model.embed_tokens.weight#linear",
                    Q::BF16,
                    256,
                    true
                ),
                Some(Q::GGUF_Q6_K)
            );
            assert_eq!(
                target(&min_cfg, "lm_head.weight", Q::GGUF_Q4_K_M, 256, false),
                None,
                "MinSize never adds bits"
            );
            assert_eq!(
                target(
                    &min_cfg,
                    "model.layers.0.self_attn.q_proj.weight",
                    Q::GGUF_Q3_K,
                    256,
                    false
                ),
                None,
                "MinSize never adds bits"
            );
            assert_eq!(
                target(&min_cfg, "blk.0.attn_output.weight", Q::BF16, 256, true),
                Some(Q::GGUF_Q4_K_M),
                "attn_output.weight is attention, not lm_head"
            );
            assert_eq!(
                target(&min_cfg, "output.weight", Q::BF16, 256, true),
                Some(Q::GGUF_Q6_K)
            );
        });
    }

    #[test]
    fn fit_bumps_until_blocks_tile() {
        assert_eq!(fit(Q::GGUF_Q4_K_M, 512, Q::FP32), Q::GGUF_Q4_K_M);
        // 576 = 2.25 x 256 (SmolLM2 hidden): block-32 Q4_0, not an 8.5 bpw Q8_0
        assert_eq!(fit(Q::GGUF_Q4_K_M, 576, Q::FP32), Q::GGUF_Q4_0);
        assert_eq!(fit(Q::GGUF_Q5_K_M, 576, Q::FP32), Q::GGUF_Q8_0);
        assert_eq!(fit(Q::GGUF_Q4_K_M, 64, Q::FP32), Q::GGUF_Q4_0);
        assert_eq!(fit(Q::GGUF_Q4_K_M, 48, Q::FP32), Q::FP32);
        assert_eq!(bump(Q::GGUF_Q8_0), None);
    }

    #[test]
    fn threaded_encode_equals_single_thread() {
        let w: Vec<f32> = (0..512 * 160)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.003)
            .collect();
        let one = encode_weight(&w, 512, 160, Q::GGUF_Q5_K_M, 1);
        let many = encode_weight(&w, 512, 160, Q::GGUF_Q5_K_M, 7);
        assert_eq!(one, many);
    }

    /// The row-run trial picks the same format and blocks as one whole-tensor encode, over a
    /// tensor spanning several runs with a short last one.
    #[test]
    fn native_trial_in_row_runs_matches_whole_tensor() {
        let (cols, src) = (256, Q::GGUF_Q8_0);
        let rows = 2 * (TRIAL_VALUES / cols) + 5;
        let mut s = 0x9e3779b97f4a7c15u64;
        let w: Vec<f32> = (0..rows * cols)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect();
        let raw = encode_weight(&w, cols, rows, src, 4);
        let cfg = crate::model::weight_format::ReencodeConfig {
            policy: MaxSpeed,
            ..Default::default()
        };
        // The whole-tensor procedure the row runs replace.
        let whole = || -> Option<(Q, Vec<u8>)> {
            let Target::Encode(f) = rule(cfg.policy, src) else {
                return None;
            };
            let f = fit(f, cols, Q::FP32);
            let (elems, block_bytes, decode) = gguf_block_codec(src)?;
            let mut data = vec![0.0f32; rows * cols];
            for (b, o) in raw
                .chunks_exact(block_bytes)
                .zip(data.chunks_exact_mut(elems))
            {
                decode(b, o);
            }
            let mut f = f;
            let mut q = encode_weight(&data, cols, rows, f, 4);
            while !is_float(f)
                && relative_rmse(&data, &dequantize_gguf_blocks(f, &q).unwrap())
                    > rmse_limit(cfg.policy)
            {
                let Some(next) = bump(f).map(|n| fit(n, cols, Q::FP32)) else {
                    break;
                };
                f = next;
                if !is_float(f) {
                    q = encode_weight(&data, cols, rows, f, 4);
                }
            }
            if is_float(f) || !beats_source(f, src, &crate::model::variant::host_gbps) {
                return None;
            }
            Some((f, q))
        };
        let want = whole();
        assert!(
            want.is_some(),
            "MaxSpeed re-stores Q8_0 here: the test compares blocks"
        );
        assert_eq!(
            reencode_native(&cfg, "layers.0.mlp.up.weight", src, rows, cols, &raw),
            want
        );
    }

    /// A `runtime.sensitive_weights` glob holds a tensor to the LM head's 0.05 error floor: the
    /// same weights re-encode to a wider format than without the glob.
    #[test]
    fn sensitive_weight_glob_tightens_rmse_limit() {
        let (rows, cols, name) = (64, 256, "model.layers.3.self_attn.q_proj.weight");
        let mut s = 0x2545f4914f6cdd1du64;
        let w: Vec<f32> = (0..rows * cols)
            .map(|_| {
                (0..4)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                    })
                    .sum::<f32>()
            })
            .collect();
        let run = |sensitive: Option<Vec<String>>| {
            let cfg = crate::model::weight_format::ReencodeConfig {
                policy: MinSize,
                sensitive_weights: sensitive.map(Arc::new),
                ..Default::default()
            };
            let (f, blocks) =
                reencode_values(&cfg, name, Q::BF16, rows, cols, || Some(w.clone())).unwrap();
            (f, relative_rmse(&w, &dequantize_gguf_blocks(f, &blocks).unwrap()))
        };
        let (loose, loose_err) = run(None);
        let (tight, tight_err) = run(Some(vec!["*.self_attn.*".into()]));
        assert!(loose_err > 0.05, "{loose:?} rmse {loose_err}");
        assert!(tight_err <= 0.05, "{tight:?} rmse {tight_err}");
        assert!(tight.bits_per_weight() > loose.bits_per_weight());
    }

    #[test]
    fn moe_router_is_never_classified_as_ffn() {
        assert_eq!(
            classify_tensor("model.layers.0.mlp.gate.weight", Some(0), Some(28)),
            TensorRole::Router
        );
        assert_eq!(
            classify_tensor(
                "model.layers.0.mlp.shared_expert_gate.weight",
                Some(0),
                Some(28)
            ),
            TensorRole::Router
        );
        assert_eq!(
            classify_tensor("blk.0.ffn_gate_inp.weight", Some(0), Some(28)),
            TensorRole::Router
        );
        assert!(!is_ffn_name("model.layers.0.mlp.gate.weight"));
        assert!(is_ffn_name("model.layers.0.mlp.gate_proj.weight"));
        assert_eq!(
            classify_tensor("model.layers.0.mlp.down_proj.weight", Some(0), Some(28)),
            TensorRole::FfnDown
        );
        assert_eq!(
            classify_tensor("model.layers.0.self_attn.q_proj.weight", Some(0), Some(28)),
            TensorRole::BoundaryAttn
        );
        let min_cfg = crate::model::weight_format::ReencodeConfig {
            policy: MinSize,
            ..Default::default()
        };
        assert_eq!(
            target(
                &min_cfg,
                "model.layers.0.mlp.gate.weight",
                Q::BF16,
                256,
                true
            ),
            None,
            "MoE router stays native under MinSize when GPU codec exists"
        );
    }
}
