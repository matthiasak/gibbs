//! Weight re-encoding policy and its quality gate. Spec loads ([`crate::arch::load`]) store
//! weights in their checkpoint encoding or re-encode them per [`ReencodePolicy`] (`--quant`:
//! [`ReencodeConfig::force`]); a re-encoded model must stay within the policy's KL limit of the
//! source model over a probe text ([`native_kl_gate`]), else the load retries more exactly.

use crate::model::reencode::ReencodeProgress;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Quality floor: largest mean KL divergence (nats) of a format's next-token distribution from
/// the f32 one over the probe. Roughly where 6-bit block formats land; 4-bit ones exceed it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReencodePolicy {
    #[serde(alias = "maxspeed", alias = "speed", alias = "MaxSpeed")]
    MaxSpeed,
    #[serde(alias = "lowbitnative", alias = "lowbit", alias = "LowBitNative")]
    LowBitNative,
    #[serde(alias = "bitexact", alias = "exact", alias = "BitExact")]
    BitExact,
    #[serde(
        rename = "min-size",
        alias = "max-compress",
        alias = "minsize",
        alias = "MinSize"
    )]
    MinSize,
}

impl ReencodePolicy {
    /// The CLI / settings spelling.
    pub fn name(self) -> &'static str {
        match self {
            Self::MaxSpeed => "max-speed",
            Self::LowBitNative => "low-bit-native",
            Self::BitExact => "bit-exact",
            Self::MinSize => "min-size",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::MaxSpeed => Self::LowBitNative,
            Self::LowBitNative => Self::BitExact,
            Self::BitExact => Self::MaxSpeed,
            Self::MinSize => Self::MaxSpeed,
        }
    }
}

impl std::fmt::Display for ReencodePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl Default for ReencodePolicy {
    fn default() -> Self {
        Self::MaxSpeed
    }
}

impl std::str::FromStr for ReencodePolicy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "max-speed" | "maxspeed" | "speed" => Ok(Self::MaxSpeed),
            "low-bit-native" | "lowbitnative" | "lowbit" => Ok(Self::LowBitNative),
            "bit-exact" | "bitexact" | "exact" => Ok(Self::BitExact),
            "min-size" | "max-compress" | "minsize" => Ok(Self::MinSize),
            _ => Err(format!("unknown reencode policy '{s}'")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VariantPolicy {
    #[serde(alias = "fast", alias = "Fastest")]
    Fastest,
    #[serde(rename = "fastest-4bit", alias = "fastest4bit", alias = "Fastest4Bit")]
    Fastest4Bit,
    #[serde(alias = "bestquality", alias = "quality", alias = "BestQuality")]
    BestQuality,
}

impl VariantPolicy {
    /// The settings / `--variant` spelling.
    pub fn name(self) -> &'static str {
        match self {
            Self::Fastest => "fastest",
            Self::Fastest4Bit => "fastest-4bit",
            Self::BestQuality => "best-quality",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Fastest => Self::Fastest4Bit,
            Self::Fastest4Bit => Self::BestQuality,
            Self::BestQuality => Self::Fastest,
        }
    }
}

impl std::fmt::Display for VariantPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl Default for VariantPolicy {
    fn default() -> Self {
        Self::Fastest
    }
}

impl std::str::FromStr for VariantPolicy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "fastest" | "fast" => Ok(Self::Fastest),
            "fastest-4bit" | "fastest4bit" => Ok(Self::Fastest4Bit),
            "best-quality" | "bestquality" | "quality" => Ok(Self::BestQuality),
            _ => Err(format!("unknown variant policy '{s}'")),
        }
    }
}

/// Quality floor: largest mean KL divergence (nats) of a re-encoded model's next-token
/// distribution from the source's over the probe. Roughly where 6-bit block formats land; 4-bit
/// ones exceed it.
pub const MAX_MEAN_KL: f64 = 0.01;
/// Teacher-forced probe positions of the gate.
const PROBE_TOKENS: usize = 48;
const PROBE_TEXT: &str = "The engine measures each weight format on this machine before it serves \
    the model. It keeps the fastest format whose predictions stay close to full precision, because \
    a format that is fast but wrong does not help anyone who reads the answers it writes.";
const PROBE_TEXT_LONG: &str = " A quantized linear layer maps an input vector x in R^d to y = W x \
    by reading packed blocks of 32 or 256 weights: for block b in 0..k/256 { let d = f16(blk[b].d); \
    acc += d * fma(q[i], x[i]); } At 293.4 GB/s ceiling, 128 positions (42, 3.14159, 0x100, -7, \
    99.9%) test whether 3-bit feed-forward projections and 4-bit attention projections preserve the \
    model's next-token distribution across prose, numbers, and code.";

/// How a GPU load re-encodes its weights (S3.5).
#[derive(Clone, Default)]
pub struct ReencodeConfig {
    pub policy: ReencodePolicy,
    /// Overrides the KL limit of the whole-model gate.
    pub max_kl: Option<f64>,
    /// Filled while tensors are re-encoded, for the load-stage text.
    pub progress: Option<Arc<ReencodeProgress>>,
    /// Receives every `reencode: ...` log line (the CLI routes it to its telemetry log).
    pub log: Option<Arc<dyn Fn(&str) + Send + Sync>>,
    /// Native loaders: only feed-forward projections are re-encoded (the second gate attempt).
    pub ffn_only: bool,
    /// The feed-forward weights `ffn_only` keeps (`arch::glob` patterns over canonical names),
    /// from an active spec's `runtime.ffn_weights`; `None` = the built-in name test.
    pub ffn_weights: Option<Arc<Vec<String>>>,
    /// Weights held to the LM head's re-encode error floor (`arch::glob` patterns over canonical
    /// names), from an active spec's `runtime.sensitive_weights`.
    pub sensitive_weights: Option<Arc<Vec<String>>>,
    /// The checkpoint's layer count (one past the highest layer index in its tensor names), so
    /// the last layer's attention counts as a boundary layer.
    pub total_layers: Option<usize>,
    /// `--quant <fmt>`: every spec-loaded weight matrix is stored as this format (f32/f16/bf16
    /// or an encodable `graph::codec` block format) instead of its checkpoint encoding.
    pub force: Option<crate::hardware::QuantFormat>,
}

/// The configuration loaders use instead of the settings file when set ([`set_reencode_override`]).
static OVERRIDE: parking_lot::RwLock<Option<ReencodeConfig>> = parking_lot::RwLock::new(None);

/// Pins the re-encode configuration every spec loader reads through
/// [`ReencodeConfig::effective`]; `None` returns to the settings file.
pub fn set_reencode_override(c: Option<ReencodeConfig>) {
    *OVERRIDE.write() = c;
}

/// The pinned configuration, if any (to restore after a temporary override).
pub fn reencode_override() -> Option<ReencodeConfig> {
    OVERRIDE.read().clone()
}

impl ReencodeConfig {
    /// `~/.cache/gibbs/settings.json` (`reencode`, `reencode_max_kl`), which the TUI writes.
    pub fn from_settings() -> Self {
        let s = crate::tui::settings::load_settings();
        Self {
            policy: s.reencode,
            max_kl: s.reencode_max_kl.map(f64::from),
            ..Self::default()
        }
    }

    /// The override ([`set_reencode_override`]) when set, else the settings file.
    pub fn effective() -> Self {
        OVERRIDE.read().clone().unwrap_or_else(Self::from_settings)
    }

    /// CLI flags (`--reencode`, `--reencode-max-kl`) over the settings file.
    pub fn resolve(cli_policy: Option<&str>, cli_max_kl: Option<f32>) -> Result<Self, String> {
        let mut c = Self::from_settings();
        if let Some(p) = cli_policy {
            c.policy = p.parse()?;
        }
        if let Some(k) = cli_max_kl {
            c.max_kl = Some(f64::from(k));
        }
        Ok(c)
    }
}

/// `--quant` spellings: the format every spec-loaded weight matrix is stored as
/// ([`ReencodeConfig::force`]). `nvfp4` is the `GGUF_NVFP4` block codec.
pub fn parse_quant(s: &str) -> Result<crate::hardware::QuantFormat, String> {
    use crate::hardware::QuantFormat as Q;
    let f = match s.to_lowercase().as_str() {
        "uncompressed" | "none" | "f32" | "fp32" => Q::FP32,
        "f16" | "fp16" => Q::FP16,
        "bf16" => Q::BF16,
        "q8_0" | "q8" | "int8" => Q::GGUF_Q8_0,
        "nvfp4" | "nvfp4-ultra" | "fp4" => Q::GGUF_NVFP4,
        other => Q::from_str_lenient(other).ok_or_else(|| format!("Unknown --quant '{other}'"))?,
    };
    if !crate::model::reencode::encodable(f) {
        return Err(format!(
            "--quant {s}: weights cannot be encoded as {}",
            f.name()
        ));
    }
    Ok(f)
}

fn gate_limits(policy: ReencodePolicy, max_kl: Option<f64>) -> (f64, f64) {
    let (kl, top1) = match policy {
        ReencodePolicy::MaxSpeed => (0.10, 0.90),
        ReencodePolicy::MinSize => (0.25, 0.80),
        _ => (MAX_MEAN_KL, 0.99),
    };
    (max_kl.unwrap_or(kl), top1)
}

/// Result of comparing a candidate model's next-token distributions with the source model's.
#[derive(Debug, Clone, Copy)]
pub struct GateReport {
    pub mean_kl: f64,
    pub top1: f64,
    pub tokens: usize,
}

/// Teacher-forces [`PROBE_TEXT`] one token at a time through `source` and `candidate` (both on
/// Metal) and compares the next-token distributions: mean KL(source ‖ candidate) in nats and top-1
/// agreement, over [`PROBE_TOKENS`] positions.
pub fn native_kl_gate(
    source: &dyn crate::model::native::NativeLm,
    candidate: &dyn crate::model::native::NativeLm,
) -> Result<GateReport, String> {
    let min_size = ReencodeConfig::effective().policy == ReencodePolicy::MinSize;
    let probe_tokens = if min_size { 128 } else { PROBE_TOKENS };
    let tok = source.tokenizer();
    let vocab = tok.vocab_size().max(1) as u32;
    let mut ids = if min_size {
        tok.encode(&format!("{PROBE_TEXT}{PROBE_TEXT_LONG}"))
    } else {
        tok.encode(PROBE_TEXT)
    };
    if ids.is_empty() {
        ids = (1..=probe_tokens as u32).collect();
    }
    let n = probe_tokens
        .min(source.context_len().min(candidate.context_len()))
        .max(2);
    let tokens: Vec<u32> = ids.iter().cycle().take(n).map(|t| t % vocab).collect();
    let run = |m: &dyn crate::model::native::NativeLm| -> Result<Vec<Vec<f32>>, String> {
        let (_, mut state) = m.prepare(vec![tokens[0]], &[])?;
        tokens
            .iter()
            .enumerate()
            .map(|(pos, &t)| {
                let logits = m.forward(&[t], pos, &mut state)?;
                Ok(log_softmax(&logits[..logits.len().min(vocab as usize)]))
            })
            .collect()
    };
    let want = run(source)?;
    let (_, mut state) = candidate.prepare(vec![tokens[0]], &[])?;
    let mut kl_sum = 0.0f64;
    let mut top1_hits = 0usize;
    for (pos, (&t, p)) in tokens.iter().zip(&want).enumerate() {
        let logits = candidate.forward(&[t], pos, &mut state)?;
        let q = log_softmax(&logits[..logits.len().min(vocab as usize)]);
        kl_sum += kl_divergence(p, &q);
        if argmax(p) == argmax(&q) {
            top1_hits += 1;
        }
    }
    let nf = n as f64;
    Ok(GateReport {
        mean_kl: kl_sum / nf,
        top1: top1_hits as f64 / nf,
        tokens: n,
    })
}

/// Whether a gate result passes the policy's limits (a NaN KL never does).
pub fn gate_passes(policy: ReencodePolicy, max_kl: Option<f64>, r: &GateReport) -> bool {
    let (kl_limit, top1_floor) = gate_limits(policy, max_kl);
    r.mean_kl <= kl_limit && r.top1 >= top1_floor
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = logits
        .iter()
        .map(|&x| ((x - max) as f64).exp())
        .sum::<f64>()
        .ln() as f32;
    logits.iter().map(|&x| x - max - log_sum).collect()
}

/// KL(p ‖ q) from log-probabilities.
fn kl_divergence(log_p: &[f32], log_q: &[f32]) -> f64 {
    log_p
        .iter()
        .zip(log_q)
        .map(|(&p, &q)| (p as f64).exp() * (p - q) as f64)
        .sum()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold(
            (0, f32::NEG_INFINITY),
            |b, (i, &x)| if x > b.1 { (i, x) } else { b },
        )
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gate thresholds: max-speed KL <= 0.10 and top-1 >= 0.90; low-bit-native KL <= 0.01 (top-1 >= 0.99);
    /// a NaN KL never passes; an explicit `max_kl` overrides the KL limit only.
    /// `--quant` names map to encodable formats; NVFP4 is the GGUF_NVFP4 block codec.
    #[test]
    fn quant_names_map_to_formats() {
        use crate::hardware::QuantFormat as Q;
        for (name, f) in [
            ("nvfp4", Q::GGUF_NVFP4),
            ("q8_0", Q::GGUF_Q8_0),
            ("f16", Q::FP16),
            ("bf16", Q::BF16),
            ("uncompressed", Q::FP32),
            ("q4_k_m", Q::GGUF_Q4_K_M),
        ] {
            assert_eq!(parse_quant(name), Ok(f), "{name}");
        }
        assert!(parse_quant("q2_k").is_err() && parse_quant("exl3").is_err());
    }

    #[test]
    fn gate_thresholds_per_policy() {
        let r = |mean_kl: f64, top1: f64| GateReport {
            mean_kl,
            top1,
            tokens: 8,
        };
        use ReencodePolicy::*;
        assert!(gate_passes(MaxSpeed, None, &r(0.10, 0.90)));
        assert!(!gate_passes(MaxSpeed, None, &r(0.1001, 0.95)));
        assert!(!gate_passes(MaxSpeed, None, &r(0.01, 0.8999)));
        assert!(gate_passes(LowBitNative, None, &r(0.01, 0.99)));
        assert!(!gate_passes(LowBitNative, None, &r(0.011, 1.0)));
        assert!(!gate_passes(MaxSpeed, None, &r(f64::NAN, 1.0)));
        assert!(gate_passes(LowBitNative, Some(0.05), &r(0.04, 0.99)));
        assert!(!gate_passes(MaxSpeed, Some(0.02), &r(0.05, 1.0)));
        assert!(gate_passes(MinSize, None, &r(0.25, 0.80)));
        assert!(!gate_passes(MinSize, None, &r(0.2501, 0.90)));
        assert!(!gate_passes(MinSize, None, &r(0.10, 0.7999)));
        assert!(gate_passes(MinSize, Some(0.15), &r(0.15, 0.80)));
        assert!(!gate_passes(MinSize, Some(0.15), &r(0.16, 0.95)));
    }
}
