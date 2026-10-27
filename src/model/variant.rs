//! Variant prediction and selection (sprint-2 S3.6): the decode speed of a checkpoint from its
//! tensor header alone, `1 / Σ_t bytes_t / (eff(fmt_t) · ceiling)` over the streamed matrices after
//! the re-encode policy, the `VariantPolicy` rules over such predictions, and the remote header
//! fetch (system curl range requests, cached by `(repo, file, etag)`).

use crate::hardware::{default_cache_dir, QuantFormat as Q};
use crate::model::loaders::gguf::{parse_gguf_header, GgufHeader, HeaderError};
use crate::model::reencode;
use crate::model::weight_format::{ReencodePolicy, VariantPolicy};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Efficiency (GB/s over the ceiling) of a format nobody has measured yet.
pub const DEFAULT_EFF: f64 = 0.9;
/// A variant fits when its resident bytes are at most this fraction of RAM.
const RAM_FRACTION: f64 = 0.7;
/// Parameters assumed for a variant whose size is unknown (only the ordering matters).
const NOMINAL_PARAMS: u64 = 1_000_000_000;

/// `eff = GB/s / ceiling` per codec, as measured by `Runtime::benchmark_codecs` on this device.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodecEff {
    /// Hash of the codec sources the numbers were measured with; a change re-measures.
    pub codec_hash: u64,
    /// `max(stream bandwidth, best codec GEMV GB/s)`.
    pub ceiling_gbps: f64,
    pub gbps: BTreeMap<String, f64>,
    pub eff: BTreeMap<String, f64>,
}

/// IQ2_S shares IQ2_M's block layout and kernel.
fn key(f: Q) -> String {
    format!("{:?}", if f == Q::GGUF_IQ2_S { Q::GGUF_IQ2_M } else { f })
}

impl CodecEff {
    pub fn key(f: Q) -> String {
        key(f)
    }
}

static HOST: RwLock<(f64, Option<CodecEff>)> = RwLock::new((0.0, None));

#[cfg(test)]
thread_local! {
    /// Rates the calling test thread sees instead of [`HOST`], which any Metal `Runtime` built by
    /// another test overwrites ([`with_unmeasured_host`]).
    static TEST_HOST: std::cell::RefCell<Option<(f64, Option<CodecEff>)>> = const { std::cell::RefCell::new(None) };
}

/// Runs `f` with this thread seeing a host whose codec benchmark has not run (default efficiencies,
/// no measured GB/s), whatever other tests publish through [`set_host_rates`].
#[cfg(test)]
pub(crate) fn with_unmeasured_host<R>(f: impl FnOnce() -> R) -> R {
    let prev = TEST_HOST.with(|t| t.replace(Some((0.0, None))));
    let r = f();
    TEST_HOST.with(|t| *t.borrow_mut() = prev);
    r
}

fn host<R>(f: impl Fn(&(f64, Option<CodecEff>)) -> R) -> R {
    // Tests see an unmeasured host unless they pin one: any Metal `Runtime` built by another test thread
    // publishes its measured rates into HOST, which would make rate-dependent rankings follow the machine.
    #[cfg(test)]
    return TEST_HOST.with(|t| match t.borrow().as_ref() {
        Some(h) => f(h),
        None => f(&(0.0, None)),
    });
    #[cfg(not(test))]
    f(&HOST.read())
}

/// Publishes this host's streaming ceiling and (once measured) codec efficiencies.
pub fn set_host_rates(ceiling_gbps: f64, eff: Option<CodecEff>) {
    *HOST.write() = (ceiling_gbps, eff);
}

pub fn host_ceiling_gbps() -> f64 {
    host(|h| h.1.as_ref().map_or(h.0, |e| e.ceiling_gbps.max(h.0)))
}

/// Measured efficiency of `f`; [`DEFAULT_EFF`] until the codec benchmark has finished.
pub fn host_eff(f: Q) -> f64 {
    host(|h| h.1.as_ref().and_then(|e| e.eff.get(&key(f)).copied())).unwrap_or(DEFAULT_EFF)
}

/// Measured GEMV GB/s of `f` on this host; `None` until the codec benchmark has measured it.
pub fn host_gbps(f: Q) -> Option<f64> {
    host(|h| h.1.as_ref().and_then(|e| e.gbps.get(&key(f)).copied()))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeaderTensor {
    pub name: String,
    pub ndim: usize,
    pub numel: u64,
    /// The contiguous dimension (the row length a block format must tile).
    pub cols: u64,
    /// `None`: packed bytes no format description covers (quantized safetensors); counted as read.
    pub fmt: Option<Q>,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightHeader {
    pub tensors: Vec<HeaderTensor>,
    pub file_bytes: u64,
    /// Share of an expert bank one token reads (`expert_used_count / expert_count`).
    pub expert_frac: f64,
}

/// Names of tables read by index, not streamed, or of towers a text decode never runs.
// ponytail: name heuristics; sharded safetensors MoE experts (one 2-D tensor each) all count.
fn gathered(name: &str, untied_head: bool) -> bool {
    const TABLES: [&str; 4] = [
        "token_embd.weight",
        "embed_tokens.weight",
        "wte.weight",
        "word_embeddings.weight",
    ];
    name.contains("per_layer_token_embd")
        || name.contains("embed_tokens_per_layer")
        || name.contains("position_embd")
        || name.contains("ple_embedding")
        || name.contains("embed_positions")
        || name.contains("position_embeddings")
        || name.contains("token_type_embeddings")
        || name.contains("rel_embeddings")
        || name.starts_with("wpe.")
        || name.contains("vision")
        || name.contains("visual")
        || name.contains("audio")
        || name.starts_with("mm.")
        || name.starts_with("mtp.")
        || name.contains(".mtp.")
        || name.starts_with("nextn.")
        || name.contains(".nextn.")
        || (untied_head && TABLES.iter().any(|t| name.ends_with(t)))
}

/// Computes `expert_frac` (`num_experts_per_tok / num_experts`) from a `config.json` value.
pub fn expert_frac_from_config(cfg: &serde_json::Value) -> Option<f64> {
    let tc = cfg.get("text_config").unwrap_or(cfg);
    let used = tc
        .get("num_experts_per_tok")
        .or_else(|| tc.get("num_selected_experts"))
        .or_else(|| tc.get("top_k"))
        .or_else(|| tc.get("top_k_experts"))
        .and_then(|v| v.as_f64())?;
    let total = tc
        .get("num_local_experts")
        .or_else(|| tc.get("num_experts"))
        .and_then(|v| v.as_f64())?;
    (total > 0.0 && used > 0.0).then(|| (used / total).min(1.0))
}

impl WeightHeader {
    pub fn from_gguf(h: &GgufHeader, file_bytes: u64) -> Result<Self, String> {
        let tensors = h
            .tensors
            .iter()
            .map(|t| {
                let (fmt, _, elems, bytes) =
                    crate::model::UniversalModelLoader::ggml_type_info(t.raw_type)
                        .map_err(|e| format!("{}: {e}", t.name))?;
                let numel: u64 = t.shape.iter().map(|&d| d as u64).product();
                Ok(HeaderTensor {
                    name: t.name.clone(),
                    ndim: t.shape.len(),
                    numel,
                    cols: t.shape.first().map_or(1, |&d| d as u64),
                    fmt: Some(fmt),
                    bytes: numel.div_ceil(elems as u64) * bytes as u64,
                })
            })
            .collect::<Result<_, String>>()?;
        let kv = |suffix: &str| {
            h.metadata
                .iter()
                .find(|(k, _)| k.ends_with(suffix))
                .and_then(|(_, v)| v.parse::<f64>().ok())
        };
        let expert_frac = match (kv(".expert_used_count"), kv(".expert_count")) {
            (Some(used), Some(n)) if n > 0.0 => (used / n).min(1.0),
            _ => 1.0,
        };
        Ok(Self {
            tensors,
            file_bytes,
            expert_frac,
        })
    }

    /// `json`: the safetensors header object (the bytes after the 8-byte length).
    pub fn from_safetensors(json: &[u8], file_bytes: u64) -> Result<Self, String> {
        Self::from_safetensors_with_expert_frac(json, file_bytes, None)
    }

    /// [`Self::from_safetensors`] with an optional MoE active-expert fraction (`expert_frac`).
    pub fn from_safetensors_with_expert_frac(
        json: &[u8],
        file_bytes: u64,
        expert_frac: Option<f64>,
    ) -> Result<Self, String> {
        let obj: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(json).map_err(|e| format!("safetensors header: {e}"))?;
        let mut tensors = Vec::new();
        for (name, t) in obj.iter().filter(|(k, _)| *k != "__metadata__") {
            let dims: Vec<u64> = t["shape"]
                .as_array()
                .ok_or_else(|| format!("{name}: no shape"))?
                .iter()
                .filter_map(|d| d.as_u64())
                .collect();
            let off = t["data_offsets"]
                .as_array()
                .map(|o| (o[0].as_u64(), o[1].as_u64()));
            let Some((Some(a), Some(b))) = off else {
                return Err(format!("{name}: no data_offsets"));
            };
            let fmt = match t["dtype"].as_str().unwrap_or("") {
                "F32" => Some(Q::FP32),
                "F16" => Some(Q::FP16),
                "BF16" => Some(Q::BF16),
                "F8_E4M3" => Some(Q::FP8_E4M3),
                "F8_E5M2" => Some(Q::FP8_E5M2),
                _ => None,
            };
            tensors.push(HeaderTensor {
                name: name.clone(),
                ndim: dims.len(),
                numel: dims.iter().product(),
                cols: dims.last().copied().unwrap_or(1),
                fmt,
                bytes: b.saturating_sub(a),
            });
        }
        Ok(Self {
            tensors,
            file_bytes,
            expert_frac: expert_frac.unwrap_or(1.0),
        })
    }

    /// The matrices one decoded token reads: 2-D tensors and expert banks, minus gathered tables.
    /// A tied embedding is one tensor, so it counts once, as the LM head.
    fn streamed(&self) -> impl Iterator<Item = &HeaderTensor> + '_ {
        let untied = self
            .tensors
            .iter()
            .any(|t| t.name == "output.weight" || t.name.ends_with("lm_head.weight"));
        self.tensors.iter().filter(move |t| {
            (t.ndim == 2 || Self::expert_bank(t)) && !gathered(&t.name, untied)
        })
    }

    /// An MoE expert bank (or one expert's matrix): a token reads `expert_frac` of it.
    fn expert_bank(t: &HeaderTensor) -> bool {
        t.ndim >= 2 && (t.name.contains(".experts.") || t.name.contains("exps"))
    }

    fn scale(&self, t: &HeaderTensor) -> f64 {
        if Self::expert_bank(t) {
            self.expert_frac
        } else {
            1.0
        }
    }

    /// Format and bytes of `t` after `policy` re-encodes it.
    fn after(policy: ReencodePolicy, t: &HeaderTensor) -> (Option<Q>, u64) {
        let Some(src) = t.fmt else {
            return (None, t.bytes);
        };
        if policy == ReencodePolicy::MinSize && t.ndim != 2 {
            return (Some(src), t.bytes);
        }
        let treat_as_lm_head = reencode::is_lm_head_name(&t.name)
            || ["embed", "token_embd", "wte"]
                .iter()
                .any(|s| t.name.contains(s));
        match reencode::target_for_policy(
            policy,
            &t.name,
            src,
            t.cols as usize,
            treat_as_lm_head,
            None,
            Q::BF16,
        ) {
            Some(f) => (Some(f), f.tensor_bytes(t.numel as usize) as u64),
            None => (Some(src), t.bytes),
        }
    }

    /// Bytes one decode step streams from memory under `policy`: the streamed matrices after
    /// re-encoding (MoE expert banks scaled by the used fraction); gathered tables are excluded.
    pub fn streamed_bytes_per_token(&self, policy: ReencodePolicy) -> u64 {
        self.streamed()
            .map(|t| (Self::after(policy, t).1 as f64 * self.scale(t)) as u64)
            .sum()
    }

    /// [`Self::streamed_bytes_per_token`] with every streamed matrix stored as `f` (`--quant`;
    /// rows `f` cannot tile take the next larger format, as the loader does).
    pub fn streamed_bytes_forced(&self, f: Q) -> u64 {
        self.streamed()
            .map(|t| {
                let g = reencode::fit(f, t.cols as usize, Q::FP16);
                (g.tensor_bytes(t.numel as usize) as f64 * self.scale(t)) as u64
            })
            .sum()
    }

    /// Bytes in memory once loaded under `policy` (gathered tables keep their source format).
    pub fn resident_bytes(&self, policy: ReencodePolicy) -> u64 {
        let streamed: std::collections::HashSet<*const HeaderTensor> =
            self.streamed().map(|t| t as *const _).collect();
        self.tensors
            .iter()
            .map(|t| {
                if streamed.contains(&(t as *const _)) {
                    Self::after(policy, t).1
                } else {
                    t.bytes
                }
            })
            .sum()
    }

    /// Mean bits per weight of the streamed matrices in their source formats: what the checkpoint
    /// carries, which re-encoding cannot improve.
    pub fn mean_bpw(&self) -> f64 {
        let (bits, n) = self.streamed().fold((0.0, 0.0), |(b, n), t| {
            (b + t.bytes as f64 * 8.0, n + t.numel as f64)
        });
        if n > 0.0 {
            bits / n
        } else {
            0.0
        }
    }
}

/// Predicted decode tokens/s: `1 / Σ_t bytes_t / (eff(fmt_t) · ceiling)` over the streamed tensors
/// after `policy`; `eff` maps a format to GB/s over the ceiling.
pub fn predict_with(
    h: &WeightHeader,
    policy: ReencodePolicy,
    ceiling_gbps: f64,
    eff: &dyn Fn(Q) -> f64,
) -> f64 {
    let secs: f64 = h
        .streamed()
        .map(|t| {
            let (f, bytes) = WeightHeader::after(policy, t);
            bytes as f64 * h.scale(t) / (f.map_or(DEFAULT_EFF, eff) * ceiling_gbps * 1e9)
        })
        .sum();
    if secs > 0.0 {
        1.0 / secs
    } else {
        0.0
    }
}

/// [`predict_with`] at this host's measured ceiling and codec efficiencies (the ceiling is clamped
/// to 1 GB/s before any profile exists, which keeps the ordering and not the magnitude).
pub fn predict_decode_tok_s(h: &WeightHeader, policy: ReencodePolicy) -> f64 {
    predict_with(h, policy, host_ceiling_gbps().max(1.0), &host_eff)
}

/// One selectable variant with its predicted speed.
#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub name: String,
    pub download_bytes: u64,
    pub resident_bytes: u64,
    pub tok_s: f64,
    /// Streamed mean bits per weight of the source.
    pub bpw: f64,
}

impl Variant {
    pub fn from_header(
        name: &str,
        download_bytes: u64,
        h: &WeightHeader,
        policy: ReencodePolicy,
        ceiling_gbps: f64,
    ) -> Self {
        Self {
            name: name.into(),
            download_bytes,
            resident_bytes: h.resident_bytes(policy),
            tok_s: predict_with(h, policy, ceiling_gbps, &host_eff),
            bpw: h.mean_bpw(),
        }
    }

    /// The fallback when no header could be read: the whole file as one matrix in `fmt`
    /// (size × bpw), sized at [`NOMINAL_PARAMS`] when the size is unknown.
    pub fn from_size(
        name: &str,
        fmt: Q,
        bytes: Option<u64>,
        policy: ReencodePolicy,
        ceiling_gbps: f64,
    ) -> Self {
        let h = size_header(fmt, bytes);
        Self::from_header(name, h.file_bytes, &h, policy, ceiling_gbps)
    }
}

/// A header of one `fmt` matrix of `bytes` (sized at [`NOMINAL_PARAMS`] when unknown): what
/// [`Variant::from_size`] predicts from.
pub fn size_header(fmt: Q, bytes: Option<u64>) -> WeightHeader {
    let numel = bytes.map_or(NOMINAL_PARAMS, |b| {
        (b as f64 * 8.0 / fmt.bits_per_weight() as f64) as u64
    });
    WeightHeader {
        tensors: vec![HeaderTensor {
            name: "blk.0.ffn_up.weight".into(),
            ndim: 2,
            numel,
            cols: 4096,
            fmt: Some(fmt),
            bytes: bytes.unwrap_or_else(|| fmt.tensor_bytes(numel as usize) as u64),
        }],
        file_bytes: bytes.unwrap_or(0),
        expert_frac: 1.0,
    }
}

/// Orders `v` best first under `policy` and returns how many are eligible (the first that many are
/// the choices; with none, the order is by predicted speed). Eligible: resident bytes at most 0.7 × `ram`,
/// and for `Fastest4Bit` a streamed mean bpw of at least 4.0. `Fastest` and `Fastest4Bit` rank by
/// predicted tok/s, `BestQuality` by bpw; ties go to fewer download bytes, then filename.
pub fn rank_variants(policy: VariantPolicy, ram: u64, v: &mut [Variant]) -> usize {
    let eligible = |x: &Variant| {
        x.resident_bytes as f64 <= RAM_FRACTION * ram as f64
            && (policy != VariantPolicy::Fastest4Bit || x.bpw >= 4.0)
    };
    v.sort_by(|a, b| {
        let primary = |x: &Variant| {
            if policy == VariantPolicy::BestQuality && eligible(x) {
                x.bpw
            } else {
                x.tok_s
            }
        };
        eligible(b)
            .cmp(&eligible(a))
            .then(primary(b).total_cmp(&primary(a)))
            .then(a.download_bytes.cmp(&b.download_bytes))
            .then(a.name.cmp(&b.name))
    });
    v.iter().filter(|x| eligible(x)).count()
}

/// Sort key for a default download plan (ascending = preferred) from a file's format and size alone:
/// predicted speed in 5% steps, then smaller bytes first. A size under 1 MiB is a placeholder, not a
/// checkpoint, and counts as unknown.
pub fn hub_rank(
    fmt: Q,
    bytes: Option<u64>,
    reencode: ReencodePolicy,
    variant: VariantPolicy,
) -> f64 {
    let bytes = bytes.filter(|&b| b >= 1 << 20);
    let v = Variant::from_size("", fmt, bytes, reencode, 1.0);
    let step = (v.tok_s.ln() / 1.05f64.ln()).round();
    let base = -step
        + bytes.unwrap_or_else(|| fmt.tensor_bytes(NOMINAL_PARAMS as usize) as u64) as f64 / 1e13;
    match variant {
        VariantPolicy::Fastest => base,
        VariantPolicy::Fastest4Bit => base + if v.bpw < 4.0 { 1e6 } else { 0.0 },
        VariantPolicy::BestQuality => -v.bpw * 1e6 + base,
    }
}

#[cfg(not(test))]
static POLICY_OVERRIDE: RwLock<Option<VariantPolicy>> = RwLock::new(None);

#[cfg(test)]
thread_local! {
    static TEST_POLICY_OVERRIDE: std::cell::Cell<Option<VariantPolicy>> = const { std::cell::Cell::new(None) };
}

/// Pins the variant policy (`--variant`) over the settings file; `None` returns to it.
pub fn set_policy_override(p: Option<VariantPolicy>) {
    #[cfg(test)]
    TEST_POLICY_OVERRIDE.with(|t| t.set(p));
    #[cfg(not(test))]
    {
        *POLICY_OVERRIDE.write() = p;
    }
}

/// The `--variant` override, else `variant` of `~/.cache/gibbs/settings.json`.
pub fn variant_policy() -> VariantPolicy {
    #[cfg(test)]
    if let Some(p) = TEST_POLICY_OVERRIDE.with(|t| t.get()) {
        return p;
    }
    #[cfg(not(test))]
    if let Some(p) = *POLICY_OVERRIDE.read() {
        return p;
    }
    crate::tui::settings::load_settings().variant
}

/// The `--reencode` override, else `reencode` of `~/.cache/gibbs/settings.json`.
pub fn reencode_policy() -> ReencodePolicy {
    crate::model::weight_format::ReencodeConfig::effective().policy
}

const FIRST_FETCH: u64 = 16 << 20;
const SECOND_FETCH: u64 = 64 << 20;
const MAX_JSON: u64 = 256 << 20;

/// Reads the header of weights file `name` through `read(start, len)` (which may return fewer bytes
/// at the end of the file): a GGUF from its first 16 MiB, retried with 64 MiB on `NeedMore`;
/// a safetensors from its 8-byte length, then the JSON header.
pub fn read_header(
    name: &str,
    file_bytes: u64,
    mut read: impl FnMut(u64, u64) -> Result<Vec<u8>, String>,
) -> Result<WeightHeader, String> {
    if name.ends_with(".safetensors") {
        let len = read(0, 8)?;
        let n = u64::from_le_bytes(
            len.as_slice()
                .try_into()
                .map_err(|_| "safetensors: short length prefix")?,
        );
        if n > MAX_JSON {
            return Err(format!("safetensors header of {n} bytes"));
        }
        return WeightHeader::from_safetensors(&read(8, n)?, file_bytes);
    }
    for len in [FIRST_FETCH, SECOND_FETCH] {
        match parse_gguf_header(&read(0, len)?) {
            Ok(h) => return WeightHeader::from_gguf(&h, file_bytes),
            Err(HeaderError::NeedMore(_)) => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    Err(format!("GGUF header exceeds {} MiB", SECOND_FETCH >> 20))
}

/// Bytes one decode step streams for the checkpoint at `path` (matrices after `policy`; gathered
/// tables excluded), the roofline numerator. The file size when its header cannot be read.
pub fn decode_bytes_per_token(path: &Path, policy: ReencodePolicy) -> u64 {
    read_local_header(path)
        .ok()
        .map(|h| h.streamed_bytes_per_token(policy))
        .filter(|&b| b > 0)
        .or_else(|| std::fs::metadata(path).ok().map(|m| m.len()))
        .unwrap_or(0)
}

/// The header of a local weights file.
pub fn read_local_header(path: &Path) -> Result<WeightHeader, String> {
    let size = std::fs::metadata(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    let mut h = read_header(&path.to_string_lossy(), size, |start, len| {
        let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
        f.seek(SeekFrom::Start(start)).map_err(|e| e.to_string())?;
        let mut buf = Vec::new();
        f.take(len)
            .read_to_end(&mut buf)
            .map_err(|e| e.to_string())?;
        Ok(buf)
    })?;
    if path.extension().is_some_and(|e| e == "safetensors") && h.expert_frac == 1.0 {
        if let Some(ef) = path
            .parent()
            .and_then(|d| std::fs::read(d.join("config.json")).ok())
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|cfg| expert_frac_from_config(&cfg))
        {
            h.expert_frac = ef;
        }
    }
    Ok(h)
}

fn curl(args: &[String], url: &str, token: Option<&str>) -> Result<Vec<u8>, String> {
    let mut cmd = std::process::Command::new("curl");
    cmd.args(["-fsSL", "--connect-timeout", "15", "-m", "120"])
        .args(args);
    if let Some(t) = token.filter(|t| !t.is_empty()) {
        cmd.arg("-H").arg(format!("Authorization: Bearer {t}"));
    }
    let out = cmd
        .arg(url)
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("Failed to execute curl: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "{url}: {}",
            stderr.lines().next().unwrap_or("curl failed").trim()
        ));
    }
    Ok(out.stdout)
}

/// `(etag, size)` from a HEAD request (redirects followed): `x-linked-etag` when the Hub sends it.
fn head(url: &str, token: Option<&str>) -> Result<(String, u64), String> {
    let text = String::from_utf8_lossy(&curl(&["-I".into()], url, token)?).to_lowercase();
    let last = |name: &str| {
        text.lines()
            .filter_map(|l| l.strip_prefix(name)?.strip_prefix(':'))
            .last()
            .map(|v| v.trim().trim_matches('"').to_string())
    };
    let etag = last("x-linked-etag")
        .or_else(|| last("etag"))
        .ok_or("no etag")?;
    let size = last("content-length")
        .and_then(|s| s.parse().ok())
        .ok_or("no content-length")?;
    Ok((etag, size))
}

fn fnv1a(parts: &[&str]) -> u64 {
    parts
        .iter()
        .flat_map(|p| p.bytes().chain([0xff]))
        .fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x100000001b3)
        })
}

/// The header of `file` in Hub repo `repo_id` (`org/repo[@rev]`): cached under
/// `~/.cache/gibbs/hub-headers/` by `(repo, file, etag)`, else range requests. A safetensors
/// file's `expert_frac` comes from the `config.json` beside it, as for a local file.
pub fn fetch_remote_header(
    repo_id: &str,
    file: &str,
    token: Option<&str>,
) -> Result<WeightHeader, String> {
    let (repo, rev) = crate::model::hub::split_revision(repo_id);
    let url = format!(
        "https://huggingface.co/{repo}/resolve/{}/{file}",
        crate::model::hub::rev_segment(rev)
    );
    let (etag, size) = head(&url, token)?;
    let cache = default_cache_dir().ok().map(|d| {
        // `expert_frac`: headers cached before the config was read carry 1.0.
        d.join("hub-headers")
            .join(format!("{:016x}.json", fnv1a(&[repo, file, &etag, "expert_frac"])))
    });
    if let Some(h) = cache
        .as_ref()
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice(&b).ok())
    {
        return Ok(h);
    }
    let mut h = read_header(file, size, |start, len| {
        curl(
            &["-r".into(), format!("{start}-{}", start + len - 1)],
            &url,
            token,
        )
    })?;
    if file.ends_with(".safetensors") && h.expert_frac == 1.0 {
        // The file's own directory in the repo, as `read_local_header` reads it.
        let cfg_url = format!("{}/config.json", url.rsplit_once('/').map_or("", |(d, _)| d));
        if let Some(ef) = curl(&[], &cfg_url, token)
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|cfg| expert_frac_from_config(&cfg))
        {
            h.expert_frac = ef;
        }
    }
    if let Some(p) = cache {
        let tmp = p.with_extension("tmp");
        let saved = p
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|_| std::fs::write(&tmp, serde_json::to_vec(&h)?))
            .and_then(|_| std::fs::rename(&tmp, &p));
        if saved.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(name: &str, shape: [u64; 2], fmt: Q) -> HeaderTensor {
        let numel = shape[0] * shape[1];
        HeaderTensor {
            name: name.into(),
            ndim: 2,
            numel,
            cols: shape[0],
            fmt: Some(fmt),
            bytes: fmt.tensor_bytes(numel as usize) as u64,
        }
    }

    fn header(tensors: Vec<HeaderTensor>) -> WeightHeader {
        WeightHeader {
            file_bytes: tensors.iter().map(|t| t.bytes).sum(),
            tensors,
            expert_frac: 1.0,
        }
    }

    #[test]
    fn higher_eff_predicts_higher_tok_s() {
        let h = header(vec![t("blk.0.ffn_up.weight", [4096, 4096], Q::GGUF_Q4_0)]);
        let at = |e: f64| predict_with(&h, ReencodePolicy::BitExact, 300.0, &|_| e);
        assert!(at(0.95) > at(0.60));
        // 9.4 MB at 300 GB/s · 0.9
        let want = 1.0 / (h.tensors[0].bytes as f64 / (0.9 * 300e9));
        assert!((at(0.9) - want).abs() / want < 1e-12);
    }

    #[test]
    fn tied_embedding_counts_once_and_gathered_tables_not_at_all() {
        let ffn = t("blk.0.ffn_up.weight", [256, 256], Q::GGUF_Q8_0);
        let emb = t("token_embd.weight", [256, 1024], Q::GGUF_Q8_0);
        let ple = t("per_layer_token_embd.weight", [256, 1024], Q::GGUF_Q8_0);
        let tied = header(vec![ffn.clone(), emb.clone(), ple.clone()]);
        let untied = header(vec![
            ffn.clone(),
            emb.clone(),
            ple,
            t("output.weight", [256, 1024], Q::GGUF_Q8_0),
        ]);
        let tok = |h: &WeightHeader| predict_with(h, ReencodePolicy::BitExact, 100.0, &|_| 1.0);
        let secs = |bytes: u64| bytes as f64 / 100e9;
        assert!((1.0 / tok(&tied) - secs(ffn.bytes + emb.bytes)).abs() < 1e-15);
        // Untied: the embedding is gathered (not streamed); only the head streams.
        assert!((1.0 / tok(&untied) - secs(ffn.bytes + emb.bytes)).abs() < 1e-15);
        assert_eq!(tied.streamed().count(), 2);
    }

    #[test]
    fn policy_reencodes_before_predicting() {
        let h = header(vec![t("blk.0.ffn_up.weight", [4096, 4096], Q::BF16)]);
        let tok = |p| predict_with(&h, p, 100.0, &|_| 1.0);
        // BF16 streams 16 bits exact, 4.5 bits as Q4_K.
        assert!(
            (tok(ReencodePolicy::MaxSpeed) / tok(ReencodePolicy::BitExact) - 16.0 / 4.5).abs()
                < 1e-9
        );
        // LowBitNative: Q8_0.
        assert!(
            (tok(ReencodePolicy::LowBitNative) / tok(ReencodePolicy::BitExact) - 16.0 / 8.5).abs()
                < 1e-9
        );
        assert!(
            h.resident_bytes(ReencodePolicy::MaxSpeed) < h.resident_bytes(ReencodePolicy::BitExact)
        );
        // MinSize: FFN -> Q3_K (110 B / 256 = 3.4375 bpw).
        assert!(
            (tok(ReencodePolicy::MinSize) / tok(ReencodePolicy::BitExact) - 16.0 / 3.4375).abs()
                < 1e-9
        );
        assert!(
            h.resident_bytes(ReencodePolicy::MinSize) < h.resident_bytes(ReencodePolicy::MaxSpeed)
        );
    }

    fn v(name: &str, dl: u64, tok_s: f64, bpw: f64, resident: u64) -> Variant {
        Variant {
            name: name.into(),
            download_bytes: dl,
            resident_bytes: resident,
            tok_s,
            bpw,
        }
    }

    #[test]
    fn variant_policies() {
        let ram = 100;
        let all = || {
            vec![
                v("a-q2", 30, 200.0, 2.6, 30),
                v("b-q4", 50, 150.0, 4.5, 50),
                v("c-q8", 90, 90.0, 8.5, 69),
                v("d-bf16", 170, 40.0, 16.0, 71), // 71 > 0.7 × 100: does not fit
            ]
        };
        let first = |p| {
            let mut x = all();
            let n = rank_variants(p, ram, &mut x);
            (x[0].name.clone(), n)
        };
        assert_eq!(first(VariantPolicy::Fastest), ("a-q2".into(), 3));
        assert_eq!(first(VariantPolicy::Fastest4Bit), ("b-q4".into(), 2));
        assert_eq!(first(VariantPolicy::BestQuality), ("c-q8".into(), 3));
        // Nothing fits: no eligible variants, ordered by speed.
        let mut x = all();
        assert_eq!(rank_variants(VariantPolicy::Fastest, 10, &mut x), 0);
        assert_eq!(x[0].name, "a-q2");
    }

    #[test]
    fn ties_prefer_fewer_download_bytes_then_filename() {
        let mut x = vec![
            v("z", 10, 100.0, 4.5, 1),
            v("m", 20, 100.0, 4.5, 1),
            v("b", 10, 100.0, 4.5, 1),
        ];
        rank_variants(VariantPolicy::Fastest, 1000, &mut x);
        assert_eq!(
            x.iter().map(|v| v.name.as_str()).collect::<Vec<_>>(),
            ["b", "z", "m"]
        );
        let mut x = vec![v("z", 10, 1.0, 6.0, 1), v("m", 20, 2.0, 6.0, 1)];
        rank_variants(VariantPolicy::BestQuality, 1000, &mut x);
        assert_eq!(x[0].name, "z");
    }

    #[test]
    fn size_fallback_orders_by_format_and_bytes() {
        with_unmeasured_host(|| {
            let p = ReencodePolicy::MaxSpeed;
            let vp = VariantPolicy::Fastest;
            let rank = |f, b| hub_rank(f, b, p, vp);
            assert!(rank(Q::GGUF_Q2_K, None) < rank(Q::GGUF_Q4_K_M, None));
            // Q8_0 re-encodes to Q4_K under max-speed: a tie broken by download size.
            // 8e9 params: Q4_K_M 4.5e9 B, Q8_0 8.5e9 B (4.5e9 B once re-encoded).
            assert!(
                rank(Q::GGUF_Q4_K_M, Some(4_500_000_000)) < rank(Q::GGUF_Q8_0, Some(8_500_000_000))
            );
            assert!(
                hub_rank(Q::GGUF_Q4_K_M, None, p, VariantPolicy::Fastest4Bit)
                    < hub_rank(Q::GGUF_Q2_K, None, p, VariantPolicy::Fastest4Bit)
            );
            assert!(
                hub_rank(Q::GGUF_Q8_0, None, p, VariantPolicy::BestQuality)
                    < hub_rank(Q::GGUF_Q4_K_M, None, p, VariantPolicy::BestQuality)
            );
            let a = Variant::from_size("a", Q::GGUF_Q4_0, Some(1 << 30), p, 300.0);
            assert!(a.tok_s > 0.0 && a.resident_bytes <= 1 << 30);
        })
    }

    #[test]
    fn safetensors_header() {
        let json = br#"{"__metadata__":{"format":"pt"},
            "model.embed_tokens.weight":{"dtype":"BF16","shape":[10,8],"data_offsets":[0,160]},
            "model.layers.0.mlp.up_proj.weight":{"dtype":"BF16","shape":[16,8],"data_offsets":[160,416]},
            "model.layers.0.norm.weight":{"dtype":"BF16","shape":[8],"data_offsets":[416,432]}}"#;
        let h = WeightHeader::from_safetensors(json, 432).unwrap();
        assert_eq!(h.tensors.len(), 3);
        // Tied (no lm_head): embedding + up_proj stream.
        assert_eq!(h.streamed().count(), 2);
        assert_eq!(h.mean_bpw(), 16.0);
        let mut f = Vec::new();
        let out = read_header("m.safetensors", 432, |s, l| {
            f.push((s, l));
            let body: Vec<u8> = json.to_vec();
            Ok(if s == 0 {
                (body.len() as u64).to_le_bytes().to_vec()
            } else {
                body
            })
        })
        .unwrap();
        assert_eq!(f, [(0, 8), (8, json.len() as u64)]);
        assert_eq!(out.tensors.len(), 3);
    }

    #[test]
    fn real_gguf_header_round_trips_and_truncation_asks_for_more() {
        let path = Path::new("disk/models--unsloth--gemma-4-E2B-it-qat-GGUF/snapshots/main/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf");
        if !path.exists() {
            println!("SKIP: {} not found", path.display());
            return;
        }
        let mut head = vec![0u8; 16 << 20];
        std::fs::File::open(path)
            .unwrap()
            .read_exact(&mut head)
            .unwrap();
        let h = parse_gguf_header(&head).unwrap();
        // Against the file reader (the full-file parser).
        let r = crate::model::loaders::gguf::GgufWeightReader::open(path).unwrap();
        assert_eq!(h.tensors.len(), r.tensor_names().count());
        for t in &h.tensors {
            assert_eq!(r.get_tensor_shape(&t.name).unwrap(), t.shape.as_slice());
            assert_eq!(r.get_tensor_byte_offset(&t.name), Some(t.byte_offset));
        }
        // Every byte of the header is needed: one short asks for exactly the header end.
        let need = h.data_start;
        let mut cut = head.len();
        while cut > need {
            cut /= 2;
        }
        let Err(HeaderError::NeedMore(n)) = parse_gguf_header(&head[..cut]) else {
            panic!("truncation at {cut} must ask for more");
        };
        assert!(n > cut && n <= need, "{n} not in ({cut}, {need}]");
        // The retry path: first fetch too small, second whole.
        let mut calls = Vec::new();
        let wh = read_header(
            "x.gguf",
            std::fs::metadata(path).unwrap().len(),
            |_, len| {
                calls.push(len);
                Ok(head[..(len as usize).min(if calls.len() == 1 {
                    1 << 20
                } else {
                    head.len()
                })]
                    .to_vec())
            },
        )
        .unwrap();
        assert_eq!(calls, [FIRST_FETCH, SECOND_FETCH]);
        assert_eq!(wh.tensors.len(), h.tensors.len());
        let tok = predict_with(&wh, ReencodePolicy::BitExact, 300.0, &|_| 0.9);
        println!(
            "E2B Q2_K_XL: {} tensors, mean {:.2} bpw, predicted {tok:.0} tok/s at 300 GB/s",
            wh.tensors.len(),
            wh.mean_bpw()
        );
        assert!(tok > 100.0 && tok < 5000.0);
    }

    #[test]
    fn safetensors_moe_and_mtp_streamed_bytes() {
        let json = br#"{"__metadata__":{"format":"pt"},
            "model.embed_tokens.weight":{"dtype":"BF16","shape":[10,256],"data_offsets":[0,5120]},
            "model.layers.0.mlp.experts.0.up_proj.weight":{"dtype":"BF16","shape":[128,256],"data_offsets":[5120,70656]},
            "mtp.layers.0.mlp.up_proj.weight":{"dtype":"BF16","shape":[128,256],"data_offsets":[70656,136192]}}"#;
        let cfg = serde_json::json!({"num_experts_per_tok": 2, "num_experts": 8});
        let ef = expert_frac_from_config(&cfg);
        assert_eq!(ef, Some(0.25));
        let h = WeightHeader::from_safetensors_with_expert_frac(json, 136192, ef).unwrap();
        // mtp.* is excluded from base-model streamed bytes; 2D .experts. scales by 0.25.
        let names: Vec<&str> = h.streamed().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "model.embed_tokens.weight",
                "model.layers.0.mlp.experts.0.up_proj.weight"
            ]
        );
        assert_eq!(
            h.streamed_bytes_per_token(ReencodePolicy::BitExact),
            5120 + (65536 / 4)
        );
    }
}
