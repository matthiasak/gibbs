//! HOPE (Hilbert Operator for Progressive Encoding, Mobahi & Bartlett, arXiv:2607.21366):
//! data-free on-the-fly functional weight compression via closed-form Gaussian Hilbert-Schmidt
//! operators, SimHash candidate pairing, Dantzig receding-horizon knapsack selection, neuron
//! merging/pruning, and macro residual-block eviction.

use crate::graph::{
    BinaryKind, DType, Graph, NodeId, Op, Tensor, UnaryKind, ValueId, ValueKind, Weights,
};
use crate::hardware::QuantFormat;
use crate::model::layers::{encode_rows, gguf_block_codec};
use crate::model::par::par_rows;
use crate::model::reencode::{encodable, is_float, smallest_ladder_ge};
use crate::model::weight_format::HopePolicy;
use crate::model::weights::Mat;
use crate::model::{f32_to_bf16, f32_to_f16};
use crate::runtime::noise_draft::fwht_in_place;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};

const SIMHASH_DIM: usize = 64;
const SIMHASH_TABLES: usize = 4;
const SIMHASH_BITS: usize = 14;
const SIMHASH_CANDIDATES: usize = 16;
const MICRO_BATCH: usize = 32;
const SILU_LAMBDA: f64 = 1.203;

static LAST_SAVED_PCT_X10: AtomicU32 = AtomicU32::new(0);
static LAST_BYTE_RATIO_X10000: AtomicU32 = AtomicU32::new(10_000);

/// Most recent measured parameter reduction percentage (falls back to policy target before first run).
pub fn last_saved_pct(policy: HopePolicy) -> f32 {
    let raw = LAST_SAVED_PCT_X10.load(Ordering::Relaxed);
    if raw > 0 {
        raw as f32 / 10.0
    } else {
        (1.0 - policy.density()) * 100.0
    }
}

/// Ratio `bytes_after / bytes_before` of the most recent active HOPE run (`1.0` when `Off`).
pub fn last_byte_ratio() -> f64 {
    (LAST_BYTE_RATIO_X10000.load(Ordering::Relaxed) as f64 / 10_000.0).clamp(0.1, 1.0)
}

fn weights_resident_bytes(weights: &Weights) -> usize {
    weights
        .iter()
        .map(|(_, t)| match t {
            Tensor::F32(v) => v.len() * 4,
            Tensor::U32(v) => v.len() * 4,
            Tensor::F16(v) | Tensor::BF16(v) => v.len() * 2,
            Tensor::Mat(m) => match m.packed() {
                Some(p) => m.rows * p.format.tensor_bytes(m.cols),
                None => m.rows * m.cols * 2,
            },
        })
        .sum()
}

/// Replayable structural compression recipe applied to reshaped prefill/verify instances.
#[derive(Debug, Clone, Default)]
pub struct HopeRecipe {
    pub resized_weights: HashMap<String, (Vec<usize>, DType)>,
    pub evicted_blocks: HashSet<String>,
    pub params_before: usize,
    pub params_after: usize,
    pub pruned_neurons: usize,
    pub merged_neurons: usize,
    pub evicted_block_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActKind {
    Relu,
    Gelu,
    Silu,
}

impl ActKind {
    fn from_unary(u: &UnaryKind) -> Option<Self> {
        match u {
            UnaryKind::Relu => Some(Self::Relu),
            UnaryKind::GeluErf | UnaryKind::GeluTanh => Some(Self::Gelu),
            UnaryKind::Silu => Some(Self::Silu),
            _ => None,
        }
    }

    fn lambda(self) -> Option<f64> {
        match self {
            Self::Relu => None,
            Self::Gelu => Some(1.0),
            Self::Silu => Some(SILU_LAMBDA),
        }
    }
}

#[derive(Debug, Clone)]
enum GroupKind {
    Plain {
        act: ActKind,
        w_in: String,
        b_in: Option<String>,
        w_out: String,
        in_spatial: usize,
        out_spatial: usize,
    },
    Glu {
        act: ActKind,
        w_gate: String,
        b_gate: Option<String>,
        w_up: String,
        b_up: Option<String>,
        sub_norm: Option<String>,
        w_down: String,
    },
}

#[derive(Debug, Clone)]
struct NeuronGroup {
    first_node: usize,
    last_node: usize,
    n: usize,
    d_in: usize,
    d_out: usize,
    align: usize,
    norm_g: Vec<f32>,
    norm_b: Vec<f32>,
    kind: GroupKind,
}

impl NeuronGroup {
    fn params_per_neuron(&self) -> usize {
        match &self.kind {
            GroupKind::Plain {
                b_in,
                in_spatial,
                out_spatial,
                ..
            } => self.d_in * *in_spatial + usize::from(b_in.is_some()) + self.d_out * *out_spatial,
            GroupKind::Glu {
                b_gate,
                b_up,
                sub_norm,
                ..
            } => {
                2 * self.d_in
                    + self.d_out
                    + usize::from(b_gate.is_some())
                    + usize::from(b_up.is_some())
                    + usize::from(sub_norm.is_some())
            }
        }
    }

    fn total_params(&self) -> usize {
        self.n * self.params_per_neuron()
    }
}

#[derive(Debug, Clone)]
struct ResidualBlock {
    add_name: String,
    sig: String,
    group_indices: Vec<usize>,
    extra_energy: f64,
    identity_energy: f64,
    total_params: usize,
}

#[derive(Debug, Clone, Copy)]
struct MergeCand {
    i: u32,
    j: u32,
    c_ker: f32,
    a: f32,
    b: f32,
}

#[derive(Debug, Clone)]
struct GroupProfile {
    caps: Vec<f32>,
    merges: Vec<MergeCand>,
    alive: Vec<bool>,
    merged_into: Vec<Option<(usize, f32, f32)>>, // survivor i -> (partner j, c_ker, target_norm)
    used_in_merge: Vec<bool>,
    n_live: usize,
    e_live: f64,
    evicted: bool,
}

/// Standard normal density $p(z)$.
#[inline]
fn norm_pdf(z: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.3989422804014327;
    INV_SQRT_2PI * (-0.5 * z * z).exp()
}

/// Standard normal cumulative distribution $F(z)$ (Abramowitz & Stegun 7.1.26, max error < 1.5e-7).
#[inline]
fn norm_cdf(z: f64) -> f64 {
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let x = z.abs() * std::f64::consts::FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = sign * (1.0 - poly * (-x * x).exp());
    0.5 * (1.0 + erf)
}

/// Self-kernel $K(i, i) = \mathbb{E}_{x \sim \mathcal{N}(0, I)}[\sigma(w^\top x + \beta)^2]$.
fn self_kernel_scalar(act: ActKind, gamma: f64, beta: f64) -> f64 {
    let g = gamma.max(1e-12);
    let ratio = (beta / g).clamp(-12.0, 12.0);
    match act.lambda() {
        None => {
            // ReLU (Eq. 79): exact 1-homogeneous Gaussian expectation.
            ((gamma * gamma + beta * beta) * norm_cdf(ratio) + beta * g * norm_pdf(ratio))
                .max(1e-24)
        }
        Some(lam) => {
            // Smooth gate (GELU / SiLU): closed-form bivariate Gaussian expectation.
            let s = lam * gamma;
            let s2 = s * s;
            let one_p_s2 = 1.0 + s2;
            let one_p_2s2 = 1.0 + 2.0 * s2;
            let frac = (s2 / one_p_s2).clamp(-1.0, 1.0);
            let k_zero = gamma
                * gamma
                * (0.25
                    + frac.asin() / (2.0 * std::f64::consts::PI)
                    + (s2 * one_p_2s2) / (2.0 * std::f64::consts::PI * one_p_s2 * one_p_2s2));
            (k_zero + beta * beta * norm_cdf(ratio)).max(1e-24)
        }
    }
}

/// Normalized Hilbert feature correlation $\langle \psi_i, \psi_j \rangle_{\mathcal{H}} \in [-1, 1]$.
#[inline]
fn cho_saul_j1(rho: f64) -> f64 {
    let r = rho.clamp(-1.0, 1.0);
    let s = (1.0 - r * r).max(0.0).sqrt();
    ((s + (std::f64::consts::PI - r.acos()) * r) / std::f64::consts::PI).clamp(-1.0, 1.0)
}

fn plain_eff_stats(w: &[f32], b: f32, g: &[f32], beta0: &[f32], out_dir: &mut [f32]) -> (f64, f64) {
    let mut g2 = 0.0f64;
    let mut b_acc = b as f64;
    for k in 0..w.len() {
        let wk = w[k] as f64;
        let eff = wk * (g[k] as f64);
        out_dir[k] = eff as f32;
        g2 += eff * eff;
        b_acc += wk * (beta0[k] as f64);
    }
    let gamma = (g2 + 1e-24).sqrt();
    let norm = (g2 + b_acc * b_acc + 1e-24).sqrt() as f32;
    let inv = 1.0 / norm;
    for v in out_dir.iter_mut().take(w.len()) {
        *v *= inv;
    }
    if out_dir.len() > w.len() {
        out_dir[w.len()] = (b_acc as f32) * inv;
    }
    (gamma, b_acc)
}

fn glu_eff_stats(
    act: ActKind,
    wg: &[f32],
    bg: f32,
    wu: &[f32],
    bu: f32,
    g: &[f32],
    beta0: &[f32],
    ug_out: &mut [f32],
    uu_out: &mut [f32],
) -> (f64, f64, f64, f64, f64) {
    let mut gg2 = 0.0f64;
    let mut gu2 = 0.0f64;
    let mut bg_acc = bg as f64;
    let mut bu_acc = bu as f64;
    for k in 0..wg.len() {
        let wgk = wg[k] as f64;
        let wuk = wu[k] as f64;
        let gk = g[k] as f64;
        let b0k = beta0[k] as f64;
        let eg = wgk * gk;
        let eu = wuk * gk;
        ug_out[k] = eg as f32;
        uu_out[k] = eu as f32;
        gg2 += eg * eg;
        gu2 += eu * eu;
        bg_acc += wgk * b0k;
        bu_acc += wuk * b0k;
    }
    let gamma_g = (gg2 + 1e-24).sqrt();
    let gamma_u = (gu2 + bu_acc * bu_acc + 1e-24).sqrt();
    let inv_g = (1.0 / gamma_g) as f32;
    let inv_u = (1.0 / gamma_u) as f32;
    let mut dot_gu = 0.0f64;
    for k in 0..wg.len() {
        ug_out[k] *= inv_g;
        uu_out[k] *= inv_u;
        dot_gu += (ug_out[k] as f64) * (uu_out[k] as f64);
    }
    let k_gate = self_kernel_scalar(act, gamma_g, bg_acc);
    let k_ii = (gamma_u * gamma_u * k_gate * (1.0 + 2.0 * dot_gu * dot_gu)).max(1e-24);
    (gamma_g, bg_acc, gamma_u, dot_gu, k_ii)
}

/// Closed-form 2×2 Gram SVD statistics $(a_{ij}, b_{ij})$ for pair $(i, j)$.
/// `a = ||f_i + f_j||_H^2`, `b = sqrt(2) * sigma_1` so Dantzig shrinkage `s*` yields exact 0 at `c = 1`.
#[inline]
fn pair_gram_ab(ni2: f64, nj2: f64, w_dot: f64, c_ker: f64) -> (f32, f32, f64, f64) {
    let c = c_ker.clamp(-1.0, 1.0);
    let a = (ni2 + nj2 + 2.0 * c * w_dot).max(0.0);
    let cross = (ni2 * nj2 - w_dot * w_dot).max(0.0);
    let det = (1.0 - c * c).max(0.0) * cross;
    let disc = (a * a - 4.0 * det).max(0.0).sqrt();
    let sigma1_sq = 0.5 * (a + disc);
    let sigma1 = sigma1_sq.max(0.0).sqrt();
    let b = std::f64::consts::SQRT_2 * sigma1;

    // Principal right eigenvector of M = G_ker * G_w:
    // M = [[ni2 + c*w_dot, w_dot + c*nj2], [c*ni2 + w_dot, c*w_dot + nj2]]
    let m00 = ni2 + c * w_dot;
    let m01 = w_dot + c * nj2;
    let m10 = c * ni2 + w_dot;
    let m11 = c * w_dot + nj2;
    let (mut b1, mut b2) =
        if (m00 - sigma1_sq).abs() + m01.abs() >= (m11 - sigma1_sq).abs() + m10.abs() {
            if m01.abs() > 1e-18 {
                (m01, sigma1_sq - m00)
            } else {
                (1.0, 0.0)
            }
        } else if m10.abs() > 1e-18 {
            (sigma1_sq - m11, m10)
        } else {
            (0.0, 1.0)
        };
    if (b1 * b1 + b2 * b2) < 1e-30 {
        b1 = ni2.sqrt();
        b2 = nj2.sqrt();
    }
    // Solve G_ker * alpha = beta => [[1, c], [c, 1]] alpha = beta.
    let (mut a1, mut a2) = if (1.0 - c * c) > 1e-8 {
        let inv = 1.0 / (1.0 - c * c);
        ((b1 - c * b2) * inv, (b2 - c * b1) * inv)
    } else {
        let s1 = ni2.sqrt();
        let s2 = nj2.sqrt();
        let denom = (s1 + s2).max(1e-12);
        (s1 / denom, s2 / denom)
    };
    let norm_ker = (a1 * a1 + a2 * a2 + 2.0 * c * a1 * a2).max(1e-24).sqrt();
    a1 /= norm_ker;
    a2 /= norm_ker;
    if a1 * ni2.sqrt() + a2 * nj2.sqrt() < 0.0 {
        a1 = -a1;
        a2 = -a2;
    }
    (a as f32, b as f32, a1, a2)
}

#[inline]
fn prune_distortion(n_live: usize, e_live: f64, cap_i: f64) -> f64 {
    (n_live as f64) * cap_i / (e_live - cap_i).max(1e-12)
}

#[inline]
fn merge_distortion(n_live: usize, e_rem: f64, a: f64, b: f64) -> (f64, f64) {
    let denom = (2.0 * e_rem + b).max(1e-12);
    let s_star = (a + b * e_rem) / denom;
    let err2 = (a + 2.0 * s_star * s_star - 2.0 * s_star * b).max(0.0);
    let j = (n_live as f64) * err2.sqrt() / (e_rem + s_star).max(1e-12);
    (j, s_star)
}

/// Decodes a 2D/ND weight tensor into row-major `[rows, cols]` f32 using `Mat::decode_row_f32` when packed.
fn decode_matrix_rows(t: &Tensor, rows: usize, cols: usize) -> Vec<f32> {
    match t {
        Tensor::Mat(m) if m.rows == rows && m.cols == cols => {
            let mut out = vec![0.0f32; rows * cols];
            par_rows(&mut out, cols, &|r, dst| m.decode_row_f32(r, dst));
            out
        }
        _ => t.to_f32(),
    }
}

/// Decodes `[d_out, n]` (with optional trailing `spatial`, i.e. `[d_out, n, spatial]`) transposed into
/// neuron-major `[n, d_out * spatial]`.
fn decode_out_columns(t: &Tensor, d_out: usize, n: usize, spatial: usize) -> Vec<f32> {
    let row_len = n * spatial;
    let raw = decode_matrix_rows(t, d_out, row_len);
    let mut cols_major = vec![0.0f32; n * d_out * spatial];
    for o in 0..d_out {
        let src_row = &raw[o * row_len..(o + 1) * row_len];
        for i in 0..n {
            let dst = &mut cols_major[(i * d_out + o) * spatial..(i * d_out + o + 1) * spatial];
            dst.copy_from_slice(&src_row[i * spatial..(i + 1) * spatial]);
        }
    }
    cols_major
}

fn tensor_quant_format(t: &Tensor) -> QuantFormat {
    match t {
        Tensor::F32(_) | Tensor::U32(_) => QuantFormat::FP32,
        Tensor::F16(_) => QuantFormat::FP16,
        Tensor::BF16(_) => QuantFormat::BF16,
        Tensor::Mat(m) => m.packed().map_or(QuantFormat::BF16, |p| p.format),
    }
}

fn block_align_for_out(t: &Tensor) -> usize {
    let f = tensor_quant_format(t);
    if is_float(f) {
        return 1;
    }
    let target = if encodable(f) {
        f
    } else {
        smallest_ladder_ge(f.bits_per_weight())
    };
    gguf_block_codec(target).map_or(32, |c| c.0)
}

fn repack_tensor(
    orig: &Tensor,
    f32_data: &[f32],
    rows: usize,
    cols: usize,
    surviving_unmerged_rows: Option<&[Option<usize>]>,
) -> Result<(Tensor, DType), String> {
    match orig {
        Tensor::F32(_) => Ok((Tensor::F32(f32_data.to_vec()), DType::F32)),
        Tensor::F16(_) => Ok((
            Tensor::F16(f32_data.iter().map(|&x| f32_to_f16(x)).collect()),
            DType::F16,
        )),
        Tensor::BF16(_) => Ok((
            Tensor::BF16(f32_data.iter().map(|&x| f32_to_bf16(x)).collect()),
            DType::BF16,
        )),
        Tensor::U32(_) => Err("cannot repack U32 weight in HOPE".into()),
        Tensor::Mat(m) => {
            let Some(p) = m.packed() else {
                let bf: Vec<u16> = f32_data.iter().map(|&x| f32_to_bf16(x)).collect();
                return Ok((Tensor::Mat(Mat::from_bf16(bf, rows, cols)), DType::BF16));
            };
            let src_fmt = p.format;
            if let Some(orig_map) = surviving_unmerged_rows {
                if cols == m.cols && orig_map.iter().all(Option::is_some) {
                    // Pure row selection: copy packed row blocks byte-for-byte (zero requantization loss,
                    // works for all 22 codecs including IQ2/IQ3/TQ/MXFP4).
                    let row_bytes = src_fmt.tensor_bytes(cols);
                    let mut bytes = Vec::with_capacity(rows * row_bytes);
                    for &orig_r in orig_map {
                        let r = orig_r.unwrap();
                        let start = p.off + r * row_bytes;
                        bytes.extend_from_slice(&p.map[start..start + row_bytes]);
                    }
                    let dtype = match src_fmt {
                        QuantFormat::FP32 => DType::F32,
                        QuantFormat::FP16 => DType::F16,
                        QuantFormat::BF16 => DType::BF16,
                        q => DType::Quant(q),
                    };
                    return Ok((
                        Tensor::Mat(Mat::from_packed_bytes(src_fmt, bytes, rows, cols)),
                        dtype,
                    ));
                }
                if cols == m.cols && encodable(src_fmt) {
                    // Copy unmerged rows verbatim; encode only merged rows into `src_fmt`.
                    let row_bytes = src_fmt.tensor_bytes(cols);
                    let mut bytes = Vec::with_capacity(rows * row_bytes);
                    for (new_r, &orig_r) in orig_map.iter().enumerate() {
                        if let Some(r) = orig_r {
                            let start = p.off + r * row_bytes;
                            bytes.extend_from_slice(&p.map[start..start + row_bytes]);
                        } else {
                            let row_slice = &f32_data[new_r * cols..(new_r + 1) * cols];
                            let enc = encode_rows(row_slice, cols, src_fmt)?;
                            bytes.extend_from_slice(&enc);
                        }
                    }
                    let dtype = match src_fmt {
                        QuantFormat::FP32 => DType::F32,
                        QuantFormat::FP16 => DType::F16,
                        QuantFormat::BF16 => DType::BF16,
                        q => DType::Quant(q),
                    };
                    return Ok((
                        Tensor::Mat(Mat::from_packed_bytes(src_fmt, bytes, rows, cols)),
                        dtype,
                    ));
                }
            }
            let target_fmt = if encodable(src_fmt) {
                src_fmt
            } else {
                smallest_ladder_ge(src_fmt.bits_per_weight())
            };
            let bytes = encode_rows(f32_data, cols, target_fmt)?;
            let dtype = match target_fmt {
                QuantFormat::FP32 => DType::F32,
                QuantFormat::FP16 => DType::F16,
                QuantFormat::BF16 => DType::BF16,
                q => DType::Quant(q),
            };
            Ok((
                Tensor::Mat(Mat::from_packed_bytes(target_fmt, bytes, rows, cols)),
                dtype,
            ))
        }
    }
}

/// Fast SimHash candidate pairing ($O(N \cdot B)$, $B = 16$) using seeded Rademacher signs + 64-d FWHT.
fn simhash_candidates(dirs: &[f32], dim: usize, n: usize) -> Vec<(usize, usize)> {
    if n <= 32 {
        let mut pairs = Vec::with_capacity(n * (n.saturating_sub(1)) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                pairs.push((i, j));
            }
        }
        return pairs;
    }
    let mut hashes = vec![[0u16; SIMHASH_TABLES]; n];
    for t in 0..SIMHASH_TABLES {
        let seed_base = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul((t as u64) + 1);
        for i in 0..n {
            let row = &dirs[i * dim..(i + 1) * dim];
            let mut buf = [0.0f32; SIMHASH_DIM];
            let chunks = dim.div_ceil(SIMHASH_DIM);
            for c in 0..chunks {
                let mut s = seed_base ^ ((c as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                let start = c * SIMHASH_DIM;
                let end = (start + SIMHASH_DIM).min(dim);
                for (k, &v) in row[start..end].iter().enumerate() {
                    s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                    let sign = if ((s ^ (s >> 31)) & 1) == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    buf[k] += v * sign;
                }
            }
            fwht_in_place(&mut buf);
            let mut h = 0u16;
            for b in 0..SIMHASH_BITS {
                if buf[b] >= 0.0 {
                    h |= 1 << b;
                }
            }
            hashes[i][t] = h;
        }
    }
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); 1 << SIMHASH_BITS];
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    for t in 0..SIMHASH_TABLES {
        for b in &mut buckets {
            b.clear();
        }
        for (i, h) in hashes.iter().enumerate() {
            buckets[h[t] as usize].push(i as u32);
        }
        for bucket in &buckets {
            if bucket.len() < 2 {
                continue;
            }
            for (idx, &i) in bucket.iter().enumerate() {
                // Probe up to SIMHASH_CANDIDATES neighbors in each bucket to bound work strictly O(N*B).
                for &j in bucket[idx + 1..].iter().take(SIMHASH_CANDIDATES) {
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    seen.insert((lo, hi));
                }
            }
        }
    }
    // Keep top SIMHASH_CANDIDATES highest-alignment candidates per neuron.
    let mut per_neuron: Vec<Vec<(f32, u32)>> = vec![Vec::new(); n];
    for &(i, j) in &seen {
        let (iu, ju) = (i as usize, j as usize);
        let ri = &dirs[iu * dim..(iu + 1) * dim];
        let rj = &dirs[ju * dim..(ju + 1) * dim];
        let mut align = 0.0f32;
        for k in 0..dim {
            align += ri[k] * rj[k];
        }
        per_neuron[iu].push((align, j));
    }
    let mut out = Vec::new();
    for (i, list) in per_neuron.iter_mut().enumerate() {
        if list.len() > SIMHASH_CANDIDATES {
            list.select_nth_unstable_by(SIMHASH_CANDIDATES, |a, b| b.0.total_cmp(&a.0));
            list.truncate(SIMHASH_CANDIDATES);
        }
        for &(_, j) in list.iter() {
            out.push((i, j as usize));
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn build_group_profile(group: &NeuronGroup, weights: &Weights) -> Option<GroupProfile> {
    let n = group.n;
    match &group.kind {
        GroupKind::Plain {
            act,
            w_in,
            b_in,
            w_out,
            in_spatial,
            out_spatial,
        } => {
            let tw_in = weights.get(w_in)?;
            let tw_out = weights.get(w_out)?;
            let in_width = group.d_in * *in_spatial;
            let out_width = group.d_out * *out_spatial;
            let w_in_rows = decode_matrix_rows(tw_in, n, in_width);
            let b_in_vec = b_in
                .as_ref()
                .and_then(|bn| weights.get(bn))
                .map(Tensor::to_f32)
                .unwrap_or_else(|| vec![0.0; n]);
            let w_out_cols = decode_out_columns(tw_out, group.d_out, n, *out_spatial);

            // Expand upstream channel norm `(g, beta0)` across spatial kernel taps if Conv.
            let mut g_exp = Vec::with_capacity(in_width);
            let mut b0_exp = Vec::with_capacity(in_width);
            for k in 0..group.d_in {
                for _ in 0..*in_spatial {
                    g_exp.push(group.norm_g[k]);
                    b0_exp.push(group.norm_b[k]);
                }
            }

            let dir_dim = in_width + 1;
            let mut dirs = vec![0.0f32; n * dir_dim];
            let mut gammas = vec![0.0f64; n];
            let mut betas = vec![0.0f64; n];
            let mut sqrt_kii = vec![0.0f64; n];
            let mut tilde_w_out = vec![0.0f32; n * out_width];
            let mut tilde_norm2 = vec![0.0f64; n];
            let mut caps = vec![0.0f32; n];

            for i in 0..n {
                let (gamma, beta) = plain_eff_stats(
                    &w_in_rows[i * in_width..(i + 1) * in_width],
                    b_in_vec[i],
                    &g_exp,
                    &b0_exp,
                    &mut dirs[i * dir_dim..(i + 1) * dir_dim],
                );
                let kii = self_kernel_scalar(*act, gamma, beta);
                let sk = kii.sqrt();
                gammas[i] = gamma;
                betas[i] = beta;
                sqrt_kii[i] = sk;

                let src = &w_out_cols[i * out_width..(i + 1) * out_width];
                let dst = &mut tilde_w_out[i * out_width..(i + 1) * out_width];
                let mut n2 = 0.0f64;
                for k in 0..out_width {
                    let v = (src[k] as f64) * sk;
                    dst[k] = v as f32;
                    n2 += v * v;
                }
                tilde_norm2[i] = n2;
                caps[i] = n2.sqrt() as f32;
            }

            let cand_pairs = simhash_candidates(&dirs, dir_dim, n);
            let mut merges = Vec::with_capacity(cand_pairs.len());
            for (i, j) in cand_pairs {
                let di = &dirs[i * dir_dim..(i + 1) * dir_dim];
                let dj = &dirs[j * dir_dim..(j + 1) * dir_dim];
                let mut rho = 0.0f64;
                for k in 0..dir_dim {
                    rho += (di[k] as f64) * (dj[k] as f64);
                }
                let rho_eff = match act.lambda() {
                    None => rho.clamp(-1.0, 1.0),
                    Some(lam) => {
                        // Eq. 80 effective correlation, reaching 1 when inputs match identically.
                        let kappa = lam * lam * gammas[i] * gammas[j] * rho;
                        let eq80 = (2.0 * kappa) / (1.0 + (1.0 + 4.0 * kappa * kappa).sqrt());
                        let rel_diff = ((gammas[i] - gammas[j]).powi(2)
                            + (betas[i] - betas[j]).powi(2))
                            / (gammas[i] * gammas[i] + gammas[j] * gammas[j] + 1e-12);
                        if rho > 0.9999 && rel_diff < 1e-6 {
                            rho.clamp(-1.0, 1.0)
                        } else {
                            eq80.max(rho * 0.95).clamp(-1.0, 1.0)
                        }
                    }
                };
                let c_ker = cho_saul_j1(rho_eff);
                if c_ker < 0.78 {
                    continue;
                }
                let wi = &tilde_w_out[i * out_width..(i + 1) * out_width];
                let wj = &tilde_w_out[j * out_width..(j + 1) * out_width];
                let mut w_dot = 0.0f64;
                for k in 0..out_width {
                    w_dot += (wi[k] as f64) * (wj[k] as f64);
                }
                let (a, b, _, _) = pair_gram_ab(tilde_norm2[i], tilde_norm2[j], w_dot, c_ker);
                merges.push(MergeCand {
                    i: i as u32,
                    j: j as u32,
                    c_ker: c_ker as f32,
                    a,
                    b,
                });
            }
            let e_live: f64 = caps.iter().map(|&c| c as f64).sum();
            Some(GroupProfile {
                caps,
                merges,
                alive: vec![true; n],
                merged_into: vec![None; n],
                used_in_merge: vec![false; n],
                n_live: n,
                e_live,
                evicted: false,
            })
        }
        GroupKind::Glu {
            act,
            w_gate,
            b_gate,
            w_up,
            b_up,
            sub_norm,
            w_down,
        } => {
            let tw_g = weights.get(w_gate)?;
            let tw_u = weights.get(w_up)?;
            let tw_d = weights.get(w_down)?;
            let wg_rows = decode_matrix_rows(tw_g, n, group.d_in);
            let wu_rows = decode_matrix_rows(tw_u, n, group.d_in);
            let bg_vec = b_gate
                .as_ref()
                .and_then(|bn| weights.get(bn))
                .map(Tensor::to_f32)
                .unwrap_or_else(|| vec![0.0; n]);
            let bu_vec = b_up
                .as_ref()
                .and_then(|bn| weights.get(bn))
                .map(Tensor::to_f32)
                .unwrap_or_else(|| vec![0.0; n]);
            let sub_vec = sub_norm
                .as_ref()
                .and_then(|sn| weights.get(sn))
                .map(Tensor::to_f32);
            let wd_cols = decode_out_columns(tw_d, group.d_out, n, 1);

            let d_in = group.d_in;
            let d_out = group.d_out;
            let mut ug = vec![0.0f32; n * d_in];
            let mut uu = vec![0.0f32; n * d_in];
            let mut cat_dirs = vec![0.0f32; n * (2 * d_in)];
            let mut dot_gu = vec![0.0f64; n];
            let mut gamma_g = vec![0.0f64; n];
            let mut beta_g = vec![0.0f64; n];
            let mut tilde_w_down = vec![0.0f32; n * d_out];
            let mut tilde_norm2 = vec![0.0f64; n];
            let mut caps = vec![0.0f32; n];

            let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
            for i in 0..n {
                let (gg, bg_eff, _gu, dgu, kii) = glu_eff_stats(
                    *act,
                    &wg_rows[i * d_in..(i + 1) * d_in],
                    bg_vec[i],
                    &wu_rows[i * d_in..(i + 1) * d_in],
                    bu_vec[i],
                    &group.norm_g,
                    &group.norm_b,
                    &mut ug[i * d_in..(i + 1) * d_in],
                    &mut uu[i * d_in..(i + 1) * d_in],
                );
                gamma_g[i] = gg;
                beta_g[i] = bg_eff;
                dot_gu[i] = dgu;
                for k in 0..d_in {
                    cat_dirs[i * 2 * d_in + k] = ug[i * d_in + k] * inv_sqrt2;
                    cat_dirs[i * 2 * d_in + d_in + k] = uu[i * d_in + k] * inv_sqrt2;
                }
                let sub_scale = sub_vec.as_ref().map_or(1.0f64, |sv| sv[i] as f64);
                let sk = kii.sqrt() * sub_scale.abs();
                let src = &wd_cols[i * d_out..(i + 1) * d_out];
                let dst = &mut tilde_w_down[i * d_out..(i + 1) * d_out];
                let sign = if sub_scale < 0.0 { -sk } else { sk };
                let mut n2 = 0.0f64;
                for k in 0..d_out {
                    let v = (src[k] as f64) * sign;
                    dst[k] = v as f32;
                    n2 += v * v;
                }
                tilde_norm2[i] = n2;
                caps[i] = n2.sqrt() as f32;
            }

            let cand_pairs = simhash_candidates(&cat_dirs, 2 * d_in, n);
            let mut merges = Vec::with_capacity(cand_pairs.len());
            for (i, j) in cand_pairs {
                let ugi = &ug[i * d_in..(i + 1) * d_in];
                let ugj = &ug[j * d_in..(j + 1) * d_in];
                let uui = &uu[i * d_in..(i + 1) * d_in];
                let uuj = &uu[j * d_in..(j + 1) * d_in];
                let (mut rg, mut ru) = (0.0f64, 0.0f64);
                for k in 0..d_in {
                    rg += (ugi[k] as f64) * (ugj[k] as f64);
                    ru += (uui[k] as f64) * (uuj[k] as f64);
                }
                let rel_g = ((gamma_g[i] - gamma_g[j]).powi(2) + (beta_g[i] - beta_g[j]).powi(2))
                    / (gamma_g[i] * gamma_g[i] + gamma_g[j] * gamma_g[j] + 1e-12);
                let j_gate = if rg > 0.9999 && rel_g < 1e-6 {
                    cho_saul_j1(rg)
                } else {
                    let lam = act.lambda().unwrap_or(1.0);
                    let kappa = lam * lam * gamma_g[i] * gamma_g[j] * rg;
                    let eq80 = (2.0 * kappa) / (1.0 + (1.0 + 4.0 * kappa * kappa).sqrt());
                    cho_saul_j1(eq80.max(rg * 0.95))
                };
                let up_corr = (ru + 2.0 * dot_gu[i] * dot_gu[j] * rg)
                    / ((1.0 + 2.0 * dot_gu[i] * dot_gu[i]) * (1.0 + 2.0 * dot_gu[j] * dot_gu[j]))
                        .sqrt();
                let c_ker = (j_gate * up_corr).clamp(-1.0, 1.0);
                if c_ker < 0.78 {
                    continue;
                }
                let wi = &tilde_w_down[i * d_out..(i + 1) * d_out];
                let wj = &tilde_w_down[j * d_out..(j + 1) * d_out];
                let mut w_dot = 0.0f64;
                for k in 0..d_out {
                    w_dot += (wi[k] as f64) * (wj[k] as f64);
                }
                let (a, b, _, _) = pair_gram_ab(tilde_norm2[i], tilde_norm2[j], w_dot, c_ker);
                merges.push(MergeCand {
                    i: i as u32,
                    j: j as u32,
                    c_ker: c_ker as f32,
                    a,
                    b,
                });
            }
            let e_live: f64 = caps.iter().map(|&c| c as f64).sum();
            Some(GroupProfile {
                caps,
                merges,
                alive: vec![true; n],
                merged_into: vec![None; n],
                used_in_merge: vec![false; n],
                n_live: n,
                e_live,
                evicted: false,
            })
        }
    }
}

fn upstream_norm_affine(
    graph: &Graph,
    weights: &Weights,
    x_val: ValueId,
    d_in: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut g = vec![1.0f32; d_in];
    let mut b = vec![0.0f32; d_in];
    let Some(prod) = graph.producer(x_val) else {
        return (g, b);
    };
    match prod.op {
        Op::RmsNorm { .. } => {
            if let Some(&w_id) = prod.inputs.get(1) {
                if let Some(tw) = weights.get(&graph.value(w_id).name) {
                    let v = tw.to_f32();
                    if v.len() == d_in {
                        g = v;
                    }
                }
            }
        }
        Op::LayerNorm { .. } | Op::GroupNorm { .. } | Op::InstanceNorm { .. } => {
            if let Some(&w_id) = prod.inputs.get(1) {
                if let Some(tw) = weights.get(&graph.value(w_id).name) {
                    let v = tw.to_f32();
                    if v.len() == d_in {
                        g = v;
                    }
                }
            }
            if let Some(&b_id) = prod.inputs.get(2) {
                if let Some(tb) = weights.get(&graph.value(b_id).name) {
                    let v = tb.to_f32();
                    if v.len() == d_in {
                        b = v;
                    }
                }
            }
        }
        _ => {}
    }
    (g, b)
}

fn discover_groups(
    graph: &Graph,
    weights: &Weights,
    val_readers: &[usize],
    weight_readers: &HashMap<&str, usize>,
    min_neurons: usize,
    min_align: usize,
) -> Vec<NeuronGroup> {
    let mut groups = Vec::new();
    let mut claimed_nodes: HashSet<usize> = HashSet::new();

    let single_weight = |v: ValueId| -> Option<&str> {
        let val = graph.value(v);
        if val.kind != ValueKind::Weight {
            return None;
        }
        let name = val.name.as_str();
        (weight_readers.get(name).copied() == Some(1) && weights.contains_key(name)).then_some(name)
    };

    for (out_idx, node_out) in graph.nodes.iter().enumerate() {
        if claimed_nodes.contains(&out_idx) || node_out.inputs.len() < 2 {
            continue;
        }
        // 1. Try Gated GLU FFN: [Linear(gate), Linear(up)] -> SwiGlu|GeGlu -> [RmsNorm?] -> Linear(down)
        if matches!(node_out.op, Op::Linear) {
            let mid_val = node_out.inputs[0];
            if let Some(w_down) = single_weight(node_out.inputs[1]) {
                let w_down_shape = &graph.value(node_out.inputs[1]).shape;
                if w_down_shape.len() == 2 && val_readers[mid_val.0 as usize] == 1 {
                    let (d_out, n) = (w_down_shape[0], w_down_shape[1]);
                    let mut glu_val = mid_val;
                    let mut sub_norm_name: Option<String> = None;
                    let mut sub_idx = None;
                    if let ValueKind::Node(sn_id) = graph.value(mid_val).kind {
                        let sn = graph.node(sn_id);
                        if matches!(sn.op, Op::RmsNorm { .. }) && sn.inputs.len() >= 2 {
                            if let Some(sn_w) = single_weight(sn.inputs[1]) {
                                if graph.value(sn.inputs[1]).shape == [n]
                                    && val_readers[sn.inputs[0].0 as usize] == 1
                                {
                                    sub_norm_name = Some(sn_w.to_string());
                                    glu_val = sn.inputs[0];
                                    sub_idx = Some(sn_id.0 as usize);
                                }
                            }
                        }
                    }
                    if n >= min_neurons {
                        if let ValueKind::Node(glu_id) = graph.value(glu_val).kind {
                            let glu_node = graph.node(glu_id);
                            let glu_act = match glu_node.op {
                                Op::SwiGlu => Some(ActKind::Silu),
                                Op::GeGlu => Some(ActKind::Gelu),
                                _ => None,
                            };
                            if let (Some(act), &[g_val, u_val]) = (glu_act, &glu_node.inputs[..]) {
                                if val_readers[g_val.0 as usize] == 1
                                    && val_readers[u_val.0 as usize] == 1
                                {
                                    if let (ValueKind::Node(gid), ValueKind::Node(uid)) =
                                        (&graph.value(g_val).kind, &graph.value(u_val).kind)
                                    {
                                        let (gn, un) = (graph.node(*gid), graph.node(*uid));
                                        if matches!(gn.op, Op::Linear)
                                            && matches!(un.op, Op::Linear)
                                            && gn.inputs.len() >= 2
                                            && un.inputs.len() >= 2
                                            && gn.inputs[0] == un.inputs[0]
                                        {
                                            let w_gate = single_weight(gn.inputs[1]);
                                            let w_up = single_weight(un.inputs[1]);
                                            let b_gate = gn.inputs.get(2).and_then(|&b| {
                                                single_weight(b).map(str::to_string)
                                            });
                                            let b_up = un.inputs.get(2).and_then(|&b| {
                                                single_weight(b).map(str::to_string)
                                            });
                                            let gs = &graph.value(gn.inputs[1]).shape;
                                            let us = &graph.value(un.inputs[1]).shape;
                                            if let (Some(wg), Some(wu)) = (w_gate, w_up) {
                                                if gs.len() == 2
                                                    && gs == us
                                                    && gs[0] == n
                                                    && (gn.inputs.len() == 2 || b_gate.is_some())
                                                    && (un.inputs.len() == 2 || b_up.is_some())
                                                {
                                                    let d_in = gs[1];
                                                    let (norm_g, norm_b) = upstream_norm_affine(
                                                        graph,
                                                        weights,
                                                        gn.inputs[0],
                                                        d_in,
                                                    );
                                                    let align = weights
                                                        .get(w_down)
                                                        .map_or(1, block_align_for_out)
                                                        .max(min_align);
                                                    let first_node =
                                                        (gid.0 as usize).min(uid.0 as usize);
                                                    claimed_nodes.extend([
                                                        gid.0 as usize,
                                                        uid.0 as usize,
                                                        glu_id.0 as usize,
                                                        out_idx,
                                                    ]);
                                                    if let Some(si) = sub_idx {
                                                        claimed_nodes.insert(si);
                                                    }
                                                    groups.push(NeuronGroup {
                                                        first_node,
                                                        last_node: out_idx,
                                                        n,
                                                        d_in,
                                                        d_out,
                                                        align,
                                                        norm_g,
                                                        norm_b,
                                                        kind: GroupKind::Glu {
                                                            act,
                                                            w_gate: wg.to_string(),
                                                            b_gate,
                                                            w_up: wu.to_string(),
                                                            b_up,
                                                            sub_norm: sub_norm_name,
                                                            w_down: w_down.to_string(),
                                                        },
                                                    });
                                                    continue;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Try Plain 2-layer FFN / Conv1d / Conv2d: Linear|Conv -> Unary(Relu|Gelu|Silu) -> Linear|Conv
        let is_plain_op = matches!(
            node_out.op,
            Op::Linear | Op::Conv1d { groups: 1, .. } | Op::Conv2d { groups: 1, .. }
        );
        if !is_plain_op {
            continue;
        }
        let act_val = node_out.inputs[0];
        let Some(w_out) = single_weight(node_out.inputs[1]) else {
            continue;
        };
        if val_readers[act_val.0 as usize] != 1 {
            continue;
        }
        let ValueKind::Node(act_nid) = graph.value(act_val).kind else {
            continue;
        };
        let act_node = graph.node(act_nid);
        let Op::Unary { kind: ukind } = &act_node.op else {
            continue;
        };
        let Some(act) = ActKind::from_unary(ukind) else {
            continue;
        };
        let in_val = act_node.inputs[0];
        if val_readers[in_val.0 as usize] != 1 {
            continue;
        }
        let ValueKind::Node(in_nid) = graph.value(in_val).kind else {
            continue;
        };
        let node_in = graph.node(in_nid);
        let same_kind = matches!(
            (&node_in.op, &node_out.op),
            (Op::Linear, Op::Linear)
                | (Op::Conv1d { groups: 1, .. }, Op::Conv1d { groups: 1, .. })
                | (Op::Conv2d { groups: 1, .. }, Op::Conv2d { groups: 1, .. })
        );
        if !same_kind || node_in.inputs.len() < 2 {
            continue;
        }
        let Some(w_in) = single_weight(node_in.inputs[1]) else {
            continue;
        };
        let b_in = node_in
            .inputs
            .get(2)
            .and_then(|&b| single_weight(b).map(str::to_string));
        if node_in.inputs.len() >= 3 && b_in.is_none() {
            continue;
        }
        let s_in = &graph.value(node_in.inputs[1]).shape;
        let s_out = &graph.value(node_out.inputs[1]).shape;
        if s_in.len() < 2
            || s_out.len() != s_in.len()
            || s_in[0] != s_out[1]
            || s_in[0] < min_neurons
        {
            continue;
        }
        let n = s_in[0];
        let d_in = s_in[1];
        let d_out = s_out[0];
        let in_spatial: usize = s_in[2..].iter().product();
        let out_spatial: usize = s_out[2..].iter().product();
        let (norm_g, norm_b) = if in_spatial == 1 {
            upstream_norm_affine(graph, weights, node_in.inputs[0], d_in)
        } else {
            (vec![1.0; d_in], vec![0.0; d_in])
        };
        let align = if out_spatial == 1 {
            weights
                .get(w_out)
                .map_or(1, block_align_for_out)
                .max(min_align)
        } else {
            min_align
        };
        claimed_nodes.extend([in_nid.0 as usize, act_nid.0 as usize, out_idx]);
        groups.push(NeuronGroup {
            first_node: in_nid.0 as usize,
            last_node: out_idx,
            n,
            d_in,
            d_out,
            align,
            norm_g,
            norm_b,
            kind: GroupKind::Plain {
                act,
                w_in: w_in.to_string(),
                b_in,
                w_out: w_out.to_string(),
                in_spatial,
                out_spatial,
            },
        });
    }
    groups
}

/// Computes closed-branch weight/state signature so reshaped instances match evicted residual blocks.
fn branch_signature(graph: &Graph, branch_nodes: &HashSet<usize>, add_name: &str) -> String {
    let mut items: Vec<&str> = Vec::new();
    for &idx in branch_nodes {
        for &inp in &graph.nodes[idx].inputs {
            match &graph.value(inp).kind {
                ValueKind::Weight => items.push(graph.value(inp).name.as_str()),
                ValueKind::State { name, .. } => items.push(name.as_str()),
                _ => {}
            }
        }
    }
    items.sort_unstable();
    items.dedup();
    if items.is_empty() {
        format!("add:{add_name}")
    } else {
        format!("w:{}", items.join("|"))
    }
}

/// Discovers closed `y = Add(skip, branch)` residual blocks; excludes first and last matched block.
fn discover_residual_blocks(
    graph: &Graph,
    weights: &Weights,
    groups: &[NeuronGroup],
) -> Vec<ResidualBlock> {
    let graph_outs: HashSet<ValueId> = graph.outputs.iter().copied().collect();
    let mut raw_blocks: Vec<ResidualBlock> = Vec::new();

    for (add_idx, node) in graph.nodes.iter().enumerate() {
        if !matches!(
            node.op,
            Op::Binary {
                kind: BinaryKind::Add
            }
        ) || node.inputs.len() != 2
        {
            continue;
        }
        let (a, b) = (node.inputs[0], node.inputs[1]);
        let (va, vb) = (graph.value(a), graph.value(b));
        if va.dtype != DType::F32
            || vb.dtype != DType::F32
            || va.shape != vb.shape
            || va.shape.is_empty()
        {
            continue;
        }
        // `branch` is produced strictly after `skip`.
        let prod_idx = |v: ValueId| match graph.value(v).kind {
            ValueKind::Node(n) => Some(n.0 as usize),
            _ => None,
        };
        let (skip_val, branch_val) = match (prod_idx(a), prod_idx(b)) {
            (Some(pa), Some(pb)) if pb > pa => (a, b),
            (Some(pa), Some(pb)) if pa > pb => (b, a),
            (None, Some(_)) => (a, b),
            (Some(_), None) => (b, a),
            _ => continue,
        };
        let Some(branch_root) = prod_idx(branch_val) else {
            continue;
        };

        // Backward traversal from `branch_root`, halting at `skip_val`, Weight, State, and non-primary Input.
        let mut branch_nodes: HashSet<usize> = HashSet::new();
        let mut q = VecDeque::new();
        q.push_back(branch_root);
        branch_nodes.insert(branch_root);
        let mut saw_skip = false;
        let mut valid = true;

        while let Some(nid) = q.pop_front() {
            for &inp in &graph.nodes[nid].inputs {
                if inp == skip_val {
                    saw_skip = true;
                    continue;
                }
                match &graph.value(inp).kind {
                    ValueKind::Weight | ValueKind::State { .. } => {}
                    ValueKind::Input => {
                        // Auxiliary index/mask inputs are allowed; floating primary stream must enter via `skip_val`.
                        if graph.value(inp).dtype == DType::F32
                            && graph.value(inp).shape == va.shape
                        {
                            valid = false;
                            break;
                        }
                    }
                    ValueKind::Node(p) => {
                        let pid = p.0 as usize;
                        if let Some(sp) = prod_idx(skip_val) {
                            if pid < sp {
                                // Reads an earlier value (e.g., PLE slice) — only legal if it does not escape;
                                // halt traversal outside the interval `[sp + 1 .. add_idx]`.
                                continue;
                            }
                        }
                        if branch_nodes.insert(pid) {
                            q.push_back(pid);
                        }
                    }
                }
            }
            if !valid {
                break;
            }
        }
        if !valid || !saw_skip {
            continue;
        }

        // Closure check: no branch value is read outside `branch_nodes` (except `branch_val` read by `add_idx`),
        // and no `State` touched inside `branch_nodes` is read/written outside `branch_nodes`.
        let mut branch_states: HashSet<&str> = HashSet::new();
        for &nid in &branch_nodes {
            for &out_v in &graph.nodes[nid].outputs {
                if graph_outs.contains(&out_v) {
                    valid = false;
                    break;
                }
            }
            for &inp in &graph.nodes[nid].inputs {
                if let ValueKind::State { name, .. } = &graph.value(inp).kind {
                    branch_states.insert(name.as_str());
                }
            }
        }
        if !valid {
            continue;
        }
        for (other_idx, other) in graph.nodes.iter().enumerate() {
            if branch_nodes.contains(&other_idx) {
                continue;
            }
            for &inp in &other.inputs {
                if other_idx == add_idx && inp == branch_val {
                    continue;
                }
                if let ValueKind::Node(p) = graph.value(inp).kind {
                    if branch_nodes.contains(&(p.0 as usize)) {
                        valid = false;
                        break;
                    }
                }
                if let ValueKind::State { name, .. } = &graph.value(inp).kind {
                    if branch_states.contains(name.as_str()) {
                        valid = false;
                        break;
                    }
                }
            }
            if !valid {
                break;
            }
        }
        if !valid {
            continue;
        }

        let group_indices: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter(|(_, g)| {
                branch_nodes.contains(&g.first_node) && branch_nodes.contains(&g.last_node)
            })
            .map(|(gi, _)| gi)
            .collect();

        // Count unique branch parameters (weights read ONLY inside this branch).
        let mut branch_weights: HashSet<&str> = HashSet::new();
        for &nid in &branch_nodes {
            for &inp in &graph.nodes[nid].inputs {
                if graph.value(inp).kind == ValueKind::Weight {
                    branch_weights.insert(graph.value(inp).name.as_str());
                }
            }
        }
        for (other_idx, other) in graph.nodes.iter().enumerate() {
            if !branch_nodes.contains(&other_idx) {
                for &inp in &other.inputs {
                    if graph.value(inp).kind == ValueKind::Weight {
                        branch_weights.remove(graph.value(inp).name.as_str());
                    }
                }
            }
        }
        let total_params: usize = branch_weights
            .iter()
            .filter_map(|name| weights.get(name).map(Tensor::len))
            .filter(|&len| len > 1)
            .sum();
        if total_params == 0 {
            continue;
        }

        // Ambient dimension & upstream norm identity capacity $E_{\mathrm{identity}}$.
        let d_amb = *va.shape.last().unwrap_or(&1);
        let mut norm_g = vec![1.0f32; d_amb];
        let mut norm_b = vec![0.0f32; d_amb];
        for &nid in &branch_nodes {
            let n = &graph.nodes[nid];
            if n.inputs.first() == Some(&skip_val) {
                if matches!(n.op, Op::RmsNorm { .. } | Op::LayerNorm { .. }) {
                    if let Some(&wid) = n.inputs.get(1) {
                        if let Some(tw) = weights.get(&graph.value(wid).name) {
                            let v = tw.to_f32();
                            if v.len() == d_amb {
                                norm_g = v;
                            }
                        }
                    }
                    if let Some(&bid) = n.inputs.get(2) {
                        if let Some(tb) = weights.get(&graph.value(bid).name) {
                            let v = tb.to_f32();
                            if v.len() == d_amb {
                                norm_b = v;
                            }
                        }
                    }
                    break;
                }
            }
        }
        // Integrated Hilbert-Schmidt skip scale so macro eviction matches granular `sum_i J_prune(i) ~ M^2`.
        let identity_energy: f64 = ((0..d_amb)
            .map(|k| (norm_g[k] as f64).powi(2) + (norm_b[k] as f64).powi(2))
            .sum::<f64>()
            / (d_amb.max(1) as f64))
            .sqrt()
            .max(1e-12);

        // Attention-only branch energy (when `group_indices.is_empty()`): rank-1 column norms of W_O * W_V
        // (or last Linear outgoing projection norm * incoming value norm).
        let mut extra_energy = 0.0f64;
        if group_indices.is_empty() {
            if let Op::Linear = graph.nodes[branch_root].op {
                if let Some(&wo_id) = graph.nodes[branch_root].inputs.get(1) {
                    if let Some(two) = weights.get(&graph.value(wo_id).name) {
                        let swo = &graph.value(wo_id).shape;
                        if swo.len() == 2 {
                            let (d_o, d_mid) = (swo[0], swo[1]);
                            let wo_cols = decode_out_columns(two, d_o, d_mid, 1);
                            // Find W_V (if CachedAttention / Attention feeds branch_root) to scale column norms.
                            let mut v_row_norms = vec![1.0f64; d_mid];
                            let attn_in = graph.nodes[branch_root].inputs[0];
                            if let Some(attn_node) = graph.producer(attn_in) {
                                if let Some(&v_in) = attn_node.inputs.get(2) {
                                    if let Some(v_lin) = graph.producer(v_in) {
                                        if let Some(&wv_id) = v_lin.inputs.get(1) {
                                            if let Some(twv) = weights.get(&graph.value(wv_id).name)
                                            {
                                                let swv = &graph.value(wv_id).shape;
                                                if swv.len() == 2 && swv[0] > 0 {
                                                    let wv_rows =
                                                        decode_matrix_rows(twv, swv[0], swv[1]);
                                                    let rep = (d_mid / swv[0]).max(1);
                                                    for r in 0..swv[0] {
                                                        let rn = wv_rows
                                                            [r * swv[1]..(r + 1) * swv[1]]
                                                            .iter()
                                                            .map(|&x| (x as f64) * (x as f64))
                                                            .sum::<f64>()
                                                            .sqrt();
                                                        for h in 0..rep {
                                                            if r * rep + h < d_mid {
                                                                v_row_norms[r * rep + h] = rn;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            let mut e_wo = 0.0f64;
                            for c in 0..d_mid {
                                let cn = wo_cols[c * d_o..(c + 1) * d_o]
                                    .iter()
                                    .map(|&x| (x as f64) * (x as f64))
                                    .sum::<f64>()
                                    .sqrt();
                                e_wo += cn * v_row_norms[c];
                            }
                            extra_energy = ((d_mid * d_mid) as f64) * e_wo;
                        }
                    }
                }
            }
        }

        let sig = branch_signature(graph, &branch_nodes, &node.name);
        raw_blocks.push(ResidualBlock {
            add_name: node.name.clone(),
            sig,
            group_indices,
            extra_energy,
            identity_energy,
            total_params,
        });
    }

    // Protect first and last matched residual block so boundary projections stay intact.
    if raw_blocks.len() <= 2 {
        return Vec::new();
    }
    raw_blocks.remove(raw_blocks.len() - 1);
    raw_blocks.remove(0);
    raw_blocks
}

#[derive(Clone, Copy)]
enum GranularAction {
    Prune {
        g: usize,
        i: usize,
        dr: f64,
    },
    Merge {
        g: usize,
        i: usize,
        j: usize,
        c_ker: f32,
        s_star: f32,
        dr: f64,
    },
}

impl GranularAction {
    fn dr(&self) -> f64 {
        match *self {
            Self::Prune { dr, .. } | Self::Merge { dr, .. } => dr,
        }
    }
}

fn synthesize_group(
    group: &NeuronGroup,
    prof: &mut GroupProfile,
    weights: &Weights,
    out: &mut Vec<(String, Tensor, Vec<usize>, DType)>,
) -> Result<(usize, usize), String> {
    let n = group.n;
    // Round surviving count up to `align` by restoring highest-capacity pruned neurons.
    let target_n = prof
        .n_live
        .max(group.align)
        .next_multiple_of(group.align)
        .min(n);
    if target_n > prof.n_live {
        let mut pruned: Vec<(f32, usize)> = (0..n)
            .filter(|&i| !prof.alive[i] && !prof.used_in_merge[i])
            .map(|i| (prof.caps[i], i))
            .collect();
        pruned.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        for &(_, idx) in pruned.iter().take(target_n - prof.n_live) {
            prof.alive[idx] = true;
            prof.n_live += 1;
        }
        if target_n > prof.n_live {
            let mut merged: Vec<(f32, usize, usize)> = (0..n)
                .filter_map(|i| prof.merged_into[i].map(|(j, _, _)| (prof.caps[i], i, j)))
                .collect();
            merged.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            for &(_, i, j) in merged.iter().take(target_n - prof.n_live) {
                prof.merged_into[i] = None;
                prof.alive[j] = true;
                prof.n_live += 1;
            }
        }
    }
    if prof.n_live == n {
        return Ok((0, 0));
    }

    let survivors: Vec<usize> = (0..n).filter(|&i| prof.alive[i]).collect();
    let n_final = survivors.len();
    let mut merged_count = 0usize;

    match &group.kind {
        GroupKind::Plain {
            act,
            w_in,
            b_in,
            w_out,
            in_spatial,
            out_spatial,
        } => {
            let in_width = group.d_in * *in_spatial;
            let out_width = group.d_out * *out_spatial;
            let tw_in = weights
                .get(w_in)
                .ok_or_else(|| format!("missing {w_in}"))?
                .clone();
            let tw_out = weights
                .get(w_out)
                .ok_or_else(|| format!("missing {w_out}"))?
                .clone();
            let w_in_rows = decode_matrix_rows(&tw_in, n, in_width);
            let b_in_vec = b_in
                .as_ref()
                .and_then(|bn| weights.get(bn))
                .map(Tensor::to_f32)
                .unwrap_or_else(|| vec![0.0; n]);
            let w_out_cols = decode_out_columns(&tw_out, group.d_out, n, *out_spatial);

            let mut g_exp = Vec::with_capacity(in_width);
            let mut b0_exp = Vec::with_capacity(in_width);
            for k in 0..group.d_in {
                for _ in 0..*in_spatial {
                    g_exp.push(group.norm_g[k]);
                    b0_exp.push(group.norm_b[k]);
                }
            }

            let mut new_w_in = vec![0.0f32; n_final * in_width];
            let mut new_b_in = vec![0.0f32; n_final];
            let mut new_w_out_cols = vec![0.0f32; n_final * out_width];
            let mut unmerged_map: Vec<Option<usize>> = Vec::with_capacity(n_final);

            let dir_dim = in_width + 1;
            let mut di = vec![0.0f32; dir_dim];
            let mut dj = vec![0.0f32; dir_dim];

            for (slot, &i) in survivors.iter().enumerate() {
                if let Some((j, c_ker, s_star)) = prof.merged_into[i] {
                    merged_count += 1;
                    unmerged_map.push(None);
                    let (gi, bi) = plain_eff_stats(
                        &w_in_rows[i * in_width..(i + 1) * in_width],
                        b_in_vec[i],
                        &g_exp,
                        &b0_exp,
                        &mut di,
                    );
                    let (gj, bj) = plain_eff_stats(
                        &w_in_rows[j * in_width..(j + 1) * in_width],
                        b_in_vec[j],
                        &g_exp,
                        &b0_exp,
                        &mut dj,
                    );
                    let ski = self_kernel_scalar(*act, gi, bi).sqrt();
                    let skj = self_kernel_scalar(*act, gj, bj).sqrt();
                    let wi = &w_out_cols[i * out_width..(i + 1) * out_width];
                    let wj = &w_out_cols[j * out_width..(j + 1) * out_width];
                    let (mut ni2, mut nj2, mut w_dot) = (0.0f64, 0.0f64, 0.0f64);
                    for k in 0..out_width {
                        let ti = (wi[k] as f64) * ski;
                        let tj = (wj[k] as f64) * skj;
                        ni2 += ti * ti;
                        nj2 += tj * tj;
                        w_dot += ti * tj;
                    }
                    let (_, b_val, a1, a2) = pair_gram_ab(ni2, nj2, w_dot, c_ker as f64);
                    let sigma1 = (b_val as f64) * std::f64::consts::FRAC_1_SQRT_2;
                    let p1 = a1.abs() / (a1.abs() + a2.abs()).max(1e-12);
                    let p2 = 1.0 - p1;

                    // Combine normalized eff directions, then restore calibrated norm `(p1*r_i + p2*r_j)`.
                    let ri = (gi * gi + bi * bi).sqrt();
                    let rj = (gj * gj + bj * bj).sqrt();
                    let rp = p1 * ri + p2 * rj;
                    let mut dp = vec![0.0f64; dir_dim];
                    let mut dp_n2 = 0.0f64;
                    for k in 0..dir_dim {
                        let v = a1 * (di[k] as f64) + a2 * (dj[k] as f64);
                        dp[k] = v;
                        dp_n2 += v * v;
                    }
                    let dp_inv = 1.0 / dp_n2.max(1e-24).sqrt();
                    let mut w_dot_b0 = 0.0f64;
                    for k in 0..in_width {
                        let eff_k = rp * dp[k] * dp_inv;
                        let wk = if g_exp[k].abs() > 1e-8 {
                            eff_k / (g_exp[k] as f64)
                        } else {
                            p1 * (w_in_rows[i * in_width + k] as f64)
                                + p2 * (w_in_rows[j * in_width + k] as f64)
                        };
                        new_w_in[slot * in_width + k] = wk as f32;
                        w_dot_b0 += wk * (b0_exp[k] as f64);
                    }
                    let bp_eff = rp * dp[in_width] * dp_inv;
                    let bp_raw = if b_in.is_some() {
                        bp_eff - w_dot_b0
                    } else {
                        0.0
                    };
                    new_b_in[slot] = bp_raw as f32;

                    // Exact output rescaling by synthesized self-kernel K(p, p).
                    let (gp, bp_actual) = plain_eff_stats(
                        &new_w_in[slot * in_width..(slot + 1) * in_width],
                        new_b_in[slot],
                        &g_exp,
                        &b0_exp,
                        &mut di,
                    );
                    let skp = self_kernel_scalar(*act, gp, bp_actual).sqrt().max(1e-12);
                    let c = c_ker as f64;
                    let c1 = a1 + c * a2;
                    let c2 = c * a1 + a2;
                    let shrink = if sigma1 > 1e-12 && c < 0.9999 {
                        (std::f64::consts::SQRT_2 * (s_star as f64)) / sigma1
                    } else {
                        1.0
                    };
                    for k in 0..out_width {
                        let tw_p = shrink * (c1 * (wi[k] as f64) * ski + c2 * (wj[k] as f64) * skj);
                        new_w_out_cols[slot * out_width + k] = (tw_p / skp) as f32;
                    }
                } else {
                    unmerged_map.push(Some(i));
                    new_w_in[slot * in_width..(slot + 1) * in_width]
                        .copy_from_slice(&w_in_rows[i * in_width..(i + 1) * in_width]);
                    new_b_in[slot] = b_in_vec[i];
                    new_w_out_cols[slot * out_width..(slot + 1) * out_width]
                        .copy_from_slice(&w_out_cols[i * out_width..(i + 1) * out_width]);
                }
            }

            // Transpose `new_w_out_cols` `[n_final, d_out * out_spatial]` back to `[d_out, n_final * out_spatial]`.
            let new_out_row_len = n_final * *out_spatial;
            let mut new_w_out = vec![0.0f32; group.d_out * new_out_row_len];
            for o in 0..group.d_out {
                for slot in 0..n_final {
                    let src = &new_w_out_cols[(slot * group.d_out + o) * *out_spatial
                        ..(slot * group.d_out + o + 1) * *out_spatial];
                    let dst = &mut new_w_out[o * new_out_row_len + slot * *out_spatial
                        ..o * new_out_row_len + (slot + 1) * *out_spatial];
                    dst.copy_from_slice(src);
                }
            }

            let in_row_map = (*in_spatial == 1).then_some(unmerged_map.as_slice());
            let (packed_in, dt_in) =
                repack_tensor(&tw_in, &new_w_in, n_final, in_width, in_row_map)?;
            out.push((w_in.clone(), packed_in, vec![n_final, group.d_in], dt_in));

            if let Some(tb) = b_in
                .as_ref()
                .and_then(|bn| weights.get(bn).map(|t| (bn, t)))
            {
                let (packed_b, dt_b) = repack_tensor(tb.1, &new_b_in, 1, n_final, None)?;
                out.push((tb.0.clone(), packed_b, vec![n_final], dt_b));
            }

            let (packed_out, dt_out) =
                repack_tensor(&tw_out, &new_w_out, group.d_out, new_out_row_len, None)?;
            out.push((
                w_out.clone(),
                packed_out,
                vec![group.d_out, n_final],
                dt_out,
            ));
        }
        GroupKind::Glu {
            act,
            w_gate,
            b_gate,
            w_up,
            b_up,
            sub_norm,
            w_down,
        } => {
            let d_in = group.d_in;
            let d_out = group.d_out;
            let tw_g = weights
                .get(w_gate)
                .ok_or_else(|| format!("missing {w_gate}"))?
                .clone();
            let tw_u = weights
                .get(w_up)
                .ok_or_else(|| format!("missing {w_up}"))?
                .clone();
            let tw_d = weights
                .get(w_down)
                .ok_or_else(|| format!("missing {w_down}"))?
                .clone();
            let wg_rows = decode_matrix_rows(&tw_g, n, d_in);
            let wu_rows = decode_matrix_rows(&tw_u, n, d_in);
            let bg_vec = b_gate
                .as_ref()
                .and_then(|bn| weights.get(bn))
                .map(Tensor::to_f32)
                .unwrap_or_else(|| vec![0.0; n]);
            let bu_vec = b_up
                .as_ref()
                .and_then(|bn| weights.get(bn))
                .map(Tensor::to_f32)
                .unwrap_or_else(|| vec![0.0; n]);
            let sub_vec = sub_norm
                .as_ref()
                .and_then(|sn| weights.get(sn))
                .map(Tensor::to_f32);
            let wd_cols = decode_out_columns(&tw_d, d_out, n, 1);

            let mut new_wg = vec![0.0f32; n_final * d_in];
            let mut new_wu = vec![0.0f32; n_final * d_in];
            let mut new_bg = vec![0.0f32; n_final];
            let mut new_bu = vec![0.0f32; n_final];
            let mut new_sub = vec![1.0f32; n_final];
            let mut new_wd_cols = vec![0.0f32; n_final * d_out];
            let mut unmerged_map: Vec<Option<usize>> = Vec::with_capacity(n_final);

            let (mut ugi, mut uui) = (vec![0.0f32; d_in], vec![0.0f32; d_in]);
            let (mut ugj, mut uuj) = (vec![0.0f32; d_in], vec![0.0f32; d_in]);

            for (slot, &i) in survivors.iter().enumerate() {
                if let Some((j, c_ker, s_star)) = prof.merged_into[i] {
                    merged_count += 1;
                    unmerged_map.push(None);
                    let (ggi, bgi, gui, _, kii_i) = glu_eff_stats(
                        *act,
                        &wg_rows[i * d_in..(i + 1) * d_in],
                        bg_vec[i],
                        &wu_rows[i * d_in..(i + 1) * d_in],
                        bu_vec[i],
                        &group.norm_g,
                        &group.norm_b,
                        &mut ugi,
                        &mut uui,
                    );
                    let (ggj, bgj, guj, _, kii_j) = glu_eff_stats(
                        *act,
                        &wg_rows[j * d_in..(j + 1) * d_in],
                        bg_vec[j],
                        &wu_rows[j * d_in..(j + 1) * d_in],
                        bu_vec[j],
                        &group.norm_g,
                        &group.norm_b,
                        &mut ugj,
                        &mut uuj,
                    );
                    let sub_i = sub_vec.as_ref().map_or(1.0f64, |sv| sv[i] as f64);
                    let sub_j = sub_vec.as_ref().map_or(1.0f64, |sv| sv[j] as f64);
                    let ski = kii_i.sqrt() * sub_i;
                    let skj = kii_j.sqrt() * sub_j;

                    let wi = &wd_cols[i * d_out..(i + 1) * d_out];
                    let wj = &wd_cols[j * d_out..(j + 1) * d_out];
                    let (mut ni2, mut nj2, mut w_dot) = (0.0f64, 0.0f64, 0.0f64);
                    for k in 0..d_out {
                        let ti = (wi[k] as f64) * ski;
                        let tj = (wj[k] as f64) * skj;
                        ni2 += ti * ti;
                        nj2 += tj * tj;
                        w_dot += ti * tj;
                    }
                    let (_, b_val, a1, a2) = pair_gram_ab(ni2, nj2, w_dot, c_ker as f64);
                    let sigma1 = (b_val as f64) * std::f64::consts::FRAC_1_SQRT_2;
                    let p1 = a1.abs() / (a1.abs() + a2.abs()).max(1e-12);
                    let p2 = 1.0 - p1;

                    // Calibrated gate norm; up blends via `(a1, a2)`.
                    let ggp = p1 * ggi + p2 * ggj;
                    let gup = p1 * gui + p2 * guj;
                    let (mut dg, mut du) = (vec![0.0f64; d_in], vec![0.0f64; d_in]);
                    let (mut dgn2, mut dun2) = (0.0f64, 0.0f64);
                    for k in 0..d_in {
                        let vg = p1 * (ugi[k] as f64) + p2 * (ugj[k] as f64);
                        let vu = a1 * (uui[k] as f64) + a2 * (uuj[k] as f64);
                        dg[k] = vg;
                        du[k] = vu;
                        dgn2 += vg * vg;
                        dun2 += vu * vu;
                    }
                    let dginv = 1.0 / dgn2.max(1e-24).sqrt();
                    let duinv = 1.0 / dun2.max(1e-24).sqrt();
                    let mut wg_dot_b0 = 0.0f64;
                    for k in 0..d_in {
                        let eg = ggp * dg[k] * dginv;
                        let eu = gup * du[k] * duinv;
                        let gk = group.norm_g[k] as f64;
                        let wgk = if gk.abs() > 1e-8 {
                            eg / gk
                        } else {
                            p1 * (wg_rows[i * d_in + k] as f64)
                                + p2 * (wg_rows[j * d_in + k] as f64)
                        };
                        let wuk = if gk.abs() > 1e-8 {
                            eu / gk
                        } else {
                            p1 * (wu_rows[i * d_in + k] as f64)
                                + p2 * (wu_rows[j * d_in + k] as f64)
                        };
                        new_wg[slot * d_in + k] = wgk as f32;
                        new_wu[slot * d_in + k] = wuk as f32;
                        wg_dot_b0 += wgk * (group.norm_b[k] as f64);
                    }
                    new_bg[slot] = if b_gate.is_some() {
                        (p1 * bgi + p2 * bgj - wg_dot_b0) as f32
                    } else {
                        0.0
                    };
                    new_bu[slot] = if b_up.is_some() {
                        (p1 * (bu_vec[i] as f64) + p2 * (bu_vec[j] as f64)) as f32
                    } else {
                        0.0
                    };
                    let sub_p = p1 * sub_i + p2 * sub_j;
                    new_sub[slot] = sub_p as f32;

                    let (_, _, _, _, kii_p) = glu_eff_stats(
                        *act,
                        &new_wg[slot * d_in..(slot + 1) * d_in],
                        new_bg[slot],
                        &new_wu[slot * d_in..(slot + 1) * d_in],
                        new_bu[slot],
                        &group.norm_g,
                        &group.norm_b,
                        &mut ugi,
                        &mut uui,
                    );
                    let denom_sk = (kii_p.sqrt() * sub_p.abs().max(1e-8)).max(1e-12);
                    let denom_signed = if sub_p < 0.0 { -denom_sk } else { denom_sk };
                    let c = c_ker as f64;
                    let c1 = a1 + c * a2;
                    let c2 = c * a1 + a2;
                    let shrink = if sigma1 > 1e-12 && c < 0.9999 {
                        (std::f64::consts::SQRT_2 * (s_star as f64)) / sigma1
                    } else {
                        1.0
                    };
                    for k in 0..d_out {
                        let tw_p = shrink * (c1 * (wi[k] as f64) * ski + c2 * (wj[k] as f64) * skj);
                        new_wd_cols[slot * d_out + k] = (tw_p / denom_signed) as f32;
                    }
                } else {
                    unmerged_map.push(Some(i));
                    new_wg[slot * d_in..(slot + 1) * d_in]
                        .copy_from_slice(&wg_rows[i * d_in..(i + 1) * d_in]);
                    new_wu[slot * d_in..(slot + 1) * d_in]
                        .copy_from_slice(&wu_rows[i * d_in..(i + 1) * d_in]);
                    new_bg[slot] = bg_vec[i];
                    new_bu[slot] = bu_vec[i];
                    new_sub[slot] = sub_vec.as_ref().map_or(1.0, |sv| sv[i]);
                    new_wd_cols[slot * d_out..(slot + 1) * d_out]
                        .copy_from_slice(&wd_cols[i * d_out..(i + 1) * d_out]);
                }
            }

            let mut new_wd = vec![0.0f32; d_out * n_final];
            for o in 0..d_out {
                for slot in 0..n_final {
                    new_wd[o * n_final + slot] = new_wd_cols[slot * d_out + o];
                }
            }

            let (pg, dt_g) = repack_tensor(&tw_g, &new_wg, n_final, d_in, Some(&unmerged_map))?;
            out.push((w_gate.clone(), pg, vec![n_final, d_in], dt_g));

            let (pu, dt_u) = repack_tensor(&tw_u, &new_wu, n_final, d_in, Some(&unmerged_map))?;
            out.push((w_up.clone(), pu, vec![n_final, d_in], dt_u));

            for (name, new) in [(b_gate, &new_bg), (b_up, &new_bu), (sub_norm, &new_sub)] {
                if let Some((name, t)) = name.as_ref().and_then(|n| weights.get(n).map(|t| (n, t)))
                {
                    let (p, dt) = repack_tensor(t, new, 1, n_final, None)?;
                    out.push((name.clone(), p, vec![n_final], dt));
                }
            }

            let (pd, dt_d) = repack_tensor(&tw_d, &new_wd, d_out, n_final, None)?;
            out.push((w_down.clone(), pd, vec![d_out, n_final], dt_d));
        }
    }

    let removed = n - n_final;
    let pruned = removed.saturating_sub(merged_count);
    Ok((pruned, merged_count))
}

/// Applies a [`HopeRecipe`] (evicted residual blocks + resized weight shapes/dtypes) to `graph`.
pub fn apply_recipe(graph: &mut Graph, recipe: &HopeRecipe) {
    if !recipe.evicted_blocks.is_empty() {
        // Match residual `Add` nodes in `graph` by name or closed-branch signature.
        let dummy_weights = Weights::new();
        let mut to_remove: HashSet<usize> = HashSet::new();
        let mut val_remap: HashMap<ValueId, ValueId> = HashMap::new();

        for (add_idx, node) in graph.nodes.iter().enumerate() {
            if !matches!(
                node.op,
                Op::Binary {
                    kind: BinaryKind::Add
                }
            ) || node.inputs.len() != 2
            {
                continue;
            }
            let (a, b) = (node.inputs[0], node.inputs[1]);
            let prod_idx = |v: ValueId| match graph.value(v).kind {
                ValueKind::Node(n) => Some(n.0 as usize),
                _ => None,
            };
            let (skip_val, branch_val) = match (prod_idx(a), prod_idx(b)) {
                (Some(pa), Some(pb)) if pb > pa => (a, b),
                (Some(pa), Some(pb)) if pa > pb => (b, a),
                (None, Some(_)) => (a, b),
                (Some(_), None) => (b, a),
                _ => continue,
            };
            let Some(branch_root) = prod_idx(branch_val) else {
                continue;
            };
            let mut branch_nodes: HashSet<usize> = HashSet::new();
            let mut q = VecDeque::new();
            q.push_back(branch_root);
            branch_nodes.insert(branch_root);
            while let Some(nid) = q.pop_front() {
                for &inp in &graph.nodes[nid].inputs {
                    if inp == skip_val {
                        continue;
                    }
                    if let ValueKind::Node(p) = graph.value(inp).kind {
                        let pid = p.0 as usize;
                        if prod_idx(skip_val).is_none_or(|sp| pid > sp) && branch_nodes.insert(pid)
                        {
                            q.push_back(pid);
                        }
                    }
                }
            }
            let sig = branch_signature(graph, &branch_nodes, &node.name);
            if recipe.evicted_blocks.contains(&sig)
                || recipe
                    .evicted_blocks
                    .contains(&format!("add:{}", node.name))
            {
                let mut resolved_skip = skip_val;
                while let Some(&r) = val_remap.get(&resolved_skip) {
                    resolved_skip = r;
                }
                val_remap.insert(node.outputs[0], resolved_skip);
                to_remove.extend(branch_nodes);
                to_remove.insert(add_idx);
            }
        }

        if !to_remove.is_empty() {
            let resolve = |mut v: ValueId| {
                while let Some(&r) = val_remap.get(&v) {
                    v = r;
                }
                v
            };
            for node in &mut graph.nodes {
                for inp in &mut node.inputs {
                    *inp = resolve(*inp);
                }
            }
            for out in &mut graph.outputs {
                *out = resolve(*out);
            }
            let mut idx = 0usize;
            graph.nodes.retain(|_| {
                let keep = !to_remove.contains(&idx);
                idx += 1;
                keep
            });
            for (i, node) in graph.nodes.iter().enumerate() {
                for &out in &node.outputs {
                    graph.values[out.0 as usize].kind = ValueKind::Node(NodeId(i as u32));
                }
            }
        }
        crate::graph::passes::canonical::run(graph, &dummy_weights);
    }

    // Update resized weight shapes/dtypes while keeping trailing spatial axes on Conv kernels.
    for val in &mut graph.values {
        if val.kind == ValueKind::Weight {
            if let Some((base_shape, dtype)) = recipe.resized_weights.get(&val.name) {
                val.dtype = *dtype;
                if val.shape.len() > base_shape.len() && base_shape.len() == 2 {
                    val.shape[0] = base_shape[0];
                    val.shape[1] = base_shape[1];
                } else {
                    val.shape = base_shape.clone();
                }
            }
        }
    }

    // Propagate updated intermediate activation shapes in topological order.
    for i in 0..graph.nodes.len() {
        let can_infer = matches!(
            graph.nodes[i].op,
            Op::Linear
                | Op::Conv1d { .. }
                | Op::Conv2d { .. }
                | Op::SwiGlu
                | Op::GeGlu
                | Op::Unary { .. }
                | Op::Binary { .. }
                | Op::RmsNorm { .. }
                | Op::LayerNorm { .. }
                | Op::GroupNorm { .. }
                | Op::InstanceNorm { .. }
        );
        if !can_infer || graph.nodes[i].outputs.len() != 1 {
            continue;
        }
        let operands: Vec<(&[usize], DType)> = graph.nodes[i]
            .inputs
            .iter()
            .map(|&v| {
                let val = &graph.values[v.0 as usize];
                (val.shape.as_slice(), val.dtype)
            })
            .collect();
        if let Ok((new_shape, _)) = crate::graph::shape::infer(&graph.nodes[i].op, &operands) {
            let out_id = graph.nodes[i].outputs[0];
            graph.values[out_id.0 as usize].shape = new_shape;
        }
    }

    compact_live_values(graph);
}

/// Drops orphaned `Value`s (such as `kv.{l}.{k,v}` states and weights of evicted blocks) while
/// preserving topological `ValueId` order and SSA invariants.
fn compact_live_values(graph: &mut Graph) {
    let mut keep = vec![false; graph.values.len()];
    for &v in &graph.inputs {
        keep[v.0 as usize] = true;
    }
    for &v in &graph.outputs {
        keep[v.0 as usize] = true;
    }
    for n in &graph.nodes {
        for &v in &n.inputs {
            keep[v.0 as usize] = true;
        }
        for &v in &n.outputs {
            keep[v.0 as usize] = true;
        }
    }
    if keep.iter().all(|&k| k) {
        return;
    }
    let mut remap = vec![ValueId(0); graph.values.len()];
    let mut new_values = Vec::with_capacity(graph.values.len());
    for (old_idx, val) in graph.values.drain(..).enumerate() {
        if keep[old_idx] {
            remap[old_idx] = ValueId(new_values.len() as u32);
            new_values.push(val);
        }
    }
    graph.values = new_values;
    for v in &mut graph.inputs {
        *v = remap[v.0 as usize];
    }
    for v in &mut graph.outputs {
        *v = remap[v.0 as usize];
    }
    for (nid, n) in graph.nodes.iter_mut().enumerate() {
        for v in &mut n.inputs {
            *v = remap[v.0 as usize];
        }
        for v in &mut n.outputs {
            *v = remap[v.0 as usize];
            graph.values[v.0 as usize].kind = ValueKind::Node(NodeId(nid as u32));
        }
    }
}

/// Compresses `graph` and `weights` in-place under `policy` (`N >= 64`, `align >= 32`).
pub fn compress(
    graph: &mut Graph,
    weights: &mut Weights,
    policy: HopePolicy,
) -> Result<Option<HopeRecipe>, String> {
    compress_with(graph, weights, policy, 64, 32)
}

/// [`compress`] with configurable minimum width and channel alignment (for small synthetic tests).
pub fn compress_with(
    graph: &mut Graph,
    weights: &mut Weights,
    policy: HopePolicy,
    min_neurons: usize,
    min_align: usize,
) -> Result<Option<HopeRecipe>, String> {
    if !policy.is_active() {
        LAST_SAVED_PCT_X10.store(0, Ordering::Relaxed);
        LAST_BYTE_RATIO_X10000.store(10_000, Ordering::Relaxed);
        return Ok(None);
    }
    let t0 = std::time::Instant::now();
    let params_before = weights.parameter_count();
    let bytes_before = weights_resident_bytes(weights);
    if params_before == 0 {
        return Ok(None);
    }

    let mut val_readers = vec![0usize; graph.values.len()];
    let mut weight_readers: HashMap<&str, usize> = HashMap::new();
    for node in &graph.nodes {
        for &inp in &node.inputs {
            val_readers[inp.0 as usize] += 1;
            if graph.value(inp).kind == ValueKind::Weight {
                *weight_readers
                    .entry(graph.value(inp).name.as_str())
                    .or_insert(0) += 1;
            }
        }
    }
    for &out in &graph.outputs {
        val_readers[out.0 as usize] += 1;
    }

    let groups = discover_groups(
        graph,
        weights,
        &val_readers,
        &weight_readers,
        min_neurons,
        min_align,
    );
    let blocks = discover_residual_blocks(graph, weights, &groups);
    if groups.is_empty() && blocks.is_empty() {
        return Ok(None);
    }

    // Profiles decode every group's weights: one pool task per group.
    let slots: Vec<std::sync::Mutex<Option<GroupProfile>>> =
        groups.iter().map(|_| Default::default()).collect();
    crate::model::par::par_for(groups.len(), &|gi| {
        *slots[gi].lock().unwrap() = build_group_profile(&groups[gi], weights);
    });
    let Some(mut profiles) = slots
        .into_iter()
        .map(|s| s.into_inner().unwrap())
        .collect::<Option<Vec<GroupProfile>>>()
    else {
        return Ok(None);
    };

    // Total compressible parameter pool P_0 (avoid double-counting groups nested inside macro blocks).
    let mut group_in_block = vec![false; groups.len()];
    for b in &blocks {
        for &gi in &b.group_indices {
            group_in_block[gi] = true;
        }
    }
    let p0: usize = blocks.iter().map(|b| b.total_params).sum::<usize>()
        + groups
            .iter()
            .enumerate()
            .filter(|&(gi, _)| !group_in_block[gi])
            .map(|(_, g)| g.total_params())
            .sum::<usize>();
    if p0 == 0 {
        return Ok(None);
    }
    let target_remove = ((1.0 - policy.density() as f64) * (p0 as f64)).round() as usize;
    if target_remove == 0 {
        return Ok(None);
    }

    let mut removed_params = 0usize;
    let mut block_evicted = vec![false; blocks.len()];

    // Receding-horizon Dantzig knapsack loop (`DR = J / delta_P_init`).
    while removed_params < target_remove {
        // 1. Best macro residual block eviction candidate.
        let mut best_block: Option<(usize, f64)> = None;
        for (bi, b) in blocks.iter().enumerate() {
            if block_evicted[bi] {
                continue;
            }
            let num: f64 = if b.group_indices.is_empty() {
                b.extra_energy
            } else {
                b.group_indices
                    .iter()
                    .filter(|&&gi| !profiles[gi].evicted)
                    .map(|&gi| {
                        let m = profiles[gi].n_live as f64;
                        m * m * profiles[gi].e_live
                    })
                    .sum::<f64>()
                    + b.extra_energy
            };
            let j_evict = num / b.identity_energy;
            let dr = j_evict / (b.total_params as f64);
            if best_block.is_none_or(|(_, best_dr)| dr < best_dr) {
                best_block = Some((bi, dr));
            }
        }

        // 2. Collect granular prune + merge candidates across live groups with room above `align`.
        let mut actions: Vec<GranularAction> = Vec::new();
        for (gi, g) in groups.iter().enumerate() {
            let prof = &profiles[gi];
            if prof.evicted || prof.n_live <= g.align {
                continue;
            }
            let boundary_mult = if groups.len() >= 8 && (gi < 2 || gi + 2 >= groups.len()) {
                4.0
            } else {
                1.0
            };
            let dp = (g.params_per_neuron() as f64) / boundary_mult;
            for i in 0..g.n {
                if !prof.alive[i] || prof.used_in_merge[i] {
                    continue;
                }
                let j_p = prune_distortion(prof.n_live, prof.e_live, prof.caps[i] as f64);
                actions.push(GranularAction::Prune {
                    g: gi,
                    i,
                    dr: j_p / dp,
                });
            }
            for mc in &prof.merges {
                let (i, j) = (mc.i as usize, mc.j as usize);
                if !prof.alive[i]
                    || !prof.alive[j]
                    || prof.used_in_merge[i]
                    || prof.used_in_merge[j]
                {
                    continue;
                }
                let e_rem =
                    (prof.e_live - (prof.caps[i] as f64) - (prof.caps[j] as f64)).max(1e-12);
                let (j_m, s_star) = merge_distortion(prof.n_live, e_rem, mc.a as f64, mc.b as f64);
                actions.push(GranularAction::Merge {
                    g: gi,
                    i,
                    j,
                    c_ker: mc.c_ker,
                    s_star: s_star as f32,
                    dr: j_m / dp,
                });
            }
        }

        let batch_limit = MICRO_BATCH * groups.len().max(1);
        let part = (batch_limit * 2).min(actions.len());
        if part > 0 && part < actions.len() {
            actions.select_nth_unstable_by(part, |a, b| a.dr().total_cmp(&b.dr()));
            actions.truncate(part);
        }
        actions.sort_unstable_by(|a, b| a.dr().total_cmp(&b.dr()));
        let best_granular_dr = actions.first().map(GranularAction::dr);

        match (best_block, best_granular_dr) {
            (None, None) => break,
            (Some((bi, _)), None) => {
                block_evicted[bi] = true;
                for &gi in &blocks[bi].group_indices {
                    profiles[gi].evicted = true;
                }
                removed_params += blocks[bi].total_params;
            }
            (Some((bi, b_dr)), Some(g_dr)) if b_dr <= g_dr => {
                block_evicted[bi] = true;
                for &gi in &blocks[bi].group_indices {
                    if !profiles[gi].evicted {
                        let already_removed =
                            (groups[gi].n - profiles[gi].n_live) * groups[gi].params_per_neuron();
                        removed_params = removed_params.saturating_sub(already_removed);
                        profiles[gi].evicted = true;
                    }
                }
                removed_params += blocks[bi].total_params;
            }
            (_, Some(_)) => {
                let mut touched: HashSet<(usize, usize)> = HashSet::with_capacity(batch_limit * 2);
                let mut executed = 0usize;
                for act in actions {
                    if executed >= batch_limit || removed_params >= target_remove {
                        break;
                    }
                    match act {
                        GranularAction::Prune { g, i, .. } => {
                            if profiles[g].n_live <= groups[g].align || !touched.insert((g, i)) {
                                continue;
                            }
                            let p = &mut profiles[g];
                            p.alive[i] = false;
                            p.n_live -= 1;
                            p.e_live = (p.e_live - p.caps[i] as f64).max(1e-12);
                            removed_params += groups[g].params_per_neuron();
                            executed += 1;
                        }
                        GranularAction::Merge {
                            g,
                            i,
                            j,
                            c_ker,
                            s_star,
                            ..
                        } => {
                            if profiles[g].n_live <= groups[g].align
                                || touched.contains(&(g, i))
                                || touched.contains(&(g, j))
                            {
                                continue;
                            }
                            touched.insert((g, i));
                            touched.insert((g, j));
                            let p = &mut profiles[g];
                            let new_cap = (std::f32::consts::SQRT_2 * s_star).max(0.0);
                            p.e_live = (p.e_live - (p.caps[i] as f64) - (p.caps[j] as f64)
                                + (new_cap as f64))
                                .max(1e-12);
                            p.alive[j] = false;
                            p.caps[i] = new_cap;
                            p.merged_into[i] = Some((j, c_ker, s_star));
                            p.used_in_merge[i] = true;
                            p.used_in_merge[j] = true;
                            p.n_live -= 1;
                            removed_params += groups[g].params_per_neuron();
                            executed += 1;
                        }
                    }
                }
                if executed == 0 {
                    if let Some((bi, _)) = best_block {
                        block_evicted[bi] = true;
                        for &gi in &blocks[bi].group_indices {
                            profiles[gi].evicted = true;
                        }
                        removed_params += blocks[bi].total_params;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    let mut recipe = HopeRecipe {
        params_before,
        ..Default::default()
    };

    for (bi, b) in blocks.iter().enumerate() {
        if block_evicted[bi] {
            recipe.evicted_blocks.insert(b.sig.clone());
            recipe.evicted_blocks.insert(format!("add:{}", b.add_name));
            recipe.evicted_block_count += 1;
        }
    }

    // Synthesis decodes and re-encodes each group's weights: one pool task per group, inserted after.
    type Synth = Result<(usize, usize, Vec<(String, Tensor, Vec<usize>, DType)>), String>;
    let cells: Vec<std::sync::Mutex<(&mut GroupProfile, Option<Synth>)>> = profiles
        .iter_mut()
        .map(|p| std::sync::Mutex::new((p, None)))
        .collect();
    let shared: &Weights = weights;
    crate::model::par::par_for(groups.len(), &|gi| {
        let mut cell = cells[gi].lock().unwrap();
        if cell.0.evicted {
            return;
        }
        let mut out = Vec::new();
        let r = synthesize_group(&groups[gi], cell.0, shared, &mut out);
        cell.1 = Some(r.map(|(p, m)| (p, m, out)));
    });
    for cell in cells {
        let Some(r) = cell.into_inner().unwrap().1 else {
            continue;
        };
        let (pruned, merged, out) = r?;
        for (name, t, shape, dt) in out {
            weights.insert(name.clone(), t);
            recipe.resized_weights.insert(name, (shape, dt));
        }
        recipe.pruned_neurons += pruned;
        recipe.merged_neurons += merged;
    }

    if recipe.resized_weights.is_empty() && recipe.evicted_blocks.is_empty() {
        return Ok(None);
    }

    apply_recipe(graph, &recipe);

    // Drop weights left completely unreferenced in `graph` (keeps 1-element scalar constants).
    let live_weights: HashSet<String> = graph
        .values
        .iter()
        .filter(|v| v.kind == ValueKind::Weight)
        .map(|v| v.name.clone())
        .collect();
    let dead_names: Vec<String> = weights
        .iter()
        .filter(|&(name, t)| t.len() > 1 && !live_weights.contains(name))
        .map(|(name, _)| name.to_string())
        .collect();
    for name in dead_names {
        weights.remove(&name);
    }

    recipe.params_after = weights.parameter_count();
    let bytes_after = weights_resident_bytes(weights);
    let saved_pct = if params_before > 0 {
        (1.0 - (recipe.params_after as f64) / (params_before as f64)).max(0.0) * 100.0
    } else {
        0.0
    };
    let byte_ratio = if bytes_before > 0 {
        ((bytes_after as f64) / (bytes_before as f64)).clamp(0.1, 1.0)
    } else {
        1.0
    };
    LAST_SAVED_PCT_X10.store((saved_pct * 10.0).round() as u32, Ordering::Relaxed);
    LAST_BYTE_RATIO_X10000.store((byte_ratio * 10_000.0).round() as u32, Ordering::Relaxed);
    let msg = format!(
        "hope ({policy}): {:.1}% params saved ({} -> {}, resident {:.1} MB -> {:.1} MB), {} pruned, {} merged, {} blocks evicted in {:.1} ms",
        saved_pct,
        recipe.params_before,
        recipe.params_after,
        bytes_before as f64 / 1e6,
        bytes_after as f64 / 1e6,
        recipe.pruned_neurons,
        recipe.merged_neurons,
        recipe.evicted_block_count,
        t0.elapsed().as_secs_f64() * 1e3
    );
    tracing::info!("{msg}");
    if let Some(f) = &crate::model::weight_format::ReencodeConfig::effective().log {
        f(&msg);
    }
    Ok(Some(recipe))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::cpu;
    use crate::graph::pattern::validate_ssa;
    use crate::graph::StateStore;

    fn rng_vec(seed: u64, len: usize, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                (((z ^ (z >> 31)) >> 40) as f32 / ((1u64 << 24) as f32) * 2.0 - 1.0) * scale
            })
            .collect()
    }

    #[test]
    fn scale_invariance_and_collinear_merge_exactness() {
        let (d_in, n, d_out) = (16usize, 8usize, 12usize);
        let lam = 3.7f32;
        let norm_g: Vec<f32> = rng_vec(10, d_in, 0.4)
            .into_iter()
            .map(|v| 1.0 + v)
            .collect();

        // 1a. Plain ReLU scale invariance `(w_in * λ, w_out / λ)`
        let build_plain = |scale: f32| {
            let mut g = Graph::new();
            let mut w = Weights::new();
            let x = g.input("x", vec![4, d_in], DType::F32);
            let nw = g.weight("norm.w", vec![d_in], DType::F32);
            let xn = g.add(
                "rn",
                Op::RmsNorm { eps: 1e-6 },
                &[x, nw],
                vec![4, d_in],
                DType::F32,
            );
            let wi = g.weight("w_in", vec![n, d_in], DType::F32);
            let h = g.add("lin_in", Op::Linear, &[xn, wi], vec![4, n], DType::F32);
            let a = g.add(
                "relu",
                Op::Unary {
                    kind: UnaryKind::Relu,
                },
                &[h],
                vec![4, n],
                DType::F32,
            );
            let wo = g.weight("w_out", vec![d_out, n], DType::F32);
            let y = g.add("lin_out", Op::Linear, &[a, wo], vec![4, d_out], DType::F32);
            g.output(y);
            w.insert("norm.w", Tensor::F32(norm_g.clone()));
            w.insert(
                "w_in",
                Tensor::F32(
                    rng_vec(1, n * d_in, 0.5)
                        .into_iter()
                        .map(|v| v * scale)
                        .collect(),
                ),
            );
            w.insert(
                "w_out",
                Tensor::F32(
                    rng_vec(2, d_out * n, 0.5)
                        .into_iter()
                        .map(|v| v / scale)
                        .collect(),
                ),
            );
            (g, w)
        };
        let (g1, w1) = build_plain(1.0);
        let (g2, w2) = build_plain(lam);
        let vr = vec![1usize; g1.values.len()];
        let wr = HashMap::from([("norm.w", 1usize), ("w_in", 1), ("w_out", 1)]);
        let grp1 = &discover_groups(&g1, &w1, &vr, &wr, 2, 1)[0];
        let grp2 = &discover_groups(&g2, &w2, &vr, &wr, 2, 1)[0];
        let p1 = build_group_profile(grp1, &w1).unwrap();
        let p2 = build_group_profile(grp2, &w2).unwrap();
        for i in 0..n {
            assert!((p1.caps[i] - p2.caps[i]).abs() < 1e-5);
            let jp1 = prune_distortion(n, p1.e_live, p1.caps[i] as f64);
            let jp2 = prune_distortion(n, p2.e_live, p2.caps[i] as f64);
            assert!((jp1 - jp2).abs() < 1e-5);
        }
        assert_eq!(p1.merges.len(), p2.merges.len());
        for (m1, m2) in p1.merges.iter().zip(&p2.merges) {
            let er1 = p1.e_live - (p1.caps[m1.i as usize] as f64) - (p1.caps[m1.j as usize] as f64);
            let er2 = p2.e_live - (p2.caps[m2.i as usize] as f64) - (p2.caps[m2.j as usize] as f64);
            let (jm1, _) = merge_distortion(n, er1, m1.a as f64, m1.b as f64);
            let (jm2, _) = merge_distortion(n, er2, m2.a as f64, m2.b as f64);
            assert!((jm1 - jm2).abs() < 1e-5, "plain merge J {jm1} vs {jm2}");
        }

        // 1b. GLU `(w_up * λ, w_down / λ)` scale invariance + exact collinear 2-neuron -> 1-neuron merge
        let build_glu = |up_scale: f32, collinear_pair: bool| {
            let nn = if collinear_pair { 2 } else { n };
            let mut g = Graph::new();
            let mut w = Weights::new();
            let x = g.input("x", vec![6, d_in], DType::F32);
            let nw = g.weight("norm.w", vec![d_in], DType::F32);
            let xn = g.add(
                "rn",
                Op::RmsNorm { eps: 1e-6 },
                &[x, nw],
                vec![6, d_in],
                DType::F32,
            );
            let wg = g.weight("w_gate", vec![nn, d_in], DType::F32);
            let wu = g.weight("w_up", vec![nn, d_in], DType::F32);
            let gv = g.add("gate", Op::Linear, &[xn, wg], vec![6, nn], DType::F32);
            let uv = g.add("up", Op::Linear, &[xn, wu], vec![6, nn], DType::F32);
            let sg = g.add("swiglu", Op::SwiGlu, &[gv, uv], vec![6, nn], DType::F32);
            let wd = g.weight("w_down", vec![d_out, nn], DType::F32);
            let y = g.add("down", Op::Linear, &[sg, wd], vec![6, d_out], DType::F32);
            g.output(y);
            w.insert("norm.w", Tensor::F32(norm_g.clone()));
            let mut gate_data = rng_vec(11, nn * d_in, 0.5);
            let mut up_data = rng_vec(12, nn * d_in, 0.5);
            if collinear_pair {
                gate_data.copy_within(0..d_in, d_in);
                for k in 0..d_in {
                    up_data[d_in + k] = 2.5 * up_data[k];
                }
            }
            w.insert("w_gate", Tensor::F32(gate_data));
            w.insert(
                "w_up",
                Tensor::F32(up_data.into_iter().map(|v| v * up_scale).collect()),
            );
            w.insert(
                "w_down",
                Tensor::F32(
                    rng_vec(13, d_out * nn, 0.5)
                        .into_iter()
                        .map(|v| v / up_scale)
                        .collect(),
                ),
            );
            (g, w)
        };
        let (gg1, gw1) = build_glu(1.0, false);
        let (gg2, gw2) = build_glu(lam, false);
        let gvr = vec![1usize; gg1.values.len()];
        let gwr = HashMap::from([
            ("norm.w", 1usize),
            ("w_gate", 1),
            ("w_up", 1),
            ("w_down", 1),
        ]);
        let ggrp1 = &discover_groups(&gg1, &gw1, &gvr, &gwr, 2, 1)[0];
        let ggrp2 = &discover_groups(&gg2, &gw2, &gvr, &gwr, 2, 1)[0];
        let gp1 = build_group_profile(ggrp1, &gw1).unwrap();
        let gp2 = build_group_profile(ggrp2, &gw2).unwrap();
        for i in 0..n {
            assert!((gp1.caps[i] - gp2.caps[i]).abs() < 1e-5);
        }
        for (m1, m2) in gp1.merges.iter().zip(&gp2.merges) {
            let er1 =
                gp1.e_live - (gp1.caps[m1.i as usize] as f64) - (gp1.caps[m1.j as usize] as f64);
            let er2 =
                gp2.e_live - (gp2.caps[m2.i as usize] as f64) - (gp2.caps[m2.j as usize] as f64);
            let (jm1, _) = merge_distortion(n, er1, m1.a as f64, m1.b as f64);
            let (jm2, _) = merge_distortion(n, er2, m2.a as f64, m2.b as f64);
            assert!((jm1 - jm2).abs() < 1e-5, "glu merge J {jm1} vs {jm2}");
        }

        // Exact collinear 2-neuron -> 1-neuron merge reproduces exact output across random inputs.
        let (mut cg, mut cw) = build_glu(1.0, true);
        let x_in = Tensor::F32(rng_vec(99, 6 * d_in, 1.0));
        let ref_out = cpu::run(&cg, &cw, &mut StateStore::new(), vec![x_in.clone()]).unwrap()[0]
            .as_f32()
            .unwrap()
            .to_vec();
        let recipe = compress_with(&mut cg, &mut cw, HopePolicy::Ratio(0.50), 2, 1)
            .unwrap()
            .expect("merged collinear pair");
        assert_eq!(recipe.merged_neurons, 1);
        assert_eq!(cw.get("w_gate").unwrap().len(), d_in); // N reduced from 2 to 1!
        validate_ssa(&cg).unwrap();
        let got_out = cpu::run(&cg, &cw, &mut StateStore::new(), vec![x_in]).unwrap()[0]
            .as_f32()
            .unwrap()
            .to_vec();
        for (a, b) in ref_out.iter().zip(&got_out) {
            assert!((a - b).abs() < 1e-5, "collinear merge mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn macro_block_eviction_removes_zero_capacity_residual_and_kv_state() {
        let (m, h, heads, hd, ctx) = (4usize, 64usize, 4usize, 16usize, 32usize);
        let mut g = Graph::new();
        let mut w = Weights::new();
        let mut x = g.input("x", vec![m, h], DType::F32);
        let pos = g.input("positions", vec![m], DType::U32);

        for l in 0..3 {
            let scale = if l == 1 { 1e-7 } else { 0.25 };
            let nw = g.weight(format!("l{l}.norm.w"), vec![h], DType::F32);
            let xn = g.add(
                format!("l{l}.rn"),
                Op::RmsNorm { eps: 1e-5 },
                &[x, nw],
                vec![m, h],
                DType::F32,
            );
            let wq = g.weight(format!("l{l}.wq"), vec![h, h], DType::F32);
            let wk = g.weight(format!("l{l}.wk"), vec![h, h], DType::F32);
            let wv = g.weight(format!("l{l}.wv"), vec![h, h], DType::F32);
            let wo = g.weight(format!("l{l}.wo"), vec![h, h], DType::F32);
            let q = g.add(
                format!("l{l}.q"),
                Op::Linear,
                &[xn, wq],
                vec![m, h],
                DType::F32,
            );
            let k = g.add(
                format!("l{l}.k"),
                Op::Linear,
                &[xn, wk],
                vec![m, h],
                DType::F32,
            );
            let v = g.add(
                format!("l{l}.v"),
                Op::Linear,
                &[xn, wv],
                vec![m, h],
                DType::F32,
            );
            let ks = g.state(format!("kv.{l}.k"), vec![ctx, h], DType::F32, false);
            let vs = g.state(format!("kv.{l}.v"), vec![ctx, h], DType::F32, false);
            let att = g.add(
                format!("l{l}.attn"),
                Op::CachedAttention {
                    heads,
                    kv_heads: heads,
                    head_dim: hd,
                    v_head_dim: hd,
                    scale: 0.25,
                    causal: true,
                    window: None,
                    soft_cap: None,
                    write: true,
                },
                &[q, k, v, ks, vs, pos],
                vec![m, h],
                DType::F32,
            );
            let o = g.add(
                format!("l{l}.o"),
                Op::Linear,
                &[att, wo],
                vec![m, h],
                DType::F32,
            );
            x = g.add(
                format!("l{l}.add"),
                Op::Binary {
                    kind: BinaryKind::Add,
                },
                &[x, o],
                vec![m, h],
                DType::F32,
            );

            w.insert(format!("l{l}.norm.w"), Tensor::F32(vec![1.0; h]));
            w.insert(
                format!("l{l}.wq"),
                Tensor::F32(rng_vec(100 + l as u64, h * h, 0.25)),
            );
            w.insert(
                format!("l{l}.wk"),
                Tensor::F32(rng_vec(200 + l as u64, h * h, 0.25)),
            );
            w.insert(
                format!("l{l}.wv"),
                Tensor::F32(rng_vec(300 + l as u64, h * h, 0.25)),
            );
            w.insert(
                format!("l{l}.wo"),
                Tensor::F32(rng_vec(400 + l as u64, h * h, scale)),
            );
        }
        g.output(x);

        let x_in = Tensor::F32(rng_vec(777, m * h, 0.5));
        let pos_in = Tensor::U32((0..m as u32).collect());
        let ref_out = cpu::run(
            &g,
            &w,
            &mut StateStore::new(),
            vec![x_in.clone(), pos_in.clone()],
        )
        .unwrap()[0]
            .as_f32()
            .unwrap()
            .to_vec();

        let recipe = compress(&mut g, &mut w, HopePolicy::Light)
            .unwrap()
            .expect("evicted middle residual block");
        assert_eq!(recipe.evicted_block_count, 1);
        assert!(w.get("l1.wo").is_none() && w.get("l1.wq").is_none());
        let paged = crate::graph::paged_states(&g);
        assert!(paged
            .iter()
            .all(|s| s.name != "kv.1.k" && s.name != "kv.1.v"));
        assert!(g
            .values
            .iter()
            .all(|v| v.name != "kv.1.k" && v.name != "kv.1.v"));
        validate_ssa(&g).unwrap();

        let got_out = cpu::run(&g, &w, &mut StateStore::new(), vec![x_in, pos_in]).unwrap()[0]
            .as_f32()
            .unwrap()
            .to_vec();
        for (a, b) in ref_out.iter().zip(&got_out) {
            assert!((a - b).abs() < 1e-4, "eviction output drift: {a} vs {b}");
        }
    }
}
