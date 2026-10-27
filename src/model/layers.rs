use crate::hardware::QuantFormat;
use crate::model::{f16_to_f32, f32_to_bf16, f32_to_f16};

/// Encodes row-major `[out_features, in_features]` f32 `weights` into `format`.
pub fn encode_rows(
    weights: &[f32],
    in_features: usize,
    format: QuantFormat,
) -> Result<Vec<u8>, String> {
    if in_features == 0 || weights.len() % in_features != 0 {
        return Err(format!(
            "weight length {} is not a multiple of in_features {in_features}",
            weights.len()
        ));
    }
    match format {
        QuantFormat::GGUF_Q4_0 => Ok(quantize_f32_to_q4_0(weights)),
        QuantFormat::GGUF_NVFP4 => {
            if in_features % 64 != 0 {
                return Err(format!(
                    "{format:?} needs in_features % 64 == 0, got {in_features}"
                ));
            }
            Ok(quantize_f32_to_gguf_nvfp4(weights))
        }
        QuantFormat::GGUF_Q8_0 => Ok(quantize_f32_to_q8_0(weights)),
        QuantFormat::GGUF_Q3_K
        | QuantFormat::GGUF_Q4_K_M
        | QuantFormat::GGUF_Q5_K_M
        | QuantFormat::GGUF_Q6_K => {
            if in_features % 256 != 0 {
                return Err(format!(
                    "{format:?} needs in_features % 256 == 0, got {in_features}"
                ));
            }
            let (bytes, row): (usize, fn(&[f32], &mut [u8])) = match format {
                QuantFormat::GGUF_Q3_K => (110, quantize_row_q3_k_ref),
                QuantFormat::GGUF_Q4_K_M => (144, quantize_row_q4_k_ref),
                QuantFormat::GGUF_Q5_K_M => (176, quantize_row_q5_k_ref),
                _ => (210, quantize_row_q6_k_ref),
            };
            Ok(quantize_k_rows(weights, in_features, bytes, row))
        }
        QuantFormat::FP16 => Ok(weights
            .iter()
            .flat_map(|&w| f32_to_f16(w).to_le_bytes())
            .collect()),
        QuantFormat::BF16 => Ok(weights
            .iter()
            .flat_map(|&w| f32_to_bf16(w).to_le_bytes())
            .collect()),
        QuantFormat::FP32 => Ok(weights.iter().flat_map(|&w| w.to_le_bytes()).collect()),
        _ => Err(format!("cannot encode as {format:?}")),
    }
}

/// Dequantizes one GGUF block into `y` in ggml `dequantize_row_*` element order.
pub type GgufBlockFn = fn(&[u8], &mut [f32]);

/// `(elements, bytes, dequantizer)` per block for every GGUF block format the engine decodes.
pub fn gguf_block_codec(format: QuantFormat) -> Option<(usize, usize, GgufBlockFn)> {
    let c = crate::graph::codec::codec(format)?;
    let dequant: GgufBlockFn = match format {
        QuantFormat::GGUF_Q4_0 => dequantize_q4_0_block,
        QuantFormat::GGUF_Q4_1 => dequantize_q4_1_block,
        QuantFormat::GGUF_Q5_0 => dequantize_q5_0_block,
        QuantFormat::GGUF_Q5_1 => dequantize_q5_1_block,
        QuantFormat::GGUF_Q8_0 => dequantize_q8_0_block,
        QuantFormat::GGUF_Q4_K_M => dequantize_q4_k_block,
        QuantFormat::GGUF_Q5_K_M => dequantize_q5_k_block,
        QuantFormat::GGUF_Q6_K => dequantize_q6_k_block,
        QuantFormat::GGUF_Q2_K => dequantize_q2_k_block,
        QuantFormat::GGUF_Q3_K => dequantize_q3_k_block,
        QuantFormat::GGUF_IQ4_NL => dequantize_iq4_nl_block,
        QuantFormat::GGUF_IQ4_XS => dequantize_iq4_xs_block,
        QuantFormat::GGUF_IQ3_XXS => dequantize_iq3_xxs_block,
        QuantFormat::GGUF_IQ3_S => dequantize_iq3_s_block,
        QuantFormat::GGUF_IQ2_XXS => dequantize_iq2_xxs_block,
        QuantFormat::GGUF_IQ2_XS => dequantize_iq2_xs_block,
        QuantFormat::GGUF_IQ2_S | QuantFormat::GGUF_IQ2_M => dequantize_iq2_s_block,
        QuantFormat::GGUF_IQ1_S => dequantize_iq1_s_block,
        QuantFormat::GGUF_IQ1_M => dequantize_iq1_m_block,
        QuantFormat::GGUF_TQ1_0 => dequantize_tq1_0_block,
        QuantFormat::GGUF_TQ2_0 => dequantize_tq2_0_block,
        QuantFormat::MXFP4 => dequantize_mxfp4_block,
        QuantFormat::GGUF_NVFP4 => dequantize_nvfp4_block,
        _ => return crate::model::repack::block_codec(format),
    };
    Some((c.block_elems as usize, c.block_bytes as usize, dequant))
}

/// Dequantizes whole GGUF blocks of `format` to f32; `None` if `format` is not a block format.
pub fn dequantize_gguf_blocks(format: QuantFormat, raw: &[u8]) -> Option<Vec<f32>> {
    let (n, bytes, dequant) = gguf_block_codec(format)?;
    let mut out = vec![0.0f32; raw.len() / bytes * n];
    for (block, y) in raw.chunks_exact(bytes).zip(out.chunks_exact_mut(n)) {
        dequant(block, y);
    }
    Some(out)
}

#[inline]
fn f16_at(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

/// Q4_0 (18 B / 32): `d`, then low nibbles = elements 0..16, high nibbles = 16..32.
pub fn dequantize_q4_0_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for (i, &q) in b[2..18].iter().enumerate() {
        y[i] = ((q & 0x0F) as i32 - 8) as f32 * d;
        y[i + 16] = ((q >> 4) as i32 - 8) as f32 * d;
    }
}

/// Q4_1 (20 B / 32): `d`, `m`, then Q4_0 nibble order; value = q * d + m.
pub fn dequantize_q4_1_block(b: &[u8], y: &mut [f32]) {
    let (d, m) = (f16_at(b, 0), f16_at(b, 2));
    for (i, &q) in b[4..20].iter().enumerate() {
        y[i] = (q & 0x0F) as f32 * d + m;
        y[i + 16] = (q >> 4) as f32 * d + m;
    }
}

/// 5th bits of element `i` / `i + 16` from a Q5_0/Q5_1 `qh` word.
#[inline]
fn q5_high_bits(qh: u32, i: usize) -> (u8, u8) {
    (
        (((qh >> i) & 1) << 4) as u8,
        (((qh >> (i + 16)) & 1) << 4) as u8,
    )
}

/// Q5_0 (22 B / 32): `d`, 32-bit `qh`, Q4_0 nibble order; value = (q | h << 4) - 16.
pub fn dequantize_q5_0_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    for (i, &q) in b[6..22].iter().enumerate() {
        let (h0, h1) = q5_high_bits(qh, i);
        y[i] = (((q & 0x0F) | h0) as i32 - 16) as f32 * d;
        y[i + 16] = (((q >> 4) | h1) as i32 - 16) as f32 * d;
    }
}

/// Q5_1 (24 B / 32): `d`, `m`, 32-bit `qh`, Q4_0 nibble order; value = (q | h << 4) * d + m.
pub fn dequantize_q5_1_block(b: &[u8], y: &mut [f32]) {
    let (d, m) = (f16_at(b, 0), f16_at(b, 2));
    let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    for (i, &q) in b[8..24].iter().enumerate() {
        let (h0, h1) = q5_high_bits(qh, i);
        y[i] = ((q & 0x0F) | h0) as f32 * d + m;
        y[i + 16] = ((q >> 4) | h1) as f32 * d + m;
    }
}

/// Q8_0 (34 B / 32): `d`, then 32 signed bytes.
pub fn dequantize_q8_0_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for (yi, &q) in y.iter_mut().zip(&b[2..34]) {
        *yi = q as i8 as f32 * d;
    }
}

/// ggml `get_scale_min_k4`: unpacks 6-bit (scale, min) pair `j` from a K-quant block's 12 scale bytes.
#[inline]
pub(crate) fn k4_scale_min(j: usize, s: &[u8]) -> (f32, f32) {
    if j < 4 {
        ((s[j] & 63) as f32, (s[j + 4] & 63) as f32)
    } else {
        (
            ((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4)) as f32,
            ((s[j + 4] >> 4) | ((s[j] >> 6) << 4)) as f32,
        )
    }
}

/// ggml `nearest_int`: round to nearest, ties to even, via the 1.5·2²³ magic number.
#[inline]
fn nearest_int(x: f32) -> i32 {
    (((x + 12582912.0).to_bits() as i32) & 0x007f_ffff) - 0x0040_0000
}

/// ggml `make_qkx2_quants` (n = x.len() ≤ 32): scale and min of the best affine grid `scale·L - min`
/// with `L ∈ 0..=nmax`, searched over `nstep + 1` candidate inverse scales by weighted squared error.
fn make_qkx2_quants(
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
) -> (f32, f32) {
    let n = x.len();
    let (mut min, mut max) = (x[0], x[0]);
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        min = min.min(x[i]);
        max = max.max(x[i]);
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l[..n].fill(0);
        return (0.0, -min);
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = q as u8;
        let diff = scale * l[i] as f32 + min - x[i];
        best_error += weights[i] * (diff * diff);
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = q as u8;
            let (w, qf) = (weights[i], q as f32);
            sum_l += w * qf;
            sum_l2 += w * qf * qf;
            sum_xl += w * qf * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let diff = this_scale * laux[i] as f32 + this_min - x[i];
                cur_error += weights[i] * (diff * diff);
            }
            if cur_error < best_error {
                l[..n].copy_from_slice(&laux[..n]);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// ggml `make_qx_quants` with `rmse_type = 1` (weights x²) and no importance weights, as Q6_K calls it.
fn make_qx_quants(nmax: i32, x: &[f32], l: &mut [i8]) -> f32 {
    let n = x.len();
    let (mut max, mut amax) = (0.0f32, 0.0f32);
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    if amax < 1e-15 {
        l[..n].fill(0);
        return 0.0;
    }
    let mut iscale = -(nmax as f32) / max;
    let (mut sumlx, mut suml2) = (0.0f32, 0.0f32);
    for i in 0..n {
        let q = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        l[i] = (q + nmax) as i8;
        let w = x[i] * x[i];
        sumlx += w * x[i] * q as f32;
        suml2 += w * q as f32 * q as f32;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;
    for is in -9..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1 * is as f32) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for i in 0..n {
            let q = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
            let w = x[i] * x[i];
            sumlx += w * x[i] * q as f32;
            suml2 += w * q as f32 * q as f32;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                l[i] = (nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1) + nmax) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// The affine K-quant search shared by Q4_K (`nmax` 15) and Q5_K (31): per 32-element sub-block a
/// weighted least-squares (scale, min), then 6-bit packing of the eight scales and mins and the
/// final levels `L`. Returns the packed 12 scale bytes, `d`, `dmin` as f16 bits and the levels.
fn k_quant_affine(
    x: &[f32],
    nmax: i32,
    rmin: f32,
    nstep: i32,
    l: &mut [u8; 256],
) -> ([u8; 12], u16, u16) {
    let (mut scales, mut mins) = ([0.0f32; 8], [0.0f32; 8]);
    let (mut laux, mut weights) = ([0u8; 32], [0.0f32; 32]);
    let (mut max_scale, mut max_min) = (0.0f32, 0.0f32);
    for j in 0..8 {
        let xs = &x[32 * j..32 * j + 32];
        let sum_x2: f32 = xs.iter().fold(0.0, |a, &v| a + v * v);
        let av_x = (sum_x2 / 32.0).sqrt();
        for (w, &v) in weights.iter_mut().zip(xs) {
            *w = av_x + v.abs();
        }
        let (s, m) = make_qkx2_quants(
            nmax,
            xs,
            &weights,
            &mut l[32 * j..32 * j + 32],
            &mut laux,
            rmin,
            0.1,
            nstep,
        );
        (scales[j], mins[j]) = (s, m);
        max_scale = max_scale.max(s);
        max_min = max_min.max(m);
    }
    let inv_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let mut packed = [0u8; 12];
    for j in 0..8 {
        let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
        let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
        if j < 4 {
            packed[j] = ls;
            packed[j + 4] = lm;
        } else {
            packed[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            packed[j - 4] |= (ls >> 4) << 6;
            packed[j] |= (lm >> 4) << 6;
        }
    }
    let (d16, dmin16) = (f32_to_f16(max_scale / 63.0), f32_to_f16(max_min / 63.0));
    let (d_f, dmin_f) = (f16_to_f32(d16), f16_to_f32(dmin16));
    for j in 0..8 {
        let (sc, m) = k4_scale_min(j, &packed);
        let d = d_f * sc;
        if d == 0.0 {
            continue;
        }
        let dm = dmin_f * m;
        for ii in 0..32 {
            l[32 * j + ii] = nearest_int((x[32 * j + ii] + dm) / d).clamp(0, nmax) as u8;
        }
    }
    (packed, d16, dmin16)
}

/// ggml `make_q3_quants` (`do_rmse = true`) for one 16-element sub-block (`nmax = 4`).
fn make_q3_quants(nmax: i32, x: &[f32], l: &mut [i8]) -> f32 {
    let n = x.len();
    let (mut max, mut amax) = (0.0f32, 0.0f32);
    for &v in x {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax < 1e-15 {
        l[..n].fill(0);
        return 0.0;
    }
    let iscale = -(nmax as f32) / max;
    let (mut sumlx, mut suml2) = (0.0f32, 0.0f32);
    for i in 0..n {
        let q = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        l[i] = q as i8;
        let w = x[i] * x[i];
        sumlx += w * x[i] * q as f32;
        suml2 += w * q as f32 * q as f32;
    }
    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = x[i] * x[i];
            let li = l[i] as i32;
            let mut slx = sumlx - w * x[i] * li as f32;
            if slx > 0.0 {
                let mut sl2 = suml2 - w * li as f32 * li as f32;
                let new_l = nearest_int(x[i] * sl2 / slx).clamp(-nmax, nmax - 1);
                if new_l != li {
                    slx += w * x[i] * new_l as f32;
                    sl2 += w * new_l as f32 * new_l as f32;
                    if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                        l[i] = new_l as i8;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    for v in &mut l[..n] {
        *v += nmax as i8;
    }
    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

/// ggml `quantize_row_q3_K_ref` for one 256-element row: 110-byte block `hmask[32], qs[64], scales[12], d`.
pub fn quantize_row_q3_k_ref(x: &[f32], out: &mut [u8]) {
    assert_eq!((x.len(), out.len()), (256, 110));
    let mut l = [0i8; 256];
    let mut scales = [0.0f32; 16];
    let (mut max_scale, mut amax) = (0.0f32, 0.0f32);
    for j in 0..16 {
        scales[j] = make_q3_quants(4, &x[16 * j..16 * j + 16], &mut l[16 * j..16 * j + 16]);
        let s = scales[j].abs();
        if s > amax {
            amax = s;
            max_scale = scales[j];
        }
    }
    out.fill(0);
    let d16 = if max_scale != 0.0 {
        let iscale = -32.0 / max_scale;
        for j in 0..16 {
            let q = (nearest_int(iscale * scales[j]).clamp(-32, 31) + 32) as u8;
            if j < 8 {
                out[96 + j] = q & 0xF;
            } else {
                out[96 + j - 8] |= (q & 0xF) << 4;
            }
            out[96 + j % 4 + 8] |= (q >> 4) << (2 * (j / 4));
        }
        f32_to_f16(1.0 / iscale)
    } else {
        f32_to_f16(0.0)
    };
    out[108..110].copy_from_slice(&d16.to_le_bytes());
    let d_all = f16_to_f32(d16);
    for j in 0..16 {
        let sc = if j < 8 {
            out[96 + j] & 0xF
        } else {
            out[96 + j - 8] >> 4
        };
        let sc = (sc | (((out[96 + 8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) as i8 - 32;
        let d = d_all * sc as f32;
        if d == 0.0 {
            continue;
        }
        for ii in 0..16 {
            l[16 * j + ii] = (nearest_int(x[16 * j + ii] / d).clamp(-4, 3) + 4) as i8;
        }
    }
    let (mut m, mut hm) = (0usize, 1u8);
    for j in 0..256 {
        if l[j] > 3 {
            out[m] |= hm;
            l[j] -= 4;
        }
        m += 1;
        if m == 32 {
            m = 0;
            hm <<= 1;
        }
    }
    for j in (0..256).step_by(128) {
        for k in 0..32 {
            out[32 + j / 4 + k] = (l[j + k] as u8)
                | ((l[j + k + 32] as u8) << 2)
                | ((l[j + k + 64] as u8) << 4)
                | ((l[j + k + 96] as u8) << 6);
        }
    }
}

/// ggml `quantize_row_q4_K_ref` for one 256-element row: 144-byte block `d, dmin, scales[12], qs[128]`.
pub fn quantize_row_q4_k_ref(x: &[f32], out: &mut [u8]) {
    assert_eq!((x.len(), out.len()), (256, 144));
    let mut l = [0u8; 256];
    let (packed, d, dmin) = k_quant_affine(x, 15, -1.0, 20, &mut l);
    out[0..2].copy_from_slice(&d.to_le_bytes());
    out[2..4].copy_from_slice(&dmin.to_le_bytes());
    out[4..16].copy_from_slice(&packed);
    for (j, q) in out[16..].chunks_exact_mut(32).enumerate() {
        for i in 0..32 {
            q[i] = l[64 * j + i] | (l[64 * j + i + 32] << 4);
        }
    }
}

/// ggml `quantize_row_q5_K_ref` for one 256-element row: 176-byte block `d, dmin, scales[12], qh[32], qs[128]`.
pub fn quantize_row_q5_k_ref(x: &[f32], out: &mut [u8]) {
    assert_eq!((x.len(), out.len()), (256, 176));
    let mut l = [0u8; 256];
    let (packed, d, dmin) = k_quant_affine(x, 31, -0.5, 15, &mut l);
    out[0..2].copy_from_slice(&d.to_le_bytes());
    out[2..4].copy_from_slice(&dmin.to_le_bytes());
    out[4..16].copy_from_slice(&packed);
    let (qh, qs) = out[16..].split_at_mut(32);
    qh.fill(0);
    let (mut m1, mut m2) = (1u8, 2u8);
    for (n, ql) in qs.chunks_exact_mut(32).enumerate() {
        for j in 0..32 {
            let (mut l1, mut l2) = (l[64 * n + j], l[64 * n + j + 32]);
            if l1 > 15 {
                l1 -= 16;
                qh[j] |= m1;
            }
            if l2 > 15 {
                l2 -= 16;
                qh[j] |= m2;
            }
            ql[j] = l1 | (l2 << 4);
        }
        m1 <<= 2;
        m2 <<= 2;
    }
}

/// ggml `quantize_row_q6_K_ref` for one 256-element row: 210-byte block `ql[128], qh[64], scales[16] (i8), d`.
pub fn quantize_row_q6_k_ref(x: &[f32], out: &mut [u8]) {
    assert_eq!((x.len(), out.len()), (256, 210));
    let mut l = [0i8; 256];
    let mut scales = [0.0f32; 16];
    let (mut max_scale, mut max_abs_scale) = (0.0f32, 0.0f32);
    for ib in 0..16 {
        let s = make_qx_quants(32, &x[16 * ib..16 * ib + 16], &mut l[16 * ib..16 * ib + 16]);
        scales[ib] = s;
        if s.abs() > max_abs_scale {
            max_abs_scale = s.abs();
            max_scale = s;
        }
    }
    out.fill(0);
    if max_abs_scale < 1e-15 {
        return;
    }
    let iscale = -128.0 / max_scale;
    let d16 = f32_to_f16(1.0 / iscale);
    out[208..210].copy_from_slice(&d16.to_le_bytes());
    let d_f = f16_to_f32(d16);
    let mut sc = [0i8; 16];
    for ib in 0..16 {
        sc[ib] = nearest_int(iscale * scales[ib]).min(127) as i8;
        out[192 + ib] = sc[ib] as u8;
    }
    for j in 0..16 {
        let d = d_f * sc[j] as f32;
        if d == 0.0 {
            continue;
        }
        for ii in 0..16 {
            l[16 * j + ii] = (nearest_int(x[16 * j + ii] / d).clamp(-32, 31) + 32) as i8;
        }
    }
    let (ql, rest) = out.split_at_mut(128);
    let qh = &mut rest[..64];
    for half in 0..2 {
        let j = 128 * half;
        for i in 0..32 {
            let q = |o: usize| l[j + i + o] as u8;
            ql[64 * half + i] = (q(0) & 0xF) | ((q(64) & 0xF) << 4);
            ql[64 * half + 32 + i] = (q(32) & 0xF) | ((q(96) & 0xF) << 4);
            qh[32 * half + i] =
                (q(0) >> 4) | ((q(32) >> 4) << 2) | ((q(64) >> 4) << 4) | ((q(96) >> 4) << 6);
        }
    }
}

/// Encodes `weights` (row-major, `cols` a multiple of 256) with one of the row encoders above.
pub fn quantize_k_rows(
    weights: &[f32],
    cols: usize,
    block_bytes: usize,
    row: fn(&[f32], &mut [u8]),
) -> Vec<u8> {
    assert!(
        cols % 256 == 0 && weights.len() % cols == 0,
        "K-quant rows need cols % 256 == 0, got {cols}"
    );
    let mut out = vec![0u8; weights.len() / 256 * block_bytes];
    for (x, y) in weights
        .chunks_exact(256)
        .zip(out.chunks_exact_mut(block_bytes))
    {
        row(x, y);
    }
    out
}

/// Dequantizes one 144-byte GGUF Q4_K block into 256 values (ggml `dequantize_row_q4_K` order).
pub fn dequantize_q4_k_block(b: &[u8], y: &mut [f32]) {
    let (d, dmin) = (f16_at(b, 0), f16_at(b, 2));
    let (scales, qs) = (&b[4..16], &b[16..144]);
    for g in 0..4 {
        let (sc0, m0) = k4_scale_min(2 * g, scales);
        let (sc1, m1) = k4_scale_min(2 * g + 1, scales);
        let (d0, min0, d1, min1) = (d * sc0, dmin * m0, d * sc1, dmin * m1);
        let q = &qs[g * 32..(g + 1) * 32];
        for l in 0..32 {
            y[g * 64 + l] = d0 * (q[l] & 0x0F) as f32 - min0;
            y[g * 64 + 32 + l] = d1 * (q[l] >> 4) as f32 - min1;
        }
    }
}

/// Dequantizes one 176-byte GGUF Q5_K block into 256 values (ggml `dequantize_row_q5_K` order).
pub fn dequantize_q5_k_block(b: &[u8], y: &mut [f32]) {
    let (d, dmin) = (f16_at(b, 0), f16_at(b, 2));
    let (scales, qh, qs) = (&b[4..16], &b[16..48], &b[48..176]);
    for g in 0..4 {
        let (sc0, m0) = k4_scale_min(2 * g, scales);
        let (sc1, m1) = k4_scale_min(2 * g + 1, scales);
        let (d0, min0, d1, min1) = (d * sc0, dmin * m0, d * sc1, dmin * m1);
        let q = &qs[g * 32..(g + 1) * 32];
        for l in 0..32 {
            let h0 = ((qh[l] >> (2 * g)) & 1) << 4;
            let h1 = ((qh[l] >> (2 * g + 1)) & 1) << 4;
            y[g * 64 + l] = d0 * ((q[l] & 0x0F) | h0) as f32 - min0;
            y[g * 64 + 32 + l] = d1 * ((q[l] >> 4) | h1) as f32 - min1;
        }
    }
}

/// Dequantizes one 210-byte GGUF Q6_K block into 256 values (ggml `dequantize_row_q6_K` order).
pub fn dequantize_q6_k_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 208);
    for n in 0..2 {
        let ql = &b[n * 64..n * 64 + 64];
        let qh = &b[128 + n * 32..128 + n * 32 + 32];
        let sc = &b[192 + n * 8..192 + n * 8 + 8];
        let y = &mut y[n * 128..n * 128 + 128];
        for l in 0..32 {
            let is = l / 16;
            let q = |lo: u8, shift: u32| ((lo | (((qh[l] >> shift) & 3) << 4)) as i32 - 32) as f32;
            y[l] = d * sc[is] as i8 as f32 * q(ql[l] & 0x0F, 0);
            y[l + 32] = d * sc[is + 2] as i8 as f32 * q(ql[l + 32] & 0x0F, 2);
            y[l + 64] = d * sc[is + 4] as i8 as f32 * q(ql[l] >> 4, 4);
            y[l + 96] = d * sc[is + 6] as i8 as f32 * q(ql[l + 32] >> 4, 6);
        }
    }
}

include!("gguf_tables.rs");

#[inline]
fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

#[inline]
fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

/// Byte `j` of a packed ggml grid entry (ggml reads the u64/u32 grids as little-endian byte arrays).
#[inline]
fn grid_byte(g: u64, j: usize) -> u8 {
    (g >> (8 * j)) as u8
}

#[inline]
fn sign(signs: u8, j: usize) -> f32 {
    if signs & (1 << j) != 0 {
        -1.0
    } else {
        1.0
    }
}

/// Q2_K (84 B / 256): scales[16], qs[64], d, dmin.
pub fn dequantize_q2_k_block(b: &[u8], y: &mut [f32]) {
    let (d, min) = (f16_at(b, 80), f16_at(b, 82));
    let (scales, qs) = (&b[0..16], &b[16..80]);
    let mut is = 0;
    let mut o = 0;
    for n in 0..2 {
        let q = &qs[n * 32..n * 32 + 32];
        for shift in [0, 2, 4, 6] {
            for half in 0..2 {
                let sc = scales[is];
                is += 1;
                let (dl, ml) = (d * (sc & 0xF) as f32, min * (sc >> 4) as f32);
                for l in 0..16 {
                    y[o] = dl * ((q[l + 16 * half] >> shift) & 3) as f32 - ml;
                    o += 1;
                }
            }
        }
    }
}

/// Q3_K (110 B / 256): hmask[32], qs[64], 6-bit scales[12], d.
pub fn dequantize_q3_k_block(b: &[u8], y: &mut [f32]) {
    let (hm, qs, d_all) = (&b[0..32], &b[32..96], f16_at(b, 108));
    let a = [u32_at(b, 96), u32_at(b, 100), u32_at(b, 104)];
    let (k1, k2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
    let aux = [
        (a[0] & k2) | ((a[2] & k1) << 4),
        (a[1] & k2) | (((a[2] >> 2) & k1) << 4),
        ((a[0] >> 4) & k2) | (((a[2] >> 4) & k1) << 4),
        ((a[1] >> 4) & k2) | (((a[2] >> 6) & k1) << 4),
    ];
    let scale = |i: usize| (aux[i / 4] >> (8 * (i % 4))) as u8 as i8 as i32 - 32;
    let (mut is, mut o, mut m) = (0, 0, 1u8);
    for n in 0..2 {
        let q = &qs[n * 32..n * 32 + 32];
        for shift in [0, 2, 4, 6] {
            for half in 0..2 {
                let dl = d_all * scale(is) as f32;
                is += 1;
                for l in 0..16 {
                    let k = l + 16 * half;
                    let h = if hm[k] & m != 0 { 0 } else { 4 };
                    y[o] = dl * (((q[k] >> shift) & 3) as i32 - h) as f32;
                    o += 1;
                }
            }
            m <<= 1;
        }
    }
}

/// TQ1_0 (54 B / 256): base-3 packed qs[48], qh[4], d.
pub fn dequantize_tq1_0_block(b: &[u8], y: &mut [f32]) {
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    let d = f16_at(b, 52);
    let trit = |q: u8, n: usize| ((q.wrapping_mul(POW3[n]) as u16 * 3) >> 8) as f32 - 1.0;
    let mut o = 0;
    for (start, width) in [(0, 32), (32, 16)] {
        for n in 0..5 {
            for m in 0..width {
                y[o] = trit(b[start + m], n) * d;
                o += 1;
            }
        }
    }
    for n in 0..4 {
        for j in 0..4 {
            y[o] = trit(b[48 + j], n) * d;
            o += 1;
        }
    }
}

/// TQ2_0 (66 B / 256): 2-bit qs[64], d; value = (q - 1) * d.
pub fn dequantize_tq2_0_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 64);
    let mut o = 0;
    for j in [0, 32] {
        for l in 0..4 {
            for m in 0..32 {
                y[o] = (((b[j + m] >> (2 * l)) & 3) as f32 - 1.0) * d;
                o += 1;
            }
        }
    }
}

/// IQ2_XXS (66 B / 256): d, then per 32: 4 grid indices + 32-bit (4x7 sign bits, 4-bit scale).
pub fn dequantize_iq2_xxs_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for ib in 0..8 {
        let q = &b[2 + 8 * ib..10 + 8 * ib];
        let aux1 = u32_at(q, 4);
        let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
        for l in 0..4 {
            let g = IQ2XXS_GRID[q[l] as usize];
            let s = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                y[ib * 32 + l * 8 + j] = db * grid_byte(g, j) as f32 * sign(s, j);
            }
        }
    }
}

/// IQ2_XS (74 B / 256): d, u16 qs[32] (9-bit grid index + 7-bit sign index), scales[8].
pub fn dequantize_iq2_xs_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for ib in 0..8 {
        let sc = b[66 + ib];
        let db = [
            d * (0.5 + (sc & 0xF) as f32) * 0.25,
            d * (0.5 + (sc >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let q = u16_at(b, 2 + 2 * (4 * ib + l)) as usize;
            let (g, s) = (IQ2XS_GRID[q & 511], KSIGNS_IQ2XS[q >> 9]);
            for j in 0..8 {
                y[ib * 32 + l * 8 + j] = db[l / 2] * grid_byte(g, j) as f32 * sign(s, j);
            }
        }
    }
}

/// IQ2_S (82 B / 256): d, qs[32] + signs[32], qh[8], scales[8].
pub fn dequantize_iq2_s_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let (qs, signs, qh, scales) = (&b[2..34], &b[34..66], &b[66..74], &b[74..82]);
    for ib in 0..8 {
        let sc = scales[ib];
        let db = [
            d * (0.5 + (sc & 0xF) as f32) * 0.25,
            d * (0.5 + (sc >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let idx = qs[4 * ib + l] as usize | ((qh[ib] as usize) << (8 - 2 * l) & 0x300);
            let (g, s) = (IQ2S_GRID[idx], signs[4 * ib + l]);
            for j in 0..8 {
                y[ib * 32 + l * 8 + j] = db[l / 2] * grid_byte(g, j) as f32 * sign(s, j);
            }
        }
    }
}

/// IQ3_XXS (98 B / 256): d, qs[64] grid indices, then per 32 a u32 (4x7 sign bits, 4-bit scale).
pub fn dequantize_iq3_xxs_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let qs = &b[2..66];
    for ib in 0..8 {
        let aux = u32_at(b, 66 + 4 * ib);
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        for l in 0..4 {
            let s = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            let g1 = IQ3XXS_GRID[qs[8 * ib + 2 * l] as usize] as u64;
            let g2 = IQ3XXS_GRID[qs[8 * ib + 2 * l + 1] as usize] as u64;
            for j in 0..4 {
                y[ib * 32 + l * 8 + j] = db * grid_byte(g1, j) as f32 * sign(s, j);
                y[ib * 32 + l * 8 + j + 4] = db * grid_byte(g2, j) as f32 * sign(s, j + 4);
            }
        }
    }
}

/// IQ3_S (110 B / 256): d, qs[64], qh[8], signs[32], scales[4].
pub fn dequantize_iq3_s_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let (qs, qh, signs, scales) = (&b[2..66], &b[66..74], &b[74..106], &b[106..110]);
    for ib in 0..8 {
        let sc = scales[ib / 2] >> (4 * (ib % 2)) & 0xF;
        let db = d * (1 + 2 * sc as i32) as f32;
        let h = qh[ib] as usize;
        for l in 0..4 {
            let g1 = IQ3S_GRID[qs[8 * ib + 2 * l] as usize | ((h << (8 - 2 * l)) & 256)] as u64;
            let g2 = IQ3S_GRID[qs[8 * ib + 2 * l + 1] as usize | ((h << (7 - 2 * l)) & 256)] as u64;
            let s = signs[4 * ib + l];
            for j in 0..4 {
                y[ib * 32 + l * 8 + j] = db * grid_byte(g1, j) as f32 * sign(s, j);
                y[ib * 32 + l * 8 + j + 4] = db * grid_byte(g2, j) as f32 * sign(s, j + 4);
            }
        }
    }
}

const IQ1S_DELTA: f32 = 0.125;

/// IQ1_S (50 B / 256): d, qs[32], u16 qh[8] (3-bit index highs, 3-bit scale, delta sign).
pub fn dequantize_iq1_s_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for ib in 0..8 {
        let qh = u16_at(b, 34 + 2 * ib) as usize;
        let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
        let delta = if qh & 0x8000 != 0 {
            -IQ1S_DELTA
        } else {
            IQ1S_DELTA
        };
        for l in 0..4 {
            let g = IQ1S_GRID[b[2 + 4 * ib + l] as usize | (((qh >> (3 * l)) & 7) << 8)];
            for j in 0..8 {
                y[ib * 32 + l * 8 + j] = dl * (grid_byte(g, j) as i8 as f32 + delta);
            }
        }
    }
}

/// IQ1_M (56 B / 256): qs[32], qh[16], scales[8] (3-bit sub-scales; f16 d spread over the top nibbles).
pub fn dequantize_iq1_m_block(b: &[u8], y: &mut [f32]) {
    let (qs, qh) = (&b[0..32], &b[32..48]);
    let sc = [u16_at(b, 48), u16_at(b, 50), u16_at(b, 52), u16_at(b, 54)];
    let d = f16_to_f32(
        (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000),
    );
    for ib in 0..8 {
        let s = sc[ib / 2] as usize >> (6 * (ib % 2));
        let dl = [
            d * (2 * (s & 7) + 1) as f32,
            d * (2 * ((s >> 3) & 7) + 1) as f32,
        ];
        let (h0, h1) = (qh[2 * ib] as usize, qh[2 * ib + 1] as usize);
        let idx = [
            (h0 << 8) & 0x700,
            (h0 << 4) & 0x700,
            (h1 << 8) & 0x700,
            (h1 << 4) & 0x700,
        ];
        let neg = [h0 & 0x08, h0 & 0x80, h1 & 0x08, h1 & 0x80];
        for l in 0..4 {
            let g = IQ1S_GRID[qs[4 * ib + l] as usize | idx[l]];
            let delta = if neg[l] != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
            for j in 0..8 {
                y[ib * 32 + l * 8 + j] = dl[l / 2] * (grid_byte(g, j) as i8 as f32 + delta);
            }
        }
    }
}

/// IQ4_NL (18 B / 32): d, then Q4_0 nibble order through the non-linear `kvalues_iq4nl` table.
pub fn dequantize_iq4_nl_block(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for (i, &q) in b[2..18].iter().enumerate() {
        y[i] = d * KVALUES_IQ4NL[(q & 0xF) as usize] as f32;
        y[i + 16] = d * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
    }
}

/// IQ4_XS (136 B / 256): d, u16 scales_h, scales_l[4], qs[128]; 6-bit sub-scales, IQ4_NL values.
pub fn dequantize_iq4_xs_block(b: &[u8], y: &mut [f32]) {
    let (d, scales_h) = (f16_at(b, 0), u16_at(b, 2));
    for ib in 0..8 {
        let ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) as i32
            | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
        let dl = d * (ls - 32) as f32;
        for (j, &q) in b[8 + 16 * ib..24 + 16 * ib].iter().enumerate() {
            y[ib * 32 + j] = dl * KVALUES_IQ4NL[(q & 0xF) as usize] as f32;
            y[ib * 32 + j + 16] = dl * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
        }
    }
}

/// MXFP4 (17 B / 32): E8M0 exponent, then Q4_0 nibble order of doubled E2M1 values; d = 2^(e-128).
pub fn dequantize_mxfp4_block(b: &[u8], y: &mut [f32]) {
    let e = b[0] as u32;
    let d = f32::from_bits(if e < 2 {
        0x0020_0000 << e
    } else {
        (e - 1) << 23
    });
    for (i, &q) in b[1..17].iter().enumerate() {
        y[i] = KVALUES_FP4[(q & 0xF) as usize] as f32 * d;
        y[i + 16] = KVALUES_FP4[(q >> 4) as usize] as f32 * d;
    }
}


/// NVFP4 (E2M1) lookup table representing 16 values for nibbles 0x0..=0xF.
pub const NVFP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];
/// ggml NVFP4 (36 B / 64; `block_nvfp4`, `dequantize_row_nvfp4`): four UE4M3 scale bytes, one per
/// 16-element sub-block, then 32 nibble bytes; within a sub-block's 8 bytes the low nibbles are
/// elements 0..8 and the high nibbles 8..16. `d = ue4m3 * 0.5` (0 and 0x7F give 0) against the
/// doubled E2M1 table.
pub fn dequantize_nvfp4_block(b: &[u8], y: &mut [f32]) {
    for s in 0..4 {
        let x = b[s] as u32;
        let (e, m) = ((x >> 3) & 0xF, x & 7);
        let d = match x {
            0 | 0x7F => 0.0,
            _ if e == 0 => m as f32 / 1024.0,
            _ => f32::from_bits(((e + 119) << 23) | (m << 20)),
        };
        for (j, &q) in b[4 + 8 * s..12 + 8 * s].iter().enumerate() {
            y[16 * s + j] = KVALUES_FP4[(q & 0xF) as usize] as f32 * d;
            y[16 * s + j + 8] = KVALUES_FP4[(q >> 4) as usize] as f32 * d;
        }
    }
}


/// The UE4M3 scale byte's value as [`dequantize_nvfp4_block`] reads it (`0` and `0x7F`: 0).
fn nvfp4_scale(x: u32) -> f32 {
    let (e, m) = ((x >> 3) & 0xF, x & 7);
    match x {
        0 | 0x7F => 0.0,
        _ if e == 0 => m as f32 / 1024.0,
        _ => f32::from_bits(((e + 119) << 23) | (m << 20)),
    }
}

/// GGML `GGUF_NVFP4` blocks of 64 values ([`dequantize_nvfp4_block`] layout): per 16-value
/// sub-block the smallest UE4M3 scale `d` with `12·d >= amax` (no value clips), then each value's
/// nearest entry of the doubled E2M1 table. `weights.len()` is a multiple of 64.
pub fn quantize_f32_to_gguf_nvfp4(weights: &[f32]) -> Vec<u8> {
    // Scale codes 1..=0x7E decode to increasing values.
    let scales: Vec<f32> = (1..0x7Fu32).map(nvfp4_scale).collect();
    let mut out = Vec::with_capacity(weights.len() / 64 * 36);
    for block in weights.chunks(64) {
        let mut head = [0u8; 4];
        let mut q = [0u8; 32];
        for (s, sub) in block.chunks(16).enumerate() {
            let amax = sub.iter().fold(0.0f32, |m, w| m.max(w.abs()));
            if amax == 0.0 {
                continue;
            }
            let i = scales.partition_point(|&d| d * 12.0 < amax).min(scales.len() - 1);
            head[s] = i as u8 + 1;
            let id = 1.0 / scales[i];
            for (j, &w) in sub.iter().enumerate() {
                let x = w * id;
                let k = (0..16)
                    .min_by(|&a, &b| {
                        (KVALUES_FP4[a] as f32 - x).abs().total_cmp(&(KVALUES_FP4[b] as f32 - x).abs())
                    })
                    .unwrap() as u8;
                // Values 0..8 in the low nibbles of bytes 4+8s.., 8..16 in the high nibbles.
                q[8 * s + j % 8] |= if j < 8 { k } else { k << 4 };
            }
        }
        out.extend_from_slice(&head);
        out.extend_from_slice(&q);
    }
    out
}

/// Quantize an f32 weight slice into GGUF Q4_0 format (18 bytes per 32 elements), as ggml's
/// `quantize_row_q4_0_ref`: `d = max/-8` for the signed element of largest magnitude, so
/// re-encoding a dequantized block made by that reference quantizer reproduces it (imatrix-tuned
/// blocks may differ; loaders carry raw blocks where they can).
pub fn quantize_f32_to_q4_0(weights: &[f32]) -> Vec<u8> {
    let num_blocks = (weights.len() + 31) / 32;
    let mut out = Vec::with_capacity(num_blocks * 18);
    for chunk in weights.chunks(32) {
        let (mut amax, mut max) = (0.0f32, 0.0f32);
        for &w in chunk {
            if amax < w.abs() {
                amax = w.abs();
                max = w;
            }
        }
        // A block of {-m, 0, m} (ternary sources such as TQ2_0) with both signs present is not
        // representable with d = max/-8 (the opposite sign clips to 7/8 m); d = m is exact (q = 8 ± 1).
        let ternary = amax > 0.0
            && chunk.iter().all(|&w| w == 0.0 || w.abs() == amax)
            && chunk.contains(&-max);
        let d = if ternary { amax } else { max / -8.0 };
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = crate::model::f32_to_f16(d);
        out.extend_from_slice(&d_f16.to_le_bytes());

        let mut nibbles = [0u8; 16];
        for i in 0..16 {
            let w0 = if i < chunk.len() { chunk[i] } else { 0.0 };
            let w1 = if i + 16 < chunk.len() {
                chunk[i + 16]
            } else {
                0.0
            };
            let q0 = ((w0 * id + 8.5) as i8).min(15) as u8;
            let q1 = ((w1 * id + 8.5) as i8).min(15) as u8;
            nibbles[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
        out.extend_from_slice(&nibbles);
    }
    out
}

/// Quantize an f32 weight slice into GGUF Q8_0 format (34 bytes per 32 elements).
pub fn quantize_f32_to_q8_0(weights: &[f32]) -> Vec<u8> {
    let num_blocks = (weights.len() + 31) / 32;
    let mut out = Vec::with_capacity(num_blocks * 34);
    for chunk in weights.chunks(32) {
        let mut max_abs = 0.0f32;
        for &w in chunk {
            let a = w.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let d = max_abs / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = crate::model::f32_to_f16(d);
        out.extend_from_slice(&d_f16.to_le_bytes());

        for i in 0..32 {
            let w = if i < chunk.len() { chunk[i] } else { 0.0 };
            let q = (w * id).round().clamp(-128.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;

    /// `GGUF_NVFP4` encoding round-trips through the decoder within half an E2M1 step of each
    /// sub-block's scale, keeps zeros and the sign, and never clips the sub-block maximum.
    #[test]
    fn gguf_nvfp4_encode_round_trips() {
        let w: Vec<f32> = (0..256).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013 * (1 + i / 64) as f32).collect();
        let raw = quantize_f32_to_gguf_nvfp4(&w);
        assert_eq!(raw.len(), 4 * 36);
        let mut y = vec![0.0f32; 256];
        for (b, out) in raw.chunks(36).zip(y.chunks_mut(64)) {
            dequantize_nvfp4_block(b, out);
        }
        for (sub, (a, b)) in w.chunks(16).zip(y.chunks(16)).enumerate() {
            let amax = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            // d <= amax/12 * 1.125 (the next UE4M3 step); the widest E2M1 gap is 4 (8..12).
            let tol = amax / 12.0 * 1.125 * 2.0 + 1e-6;
            for (x, z) in a.iter().zip(b) {
                assert!((x - z).abs() <= tol, "sub {sub}: {x} -> {z} (tol {tol})");
                assert!(*x == 0.0 || z.signum() == x.signum() || *z == 0.0, "sub {sub}: sign of {x} -> {z}");
            }
        }
    }

    /// Through `encode_rows`: values on the grid (E2M1 entry × a representable
    /// scale) round-trip exactly, an all-zero sub-block encodes scale 0, and sub-blocks beyond the
    /// largest scale clamp to code 0x7E (never 0x7F, which decodes to 0) without NaN.
    #[test]
    fn gguf_nvfp4_from_f32_edges() {
        let d = nvfp4_scale(0x3A);
        let mut w: Vec<f32> = (0..64).map(|i| KVALUES_FP4[i % 16] as f32 * d).collect();
        w[0] = 12.0 * d; // every sub-block's amax is 12·d
        for i in [16, 32] {
            w[i] = -12.0 * d;
        }
        w.extend(std::iter::repeat_n(0.0, 64));
        w.extend((0..64).map(|i| if i % 2 == 0 { 1e6 } else { -3e5 }));
        let raw_data = encode_rows(&w, 192, QuantFormat::GGUF_NVFP4).unwrap();
        let mut y = vec![0.0f32; 192];
        for (b, out) in raw_data.chunks(36).zip(y.chunks_mut(64)) {
            dequantize_nvfp4_block(b, out);
        }
        assert_eq!(&y[..64], &w[..64], "on-grid values");
        assert!(raw_data[36..40].iter().all(|&c| c == 0) && y[64..128].iter().all(|&v| v == 0.0));
        assert!(raw_data[72..76].iter().all(|&c| c == 0x7E), "{:x?}", &raw_data[72..76]);
        assert!(y[128..].iter().all(|v| v.is_finite() && *v != 0.0));
    }
    use crate::hardware::QuantFormat;

    /// Deterministic bell-shaped weights with per-block scale and an outlier every fourth block;
    /// the same generator (and f32 operation order) as `/tmp/ggml_ref/gold.c`, which ran ggml's own
    /// `quantize_row_*_ref` / `dequantize_row_*` (llama.cpp master, `-ffp-contract=off`).
    fn golden_input(seed: u64) -> Vec<f32> {
        let mut st = seed;
        let mut u = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((st >> 33) as u32) as f32 / 2147483648.0
        };
        let mut x = vec![0.0f32; 16384];
        for (i, v) in x.iter_mut().enumerate() {
            let (a, b, c, d) = (u(), u(), u(), u());
            *v = (a + b + c + d - 2.0) * (0.05 * (1 + (i / 256) % 5) as f32);
        }
        for b in (0..x.len() / 256).step_by(4) {
            x[b * 256 + 7] *= 9.0;
        }
        x
    }

    fn fnv1a64(b: &[u8]) -> u64 {
        b.iter().fold(0xcbf29ce484222325u64, |h, &c| {
            (h ^ c as u64).wrapping_mul(0x100000001b3)
        })
    }

    /// Encoders reproduce ggml byte-for-byte (hash of all blocks) and so have ggml's exact RMSE.
    #[test]
    fn encoders_match_ggml_reference() {
        // (seed, format, ggml hash, ggml RMSE vs the f32 input)
        let golden: [(u64, QuantFormat, u64, f64); 10] = [
            (1, QuantFormat::GGUF_Q4_K_M, 0x7c024b97d3740200, 0.006986602),
            (1, QuantFormat::GGUF_Q5_K_M, 0x4a8d702cff5d76af, 0.003506242),
            (1, QuantFormat::GGUF_Q6_K, 0xd15767a0257f0055, 0.001763364),
            (1, QuantFormat::GGUF_Q8_0, 0x5794e6f1110df100, 0.000538608),
            (1, QuantFormat::GGUF_Q4_0, 0xbfd1f7415032602d, 0.008640246),
            (2, QuantFormat::GGUF_Q4_K_M, 0x8e7b0735b1b75668, 0.007383891),
            (2, QuantFormat::GGUF_Q5_K_M, 0x633fa05533db0eee, 0.003749277),
            (2, QuantFormat::GGUF_Q6_K, 0x7467d429f40c107c, 0.001980719),
            (2, QuantFormat::GGUF_Q8_0, 0x727585e87d113474, 0.000621894),
            (2, QuantFormat::GGUF_Q4_0, 0x1cdb4f9057e69f39, 0.009815987),
        ];
        for (seed, format, hash, rmse_ref) in golden {
            let x = golden_input(seed);
            let raw_data = encode_rows(&x, 256, format).unwrap();
            assert_eq!(
                fnv1a64(&raw_data),
                hash,
                "{format:?} seed {seed}: bytes differ from ggml"
            );
            let y = dequantize_gguf_blocks(format, &raw_data).unwrap();
            let sq: f64 = x
                .iter()
                .zip(&y)
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum();
            let rmse = (sq / x.len() as f64).sqrt();
            assert!(
                (rmse - rmse_ref).abs() < 1e-8,
                "{format:?} seed {seed}: RMSE {rmse:.9} vs ggml {rmse_ref:.9}"
            );
        }
    }

    /// A Q4_0 block's own dequantized values re-encode to the same bytes (what makes "native" Q4_0
    /// lossless when a loader has to dequantize first).
    #[test]
    fn q4_0_and_q8_0_requantize_is_idempotent() {
        let x = golden_input(3);
        for (format, enc) in [
            (
                QuantFormat::GGUF_Q4_0,
                quantize_f32_to_q4_0 as fn(&[f32]) -> Vec<u8>,
            ),
            (QuantFormat::GGUF_Q8_0, quantize_f32_to_q8_0),
        ] {
            let once = enc(&x);
            let twice = enc(&dequantize_gguf_blocks(format, &once).unwrap());
            assert_eq!(once, twice, "{format:?}");
        }
    }
}
