//! SIMD primitives for Gibbs.
//!
//! Vector routines for RMSNorm, Softmax, and dot products using ARM NEON on aarch64.

#[inline]
pub fn vector_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        while i + 16 <= len {
            let va0 = vld1q_f32(a.as_ptr().add(i));
            let vb0 = vld1q_f32(b.as_ptr().add(i));
            acc0 = vfmaq_f32(acc0, va0, vb0);

            let va1 = vld1q_f32(a.as_ptr().add(i + 4));
            let vb1 = vld1q_f32(b.as_ptr().add(i + 4));
            acc1 = vfmaq_f32(acc1, va1, vb1);

            let va2 = vld1q_f32(a.as_ptr().add(i + 8));
            let vb2 = vld1q_f32(b.as_ptr().add(i + 8));
            acc2 = vfmaq_f32(acc2, va2, vb2);

            let va3 = vld1q_f32(a.as_ptr().add(i + 12));
            let vb3 = vld1q_f32(b.as_ptr().add(i + 12));
            acc3 = vfmaq_f32(acc3, va3, vb3);

            i += 16;
        }

        let sum4 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut dot = vaddvq_f32(sum4);

        while i < len {
            dot += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
        dot
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;

        while i + 8 <= len {
            acc0 += a[i] * b[i] + a[i + 1] * b[i + 1];
            acc1 += a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
            acc2 += a[i + 4] * b[i + 4] + a[i + 5] * b[i + 5];
            acc3 += a[i + 6] * b[i + 6] + a[i + 7] * b[i + 7];
            i += 8;
        }
        let mut dot = acc0 + acc1 + acc2 + acc3;
        while i < len {
            dot += a[i] * b[i];
            i += 1;
        }
        dot
    }
}

/// In-place or out-of-place vectorized RMSNorm: out = x * weight / sqrt(mean(x^2) + eps).
#[inline]
pub fn vector_rmsnorm(x: &[f32], weight: &[f32], out: &mut [f32], eps: f32) {
    let n = x.len();
    assert_eq!(n, weight.len());
    assert_eq!(n, out.len());
    if n == 0 {
        return;
    }

    let sum_sq = vector_dot_product(x, x);
    let scale = 1.0 / (sum_sq / n as f32 + eps).sqrt();

    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let vscale = vdupq_n_f32(scale);
        while i + 8 <= n {
            let vx0 = vld1q_f32(x.as_ptr().add(i));
            let vw0 = vld1q_f32(weight.as_ptr().add(i));
            let vo0 = vmulq_f32(vmulq_f32(vx0, vscale), vw0);
            vst1q_f32(out.as_mut_ptr().add(i), vo0);

            let vx1 = vld1q_f32(x.as_ptr().add(i + 4));
            let vw1 = vld1q_f32(weight.as_ptr().add(i + 4));
            let vo1 = vmulq_f32(vmulq_f32(vx1, vscale), vw1);
            vst1q_f32(out.as_mut_ptr().add(i + 4), vo1);

            i += 8;
        }
    }

    while i < n {
        out[i] = x[i] * scale * weight[i];
        i += 1;
    }
}

/// Vectorized in-place numerically stable Softmax: x[i] = exp(x[i] - max) / sum(exp).
#[inline]
pub fn vector_softmax(x: &mut [f32]) {
    let n = x.len();
    if n == 0 {
        return;
    }

    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    for val in x.iter_mut() {
        let e = (*val - max_val).exp();
        *val = e;
        sum_exp += e;
    }

    let inv_sum = 1.0 / sum_exp.max(1e-12);
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let vinv = vdupq_n_f32(inv_sum);
        while i + 8 <= n {
            let vx0 = vld1q_f32(x.as_ptr().add(i));
            vst1q_f32(x.as_mut_ptr().add(i), vmulq_f32(vx0, vinv));

            let vx1 = vld1q_f32(x.as_ptr().add(i + 4));
            vst1q_f32(x.as_mut_ptr().add(i + 4), vmulq_f32(vx1, vinv));

            i += 8;
        }
    }

    while i < n {
        x[i] *= inv_sum;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_dot_product_matches_scalar() {
        let a = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0,
        ];
        let b = vec![
            0.5, -0.5, 1.0, 2.0, -1.0, 0.5, 0.25, -0.25, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4,
            -0.4, 0.5,
        ];
        let expected: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let actual = vector_dot_product(&a, &b);
        assert!((actual - expected).abs() < 1e-4);
    }

    #[test]
    fn test_simd_rmsnorm_matches_scalar() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let w = vec![1.0; 8];
        let mut out = vec![0.0f32; 8];
        vector_rmsnorm(&x, &w, &mut out, 1e-5);
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let scale = 1.0 / (sum_sq / 8.0 + 1e-5).sqrt();
        for i in 0..8 {
            assert!((out[i] - x[i] * scale).abs() < 1e-4);
        }
    }
}
