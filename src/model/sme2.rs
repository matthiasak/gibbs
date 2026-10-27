//! SME2 outer-product tiles for the dense CPU GEMM: one call accumulates a 32×32 f32 block in the
//! four 16×16 `ZA` tiles of a 512-bit streaming vector length, one `fmopa` per tile per `k`.

use std::sync::LazyLock;

/// The host has SME2 with the 512-bit streaming vectors the tile shape assumes.
pub fn available() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        crate::hardware::HardwareCapability::probe_system().cpu_has_arm_sme2
            && crate::hardware::sysctl::int(c"hw.optional.arm.sme_max_svl_b") == Some(64)
    });
    *ON
}

// x0 = a, x1 = b, x2 = c, x3 = k, x4 = ldc (elements), w5 = accumulate.
// A and B are packed 32 wide per k: a[p * 32 + i] = A[i][p], b[p * 32 + j] = B[p][j].
// za0 = C[0..16][0..16], za1 = C[0..16][16..32], za2 = C[16..32][0..16], za3 = C[16..32][16..32].
std::arch::global_asm!(
    ".arch armv9-a+sme2",
    ".text",
    ".p2align 4",
    ".globl _gibbs_sme2_sgemm_32x32",
    "_gibbs_sme2_sgemm_32x32:",
    "stp d8, d9, [sp, #-64]!",
    "stp d10, d11, [sp, #16]",
    "stp d12, d13, [sp, #32]",
    "stp d14, d15, [sp, #48]",
    "smstart",
    "ptrue p0.s",
    "lsl x4, x4, #2",
    "add x10, x2, x4, lsl #4",
    "cbz w5, 1f",
    "mov w12, #0",
    "mov x9, x2",
    "2:",
    "ld1w {{za0h.s[w12, 0]}}, p0/z, [x9]",
    "add x11, x9, #64",
    "ld1w {{za1h.s[w12, 0]}}, p0/z, [x11]",
    "ld1w {{za2h.s[w12, 0]}}, p0/z, [x10]",
    "add x11, x10, #64",
    "ld1w {{za3h.s[w12, 0]}}, p0/z, [x11]",
    "add x9, x9, x4",
    "add x10, x10, x4",
    "add w12, w12, #1",
    "cmp w12, #16",
    "b.lt 2b",
    "add x10, x2, x4, lsl #4",
    "b 3f",
    "1:",
    "zero {{za}}",
    "3:",
    "cbz x3, 5f",
    "4:",
    "ld1w {{z0.s}}, p0/z, [x0]",
    "ld1w {{z1.s}}, p0/z, [x0, #1, mul vl]",
    "ld1w {{z2.s}}, p0/z, [x1]",
    "ld1w {{z3.s}}, p0/z, [x1, #1, mul vl]",
    "fmopa za0.s, p0/m, p0/m, z0.s, z2.s",
    "fmopa za1.s, p0/m, p0/m, z0.s, z3.s",
    "fmopa za2.s, p0/m, p0/m, z1.s, z2.s",
    "fmopa za3.s, p0/m, p0/m, z1.s, z3.s",
    "add x0, x0, #128",
    "add x1, x1, #128",
    "subs x3, x3, #1",
    "b.ne 4b",
    "5:",
    "mov w12, #0",
    "mov x9, x2",
    "6:",
    "st1w {{za0h.s[w12, 0]}}, p0, [x9]",
    "add x11, x9, #64",
    "st1w {{za1h.s[w12, 0]}}, p0, [x11]",
    "st1w {{za2h.s[w12, 0]}}, p0, [x10]",
    "add x11, x10, #64",
    "st1w {{za3h.s[w12, 0]}}, p0, [x11]",
    "add x9, x9, x4",
    "add x10, x10, x4",
    "add w12, w12, #1",
    "cmp w12, #16",
    "b.lt 6b",
    "smstop",
    "ldp d14, d15, [sp, #48]",
    "ldp d12, d13, [sp, #32]",
    "ldp d10, d11, [sp, #16]",
    "ldp d8, d9, [sp], #64",
    "ret",
);

extern "C" {
    fn gibbs_sme2_sgemm_32x32(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        k: usize,
        ldc: usize,
        accumulate: u32,
    );
}

/// `c[0..32][0..32] (+)= Σ_p A[·][p] ⊗ B[p][·]` over `k` packed columns of A and rows of B;
/// row `i` of `c` starts at `c[i * ldc]`.
pub fn sgemm_32x32(a: &[f32], b: &[f32], c: &mut [f32], k: usize, ldc: usize, accumulate: bool) {
    assert!(available(), "sgemm_32x32 needs SME2 with 512-bit streaming vectors");
    assert!(a.len() >= 32 * k && b.len() >= 32 * k, "sgemm_32x32: short panels");
    assert!(ldc >= 32 && c.len() >= 31 * ldc + 32, "sgemm_32x32: short output");
    // SAFETY: the asserts bound every load and store; the routine saves the callee-saved
    // registers streaming mode clobbers and leaves streaming mode before returning.
    unsafe {
        gibbs_sme2_sgemm_32x32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), k, ldc, accumulate as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_match_neon_gemm() {
        if !available() {
            eprintln!("skipped: no SME2 with 512-bit streaming vectors");
            return;
        }
        let val = |i: usize| ((i * 37 % 101) as f32 - 50.0) / 25.0;
        // Raw tile, both modes, against a scalar reference.
        let k = 19;
        let (a, b): (Vec<f32>, Vec<f32>) = ((0..32 * k).map(val).collect(), (0..32 * k).map(|i| val(i + 7)).collect());
        let ldc = 40;
        let c0: Vec<f32> = (0..32 * ldc).map(|i| val(i + 3)).collect();
        for acc in [false, true] {
            let mut c = c0.clone();
            sgemm_32x32(&a, &b, &mut c, k, ldc, acc);
            for i in 0..32 {
                for j in 0..ldc {
                    let want = if j < 32 {
                        (0..k).map(|p| a[p * 32 + i] * b[p * 32 + j]).sum::<f32>() + if acc { c0[i * ldc + j] } else { 0.0 }
                    } else {
                        c0[i * ldc + j]
                    };
                    assert!((c[i * ldc + j] - want).abs() <= 1e-4 * want.abs().max(1.0), "acc {acc} ({i}, {j})");
                }
            }
        }
        // Through the GEMM: SME2 tiles against the NEON 8×12 kernel, with ragged edges and bias.
        let (m, n, kk) = (77, 101, 300);
        let x: Vec<f32> = (0..m * kk).map(val).collect();
        let w: Vec<f32> = (0..n * kk).map(|i| val(i + 11)).collect();
        let bias: Vec<f32> = (0..n).map(|i| val(i + 5)).collect();
        let (mut sme, mut neon) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
        crate::model::nn::matmul_nt_with(&x, &w, Some(&bias), m, kk, n, &mut sme, true);
        crate::model::nn::matmul_nt_with(&x, &w, Some(&bias), m, kk, n, &mut neon, false);
        for (i, (s, r)) in sme.iter().zip(&neon).enumerate() {
            assert!((s - r).abs() <= 1e-4 * r.abs().max(1.0), "{i}: sme {s} vs neon {r}");
        }
    }
}
