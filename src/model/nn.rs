//! Dense f32 neural-network primitives for the Stable Diffusion path (CLIP, UNet, VAE).
//!
//! Layout: row-major, single image, no batch dim. Images are `[c, h, w]`, token
//! sequences are `[rows, features]`. Every routine is deterministic: each output
//! element is produced by exactly one thread with a fixed reduction order.

use crate::model::bf16_to_f32;
use crate::model::simd::vector_softmax;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;

/// Micro-tile rows (A side, two NEON vectors).
const MR: usize = 8;
/// Micro-tile cols (B side, three NEON vectors).
const NR: usize = 12;
/// Reduction block: an MR x KC A panel (8 KiB) and NR x KC B panel (12 KiB) sit in L1.
const KC: usize = 256;
/// Thread tile rows (multiple of MR); the packed A block (128 KiB) sits in L2.
const MC: usize = 128;
/// Thread tile cols (multiple of NR).
const NC: usize = 192;
/// Query rows per attention task: scores scratch is QB x nk.
const QB: usize = 64;
/// Below this many multiply-adds a GEMM runs on the calling thread.
pub(crate) const PAR_MIN_FLOPS: usize = 1 << 18;
/// Below this many elements an elementwise/normalisation pass runs on the calling thread.
const PAR_MIN_ELEMS: usize = 1 << 15;
/// Elementwise task granularity.
const ELEM_CHUNK: usize = 1 << 13;

static THREADS: LazyLock<usize> =
    LazyLock::new(|| std::thread::available_parallelism().map_or(1, |n| n.get()));

pub static PAR_LIMIT: AtomicUsize = AtomicUsize::new(0);

pub fn set_par_limit(limit: usize) {
    PAR_LIMIT.store(limit, Ordering::Relaxed);
}
/// Raw output pointer shared across workers that write provably disjoint regions.
#[derive(Clone, Copy)]
struct Out {
    ptr: *mut f32,
    ld: usize,
}
// SAFETY: every user partitions the pointee into disjoint per-task regions.
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

impl Out {
    fn new(buf: &mut [f32], ld: usize) -> Self {
        Out {
            ptr: buf.as_mut_ptr(),
            ld,
        }
    }
    fn at(self, off: usize) -> Self {
        Out {
            ptr: self.ptr.wrapping_add(off),
            ld: self.ld,
        }
    }
}

/// Runs `tasks` independent tasks, pulling indices from a shared counter so the
/// slower efficiency cores simply take fewer tasks. `S` is per-worker scratch.
// ponytail: spawns scoped threads per call (~tens of µs); a persistent pool if small-op overhead shows in profiles.
pub(crate) fn par_tasks<S: Default>(tasks: usize, parallel: bool, f: impl Fn(&mut S, usize) + Sync) {
    let limit = PAR_LIMIT.load(Ordering::Relaxed);
    let max_t = if limit > 0 {
        (*THREADS).min(limit)
    } else {
        *THREADS
    };
    let threads = if parallel { max_t.min(tasks) } else { 1 };
    let next = AtomicUsize::new(0);
    let work = || {
        let mut state = S::default();
        loop {
            let t = next.fetch_add(1, Ordering::Relaxed);
            if t >= tasks {
                break;
            }
            f(&mut state, t);
        }
    };
    if threads <= 1 {
        work();
        return;
    }
    crate::model::par::par_for(threads, &|_| work());
}

/// Applies `f(chunk_index, chunk)` over `chunk`-sized pieces of `x` (last may be short).
fn par_chunks(x: &mut [f32], chunk: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    if x.is_empty() {
        return;
    }
    let len = x.len();
    let base = Out::new(x, 0);
    par_tasks::<()>(len.div_ceil(chunk), len >= PAR_MIN_ELEMS, |_, t| {
        let start = t * chunk;
        // SAFETY: task indices are unique, so chunks are disjoint and inside `x`.
        let c =
            unsafe { std::slice::from_raw_parts_mut(base.at(start).ptr, chunk.min(len - start)) };
        f(t, c);
    });
}

// ---------------------------------------------------------------------------
// GEMM: C[m, n] = A[m, k] . B[n, k]^T, both operands k-contiguous.
// ---------------------------------------------------------------------------

/// Row-major strided view: element (r, c) = data[r * ld + c].
#[derive(Clone, Copy)]
struct Rows<'a> {
    data: &'a [f32],
    ld: usize,
}

/// Source of the B operand, packed into NR-wide k-major panels.
trait PackB: Sync {
    /// Packs rows `j0..j0+nc`, cols `p0..p0+kc` into `dst` as panels of `NR * kc`,
    /// element (j, p) at `panel[p * NR + j]`, zero-padding rows past `nc`.
    fn pack(&self, dst: &mut [f32], j0: usize, nc: usize, p0: usize, kc: usize);
}

fn pack_rows<const R: usize>(
    src: &Rows,
    r0: usize,
    count: usize,
    p0: usize,
    kc: usize,
    dst: &mut [f32],
) {
    let panels = count.div_ceil(R);
    for (pi, panel) in dst[..panels * R * kc].chunks_exact_mut(R * kc).enumerate() {
        for rr in 0..R {
            let r = pi * R + rr;
            if r < count {
                let s = &src.data[(r0 + r) * src.ld + p0..][..kc];
                for (kk, &v) in s.iter().enumerate() {
                    panel[kk * R + rr] = v;
                }
            } else {
                for kk in 0..kc {
                    panel[kk * R + rr] = 0.0;
                }
            }
        }
    }
}

impl PackB for Rows<'_> {
    fn pack(&self, dst: &mut [f32], j0: usize, nc: usize, p0: usize, kc: usize) {
        pack_rows::<NR>(self, j0, nc, p0, kc, dst);
    }
}

/// Implicit im2col: row j = output pixel, col p = (ci, ky, kx) matching the
/// `[c_out, c_in, kh, kw]` weight layout. No patch matrix is materialised.
struct Im2col<'a> {
    x: &'a [f32],
    h: usize,
    w: usize,
    k: usize,
    stride: usize,
    pad: usize,
    ow: usize,
}

impl PackB for Im2col<'_> {
    fn pack(&self, dst: &mut [f32], j0: usize, nc: usize, p0: usize, kc: usize) {
        let (h, w, k) = (self.h, self.w, self.k);
        let hw = h * w;
        let kk2 = k * k;
        let direct = k == 1 && self.stride == 1 && self.pad == 0;
        for (pi, panel) in dst[..nc.div_ceil(NR) * NR * kc]
            .chunks_exact_mut(NR * kc)
            .enumerate()
        {
            let jb = j0 + pi * NR;
            let cnt = NR.min(nc - pi * NR);
            if direct {
                // 1x1 stride 1: B[pixel][ci] = x[ci][pixel], a contiguous run of pixels per channel.
                for (kk, d) in panel.chunks_exact_mut(NR).enumerate() {
                    d[..cnt].copy_from_slice(&self.x[(p0 + kk) * hw + jb..][..cnt]);
                    d[cnt..].fill(0.0);
                }
                continue;
            }
            let mut by = [0isize; NR];
            let mut bx = [0isize; NR];
            for jj in 0..cnt {
                let p = jb + jj;
                by[jj] = ((p / self.ow) * self.stride) as isize - self.pad as isize;
                bx[jj] = ((p % self.ow) * self.stride) as isize - self.pad as isize;
            }
            let (mut ci, mut ky, mut kx) = (p0 / kk2, (p0 % kk2) / k, p0 % k);
            for d in panel.chunks_exact_mut(NR) {
                let plane = &self.x[ci * hw..][..hw];
                for jj in 0..NR {
                    let iy = (by[jj] + ky as isize) as usize;
                    let ix = (bx[jj] + kx as isize) as usize;
                    // Negative coordinates wrap to huge usize and fail the bound check.
                    d[jj] = if jj < cnt && iy < h && ix < w {
                        plane[iy * w + ix]
                    } else {
                        0.0
                    };
                }
                kx += 1;
                if kx == k {
                    kx = 0;
                    ky += 1;
                    if ky == k {
                        ky = 0;
                        ci += 1;
                    }
                }
            }
        }
    }
}

/// tile[MR x NR] = sum over kc of packed A panel (k-major, MR wide) x packed B panel (k-major, NR wide).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn kernel(kc: usize, a: &[f32], b: &[f32], tile: &mut [f32; MR * NR]) {
    use core::arch::aarch64::*;
    assert!(a.len() >= kc * MR && b.len() >= kc * NR);
    // SAFETY: the assert above bounds every load; stores stay inside `tile`.
    unsafe {
        let z = vdupq_n_f32(0.0);
        let mut acc = [[z; 3]; MR];
        let (mut pa, mut pb) = (a.as_ptr(), b.as_ptr());
        for _ in 0..kc {
            let a0 = vld1q_f32(pa);
            let a1 = vld1q_f32(pa.add(4));
            let b0 = vld1q_f32(pb);
            let b1 = vld1q_f32(pb.add(4));
            let b2 = vld1q_f32(pb.add(8));
            macro_rules! row {
                ($i:literal, $a:ident, $l:literal) => {
                    acc[$i][0] = vfmaq_laneq_f32::<$l>(acc[$i][0], b0, $a);
                    acc[$i][1] = vfmaq_laneq_f32::<$l>(acc[$i][1], b1, $a);
                    acc[$i][2] = vfmaq_laneq_f32::<$l>(acc[$i][2], b2, $a);
                };
            }
            row!(0, a0, 0);
            row!(1, a0, 1);
            row!(2, a0, 2);
            row!(3, a0, 3);
            row!(4, a1, 0);
            row!(5, a1, 1);
            row!(6, a1, 2);
            row!(7, a1, 3);
            pa = pa.add(MR);
            pb = pb.add(NR);
        }
        for (i, r) in acc.iter().enumerate() {
            for (v, &q) in r.iter().enumerate() {
                vst1q_f32(tile.as_mut_ptr().add(i * NR + v * 4), q);
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn kernel(kc: usize, a: &[f32], b: &[f32], tile: &mut [f32; MR * NR]) {
    tile.fill(0.0);
    for (ap, bp) in a[..kc * MR]
        .chunks_exact(MR)
        .zip(b[..kc * NR].chunks_exact(NR))
    {
        for (i, &av) in ap.iter().enumerate() {
            for (t, &bv) in tile[i * NR..][..NR].iter_mut().zip(bp) {
                *t += av * bv;
            }
        }
    }
}

/// Computes the full-k result for C rows `i0..i0+mc`, cols `j0..j0+nc`.
#[allow(clippy::too_many_arguments)]
fn gemm_tile<B: PackB>(
    a: &Rows,
    b: &B,
    i0: usize,
    mc: usize,
    j0: usize,
    nc: usize,
    k: usize,
    bias: Option<&[f32]>,
    out: Out,
    scratch: &mut Vec<f32>,
) {
    // SAFETY (all row slices below): rows i0..i0+mc and cols j0..j0+nc of `out` belong
    // to this tile alone; callers size `out` to cover the whole m x n result.
    let out_row = |r: usize, c: usize, len: usize| unsafe {
        std::slice::from_raw_parts_mut(out.ptr.add(r * out.ld + c), len)
    };
    if k == 0 {
        for i in 0..mc {
            let row = out_row(i0 + i, j0, nc);
            match bias {
                Some(bb) => row.copy_from_slice(&bb[j0..j0 + nc]),
                None => row.fill(0.0),
            }
        }
        return;
    }
    let (pm, pn) = (mc.div_ceil(MR), nc.div_ceil(NR));
    let kc_max = k.min(KC);
    let need = (pm * MR + pn * NR) * kc_max;
    if scratch.len() < need {
        scratch.resize(need, 0.0);
    }
    let (ap, bp) = scratch.split_at_mut(pm * MR * kc_max);
    let mut tile = [0.0f32; MR * NR];
    for p0 in (0..k).step_by(KC) {
        let kc = KC.min(k - p0);
        pack_rows::<MR>(a, i0, mc, p0, kc, ap);
        b.pack(bp, j0, nc, p0, kc);
        for jp in 0..pn {
            let bpan = &bp[jp * NR * kc..][..NR * kc];
            let jb = j0 + jp * NR;
            let nr = NR.min(nc - jp * NR);
            for ip in 0..pm {
                kernel(kc, &ap[ip * MR * kc..][..MR * kc], bpan, &mut tile);
                for i in 0..MR.min(mc - ip * MR) {
                    let row = out_row(i0 + ip * MR + i, jb, nr);
                    let src = &tile[i * NR..][..nr];
                    if p0 > 0 {
                        for (d, s) in row.iter_mut().zip(src) {
                            *d += s;
                        }
                    } else if let Some(bb) = bias {
                        for ((d, s), bv) in row.iter_mut().zip(src).zip(&bb[jb..jb + nr]) {
                            *d = s + bv;
                        }
                    } else {
                        row.copy_from_slice(src);
                    }
                }
            }
        }
    }
}

/// Single-threaded GEMM (used inside already-parallel tasks).
#[allow(clippy::too_many_arguments)]
fn gemm_serial<B: PackB>(
    a: &Rows,
    b: &B,
    m: usize,
    n: usize,
    k: usize,
    bias: Option<&[f32]>,
    out: Out,
    scratch: &mut Vec<f32>,
) {
    for i0 in (0..m).step_by(MC) {
        for j0 in (0..n).step_by(NC) {
            gemm_tile(
                a,
                b,
                i0,
                MC.min(m - i0),
                j0,
                NC.min(n - j0),
                k,
                bias,
                out,
                scratch,
            );
        }
    }
}

/// Multithreaded GEMM over MC x NC output tiles.
// ponytail: m < MR (e.g. a single-row GEMV) pads to MR rows; fine while such calls are rare (time embedding).
fn gemm<B: PackB>(a: Rows, b: &B, m: usize, n: usize, k: usize, bias: Option<&[f32]>, out: Out) {
    let tn = n.div_ceil(NC);
    let tiles = m.div_ceil(MC) * tn;
    par_tasks::<Vec<f32>>(tiles, m * n * k >= PAR_MIN_FLOPS, |scratch, t| {
        let (i0, j0) = ((t / tn) * MC, (t % tn) * NC);
        gemm_tile(
            &a,
            b,
            i0,
            MC.min(m - i0),
            j0,
            NC.min(n - j0),
            k,
            bias,
            out,
            scratch,
        );
    });
}

/// out[m*n] = a[m*k] . b[n*k]^T (+ bias[n]).
pub fn matmul_nt(
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    assert_eq!(a.len(), m * k, "matmul_nt: a is not m*k");
    assert_eq!(b.len(), n * k, "matmul_nt: b is not n*k");
    assert_eq!(out.len(), m * n, "matmul_nt: out is not m*n");
    if let Some(bb) = bias {
        assert_eq!(bb.len(), n, "matmul_nt: bias is not n");
    }
    gemm(
        Rows { data: a, ld: k },
        &Rows { data: b, ld: k },
        m,
        n,
        k,
        bias,
        Out::new(out, n),
    );
}

/// B rows produced on demand by `f(j, p0, dst)`: fills `dst` (len <= KC) with B[j][p0..p0 + dst.len()];
/// `p0` is always a multiple of 256, so 32/256-element quant blocks stay aligned.
struct RowFn<'a>(&'a (dyn Fn(usize, usize, &mut [f32]) + Sync));

impl PackB for RowFn<'_> {
    fn pack(&self, dst: &mut [f32], j0: usize, nc: usize, p0: usize, kc: usize) {
        let mut row = [0.0f32; KC];
        for (pi, panel) in dst[..nc.div_ceil(NR) * NR * kc]
            .chunks_exact_mut(NR * kc)
            .enumerate()
        {
            for rr in 0..NR {
                let j = pi * NR + rr;
                if j < nc {
                    (self.0)(j0 + j, p0, &mut row[..kc]);
                } else {
                    row[..kc].fill(0.0);
                }
                for (kk, &v) in row[..kc].iter().enumerate() {
                    panel[kk * NR + rr] = v;
                }
            }
        }
    }
}

/// out[m*n] = a[m*k] . B^T (+ bias[n]) with B [n, k] decoded row-slice by row-slice through `b`
/// (see [`RowFn`]): quantized or bf16 weights go through the f32 GEMM without an f32 copy.
#[allow(clippy::too_many_arguments)]
pub fn matmul_rows(
    a: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    b: &(dyn Fn(usize, usize, &mut [f32]) + Sync),
) {
    assert_eq!(a.len(), m * k, "matmul_rows: a is not m*k");
    assert_eq!(out.len(), m * n, "matmul_rows: out is not m*n");
    if let Some(bb) = bias {
        assert_eq!(bb.len(), n, "matmul_rows: bias is not n");
    }
    gemm(
        Rows { data: a, ld: k },
        &RowFn(b),
        m,
        n,
        k,
        bias,
        Out::new(out, n),
    );
}

/// [`matmul_nt`] with bf16 weights `w` [n, k].
pub fn matmul_bf16(
    a: &[f32],
    w: &[u16],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    assert_eq!(w.len(), n * k, "matmul_bf16: w is not n*k");
    matmul_rows(a, m, k, n, bias, out, &|j, p0, d| {
        for (d, &s) in d.iter_mut().zip(&w[j * k + p0..]) {
            *d = bf16_to_f32(s);
        }
    });
}

/// Direct evaluation of `x [rows, in_f] * w_bf16^T [in_f, out_f] -> out [rows, out_f]`
/// without full-matrix f32 dequantization allocation.
pub fn linear_bf16_direct(
    x: &[f32],
    w_bf16: &[u16],
    rows: usize,
    in_f: usize,
    out_f: usize,
    out: &mut [f32],
) {
    matmul_bf16(x, w_bf16, None, rows, in_f, out_f, out);
}

/// x[rows*in_f] -> [rows*out_f] with PyTorch weight layout [out_f, in_f].
pub fn linear(
    x: &[f32],
    rows: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    in_f: usize,
    out_f: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; rows * out_f];
    matmul_nt(x, weight, bias, rows, in_f, out_f, &mut out);
    out
}

/// [c_in, h, w] -> ([c_out, oh, ow], oh, ow); weight [c_out, c_in, k, k], symmetric zero pad.
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    x: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    c_out: usize,
    k: usize,
    stride: usize,
    pad: usize,
) -> (Vec<f32>, usize, usize) {
    assert_eq!(x.len(), c_in * h * w, "conv2d: input is not c_in*h*w");
    assert_eq!(
        weight.len(),
        c_out * c_in * k * k,
        "conv2d: weight is not c_out*c_in*k*k"
    );
    assert!(
        k > 0 && stride > 0 && h + 2 * pad >= k && w + 2 * pad >= k,
        "conv2d: bad geometry"
    );
    let oh = (h + 2 * pad - k) / stride + 1;
    let ow = (w + 2 * pad - k) / stride + 1;
    let n = oh * ow;
    let mut out = vec![0.0; c_out * n];
    let src = Im2col {
        x,
        h,
        w,
        k,
        stride,
        pad,
        ow,
    };
    gemm(
        Rows {
            data: weight,
            ld: c_in * k * k,
        },
        &src,
        c_out,
        n,
        c_in * k * k,
        None,
        Out::new(&mut out, n),
    );
    if let Some(bb) = bias {
        assert_eq!(bb.len(), c_out, "conv2d: bias is not c_out");
        par_chunks(&mut out, n, |co, plane| {
            for v in plane {
                *v += bb[co];
            }
        });
    }
    (out, oh, ow)
}

/// In-place GroupNorm over x [c, hw]: biased variance per group, then per-channel affine.
pub fn group_norm(
    x: &mut [f32],
    c: usize,
    hw: usize,
    groups: usize,
    eps: f32,
    gamma: &[f32],
    beta: &[f32],
) {
    assert_eq!(x.len(), c * hw, "group_norm: x is not c*hw");
    assert!(
        groups > 0 && c % groups == 0,
        "group_norm: c not divisible by groups"
    );
    assert!(
        gamma.len() == c && beta.len() == c,
        "group_norm: gamma/beta are not c"
    );
    let cg = c / groups;
    par_chunks(x, cg * hw, |g, xs| {
        let n = xs.len() as f64;
        let mean = xs.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = xs.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        let inv = 1.0 / (var + eps as f64).sqrt();
        for (ci, ch) in xs.chunks_exact_mut(hw).enumerate() {
            let sc = (gamma[g * cg + ci] as f64 * inv) as f32;
            let sh = beta[g * cg + ci] - (mean as f32) * sc;
            for v in ch {
                *v = *v * sc + sh;
            }
        }
    });
}

/// LayerNorm over the last dim of x [rows, d] (biased variance).
pub fn layer_norm(
    x: &[f32],
    rows: usize,
    d: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Vec<f32> {
    assert_eq!(x.len(), rows * d, "layer_norm: x is not rows*d");
    assert!(
        gamma.len() == d && beta.len() == d,
        "layer_norm: gamma/beta are not d"
    );
    let mut out = x.to_vec();
    par_chunks(&mut out, d, |_, row| {
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for ((v, g), b) in row.iter_mut().zip(gamma).zip(beta) {
            *v = (*v - mean) * inv * g + b;
        }
    });
    out
}

pub fn silu(x: &mut [f32]) {
    par_chunks(x, ELEM_CHUNK, |_, c| {
        for v in c {
            *v /= 1.0 + (-*v).exp();
        }
    });
}

/// erf via the Numerical Recipes Chebyshev erfc fit; |error| < 1.2e-7 (8.3e-8 measured on [-8, 8]).
fn erf(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.26551223
        + t * (1.00002368
            + t * (0.37409196
                + t * (0.09678418
                    + t * (-0.18628806
                        + t * (0.27886807
                            + t * (-1.13520398
                                + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
    let erfc = t * poly.exp();
    if x >= 0.0 {
        1.0 - erfc
    } else {
        erfc - 1.0
    }
}

/// Exact GELU: 0.5 x (1 + erf(x / sqrt 2)).
pub fn gelu(x: &mut [f32]) {
    par_chunks(x, ELEM_CHUNK, |_, c| {
        for v in c {
            let z = *v as f64;
            *v = (0.5 * z * (1.0 + erf(z * std::f64::consts::FRAC_1_SQRT_2))) as f32;
        }
    });
}

/// Tanh-approximated GELU (`gelu_pytorch_tanh`): 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).
pub fn gelu_tanh(x: &mut [f32]) {
    par_chunks(x, ELEM_CHUNK, |_, c| {
        for v in c {
            let z = *v;
            *v = 0.5 * z * (1.0 + (0.797_884_6 * (z + 0.044715 * z * z * z)).tanh());
        }
    });
}

/// Multi-head scaled dot-product attention.
/// q [nq, heads*head_dim], k/v [nk, heads*head_dim] -> [nq, heads*head_dim].
// ponytail: materialises a QB x nk score block per task (1 MiB at nk=4096) instead of flash-style online softmax.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    nq: usize,
    nk: usize,
    heads: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f32> {
    let hd = heads * head_dim;
    assert_eq!(q.len(), nq * hd, "attention: q is not nq*heads*head_dim");
    assert_eq!(k.len(), nk * hd, "attention: k is not nk*heads*head_dim");
    assert_eq!(v.len(), nk * hd, "attention: v is not nk*heads*head_dim");
    let mut out = vec![0.0; nq * hd];
    if nq == 0 || hd == 0 {
        return out;
    }
    assert!(nk > 0, "attention: no keys");
    let vt = transpose(v, nk, hd); // [hd, nk]: P . V becomes an nt GEMM.
    let scale = 1.0 / (head_dim as f32).sqrt();
    let qblocks = nq.div_ceil(QB);
    let o = Out::new(&mut out, hd);
    par_tasks::<(Vec<f32>, Vec<f32>)>(
        heads * qblocks,
        nq * nk * hd >= PAR_MIN_FLOPS,
        |(scratch, scores), t| {
            let (h, q0) = (t / qblocks, (t % qblocks) * QB);
            let qb = QB.min(nq - q0);
            // Keys past the last query of the block are fully masked under causality.
            let kn = if causal { nk.min(q0 + qb) } else { nk };
            scores.resize(qb * kn, 0.0);
            let qs = Rows {
                data: &q[q0 * hd + h * head_dim..],
                ld: hd,
            };
            let ks = Rows {
                data: &k[h * head_dim..],
                ld: hd,
            };
            gemm_serial(
                &qs,
                &ks,
                qb,
                kn,
                head_dim,
                None,
                Out::new(scores, kn),
                scratch,
            );
            for (i, row) in scores.chunks_exact_mut(kn).enumerate() {
                let lim = if causal { (q0 + i + 1).min(kn) } else { kn };
                for s in &mut row[..lim] {
                    *s *= scale;
                }
                vector_softmax(&mut row[..lim]);
                row[lim..].fill(0.0);
            }
            let ps = Rows {
                data: &scores[..],
                ld: kn,
            };
            let vs = Rows {
                data: &vt[h * head_dim * nk..],
                ld: nk,
            };
            // Each task owns rows q0..q0+qb of head h's column band.
            gemm_serial(
                &ps,
                &vs,
                qb,
                head_dim,
                kn,
                None,
                o.at(q0 * hd + h * head_dim),
                scratch,
            );
        },
    );
    out
}

/// [rows, cols] -> [cols, rows], 32x32 cache blocks.
pub fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * cols, "transpose: x is not rows*cols");
    const B: usize = 32;
    let mut out = vec![0.0; rows * cols];
    for r0 in (0..rows).step_by(B) {
        for c0 in (0..cols).step_by(B) {
            for r in r0..(r0 + B).min(rows) {
                for c in c0..(c0 + B).min(cols) {
                    out[c * rows + r] = x[r * cols + c];
                }
            }
        }
    }
    out
}

pub fn add_in_place(dst: &mut [f32], src: &[f32]) {
    dst.iter_mut().zip(src).for_each(|(d, s)| *d += s);
}

/// Mean of rows of `x` [n, d] whose mask entry is non-zero.
pub fn mean_pool(x: &[f32], mask: &[u32], d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; d];
    let mut count = 0usize;
    for (row, &m) in x.chunks_exact(d).zip(mask) {
        if m != 0 {
            add_in_place(&mut out, row);
            count += 1;
        }
    }
    let c = count.max(1) as f32;
    out.iter_mut().for_each(|v| *v /= c);
    out
}

pub fn l2_normalize(v: &mut [f32]) {
    let n = v
        .iter()
        .zip(v.iter())
        .map(|(x, y)| x * y)
        .sum::<f32>()
        .sqrt()
        .max(1e-12);
    v.iter_mut().for_each(|x| *x /= n);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_respects_mask() {
        let x = [1.0, 2.0, 3.0, 4.0, 100.0, 100.0];
        assert_eq!(mean_pool(&x, &[1, 1, 0], 2), vec![2.0, 3.0]);
    }

    #[test]
    fn l2_normalize_unit_length() {
        let mut v = [3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
    }

    struct Lcg(u64);
    impl Lcg {
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n)
                .map(|_| {
                    self.0 = self
                        .0
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
                })
                .collect()
        }
    }

    fn close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                "index {i}: got {g}, want {w}"
            );
        }
    }

    fn naive_matmul(
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = bias.map_or(0.0, |bb| bb[j] as f64);
                for p in 0..k {
                    s += a[i * k + p] as f64 * b[j * k + p] as f64;
                }
                out[i * n + j] = s as f32;
            }
        }
        out
    }

    #[test]
    fn matmul_matches_naive() {
        let mut r = Lcg(1);
        // Tiny serial case, multi-tile parallel case with k spanning two KC blocks, single-row GEMV.
        for &(m, k, n) in &[(5, 7, 3), (137, 300, 203), (1, 513, 250)] {
            let a = r.vec(m * k);
            let b = r.vec(n * k);
            let bias = r.vec(n);
            for bias in [None, Some(bias.as_slice())] {
                let mut out = vec![f32::NAN; m * n];
                matmul_nt(&a, &b, bias, m, k, n, &mut out);
                close(&out, &naive_matmul(&a, &b, bias, m, k, n));
            }
        }
    }

    #[test]
    fn matmul_bf16_matches_f32_on_bf16_exact_weights() {
        let mut r = Lcg(3);
        for &(m, k, n) in &[(3, 5, 7), (130, 600, 200), (1, 300, 13)] {
            let a = r.vec(m * k);
            let w: Vec<u16> = r
                .vec(n * k)
                .into_iter()
                .map(crate::model::f32_to_bf16)
                .collect();
            let wf: Vec<f32> = w.iter().map(|&v| bf16_to_f32(v)).collect();
            let bias = r.vec(n);
            let mut out = vec![f32::NAN; m * n];
            matmul_bf16(&a, &w, Some(&bias), m, k, n, &mut out);
            close(&out, &naive_matmul(&a, &wf, Some(&bias), m, k, n));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn naive_conv(
        x: &[f32],
        c_in: usize,
        h: usize,
        w: usize,
        wt: &[f32],
        bias: &[f32],
        c_out: usize,
        k: usize,
        s: usize,
        p: usize,
    ) -> Vec<f32> {
        let oh = (h + 2 * p - k) / s + 1;
        let ow = (w + 2 * p - k) / s + 1;
        let mut out = vec![0.0; c_out * oh * ow];
        for co in 0..c_out {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = bias[co] as f64;
                    for ci in 0..c_in {
                        for ky in 0..k {
                            for kx in 0..k {
                                let iy = (oy * s + ky) as isize - p as isize;
                                let ix = (ox * s + kx) as isize - p as isize;
                                if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                                    acc += x[ci * h * w + iy as usize * w + ix as usize] as f64
                                        * wt[((co * c_in + ci) * k + ky) * k + kx] as f64;
                                }
                            }
                        }
                    }
                    out[(co * oh + oy) * ow + ox] = acc as f32;
                }
            }
        }
        out
    }

    #[test]
    fn conv2d_matches_naive() {
        let mut r = Lcg(2);
        for &(c_in, h, w, c_out, k, s, p) in &[
            (5, 9, 11, 7, 3, 1, 1),
            (5, 9, 11, 7, 3, 2, 1),
            (5, 9, 11, 7, 1, 1, 0),
            (32, 20, 19, 40, 3, 1, 1), // parallel, K = 288 spans two KC blocks
            (32, 20, 19, 40, 1, 1, 0),
        ] {
            let x = r.vec(c_in * h * w);
            let wt = r.vec(c_out * c_in * k * k);
            let bias = r.vec(c_out);
            let (got, oh, ow) = conv2d(&x, c_in, h, w, &wt, Some(bias.as_slice()), c_out, k, s, p);
            assert_eq!((oh, ow), ((h + 2 * p - k) / s + 1, (w + 2 * p - k) / s + 1));
            close(
                &got,
                &naive_conv(&x, c_in, h, w, &wt, &bias, c_out, k, s, p),
            );
        }
    }

    #[test]
    fn group_norm_matches_naive() {
        let mut r = Lcg(3);
        let (c, hw, groups, eps) = (64, 35, 32, 1e-5);
        let x = r
            .vec(c * hw)
            .iter()
            .map(|v| v * 3.0 + 0.5)
            .collect::<Vec<_>>();
        let gamma = r.vec(c);
        let beta = r.vec(c);
        let mut got = x.clone();
        group_norm(&mut got, c, hw, groups, eps, &gamma, &beta);
        let cg = c / groups;
        let mut want = vec![0.0; c * hw];
        for g in 0..groups {
            let xs = &x[g * cg * hw..(g + 1) * cg * hw];
            let n = xs.len() as f64;
            let mean = xs.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = xs.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
            for i in 0..cg * hw {
                let ch = g * cg + i / hw;
                let norm = (xs[i] as f64 - mean) / (var + eps as f64).sqrt();
                want[g * cg * hw + i] = (norm * gamma[ch] as f64 + beta[ch] as f64) as f32;
            }
        }
        close(&got, &want);
    }

    #[allow(clippy::too_many_arguments)]
    fn naive_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        nq: usize,
        nk: usize,
        heads: usize,
        d: usize,
        causal: bool,
    ) -> Vec<f32> {
        let hd = heads * d;
        let mut out = vec![0.0; nq * hd];
        for h in 0..heads {
            for i in 0..nq {
                let lim = if causal { (i + 1).min(nk) } else { nk };
                let s: Vec<f64> = (0..lim)
                    .map(|j| {
                        (0..d)
                            .map(|e| q[i * hd + h * d + e] as f64 * k[j * hd + h * d + e] as f64)
                            .sum::<f64>()
                            / (d as f64).sqrt()
                    })
                    .collect();
                let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
                let z: f64 = e.iter().sum();
                for c in 0..d {
                    let acc: f64 = (0..lim)
                        .map(|j| e[j] / z * v[j * hd + h * d + c] as f64)
                        .sum();
                    out[i * hd + h * d + c] = acc as f32;
                }
            }
        }
        out
    }

    #[test]
    fn attention_matches_naive() {
        let mut r = Lcg(4);
        // nq spans several QB blocks (last partial); nk both above and below nq.
        for &(nq, nk, heads, d) in &[(70, 50, 3, 8), (130, 150, 4, 16), (130, 40, 2, 24)] {
            let hd = heads * d;
            let q = r.vec(nq * hd);
            let k = r.vec(nk * hd);
            let v = r.vec(nk * hd);
            for causal in [false, true] {
                close(
                    &attention(&q, &k, &v, nq, nk, heads, d, causal),
                    &naive_attention(&q, &k, &v, nq, nk, heads, d, causal),
                );
            }
        }
    }

    #[test]
    fn gelu_known_points() {
        let mut x = [0.0f32, 0.5, 1.0, -1.0, 2.0, -3.0];
        gelu(&mut x);
        close(
            &x,
            &[
                0.0,
                0.345_731_23,
                0.841_344_75,
                -0.158_655_25,
                1.954_499_7,
                -0.004_049_694,
            ],
        );
        // Small-magnitude values must be accurate relative to themselves, not just absolutely.
        let mut t = [-0.003_f32];
        gelu(&mut t);
        let want = 0.5 * -0.003 * (1.0 + erf_series(-0.003 / std::f64::consts::SQRT_2));
        assert!((t[0] as f64 - want).abs() <= 1e-4 * want.abs());
    }

    /// Maclaurin series of erf, exact to f64 for |x| < 0.1.
    fn erf_series(x: f64) -> f64 {
        let x2 = x * x;
        2.0 / std::f64::consts::PI.sqrt()
            * x
            * (1.0 - x2 / 3.0 + x2 * x2 / 10.0 - x2 * x2 * x2 / 42.0)
    }

    #[test]
    fn transpose_matches_naive() {
        let mut r = Lcg(5);
        let (rows, cols) = (37, 53);
        let m = r.vec(rows * cols);
        let t = transpose(&m, rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(t[j * rows + i], m[i * cols + j]);
            }
        }
    }
}
