//! Direct quantized linears for the CPU executor. Activations are quantized once per call into
//! ggml's `Q8_0` (32 × i8, f32 `d`) or `Q8_K` (256 × i8, f32 `d`, 16 × i16 `bsums`) layout and
//! dotted against the packed weight blocks with `sdot`; every other format decodes each weight
//! block once per task and FMAs it against every activation row. CPU runs that serve as a
//! numerical reference hold [`exact`], which keeps activations in f32.

use crate::hardware::QuantFormat as Q;
use crate::model::nn;
use crate::model::par::OutPtr;
use crate::model::weights::Mat;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

static EXACT: AtomicUsize = AtomicUsize::new(0);

/// While held (by any thread), linears keep activations in f32 instead of quantizing them to
/// int8: load-time constant evaluation, parity checks and kernel tests use the CPU as an exact
/// reference.
#[must_use]
pub fn exact() -> ExactGuard {
    EXACT.fetch_add(1, Ordering::SeqCst);
    ExactGuard(())
}

pub struct ExactGuard(());

impl Drop for ExactGuard {
    fn drop(&mut self) {
        EXACT.fetch_sub(1, Ordering::SeqCst);
    }
}

fn quantize_activations() -> bool {
    EXACT.load(Ordering::SeqCst) == 0
}

/// Output rows per task.
const TASK_ROWS: usize = 64;

/// Kernel family `linear_q` runs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Path {
    Generic,
    Dot,
    I8mm,
}

/// Quantized activation layout a dot kernel reads.
#[derive(Clone, Copy, PartialEq)]
enum Act {
    Q80,
    Q8K,
}

impl Act {
    fn block(self) -> usize {
        match self {
            Act::Q80 => 32,
            Act::Q8K => 256,
        }
    }
}

fn act_of(f: Q) -> Option<Act> {
    match f {
        Q::GGUF_Q4_0 | Q::GGUF_Q8_0 | Q::GGUF_IQ4_NL => Some(Act::Q80),
        Q::GGUF_Q4_K_M | Q::GGUF_Q6_K | Q::GGUF_Q2_K | Q::GGUF_TQ2_0 => Some(Act::Q8K),
        _ => None,
    }
}

fn dotprod() -> bool {
    #[cfg(target_arch = "aarch64")]
    return std::arch::is_aarch64_feature_detected!("dotprod");
    #[cfg(not(target_arch = "aarch64"))]
    false
}

fn i8mm() -> bool {
    #[cfg(target_arch = "aarch64")]
    return std::arch::is_aarch64_feature_detected!("i8mm");
    #[cfg(not(target_arch = "aarch64"))]
    false
}

/// `linear_q` runs `smmla` kernels for `f` when the activation has two or more rows (never under
/// [`exact`]).
pub fn i8mm_format(f: Q) -> bool {
    quantize_activations() && i8mm() && kernel(Path::I8mm, f).is_some()
}

/// Quantized activations: `qs` [rows, cols], `d` one scale per block, `bs` per-16 sums (`Q8_K`).
#[derive(Default)]
struct QAct {
    qs: Vec<i8>,
    d: Vec<f32>,
    bs: Vec<i16>,
}

thread_local! {
    /// Grown, never shrunk: a call reuses the previous call's capacity.
    static SCRATCH: RefCell<QAct> = RefCell::default();
}

fn quantize(act: Act, x: &[f32], cols: usize, a: &mut QAct) {
    let blk = act.block();
    a.qs.resize(x.len(), 0);
    a.d.resize(x.len() / blk, 0.0);
    for ((xb, qb), d) in x
        .chunks_exact(blk)
        .zip(a.qs.chunks_exact_mut(blk))
        .zip(a.d.iter_mut())
    {
        let amax = xb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        *d = amax / 127.0;
        let id = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        for (q, &v) in qb.iter_mut().zip(xb) {
            *q = (v * id).round() as i8;
        }
    }
    if act == Act::Q8K {
        a.bs.resize(x.len() / 16, 0);
        for (s, q) in a.bs.iter_mut().zip(a.qs.chunks_exact(16)) {
            *s = q.iter().map(|&v| v as i16).sum();
        }
    }
    debug_assert_eq!(x.len() % cols, 0);
}

/// Quantized activation rows starting at one row of a [`QAct`].
#[derive(Clone, Copy)]
struct ActRef<'a> {
    qs: &'a [i8],
    d: &'a [f32],
    bs: &'a [i16],
}

impl QAct {
    fn view(&self, act: Act, cols: usize, r0: usize) -> ActRef<'_> {
        ActRef {
            qs: &self.qs[r0 * cols..],
            d: &self.d[r0 * cols / act.block()..],
            bs: if act == Act::Q8K {
                &self.bs[r0 * cols / 16..]
            } else {
                &[]
            },
        }
    }
}

/// One linear over weight rows `wrow0 + j` and activation rows `0..rows`; output `(r, j)` lands at
/// `out_off + r * ldo + j`.
struct Job<'a> {
    m: &'a Mat,
    /// Packed bytes of the whole matrix (dot kernels only).
    w: &'a [u8],
    row_bytes: usize,
    wrow0: usize,
    cols: usize,
    rows: usize,
    x: &'a [f32],
    a: ActRef<'a>,
    out: OutPtr,
    out_off: usize,
    ldo: usize,
    bias: Option<&'a [f32]>,
}

impl Job<'_> {
    /// SAFETY: `(r, j)` belongs to the calling task alone and lies inside the output.
    #[inline(always)]
    unsafe fn put(&self, r: usize, j: usize, v: f32) {
        self.out.slice(self.out_off + r * self.ldo + j, 1)[0] = v + self.bias.map_or(0.0, |b| b[j]);
    }
}

/// Computes output rows `j0..j1` of a [`Job`]. SAFETY: tasks own disjoint `j` ranges, and the
/// job's slices cover every row and block the kernel reads.
type Kernel = unsafe fn(&Job, usize, usize);

/// Decodes each 256-wide weight slice once and dots it with up to 8 activation rows at a time.
unsafe fn generic(job: &Job, j0: usize, j1: usize) {
    let cols = job.cols;
    let mut buf = [0.0f32; 256];
    for rc in (0..job.rows).step_by(8) {
        let nr = (job.rows - rc).min(8);
        for j in j0..j1 {
            let mut acc = [0.0f32; 8];
            for p0 in (0..cols).step_by(256) {
                let n = (cols - p0).min(256);
                job.m.decode_slice_f32(job.wrow0 + j, p0, &mut buf[..n]);
                for (r, a) in acc[..nr].iter_mut().enumerate() {
                    *a += crate::model::simd::vector_dot_product(
                        &buf[..n],
                        &job.x[(rc + r) * cols + p0..][..n],
                    );
                }
            }
            for (r, &a) in acc[..nr].iter().enumerate() {
                job.put(rc + r, j, a);
            }
        }
    }
}

fn kernel(path: Path, f: Q) -> Option<Kernel> {
    #[cfg(target_arch = "aarch64")]
    {
        use neon::*;
        let k: Kernel = match (path, f) {
            (Path::Dot, Q::GGUF_Q4_0) => dot32::<Q40>,
            (Path::Dot, Q::GGUF_Q8_0) => dot32::<Q80>,
            (Path::Dot, Q::GGUF_IQ4_NL) => dot32::<Iq4Nl>,
            (Path::Dot, Q::GGUF_Q4_K_M) => dotk::<Q4K>,
            (Path::Dot, Q::GGUF_Q6_K) => dotk::<Q6K>,
            (Path::Dot, Q::GGUF_Q2_K) => dotk::<Q2K>,
            (Path::Dot, Q::GGUF_TQ2_0) => dotk::<Tq20>,
            (Path::I8mm, Q::GGUF_Q4_0) => mm32::<Q40>,
            (Path::I8mm, Q::GGUF_Q8_0) => mm32::<Q80>,
            (Path::I8mm, Q::GGUF_Q4_K_M) => mmk::<Q4K>,
            (Path::I8mm, Q::GGUF_Q6_K) => mmk::<Q6K>,
            _ => return None,
        };
        Some(k)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (path, f);
        None
    }
}

/// The quantized kernel and activation layout for `m`, when one applies.
fn quant_kernel(m: &Mat, path: Path) -> Option<(Kernel, Act)> {
    let f = m.packed()?.format;
    let act = act_of(f)?;
    if m.cols % act.block() != 0 {
        return None;
    }
    Some((kernel(path, f)?, act))
}

fn path_for(m: &Mat, rows: usize) -> Path {
    match m.packed() {
        Some(p) if rows >= 2 && i8mm_format(p.format) => Path::I8mm,
        _ if dotprod() && quantize_activations() => Path::Dot,
        _ => Path::Generic,
    }
}

/// `out[rows, m.rows] = x[rows, m.cols] · mᵀ (+ bias)`, straight from `m`'s stored encoding.
pub fn linear_q(m: &Mat, x: &[f32], rows: usize, bias: Option<&[f32]>, out: &mut [f32]) {
    linear_with(m, x, rows, bias, out, path_for(m, rows));
}

fn linear_with(m: &Mat, x: &[f32], rows: usize, bias: Option<&[f32]>, out: &mut [f32], path: Path) {
    let (n, cols) = (m.rows, m.cols);
    assert_eq!(x.len(), rows * cols, "linear_q: x is not rows * cols");
    assert_eq!(out.len(), rows * n, "linear_q: out is not rows * n");
    assert!(bias.is_none_or(|b| b.len() == n), "linear_q: bias is not n");
    let job = Job {
        m,
        w: &[],
        row_bytes: 0,
        wrow0: 0,
        cols,
        rows,
        x,
        a: ActRef { qs: &[], d: &[], bs: &[] },
        out: OutPtr::new(out),
        out_off: 0,
        ldo: n,
        bias,
    };
    let parallel = rows * n * cols >= nn::PAR_MIN_FLOPS;
    let tasks = n.div_ceil(TASK_ROWS);
    let run = |job: &Job, k: Kernel| {
        // SAFETY: task `t` owns output rows `t * TASK_ROWS..`; `bind` sized every input slice.
        nn::par_tasks::<()>(tasks, parallel, |_, t| unsafe {
            k(job, t * TASK_ROWS, ((t + 1) * TASK_ROWS).min(n))
        });
    };
    let Some((k, act)) = quant_kernel(m, path) else {
        return run(&job, generic);
    };
    SCRATCH.with_borrow_mut(move |a| {
        let mut job: Job<'_> = job;
        quantize(act, x, cols, a);
        bind(&mut job, act, a, 0);
        run(&job, k);
    });
}

/// Points `job` at `m`'s packed bytes and activation row `r0` of `a`, checking every bound the
/// kernels read without checks.
fn bind<'a>(job: &mut Job<'a>, act: Act, a: &'a QAct, r0: usize) {
    let p = job.m.packed().expect("quantized kernels read packed weights");
    job.row_bytes = p.format.tensor_bytes(job.cols);
    job.w = &p.map[p.off..p.off + job.m.rows * job.row_bytes];
    job.a = a.view(act, job.cols, r0);
    assert!(job.a.qs.len() >= job.rows * job.cols);
}

/// `Op::MoeLinear` rows: `out[s * n + j] = m[w_row[s] + j] · x[x_row[s]]` for every slot `s` and
/// `j < n`; each `x` row is quantized once however many slots read it.
pub(crate) fn gather_linear(
    m: &Mat,
    x: &[f32],
    slots: &[(usize, usize)],
    n: usize,
    out: &mut [f32],
) {
    let cols = m.cols;
    assert_eq!(x.len() % cols, 0, "gather_linear: x is not rows * cols");
    assert_eq!(out.len(), slots.len() * n, "gather_linear: out is not slots * n");
    assert!(
        slots
            .iter()
            .all(|&(xr, wr)| (xr + 1) * cols <= x.len() && wr + n <= m.rows),
        "gather_linear: slot out of range"
    );
    let o = OutPtr::new(out);
    let blocks = n.div_ceil(TASK_ROWS);
    let parallel = slots.len() * n * cols >= nn::PAR_MIN_FLOPS;
    let quant = quant_kernel(m, path_for(m, 1));
    let go = |a: Option<&QAct>| {
        nn::par_tasks::<()>(slots.len() * blocks, parallel, |_, t| {
            let (s, b) = (t / blocks, t % blocks);
            let (xr, wr) = slots[s];
            let mut job = Job {
                m,
                w: &[],
                row_bytes: 0,
                wrow0: wr,
                cols,
                rows: 1,
                x: &x[xr * cols..][..cols],
                a: ActRef { qs: &[], d: &[], bs: &[] },
                out: o,
                out_off: s * n,
                ldo: n,
                bias: None,
            };
            let k = match (quant, a) {
                (Some((k, act)), Some(a)) => {
                    bind(&mut job, act, a, xr);
                    k
                }
                _ => generic,
            };
            // SAFETY: task `t` owns cells `(s, b * TASK_ROWS..)`; `bind` checked the inputs.
            unsafe { k(&job, b * TASK_ROWS, ((b + 1) * TASK_ROWS).min(n)) }
        });
    };
    match quant {
        Some((_, act)) => SCRATCH.with_borrow_mut(|a| {
            quantize(act, x, cols, a);
            go(Some(a));
        }),
        None => go(None),
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::Job;
    use core::arch::aarch64::*;

    #[inline(always)]
    unsafe fn f16(b: *const u8) -> f32 {
        crate::model::f16_to_f32(u16::from_le_bytes([*b, *b.add(1)]))
    }

    /// A 32-element weight block as signed values plus its scale.
    pub trait B32 {
        const BYTES: usize;
        unsafe fn unpack(b: *const u8) -> (int8x16_t, int8x16_t, f32);
    }

    /// Low nibbles are elements 0..16, high nibbles 16..32 (Q4_0 order).
    #[inline(always)]
    unsafe fn nibbles(b: *const u8) -> (uint8x16_t, uint8x16_t) {
        let q = vld1q_u8(b);
        (vandq_u8(q, vdupq_n_u8(0x0F)), vshrq_n_u8::<4>(q))
    }

    pub struct Q40;
    impl B32 for Q40 {
        const BYTES: usize = 18;
        #[inline(always)]
        unsafe fn unpack(b: *const u8) -> (int8x16_t, int8x16_t, f32) {
            let (lo, hi) = nibbles(b.add(2));
            let off = vdupq_n_s8(8);
            (
                vsubq_s8(vreinterpretq_s8_u8(lo), off),
                vsubq_s8(vreinterpretq_s8_u8(hi), off),
                f16(b),
            )
        }
    }

    pub struct Q80;
    impl B32 for Q80 {
        const BYTES: usize = 34;
        #[inline(always)]
        unsafe fn unpack(b: *const u8) -> (int8x16_t, int8x16_t, f32) {
            let q = b.add(2) as *const i8;
            (vld1q_s8(q), vld1q_s8(q.add(16)), f16(b))
        }
    }

    pub struct Iq4Nl;
    impl B32 for Iq4Nl {
        const BYTES: usize = 18;
        #[inline(always)]
        unsafe fn unpack(b: *const u8) -> (int8x16_t, int8x16_t, f32) {
            let t = vld1q_s8(crate::model::layers::KVALUES_IQ4NL.as_ptr());
            let (lo, hi) = nibbles(b.add(2));
            (vqtbl1q_s8(t, lo), vqtbl1q_s8(t, hi), f16(b))
        }
    }

    /// A 256-element K-quant block: value `i` of 16-group `g` is
    /// `d · sc[g] · q[g][i] − dmin · mn[g]`.
    pub struct KBlk {
        pub q: [int8x16_t; 16],
        pub sc: [i32; 16],
        pub mn: [i16; 16],
        pub d: f32,
        pub dmin: f32,
    }

    pub trait BK {
        const BYTES: usize;
        unsafe fn unpack(b: *const u8) -> KBlk;
    }

    pub struct Q4K;
    impl BK for Q4K {
        const BYTES: usize = 144;
        #[inline(always)]
        unsafe fn unpack(b: *const u8) -> KBlk {
            let s = std::slice::from_raw_parts(b.add(4), 12);
            let mut k = KBlk {
                q: [vdupq_n_s8(0); 16],
                sc: [0; 16],
                mn: [0; 16],
                d: f16(b),
                dmin: f16(b.add(2)),
            };
            for g in 0..4 {
                let (l0, h0) = nibbles(b.add(16 + 32 * g));
                let (l1, h1) = nibbles(b.add(32 + 32 * g));
                k.q[4 * g] = vreinterpretq_s8_u8(l0);
                k.q[4 * g + 1] = vreinterpretq_s8_u8(l1);
                k.q[4 * g + 2] = vreinterpretq_s8_u8(h0);
                k.q[4 * g + 3] = vreinterpretq_s8_u8(h1);
                for h in 0..2 {
                    let (sc, m) = crate::model::layers::k4_scale_min(2 * g + h, s);
                    for i in 0..2 {
                        k.sc[4 * g + 2 * h + i] = sc as i32;
                        k.mn[4 * g + 2 * h + i] = m as i16;
                    }
                }
            }
            k
        }
    }

    pub struct Q6K;
    impl BK for Q6K {
        const BYTES: usize = 210;
        #[inline(always)]
        unsafe fn unpack(b: *const u8) -> KBlk {
            let d = f16(b.add(208));
            let mut k = KBlk {
                q: [vdupq_n_s8(0); 16],
                sc: [0; 16],
                mn: [0; 16],
                d,
                dmin: d,
            };
            let (m4, m3) = (vdupq_n_u8(0x0F), vdupq_n_u8(3));
            for n in 0..2 {
                let ql = b.add(64 * n);
                let qh = b.add(128 + 32 * n);
                let (ql0, ql1, ql2, ql3) = (
                    vld1q_u8(ql),
                    vld1q_u8(ql.add(16)),
                    vld1q_u8(ql.add(32)),
                    vld1q_u8(ql.add(48)),
                );
                let (qh0, qh1) = (vld1q_u8(qh), vld1q_u8(qh.add(16)));
                let hi = |h: uint8x16_t| vshlq_n_u8::<4>(vandq_u8(h, m3));
                let q = [
                    vorrq_u8(vandq_u8(ql0, m4), hi(qh0)),
                    vorrq_u8(vandq_u8(ql1, m4), hi(qh1)),
                    vorrq_u8(vandq_u8(ql2, m4), hi(vshrq_n_u8::<2>(qh0))),
                    vorrq_u8(vandq_u8(ql3, m4), hi(vshrq_n_u8::<2>(qh1))),
                    vorrq_u8(vshrq_n_u8::<4>(ql0), hi(vshrq_n_u8::<4>(qh0))),
                    vorrq_u8(vshrq_n_u8::<4>(ql1), hi(vshrq_n_u8::<4>(qh1))),
                    vorrq_u8(vshrq_n_u8::<4>(ql2), hi(vshrq_n_u8::<6>(qh0))),
                    vorrq_u8(vshrq_n_u8::<4>(ql3), hi(vshrq_n_u8::<6>(qh1))),
                ];
                for (i, v) in q.into_iter().enumerate() {
                    let g = 8 * n + i;
                    k.q[g] = vreinterpretq_s8_u8(v);
                    let sc = *b.add(192 + g) as i8 as i32;
                    k.sc[g] = sc;
                    k.mn[g] = (32 * sc) as i16;
                }
            }
            k
        }
    }

    /// 2-bit planes of `qs[0..32]` (Q2_K / TQ2_0 order): group `2s` is `(qs[0..16] >> 2s) & 3`,
    /// group `2s + 1` is `(qs[16..32] >> 2s) & 3`.
    #[inline(always)]
    unsafe fn crumbs(qs: *const u8, q: &mut [int8x16_t]) {
        let m3 = vdupq_n_u8(3);
        let (a, b) = (vld1q_u8(qs), vld1q_u8(qs.add(16)));
        let planes = [
            (a, b),
            (vshrq_n_u8::<2>(a), vshrq_n_u8::<2>(b)),
            (vshrq_n_u8::<4>(a), vshrq_n_u8::<4>(b)),
            (vshrq_n_u8::<6>(a), vshrq_n_u8::<6>(b)),
        ];
        for (s, (x, y)) in planes.into_iter().enumerate() {
            q[2 * s] = vreinterpretq_s8_u8(vandq_u8(x, m3));
            q[2 * s + 1] = vreinterpretq_s8_u8(vandq_u8(y, m3));
        }
    }

    pub struct Q2K;
    impl BK for Q2K {
        const BYTES: usize = 84;
        #[inline(always)]
        unsafe fn unpack(b: *const u8) -> KBlk {
            let mut k = KBlk {
                q: [vdupq_n_s8(0); 16],
                sc: [0; 16],
                mn: [0; 16],
                d: f16(b.add(80)),
                dmin: f16(b.add(82)),
            };
            for n in 0..2 {
                crumbs(b.add(16 + 32 * n), &mut k.q[8 * n..]);
            }
            for g in 0..16 {
                let s = *b.add(g);
                k.sc[g] = (s & 0x0F) as i32;
                k.mn[g] = (s >> 4) as i16;
            }
            k
        }
    }

    pub struct Tq20;
    impl BK for Tq20 {
        const BYTES: usize = 66;
        #[inline(always)]
        unsafe fn unpack(b: *const u8) -> KBlk {
            let d = f16(b.add(64));
            let mut k = KBlk {
                q: [vdupq_n_s8(0); 16],
                sc: [1; 16],
                mn: [1; 16],
                d,
                dmin: d,
            };
            for n in 0..2 {
                crumbs(b.add(32 * n), &mut k.q[8 * n..]);
            }
            k
        }
    }

    /// `Σ mn[g] · bs[g]` over a block's 16 groups.
    #[inline(always)]
    pub unsafe fn mins(mn: &[i16; 16], bs: *const i16) -> i32 {
        let (m0, m1) = (vld1q_s16(mn.as_ptr()), vld1q_s16(mn.as_ptr().add(8)));
        let (b0, b1) = (vld1q_s16(bs), vld1q_s16(bs.add(8)));
        let p = vmull_s16(vget_low_s16(m0), vget_low_s16(b0));
        let p = vmlal_high_s16(p, m0, b0);
        let p = vmlal_s16(p, vget_low_s16(m1), vget_low_s16(b1));
        vaddvq_s32(vmlal_high_s16(p, m1, b1))
    }

    #[target_feature(enable = "dotprod")]
    pub unsafe fn dot32<T: B32>(job: &Job, j0: usize, j1: usize) {
        let (cols, nb) = (job.cols, job.cols / 32);
        let (qs, ds) = (job.a.qs.as_ptr(), job.a.d.as_ptr());
        for rc in (0..job.rows).step_by(8) {
            let nr = (job.rows - rc).min(8);
            for j in j0..j1 {
                let w = job.w.as_ptr().add((job.wrow0 + j) * job.row_bytes);
                let mut acc = [vdupq_n_f32(0.0); 8];
                for b in 0..nb {
                    let (lo, hi, d) = T::unpack(w.add(b * T::BYTES));
                    for (r, acc) in acc[..nr].iter_mut().enumerate() {
                        let y = qs.add((rc + r) * cols + b * 32);
                        let s = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(y));
                        let s = vdotq_s32(s, hi, vld1q_s8(y.add(16)));
                        let dy = *ds.add((rc + r) * nb + b);
                        *acc = vfmaq_n_f32(*acc, vcvtq_f32_s32(s), d * dy);
                    }
                }
                for (r, a) in acc[..nr].iter().enumerate() {
                    job.put(rc + r, j, vaddvq_f32(*a));
                }
            }
        }
    }

    #[target_feature(enable = "dotprod")]
    pub unsafe fn dotk<T: BK>(job: &Job, j0: usize, j1: usize) {
        let nb = job.cols / 256;
        let (qs, ds, bs) = (job.a.qs.as_ptr(), job.a.d.as_ptr(), job.a.bs.as_ptr());
        for rc in (0..job.rows).step_by(8) {
            let nr = (job.rows - rc).min(8);
            for j in j0..j1 {
                let w = job.w.as_ptr().add((job.wrow0 + j) * job.row_bytes);
                let mut acc = [0.0f32; 8];
                for b in 0..nb {
                    let k = T::unpack(w.add(b * T::BYTES));
                    for (r, acc) in acc[..nr].iter_mut().enumerate() {
                        let blk = (rc + r) * nb + b;
                        let y = qs.add(blk * 256);
                        let mut s = vdupq_n_s32(0);
                        for g in 0..16 {
                            let p = vdotq_s32(vdupq_n_s32(0), k.q[g], vld1q_s8(y.add(16 * g)));
                            s = vmlaq_n_s32(s, p, k.sc[g]);
                        }
                        let m = mins(&k.mn, bs.add(blk * 16));
                        *acc += *ds.add(blk) * (k.d * vaddvq_s32(s) as f32 - k.dmin * m as f32);
                    }
                }
                for (r, &a) in acc[..nr].iter().enumerate() {
                    job.put(rc + r, j, a);
                }
            }
        }
    }

    /// `acc + [w0·x0, w0·x1, w1·x0, w1·x1]` over 16 elements: two `smmla` on 8-wide halves.
    #[target_feature(enable = "i8mm")]
    #[inline]
    unsafe fn mm16(
        acc: int32x4_t,
        w0: int8x16_t,
        w1: int8x16_t,
        x0: int8x16_t,
        x1: int8x16_t,
    ) -> int32x4_t {
        let zip = |a: int8x16_t, b: int8x16_t| {
            let (a, b) = (vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b));
            (
                vreinterpretq_s8_s64(vzip1q_s64(a, b)),
                vreinterpretq_s8_s64(vzip2q_s64(a, b)),
            )
        };
        let ((wa, wb), (xa, xb)) = (zip(w0, w1), zip(x0, x1));
        let mut acc = acc;
        // `vmmlaq_s32` is still unstable (`stdarch_neon_i8mm`).
        core::arch::asm!(
            "smmla {acc:v}.4s, {wa:v}.16b, {xa:v}.16b",
            "smmla {acc:v}.4s, {wb:v}.16b, {xb:v}.16b",
            acc = inout(vreg) acc,
            wa = in(vreg) wa,
            xa = in(vreg) xa,
            wb = in(vreg) wb,
            xb = in(vreg) xb,
            options(pure, nomem, nostack),
        );
        acc
    }

    /// Two weight rows × two activation rows per step; output lanes are `(r0, j)`, `(r1, j)`,
    /// `(r0, j + 1)`, `(r1, j + 1)`. An odd last row of either side pairs with itself.
    #[inline(always)]
    unsafe fn put4(job: &Job, r0: usize, r1: usize, j: usize, jb: usize, v: float32x4_t) {
        let v: [f32; 4] = std::mem::transmute(v);
        job.put(r0, j, v[0]);
        if r1 != r0 {
            job.put(r1, j, v[1]);
        }
        if jb != j {
            job.put(r0, jb, v[2]);
            if r1 != r0 {
                job.put(r1, jb, v[3]);
            }
        }
    }

    #[target_feature(enable = "i8mm")]
    pub unsafe fn mm32<T: B32>(job: &Job, j0: usize, j1: usize) {
        let (cols, nb) = (job.cols, job.cols / 32);
        let (qs, ds) = (job.a.qs.as_ptr(), job.a.d.as_ptr());
        for rc in (0..job.rows).step_by(16) {
            let pairs = (job.rows - rc).min(16).div_ceil(2);
            for j in (j0..j1).step_by(2) {
                let jb = (j + 1).min(j1 - 1);
                let w0 = job.w.as_ptr().add((job.wrow0 + j) * job.row_bytes);
                let w1 = job.w.as_ptr().add((job.wrow0 + jb) * job.row_bytes);
                let mut acc = [vdupq_n_f32(0.0); 8];
                for b in 0..nb {
                    let (a0, a1, d0) = T::unpack(w0.add(b * T::BYTES));
                    let (b0, b1, d1) = T::unpack(w1.add(b * T::BYTES));
                    let dw = vcombine_f32(vdup_n_f32(d0), vdup_n_f32(d1));
                    for (p, acc) in acc[..pairs].iter_mut().enumerate() {
                        let r0 = rc + 2 * p;
                        let r1 = (r0 + 1).min(job.rows - 1);
                        let (y0, y1) = (qs.add(r0 * cols + b * 32), qs.add(r1 * cols + b * 32));
                        let s = mm16(vdupq_n_s32(0), a0, b0, vld1q_s8(y0), vld1q_s8(y1));
                        let s = mm16(s, a1, b1, vld1q_s8(y0.add(16)), vld1q_s8(y1.add(16)));
                        let dy = vld1_f32([*ds.add(r0 * nb + b), *ds.add(r1 * nb + b)].as_ptr());
                        *acc = vfmaq_f32(*acc, vcvtq_f32_s32(s), vmulq_f32(dw, vcombine_f32(dy, dy)));
                    }
                }
                for (p, &a) in acc[..pairs].iter().enumerate() {
                    let r0 = rc + 2 * p;
                    put4(job, r0, (r0 + 1).min(job.rows - 1), j, jb, a);
                }
            }
        }
    }

    #[target_feature(enable = "i8mm")]
    pub unsafe fn mmk<T: BK>(job: &Job, j0: usize, j1: usize) {
        let nb = job.cols / 256;
        let (qs, ds, bs) = (job.a.qs.as_ptr(), job.a.d.as_ptr(), job.a.bs.as_ptr());
        let pair = |a: f32, b: f32| vcombine_f32(vdup_n_f32(a), vdup_n_f32(b));
        for rc in (0..job.rows).step_by(16) {
            let pairs = (job.rows - rc).min(16).div_ceil(2);
            for j in (j0..j1).step_by(2) {
                let jb = (j + 1).min(j1 - 1);
                let w0 = job.w.as_ptr().add((job.wrow0 + j) * job.row_bytes);
                let w1 = job.w.as_ptr().add((job.wrow0 + jb) * job.row_bytes);
                let mut acc = [vdupq_n_f32(0.0); 8];
                for b in 0..nb {
                    let k0 = T::unpack(w0.add(b * T::BYTES));
                    let k1 = T::unpack(w1.add(b * T::BYTES));
                    let sc: [int32x4_t; 16] = std::array::from_fn(|g| {
                        vcombine_s32(vdup_n_s32(k0.sc[g]), vdup_n_s32(k1.sc[g]))
                    });
                    let (dw, dm) = (pair(k0.d, k1.d), pair(k0.dmin, k1.dmin));
                    for (p, acc) in acc[..pairs].iter_mut().enumerate() {
                        let r0 = rc + 2 * p;
                        let r1 = (r0 + 1).min(job.rows - 1);
                        let (b0, b1) = (r0 * nb + b, r1 * nb + b);
                        let (y0, y1) = (qs.add(b0 * 256), qs.add(b1 * 256));
                        let mut s = vdupq_n_s32(0);
                        for g in 0..16 {
                            let y = (vld1q_s8(y0.add(16 * g)), vld1q_s8(y1.add(16 * g)));
                            let t = mm16(vdupq_n_s32(0), k0.q[g], k1.q[g], y.0, y.1);
                            s = vmlaq_s32(s, t, sc[g]);
                        }
                        let (s0, s1) = (bs.add(b0 * 16), bs.add(b1 * 16));
                        let m = [
                            mins(&k0.mn, s0) as f32,
                            mins(&k0.mn, s1) as f32,
                            mins(&k1.mn, s0) as f32,
                            mins(&k1.mn, s1) as f32,
                        ];
                        let dy = vld1_f32([*ds.add(b0), *ds.add(b1)].as_ptr());
                        let v = vmulq_f32(dw, vcvtq_f32_s32(s));
                        let v = vfmsq_f32(v, dm, vld1q_f32(m.as_ptr()));
                        *acc = vfmaq_f32(*acc, v, vcombine_f32(dy, dy));
                    }
                }
                for (p, &a) in acc[..pairs].iter().enumerate() {
                    let r0 = rc + 2 * p;
                    put4(job, r0, (r0 + 1).min(job.rows - 1), j, jb, a);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::layers::gguf_block_codec;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn f32(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    /// Random `[n, cols]` blocks of `f` whose decoded values are finite and moderate.
    fn random_weights(f: Q, n: usize, cols: usize, rng: &mut Rng) -> (Vec<u8>, Vec<f32>) {
        let (elems, bytes, dec) = gguf_block_codec(f).unwrap();
        let blocks = n * cols / elems;
        let (mut raw, mut w) = (vec![0u8; blocks * bytes], vec![0.0f32; n * cols]);
        for (b, y) in raw.chunks_exact_mut(bytes).zip(w.chunks_exact_mut(elems)) {
            for attempt in 0.. {
                assert!(attempt < 100_000, "{f:?}: no finite random block");
                b.iter_mut().for_each(|v| *v = rng.next() as u8);
                dec(b, y);
                if y.iter().all(|v| v.is_finite() && v.abs() < 1e4) {
                    break;
                }
            }
        }
        (raw, w)
    }

    #[test]
    fn every_format_matches_dequantized_reference() {
        let mut rng = Rng(7);
        let (n, cols) = (67, 512);
        let mut exercised = std::collections::HashMap::<Path, usize>::new();
        for &f in Q::ALL {
            if gguf_block_codec(f).is_none() {
                continue;
            }
            let (raw, w) = random_weights(f, n, cols, &mut rng);
            let m = Mat::from_packed_bytes(f, raw, n, cols);
            let bias: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            for rows in [1, 2, 7, 16] {
                let x: Vec<f32> = (0..rows * cols).map(|_| rng.f32()).collect();
                let mut paths = vec![Path::Generic];
                if dotprod() && kernel(Path::Dot, f).is_some() {
                    paths.push(Path::Dot);
                }
                if i8mm() && kernel(Path::I8mm, f).is_some() {
                    paths.push(Path::I8mm);
                }
                for path in paths {
                    let mut out = vec![0.0f32; rows * n];
                    linear_with(&m, &x, rows, Some(&bias), &mut out, path);
                    *exercised.entry(path).or_default() += 1;
                    for r in 0..rows {
                        let xr = &x[r * cols..][..cols];
                        let amax = xr.iter().fold(0.0f32, |a, v| a.max(v.abs())) as f64;
                        for j in 0..n {
                            let wr = &w[j * cols..][..cols];
                            let want: f64 = bias[j] as f64
                                + wr.iter().zip(xr).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>();
                            let abs: f64 = wr.iter().zip(xr).map(|(&a, &b)| (a * b).abs() as f64).sum();
                            // Int8 activations err by at most half a step (amax / 254) per element.
                            let quant = if path == Path::Generic {
                                0.0
                            } else {
                                wr.iter().map(|&a| a.abs() as f64).sum::<f64>() * amax / 250.0
                            };
                            let got = out[r * n + j] as f64;
                            assert!(
                                (got - want).abs() <= quant + 1e-4 * abs + 1e-5,
                                "{f:?} {path:?} rows {rows} ({r}, {j}): {got} vs {want}"
                            );
                        }
                    }
                }
            }
        }
        assert!(exercised[&Path::Generic] > 0);
        if dotprod() {
            assert!(exercised[&Path::Dot] >= 7 * 4, "{exercised:?}");
        }
        if i8mm() {
            assert!(exercised[&Path::I8mm] >= 4 * 4, "{exercised:?}");
        }
    }
}
