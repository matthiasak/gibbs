//! CPU parallelism shared by the native models: one persistent worker pool for the process, a
//! disjoint-output pointer for its tasks, and the bf16 x f32 dot product their linears stream.

use crate::model::bf16_to_f32;
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};

/// A job published to the workers: `f(t)` for every `t < tasks`, indices handed out by `next`.
/// Shared by `Arc`, so a worker that wakes after the job finished only bumps `next` past
/// `tasks` and never touches `f`.
struct Job {
    f: *const (dyn Fn(usize) + Sync),
    tasks: usize,
    next: AtomicUsize,
    done: AtomicUsize,
    panicked: AtomicBool,
    /// Worker seats left: `nn::par_limit()` participants, the caller included.
    seats: AtomicUsize,
}

thread_local! {
    /// Set while a thread runs pool tasks: a nested `par_for` runs inline instead of deadlocking.
    static IN_POOL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
// SAFETY: `f` is called only for claimed task indices, all of which complete before `par_for`
// (which owns the closure) returns.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

impl Job {
    fn admit(&self) -> bool {
        self.seats
            .try_update(Ordering::AcqRel, Ordering::Acquire, |s| s.checked_sub(1))
            .is_ok()
    }

    fn run(&self) {
        IN_POOL.set(true);
        loop {
            let t = self.next.fetch_add(1, Ordering::Relaxed);
            if t >= self.tasks {
                break;
            }
            // SAFETY: t is unclaimed, so `par_for` is still waiting on it and `f` is alive.
            let f = unsafe { &*self.f };
            // A panicking task still counts as done; `par_for` re-raises after the join.
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(t))).is_err() {
                self.panicked.store(true, Ordering::Relaxed);
            }
            self.done.fetch_add(1, Ordering::Release);
        }
        IN_POOL.set(false);
    }
}

/// Persistent workers: a decode step issues hundreds of parallel ops, too many to spawn threads
/// for. Completion waits on tasks, not on workers, so a descheduled worker never stalls a step.
struct Pool {
    /// Serialises callers (one job at a time).
    submit: Mutex<()>,
    job: Mutex<(u64, Option<Arc<Job>>)>,
    /// Mirror of the job epoch for workers spinning between back-to-back jobs.
    epoch: AtomicU64,
    wake: Condvar,
    workers: usize,
}

/// Spins a worker makes for a new job before sleeping (~tens of µs).
const SPIN: usize = 1 << 14;

static POOL: LazyLock<Pool> = LazyLock::new(|| {
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get()) - 1;
    for _ in 0..workers {
        std::thread::spawn(|| {
            // macOS runs user-interactive threads on the performance cores first.
            #[cfg(target_os = "macos")]
            // SAFETY: sets the calling thread's own QoS class.
            unsafe {
                libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
            }
            let pool = &*POOL;
            let mut seen = 0;
            loop {
                for _ in 0..SPIN {
                    if pool.epoch.load(Ordering::Acquire) != seen {
                        break;
                    }
                    std::hint::spin_loop();
                }
                let job = {
                    let mut st = pool.job.lock();
                    while st.0 == seen {
                        pool.wake.wait(&mut st);
                    }
                    seen = st.0;
                    st.1.clone()
                };
                if let Some(job) = job.filter(|j| j.admit()) {
                    job.run();
                }
            }
        });
    }
    Pool {
        submit: Mutex::new(()),
        job: Mutex::new((0, None)),
        epoch: AtomicU64::new(0),
        wake: Condvar::new(),
        workers,
    }
});

/// Number of threads `par_for` runs on.
pub fn threads() -> usize {
    POOL.workers + 1
}

/// Runs `f(t)` for `t in 0..tasks` across the pool (the caller works too), on at most
/// `nn::par_limit()` threads. Tasks must touch disjoint outputs.
// ponytail: one job at a time across all callers; per-sequence pools if concurrent batches matter.
pub fn par_for(tasks: usize, f: &(dyn Fn(usize) + Sync)) {
    let pool = &*POOL;
    let limit = crate::model::nn::par_limit();
    if tasks <= 1 || pool.workers == 0 || limit <= 1 || IN_POOL.get() {
        (0..tasks).for_each(f);
        return;
    }
    let _one = pool.submit.lock();
    // SAFETY: the erased lifetime is never used past this call: we return only after every task
    // index has been claimed and finished.
    let f: &'static (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
    let job = Arc::new(Job {
        f,
        tasks,
        next: AtomicUsize::new(0),
        done: AtomicUsize::new(0),
        panicked: Default::default(),
        seats: AtomicUsize::new(limit - 1),
    });
    {
        let mut st = pool.job.lock();
        st.0 += 1;
        st.1 = Some(job.clone());
        pool.epoch.store(st.0, Ordering::Release);
        pool.wake.notify_all();
    }
    job.run();
    let mut spins = 0u32;
    while job.done.load(Ordering::Acquire) < tasks {
        spins += 1;
        if spins < 1 << 12 {
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
    pool.job.lock().1 = None;
    if job.panicked.load(Ordering::Relaxed) {
        panic!("a parallel task panicked");
    }
}

/// Shared output pointer for tasks writing disjoint regions.
#[derive(Clone, Copy)]
pub struct OutPtr(*mut f32);
// SAFETY: users partition the pointee into disjoint per-task regions.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

impl OutPtr {
    pub fn new(buf: &mut [f32]) -> Self {
        OutPtr(buf.as_mut_ptr())
    }
    /// SAFETY: `off..off + len` must lie inside the buffer and belong to this task alone.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice(&self, off: usize, len: usize) -> &mut [f32] {
        std::slice::from_raw_parts_mut(self.0.add(off), len)
    }
    /// [`Self::new`] over spare capacity (`Vec::spare_capacity_mut`), filled with [`Self::write`]
    /// before the owner's `set_len`; saves zero-filling a buffer that is about to be overwritten.
    pub fn uninit(buf: &mut [std::mem::MaybeUninit<f32>]) -> Self {
        OutPtr(buf.as_mut_ptr().cast())
    }
    /// Copies `src` to `off..`. SAFETY: as [`Self::slice`].
    pub unsafe fn write(&self, off: usize, src: &[f32]) {
        std::ptr::copy_nonoverlapping(src.as_ptr(), self.0.add(off), src.len());
    }
}

/// `f(row_index, row)` over `rows`-sized chunks of `x`, in parallel when there is enough work.
pub fn par_rows(x: &mut [f32], width: usize, f: &(dyn Fn(usize, &mut [f32]) + Sync)) {
    let rows = x.len() / width.max(1);
    if rows * width < 1 << 14 {
        x.chunks_exact_mut(width)
            .enumerate()
            .for_each(|(i, r)| f(i, r));
        return;
    }
    let out = OutPtr::new(x);
    let per = rows.div_ceil(threads() * 4);
    par_for(rows.div_ceil(per), &|t| {
        for r in t * per..((t + 1) * per).min(rows) {
            // SAFETY: row r belongs to task t only.
            f(r, unsafe { out.slice(r * width, width) });
        }
    });
}

/// dot(bf16 w, f32 x).
#[inline]
pub fn dot_bf16(w: &[u16], x: &[f32]) -> f32 {
    debug_assert_eq!(w.len(), x.len());
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; loads stay inside `n` elements.
    unsafe {
        use std::arch::aarch64::*;
        let n = w.len().min(x.len());
        let (wp, xp) = (w.as_ptr(), x.as_ptr());
        let mut acc = [vdupq_n_f32(0.0); 4];
        let mut i = 0;
        while i + 16 <= n {
            let a = vld1q_u16(wp.add(i));
            let b = vld1q_u16(wp.add(i + 8));
            let w0 = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(a)));
            let w1 = vreinterpretq_f32_u32(vshll_high_n_u16::<16>(a));
            let w2 = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(b)));
            let w3 = vreinterpretq_f32_u32(vshll_high_n_u16::<16>(b));
            acc[0] = vfmaq_f32(acc[0], w0, vld1q_f32(xp.add(i)));
            acc[1] = vfmaq_f32(acc[1], w1, vld1q_f32(xp.add(i + 4)));
            acc[2] = vfmaq_f32(acc[2], w2, vld1q_f32(xp.add(i + 8)));
            acc[3] = vfmaq_f32(acc[3], w3, vld1q_f32(xp.add(i + 12)));
            i += 16;
        }
        let mut s = vaddvq_f32(vaddq_f32(
            vaddq_f32(acc[0], acc[1]),
            vaddq_f32(acc[2], acc[3]),
        ));
        while i < n {
            s += bf16_to_f32(w[i]) * x[i];
            i += 1;
        }
        s
    }
    #[cfg(not(target_arch = "aarch64"))]
    w.iter().zip(x).map(|(&w, &x)| bf16_to_f32(w) * x).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_runs_every_task_once() {
        let hits: Vec<AtomicUsize> = (0..1000).map(|_| AtomicUsize::new(0)).collect();
        for _ in 0..50 {
            par_for(hits.len(), &|t| {
                hits[t].fetch_add(1, Ordering::Relaxed);
            });
        }
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 50));
    }

    #[test]
    fn pool_nests_inline_and_reraises_panics() {
        let hits = AtomicUsize::new(0);
        par_for(16, &|_| {
            par_for(8, &|_| {
                hits.fetch_add(1, Ordering::Relaxed);
            })
        });
        assert_eq!(hits.load(Ordering::Relaxed), 128);
        let r = std::panic::catch_unwind(|| par_for(64, &|t| assert!(t != 37)));
        assert!(r.is_err());
        par_for(64, &|_| {}); // the pool still works afterwards
    }

    #[test]
    fn dot_bf16_matches_scalar() {
        let w: Vec<u16> = (0..37)
            .map(|i| crate::model::f32_to_bf16(i as f32 * 0.25 - 3.0))
            .collect();
        let x: Vec<f32> = (0..37).map(|i| (i as f32).sin()).collect();
        let want: f32 = w.iter().zip(&x).map(|(&w, &x)| bf16_to_f32(w) * x).sum();
        assert!((dot_bf16(&w, &x) - want).abs() < 1e-4);
    }
}
