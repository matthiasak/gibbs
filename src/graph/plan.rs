//! Device-agnostic graph arena planner ([`SharedArenaPlan`]) and runtime shared-library loader ([`DynLib`]).
//!
//! Computes exact SSA value liveness intervals `[first_def, last_use]` for any [`Graph`] and packs
//! non-aliasing intermediate tensors into a single aligned arena buffer (256-byte alignment by
//! default) using best-fit free-span recycling. Used by the portable (CUDA, ROCm, Intel XPU)
//! executor.

use crate::graph::{DType, Graph, ValueId, ValueKind};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};

/// Default arena alignment in bytes across GPU backends (Metal/CUDA/HIP/OpenCL).
pub const DEFAULT_ARENA_ALIGNMENT: usize = 256;

/// Byte offset and byte length of an intermediate value inside a device arena buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaSpan {
    pub offset: usize,
    pub bytes: usize,
    pub first_step: usize,
    pub last_step: usize,
}

/// Result of device-agnostic liveness analysis and best-fit arena packing on a [`Graph`].
#[derive(Debug, Clone, Default)]
pub struct SharedArenaPlan {
    /// Total arena size in bytes required to execute `graph` with zero runtime allocations.
    pub total_bytes: usize,
    /// Per-value arena span for intermediate node outputs.
    pub spans: HashMap<ValueId, ArenaSpan>,
    /// Unpacked sum of all intermediate tensor byte sizes (for memory-savings telemetry).
    pub unpacked_bytes: usize,
}

impl SharedArenaPlan {
    /// Allocates arena offsets for all intermediate node outputs in `graph`.
    pub fn build(graph: &Graph, alignment: usize) -> Self {
        let align = alignment.max(16).next_power_of_two();
        let align_up = |n: usize| (n + align - 1) & !(align - 1);

        let n_nodes = graph.nodes.len();
        let mut def_step: HashMap<ValueId, usize> = HashMap::new();
        let mut last_use: HashMap<ValueId, usize> = HashMap::new();

        for (step, node) in graph.nodes.iter().enumerate() {
            for &out in &node.outputs {
                def_step.entry(out).or_insert(step);
                last_use.entry(out).or_insert(step);
            }
            for &inp in &node.inputs {
                last_use.insert(inp, step);
            }
        }
        // Graph outputs live until the end of the pass.
        for &out in &graph.outputs {
            last_use.insert(out, n_nodes);
        }

        // Collect values produced at each step and values expiring after each step.
        let mut born_at: Vec<Vec<ValueId>> = vec![Vec::new(); n_nodes];
        for (&vid, &step) in &def_step {
            let v = graph.value(vid);
            if matches!(v.kind, ValueKind::Node(_)) && step < n_nodes {
                born_at[step].push(vid);
            }
        }

        let mut free_blocks: Vec<(usize, usize)> = Vec::new(); // (offset, size)
        let mut active_expires: Vec<(usize, usize, usize)> = Vec::new(); // (last_step, offset, size)
        let mut spans: HashMap<ValueId, ArenaSpan> = HashMap::new();
        let mut total_bytes = 0usize;
        let mut unpacked_bytes = 0usize;

        for step in 0..n_nodes {
            // 1. Reclaim blocks whose last use was strictly before `step`.
            let mut i = 0;
            while i < active_expires.len() {
                if active_expires[i].0 < step {
                    let (_, off, sz) = active_expires.swap_remove(i);
                    insert_and_coalesce(&mut free_blocks, off, sz);
                } else {
                    i += 1;
                }
            }

            // 2. Allocate spans for values born at `step`.
            for &vid in &born_at[step] {
                let v = graph.value(vid);
                let elem_bytes = match v.dtype {
                    DType::F32 | DType::U32 => 4,
                    DType::F16 | DType::BF16 => 2,
                    DType::Quant(q) => {
                        let elems = v.shape.iter().product::<usize>().max(1);
                        let raw = q.tensor_bytes(elems).max(4);
                        let req = align_up(raw);
                        unpacked_bytes += req;
                        let end_step = last_use.get(&vid).copied().unwrap_or(step);
                        let offset = take_best_fit(&mut free_blocks, req, &mut total_bytes);
                        active_expires.push((end_step, offset, req));
                        spans.insert(
                            vid,
                            ArenaSpan {
                                offset,
                                bytes: req,
                                first_step: step,
                                last_step: end_step,
                            },
                        );
                        continue;
                    }
                };
                let elems = v.shape.iter().product::<usize>().max(1);
                let req = align_up(elems * elem_bytes);
                unpacked_bytes += req;
                let end_step = last_use.get(&vid).copied().unwrap_or(step);
                let offset = take_best_fit(&mut free_blocks, req, &mut total_bytes);
                active_expires.push((end_step, offset, req));
                spans.insert(
                    vid,
                    ArenaSpan {
                        offset,
                        bytes: req,
                        first_step: step,
                        last_step: end_step,
                    },
                );
            }
        }

        Self {
            total_bytes,
            spans,
            unpacked_bytes,
        }
    }
}

fn take_best_fit(free: &mut Vec<(usize, usize)>, req: usize, high_water: &mut usize) -> usize {
    let best = free
        .iter()
        .enumerate()
        .filter(|(_, &(_, sz))| sz >= req)
        .min_by_key(|(_, &(_, sz))| sz)
        .map(|(idx, _)| idx);
    if let Some(idx) = best {
        let (off, sz) = free.remove(idx);
        if sz > req {
            insert_and_coalesce(free, off + req, sz - req);
        }
        off
    } else {
        let off = *high_water;
        *high_water += req;
        off
    }
}

fn insert_and_coalesce(free: &mut Vec<(usize, usize)>, offset: usize, size: usize) {
    if size == 0 {
        return;
    }
    let pos = free.partition_point(|&(off, _)| off < offset);
    free.insert(pos, (offset, size));
    // Coalesce adjacent free spans.
    let mut i = 0;
    while i + 1 < free.len() {
        if free[i].0 + free[i].1 == free[i + 1].0 {
            free[i].1 += free[i + 1].1;
            free.remove(i + 1);
        } else {
            i += 1;
        }
    }
}

// ---- Runtime Dynamic Library Loader (`dlopen` / `dlsym`) --------------------------------------

#[cfg(unix)]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

#[cfg(unix)]
const RTLD_LAZY: c_int = 0x1;
#[cfg(unix)]
const RTLD_LOCAL: c_int = 0x4;

/// Runtime-loaded shared library handle (`libcuda`, `libamdhip64`, `libOpenCL`, PJRT plugins, etc.).
pub struct DynLib {
    handle: *mut c_void,
    pub path: String,
}

// SAFETY: `dlopen` handles and resolved function pointers are process-wide and thread-safe.
unsafe impl Send for DynLib {}
unsafe impl Sync for DynLib {}

impl DynLib {
    /// Tries to open the first candidate library path that exists on this host.
    pub fn open_any(candidates: &[&str]) -> Option<Self> {
        #[cfg(unix)]
        for &cand in candidates {
            let Ok(cstr) = CString::new(cand) else {
                continue;
            };
            let handle = unsafe { dlopen(cstr.as_ptr(), RTLD_LAZY | RTLD_LOCAL) };
            if !handle.is_null() {
                return Some(Self {
                    handle,
                    path: cand.to_string(),
                });
            }
        }
        None
    }

    /// Looks up raw symbol address `name` in the loaded library.
    pub fn sym_ptr(&self, name: &CStr) -> Option<*mut c_void> {
        #[cfg(unix)]
        unsafe {
            let p = dlsym(self.handle, name.as_ptr());
            (!p.is_null()).then_some(p)
        }
        #[cfg(not(unix))]
        {
            let _ = name;
            None
        }
    }
}

impl Drop for DynLib {
    fn drop(&mut self) {
        #[cfg(unix)]
        if !self.handle.is_null() {
            unsafe {
                let _ = dlclose(self.handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{BinaryKind, Op, UnaryKind};

    #[test]
    fn shared_arena_plan_reuses_expired_intermediate_spans() {
        let mut g = Graph::new();
        let x = g.input("x", vec![4, 64], DType::F32);
        let a = g.add(
            "a",
            Op::Unary {
                kind: UnaryKind::Silu,
            },
            &[x],
            vec![4, 64],
            DType::F32,
        );
        let b = g.add(
            "b",
            Op::Unary {
                kind: UnaryKind::Relu,
            },
            &[a],
            vec![4, 64],
            DType::F32,
        );
        let c = g.add(
            "c",
            Op::Unary {
                kind: UnaryKind::Neg,
            },
            &[b],
            vec![4, 64],
            DType::F32,
        );
        let d = g.add(
            "d",
            Op::Binary {
                kind: BinaryKind::Add,
            },
            &[c, x],
            vec![4, 64],
            DType::F32,
        );
        g.output(d);

        let plan = SharedArenaPlan::build(&g, DEFAULT_ARENA_ALIGNMENT);
        // Each tensor is 4 * 64 * 4 = 1024 bytes; `a` expires at step 1, so `c` or `d` reuses `a`'s offset!
        assert_eq!(plan.unpacked_bytes, 4 * 1024);
        assert!(
            plan.total_bytes <= 2 * 1024,
            "expected arena reuse (<= 2048 bytes), got {}",
            plan.total_bytes
        );
        assert_eq!(plan.spans[&a].offset, 0);
        assert_eq!(
            plan.spans[&c].offset, 0,
            "c should reuse a's expired span at offset 0"
        );
    }
}
