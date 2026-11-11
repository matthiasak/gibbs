//! Metal lowering group `movement`: Reshape/Squeeze/Unsqueeze (aliases), Slice, Concat, Transpose,
//! Pad, Tile, Gather, Scatter, Upsample, StateWrite. Kernels live in `movement.metal`.
//!
//! Most ops are one generic strided copy (`mv_strided`, see [`Map`]); views that keep the memory
//! layout alias their operand instead of copying.

use super::super::{set_bytes, u32_of, Planned, Session};
use super::{Enc, PlanOut};
use crate::graph::{DType, Graph, Node, Op, UpsampleMode, ValueId, ValueKind};
use std::collections::HashMap;

pub(super) const MSL: &str = include_str!("movement.metal");
/// Compute functions in `movement.metal`.
pub(super) const KERNELS: &[&str] = &[
    "mv_strided",
    "mv_fill",
    "mv_gather",
    "mv_scatter",
    "mv_upsample",
    "mv_state_write",
];

const MAX_RANK: usize = 6;
const TG: usize = 256;

/// Parameters of `mv_strided`; layout mirrors `MvMap` in `movement.metal`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Map {
    n: u32,
    rank: u32,
    odim: [u32; MAX_RANK],
    ostride: [u32; MAX_RANK],
    istride: [u32; MAX_RANK],
    idiv: [u32; MAX_RANK],
    imod: [u32; MAX_RANK],
    ibase: u32,
    obase: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GatherP {
    n: u32,
    m: u32,
    d: u32,
    inner: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ScatterP {
    m: u32,
    d: u32,
    inner: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct UpP {
    n: u32,
    d: u32,
    h: u32,
    w: u32,
    sd: u32,
    scale: u32,
    mode: u32,
}

/// Steps this group adds to the executor.
#[derive(Debug, Clone)]
pub enum Step {
    /// Optional constant fill of the whole output (`value`, element count), then one strided copy
    /// per source into it.
    Copy {
        fill: Option<(f32, usize)>,
        parts: Vec<(ValueId, Map)>,
    },
    Gather {
        x: ValueId,
        idx: ValueId,
        p: GatherP,
    },
    /// Copy `base` into the output, then write the `rows` update rows.
    Scatter {
        base: ValueId,
        idx: ValueId,
        upd: ValueId,
        copy: Map,
        rows: usize,
        p: ScatterP,
    },
    Upsample {
        x: ValueId,
        p: UpP,
    },
    /// Write `m` rows of width `w` into the `[cap, w]` state at row `pos0[0]`; the output is the token.
    StateWrite {
        state: ValueId,
        rows: ValueId,
        pos0: ValueId,
        m: usize,
        w: usize,
        cap: usize,
    },
}

/// Whether the Metal executor lowers `op` for `dtype` (the weight/output dtype the placer passes).
pub(super) fn supports(op: &Op, dtype: DType) -> bool {
    dtype == DType::F32
        && matches!(
            op,
            Op::Reshape { .. }
                | Op::Squeeze { .. }
                | Op::Unsqueeze { .. }
                | Op::Slice { .. }
                | Op::Concat { .. }
                | Op::Transpose { .. }
                | Op::Pad { .. }
                | Op::Tile { .. }
                | Op::Gather { .. }
                | Op::Scatter { .. }
                | Op::Upsample { .. }
                | Op::StateWrite
        )
}

/// One output dim of a strided copy.
struct Dim {
    n: usize,
    ostride: usize,
    istride: usize,
    /// Input coordinate is `(c / div) % md`; `md == 0` means no wrap.
    div: usize,
    md: usize,
}

impl Dim {
    fn dense(n: usize, ostride: usize, istride: usize) -> Dim {
        Dim {
            n,
            ostride,
            istride,
            div: 1,
            md: 0,
        }
    }
    fn plain(&self) -> bool {
        self.div == 1 && self.md == 0
    }
}

fn dense_strides(shape: &[usize]) -> Vec<usize> {
    let mut st = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        st[d] = st[d + 1] * shape[d + 1];
    }
    st
}

/// Builds the kernel parameters, dropping size-1 dims and merging dims contiguous in both buffers.
fn map(mut dims: Vec<Dim>, ibase: usize, obase: usize, name: &str) -> Result<Map, String> {
    dims.retain(|d| d.n != 1);
    let mut i = dims.len();
    while i > 1 {
        i -= 1;
        let (a, b) = (&dims[i - 1], &dims[i]);
        if a.plain() && b.plain() && a.istride == b.n * b.istride && a.ostride == b.n * b.ostride {
            let merged = Dim::dense(a.n * b.n, b.ostride, b.istride);
            dims[i - 1] = merged;
            dims.remove(i);
        }
    }
    if dims.is_empty() {
        dims.push(Dim::dense(1, 1, 1));
    }
    if dims.len() > MAX_RANK {
        return Err(format!(
            "node '{name}': copy needs {} non-mergeable dims; the Metal kernel takes {MAX_RANK}",
            dims.len()
        ));
    }
    let mut m = Map {
        n: u32_of(dims.iter().map(|d| d.n).product())?,
        rank: dims.len() as u32,
        odim: [1; MAX_RANK],
        ostride: [0; MAX_RANK],
        istride: [0; MAX_RANK],
        idiv: [1; MAX_RANK],
        imod: [u32::MAX; MAX_RANK],
        ibase: u32_of(ibase)?,
        obase: u32_of(obase)?,
    };
    for (k, d) in dims.iter().enumerate() {
        m.odim[k] = u32_of(d.n)?;
        m.ostride[k] = u32_of(d.ostride)?;
        m.istride[k] = u32_of(d.istride)?;
        m.idiv[k] = u32_of(d.div)?;
        if d.md != 0 {
            m.imod[k] = u32_of(d.md)?;
        }
    }
    Ok(m)
}

/// Dense copy of `n` elements.
fn flat(n: usize, name: &str) -> Result<Map, String> {
    map(vec![Dim::dense(n, 1, 1)], 0, 0, name)
}

fn shape_of(g: &Graph, v: ValueId) -> Result<&[usize], String> {
    let val = g.value(v);
    if val.shape.is_empty() {
        return Err(format!(
            "value '{}' has no static shape (the Metal executor needs one)",
            val.name
        ));
    }
    Ok(&val.shape)
}

fn axis_of(axis: isize, rank: usize, name: &str) -> Result<usize, String> {
    let a = if axis < 0 { axis + rank as isize } else { axis };
    if a < 0 || a as usize >= rank {
        return Err(format!(
            "node '{name}': axis {axis} out of range for rank {rank}"
        ));
    }
    Ok(a as usize)
}

/// Binds `out` to `x` at `byte_off` when `x`'s root lives in the arena; otherwise (weights, state
/// buffers) runs `copy`.
fn view(
    g: &Graph,
    aliases: &mut HashMap<ValueId, (ValueId, usize)>,
    x: ValueId,
    out: ValueId,
    byte_off: usize,
    copy: impl FnOnce() -> Result<Step, String>,
) -> Result<PlanOut<Step>, String> {
    let (root, base) = aliases.get(&x).copied().unwrap_or((x, 0));
    if matches!(g.value(root).kind, ValueKind::Node(_) | ValueKind::Input) {
        aliases.insert(out, (root, base + byte_off));
        Ok(PlanOut::Alias)
    } else {
        let n = shape_of(g, out)?.iter().product();
        copy().map(|s| PlanOut::Step(s, n))
    }
}

/// `None`: this group does not own `node.op`. Aliased outputs go into `aliases` as
/// `out -> (root value, byte offset)` and return `PlanOut::Alias`.
pub(super) fn plan(
    g: &Graph,
    node: &Node,
    aliases: &mut HashMap<ValueId, (ValueId, usize)>,
    segmented: &HashMap<ValueId, (usize, Vec<usize>)>,
) -> Option<Result<PlanOut<Step>, String>> {
    if !supports(&node.op, DType::F32) {
        return None;
    }
    Some(plan_op(g, node, aliases, segmented))
}

fn plan_op(
    g: &Graph,
    node: &Node,
    aliases: &mut HashMap<ValueId, (ValueId, usize)>,
    segmented: &HashMap<ValueId, (usize, Vec<usize>)>,
) -> Result<PlanOut<Step>, String> {
    let name = node.name.as_str();
    let arg = |k: usize| {
        node.inputs
            .get(k)
            .copied()
            .ok_or_else(|| format!("node '{name}': {} needs operand {k}", node.op.name()))
    };
    let out = *node
        .outputs
        .first()
        .ok_or_else(|| format!("node '{name}' has no output"))?;
    let out_n: usize = shape_of(g, out)?.iter().product();
    let copy_step =
        |parts: Vec<(ValueId, Map)>, fill: Option<f32>| -> Result<PlanOut<Step>, String> {
            Ok(PlanOut::Step(
                Step::Copy {
                    fill: fill.map(|v| (v, out_n)),
                    parts,
                },
                out_n,
            ))
        };
    match &node.op {
        Op::Reshape { .. } | Op::Squeeze { .. } | Op::Unsqueeze { .. } => {
            let x = arg(0)?;
            let n: usize = shape_of(g, x)?.iter().product();
            if n != out_n {
                return Err(format!(
                    "node '{name}': {} turns {n} elements into {out_n}",
                    node.op.name()
                ));
            }
            view(g, aliases, x, out, 0, || {
                Ok(Step::Copy {
                    fill: None,
                    parts: vec![(x, flat(n, name)?)],
                })
            })
        }
        Op::Slice { axis, start, end } => {
            let x = arg(0)?;
            let shape = shape_of(g, x)?;
            let ax = axis_of(*axis, shape.len(), name)?;
            let dim = shape[ax] as i64;
            let clampi = |v: i64| (if v < 0 { (dim + v).max(0) } else { v.min(dim) }) as usize;
            let (s, e) = (clampi(*start), clampi(*end));
            let len = e.saturating_sub(s);
            let outer: usize = shape[..ax].iter().product();
            let inner: usize = shape[ax + 1..].iter().product();
            if outer * len * inner != out_n {
                return Err(format!(
                    "node '{name}': slice {s}..{e} of axis {ax} of {shape:?} has {} elements, output declares {out_n}",
                    outer * len * inner
                ));
            }
            if let Some((seg_m, split)) = segmented.get(&x) {
                let mut bounds = Vec::with_capacity(split.len() + 2);
                bounds.push(0);
                bounds.extend_from_slice(split);
                bounds.push(shape[ax]);
                if ax + 1 == shape.len()
                    && outer == *seg_m
                    && bounds.windows(2).any(|w| w[0] == s && w[1] == e)
                {
                    return view(g, aliases, x, out, seg_m * s * 4, || {
                        Err(format!(
                            "node '{name}': segmented value cannot fall back to copy"
                        ))
                    });
                }
                return Err(format!(
                    "node '{name}': slice {s}..{e} on axis {ax} does not match segmented layout {split:?}"
                ));
            }
            let in_st = dense_strides(shape);
            let mut odims = shape.to_vec();
            odims[ax] = len;
            let out_st = dense_strides(&odims);
            let dims = (0..shape.len())
                .map(|d| Dim::dense(odims[d], out_st[d], in_st[d]))
                .collect();
            let ibase = s * in_st[ax];
            let m = map(dims, ibase, 0, name)?;
            if outer == 1 || len == shape[ax] {
                view(g, aliases, x, out, ibase * 4, || {
                    Ok(Step::Copy {
                        fill: None,
                        parts: vec![(x, m)],
                    })
                })
            } else {
                copy_step(vec![(x, m)], None)
            }
        }
        Op::Concat { axis } => {
            let first = shape_of(g, arg(0)?)?;
            let ax = axis_of(*axis, first.len(), name)?;
            let mut odims = first.to_vec();
            odims[ax] = 0;
            for &v in &node.inputs {
                let sh = shape_of(g, v)?;
                if sh.len() != first.len() || (0..sh.len()).any(|d| d != ax && sh[d] != first[d]) {
                    return Err(format!(
                        "node '{name}': concat operands disagree off axis {ax}"
                    ));
                }
                odims[ax] += sh[ax];
            }
            if odims.iter().product::<usize>() != out_n {
                return Err(format!("node '{name}': concat of {first:?}.. is {odims:?}, output declares {out_n} elements"));
            }
            let out_st = dense_strides(&odims);
            let mut parts = Vec::new();
            let mut at = 0;
            for &v in &node.inputs {
                let sh = shape_of(g, v)?;
                let in_st = dense_strides(sh);
                let dims = (0..sh.len())
                    .map(|d| Dim::dense(sh[d], out_st[d], in_st[d]))
                    .collect();
                parts.push((v, map(dims, 0, at * out_st[ax], name)?));
                at += sh[ax];
            }
            copy_step(parts, None)
        }
        Op::Transpose { perm } => {
            let x = arg(0)?;
            let shape = shape_of(g, x)?;
            let mut seen = vec![false; shape.len()];
            if perm.len() != shape.len()
                || perm
                    .iter()
                    .any(|&p| p >= shape.len() || std::mem::replace(&mut seen[p], true))
            {
                return Err(format!(
                    "node '{name}': {perm:?} is not a permutation of rank {}",
                    shape.len()
                ));
            }
            let in_st = dense_strides(shape);
            let odims: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
            let out_st = dense_strides(&odims);
            let dims = (0..perm.len())
                .map(|d| Dim::dense(odims[d], out_st[d], in_st[perm[d]]))
                .collect();
            copy_step(vec![(x, map(dims, 0, 0, name)?)], None)
        }
        Op::Tile { reps } => {
            let x = arg(0)?;
            let shape = shape_of(g, x)?;
            let r = shape.len().max(reps.len());
            let lead = |v: &[usize]| -> Vec<usize> {
                std::iter::repeat_n(1, r - v.len())
                    .chain(v.iter().copied())
                    .collect()
            };
            let (ish, rp) = (lead(shape), lead(reps));
            let odims: Vec<usize> = ish.iter().zip(&rp).map(|(a, b)| a * b).collect();
            if odims.iter().product::<usize>() != out_n {
                return Err(format!("node '{name}': tile of {shape:?} by {reps:?} is {odims:?}, output declares {out_n} elements"));
            }
            let (in_st, out_st) = (dense_strides(&ish), dense_strides(&odims));
            let dims = (0..r)
                .map(|d| Dim {
                    n: odims[d],
                    ostride: out_st[d],
                    istride: in_st[d],
                    div: 1,
                    md: ish[d],
                })
                .collect();
            copy_step(vec![(x, map(dims, 0, 0, name)?)], None)
        }
        Op::Pad { padding, value } => {
            let x = arg(0)?;
            let shape = shape_of(g, x)?;
            let (r, k) = (shape.len(), padding.len() / 2);
            if padding.len() % 2 != 0 || k > r {
                return Err(format!(
                    "node '{name}': Pad {padding:?} does not fit shape {shape:?}"
                ));
            }
            let before = |d: usize| {
                if d + k >= r {
                    padding[2 * (d + k - r)]
                } else {
                    0
                }
            };
            let after = |d: usize| {
                if d + k >= r {
                    padding[2 * (d + k - r) + 1]
                } else {
                    0
                }
            };
            let odims: Vec<usize> = (0..r).map(|d| shape[d] + before(d) + after(d)).collect();
            if odims.iter().product::<usize>() != out_n {
                return Err(format!("node '{name}': Pad of {shape:?} by {padding:?} is {odims:?}, output declares {out_n} elements"));
            }
            let (in_st, out_st) = (dense_strides(shape), dense_strides(&odims));
            let obase = (0..r).map(|d| before(d) * out_st[d]).sum();
            let dims = (0..r)
                .map(|d| Dim::dense(shape[d], out_st[d], in_st[d]))
                .collect();
            copy_step(vec![(x, map(dims, 0, obase, name)?)], Some(*value))
        }
        Op::Gather { axis } => {
            let (x, idx) = (arg(0)?, arg(1)?);
            let shape = shape_of(g, x)?;
            let ax = axis_of(*axis, shape.len(), name)?;
            if g.value(idx).dtype != DType::U32 {
                return Err(format!("node '{name}': Gather indices must be u32"));
            }
            let m: usize = shape_of(g, idx)?.iter().product();
            let (outer, d, inner) = (
                shape[..ax].iter().product::<usize>(),
                shape[ax],
                shape[ax + 1..].iter().product::<usize>(),
            );
            if outer * m * inner != out_n {
                return Err(format!("node '{name}': gather of {m} rows from {shape:?} is {} elements, output declares {out_n}", outer * m * inner));
            }
            let p = GatherP {
                n: u32_of(out_n)?,
                m: u32_of(m)?,
                d: u32_of(d)?,
                inner: u32_of(inner)?,
            };
            Ok(PlanOut::Step(Step::Gather { x, idx, p }, out_n))
        }
        Op::Scatter { axis } => {
            let (base, idx, upd) = (arg(0)?, arg(1)?, arg(2)?);
            let shape = shape_of(g, base)?;
            let ax = axis_of(*axis, shape.len(), name)?;
            if g.value(idx).dtype != DType::U32 {
                return Err(format!("node '{name}': Scatter indices must be u32"));
            }
            let m: usize = shape_of(g, idx)?.iter().product();
            let (outer, d, inner) = (
                shape[..ax].iter().product::<usize>(),
                shape[ax],
                shape[ax + 1..].iter().product::<usize>(),
            );
            let un: usize = shape_of(g, upd)?.iter().product();
            if un != outer * m * inner || out_n != outer * d * inner {
                return Err(format!("node '{name}': scatter of {un} update elements into {shape:?} at {m} rows does not fit"));
            }
            let p = ScatterP {
                m: u32_of(m)?,
                d: u32_of(d)?,
                inner: u32_of(inner)?,
            };
            let step = Step::Scatter {
                base,
                idx,
                upd,
                copy: flat(out_n, name)?,
                rows: outer * m,
                p,
            };
            Ok(PlanOut::Step(step, out_n))
        }
        Op::Upsample { scale, mode } => {
            let x = arg(0)?;
            let (inp, outp) = (shape_of(g, x)?, shape_of(g, out)?);
            let nsp = crate::graph::cpu::upsample_nsp(inp, outp, *scale)
                .map_err(|e| format!("node '{name}': {e}"))?;
            let r = inp.len();
            let (d, sd) = if nsp == 3 {
                (inp[r - 3], *scale)
            } else {
                (1, 1)
            };
            let p = UpP {
                n: u32_of(out_n)?,
                d: u32_of(d)?,
                h: u32_of(inp[r - 2])?,
                w: u32_of(inp[r - 1])?,
                sd: u32_of(sd)?,
                scale: u32_of(*scale)?,
                mode: match mode {
                    UpsampleMode::Nearest => 0,
                    UpsampleMode::Bilinear => 1,
                    UpsampleMode::Bicubic => 2,
                },
            };
            Ok(PlanOut::Step(Step::Upsample { x, p }, out_n))
        }
        Op::StateWrite => {
            let (state, rows, pos0) = (arg(0)?, arg(1)?, arg(2)?);
            if !matches!(g.value(state).kind, ValueKind::State { .. }) {
                return Err(format!(
                    "node '{name}': StateWrite operand 0 must be a graph state"
                ));
            }
            let &[cap, w] = shape_of(g, state)? else {
                return Err(format!("node '{name}': StateWrite state is not [cap, w]"));
            };
            let rn: usize = shape_of(g, rows)?.iter().product();
            if w == 0 || rn % w != 0 || g.value(pos0).dtype != DType::U32 {
                return Err(format!(
                    "node '{name}': StateWrite needs rows [m, {w}] and a u32 pos0"
                ));
            }
            u32_of(cap * w)?;
            Ok(PlanOut::Step(
                Step::StateWrite {
                    state,
                    rows,
                    pos0,
                    m: rn / w,
                    w,
                    cap,
                },
                1,
            ))
        }
        _ => Err(format!(
            "node '{name}': movement group does not own {}",
            node.op.name()
        )),
    }
}

/// Every value the step reads (drives arena liveness and barriers).
pub(super) fn reads(st: &Step) -> Vec<ValueId> {
    match st {
        Step::Copy { parts, .. } => parts.iter().map(|p| p.0).collect(),
        Step::Gather { x, idx, .. } => vec![*x, *idx],
        Step::Scatter { base, idx, upd, .. } => vec![*base, *idx, *upd],
        Step::Upsample { x, .. } => vec![*x],
        Step::StateWrite {
            state, rows, pos0, ..
        } => vec![*state, *rows, *pos0],
    }
}

/// Operands allowed to be non-f32 (u32 ids, packed weights).
pub(super) fn allows_non_f32(st: &Step, v: ValueId) -> bool {
    match st {
        Step::Gather { idx, .. } | Step::Scatter { idx, .. } => v == *idx,
        Step::StateWrite { pos0, .. } => v == *pos0,
        _ => false,
    }
}

fn copy(sess: &Session, e: &Enc, x: ValueId, out: ValueId, m: &Map) -> Result<(), String> {
    if m.n == 0 {
        return Ok(());
    }
    e.enc.set_compute_pipeline_state(sess.rt.pso("mv_strided")?);
    e.bind(0, x);
    e.bind(1, out);
    set_bytes(e.enc, 2, m);
    e.grid1(m.n as usize);
    Ok(())
}

pub(super) fn encode(sess: &Session, e: &Enc, s: &Planned, st: &Step) -> Result<(), String> {
    match st {
        Step::Copy { fill, parts } => {
            if let Some((value, n)) = *fill {
                e.enc.set_compute_pipeline_state(sess.rt.pso("mv_fill")?);
                e.bind(0, s.out);
                set_bytes(e.enc, 1, &value);
                set_bytes(e.enc, 2, &u32_of(n)?);
                e.grid1(n);
            }
            for (x, m) in parts {
                copy(sess, e, *x, s.out, m)?;
            }
        }
        Step::Gather { x, idx, p } => {
            if p.n > 0 {
                e.enc.set_compute_pipeline_state(sess.rt.pso("mv_gather")?);
                e.bind(0, *x);
                e.bind(1, *idx);
                e.bind(2, s.out);
                set_bytes(e.enc, 3, p);
                e.grid1(p.n as usize);
            }
        }
        Step::Scatter {
            base,
            idx,
            upd,
            copy: m,
            rows,
            p,
        } => {
            copy(sess, e, *base, s.out, m)?;
            if *rows > 0 {
                e.enc.set_compute_pipeline_state(sess.rt.pso("mv_scatter")?);
                e.bind(0, *idx);
                e.bind(1, *upd);
                e.bind(2, s.out);
                set_bytes(e.enc, 3, p);
                e.enc.dispatch_thread_groups(
                    metal::MTLSize::new(*rows as u64, 1, 1),
                    metal::MTLSize::new(TG as u64, 1, 1),
                );
            }
        }
        Step::Upsample { x, p } => {
            if p.n > 0 {
                e.enc
                    .set_compute_pipeline_state(sess.rt.pso("mv_upsample")?);
                e.bind(0, *x);
                e.bind(1, s.out);
                set_bytes(e.enc, 2, p);
                e.grid1(p.n as usize);
            }
        }
        Step::StateWrite {
            state,
            rows,
            pos0,
            m,
            w,
            cap,
        } => {
            // The state buffer may hold fewer rows than declared (lazy growth): never write past it.
            let (buf, off) = (e.loc)(*state);
            let held = (buf.length() as usize).saturating_sub(off) / (w * 4);
            e.enc
                .set_compute_pipeline_state(sess.rt.pso("mv_state_write")?);
            e.bind(0, *rows);
            e.bind(1, *pos0);
            e.bind(2, *state);
            e.bind(3, s.out);
            set_bytes(
                e.enc,
                4,
                &[u32_of(m * w)?, u32_of(*w)?, u32_of(held.min(*cap))?],
            );
            e.grid1((m * w).max(1));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::graph::metal::testutil::{both, rand, rel_err, rt};
    use crate::graph::{
        cpu,
        metal::{self, Session},
        DType, Graph, Op, Tensor, UpsampleMode, ValueId, Weights,
    };

    const TOL: f32 = 1e-5;

    /// A graph under construction plus its f32/u32 inputs.
    struct B {
        g: Graph,
        ins: Vec<Tensor>,
    }

    impl B {
        fn new() -> B {
            B {
                g: Graph::new(),
                ins: Vec::new(),
            }
        }
        fn f32(&mut self, shape: &[usize]) -> ValueId {
            let id = self
                .g
                .input(format!("x{}", self.ins.len()), shape.to_vec(), DType::F32);
            self.ins.push(Tensor::F32(rand(
                shape.iter().product(),
                7 + self.ins.len() as u64,
            )));
            id
        }
        fn u32(&mut self, shape: &[usize], vals: Vec<u32>) -> ValueId {
            let id = self
                .g
                .input(format!("x{}", self.ins.len()), shape.to_vec(), DType::U32);
            self.ins.push(Tensor::U32(vals));
            id
        }
        fn op(&mut self, op: Op, ins: &[ValueId], shape: &[usize]) -> ValueId {
            let name = format!("n{}", self.g.nodes.len());
            self.g.add(name, op, ins, shape.to_vec(), DType::F32)
        }
        /// Runs CPU and Metal on the first output and compares; returns the CPU result.
        fn check(mut self, out: ValueId, tol: f32) -> Vec<f32> {
            self.g.output(out);
            let (want, got) = both(&self.g, &Weights::new(), self.ins);
            let e = rel_err(&want, &got);
            assert!(e <= tol, "rel err {e} > {tol}");
            want
        }
    }

    fn len_of(dim: usize, start: i64, end: i64) -> usize {
        let c = |v: i64| if v < 0 { (dim as i64 + v).max(0) } else { v.min(dim as i64) } as usize;
        c(end).saturating_sub(c(start))
    }

    #[test]
    fn supports_f32_only() {
        let ops = [
            Op::Reshape { shape: None },
            Op::Squeeze { axes: vec![0] },
            Op::Unsqueeze { axes: vec![0] },
            Op::Slice {
                axis: 0,
                start: 0,
                end: 1,
            },
            Op::Concat { axis: 0 },
            Op::Transpose { perm: vec![1, 0] },
            Op::Pad {
                padding: vec![0, 1],
                value: 0.0,
            },
            Op::Tile { reps: vec![2] },
            Op::Gather { axis: 0 },
            Op::Scatter { axis: 0 },
            Op::Upsample {
                scale: 2,
                mode: UpsampleMode::Bicubic,
            },
        ];
        for op in &ops {
            assert!(metal::supports_op(op, DType::F32), "{op:?}");
            assert!(!super::supports(op, DType::F16), "{op:?}");
        }
    }

    #[test]
    fn reshape_squeeze_unsqueeze_alias_and_copy() {
        // Input aliased straight to the output.
        let mut b = B::new();
        let x = b.f32(&[3, 5]);
        let r = b.op(
            Op::Reshape {
                shape: Some(vec![5, 3]),
            },
            &[x],
            &[5, 3],
        );
        let want = b.check(r, 0.0);
        assert_eq!(want.len(), 15);

        // Chain of views ending in a real copy, with a view read by a core op after the producer.
        let mut b = B::new();
        let x = b.f32(&[1, 3, 1, 5]);
        let s = b.op(Op::Squeeze { axes: vec![0, 2] }, &[x], &[3, 5]);
        let u = b.op(Op::Unsqueeze { axes: vec![-1] }, &[s], &[3, 5, 1]);
        let t = b.op(
            Op::Transpose {
                perm: vec![1, 0, 2],
            },
            &[u],
            &[5, 3, 1],
        );
        let y = b.f32(&[15]);
        let rt_ = b.op(
            Op::Reshape {
                shape: Some(vec![15]),
            },
            &[t],
            &[15],
        );
        let sum = b.op(
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[rt_, y],
            &[15],
        );
        b.check(sum, TOL);

        // Runtime-shape reshape (two operands) is still an alias.
        let mut b = B::new();
        let x = b.f32(&[2, 6]);
        let shp = b.u32(&[2], vec![3, 4]);
        let r = b.op(Op::Reshape { shape: None }, &[x, shp], &[3, 4]);
        b.check(r, 0.0);
    }

    #[test]
    fn reshape_of_weight_copies() {
        let mut g = Graph::new();
        let w = g.weight("w", vec![4, 6], DType::F32);
        let r = g.add(
            "r",
            Op::Reshape {
                shape: Some(vec![6, 4]),
            },
            &[w],
            vec![6, 4],
            DType::F32,
        );
        let s = g.add(
            "s",
            Op::Slice {
                axis: 0,
                start: 1,
                end: 3,
            },
            &[w],
            vec![2, 6],
            DType::F32,
        );
        g.output(r);
        g.output(s);
        let mut ws = Weights::new();
        let data = rand(24, 3);
        ws.insert("w", Tensor::F32(data.clone()));
        let out = Session::new(rt(), &ws).unwrap().run(&g, vec![]).unwrap();
        assert_eq!(out[0].as_f32().unwrap(), &data[..]);
        assert_eq!(out[1].as_f32().unwrap(), &data[6..18]);
    }

    #[test]
    fn slice_axes_and_bounds() {
        let shapes: &[&[usize]] = &[
            &[3, 5, 7],
            &[1, 37],
            &[129, 3],
            &[2, 129],
            &[37],
            &[2, 3, 4, 5],
        ];
        let ranges = [
            (0, 1),
            (1, 3),
            (-3, -1),
            (0, 100),
            (-100, 2),
            (2, -1),
            (1, 1),
        ];
        let mut cases = 0;
        for shape in shapes {
            for axis in -(shape.len() as isize)..shape.len() as isize {
                let ax = (axis + shape.len() as isize) as usize % shape.len();
                for &(s, e) in &ranges {
                    let len = len_of(shape[ax], s, e);
                    if len == 0 {
                        continue;
                    }
                    let mut os = shape.to_vec();
                    os[ax] = len;
                    let mut b = B::new();
                    let x = b.f32(shape);
                    let o = b.op(
                        Op::Slice {
                            axis,
                            start: s,
                            end: e,
                        },
                        &[x],
                        &os,
                    );
                    b.check(o, 0.0);
                    cases += 1;
                }
            }
        }
        assert!(cases > 100, "{cases}");
    }

    #[test]
    fn slice_literal_and_chains() {
        // [[1,2,3],[4,5,6]] column 1..3 -> [[2,3],[5,6]]
        let mut g = Graph::new();
        let x = g.input("x", vec![2, 3], DType::F32);
        let o = g.add(
            "s",
            Op::Slice {
                axis: 1,
                start: 1,
                end: 3,
            },
            &[x],
            vec![2, 2],
            DType::F32,
        );
        g.output(o);
        let ins = vec![Tensor::F32(vec![1., 2., 3., 4., 5., 6.])];
        let (want, got) = both(&g, &Weights::new(), ins);
        assert_eq!(want, [2., 3., 5., 6.]);
        assert_eq!(got, want);

        // Slice of a slice (offset aliases), a strided copy, then a reshape alias as the output.
        let mut b = B::new();
        let x = b.f32(&[6, 4, 5]);
        let a = b.op(
            Op::Slice {
                axis: 0,
                start: 1,
                end: 5,
            },
            &[x],
            &[4, 4, 5],
        );
        let a2 = b.op(
            Op::Slice {
                axis: 0,
                start: 1,
                end: 3,
            },
            &[a],
            &[2, 4, 5],
        );
        let c = b.op(
            Op::Slice {
                axis: 2,
                start: 1,
                end: 4,
            },
            &[a2],
            &[2, 4, 3],
        );
        let r = b.op(
            Op::Reshape {
                shape: Some(vec![8, 3]),
            },
            &[c],
            &[8, 3],
        );
        b.check(r, 0.0);
    }

    #[test]
    fn concat_axes_and_arity() {
        let cases: &[(&[usize], isize, &[usize])] = &[
            (&[3, 5], 0, &[1, 2, 7]),
            (&[3, 5], 1, &[1, 4, 33]),
            (&[3, 5], -1, &[2, 2]),
            (&[2, 3, 4], 1, &[1, 3, 2]),
            (&[2, 3, 4], 2, &[1, 5, 129]),
            (&[2, 3, 4], 0, &[3, 1, 2]),
            (&[1, 37], 0, &[1, 1, 1, 1]),
            (&[37], 0, &[5, 3]),
            (&[2, 2, 3, 2], -2, &[1, 4]),
        ];
        for &(base, axis, sizes) in cases {
            let ax = (axis + base.len() as isize) as usize % base.len();
            let mut b = B::new();
            let mut ids = Vec::new();
            let mut total = 0;
            for &s in sizes {
                let mut sh = base.to_vec();
                sh[ax] = s;
                total += s;
                ids.push(b.f32(&sh));
            }
            let mut os = base.to_vec();
            os[ax] = total;
            let o = b.op(Op::Concat { axis }, &ids, &os);
            b.check(o, 0.0);
        }
        // Literal: [[1,2]] ++ [[3,4]] along axis 0.
        let mut g = Graph::new();
        let a = g.input("a", vec![1, 2], DType::F32);
        let c = g.input("c", vec![1, 2], DType::F32);
        let o = g.add(
            "cat",
            Op::Concat { axis: 1 },
            &[a, c],
            vec![1, 4],
            DType::F32,
        );
        g.output(o);
        let ins = vec![Tensor::F32(vec![1., 2.]), Tensor::F32(vec![3., 4.])];
        let (want, got) = both(&g, &Weights::new(), ins);
        assert_eq!(want, [1., 2., 3., 4.]);
        assert_eq!(got, want);
    }

    fn perms(n: usize) -> Vec<Vec<usize>> {
        if n == 0 {
            return vec![vec![]];
        }
        let mut out = Vec::new();
        for p in perms(n - 1) {
            for at in 0..n {
                let mut q = p.clone();
                q.insert(at, n - 1);
                out.push(q);
            }
        }
        out
    }

    #[test]
    fn transpose_every_permutation() {
        let shapes: &[&[usize]] = &[
            &[37],
            &[3, 129],
            &[3, 5, 7],
            &[2, 3, 5, 4],
            &[2, 1, 3, 2, 5],
        ];
        let mut cases = 0;
        for shape in shapes {
            let all = perms(shape.len());
            // rank 5 has 120 permutations; take every third to bound runtime
            for p in all.iter().step_by(if shape.len() == 5 { 3 } else { 1 }) {
                let os: Vec<usize> = p.iter().map(|&i| shape[i]).collect();
                let mut b = B::new();
                let x = b.f32(shape);
                let o = b.op(Op::Transpose { perm: p.clone() }, &[x], &os);
                b.check(o, 0.0);
                cases += 1;
            }
        }
        assert!(cases > 60, "{cases}");
        // Literal: [[1,2,3],[4,5,6]]^T
        let mut g = Graph::new();
        let x = g.input("x", vec![2, 3], DType::F32);
        let o = g.add(
            "t",
            Op::Transpose { perm: vec![1, 0] },
            &[x],
            vec![3, 2],
            DType::F32,
        );
        g.output(o);
        let (want, got) = both(
            &g,
            &Weights::new(),
            vec![Tensor::F32(vec![1., 2., 3., 4., 5., 6.])],
        );
        assert_eq!(want, [1., 4., 2., 5., 3., 6.]);
        assert_eq!(got, want);
    }

    #[test]
    fn pad_ranks_and_sides() {
        let cases: &[(&[usize], &[usize], f32)] = &[
            (&[5], &[2, 3], 0.0),
            (&[3, 7], &[1, 2], 0.0),
            (&[3, 7], &[1, 0, 2, 3], 0.0),
            (&[2, 3, 5], &[1, 1, 0, 2], 2.5),
            (&[2, 3, 5], &[1, 0, 1, 2, 2, 1], -1.0),
            (&[1, 1], &[0, 0, 0, 0], 9.0),
            (&[37, 3], &[0, 2, 1, 0], -1.5),
            (&[2, 2, 3, 5], &[1, 1], 0.5),
            (&[1, 129], &[0, 1], 3.0),
            (&[2, 2, 3, 5], &[1, 0, 0, 2, 1, 1], 7.0),
        ];
        for &(shape, padding, value) in cases {
            let (r, k) = (shape.len(), padding.len() / 2);
            let mut os = shape.to_vec();
            for i in 0..k {
                os[r - k + i] += padding[2 * i] + padding[2 * i + 1];
            }
            let mut b = B::new();
            let x = b.f32(shape);
            let o = b.op(
                Op::Pad {
                    padding: padding.to_vec(),
                    value,
                },
                &[x],
                &os,
            );
            b.check(o, 0.0);
        }
        // Literal: [1,2] padded (1,2) with 9.
        let mut g = Graph::new();
        let x = g.input("x", vec![2], DType::F32);
        let o = g.add(
            "p",
            Op::Pad {
                padding: vec![1, 2],
                value: 9.0,
            },
            &[x],
            vec![5],
            DType::F32,
        );
        g.output(o);
        let (want, got) = both(&g, &Weights::new(), vec![Tensor::F32(vec![1., 2.])]);
        assert_eq!(want, [9., 1., 2., 9., 9.]);
        assert_eq!(got, want);
    }

    #[test]
    fn tile_reps_forms() {
        let cases: &[(&[usize], &[usize])] = &[
            (&[3], &[2]),
            (&[2, 3], &[2, 1]),
            (&[2, 3], &[1, 3]),
            (&[3], &[2, 2]),
            (&[2, 3, 4], &[2]),
            (&[1, 37], &[3, 2]),
            (&[5, 1, 3], &[1, 4, 2]),
            (&[2, 2], &[1, 1]),
            (&[129], &[3]),
        ];
        for &(shape, reps) in cases {
            let r = shape.len().max(reps.len());
            let mut os = vec![1; r];
            for d in 0..r {
                let s = if d + shape.len() >= r {
                    shape[d + shape.len() - r]
                } else {
                    1
                };
                let p = if d + reps.len() >= r {
                    reps[d + reps.len() - r]
                } else {
                    1
                };
                os[d] = s * p;
            }
            let mut b = B::new();
            let x = b.f32(shape);
            let o = b.op(
                Op::Tile {
                    reps: reps.to_vec(),
                },
                &[x],
                &os,
            );
            b.check(o, 0.0);
        }
        // Literal: [1,2] tiled 3x.
        let mut g = Graph::new();
        let x = g.input("x", vec![2], DType::F32);
        let o = g.add("t", Op::Tile { reps: vec![3] }, &[x], vec![6], DType::F32);
        g.output(o);
        let (want, got) = both(&g, &Weights::new(), vec![Tensor::F32(vec![1., 2.])]);
        assert_eq!(want, [1., 2., 1., 2., 1., 2.]);
        assert_eq!(got, want);
    }

    /// Deterministic indices below `bound`.
    fn idx(m: usize, bound: usize, seed: u64) -> Vec<u32> {
        rand(m, seed)
            .iter()
            .map(|v| (((v + 1.0) * 0.5 * bound as f32) as usize).min(bound - 1) as u32)
            .collect()
    }

    #[test]
    fn gather_axes() {
        let cases: &[(&[usize], isize, usize)] = &[
            (&[10, 4], 0, 3),
            (&[4, 10], 1, 5),
            (&[3, 5, 7], -1, 4),
            (&[3, 5, 7], 1, 6),
            (&[3, 5, 7], 0, 1),
            (&[37], 0, 37),
            (&[129, 3], 0, 129),
            (&[2, 300], 1, 300),
            (&[2, 2, 3, 4], 2, 2),
        ];
        for &(shape, axis, m) in cases {
            let ax = (axis + shape.len() as isize) as usize % shape.len();
            let outer: usize = shape[..ax].iter().product();
            let inner: usize = shape[ax + 1..].iter().product();
            let mut b = B::new();
            let x = b.f32(shape);
            let i = b.u32(&[m], idx(m, shape[ax], 11));
            let o = b.op(Op::Gather { axis }, &[x, i], &[outer * m * inner]);
            b.check(o, 0.0);
        }
        // Rank-2 index tensor is flattened.
        let mut b = B::new();
        let x = b.f32(&[6, 3]);
        let i = b.u32(&[2, 2], vec![5, 0, 3, 3]);
        let o = b.op(Op::Gather { axis: 0 }, &[x, i], &[2, 2, 3]);
        b.check(o, 0.0);
    }

    #[test]
    fn gather_out_of_range_is_zero() {
        let mut g = Graph::new();
        let x = g.input("x", vec![3, 2], DType::F32);
        let i = g.input("i", vec![3], DType::U32);
        let o = g.add("g", Op::Gather { axis: 0 }, &[x, i], vec![3, 2], DType::F32);
        g.output(o);
        let ins = vec![
            Tensor::F32(vec![1., 2., 3., 4., 5., 6.]),
            Tensor::U32(vec![2, 3, 0]),
        ];
        let out = Session::new(rt(), &Weights::new())
            .unwrap()
            .run(&g, ins)
            .unwrap();
        assert_eq!(out[0].as_f32().unwrap(), &[5., 6., 0., 0., 1., 2.]);
    }

    #[test]
    fn scatter_axes_and_duplicates() {
        // (base shape, axis, indices)
        let cases: Vec<(Vec<usize>, isize, Vec<u32>)> = vec![
            (vec![10, 4], 0, vec![3, 7, 1]),
            (vec![10, 4], 0, vec![2, 2, 5, 2]),
            (vec![3, 6], 1, vec![4, 0]),
            (vec![2, 3, 5], -1, vec![1, 4, 0]),
            (vec![2, 5, 3], 1, vec![4, 4, 1, 0]),
            (vec![300, 3], 0, idx(300, 300, 21)),
            (vec![4, 300], 0, vec![3, 1]),
            (vec![37], 0, vec![36, 0, 36]),
            (vec![2, 2, 3, 4], 2, vec![2, 0]),
        ];
        for (shape, axis, ids) in cases {
            let ax = (axis + shape.len() as isize) as usize % shape.len();
            let outer: usize = shape[..ax].iter().product();
            let inner: usize = shape[ax + 1..].iter().product();
            let m = ids.len();
            let mut b = B::new();
            let base = b.f32(&shape);
            let i = b.u32(&[m], ids);
            let upd = b.f32(&[outer * m * inner]);
            let o = b.op(Op::Scatter { axis }, &[base, i, upd], &shape);
            b.check(o, 0.0);
        }
    }

    #[test]
    fn upsample_modes_and_ranks() {
        // (input shape, scale, nsp)
        let cases: &[(&[usize], usize, usize)] = &[
            (&[3, 5], 2, 2),
            (&[2, 3, 5], 3, 2),
            (&[2, 3, 4, 5], 2, 2),
            (&[1, 1, 37], 2, 2),
            (&[2, 3, 4, 5], 2, 3),
            (&[2, 2, 3, 4, 5], 2, 3),
            (&[1, 2, 3, 1], 3, 3),
            (&[2, 4, 4], 1, 2),
            (&[1, 1, 3, 129], 2, 2),
        ];
        for &(shape, scale, nsp) in cases {
            for mode in [
                UpsampleMode::Nearest,
                UpsampleMode::Bilinear,
                UpsampleMode::Bicubic,
            ] {
                let r = shape.len();
                let mut os = shape.to_vec();
                for d in r - nsp..r {
                    os[d] *= scale;
                }
                let mut b = B::new();
                let x = b.f32(shape);
                let o = b.op(Op::Upsample { scale, mode }, &[x], &os);
                b.check(o, TOL);
            }
        }
    }

    #[test]
    fn upsample_literals() {
        let run = |mode| {
            let mut g = Graph::new();
            let x = g.input("x", vec![2, 2], DType::F32);
            let o = g.add(
                "u",
                Op::Upsample { scale: 2, mode },
                &[x],
                vec![4, 4],
                DType::F32,
            );
            g.output(o);
            both(&g, &Weights::new(), vec![Tensor::F32(vec![1., 2., 3., 4.])])
        };
        let (want, got) = run(UpsampleMode::Nearest);
        assert_eq!(
            want,
            [1., 1., 2., 2., 1., 1., 2., 2., 3., 3., 4., 4., 3., 3., 4., 4.]
        );
        assert_eq!(got, want);
        // PyTorch `interpolate(mode="bilinear", align_corners=False)`.
        let (want, got) = run(UpsampleMode::Bilinear);
        let exp = [
            1., 1.25, 1.75, 2., 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3., 3.25, 3.75, 4.,
        ];
        assert!(
            rel_err(&exp, &want) < 1e-6 && rel_err(&exp, &got) < 1e-6,
            "{want:?}"
        );
        // PyTorch `mode="bicubic", align_corners=False` corner value.
        let (want, got) = run(UpsampleMode::Bicubic);
        assert!((want[0] - 0.683_593_75).abs() < 1e-5, "{want:?}");
        assert!(rel_err(&want, &got) < 1e-5);
    }

    #[test]
    fn mixed_graph_arena_reuse() {
        // Views and copies feeding each other and a core op; exercises liveness of aliased roots.
        let mut b = B::new();
        let x = b.f32(&[4, 6]);
        let y = b.f32(&[4, 6]);
        let s = b.op(
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[x, y],
            &[4, 6],
        );
        let t = b.op(Op::Transpose { perm: vec![1, 0] }, &[s], &[6, 4]);
        let a = b.op(
            Op::Slice {
                axis: 0,
                start: 2,
                end: 6,
            },
            &[t],
            &[4, 4],
        );
        let u = b.op(
            Op::Upsample {
                scale: 2,
                mode: UpsampleMode::Nearest,
            },
            &[a],
            &[8, 8],
        );
        let p = b.op(
            Op::Pad {
                padding: vec![1, 1, 1, 1],
                value: 0.0,
            },
            &[u],
            &[10, 10],
        );
        let c = b.op(Op::Concat { axis: 0 }, &[p, p], &[20, 10]);
        let r = b.op(
            Op::Reshape {
                shape: Some(vec![200]),
            },
            &[c],
            &[200],
        );
        let tl = b.op(Op::Tile { reps: vec![2] }, &[r], &[400]);
        let d = b.op(
            Op::Slice {
                axis: 0,
                start: 100,
                end: 300,
            },
            &[tl],
            &[200],
        );
        let e = b.op(
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[d, r],
            &[200],
        );
        b.check(e, TOL);
    }

    #[test]
    fn cpu_reference_rejects_bad_shapes() {
        let mut g = Graph::new();
        let x = g.input("x", vec![2, 3], DType::F32);
        let o = g.add(
            "t",
            Op::Transpose { perm: vec![0, 0] },
            &[x],
            vec![2, 3],
            DType::F32,
        );
        g.output(o);
        assert!(cpu::run_stateless(&g, &Weights::new(), vec![Tensor::F32(vec![0.0; 6])]).is_err());
    }
}
