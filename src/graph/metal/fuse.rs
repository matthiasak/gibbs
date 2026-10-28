//! Planner fusions as e-graph rules. [`decide`] encodes the lowered graph's nodes, the operand
//! facts the Metal kernels need (element counts, producer order, op parameters, weight formats),
//! saturates the rules below, and returns the fusions `plan` lowers: a Linear absorbing its single
//! reader as a GEMV epilogue, an RmsNorm absorbing a residual Add, a scalar Mul and/or a RoPE, and a
//! cached attention absorbing its q/k RoPEs.
//!
//! Every rule claims a node read by exactly one other node, and the operands it captures must exist
//! before the claiming node, so at most one rule claims any node: extraction takes every match.

use super::{fusion_enabled, linear_format, numel, Epi, Planned, Step, ATTN_MAX_HEAD_DIM};
use crate::graph::{DType, Graph, Op, RopeVariant, ValueId, ValueKind};
use egglog::{EGraph, Write as _};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

// Flat i64 facts keyed by node and value ids: egglog's generic join orders the variables of nested
// term patterns badly here (a ~780-node graph took 170 ms, quadratic in the pattern variables), so
// each fact carries the columns its rules join on.
const PROGRAM: &str = r#"
(relation Out (i64 i64))
(relation Single (i64))
(relation Val (i64 i64 i64))
(relation F32 (i64))
(relation NotNorm (i64))
(relation Rd1 (i64 String i64))
(relation Rd2 (i64 String i64 i64))
(relation Lin (i64 i64 i64))
(relation PackedEpi (i64))
(relation Norm (i64 i64 i64 i64))
(relation NormNoGamma (i64 i64 i64 i64))
(relation NormGamma (i64 i64 i64))
(relation Rope (i64 i64 i64 i64 i64))
(relation RopeP (i64 i64 i64 i64 i64))
(relation Attn (i64 i64 i64 i64 i64 i64 i64))
(relation Epi (i64 i64 i64 i64))
(relation NormRes (i64 i64 i64))
(relation NormScale (i64 i64 i64))
(relation NormRope (i64 i64))
(relation AttnRope (i64 i64 i64))
(ruleset plan)

; Linear `li` (output `l` of m*n elements, one reader `j`) → GEMV epilogue (Epi li j kind side|-1).
(rule ((Lin li l mn) (Single l) (Rd1 j "GeluErf" l)) ((Epi li j 1 -1)) :ruleset plan)
(rule ((Lin li l mn) (PackedEpi li) (Single l) (Rd1 j "Silu" l)) ((Epi li j 2 -1)) :ruleset plan)
(rule ((Lin li l mn) (PackedEpi li) (Single l) (Rd2 j "SwiGlu" l o) (Val o po mn) (< po li))
      ((Epi li j 3 o)) :ruleset plan)
(rule ((Lin li l mn) (PackedEpi li) (Single l) (Rd2 j "SwiGlu" o l) (Val o po mn) (< po li))
      ((Epi li j 4 o)) :ruleset plan)
(rule ((Lin li l mn) (PackedEpi li) (Single l) (Rd2 j "GeGlu" l o) (Val o po mn) (< po li))
      ((Epi li j 5 o)) :ruleset plan)
(rule ((Lin li l mn) (PackedEpi li) (Single l) (Rd2 j "GeGlu" o l) (Val o po mn) (< po li))
      ((Epi li j 6 o)) :ruleset plan)
(rule ((Lin li l mn) (Single l) (Rd2 j "Add" l o) (Val o po mn) (< po li))
      ((Epi li j 7 o)) :ruleset plan)
(rule ((Lin li l mn) (Single l) (Rd2 j "Add" o l) (Val o po mn) (< po li))
      ((Epi li j 7 o)) :ruleset plan)

; RmsNorm `ni` (output `n`, gamma width d, x of xl elements): `+ r`, then `* s` (scalar f32), or a
; RoPE.
(rule ((NormNoGamma ni n d xl) (Single n) (Rd2 j "Mul" n g) (Val g pg d) (F32 g) (< pg ni))
      ((NormGamma ni j g)) :ruleset plan)
(rule ((NormNoGamma ni n d xl) (Single n) (Rd2 j "Mul" g n) (Val g pg d) (F32 g) (< pg ni))
      ((NormGamma ni j g)) :ruleset plan)
(rule ((Norm ni n d xl) (Single n) (Rd2 j "Add" n o) (Val o po xl) (< po ni))
      ((NormRes ni j o)) :ruleset plan)
(rule ((Norm ni n d xl) (Single n) (Rd2 j "Add" o n) (Val o po xl) (< po ni))
      ((NormRes ni j o)) :ruleset plan)
(rule ((Norm ni n d xl) (Single n) (Rd2 k "Mul" n o) (Val o po 1) (F32 o) (< po ni))
      ((NormScale ni k o)) :ruleset plan)
(rule ((Norm ni n d xl) (Single n) (Rd2 k "Mul" o n) (Val o po 1) (F32 o) (< po ni))
      ((NormScale ni k o)) :ruleset plan)
(rule ((NormRes ni j vr) (Out j n) (Single n) (Rd2 k "Mul" n o) (Val o po 1) (F32 o) (< po ni))
      ((NormScale ni k o)) :ruleset plan)
(rule ((NormRes ni j vr) (Out j n) (Single n) (Rd2 k "Mul" o n) (Val o po 1) (F32 o) (< po ni))
      ((NormScale ni k o)) :ruleset plan)
(rule ((Norm ni n d xl) (Single n) (Rope j r n pos inv) (RopeP j d rot var ms) (<= rot d)
       (Val pos pp ns) (< pp ni) (Val inv pv ni_) (< pv ni) (> ns 0) (= 0 (% (/ xl d) ns)))
      ((NormRope ni j)) :ruleset plan)

; Cached attention `ai` writing its KV state: q and k each the single-reader RoPE of a non-norm.
(rule ((Attn ai h kvh hd q k pos) (Single q) (Single k)
       (Rope qi q qx pos inv) (Rope ki k kx pos inv)
       (RopeP qi hd rot var ms) (RopeP ki hd rot var ms) (> rot 0) (<= rot hd)
       (NotNorm qx) (NotNorm kx)
       (Val qx pq nq) (= nq (* h hd)) (Val kx pk nk) (= nk (* kvh hd))
       (Val pos pp 1) (Val inv pi ni) (>= ni (/ rot 2)))
      ((AttnRope ai qi ki)) :ruleset plan)

(run-schedule (saturate (run plan)))
"#;

/// What an RmsNorm absorbs: the residual `Add` node and its other operand, the scalar `Mul` node
/// and its scale, the RoPE node.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct NormFusion {
    pub gamma: Option<(usize, ValueId)>,
    pub res: Option<(usize, ValueId)>,
    pub scale: Option<(usize, ValueId)>,
    pub rope: Option<usize>,
}

/// The fusions `plan` lowers, keyed by the absorbing node.
#[derive(Debug, Default)]
pub(super) struct Decisions {
    /// Linear → (absorbed reader, epilogue, its second operand).
    pub epi: HashMap<usize, (usize, Epi, Option<ValueId>)>,
    pub norm: HashMap<usize, NormFusion>,
    /// CachedAttention → its q and k RoPE nodes.
    pub attn_rope: HashMap<usize, (usize, usize)>,
}

impl Decisions {
    /// Nodes lowered inside another node's step.
    pub fn absorbed(&self) -> impl Iterator<Item = usize> + '_ {
        let epi = self.epi.values().map(|e| e.0);
        let norm = self.norm.values().flat_map(|n| {
            [
                n.gamma.map(|g| g.0),
                n.res.map(|r| r.0),
                n.scale.map(|s| s.0),
                n.rope,
            ]
            .into_iter()
            .flatten()
        });
        let attn = self.attn_rope.values().flat_map(|&(q, k)| [q, k]);
        epi.chain(norm).chain(attn)
    }
}

pub(super) fn err(e: egglog::Error) -> String {
    format!("planner e-graph: {e}")
}

/// A fresh e-graph with `program` (relations and rules) loaded: a clone of this thread's parsed
/// copy, since parsing (~1 ms) costs more than most plans' encoding and saturation together.
pub(super) fn loaded(program: &str) -> Result<EGraph, String> {
    thread_local!(static PARSED: std::cell::RefCell<HashMap<String, EGraph>> = Default::default());
    PARSED.with_borrow_mut(|parsed| {
        if let Some(eg) = parsed.get(program) {
            return Ok(eg.clone());
        }
        let mut eg = EGraph::default();
        eg.parse_and_run_program(None, program).map_err(err)?;
        parsed.insert(program.to_string(), eg.clone());
        Ok(eg)
    })
}

/// The rows of the all-`i64` relation `name`.
pub(super) fn relation(eg: &EGraph, name: &str) -> Result<Vec<Vec<i64>>, String> {
    let mut out = Vec::new();
    eg.constructor_enodes(name, |en| {
        out.push(
            en.children
                .iter()
                .map(|&v| eg.value_to_base::<i64>(v))
                .collect(),
        )
    })
    .map_err(err)?;
    Ok(out)
}

/// Saturates the planner rules over `g` (see the module doc).
pub(super) fn decide(g: &Graph) -> Result<Decisions, String> {
    let mut uses = vec![0i64; g.values.len()];
    for n in &g.nodes {
        for v in &n.inputs {
            uses[v.0 as usize] += 1;
        }
    }
    let fused = fusion_enabled();
    let produced_by_norm = |v: ValueId| {
        matches!(g.value(v).kind,
            ValueKind::Node(n) if matches!(g.nodes[n.0 as usize].op, Op::RmsNorm { .. }))
    };
    let (schema, run) = PROGRAM.split_at(PROGRAM.find("(run-schedule").expect("schedule"));
    let mut eg = loaded(schema)?;
    eg.update(|mut fs| -> Result<(), egglog::Error> {
        // Facts for the values nodes read or produce (passes leave dead values behind).
        let mut live = vec![false; g.values.len()];
        for n in &g.nodes {
            for v in n.inputs.iter().chain(n.outputs.first()) {
                live[v.0 as usize] = true;
            }
        }
        for vi in (0..g.values.len()).filter(|&vi| live[vi]) {
            let (id, v) = (vi as i64, ValueId(vi as u32));
            let pos = match g.value(v).kind {
                ValueKind::Node(n) => n.0 as i64,
                _ => -1,
            };
            fs.add("Val", (id, pos, numel(g, v).map_or(-1, |n| n as i64)))?;
            if g.value(v).dtype == DType::F32 {
                fs.add("F32", id)?;
            }
            if !produced_by_norm(v) {
                fs.add("NotNorm", id)?;
            }
        }
        for (i, n) in g.nodes.iter().enumerate() {
            let id = i as i64;
            let Some(&out) = n.outputs.first() else {
                continue;
            };
            let o = out.0 as i64;
            fs.add("Out", (id, o))?;
            if uses[out.0 as usize] == 1 && !g.outputs.contains(&out) {
                fs.add("Single", o)?;
            }
            let arg = |k: usize| n.inputs.get(k).map(|v| v.0 as i64);
            let tag = match &n.op {
                Op::Unary {
                    kind: crate::graph::UnaryKind::GeluErf,
                } => Some("GeluErf"),
                Op::Unary {
                    kind: crate::graph::UnaryKind::Silu,
                } => Some("Silu"),
                Op::SwiGlu => Some("SwiGlu"),
                Op::GeGlu => Some("GeGlu"),
                Op::Binary {
                    kind: crate::graph::BinaryKind::Add,
                } => Some("Add"),
                Op::Binary {
                    kind: crate::graph::BinaryKind::Mul,
                } => Some("Mul"),
                _ => None,
            };
            if let (Some(t), Some(a)) = (tag, arg(0)) {
                fs.add("Rd1", (id, t, a))?;
                if let (Some(b), 2) = (arg(1), n.inputs.len()) {
                    fs.add("Rd2", (id, t, a, b))?;
                }
            }
            match &n.op {
                Op::Linear if fused => {
                    let (Some(&x), Some(&w)) = (n.inputs.first(), n.inputs.get(1)) else {
                        continue;
                    };
                    let (&[rows, k], Ok(xl)) = (&g.value(w).shape[..], numel(g, x)) else {
                        continue;
                    };
                    let format = linear_format(g, w);
                    if k == 0 {
                        continue;
                    }
                    let m = xl / k;
                    fs.add("Lin", (id, o, (m * rows) as i64))?;
                    if format.is_some_and(|f| !(m >= 16 && f.batched_gemm()) || k % 8 == 0) {
                        fs.add("PackedEpi", id)?;
                    }
                }
                Op::RmsNorm { .. } if fused && n.inputs.len() >= 2 => {
                    let d = g.value(n.inputs[1]).shape.last().copied().unwrap_or(0);
                    if let Ok(xl) = numel(g, n.inputs[0]) {
                        if d > 0 && xl % d == 0 {
                            fs.add("Norm", (id, o, d as i64, xl as i64))?;
                        }
                    }
                }
                Op::RmsNorm { .. } if fused && n.inputs.len() == 1 => {
                    let d = g.value(n.inputs[0]).shape.last().copied().unwrap_or(0);
                    if let Ok(xl) = numel(g, n.inputs[0]) {
                        if d > 0 && xl % d == 0 {
                            fs.add("NormNoGamma", (id, o, d as i64, xl as i64))?;
                        }
                    }
                }
                Op::Rope {
                    head_dim,
                    rotary_dim,
                    variant,
                    mscale,
                } if rotary_dim % 2 == 0 => {
                    let var = match variant {
                        RopeVariant::Interleaved => 2,
                        RopeVariant::HalfSplit => 1,
                        RopeVariant::MRope { .. } => continue,
                    };
                    let p = (id, *head_dim as i64, *rotary_dim as i64, var as i64);
                    fs.add("RopeP", (p.0, p.1, p.2, p.3, mscale.to_bits() as i64))?;
                    if let (Some(x), Some(pos), Some(inv)) = (arg(0), arg(1), arg(2)) {
                        fs.add("Rope", (id, o, x, pos, inv))?;
                    }
                }
                Op::CachedAttention {
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    write: true,
                    ..
                } if fused
                    && head_dim == v_head_dim
                    && *head_dim <= ATTN_MAX_HEAD_DIM
                    && n.inputs.len() == 6 =>
                {
                    let p = (*heads as i64, *kv_heads as i64, *head_dim as i64);
                    let (q, k, pos) = (n.inputs[0].0 as i64, n.inputs[1].0 as i64, n.inputs[5].0);
                    fs.add("Attn", (id, p.0, p.1, p.2, q, k, pos as i64))?;
                }
                _ => {}
            }
        }
        Ok(())
    })
    .map_err(err)?;
    eg.parse_and_run_program(None, run).map_err(err)?;

    let rows = |eg: &EGraph, name: &str| relation(eg, name);
    let val = |v: i64| (v >= 0).then(|| ValueId(v as u32));
    let mut d = Decisions::default();
    for r in rows(&eg, "Epi")? {
        let epi = match r[2] {
            1 => Epi::Gelu,
            2 => Epi::Silu,
            3 => Epi::SwiGlu,
            4 => Epi::UpSwiGlu,
            5 => Epi::GeGlu,
            6 => Epi::UpGeGlu,
            _ => Epi::Residual,
        };
        d.epi.insert(r[0] as usize, (r[1] as usize, epi, val(r[3])));
    }
    for r in rows(&eg, "NormGamma")? {
        d.norm.entry(r[0] as usize).or_default().gamma =
            Some((r[1] as usize, ValueId(r[2] as u32)));
    }
    for r in rows(&eg, "NormRes")? {
        d.norm.entry(r[0] as usize).or_default().res = Some((r[1] as usize, ValueId(r[2] as u32)));
    }
    for r in rows(&eg, "NormScale")? {
        d.norm.entry(r[0] as usize).or_default().scale =
            Some((r[1] as usize, ValueId(r[2] as u32)));
    }
    for r in rows(&eg, "NormRope")? {
        d.norm.entry(r[0] as usize).or_default().rope = Some(r[1] as usize);
    }
    for r in rows(&eg, "AttnRope")? {
        d.attn_rope
            .insert(r[0] as usize, (r[1] as usize, r[2] as usize));
    }
    Ok(d)
}

/// Horizontal batching over the step list, as [`batches`] extracts it.
#[derive(Debug, Default)]
pub(super) struct Batches {
    /// First member → the members (ascending, 2..=4) of one batched dispatch. Steps between members
    /// are independent GEMVs that run ahead of the batch.
    pub runs: HashMap<usize, Vec<usize>>,
    /// An RmsNorm with a residual or scale whose next step, a plain RmsNorm of its output, chains in.
    pub chain: HashSet<usize>,
    /// A one-row RmsNorm with a residual or scale that the next step, a MegakernelMlp of its output,
    /// absorbs.
    pub absorb: HashSet<usize>,
}

/// Step-order facts: `BKind p k` (1: RmsNorm without residual/scale/chain; 1000+: a Rope geometry),
/// `RopeKey p rk` (the RoPE fused into that norm, 0: none), `Indep a b` (step `b` reads no output of
/// `a`, `b - a < 8`). `R{L}` is a run of `L` members from `s` whose next candidate is at `c` and
/// whose RoPE is `rk`; it grows by the next step when that is of the run's kind, independent of
/// every member and RoPE-compatible, or — for norm runs — by the step after an independent
/// `Movable` GEMV.
const BATCH_PROGRAM: &str = r#"
(relation BKind (i64 i64))
(relation RopeKey (i64 i64))
(relation Solo (i64))
(relation Movable (i64))
(relation Indep (i64 i64))
(relation NormRS (i64 i64 i64 i64))
(relation NormPlain0 (i64 i64 i64 i64))
(relation NormRS1 (i64 i64 i64))
(relation MlpIn (i64 i64 i64))
(relation R1 (i64 i64 i64))
(relation R2 (i64 i64 i64 i64))
(relation R3 (i64 i64 i64 i64 i64))
(relation R4 (i64 i64 i64 i64 i64 i64))
(relation Chain (i64))
(relation Absorb (i64))
(ruleset batch)
(rule ((BKind s k) (RopeKey s rk)) ((R1 s (+ s 1) rk)) :ruleset batch)
(rule ((NormRS s rows d o) (= n (+ s 1)) (NormPlain0 n rows d o)) ((Chain s)) :ruleset batch)
(rule ((NormRS1 s d o) (= n (+ s 1)) (MlpIn n o d)) ((Absorb s)) :ruleset batch)
"#;

/// [`BATCH_PROGRAM`] with its run-growing rules for runs of 1..=3 members.
static BATCH: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    let mut p = BATCH_PROGRAM.to_string();
    for l in 1..=3 {
        let ms: Vec<String> = (2..=l).map(|i| format!("m{i}")).collect();
        let members: Vec<&str> = std::iter::once("s")
            .chain(ms.iter().map(String::as_str))
            .collect();
        let indep = |at: &str| -> String {
            members
                .iter()
                .map(|m| format!("(Indep {m} {at}) "))
                .collect()
        };
        let state = format!("(R{l} s {} c rk)", ms.join(" "));
        let grow = [
            ("(BKind s k)".to_string(), "c"),
            (
                format!(
                    "(BKind s k) (= k 1) (Movable c) {}(= at (+ c 1))",
                    indep("c")
                ),
                "at",
            ),
        ];
        for (pre, at) in &grow {
            for (compat, rk) in [
                (format!("(RopeKey {at} 0)"), "rk"),
                (format!("(RopeKey {at} nk) (> nk 0) (= rk 0)"), "nk"),
                (format!("(RopeKey {at} rk) (> rk 0)"), "rk"),
            ] {
                let _ = writeln!(
                    p,
                    "(rule ({state} {pre} (BKind {at} k) (Solo {at}) {}{compat}) \
                     ((R{} s {} {at} (+ {at} 1) {rk})) :ruleset batch)",
                    indep(at),
                    l + 1,
                    ms.join(" "),
                );
            }
        }
    }
    p
});

fn intern<K: std::hash::Hash + Eq>(m: &mut HashMap<K, i64>, k: K) -> i64 {
    let n = m.len() as i64 + 1;
    *m.entry(k).or_insert(n)
}

/// Saturates the batching rules over `steps` and extracts what the greedy in-order walk takes:
/// at each step, the longest run starting there, else its chain, else its absorption.
pub(super) fn batches(
    steps: &[Planned],
    aliases: &HashMap<ValueId, (ValueId, usize)>,
) -> Result<Batches, String> {
    let root = |v: ValueId| aliases.get(&v).map_or(v, |a| a.0);
    let mut eg = loaded(&BATCH)?;
    let mut ropes = HashMap::new();
    let mut geoms = HashMap::new();
    eg.update(|mut fs| -> Result<(), egglog::Error> {
        for (p, s) in steps.iter().enumerate() {
            let pi = p as i64;
            let solo = s.extra_outs.is_empty();
            if solo {
                fs.add("Solo", pi)?;
            }
            let kind = match &s.op {
                Step::RmsNorm {
                    x,
                    res: None,
                    scale: None,
                    chain: None,
                    rope,
                    rows,
                    d,
                    ..
                } => {
                    let rk = rope.map_or(0, |r| {
                        let key = (r.pos, r.inv_freq, r.n_seq, r.rotary_dim, r.pair_stride);
                        intern(&mut ropes, (key, r.mscale.to_bits()))
                    });
                    fs.add("RopeKey", (pi, rk))?;
                    if rope.is_none() && solo {
                        let at = (pi, *rows as i64, *d as i64, root(*x).0 as i64);
                        fs.add("NormPlain0", at)?;
                    }
                    Some(1)
                }
                Step::Rope {
                    pos,
                    inv_freq,
                    n_seq,
                    head_dim,
                    rotary_dim,
                    pair_stride,
                    mscale,
                    ..
                } => {
                    fs.add("RopeKey", (pi, 0i64))?;
                    let key = (
                        *pos,
                        *inv_freq,
                        *n_seq,
                        *head_dim,
                        *rotary_dim,
                        *pair_stride,
                    );
                    Some(1000 + intern(&mut geoms, (key, mscale.to_bits())))
                }
                Step::RmsNorm {
                    chain: None,
                    rope: None,
                    rows,
                    d,
                    ..
                } => {
                    let (rows, d) = (*rows as i64, *d as i64);
                    fs.add("NormRS", (pi, rows, d, root(s.out).0 as i64))?;
                    if rows == 1 && solo && !aliases.contains_key(&s.out) {
                        fs.add("NormRS1", (pi, d, s.out.0 as i64))?;
                    }
                    None
                }
                Step::MegakernelMlp {
                    x,
                    pre_res: None,
                    hidden,
                    ..
                } => {
                    fs.add("MlpIn", (pi, x.0 as i64, *hidden as i64))?;
                    None
                }
                _ => None,
            };
            let gemv = matches!(s.op, Step::GemvPacked { .. });
            let writes_state = matches!(
                s.op,
                Step::Ext(super::ext::ExtStep::Movement(
                    super::ext::movement::Step::StateWrite { .. }
                ))
            );
            if kind.is_none() && solo && !writes_state && gemv {
                fs.add("Movable", pi)?;
            }
            if let Some(k) = kind {
                fs.add("BKind", (pi, k))?;
                let out = root(s.out);
                for (b, t) in steps.iter().enumerate().take(p + 8).skip(p + 1) {
                    if !t.reads.iter().any(|&r| root(r) == out) {
                        fs.add("Indep", (pi, b as i64))?;
                    }
                }
            }
        }
        Ok(())
    })
    .map_err(err)?;
    eg.parse_and_run_program(None, "(run-schedule (saturate (run batch)))")
        .map_err(err)?;

    let mut b = Batches::default();
    for l in 2..=4usize {
        for r in relation(&eg, &format!("R{l}"))? {
            let members: Vec<usize> = r[..l].iter().map(|&m| m as usize).collect();
            // Runs grow one member at a time: the longest one from `s` is the extracted one.
            b.runs.insert(members[0], members);
        }
    }
    b.chain = relation(&eg, "Chain")?
        .iter()
        .map(|r| r[0] as usize)
        .collect();
    b.absorb = relation(&eg, "Absorb")?
        .iter()
        .map(|r| r[0] as usize)
        .collect();
    Ok(b)
}

/// One MoE subgraph fusion [`moe`] extracts, at its first step.
#[derive(Debug, Clone, Copy)]
pub(super) enum MoeFusion {
    /// TopK + GatherCols(logits) + Softmax (3 steps): the softmax output becomes TopK's weights.
    TopKSoftmax { weights: ValueId },
    /// ... + GatherCols(scale) + Mul (5 steps): scaled weights.
    TopKSoftmaxScaled { scale: ValueId, weights: ValueId },
    /// Copy(lower half) + Copy(upper half) + GeGlu/SwiGlu (3 steps) of one `[rows, 2 * half]` value.
    SplitGlu {
        x: ValueId,
        rows: usize,
        half_cols: usize,
        kind: u32,
    },
    /// Mul([slots, h] by [slots, 1]) + Reduce(Sum) over `k` (2 steps).
    WeightedSum {
        down: ValueId,
        weights: ValueId,
        n: usize,
        k: usize,
        h: usize,
    },
}

impl MoeFusion {
    /// Steps the fusion replaces, starting at its position.
    pub fn len(&self) -> usize {
        match self {
            MoeFusion::TopKSoftmax { .. } | MoeFusion::SplitGlu { .. } => 3,
            MoeFusion::TopKSoftmaxScaled { .. } => 5,
            MoeFusion::WeightedSum { .. } => 2,
        }
    }
}

/// Step-window facts of [`moe`]: `Single v` (one reader across steps and graph outputs, by alias
/// root), `Solo p` (no extra outputs), and per-kind step fields.
const MOE_PROGRAM: &str = r#"
(relation Single (i64))
(relation Solo (i64))
(relation TopK (i64 i64 i64 i64 i64 i64))
(relation Gat (i64 i64 i64 i64 i64 i64 i64 i64))
(relation Smax (i64 i64 i64 i64 i64))
(relation Bin (i64 i64 i64 i64 i64 i64 i64))
(relation Copy1 (i64 i64))
(relation Glu (i64 i64 i64 i64 i64 i64))
(relation Half (i64 i64 i64 i64 i64))
(relation RedSum (i64 i64 i64 i64 i64 i64 i64 i64))
(relation Shape2 (i64 i64 i64))
(relation MoeTopK (i64 i64))
(relation MoeScaled (i64 i64 i64))
(relation SplitG (i64 i64 i64 i64 i64))
(relation WSum (i64 i64 i64 i64 i64 i64 i64))
(ruleset moe)
; TopK(logits) -> ids; GatherCols(logits, ids) single-use; Softmax over k.
(rule ((TopK p l rows cols k ids) (= p1 (+ p 1)) (Gat p1 l ids tot cols k 0 go) (= tot (* rows k))
       (Single go) (= p2 (+ p 2)) (Smax p2 go rows k so))
      ((MoeTopK p so)) :ruleset moe)
(rule ((TopK p l 1 cols k ids) (= p1 (+ p 1)) (Gat p1 l ids k cols k 1 go)
       (Single go) (= p2 (+ p 2)) (Smax p2 go 1 k so))
      ((MoeTopK p so)) :ruleset moe)
; ... * GatherCols(scale table, ids) (shared), both single-use.
(rule ((MoeTopK p so) (TopK p l rows cols k ids) (Single so) (= p3 (+ p 3))
       (Gat p3 sct ids tot cols k 1 sco) (= tot (* rows k)) (Single sco)
       (= p4 (+ p 4)) (Bin p4 so sco tot 2 1 wo))
      ((MoeScaled p sct wo)) :ruleset moe)
(rule ((MoeTopK p so) (TopK p l rows cols k ids) (Single so) (= p3 (+ p 3))
       (Gat p3 sct ids tot cols k 1 sco) (= tot (* rows k)) (Single sco)
       (= p4 (+ p 4)) (Bin p4 sco so tot 2 1 wo))
      ((MoeScaled p sct wo)) :ruleset moe)
; Copy, Copy, Glu(gate, up): gate the lower and up the upper last-axis half of one value.
(rule ((Copy1 p o0) (Solo p) (= p1 (+ p 1)) (Copy1 p1 o1) (Solo p1) (= p2 (+ p 2))
       (Glu p2 o0 o1 n kind go) (Single o0) (Single o1)
       (Half o0 src r hc 0) (Half o1 src r hc 1) (= n (* r hc)))
      ((SplitG p src r hc kind)) :ruleset moe)
(rule ((Copy1 p o0) (Solo p) (= p1 (+ p 1)) (Copy1 p1 o1) (Solo p1) (= p2 (+ p 2))
       (Glu p2 o1 o0 n kind go) (Single o0) (Single o1)
       (Half o1 src r hc 0) (Half o0 src r hc 1) (= n (* r hc)))
      ((SplitG p src r hc kind)) :ruleset moe)
; Mul(down [slots, h], w [slots, 1]) single-use, then its Sum over k: WSum p down w n k h.
(rule ((Bin p a b mn 2 rk mo) (Solo p) (Single mo) (= p1 (+ p 1)) (RedSum p1 mo 0 nout ro n k h)
       (Solo p1) (= slots (* n k)) (Shape2 mo slots h) (= mn (* slots h)) (= nout (* n h))
       (Shape2 a slots h) (Shape2 b slots 1))
      ((WSum p 0 a b n k h)) :ruleset moe)
(rule ((Bin p a b mn 2 rk mo) (Solo p) (Single mo) (= p1 (+ p 1)) (RedSum p1 mo 0 nout ro n k h)
       (Solo p1) (= slots (* n k)) (Shape2 mo slots h) (= mn (* slots h)) (= nout (* n h))
       (Shape2 b slots h) (Shape2 a slots 1))
      ((WSum p 1 b a n k h)) :ruleset moe)
(run-schedule (saturate (run moe)))
"#;

/// The last-axis half `v` copies, when `v` is `Slice(src)` of exactly the lower or upper half of an
/// even last axis: `(src, rows, half, upper)`.
fn slice_half(g: &Graph, v: ValueId) -> Option<(ValueId, usize, usize, bool)> {
    let ValueKind::Node(nid) = g.value(v).kind else {
        return None;
    };
    let node = &g.nodes[nid.0 as usize];
    let Op::Slice { axis, start, end } = node.op else {
        return None;
    };
    let src = *node.inputs.first()?;
    let sh = &g.value(src).shape;
    let rank = sh.len();
    let ax = if axis < 0 { axis + rank as isize } else { axis };
    if rank < 2 || ax != rank as isize - 1 {
        return None;
    }
    let cols = sh[rank - 1];
    if cols == 0 || cols % 2 != 0 {
        return None;
    }
    let half = cols / 2;
    let rows: usize = sh[..rank - 1].iter().product();
    let dim = cols as i64;
    let clamp = |x: i64| {
        if x < 0 {
            (dim + x).max(0) as usize
        } else {
            x.min(dim) as usize
        }
    };
    match (clamp(start), clamp(end)) {
        (0, e) if e == half => Some((src, rows, half, false)),
        (s, e) if s == half && e == cols => Some((src, rows, half, true)),
        _ => None,
    }
}

/// Saturates the MoE window rules over `steps` and extracts, in step order, the fusion each
/// window start takes (scaled TopK before plain TopK; for a weighted sum, `down` is the first
/// operand when both operand orders fit).
pub(super) fn moe(
    g: &Graph,
    steps: &[Planned],
    aliases: &HashMap<ValueId, (ValueId, usize)>,
) -> Result<HashMap<usize, MoeFusion>, String> {
    use super::ext::{elementwise, movement, norm, ExtStep};
    // Every rule is anchored on a TopK, a GLU or a Reduce step: skip the e-graph without one.
    if !steps.iter().any(|s| {
        matches!(
            s.op,
            Step::Ext(ExtStep::Experts(super::ext::moe::Step::TopK { .. }))
                | Step::Ext(ExtStep::Norm(norm::Step::Reduce { .. }))
                | Step::GeGlu { .. }
                | Step::SwiGlu { .. }
        )
    }) {
        return Ok(HashMap::new());
    }
    let root = |v: ValueId| aliases.get(&v).map_or(v, |a| a.0);
    let mut use_count: HashMap<ValueId, usize> = HashMap::new();
    for v in steps.iter().flat_map(|s| &s.reads).chain(&g.outputs) {
        *use_count.entry(root(*v)).or_default() += 1;
    }
    let vi = |v: ValueId| v.0 as i64;
    let (schema, run) = MOE_PROGRAM.split_at(MOE_PROGRAM.find("(run-schedule").expect("schedule"));
    let mut eg = loaded(schema)?;
    eg.update(|mut fs| -> Result<(), egglog::Error> {
        for (v, _) in use_count.iter().filter(|(_, &n)| n == 1) {
            fs.add("Single", vi(*v))?;
        }
        let shape2 = |fs: &mut egglog::FullState<'_, '_>, v: ValueId| match g.value(v).shape[..] {
            [a, b] => fs.add("Shape2", (vi(v), a as i64, b as i64)).map(|_| ()),
            _ => Ok(()),
        };
        for (p, s) in steps.iter().enumerate() {
            let pi = p as i64;
            if s.extra_outs.is_empty() {
                fs.add("Solo", pi)?;
            }
            let o = vi(s.out);
            match &s.op {
                Step::Ext(ExtStep::Experts(super::ext::moe::Step::TopK {
                    x,
                    rows,
                    cols,
                    k,
                    weights_out: None,
                    scale: None,
                })) if *k <= 32 => {
                    let f = (pi, vi(*x), *rows as i64, *cols as i64, *k as i64, o);
                    fs.add("TopK", f)?;
                }
                Step::Ext(ExtStep::Experts(super::ext::moe::Step::GatherCols {
                    x,
                    idx,
                    total,
                    cols,
                    k,
                    shared,
                })) => {
                    let f = (pi, vi(*x), vi(*idx), *total as i64, *cols as i64, *k as i64);
                    fs.add("Gat", (f.0, f.1, f.2, f.3, f.4, f.5, *shared as i64, o))?;
                }
                Step::Ext(ExtStep::Norm(norm::Step::Softmax {
                    x,
                    outer,
                    n,
                    inner: 1,
                })) => {
                    fs.add("Smax", (pi, vi(*x), *outer as i64, *n as i64, o))?;
                }
                Step::Ext(ExtStep::Elementwise(elementwise::Step::Binary {
                    a,
                    b,
                    n,
                    kind,
                    rank,
                    ..
                })) => {
                    let f = (pi, vi(*a), vi(*b), *n as i64, *kind as i64, *rank as i64, o);
                    fs.add("Bin", f)?;
                    for v in [*a, *b, s.out] {
                        shape2(&mut fs, v)?;
                    }
                }
                Step::Ext(ExtStep::Movement(movement::Step::Copy { fill: None, parts }))
                    if parts.len() == 1 =>
                {
                    fs.add("Copy1", (pi, o))?;
                }
                Step::GeGlu { gate, up, n } | Step::SwiGlu { gate, up, n } => {
                    let kind = if matches!(s.op, Step::GeGlu { .. }) {
                        1
                    } else {
                        2
                    };
                    fs.add("Glu", (pi, vi(*gate), vi(*up), *n as i64, kind, o))?;
                    for v in [*gate, *up] {
                        if let Some((src, rows, half, upper)) = slice_half(g, v) {
                            let f = (vi(v), vi(src), rows as i64, half as i64, upper as i64);
                            fs.add("Half", f)?;
                        }
                    }
                }
                Step::Ext(ExtStep::Norm(norm::Step::Reduce { x, n_out, .. })) => {
                    let ValueKind::Node(nid) = g.value(s.out).kind else {
                        continue;
                    };
                    let Op::Reduce {
                        kind: crate::graph::ReduceKind::Sum,
                        axes,
                        ..
                    } = &g.nodes[nid.0 as usize].op
                    else {
                        continue;
                    };
                    let rsh = &g.value(*x).shape;
                    let ax = match axes[..] {
                        [a] if a < 0 => a + rsh.len() as isize,
                        [a] => a,
                        _ => continue,
                    };
                    if ax < 0 || ax as usize >= rsh.len() {
                        continue;
                    }
                    let ax = ax as usize;
                    let n: usize = rsh[..ax].iter().product();
                    let h: usize = rsh[ax + 1..].iter().product();
                    let (rx, off) = aliases.get(x).copied().unwrap_or((*x, 0));
                    let f = (pi, vi(rx), off as i64, *n_out as i64, o, n as i64);
                    fs.add(
                        "RedSum",
                        (f.0, f.1, f.2, f.3, f.4, f.5, rsh[ax] as i64, h as i64),
                    )?;
                }
                _ => {}
            }
        }
        Ok(())
    })
    .map_err(err)?;
    eg.parse_and_run_program(None, run).map_err(err)?;

    let v = |x: i64| ValueId(x as u32);
    let mut d: HashMap<usize, MoeFusion> = HashMap::new();
    // Lowest priority first: later inserts at the same start win.
    let mut wsum = relation(&eg, "WSum")?;
    wsum.sort_by_key(|r| std::cmp::Reverse(r[1]));
    for r in wsum {
        let (n, k, h) = (r[4] as usize, r[5] as usize, r[6] as usize);
        let f = MoeFusion::WeightedSum {
            down: v(r[2]),
            weights: v(r[3]),
            n,
            k,
            h,
        };
        d.insert(r[0] as usize, f);
    }
    for r in relation(&eg, "SplitG")? {
        let f = MoeFusion::SplitGlu {
            x: v(r[1]),
            rows: r[2] as usize,
            half_cols: r[3] as usize,
            kind: r[4] as u32,
        };
        d.insert(r[0] as usize, f);
    }
    for r in relation(&eg, "MoeTopK")? {
        d.insert(r[0] as usize, MoeFusion::TopKSoftmax { weights: v(r[1]) });
    }
    for r in relation(&eg, "MoeScaled")? {
        let f = MoeFusion::TopKSoftmaxScaled {
            scale: v(r[1]),
            weights: v(r[2]),
        };
        d.insert(r[0] as usize, f);
    }
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::super::{batch_horizontal, plan};
    use super::*;
    use crate::graph::{ReduceKind, Weights};

    const F: DType = DType::F32;

    /// Linear epilogue (residual), RmsNorm residual + scale, and attention q/k RoPE are each
    /// claimed once, by the node the rules name.
    #[test]
    fn decide_claims_epilogue_norm_and_attention_rope() {
        let (h, hd) = (64usize, 16usize);
        let mut g = Graph::new();
        let x = g.input("x", vec![1, h], F);
        let pos = g.input("pos", vec![1], DType::U32);
        let gw = g.weight("g", vec![h], F);
        let s = g.weight("s", vec![1], F);
        let inv = g.weight("inv", vec![hd / 2], F);
        let w = |g: &mut Graph, n: &str| g.weight(n, vec![h, h], DType::F16);
        let (wq, wk, wv, wo) = (
            w(&mut g, "wq"),
            w(&mut g, "wk"),
            w(&mut g, "wv"),
            w(&mut g, "wo"),
        );
        let n0 = g.add("n0", Op::RmsNorm { eps: 1e-5 }, &[x, gw], vec![1, h], F);
        let o = g.add("o", Op::Linear, &[n0, wo], vec![1, h], F);
        let r = g.add(
            "r",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[o, x],
            vec![1, h],
            F,
        );
        let n1 = g.add("n1", Op::RmsNorm { eps: 1e-5 }, &[r, gw], vec![1, h], F);
        let r1 = g.add(
            "r1",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[r, n1],
            vec![1, h],
            F,
        );
        let sc = g.add(
            "sc",
            Op::Binary {
                kind: crate::graph::BinaryKind::Mul,
            },
            &[r1, s],
            vec![1, h],
            F,
        );
        let rope = Op::Rope {
            head_dim: hd,
            rotary_dim: hd,
            variant: RopeVariant::HalfSplit,
            mscale: 1.0,
        };
        let q = g.add("q", Op::Linear, &[sc, wq], vec![1, h], F);
        let k = g.add("k", Op::Linear, &[sc, wk], vec![1, h], F);
        let v = g.add("v", Op::Linear, &[sc, wv], vec![1, h], F);
        let qr = g.add("qr", rope.clone(), &[q, pos, inv], vec![1, h], F);
        let kr = g.add("kr", rope, &[k, pos, inv], vec![1, h], F);
        let ks = g.state("ks", vec![8, h], F, false);
        let vs = g.state("vs", vec![8, h], F, false);
        let attn = Op::CachedAttention {
            heads: h / hd,
            kv_heads: h / hd,
            head_dim: hd,
            v_head_dim: hd,
            scale: 0.25,
            causal: true,
            window: None,
            soft_cap: None,
            write: true,
        };
        let a = g.add("a", attn, &[qr, kr, v, ks, vs, pos], vec![1, h], F);
        g.output(a);
        let node = |v: ValueId| match g.value(v).kind {
            ValueKind::Node(n) => n.0 as usize,
            _ => unreachable!(),
        };
        let d = decide(&g).unwrap();
        assert_eq!(d.epi.len(), 1);
        let &(j, epi, side) = &d.epi[&node(o)];
        assert_eq!((j, epi, side), (node(r), Epi::Residual, Some(x)));
        let nf = d.norm[&node(n1)];
        assert_eq!(nf.res, Some((node(r1), r)));
        assert_eq!(nf.scale, Some((node(sc), s)));
        assert!(nf.rope.is_none() && !d.norm.contains_key(&node(n0)));
        assert_eq!(d.attn_rope[&node(a)], (node(qr), node(kr)));
        let mut absorbed: Vec<usize> = d.absorbed().collect();
        absorbed.sort_unstable();
        let mut want = vec![node(r), node(r1), node(sc), node(qr), node(kr)];
        want.sort_unstable();
        assert_eq!(absorbed, want);
    }

    /// Passes leave values whose producer node is gone: `decide` encodes only referenced values.
    #[test]
    fn decide_skips_values_of_removed_nodes() {
        let mut g = Graph::new();
        let x = g.input("x", vec![1, 64], F);
        let gw = g.weight("g", vec![64], F);
        let n = g.add("n", Op::RmsNorm { eps: 1e-5 }, &[x, gw], vec![1, 64], F);
        let r = g.add(
            "r",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[n, x],
            vec![1, 64],
            F,
        );
        g.add(
            "dead",
            Op::Unary {
                kind: crate::graph::UnaryKind::Silu,
            },
            &[r],
            vec![1, 64],
            F,
        );
        g.nodes.pop();
        g.output(r);
        let d = decide(&g).unwrap();
        assert_eq!(d.norm[&0].res, Some((1, x)));
    }

    /// Norms around an independent elementwise step do not batch; norms around a GEMV batch.
    #[test]
    fn batches_commute_gemvs() {
        let mut g = Graph::new();
        let x = g.input("x", vec![1, 64], F);
        let y = g.input("y", vec![1, 64], F);
        let gw = g.weight("g", vec![64], F);
        let w = g.weight("w", vec![64, 64], DType::F16);
        let a = g.add("a", Op::RmsNorm { eps: 1e-5 }, &[x, gw], vec![1, 64], F);
        let neg = Op::Unary {
            kind: crate::graph::UnaryKind::Neg,
        };
        let m = g.add("m", neg, &[y], vec![1, 64], F);
        let b = g.add("b", Op::RmsNorm { eps: 1e-5 }, &[y, gw], vec![1, 64], F);
        let l = g.add("l", Op::Linear, &[x, w], vec![1, 64], F);
        let c = g.add("c", Op::RmsNorm { eps: 1e-5 }, &[x, gw], vec![1, 64], F);
        for o in [a, m, b, l, c] {
            g.output(o);
        }
        let (steps, aliases) = plan(&g).unwrap();
        assert_eq!(steps.len(), 5);
        let off = batches(&steps, &aliases).unwrap();
        assert_eq!(off.runs, HashMap::from([(2, vec![2, 4])]));
        let kinds: Vec<String> = batch_horizontal(steps, &aliases)
            .unwrap()
            .iter()
            .map(|s| super::super::kernel_kind(&format!("{:?}", s.op)).to_string())
            .collect();
        assert_eq!(kinds, ["RmsNorm", "Unary", "GemvPacked", "RmsNormBatch"]);
    }

    /// Router TopK + GatherCols + Softmax (+ scale), the gate/up half-split GLU and the
    /// expert-weighted Sum each plan as one step.
    #[test]
    fn moe_windows_plan_as_fused_steps() {
        let (m, h, ff, experts, top_k) = (2usize, 16usize, 16usize, 6usize, 3usize);
        let mut g = Graph::new();
        let x = g.input("x", vec![m, h], F);
        let logits = g.input("logits", vec![m, experts], F);
        let scale_w = g.weight("scale", vec![1, experts], F);
        let gu_w = g.weight("gate_up", vec![experts * 2 * ff, h], F);
        let down_w = g.weight("down", vec![experts * h, ff], F);
        let ids = g.add(
            "topk",
            Op::TopK { k: top_k },
            &[logits],
            vec![m, top_k],
            DType::U32,
        );
        let picked = g.add("picked", Op::GatherCols, &[logits, ids], vec![m, top_k], F);
        let probs = g.add(
            "probs",
            Op::Softmax { axis: -1 },
            &[picked],
            vec![m, top_k],
            F,
        );
        let sc = g.add("sc", Op::GatherCols, &[scale_w, ids], vec![m, top_k], F);
        let wv = g.add(
            "weights",
            Op::Binary {
                kind: crate::graph::BinaryKind::Mul,
            },
            &[probs, sc],
            vec![m, top_k],
            F,
        );
        let moe = Op::MoeLinear { experts, top_k };
        let gu = g.add(
            "gu",
            moe.clone(),
            &[x, gu_w, ids],
            vec![m * top_k, 2 * ff],
            F,
        );
        let half = |g: &mut Graph, n: &str, lo: usize| {
            let op = Op::Slice {
                axis: 1,
                start: lo as i64,
                end: (lo + ff) as i64,
            };
            g.add(n, op, &[gu], vec![m * top_k, ff], F)
        };
        let (gate, up) = (half(&mut g, "gate", 0), half(&mut g, "up", ff));
        let act = g.add("act", Op::GeGlu, &[gate, up], vec![m * top_k, ff], F);
        let down = g.add("down", moe, &[act, down_w, ids], vec![m * top_k, h], F);
        let col = Op::Reshape {
            shape: Some(vec![m * top_k, 1]),
        };
        let w_col = g.add("w_col", col, &[wv], vec![m * top_k, 1], F);
        let weighted = g.add(
            "weighted",
            Op::Binary {
                kind: crate::graph::BinaryKind::Mul,
            },
            &[down, w_col],
            vec![m * top_k, h],
            F,
        );
        let slots = Op::Reshape {
            shape: Some(vec![m, top_k, h]),
        };
        let slots = g.add("slots", slots, &[weighted], vec![m, top_k, h], F);
        let sum = Op::Reduce {
            kind: ReduceKind::Sum,
            axes: vec![1],
            keep_dims: false,
        };
        let out = g.add("out", sum, &[slots], vec![m, h], F);
        g.output(out);
        let _ = Weights::new();
        let (steps, _) = plan(&g).unwrap();
        let ops: Vec<String> = steps.iter().map(|s| format!("{:?}", s.op)).collect();
        assert_eq!(steps.len(), 5, "{ops:#?}");
        assert!(
            ops[0].contains("TopK") && ops[0].contains("scale: Some"),
            "{}",
            ops[0]
        );
        assert!(ops[2].starts_with("SplitGlu"), "{}", ops[2]);
        assert!(ops[4].starts_with("MoeWeightedSum"), "{}", ops[4]);
    }
}
