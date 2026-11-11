//! Drafter steps on the ANE, the one place state ops reach it. [`lower_drafter_step`] rewrites a
//! one-row decode instance: the `Embedding` row of `token_ids` becomes input `embed` (the host
//! decodes it), and each `CachedAttention` becomes an `Attention` over a static `W`-row key/value
//! window passed as inputs (`ring.<state>`), with the step's own row put in slot `pos % W`
//! (input `ring.onehot`) and a validity mask (`ring.mask.<window>`) as bias. The new K/V rows are
//! extra outputs, which [`AneDraft::run`] writes into that slot of the window surfaces, so the
//! window lives in the programs' own input surfaces and no step copies it.

use super::{lower, rope_table, AneInfo, AneProgram, IoLayout, IoSource};
use crate::graph::{BinaryKind, DType, Graph, Op, ValueId, ValueKind, Weights};
use std::collections::HashMap;
use std::sync::{Arc, Weak};

/// Key/value rows of each drafter attention window on the ANE.
pub const DRAFT_WINDOW: usize = 1024;
/// Newest window rows a step never attends: a fork drafting up to `k + 1` positions ahead of its
/// sequence (`k <= 15`) and a sequence rewinding up to `k` positions share one window, so neither
/// may read a slot the other overwrote.
const RING_MARGIN: usize = 32;
const MASKED: f32 = -65504.0;
const EMPTY: u32 = u32::MAX;

/// A one-row decode instance rewritten by [`lower_drafter_step`].
pub struct StepGraph {
    pub graph: Graph,
    /// The `Embedding` table the host decodes input `embed` from, and the row width.
    pub embed: (String, usize),
    /// Extra `Embedding` inputs (`(input_name, weight_name, row_width)`) when >1 Embedding reads `token_ids`.
    pub extra_embeds: Vec<(String, String, usize)>,
    /// Active K/V ring window capacity (`<= DRAFT_WINDOW`; shrunk to power-of-2 when all layers are sliding-window).
    pub window_len: usize,
    /// Window inputs `[W, row]` and the state each stands for.
    rings: Vec<(ValueId, String)>,
    onehot: ValueId,
    /// Validity-mask inputs `[W]` by attention window.
    masks: Vec<(ValueId, Option<usize>)>,
    /// The instance's output count; outputs past it are the new rows of `rows`' states.
    outs: usize,
    rows: Vec<String>,
}

/// Rewrites the one-row decode instance `g` for the ANE with [`DRAFT_WINDOW`]-row K/V windows (see
/// the module docs). Errs, naming the node, on anything the ANE cannot run: a second `Embedding`, other
/// state ops, a bidirectional or multi-row cached attention, or an op [`super::check`] refuses.
pub fn lower_drafter_step(g: &Graph) -> Result<StepGraph, String> {
    let f32_ = DType::F32;
    // Size the ring window tightly when every CachedAttention in `g` uses a sliding window.
    let mut all_windowed = true;
    let mut max_win = 0usize;
    let mut saw_attn = false;
    for n in &g.nodes {
        if let Op::CachedAttention { window, .. } = n.op {
            saw_attn = true;
            match window {
                Some(win) => max_win = max_win.max(win),
                None => all_windowed = false,
            }
        }
    }
    let w = if saw_attn && all_windowed {
        (max_win + RING_MARGIN)
            .max(64)
            .next_power_of_two()
            .min(DRAFT_WINDOW)
    } else {
        DRAFT_WINDOW
    };
    let mut s = Graph::new();
    let mut map: HashMap<ValueId, ValueId> = HashMap::new();
    let ids = g
        .inputs
        .iter()
        .copied()
        .find(|&v| g.value(v).name == "token_ids");
    let mut embedded: Option<(String, usize)> = None;
    let mut extra_embeds: Vec<(String, String, usize)> = Vec::new();
    let mut embed_by_table: HashMap<String, ValueId> = HashMap::new();
    let mut onehot = None;
    let (mut rings, mut masks, mut rows) =
        (Vec::new(), Vec::new(), Vec::<(ValueId, String)>::new());
    // Written ring value per state name (supports independent shared K/V and `k_eq_v` where `ks == vs`).
    let mut written: HashMap<String, ValueId> = HashMap::new();
    for n in &g.nodes {
        let name = &n.name;
        let [out] = n.outputs[..] else {
            return Err(format!("node '{name}' has {} outputs", n.outputs.len()));
        };
        let ov = g.value(out);
        // Operand `v` in the step graph: graph inputs and weights are declared on first use.
        let mut operand = |s: &mut Graph, v: ValueId| -> Result<ValueId, String> {
            if let Some(&x) = map.get(&v) {
                return Ok(x);
            }
            let val = g.value(v);
            let x = match &val.kind {
                ValueKind::Input => s.input(val.name.clone(), val.shape.clone(), val.dtype),
                ValueKind::Weight => s.weight(val.name.clone(), val.shape.clone(), val.dtype),
                _ => {
                    return Err(format!(
                        "node '{name}' reads '{}', which the ANE step cannot hold",
                        val.name
                    ))
                }
            };
            map.insert(v, x);
            Ok(x)
        };
        match &n.op {
            Op::Embedding if ids.is_some() && n.inputs.get(1) == ids.as_ref() => {
                let table = g.value(n.inputs[0]).name.clone();
                let width: usize = ov.shape.iter().product();
                let e = if let Some(&existing) = embed_by_table.get(&table) {
                    existing
                } else if embedded.is_none() {
                    embedded = Some((table.clone(), width));
                    let inp = s.input("embed", ov.shape.clone(), f32_);
                    embed_by_table.insert(table, inp);
                    inp
                } else {
                    let in_name = format!("embed.{}", extra_embeds.len() + 1);
                    let inp = s.input(in_name.clone(), ov.shape.clone(), f32_);
                    extra_embeds.push((in_name, table.clone(), width));
                    embed_by_table.insert(table, inp);
                    inp
                };
                map.insert(out, e);
            }
            &Op::CachedAttention {
                heads,
                kv_heads,
                head_dim,
                v_head_dim,
                scale,
                causal,
                window,
                soft_cap,
                write,
            } => {
                let state = |k: usize| match n.inputs.get(k).map(|&v| &g.value(v).kind) {
                    Some(ValueKind::State { name, .. }) => Ok(name.clone()),
                    _ => Err(format!("node '{name}' attends no cached state")),
                };
                let (ks, vs) = (state(3)?, state(4)?);
                let q = operand(&mut s, n.inputs[0])?;
                let rows_q = s.value(q).shape.iter().product::<usize>() / (heads * head_dim).max(1);
                if !causal || rows_q != 1 {
                    return Err(format!(
                        "node '{name}': only a one-row causal cached attention lowers"
                    ));
                }
                let (keys, vals) = if write {
                    let oh =
                        *onehot.get_or_insert_with(|| s.input("ring.onehot", vec![w, 1], f32_));
                    let mut ring = |s: &mut Graph,
                                    written: &mut HashMap<String, ValueId>,
                                    st: &str,
                                    x: ValueId,
                                    width: usize|
                     -> Result<ValueId, String> {
                        if let Some(&already) = written.get(st) {
                            return Ok(already);
                        }
                        let x = operand(s, x)?;
                        let shape = Some(vec![1, width]);
                        let row = s.add(
                            format!("{st}.row"),
                            Op::Reshape { shape },
                            &[x],
                            vec![1, width],
                            f32_,
                        );
                        rows.push((row, st.to_string()));
                        let win = s.input(format!("ring.{st}"), vec![w, width], f32_);
                        rings.push((win, st.to_string()));
                        let bin = |kind| Op::Binary { kind };
                        let d = s.add(
                            format!("{st}.d"),
                            bin(BinaryKind::Sub),
                            &[row, win],
                            vec![w, width],
                            f32_,
                        );
                        let d = s.add(
                            format!("{st}.dm"),
                            bin(BinaryKind::Mul),
                            &[d, oh],
                            vec![w, width],
                            f32_,
                        );
                        let upd = s.add(
                            format!("{st}.win"),
                            bin(BinaryKind::Add),
                            &[win, d],
                            vec![w, width],
                            f32_,
                        );
                        written.insert(st.to_string(), upd);
                        Ok(upd)
                    };
                    let k_win = ring(&mut s, &mut written, &ks, n.inputs[1], kv_heads * head_dim)?;
                    let v_win = ring(
                        &mut s,
                        &mut written,
                        &vs,
                        n.inputs[2],
                        kv_heads * v_head_dim,
                    )?;
                    (k_win, v_win)
                } else {
                    let k_win = *written.get(&ks).ok_or_else(|| {
                        format!("node '{name}' reads K state {ks} before any write")
                    })?;
                    let v_win = *written.get(&vs).ok_or_else(|| {
                        format!("node '{name}' reads V state {vs} before any write")
                    })?;
                    (k_win, v_win)
                };
                let mask = match masks.iter().find(|(_, mw)| *mw == window) {
                    Some(&(m, _)) => m,
                    None => {
                        let label = window.map_or("all".into(), |x| x.to_string());
                        let m = s.input(format!("ring.mask.{label}"), vec![w], f32_);
                        masks.push((m, window));
                        m
                    }
                };
                let op = Op::Attention {
                    heads,
                    kv_heads,
                    causal: false,
                    window: None,
                    soft_cap,
                    scale: Some(scale),
                };
                let o = s.add(
                    name.clone(),
                    op,
                    &[q, keys, vals, mask],
                    ov.shape.clone(),
                    ov.dtype,
                );
                map.insert(out, o);
            }
            op => {
                let ins = n
                    .inputs
                    .iter()
                    .map(|&v| operand(&mut s, v))
                    .collect::<Result<Vec<_>, _>>()?;
                let o = s.add(name.clone(), op.clone(), &ins, ov.shape.clone(), ov.dtype);
                map.insert(out, o);
            }
        }
    }
    let embed = embedded.ok_or("no Embedding reads token_ids")?;
    let onehot = onehot.ok_or("the step attends no cache")?;
    for &o in &g.outputs {
        let v = *map.get(&o).ok_or("an output is not produced by a node")?;
        s.output(v);
    }
    for &(r, _) in &rows {
        s.output(r);
    }
    for node in &s.nodes {
        super::check(&s, node).map_err(|e| format!("node '{}': {e}", node.name))?;
    }
    Ok(StepGraph {
        embed,
        extra_embeds,
        window_len: w,
        outs: g.outputs.len(),
        rows: rows.into_iter().map(|(_, st)| st).collect(),
        graph: s,
        rings,
        onehot,
        masks,
    })
}

/// One drafter step compiled for the ANE.
pub struct DraftStep {
    graph: Graph,
    program: AneProgram,
    io: IoLayout,
    /// The `inv_freq` weights its host-side Rope tables read.
    tables: Weights,
    /// Input surface of each state's window.
    rings: HashMap<String, usize>,
    onehot: usize,
    masks: Vec<(usize, Option<usize>)>,
    outs: usize,
    rows: Vec<String>,
    /// FNV-64 of the MIL text.
    sig: u64,
    window_len: usize,
    extra_tables: Vec<(usize, crate::graph::Tensor, usize)>,
}

impl DraftStep {
    pub fn compile(sg: StepGraph, weights: &Weights, info: &AneInfo) -> Result<Self, String> {
        let (mil, blob, io) = lower(&sg.graph, weights, info)?;
        let surface = |v: ValueId| {
            io.inputs
                .iter()
                .position(|b| b.source == IoSource::Value(v))
                .ok_or_else(|| format!("input '{}' is not bound", sg.graph.value(v).name))
        };
        let rings = sg
            .rings
            .iter()
            .map(|(v, st)| Ok((st.clone(), surface(*v)?)))
            .collect::<Result<_, String>>()?;
        let masks = sg
            .masks
            .iter()
            .map(|&(v, w)| Ok((surface(v)?, w)))
            .collect::<Result<_, String>>()?;
        let mut tables = Weights::new();
        for b in &io.inputs {
            if let IoSource::RopeTable(n) = b.source {
                let name = &sg.graph.value(sg.graph.nodes[n].inputs[2]).name;
                let t = weights
                    .get(name)
                    .ok_or_else(|| format!("weight '{name}' is not bound"))?;
                tables.insert(name.clone(), t.clone());
            }
        }
        let mut extra_tables = Vec::new();
        for (in_name, w_name, width) in &sg.extra_embeds {
            if let Some(pos_idx) = io.inputs.iter().position(|b| match b.source {
                IoSource::Value(v) => sg.graph.value(v).name == *in_name,
                _ => false,
            }) {
                let t = weights
                    .get(w_name)
                    .ok_or_else(|| format!("weight '{w_name}' is not bound"))?;
                extra_tables.push((pos_idx, t.clone(), *width));
            }
        }
        let program = AneProgram::compile(
            &mil,
            &blob,
            io.inputs.len(),
            io.in_bytes(),
            io.outputs.len(),
            io.out_bytes(),
        )?;
        Ok(Self {
            onehot: surface(sg.onehot)?,
            sig: super::fnv64(&[mil.as_bytes()]),
            window_len: sg.window_len,
            extra_tables,
            graph: sg.graph,
            program,
            io,
            tables,
            rings,
            masks,
            outs: sg.outs,
            rows: sg.rows,
        })
    }

    /// Evaluates position `pos` (`w`-row windows) with F32 inputs by name (`embed`, ...), the
    /// mask of each attention window from `mask`; every output, the instance's first.
    fn eval(
        &mut self,
        inputs: &[(&str, &[f32])],
        pos: usize,
        tok_opt: Option<u32>,
        w: usize,
        mask: &dyn Fn(Option<usize>) -> Vec<f32>,
    ) -> Result<Vec<Vec<f32>>, String> {
        for (i, b) in self.io.inputs.iter().enumerate() {
            let data = match b.source {
                _ if self.rings.values().any(|&r| r == i) => continue,
                _ if i == self.onehot => (0..w).map(|s| (s == pos % w) as u8 as f32).collect(),
                IoSource::Value(v) => match self.masks.iter().find(|m| m.0 == i) {
                    Some(&(_, win)) => mask(win),
                    None => {
                        let name = &self.graph.value(v).name;
                        match inputs.iter().find(|(n, _)| n == name) {
                            Some((_, d)) => d.to_vec(),
                            None if name == "positions" => vec![pos as f32],
                            None => {
                                if let Some((_, tbl, width)) =
                                    self.extra_tables.iter().find(|(si, ..)| *si == i)
                                {
                                    let tid = tok_opt.unwrap_or(0) as usize;
                                    let all = tbl.to_f32();
                                    let start = tid.saturating_mul(*width);
                                    if start + *width <= all.len() {
                                        all[start..start + *width].to_vec()
                                    } else {
                                        vec![0.0; *width]
                                    }
                                } else {
                                    return Err(format!(
                                        "ANE draft step input '{name}' is not supplied"
                                    ));
                                }
                            }
                        }
                    }
                },
                IoSource::RopeTable(n) => {
                    let node = &self.graph.nodes[n];
                    let inv = &self.graph.value(node.inputs[2]).name;
                    let inv = self
                        .tables
                        .get(inv)
                        .ok_or("rope table weight is not bound")?
                        .to_f32();
                    let per_row = self
                        .graph
                        .value(node.inputs[1])
                        .shape
                        .iter()
                        .product::<usize>();
                    rope_table(&node.op, &vec![pos as u32; per_row], &inv, b.spatial)?
                }
            };
            self.program.input(i).write(|d| b.pack(&data, d))?;
        }
        self.program.eval()?;
        self.io
            .outputs
            .iter()
            .enumerate()
            .map(|(i, b)| self.program.output(i).read(|s| b.unpack(s)))
            .collect()
    }

    /// Writes `row` into slot `slot` of `state`'s window surface (if this step reads it).
    fn write_row(&self, state: &str, slot: usize, row: &[f32]) {
        let Some(&i) = self.rings.get(state) else {
            return;
        };
        let b = &self.io.inputs[i];
        self.program.input(i).write(|d| {
            for (j, x) in row.iter().enumerate() {
                let o = b.offset(slot * row.len() + j);
                d[o..o + 4].copy_from_slice(&x.to_le_bytes());
            }
        });
    }
}

/// One sequence's view of an [`AneDraft`]'s windows: the position each slot holds and the token
/// ids of the positions run. Clones (forks) share the windows: [`RING_MARGIN`] keeps them apart.
#[derive(Clone)]
pub struct Ring {
    token: Arc<()>,
    slots: Vec<u32>,
    pub ids: Vec<u32>,
}

impl Ring {
    pub fn new() -> Self {
        Self {
            token: Arc::new(()),
            slots: vec![EMPTY; DRAFT_WINDOW],
            ids: Vec::new(),
        }
    }

    /// Forgets positions `len..`.
    pub fn truncate(&mut self, len: usize) {
        self.ids.truncate(len);
        for s in &mut self.slots {
            if *s != EMPTY && *s as usize >= len {
                *s = EMPTY;
            }
        }
    }
}

impl Default for Ring {
    fn default() -> Self {
        Self::new()
    }
}

/// A drafter's ANE steps (e.g. catch-up and draft instances) sharing one set of windows, owned
/// by one sequence (and its forks) at a time.
pub struct AneDraft {
    steps: Vec<DraftStep>,
    owner: Weak<()>,
}

impl AneDraft {
    pub fn new(steps: Vec<DraftStep>) -> Self {
        Self {
            steps,
            owner: Weak::new(),
        }
    }

    /// FNV-64 over the steps' MIL.
    pub fn sig(&self) -> u64 {
        let sigs: Vec<[u8; 8]> = self.steps.iter().map(|s| s.sig.to_le_bytes()).collect();
        super::fnv64(&sigs.iter().map(|s| &s[..]).collect::<Vec<_>>())
    }

    /// Whether `r`'s sequence may use the windows: it (or a fork) owns them, or nobody does and
    /// it takes them, starting empty.
    // ponytail: one sequence drafts on the ANE at a time, the others on Metal; give each its own
    // windows (more surfaces, one request each) if concurrent ANE drafting matters.
    pub fn claim(&mut self, r: &mut Ring) -> bool {
        match self.owner.upgrade() {
            Some(t) => Arc::ptr_eq(&t, &r.token),
            None => {
                self.owner = Arc::downgrade(&r.token);
                r.slots.fill(EMPTY);
                r.ids.clear();
                true
            }
        }
    }

    /// Runs step `i` at position `pos` of `r` (claimed) with F32 inputs by name; writes the new
    /// K/V rows into every step's windows and returns the instance's outputs.
    pub fn run(
        &mut self,
        i: usize,
        r: &mut Ring,
        pos: usize,
        inputs: &[(&str, &[f32])],
    ) -> Result<Vec<Vec<f32>>, String> {
        let w = self.steps[i].window_len;
        let slot = pos % w;
        if r.slots.len() < w {
            r.slots.resize(w, EMPTY);
        }
        let reach = w.saturating_sub(RING_MARGIN);
        let slots = &r.slots;
        let mask = |win: Option<usize>| {
            let lim = win.unwrap_or(usize::MAX).min(reach);
            (0..w)
                .map(|s| {
                    let p = slots[s] as usize;
                    let seen = s == slot || (slots[s] != EMPTY && p < pos && pos - p < lim);
                    if seen {
                        0.0
                    } else {
                        MASKED
                    }
                })
                .collect()
        };
        let tok_opt = r.ids.get(pos).copied();
        let mut out = self.steps[i].eval(inputs, pos, tok_opt, w, &mask)?;
        let new = out.split_off(self.steps[i].outs);
        for st in &self.steps {
            for (state, row) in self.steps[i].rows.iter().zip(&new) {
                st.write_row(state, slot, row);
            }
        }
        r.slots[slot] = pos as u32;
        Ok(out)
    }
}
