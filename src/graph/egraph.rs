//! Equality-saturation fusion: the graph-level fusions (megakernel MLP, gate/up, QKV, and the
//! opt-in QKV-slices→attention rewrite) as non-destructive egglog rules. Every fused form is
//! unioned into the e-class of the value it computes, so the unfused forms stay available, and one
//! extraction picks a consistent set of fusions with a cost model. The chosen fusions are then
//! materialized by the rewriters in [`crate::graph::passes`] (`fuse_mlp_blocks`, `rewrite_gate_up`,
//! `rewrite_qkv`), restricted to the extracted members.
//!
//! * **Encoding.** Every value is an e-class named `(V value-id)` (no global `let`s: they
//!   dominate parse time on model graphs); a node's class also holds `(Node node-id args)`.
//!   Facts carry what the rules test: op kind `(K id "Linear")`,
//!   consumer count `Uses`, single-row activations `OneRow`, megakernel-decodable weights `MkW`.
//!   Two-input Linears with a concatenable rank-2 weight also get the view `(Lin key x (Wt w))`
//!   with `wrows`/`wkey` (dtype, columns and storage, so a rule only fuses weights the
//!   concatenation accepts).
//! * **Rules** (ruleset `fuse`): `mlp-down` → `mlp-gate` → `mlp-up` → `mlp-norm` → `mlp-body`
//!   find the MLP chain one node per join (one 20-atom join took 43 s on a 30-layer graph),
//!   then `mlp-add[-swapped]` / `mlp-post-norm-add[-swapped]`
//!   union the residual Add with `Mlp`/`MlpPn`; `gate-up` unions gate and up with `Slice`s of
//!   one `Lin` over `(Cat2 wg wu)`; `qkv` unions three Linears sharing a node input with
//!   `Slice`s of a `Lin` over `Cat3`.
//! * **Budget.** [`Budget::match_limit`] caps the matches each rule may fire in total (luminal's
//!   `AlgebraBudget`), [`Budget::max_iters`] the scheduler steps.
//! * **Extraction.** DAG cost with sharing: every e-class reachable from the outputs (or from a
//!   node nobody reads) costs its chosen e-node once,
//!   `dispatch_us·[launches] + activation bytes / ceiling + Σ weight bytes / rate(format)`
//!   ([`CostModel`]). A fusion group (all `Slice`s of one fused `Lin`, or one `Mlp`/`Rewired`)
//!   switches its member classes together; groups are tried greedily by cost delta (ties: lowest
//!   member node ids, the passes' match order) and kept when the DAG gets cheaper.
//!
//! # Planner fusions
//! The Metal planner's fusions are rules too ([`crate::graph::metal`]'s `fuse` module, and the
//! `ple`/`deltanet` groups): GEMV epilogues, RmsNorm residual/scale/RoPE, attention q/k RoPE,
//! horizontal batching, MoE, PLE and DeltaNet chains are extracted from saturated node/step facts
//! and only lowered by the planner. The allocation-dependent GEMV pairing
//! (`pair_small_hand_gemvs`, which needs arena offsets and tuned variants) stays a planner post-pass.

use crate::graph::passes::fusion::{self, FuseWeights};
use crate::graph::passes::megakernel;
use crate::graph::pattern::MatchContext;
use crate::graph::{DType, Graph, NodeId, Op, Tensor, ValueId, ValueKind, Weights};
use egglog::scheduler::{Matches, Scheduler};
use egglog::{EGraph, Value, Write as _};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Key offset of a fused gate/up `Lin` (`+ glu node id`): after every original node, like the
/// pass appends it.
const FUSED_KEY_BASE: i64 = 1 << 40;

const SCHEMA: &str = r#"
(datatype W (Wt i64) (Cat2 W W) (Cat3 W W W))
(datatype*
  (E (V i64)
     (Node i64 L)
     (Lin i64 E W)
     (Slice E i64 i64)
     (Mlp i64 i64 E E E E E)
     (MlpPn i64 i64 E E E E E i64 E))
  (L (Nil) (Cons E L)))
(relation K (i64 String))
(function Uses (E) i64 :no-merge)
(relation OneRow (E))
(relation MkW (E))
(function wrows (W) i64 :no-merge)
(function wkey (W) String :no-merge)
(relation QkvIn (W))
(relation GluIn (W))
(relation MlpDown (E E E i64 E E))
(relation MlpGate (E E E E E i64 E E))
(relation MlpUp (E E E E E i64 E E E))
(relation MlpChain (E E E E E i64 i64 E E E E E))
(relation MlpBody (E i64 i64 E E E E E))
(ruleset fuse)
"#;

/// The node's kind fact: `"Add"` for a same-shape `Binary{Add}` (the residual the MLP rules
/// anchor on, and the pattern-key text refbase recorded), else [`Op::name`].
fn rule_key(g: &Graph, n: &crate::graph::Node) -> &'static str {
    match (&n.op, &n.inputs[..]) {
        (
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[a, b],
        ) if g.value(a).shape == g.value(b).shape => "Add",
        _ => n.op.name(),
    }
}

/// One ported fusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// `RmsNorm -> gate/up -> GLU -> down -> Add` → one `Megakernel` MLP node.
    MegakernelMlp,
    /// Gate and up Linears of one GLU → one Linear over the concatenated weight.
    GateUp,
    /// Q, K, V Linears of one input → one Linear (all three of one format).
    Qkv,
}

/// The rules of the default pass pipeline.
pub const DEFAULT_RULES: &[Rule] = &[Rule::MegakernelMlp, Rule::GateUp, Rule::Qkv];

impl Rule {
    fn text(self) -> &'static str {
        match self {
            Rule::MegakernelMlp => {
                r#"
(rule ((= d (Node did (Cons s (Cons wd (Nil))))) (K did "Linear")
       (= s (Node lid (Cons g (Cons u (Nil))))) (K lid "Glu"))
      ((MlpDown d s wd lid g u))
      :ruleset fuse :name "mlp-down")
(rule ((MlpDown d s wd lid g u) (= g (Node gid (Cons n (Cons wg (Nil))))) (K gid "Linear"))
      ((MlpGate d s g u n lid wg wd))
      :ruleset fuse :name "mlp-gate")
(rule ((MlpGate d s g u n lid wg wd) (= u (Node uid (Cons n (Cons wu (Nil))))) (K uid "Linear"))
      ((MlpUp d s g u n lid wg wu wd))
      :ruleset fuse :name "mlp-up")
(rule ((MlpUp d s g u n lid wg wu wd) (= n (Node nid (Cons r (Cons wn nr)))) (K nid "RmsNorm"))
      ((MlpChain d s g u n nid lid r wn wg wu wd))
      :ruleset fuse :name "mlp-norm")
(rule ((MlpChain d s g u n nid lid r wn wg wu wd)
       (= (Uses d) 1) (= (Uses s) 1) (= (Uses g) 1) (= (Uses u) 1) (= un (Uses n)) (<= un 2)
       (OneRow s) (OneRow r) (MkW wg) (MkW wu) (MkW wd))
      ((MlpBody d nid lid r wn wg wu wd))
      :ruleset fuse :name "mlp-body")
(rule ((MlpBody d nid lid r wn wg wu wd) (= a (Node aid (Cons r (Cons d (Nil))))) (K aid "Add"))
      ((union a (Mlp nid lid r wn wg wu wd)))
      :ruleset fuse :name "mlp-add")
(rule ((MlpBody d nid lid r wn wg wu wd) (= a (Node aid (Cons d (Cons r (Nil))))) (K aid "Add"))
      ((union a (Mlp nid lid r wn wg wu wd)))
      :ruleset fuse :name "mlp-add-swapped")
(rule ((MlpBody d nid lid r wn wg wu wd)
       (= p (Node pid (Cons d (Cons pw pr)))) (K pid "RmsNorm") (= (Uses p) 1)
       (= a (Node aid (Cons r (Cons p (Nil))))) (K aid "Add"))
      ((union a (MlpPn nid lid r wn wg wu wd pid pw)))
      :ruleset fuse :name "mlp-post-norm-add")
(rule ((MlpBody d nid lid r wn wg wu wd)
       (= p (Node pid (Cons d (Cons pw pr)))) (K pid "RmsNorm") (= (Uses p) 1)
       (= a (Node aid (Cons p (Cons r (Nil))))) (K aid "Add"))
      ((union a (MlpPn nid lid r wn wg wu wd pid pw)))
      :ruleset fuse :name "mlp-post-norm-add-swapped")
"#
            }
            Rule::GateUp => {
                r#"
(rule ((= s (Node lid (Cons g (Cons u (Nil))))) (K lid "Glu")
       (= g (Lin gk x wg)) (= u (Lin uk x wu)) (!= gk uk) (GluIn wg) (GluIn wu)
       (= key (wkey wg)) (= key (wkey wu)) (= rg (wrows wg)) (= ru (wrows wu)))
      ((let w (Cat2 wg wu))
       (set (wrows w) (+ rg ru)) (set (wkey w) key) (QkvIn w)
       (let f (Lin (+ lid 1099511627776) x w))
       (union g (Slice f 0 rg))
       (union u (Slice f rg (+ rg ru))))
      :ruleset fuse :name "gate-up")
"#
            }
            Rule::Qkv => {
                r#"
(rule ((= q (Lin kq x wq)) (= k (Lin kk x wk)) (< kq kk) (= v (Lin kv x wv)) (< kk kv)
       (QkvIn wq) (QkvIn wk) (QkvIn wv)
       (= key (wkey wq)) (= key (wkey wk)) (= key (wkey wv))
       (= x (Node xid xa))
       (= rq (wrows wq)) (= rk (wrows wk)) (= rv (wrows wv)))
      ((let w (Cat3 wq wk wv))
       (set (wrows w) (+ rq (+ rk rv))) (set (wkey w) key)
       (let f (Lin kq x w))
       (union q (Slice f 0 rq))
       (union k (Slice f rq (+ rq rk)))
       (union v (Slice f (+ rq rk) (+ rq (+ rk rv)))))
      :ruleset fuse :name "qkv")
"#
            }
        }
    }
}

/// Saturation bound: like luminal's `AlgebraBudget`, a cumulative per-rule match cap.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub match_limit: usize,
    pub max_iters: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            match_limit: 4096,
            max_iters: 16,
        }
    }
}

#[derive(Clone)]
struct MatchBudget {
    limit: usize,
    used: HashMap<String, usize>,
}

impl Scheduler for MatchBudget {
    fn filter_matches(&mut self, rule: &str, _: &str, matches: &mut Matches) -> bool {
        let used = self.used.entry(rule.to_owned()).or_default();
        let count = matches.match_size().min(self.limit.saturating_sub(*used));
        for i in 0..count {
            matches.choose(i);
        }
        *used += count;
        *used < self.limit
    }
}

/// Extraction cost of one e-node in µs:
/// `dispatch_us·[launches] + activation bytes / ceiling + Σ weight bytes / rate(format)`.
#[derive(Debug, Clone)]
pub struct CostModel {
    pub ceiling_gbps: f64,
    pub dispatch_us: f64,
    /// Streaming rate of a quantized weight format (`QuantFormat` debug name → GB/s); formats
    /// without an entry stream at the ceiling.
    pub rates: HashMap<String, f64>,
}

impl CostModel {
    pub fn constants() -> Self {
        Self {
            ceiling_gbps: 286.9,
            dispatch_us: 1.2,
            rates: HashMap::new(),
        }
    }

    /// [`Self::constants`] with each format's rate = mean GB/s of its entries in the tune cache
    /// files (`<cache>/host/tune-*.json`, any device), when present.
    pub fn from_tune_cache() -> Self {
        let mut sums: HashMap<String, (f64, f64)> = HashMap::new();
        let dir = crate::hardware::default_cache_dir().map(|d| d.join("host"));
        for e in dir.iter().flat_map(std::fs::read_dir).flatten().flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if !(name.starts_with("tune-") && name.ends_with(".json")) {
                continue;
            }
            let Some(map) = std::fs::read(e.path()).ok().and_then(|b| {
                serde_json::from_slice::<HashMap<String, serde_json::Value>>(&b).ok()
            }) else {
                continue;
            };
            for (k, v) in map {
                let (Some(fmt), Some(gbps)) = (k.split('|').next(), v["gbps"].as_f64()) else {
                    continue;
                };
                let s = sums.entry(fmt.to_owned()).or_default();
                *s = (s.0 + gbps, s.1 + 1.0);
            }
        }
        Self {
            rates: sums.into_iter().map(|(k, (s, n))| (k, s / n)).collect(),
            ..Self::constants()
        }
    }

    fn act_us(&self, bytes: f64) -> f64 {
        bytes / (self.ceiling_gbps * 1e3)
    }

    /// Time to read value `v`: weights at their format's rate, everything else at the ceiling.
    fn read_us(&self, g: &Graph, v: ValueId) -> f64 {
        let val = g.value(v);
        let elems = val.shape.iter().product::<usize>() as f64;
        match (&val.kind, val.dtype) {
            (ValueKind::Weight, DType::Quant(f)) => {
                let rate = self
                    .rates
                    .get(&format!("{f:?}"))
                    .copied()
                    .unwrap_or(self.ceiling_gbps);
                elems * f.bits_per_weight() as f64 / 8.0 / (rate * 1e3)
            }
            _ => self.act_us(elems * dtype_bytes(val.dtype)),
        }
    }
}

fn dtype_bytes(d: DType) -> f64 {
    match d {
        DType::F32 | DType::U32 => 4.0,
        DType::F16 | DType::BF16 => 2.0,
        DType::Quant(f) => f.bits_per_weight() as f64 / 8.0,
    }
}

fn rows(shape: &[usize]) -> usize {
    shape.iter().rev().skip(1).product()
}

/// Per-graph timings and counts of one [`fuse`] run.
#[derive(Debug, Clone, Default)]
pub struct Report {
    pub encode_ms: f64,
    pub saturate_ms: f64,
    pub extract_ms: f64,
    pub decode_ms: f64,
    pub enodes: usize,
    pub groups: usize,
    /// Fusions applied: megakernel MLP, gate/up, QKV.
    pub applied: [usize; 3],
}

impl Report {
    pub fn compile_ms(&self) -> f64 {
        self.encode_ms + self.saturate_ms + self.extract_ms
    }
}

/// Concatenation class of weight `w` for the gate/up and QKV rules (`None`: never fused, as
/// `concat_weight_tensors` / `matches_dtype` would refuse it). Model-loader `Tensor::Mat`s fuse
/// only as QKV (the Metal planner folds the GLU into the up GEMV epilogue `Epi::UpGeGlu/UpSwiGlu`).
fn cat_key(g: &Graph, w: ValueId, weights: Option<&Weights>) -> Option<String> {
    let val = g.value(w);
    if val.kind != ValueKind::Weight || val.shape.len() != 2 {
        return None;
    }
    let cols = val.shape[1];
    if let Some(ws) = weights {
        let t = ws.get(&val.name)?;
        if !fusion::matches_dtype(val.dtype, t) {
            return None;
        }
    }
    Some(format!("{:?}/{cols}", val.dtype))
}

/// Inserts the facts of `g` into `eg` (schema and rules already declared) through the row API:
/// parsing an equivalent program costs several times the saturation on model graphs.
fn encode(eg: &mut EGraph, g: &Graph, weights: Option<&Weights>) -> Result<(), String> {
    let mut uses = vec![0i64; g.values.len()];
    let mut declared = vec![false; g.values.len()];
    for n in &g.nodes {
        if n.outputs.len() != 1 {
            return Err(format!(
                "egraph: node {} has {} outputs",
                n.name,
                n.outputs.len()
            ));
        }
        for &v in &n.inputs {
            if !declared[v.0 as usize] && matches!(g.value(v).kind, ValueKind::Node(_)) {
                return Err(format!(
                    "egraph: node {} reads {} before it is produced",
                    n.name, v.0
                ));
            }
            uses[v.0 as usize] += 1;
        }
        declared[n.outputs[0].0 as usize] = true;
    }
    eg.update(|mut fs| -> Result<(), egglog::Error> {
        let mut seen = vec![false; g.values.len()];
        let mut w_seen = vec![false; g.values.len()];
        for (id, n) in g.nodes.iter().enumerate() {
            let id = id as i64;
            let mut args = Vec::with_capacity(n.inputs.len());
            for &v in &n.inputs {
                let vi = v.0 as usize;
                let c = fs.add("V", vi as i64)?;
                if !seen[vi] {
                    seen[vi] = true;
                    if rows(&g.value(v).shape) == 1 {
                        fs.add("OneRow", c)?;
                    }
                    if megakernel::fusible_weight(g, v) {
                        fs.add("MkW", c)?;
                    }
                }
                args.push(c);
            }
            let mut list = fs.add("Nil", egglog::api::RawValues(Vec::new()))?;
            for &c in args.iter().rev() {
                list = fs.add("Cons", (c, list))?;
            }
            let out = n.outputs[0];
            let o = fs.add("V", out.0 as i64)?;
            seen[out.0 as usize] = true;
            let node = fs.add("Node", (id, list))?;
            fs.union(o, node)?;
            fs.add("K", (id, rule_key(g, n)))?;
            if matches!(n.op, Op::SwiGlu | Op::GeGlu) {
                fs.add("K", (id, "Glu"))?;
            }
            fs.set("Uses", o, uses[out.0 as usize])?;
            if rows(&g.value(out).shape) == 1 {
                fs.add("OneRow", o)?;
            }
            if let (Op::Linear, [_, w]) = (&n.op, &n.inputs[..]) {
                if let Some(key) = cat_key(g, *w, weights) {
                    let wi = w.0 as usize;
                    let wt = fs.add("Wt", wi as i64)?;
                    if !w_seen[wi] {
                        w_seen[wi] = true;
                        fs.set("wrows", wt, g.value(*w).shape[0] as i64)?;
                        fs.set("wkey", wt, key)?;
                        fs.add("QkvIn", wt)?;
                        let mat = weights
                            .and_then(|ws| ws.get(&g.value(*w).name))
                            .is_some_and(|t| matches!(t, Tensor::Mat(_)));
                        // Separate gate/up lets Metal fold GLU into the up GEMV epilogue (`Epi::UpGeGlu/UpSwiGlu`);
                        // fusing them costs an extra activation round trip per layer.
                        if !mat {
                            fs.add("GluIn", wt)?;
                        }
                    }
                    let lin = fs.add("Lin", (id, args[0], wt))?;
                    fs.union(o, lin)?;
                }
            }
        }
        Ok(())
    })
    .map_err(|e| format!("egraph: {e}"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Mlp = 0,
    GateUp = 1,
    Qkv = 2,
}

/// One e-node as extraction sees it.
struct ENode {
    cost: f64,
    /// Non-leaf child classes.
    kids: Vec<Value>,
}

struct Class {
    nodes: Vec<ENode>,
    /// Original node id, or a fused `Lin`'s key.
    key: i64,
}

/// Fusion alternatives that switch together.
struct Group {
    kind: Kind,
    /// `(class, e-node index)`, in fused-output order.
    members: Vec<(Value, usize)>,
    /// Member keys, for the deterministic tie-break.
    keys: Vec<i64>,
    /// Anchor node id (`Mlp`: the Add node).
    anchor: i64,
    /// Class of the fused `Lin` whose slices the members are (gate/up, QKV).
    fused: Option<Value>,
}

struct Extract<'a> {
    classes: &'a HashMap<Value, Class>,
    choice: HashMap<Value, usize>,
    rc: HashMap<Value, u32>,
}

impl Extract<'_> {
    fn chosen(&self, c: Value) -> &ENode {
        &self.classes[&c].nodes[self.choice.get(&c).copied().unwrap_or(0)]
    }

    /// Adds a reference to `c`; returns the cost that became live.
    fn add_ref(&mut self, c: Value) -> f64 {
        let (mut d, mut stack) = (0.0, vec![c]);
        while let Some(c) = stack.pop() {
            let r = self.rc.entry(c).or_default();
            *r += 1;
            if *r == 1 {
                let n = self.chosen(c);
                d += n.cost;
                stack.extend(n.kids.iter().copied());
            }
        }
        d
    }

    /// Drops a reference to `c`; returns the (negative) cost that died.
    fn drop_ref(&mut self, c: Value) -> f64 {
        let (mut d, mut stack) = (0.0, vec![c]);
        while let Some(c) = stack.pop() {
            let r = self.rc.get_mut(&c).expect("refcounted class");
            *r -= 1;
            if *r == 0 {
                let n = self.chosen(c);
                d -= n.cost;
                stack.extend(n.kids.iter().copied());
            }
        }
        d
    }

    fn live(&self, c: Value) -> bool {
        self.rc.get(&c).is_some_and(|&r| r > 0)
    }

    fn switch(&mut self, c: Value, idx: usize) -> f64 {
        let old = self.choice.insert(c, idx).unwrap_or(0);
        if old == idx || !self.live(c) {
            return 0.0;
        }
        let classes = self.classes;
        let (new_n, old_n) = (&classes[&c].nodes[idx], &classes[&c].nodes[old]);
        let mut d = new_n.cost - old_n.cost;
        for &k in &new_n.kids {
            d += self.add_ref(k);
        }
        for &k in &old_n.kids {
            d += self.drop_ref(k);
        }
        d
    }

    fn apply(&mut self, g: &Group, to_base: bool) -> f64 {
        g.members
            .iter()
            .map(|&(c, i)| self.switch(c, if to_base { 0 } else { i }))
            .sum()
    }
}


/// How to materialize one choice.
struct Pick {
    on: bool,
    kind: Kind,
    /// Add output (`Mlp`).
    anchor: Option<ValueId>,
    /// Gate/up and QKV members; `Err(glu)`: a fused gate/up Linear.
    members: Vec<Result<ValueId, ValueId>>,
    /// Original node outputs the fusion removes, which another fusion must not remove first.
    covers: Vec<ValueId>,
    /// Fused gate/up picks a QKV pick reads.
    requires: Vec<usize>,
}

impl Pick {
    fn conflicts(&self, o: &Pick) -> bool {
        self.covers.iter().any(|v| o.covers.contains(v))
    }
}

/// Node outputs a megakernel MLP anchored at Add output `add` replaces (the pass's chain).
fn mlp_cover(g: &Graph, add: ValueId) -> Vec<ValueId> {
    let prod = |v: ValueId| match g.value(v).kind {
        ValueKind::Node(n) => Some(&g.nodes[n.0 as usize]),
        _ => None,
    };
    let mut out = vec![add];
    let Some(a) = prod(add) else { return out };
    for &d in &a.inputs {
        let mut d = d;
        let mut chain = vec![d];
        if let Some(p) = prod(d).filter(|p| matches!(p.op, Op::RmsNorm { .. })) {
            d = p.inputs[0];
            chain.push(d);
        }
        let Some(down) = prod(d).filter(|p| p.op == Op::Linear) else {
            continue;
        };
        let Some(glu) = prod(down.inputs[0]).filter(|p| matches!(p.op, Op::SwiGlu | Op::GeGlu))
        else {
            continue;
        };
        chain.push(down.inputs[0]);
        chain.extend(&glu.inputs);
        if let Some(gate) = prod(glu.inputs[0]) {
            chain.push(gate.inputs[0]);
        }
        out.extend(chain);
        break;
    }
    out
}


/// A saturated and extracted e-graph of one graph: its fusion choices, and the materialization of
/// the cost model's set ([`Fusion::apply`] with no flips) or of any other consistent set.
pub struct Fusion {
    picks: Vec<Pick>,
    report: Report,
}

/// Runs the e-graph fusion on `graph` (after canonical/dtype_adapt), applying the
/// extracted fusions in place and treating `weights` like the passes do.
pub fn fuse(
    graph: &mut Graph,
    weights: FuseWeights<'_>,
    rules: &[Rule],
    cost: &CostModel,
    budget: Budget,
) -> Result<Report, String> {
    Fusion::new(graph, weights.tensors(), rules, cost, budget)?.apply(graph, weights, &[])
}

/// `rule` alone with constant costs (tests of one fusion): how many it applied.
#[cfg(test)]
pub(crate) fn fuse_rule(graph: &mut Graph, weights: FuseWeights<'_>, rule: Rule) -> usize {
    let r = fuse(
        graph,
        weights,
        &[rule],
        &CostModel::constants(),
        Budget::default(),
    );
    r.expect("egraph fusion").applied[rule as usize]
}

impl Fusion {
    /// Encodes, saturates and extracts `g` (`weights` decide which concatenations are possible).
    pub fn new(
        g: &Graph,
        weights: Option<&Weights>,
        rules: &[Rule],
        cost: &CostModel,
        budget: Budget,
    ) -> Result<Self, String> {
        Self::build(g, weights, rules, cost, budget)
    }

    /// Timings of [`Fusion::new`] (encode, saturate, extract) and counts.
    pub fn report(&self) -> &Report {
        &self.report
    }
}

impl Fusion {
    fn build(
        g: &Graph,
        weights: Option<&Weights>,
        rules: &[Rule],
        cost: &CostModel,
        budget: Budget,
    ) -> Result<Self, String> {
        let mut report = Report::default();
        let err = |e: egglog::Error| format!("egraph: {e}");

        let t = Instant::now();
        let mut eg = EGraph::default();
        let mut program = String::from(SCHEMA);
        for r in rules {
            program.push_str(r.text());
        }
        eg.parse_and_run_program(None, &program).map_err(err)?;
        encode(&mut eg, g, weights)?;
        report.encode_ms = t.elapsed().as_secs_f64() * 1e3;

        let t = Instant::now();
        let sid = eg.add_scheduler(Box::new(MatchBudget {
            limit: budget.match_limit,
            used: HashMap::new(),
        }));
        for _ in 0..budget.max_iters {
            if !eg
                .step_rules_with_scheduler(sid, "fuse")
                .map_err(err)?
                .updated
            {
                break;
            }
        }
        eg.remove_scheduler(sid);
        report.saturate_ms = t.elapsed().as_secs_f64() * 1e3;

        let t = Instant::now();
        let rows_of = |name: &str| -> Result<Vec<(Value, Vec<Value>)>, String> {
            let mut out = Vec::new();
            eg.constructor_enodes(name, |en| out.push((en.eclass, en.children.to_vec())))
                .map_err(err)?;
            Ok(out)
        };
        let int = |v: Value| eg.value_to_base::<i64>(v);

        // Lists, weight terms, leaves.
        let mut lists: HashMap<Value, Option<(Value, Value)>> = HashMap::new();
        for (c, ch) in rows_of("Cons")? {
            lists.insert(c, Some((ch[0], ch[1])));
        }
        for (c, _) in rows_of("Nil")? {
            lists.insert(c, None);
        }
        let flatten = |mut l: Value| -> Vec<Value> {
            let mut out = Vec::new();
            while let Some(Some((h, t))) = lists.get(&l) {
                out.push(*h);
                l = *t;
            }
            out
        };
        let mut wterm: HashMap<Value, Result<ValueId, Vec<Value>>> = HashMap::new();
        for (c, ch) in rows_of("Wt")? {
            wterm.insert(c, Ok(ValueId(int(ch[0]) as u32)));
        }
        for name in ["Cat2", "Cat3"] {
            for (c, ch) in rows_of(name)? {
                wterm.insert(c, Err(ch));
            }
        }
        let wleaves = |w: Value| -> Vec<ValueId> {
            let (mut out, mut stack) = (Vec::new(), vec![w]);
            while let Some(w) = stack.pop() {
                match &wterm[&w] {
                    Ok(v) => out.push(*v),
                    Err(ch) => stack.extend(ch.iter().rev()),
                }
            }
            out
        };
        // Graph value of every e-class the encoding named.
        let mut value_of: HashMap<Value, ValueId> = HashMap::new();
        for (c, ch) in rows_of("V")? {
            value_of.insert(c, ValueId(int(ch[0]) as u32));
        }
        let node_rows = rows_of("Node")?;
        let is_leaf = |c: &Value| {
            value_of
                .get(c)
                .is_some_and(|v| !matches!(g.value(*v).kind, ValueKind::Node(_)))
        };
        let kids_of =
            |cs: &[Value]| -> Vec<Value> { cs.iter().filter(|c| !is_leaf(c)).copied().collect() };
        let reads_us =
            |cs: &[Value]| -> f64 { cs.iter().map(|c| cost.read_us(g, value_of[c])).sum() };
        let op_cost = |id: usize, args: &[Value]| -> f64 {
            let n = &g.nodes[id];
            if matches!(
                n.op,
                Op::Slice { .. } | Op::Reshape { .. } | Op::Squeeze { .. } | Op::Unsqueeze { .. }
            ) {
                return 0.0;
            }
            let out = g.value(n.outputs[0]);
            cost.dispatch_us
                + reads_us(args)
                + cost.act_us(out.shape.iter().product::<usize>() as f64 * dtype_bytes(out.dtype))
        };

        let mut classes: HashMap<Value, Class> = HashMap::new();
        for (c, ch) in &node_rows {
            let id = int(ch[0]);
            let args = flatten(ch[1]);
            classes.insert(
                *c,
                Class {
                    nodes: vec![ENode {
                        cost: op_cost(id as usize, &args),
                        kids: kids_of(&args),
                    }],
                    key: id,
                },
            );
        }
        // Fused `Lin`s (over a `Cat`) are classes of their own; `Lin` over `Wt` is a view.
        let mut fused_kind: HashMap<Value, Kind> = HashMap::new();
        for (c, ch) in rows_of("Lin")? {
            let Err(parts) = &wterm[&ch[2]] else { continue };
            let x = ch[1];
            let leaves = wleaves(ch[2]);
            let out_rows: usize = leaves.iter().map(|w| g.value(*w).shape[0]).sum();
            let xv = g.value(value_of[&x]);
            let cost_us = cost.dispatch_us
                + cost.read_us(g, value_of[&x])
                + leaves.iter().map(|w| cost.read_us(g, *w)).sum::<f64>()
                + cost.act_us((rows(&xv.shape) * out_rows * 4) as f64);
            fused_kind.insert(
                c,
                if parts.len() == 3 {
                    Kind::Qkv
                } else {
                    Kind::GateUp
                },
            );
            classes.insert(
                c,
                Class {
                    nodes: vec![ENode {
                        cost: cost_us,
                        kids: kids_of(&[x]),
                    }],
                    key: int(ch[0]),
                },
            );
        }

        // Alternatives and their groups.
        let mut groups: Vec<Group> = Vec::new();
        let mut by_fused: HashMap<Value, usize> = HashMap::new();
        let mut slices = rows_of("Slice")?;
        slices.sort_by_key(|(_, ch)| int(ch[1]));
        for (c, ch) in slices {
            let f = ch[0];
            let class = classes
                .get_mut(&c)
                .ok_or("egraph: Slice outside a known class")?;
            class.nodes.push(ENode {
                cost: 0.0,
                kids: vec![f],
            });
            let (idx, key) = (class.nodes.len() - 1, class.key);
            let gi = *by_fused.entry(f).or_insert_with(|| {
                groups.push(Group {
                    kind: fused_kind[&f],
                    members: Vec::new(),
                    keys: Vec::new(),
                    anchor: -1,
                    fused: Some(f),
                });
                groups.len() - 1
            });
            groups[gi].members.push((c, idx));
            groups[gi].keys.push(key);
        }
        let mut singles =
            |name: &str, kind: Kind, args: &dyn Fn(&[Value]) -> Vec<Value>| -> Result<(), String> {
                for (c, ch) in rows_of(name)? {
                    let a = args(&ch);
                    let class = classes
                        .get_mut(&c)
                        .ok_or("egraph: fused e-node outside a known class")?;
                    let anchor = class.key;
                    let out = g.value(g.nodes[anchor as usize].outputs[0]);
                    let cost_us = cost.dispatch_us
                        + reads_us(&a)
                        + cost.act_us(out.shape.iter().product::<usize>() as f64 * 4.0);
                    class.nodes.push(ENode {
                        cost: cost_us,
                        kids: kids_of(&a),
                    });
                    groups.push(Group {
                        kind,
                        members: vec![(c, class.nodes.len() - 1)],
                        keys: vec![anchor],
                        anchor,
                        fused: None,
                    });
                }
                Ok(())
            };
        singles("Mlp", Kind::Mlp, &|ch| ch[2..7].to_vec())?;
        singles("MlpPn", Kind::Mlp, &|ch| [&ch[2..7], &ch[8..9]].concat())?;
        report.groups = groups.len();
        report.enodes = classes.values().map(|c| c.nodes.len()).sum();

        // Greedy DAG extraction.
        let mut ex = Extract {
            classes: &classes,
            choice: HashMap::new(),
            rc: HashMap::new(),
        };
        let mut uses = vec![0usize; g.values.len()];
        for n in &g.nodes {
            for i in &n.inputs {
                uses[i.0 as usize] += 1;
            }
        }
        for (c, ch) in &node_rows {
            let out = g.nodes[int(ch[0]) as usize].outputs[0];
            if uses[out.0 as usize] == 0 || g.outputs.contains(&out) {
                ex.add_ref(*c);
            }
        }
        let usable = |ex: &Extract, gr: &Group| {
            gr.members
                .iter()
                .all(|&(c, _)| ex.live(c) && ex.choice.get(&c).copied().unwrap_or(0) == 0)
        };
        let mut order: Vec<(i64, usize)> = Vec::with_capacity(groups.len());
        for (i, gr) in groups.iter().enumerate() {
            let d = if usable(&ex, gr) {
                let d = ex.apply(gr, false);
                ex.apply(gr, true);
                d
            } else {
                0.0
            };
            order.push(((d * 1e3).round() as i64, i));
        }
        order.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| groups[a.1].keys.cmp(&groups[b.1].keys))
        });
        let mut committed = vec![false; groups.len()];
        loop {
            let mut changed = false;
            for &(_, i) in &order {
                if committed[i] || !usable(&ex, &groups[i]) {
                    continue;
                }
                let d = ex.apply(&groups[i], false);
                if d < -1e-9 {
                    committed[i] = true;
                    changed = true;
                } else {
                    ex.apply(&groups[i], true);
                }
            }
            if !changed {
                break;
            }
        }
        report.extract_ms = t.elapsed().as_secs_f64() * 1e3;

        // Choices: members as graph values; a fused gate/up `Lin` is named by its GLU.
        let orig_out = |id: i64| g.nodes[id as usize].outputs[0];
        let fused_pick: HashMap<Value, usize> = groups
            .iter()
            .enumerate()
            .filter_map(|(i, gr)| Some((gr.fused?, i)))
            .collect();
        let mut picks = Vec::with_capacity(groups.len());
        for (i, gr) in groups.iter().enumerate() {
            let members: Vec<Result<ValueId, ValueId>> = match gr.kind {
                Kind::GateUp | Kind::Qkv => gr
                    .members
                    .iter()
                    .map(|(c, _)| match value_of.get(c) {
                        Some(v) => Ok(*v),
                        None => Err(orig_out(classes[c].key - FUSED_KEY_BASE)),
                    })
                    .collect(),
                Kind::Mlp => Vec::new(),
            };
            let anchor = matches!(gr.kind, Kind::Mlp).then(|| orig_out(gr.anchor));
            let covers: Vec<ValueId> = match (gr.kind, anchor) {
                (Kind::Mlp, Some(a)) => mlp_cover(g, a),
                _ => members.iter().filter_map(|m| m.ok()).collect(),
            };
            picks.push(Pick {
                on: committed[i],
                kind: gr.kind,
                anchor,
                members,
                covers,
                requires: gr
                    .members
                    .iter()
                    .filter_map(|(c, _)| fused_pick.get(c).copied())
                    .collect(),
            });
        }
        Ok(Self {
            picks,
            report,
        })
    }

    /// Applies the cost model's choices with `flip` toggled to `graph` (the graph [`Fusion::new`]
    /// saw, unchanged), treating `weights` like the passes do. A choice flipped on wins: the
    /// cost model's choices it overlaps are switched off and the gate/up fusions it reads are
    /// switched on; choices left reading a gate/up fusion that is off are switched off. Two
    /// flipped-on choices that overlap are an error. Under [`FuseWeights::ReuseOnly`] a choice
    /// whose fused weight the program does not bind stays unfused, and so does a QKV choice
    /// reading a gate/up fusion that stayed unfused.
    pub fn apply(
        &self,
        graph: &mut Graph,
        mut weights: FuseWeights<'_>,
        flip: &[usize],
    ) -> Result<Report, String> {
        let mut report = self.report.clone();
        let mut on: Vec<bool> = self.picks.iter().map(|p| p.on).collect();
        let mut forced = vec![false; on.len()];
        for &i in flip {
            let o = on
                .get_mut(i)
                .ok_or_else(|| format!("egraph: no choice {i}"))?;
            *o = !*o;
            forced[i] = *o;
        }
        let mut work: Vec<usize> = (0..on.len()).filter(|&i| forced[i]).collect();
        while let Some(i) = work.pop() {
            for &r in &self.picks[i].requires {
                if !on[r] {
                    on[r] = true;
                    forced[r] = true;
                    work.push(r);
                }
            }
            for j in 0..on.len() {
                if j != i && on[j] && self.picks[i].conflicts(&self.picks[j]) {
                    if forced[j] {
                        return Err(format!("egraph: choices {i} and {j} overlap"));
                    }
                    on[j] = false;
                }
            }
        }
        while let Some(j) =
            (0..on.len()).find(|&j| on[j] && self.picks[j].requires.iter().any(|&r| !on[r]))
        {
            on[j] = false;
        }

        // Materialize with the passes' rewriters, in pipeline order.
        let t = Instant::now();
        let chosen = |k: Kind| {
            self.picks
                .iter()
                .zip(&on)
                .filter(move |(p, &c)| c && p.kind == k)
                .map(|(p, _)| p)
        };
        let mlp_adds: Vec<ValueId> = chosen(Kind::Mlp).filter_map(|p| p.anchor).collect();
        let gate_up: Vec<&[Result<ValueId, ValueId>]> =
            chosen(Kind::GateUp).map(|p| &p.members[..]).collect();
        let qkv: Vec<&[Result<ValueId, ValueId>]> =
            chosen(Kind::Qkv).map(|p| &p.members[..]).collect();

        let node_of = |g: &Graph, v: ValueId| match g.value(v).kind {
            ValueKind::Node(n) => Some(n),
            _ => None,
        };
        let resolve = |g: &Graph, m: &Result<ValueId, ValueId>| -> Option<NodeId> {
            match m {
                Ok(v) => node_of(g, *v),
                // GLU input 0 is the gate slice of the fused gate/up Linear (when it was fused).
                Err(glu) => {
                    let slice = g.nodes[node_of(g, *glu)?.0 as usize].inputs[0];
                    let s = &g.nodes[node_of(g, slice)?.0 as usize];
                    matches!(s.op, Op::Slice { .. })
                        .then(|| node_of(g, s.inputs[0]))
                        .flatten()
                }
            }
        };
        let contexts = |g: &Graph,
                        sets: &[&[Result<ValueId, ValueId>]],
                        names: &[&str]|
         -> Vec<MatchContext> {
            let mut ms: Vec<Vec<NodeId>> = sets
                .iter()
                .filter_map(|s| {
                    let mut ns: Vec<NodeId> =
                        s.iter().map(|m| resolve(g, m)).collect::<Option<_>>()?;
                    ns.sort_by_key(|n| n.0);
                    Some(ns)
                })
                .collect();
            ms.sort_by_key(|ns| ns.iter().map(|n| n.0).collect::<Vec<_>>());
            ms.into_iter()
                .map(|ns| {
                    let mut m = MatchContext::default();
                    for (name, n) in names.iter().zip(ns) {
                        m.insert(*name, n);
                    }
                    m
                })
                .collect()
        };
        let only = |g: &Graph, vs: &[ValueId]| -> HashSet<NodeId> {
            vs.iter().filter_map(|v| node_of(g, *v)).collect()
        };
        let want = [mlp_adds.len(), gate_up.len(), qkv.len()];
        report.applied = [0; 3];
        if !mlp_adds.is_empty() {
            let s = only(graph, &mlp_adds);
            report.applied[0] = megakernel::fuse_mlp_blocks(graph, &s);
        }
        let ms = contexts(graph, &gate_up, &["gate", "up"]);
        report.applied[1] = fusion::rewrite_gate_up(graph, &ms, weights.reborrow());
        let ms = contexts(graph, &qkv, &["q", "k", "v"]);
        report.applied[2] = fusion::rewrite_qkv(graph, &ms, weights.reborrow());
        report.decode_ms = t.elapsed().as_secs_f64() * 1e3;
        if report.applied != want && !matches!(weights, FuseWeights::ReuseOnly(_)) {
            return Err(format!(
                "egraph: extracted fusions {want:?} but the rewriters applied {:?}",
                report.applied
            ));
        }
        Ok(report)
    }
}

#[cfg(test)]
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    type Gw = (Graph, Weights);

    fn by_egraph((mut g, mut w): Gw, rules: &[Rule], cost: &CostModel) -> (Gw, Report) {
        crate::graph::passes::prefuse(&mut g);
        let r = fuse(&mut g, FuseWeights::Materialize(&mut w), rules, cost, Budget::default())
            .expect("egraph fusion");
        crate::graph::pattern::validate_ssa(&g).expect("SSA after egraph fusion");
        ((g, w), r)
    }

    /// The default rules with constant costs; `golden` is the [`digest`] of the result, recorded
    /// from the destructive passes these rules replaced (stage 1 asserted equality with them).
    fn check(what: &str, gw: &Gw, golden: u64) -> Report {
        let (got, r) = by_egraph(gw.clone(), DEFAULT_RULES, &CostModel::constants());
        assert_eq!(
            digest(&got),
            golden,
            "{what}: fused graph differs from its golden"
        );
        r
    }

    /// Node order, structure, values and weights of a fused graph (a golden for the fusions).
    fn digest((g, w): &Gw) -> u64 {
        use std::fmt::Write as _;
        let mut s = format!("{:?}|{:?}|{:?}|", g.nodes, g.values, g.outputs);
        let mut names: Vec<&str> = w.iter().map(|(n, _)| n).collect();
        names.sort_unstable();
        for n in names {
            let _ = write!(s, "{n}:{:?};", w.get(n));
        }
        fnv1a(s.as_bytes())
    }

    struct B {
        g: Graph,
        w: Weights,
        seed: f32,
    }

    impl B {
        fn new() -> Self {
            Self {
                g: Graph::new(),
                w: Weights::new(),
                seed: 0.0,
            }
        }
        fn weight(&mut self, name: &str, shape: Vec<usize>, dtype: DType) -> ValueId {
            self.seed += 1.0;
            let (s, n) = (self.seed, shape.iter().product::<usize>());
            let data = (0..n).map(|i| s + i as f32 * 1e-3);
            let t = match dtype {
                DType::BF16 => Tensor::BF16(data.map(|x| (x.to_bits() >> 16) as u16).collect()),
                _ => Tensor::F32(data.collect()),
            };
            self.w.insert(name, t);
            self.g.weight(name, shape, dtype)
        }
        fn lin_t(&mut self, name: &str, x: ValueId, out: usize, dtype: DType) -> ValueId {
            let s = self.g.value(x).shape.clone();
            let w = self.weight(&format!("{name}.w"), vec![out, s[1]], dtype);
            self.g
                .add(name, Op::Linear, &[x, w], vec![s[0], out], DType::F32)
        }
        fn lin(&mut self, name: &str, x: ValueId, out: usize) -> ValueId {
            self.lin_t(name, x, out, DType::F32)
        }
        fn norm(&mut self, name: &str, x: ValueId) -> ValueId {
            let s = self.g.value(x).shape.clone();
            let w = self.weight(&format!("{name}.w"), vec![s[1]], DType::F32);
            self.g
                .add(name, Op::RmsNorm { eps: 1e-5 }, &[x, w], s, DType::F32)
        }
        fn op(&mut self, name: &str, op: Op, ins: &[ValueId], shape: Vec<usize>) -> ValueId {
            self.g.add(name, op, ins, shape, DType::F32)
        }
        fn done(self) -> Gw {
            (self.g, self.w)
        }
    }

    /// Attention block + MLP block over `rows` rows; optional post-MLP norm, swapped Add operands,
    /// GeGLU.
    fn layer(rows: usize, post_norm: bool, swap_add: bool, geglu: bool) -> Gw {
        let mut b = B::new();
        let x = b.g.input("x", vec![rows, 64], DType::F32);
        let n1 = b.norm("n1", x);
        let q = b.lin("q", n1, 64);
        let k = b.lin("k", n1, 32);
        let v = b.lin("v", n1, 32);
        let attn = Op::Attention {
            heads: 2,
            kv_heads: 1,
            causal: true,
            window: None,
            soft_cap: None,
            scale: None,
        };
        let a = b.op("attn", attn, &[q, k, v], vec![rows, 64]);
        let o = b.lin("o", a, 64);
        let r1 = b.op(
            "r1",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[o, x],
            vec![rows, 64],
        );
        let n2 = b.norm("n2", r1);
        let gate = b.lin("gate", n2, 128);
        let up = b.lin("up", n2, 128);
        let glu = if geglu { Op::GeGlu } else { Op::SwiGlu };
        let m = b.op("glu", glu, &[gate, up], vec![rows, 128]);
        let mut d = b.lin("down", m, 64);
        if post_norm {
            d = b.norm("post", d);
        }
        let ins = if swap_add { [d, r1] } else { [r1, d] };
        let r2 = b.op(
            "r2",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &ins,
            vec![rows, 64],
        );
        b.g.output(r2);
        b.done()
    }

    #[test]
    fn megakernel_mlp_and_qkv_fuse() {
        let r = check("layer", &layer(1, false, false, false), 0x8704a1b9129c7244);
        assert_eq!(r.applied, [1, 0, 1]);
        let r = check(
            "post-norm, swapped",
            &layer(1, true, true, false),
            0x23b98b5d9122b1c6,
        );
        assert_eq!(r.applied, [1, 0, 1]);
        let r = check("geglu", &layer(1, false, true, true), 0xac127d60b16a6a85);
        assert_eq!(r.applied, [1, 0, 1]);
    }

    #[test]
    fn gate_up_fuses_when_megakernel_refuses() {
        // Two rows: the megakernel is a one-row chain, so gate/up and QKV fuse instead.
        let r = check(
            "two rows",
            &layer(2, false, false, false),
            0x5d6a683c206fcc55,
        );
        assert_eq!(r.applied, [0, 1, 1]);
    }

    #[test]
    fn qkv_picks_the_first_compatible_triple_among_four_linears() {
        // `b` has another dtype: (a,b,c) and (a,b,d) are rejected and (a,c,d) fuses.
        let mut b = B::new();
        let x = b.g.input("x", vec![1, 64], DType::F32);
        let n = b.norm("n", x);
        let outs = [
            b.lin("a", n, 32),
            b.lin_t("b", n, 32, DType::BF16),
            b.lin("c", n, 16),
            b.lin("d", n, 48),
        ];
        for o in outs {
            b.g.output(o);
        }
        let r = check("four linears", &b.done(), 0x566dfadfc7c6bb6d);
        assert_eq!(r.applied, [0, 0, 1]);

        // Five same-dtype linears: the lowest triple wins.
        let mut b = B::new();
        let x = b.g.input("x", vec![1, 64], DType::F32);
        let n = b.norm("n", x);
        for i in 0..5 {
            let o = b.lin(&format!("l{i}"), n, 16 * (i + 1));
            b.g.output(o);
        }
        let r = check("five linears", &b.done(), 0x951cf9bcebed410d);
        assert_eq!(r.applied, [0, 0, 1]);
    }

    #[test]
    fn parallel_block_fuses_gate_up_then_qkv() {
        // One norm feeds q, k, v, gate and up (parallel attention + MLP), two rows.
        let mut b = B::new();
        let x = b.g.input("x", vec![2, 64], DType::F32);
        let n = b.norm("n", x);
        let q = b.lin("q", n, 64);
        let k = b.lin("k", n, 64);
        let v = b.lin("v", n, 64);
        let gate = b.lin("gate", n, 128);
        let up = b.lin("up", n, 128);
        let m = b.op("glu", Op::SwiGlu, &[gate, up], vec![2, 128]);
        for o in [q, k, v, m] {
            b.g.output(o);
        }
        let r = check("parallel", &b.done(), 0xff392cdb64415b10);
        assert_eq!(r.applied, [0, 1, 1]);
    }


    /// The llama spec's decode instance (one row, last-row head) on the synthetic checkpoint of
    /// [`crate::arch::testing::synthetic_llama`].
    fn llama_decode() -> Gw {
        use crate::arch::expr::Val;
        let dir = tempfile::tempdir().unwrap();
        let a = crate::arch::testing::synthetic_llama(dir.path(), serde_json::json!({}));
        let mut r = crate::arch::testing::Runner::from_artifacts(
            &a,
            Some("llama"),
            crate::hardware::Device::Cpu,
            512,
        )
        .unwrap();
        let vars = [("rows", Val::Num(1.0)), ("head", Val::Str("last".into()))];
        let i = r.instance("text", &vars).unwrap();
        (i.graph, r.model.store.materialize(&i.weights).unwrap())
    }

    #[test]
    fn synthetic_models_match_goldens() {
        let models: Vec<(&str, Gw)> = vec![
            (
                "dense decoder (4 layers)",
                crate::graph::synth_decoder::synthetic_decoder(4, 64, 128, 4, 2, 16),
            ),
            ("llama spec (synthetic, 2 layers)", llama_decode()),
        ];
        let want = [
            ([4, 0, 4], 0x4f2daf5f8f6489de),
            ([2, 0, 2], 2219546655677575810),
        ];
        for ((what, gw), (want, golden)) in models.iter().zip(want) {
            let r = check(what, gw, golden);
            println!("{what}: {} nodes, {r:?}", gw.0.nodes.len());
            assert_eq!(r.applied, want, "{what}: fusions applied");
        }
    }

    /// Real checkpoints, CPU only. `GIBBS_EGRAPH_MODELS` = `:`-separated GGUF paths (default: the
    /// suite's Llama-3.2-1B Q4_0 in the HF cache). Run with `--ignored --nocapture`.
    #[test]
    #[ignore = "loads real checkpoints (GIBBS_EGRAPH_MODELS)"]
    fn real_models_fuse() {
        let hub =
            std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/huggingface/hub");
        let paths: Vec<std::path::PathBuf> = match std::env::var("GIBBS_EGRAPH_MODELS") {
            Ok(s) => s.split(':').map(Into::into).collect(),
            Err(_) => vec![hub.join(
                "models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/main/Llama-3.2-1B-Instruct-Q4_0.gguf",
            )],
        };
        for p in paths {
            if !p.exists() {
                println!("SKIP {}: not found", p.display());
                continue;
            }
            let arch = crate::model::loaders::gguf::architecture(&p).unwrap_or_default();
            // The detected family spec's decode instance (one row, last-row head).
            let gw: Gw = {
                use crate::arch::expr::Val;
                let a = crate::model::hub::ModelArtifacts {
                    model_id: "egraph/real".into(),
                    weights_path: Some(p.clone()),
                    ..Default::default()
                };
                let mut r = crate::arch::testing::Runner::from_artifacts(
                    &a,
                    None,
                    crate::hardware::Device::Cpu,
                    4096,
                )
                .unwrap();
                let vars = [("rows", Val::Num(1.0)), ("head", Val::Str("last".into()))];
                let i = r.instance("text", &vars).unwrap();
                (i.graph, r.model.store.materialize(&i.weights).unwrap())
            };
            let nodes = gw.0.nodes.len();
            // The shipped configuration: rates from the tune cache when present.
            let (_, r) = by_egraph(gw, DEFAULT_RULES, &CostModel::from_tune_cache());
            println!(
                "{} ({arch}): {nodes} nodes, applied {:?}; egraph compile {:.1} ms \
                 (encode {:.1} + saturate {:.1} + extract {:.1}), decode {:.1} ms; {r:?}",
                p.file_name().unwrap().to_string_lossy(),
                r.applied,
                r.compile_ms(),
                r.encode_ms,
                r.saturate_ms,
                r.extract_ms,
                r.decode_ms,
            );
        }
    }
}
