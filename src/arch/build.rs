//! Spec interpreter: runs a component's body with instance variables bound and emits a
//! `graph::Graph`. Node kinds (one JSON object each, see the README "Model specs"):
//!
//! - op:     `{op, args, in, out, name?, when?}`; `args` deserialize into `graph::Op`.
//! - state:  `{state, shape, dtype?, recurrent?, when?}` declares persistent state.
//! - use:    `{use, args?, in?, out?, when?}` expands a macro in a fresh value scope.
//! - repeat: `{repeat: var, count | over, body}` (shares the enclosing value scope).
//! - block:  `{block: [...], when}` conditional group (shares the enclosing value scope).
//! - let:    `{let: {var: expr, ...}}` binds expression variables in the current frame.
//! - tap:    `{tap: name, in}` an extra output when the instance's `taps` list names it (`"*"`: all).
//!
//! Operand refs: `x` (scope value), `v?:x` (scope value, dropped when unbound), `w:name` (weight),
//! `w?:name` (weight, dropped when absent), `s:name` (state), `c:name` (constant), `i:name` (input).
//! Every name interpolates `{expr}`.

use super::expr::{eval_json, eval_str, interp, walk, Env, Val};
use super::{InputDecl, Lora, Network, Source, Spec};
use crate::graph::{DType, Graph, Op, ValueId, ValueKind};
use serde_json::{Map, Value as J};
use std::collections::{HashMap, HashSet};

/// Name suffix of a weight's projection copy (see `Builder::projection`).
pub const LINEAR_COPY: &str = "#linear";

/// One graph input of an instance, in declaration (= binding) order.
#[derive(Debug, Clone)]
pub struct InputPlan {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// Generator (and evaluated args) the runner fills it with; `None` = caller supplies it.
    pub gen: Option<(String, J)>,
}

/// A built instance of a component.
#[derive(Debug, Clone)]
pub struct Instance {
    pub graph: Graph,
    pub inputs: Vec<InputPlan>,
    /// Output names, graph output order (declared outputs, then requested taps).
    pub outputs: Vec<String>,
    /// Weights and constants the graph references (bind these).
    pub weights: Vec<String>,
}

impl Instance {
    pub fn input_index(&self, name: &str) -> Option<usize> {
        self.inputs.iter().position(|i| i.name == name)
    }
    pub fn output_index(&self, name: &str) -> Option<usize> {
        self.outputs.iter().position(|o| o == name)
    }
}

pub fn parse_dtype(s: &str) -> Result<DType, String> {
    Ok(match s {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "u32" => DType::U32,
        other => return Err(format!("unknown dtype `{other}`")),
    })
}

/// Builds `component` of `spec` with instance `vars`. `model` resolves config vars and scopes
/// (`cfg`, `gguf`, `json`, `runtime`) and tensor existence; `src` gives weight shapes/dtypes.
pub fn build(
    spec: &Spec,
    component: &str,
    model: &dyn Env,
    src: &dyn Source,
    vars: &[(&str, Val)],
) -> Result<Instance, String> {
    let net = match spec.component(component)? {
        super::Component::Net(n) => n,
        super::Component::Ref { family, .. } => {
            return Err(format!(
                "{}: component `{component}` is family `{family}`; build it from that spec",
                spec.family
            ))
        }
    };
    build_network(spec, &net, model, src, vars)
        .map_err(|e| format!("{}/{component}: {e}", spec.family))
}

pub fn build_network(
    spec: &Spec,
    net: &Network,
    model: &dyn Env,
    src: &dyn Source,
    vars: &[(&str, Val)],
) -> Result<Instance, String> {
    for v in &net.vars {
        if !vars.iter().any(|(k, _)| k == v) && *v != "taps" {
            return Err(format!("instance variable `{v}` not given"));
        }
    }
    let mut b = Builder {
        spec,
        model,
        src,
        g: Graph::new(),
        frames: vec![vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()],
        values: vec![HashMap::new()],
        weights: HashMap::new(),
        weight_order: Vec::new(),
        gathered: HashSet::new(),
        states: HashMap::new(),
        inputs: HashMap::new(),
        taps: HashMap::new(),
        prefix: Vec::new(),
        depth: 0,
    };
    let mut plans = Vec::new();
    for d in &net.inputs {
        if let Some(p) = b.input(d)? {
            plans.push(p);
        }
    }
    b.body(&net.body)?;
    let mut outputs = Vec::new();
    for o in &net.outputs {
        let (name, value, when) = match o {
            J::String(s) => (s.clone(), None, None),
            J::Object(m) => (
                m.get("name")
                    .and_then(J::as_str)
                    .ok_or("output object needs `name`")?
                    .to_string(),
                m.get("value").and_then(J::as_str),
                m.get("when").cloned(),
            ),
            _ => return Err(format!("bad output {o}")),
        };
        if let Some(w) = when {
            if !b.eval(&w)?.truthy() {
                continue;
            }
        }
        // `{repeat: var, count: expr, name: "x.{var}"}`: one output per index.
        let rep = match o.get("repeat").and_then(J::as_str) {
            Some(var) => {
                let n = b
                    .eval(o.get("count").ok_or("output `repeat` needs `count`")?)?
                    .usize()?;
                Some((var.to_string(), n))
            }
            None => None,
        };
        let idx: Vec<Option<usize>> = match &rep {
            Some((_, n)) => (0..*n).map(Some).collect(),
            None => vec![None],
        };
        for i in idx {
            if let (Some((var, _)), Some(i)) = (&rep, i) {
                b.frames.push(vec![(var.clone(), Val::Num(i as f64))]);
            }
            let r = (|| {
                let name = b.interp(&name)?;
                let v = b.value(&match value {
                    Some(s) => b.interp(s)?,
                    None => name.clone(),
                })?;
                Ok::<_, String>((name, v))
            })();
            if rep.is_some() {
                b.frames.pop();
            }
            let (name, v) = r?;
            b.g.output(v);
            outputs.push(name);
        }
    }
    let want: Vec<String> = match b.var("taps") {
        Some(Val::List(l)) => l.iter().map(|t| t.to_string()).collect(),
        // Every emitted tap, by name (the `precompute` component's weights).
        Some(Val::Str(s)) if s == "*" => {
            let mut t: Vec<String> = b.taps.keys().cloned().collect();
            t.sort();
            t
        }
        _ => Vec::new(),
    };
    for t in want {
        let v = *b
            .taps
            .get(&t)
            .ok_or_else(|| format!("tap `{t}` is not emitted by this component"))?;
        b.g.output(v);
        outputs.push(t);
    }
    Ok(Instance {
        graph: b.g,
        inputs: plans,
        outputs,
        weights: b.weight_order,
    })
}

struct Builder<'a> {
    spec: &'a Spec,
    model: &'a dyn Env,
    src: &'a dyn Source,
    g: Graph,
    /// Expression variable frames, innermost last.
    frames: Vec<Vec<(String, Val)>>,
    /// Value scopes; lookups see only the innermost (macros are isolated).
    values: Vec<HashMap<String, ValueId>>,
    weights: HashMap<String, ValueId>,
    weight_order: Vec<String>,
    /// Values some non-`Linear` op reads.
    gathered: HashSet<ValueId>,
    states: HashMap<String, ValueId>,
    inputs: HashMap<String, ValueId>,
    taps: HashMap<String, ValueId>,
    /// Node-name prefixes of the enclosing macro calls.
    prefix: Vec<String>,
    depth: usize,
}

/// The builder's expression environment: frames, then the model env.
struct BEnv<'b, 'a>(&'b Builder<'a>);

impl Env for BEnv<'_, '_> {
    fn lookup(&self, root: &str, path: &[Val]) -> Option<Val> {
        match self.0.var(root) {
            Some(v) => walk(v, path),
            None => self.0.model.lookup(root, path),
        }
    }
    fn has(&self, t: &str) -> bool {
        self.0.src.tensor(t).is_some() || self.0.model.has(t)
    }
    fn shape(&self, t: &str) -> Option<Vec<usize>> {
        if let Some(v) = self.0.values.last().unwrap().get(t) {
            return Some(self.0.g.value(*v).shape.clone());
        }
        self.0
            .src
            .tensor(t)
            .map(|(s, _)| s)
            .or_else(|| self.0.model.shape(t))
    }
    fn block(&self, t: &str) -> Option<usize> {
        self.0.model.block(t)
    }
    fn raw_has(&self, p: &str) -> bool {
        self.0.model.raw_has(p)
    }
    fn values(&self, t: &str) -> Option<Vec<f32>> {
        self.0.model.values(t)
    }
    fn dtype(&self, t: &str) -> Option<&'static str> {
        self.0.model.dtype(t)
    }
    fn token(&self, s: &str) -> Option<u32> {
        self.0.model.token(s)
    }
}

impl<'a> Builder<'a> {
    fn var(&self, name: &str) -> Option<Val> {
        self.frames.iter().rev().find_map(|f| {
            f.iter()
                .rev()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.clone())
        })
    }
    fn eval(&self, j: &J) -> Result<Val, String> {
        eval_json(j, &BEnv(self))
    }
    fn eval_s(&self, s: &str) -> Result<Val, String> {
        eval_str(s, &BEnv(self))
    }
    fn interp(&self, s: &str) -> Result<String, String> {
        interp(s, &BEnv(self))
    }
    fn when(&self, node: &Map<String, J>) -> Result<bool, String> {
        match node.get("when") {
            None => Ok(true),
            Some(J::String(s)) => Ok(self.eval_s(s)?.truthy()),
            Some(w) => Ok(self.eval(w)?.truthy()),
        }
    }
    fn shape_of(&self, dims: &[J]) -> Result<Vec<usize>, String> {
        dims.iter().map(|d| self.eval(d)?.usize()).collect()
    }

    fn input(&mut self, d: &InputDecl) -> Result<Option<InputPlan>, String> {
        if let Some(w) = &d.when {
            if !self.eval_s(w)?.truthy() {
                return Ok(None);
            }
        }
        let name = self.interp(&d.name)?;
        let shape = self.shape_of(&d.shape)?;
        let dtype = parse_dtype(&d.dtype)?;
        let v = self.g.input(name.clone(), shape.clone(), dtype);
        self.inputs.insert(name.clone(), v);
        let gen = match &d.gen {
            Some(gname) => Some((gname.clone(), self.gen_args(&d.args)?)),
            None => None,
        };
        Ok(Some(InputPlan {
            name,
            shape,
            dtype,
            gen,
        }))
    }

    /// Evaluated gen-input args; strings `"step:<expr>"` stay verbatim for the runner to evaluate
    /// per run (`runtime::diffusion`).
    fn gen_args(&self, j: &J) -> Result<J, String> {
        Ok(match j {
            J::String(s) if s.starts_with("step:") => j.clone(),
            J::Object(o) => J::Object(
                o.iter()
                    .map(|(k, v)| Ok((k.clone(), self.gen_args(v)?)))
                    .collect::<Result<_, String>>()?,
            ),
            _ => self.eval(j)?.to_json(),
        })
    }

    fn body(&mut self, nodes: &[J]) -> Result<(), String> {
        for n in nodes {
            let m = n
                .as_object()
                .ok_or_else(|| format!("node must be an object: {n}"))?;
            self.node(m).map_err(|e| {
                let mut what = ["op", "use", "state", "repeat", "tap", "block", "let"]
                    .iter()
                    .find_map(|k| m.get(*k).map(|v| format!("{k} {v}")))
                    .unwrap_or_default();
                if what.len() > 60 {
                    what = format!("{}…", &what[..what.floor_char_boundary(60)]);
                }
                let out = m.get("out").map(|o| format!(" -> {o}")).unwrap_or_default();
                format!("{e}\n  at {what}{out}")
            })?;
        }
        Ok(())
    }

    fn node(&mut self, m: &Map<String, J>) -> Result<(), String> {
        if let Some(lets) = m.get("let") {
            let lets = lets.as_object().ok_or("`let` takes an object")?;
            for (k, e) in lets {
                let v = self.eval(e)?;
                self.frames.last_mut().unwrap().push((k.clone(), v));
            }
            return Ok(());
        }
        if !self.when(m)? {
            return Ok(());
        }
        if let Some(op) = m.get("op") {
            return self.op(op.as_str().ok_or("`op` must be a string")?, m);
        }
        if let Some(st) = m.get("state") {
            let name = self.interp(st.as_str().ok_or("`state` must be a string")?)?;
            let shape = self.shape_of(
                m.get("shape")
                    .and_then(J::as_array)
                    .ok_or("state needs `shape`")?,
            )?;
            let dtype = parse_dtype(m.get("dtype").and_then(J::as_str).unwrap_or("f32"))?;
            if self.states.contains_key(&name) {
                return Err(format!("state `{name}` declared twice"));
            }
            let recurrent = m
                .get("recurrent")
                .map(|r| self.eval(r))
                .transpose()?
                .is_some_and(|r| r.truthy());
            let v = self.g.state(name.clone(), shape, dtype, recurrent);
            self.states.insert(name, v);
            return Ok(());
        }
        if let Some(var) = m.get("repeat") {
            let var = var.as_str().ok_or("`repeat` names a variable")?.to_string();
            let items: Vec<Val> = match (m.get("count"), m.get("over")) {
                (Some(c), None) => (0..self.eval(c)?.usize()?)
                    .map(|i| Val::Num(i as f64))
                    .collect(),
                (None, Some(o)) => self.eval(o)?.list()?.to_vec(),
                _ => return Err("repeat needs exactly one of `count`, `over`".into()),
            };
            let body = m
                .get("body")
                .and_then(J::as_array)
                .ok_or("repeat needs `body`")?;
            for it in items {
                self.frames.push(vec![(var.clone(), it)]);
                let r = self.body(body);
                self.frames.pop();
                r?;
            }
            return Ok(());
        }
        if let Some(b) = m.get("block") {
            let body = b.as_array().ok_or("`block` takes a node list")?;
            self.frames.push(Vec::new());
            let r = self.body(body);
            self.frames.pop();
            return r;
        }
        if let Some(t) = m.get("tap") {
            let name = self.interp(t.as_str().ok_or("`tap` must be a string")?)?;
            let src = m.get("in").and_then(J::as_str).ok_or("tap needs `in`")?;
            let v = self.value(&self.interp(src)?)?;
            self.taps.insert(name, v);
            return Ok(());
        }
        if let Some(u) = m.get("use") {
            return self.use_macro(u.as_str().ok_or("`use` must be a string")?, m);
        }
        Err(format!("unknown node kind: {}", J::Object(m.clone())))
    }

    /// A value in the current scope.
    fn value(&self, name: &str) -> Result<ValueId, String> {
        self.values
            .last()
            .unwrap()
            .get(name)
            .copied()
            .ok_or_else(|| format!("`{name}` is not bound"))
    }

    fn bind(&mut self, name: String, v: ValueId) {
        self.values.last_mut().unwrap().insert(name, v);
    }

    /// Resolves an operand ref; `None` when an optional ref is absent.
    fn operand(&mut self, r: &str) -> Result<Option<ValueId>, String> {
        let (kind, name) = match r.split_once(':') {
            Some((k @ ("w" | "w?" | "s" | "c" | "i" | "v?"), n)) => (k, n),
            _ => ("", r),
        };
        let name = self.interp(name)?;
        match kind {
            "" => self.value(&name).map(Some),
            "v?" => Ok(self.values.last().unwrap().get(&name).copied()),
            "w" | "c" | "w?" => match self.weight(&name) {
                Some(v) => Ok(Some(v)),
                None if kind == "w?" => Ok(None),
                None => Err(format!(
                    "{} `{name}` is missing",
                    if kind == "c" { "constant" } else { "weight" }
                )),
            },
            "s" => self
                .states
                .get(&name)
                .copied()
                .map(Some)
                .ok_or_else(|| format!("state `{name}` is not declared")),
            "i" => self
                .inputs
                .get(&name)
                .copied()
                .map(Some)
                .ok_or_else(|| format!("input `{name}` is not declared")),
            _ => unreachable!(),
        }
    }

    fn weight(&mut self, name: &str) -> Option<ValueId> {
        if let Some(v) = self.weights.get(name) {
            return Some(*v);
        }
        let (shape, dtype) = self.src.tensor(name)?;
        let v = self.g.weight(name.to_string(), shape, dtype);
        self.weights.insert(name.to_string(), v);
        self.weight_order.push(name.to_string());
        Some(v)
    }

    /// The weight a `Linear` reads for `w`: `w` itself, or when another op (a tied embedding
    /// gather) already reads `w`, its projection copy `<name>#linear`, which the loader may
    /// re-encode for streaming while the gather keeps the table's own encoding.
    // ponytail: only a gather built before the Linear is seen (embedding first, tied head last).
    fn projection(&mut self, w: ValueId) -> ValueId {
        let name = self.g.value(w).name.clone();
        if !self.gathered.contains(&w) || self.g.value(w).kind != ValueKind::Weight {
            return w;
        }
        let copy = format!("{name}{LINEAR_COPY}");
        if self
            .src
            .tensor(&copy)
            .is_none_or(|(_, dt)| dt == self.g.value(w).dtype)
        {
            return w;
        }
        self.weight(&copy).unwrap_or(w)
    }

    fn node_name(&self, m: &Map<String, J>, out: &str) -> Result<String, String> {
        let base = match m.get("name").and_then(J::as_str) {
            Some(n) => self.interp(n)?,
            None => out.to_string(),
        };
        Ok(match self.prefix.last().filter(|p| !p.is_empty()) {
            Some(p) => format!("{p}.{base}"),
            None => base,
        })
    }

    fn op(&mut self, op: &str, m: &Map<String, J>) -> Result<(), String> {
        let args = match m.get("args") {
            Some(a) => self.eval(a)?.to_json(),
            None => J::Object(Map::new()),
        };
        let parsed = parse_op(op, args)?;
        let mut ins = Vec::new();
        for r in m
            .get("in")
            .and_then(J::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            if let Some(v) = self.operand(r.as_str().ok_or("operand refs are strings")?)? {
                ins.push(v);
            }
        }
        if let (Op::Linear, Some(w)) = (&parsed, ins.get_mut(1)) {
            *w = self.projection(*w);
        } else {
            self.gathered.extend(ins.iter().copied());
        }
        let out = self.interp(m.get("out").and_then(J::as_str).ok_or("op needs `out`")?)?;
        let name = self.node_name(m, &out)?;
        let lora = match (&parsed, ins.get(1)) {
            (Op::Linear, Some(&w)) => self.src.lora(&self.g.value(w).name),
            _ => None,
        };
        let mut v = self.g.push(name.clone(), parsed, &ins)?;
        if let Some(l) = lora {
            v = self.lora_branch(&name, ins[0], v, &l)?;
        }
        self.bind(out, v);
        Ok(())
    }

    /// `y + (x·Aᵀ·Bᵀ)·scale (+ bias delta)`.
    fn lora_branch(
        &mut self,
        name: &str,
        x: ValueId,
        y: ValueId,
        l: &Lora,
    ) -> Result<ValueId, String> {
        use crate::graph::{BinaryKind, UnaryKind};
        let a = self
            .weight(&l.a)
            .ok_or_else(|| format!("LoRA weight `{}` missing", l.a))?;
        let b = self
            .weight(&l.b)
            .ok_or_else(|| format!("LoRA weight `{}` missing", l.b))?;
        let t = self.g.push(format!("{name}.lora_a"), Op::Linear, &[x, a])?;
        let u = self.g.push(format!("{name}.lora_b"), Op::Linear, &[t, b])?;
        let s = self.g.push(
            format!("{name}.lora_scale"),
            Op::Unary {
                kind: UnaryKind::Scale(l.scale),
            },
            &[u],
        )?;
        let mut out = self.g.push(
            format!("{name}.lora_add"),
            Op::Binary {
                kind: BinaryKind::Add,
            },
            &[y, s],
        )?;
        if let Some(bd) = &l.bias_delta {
            let d = self
                .weight(bd)
                .ok_or_else(|| format!("LoRA bias `{bd}` missing"))?;
            out = self.g.push(
                format!("{name}.lora_bias"),
                Op::Binary {
                    kind: BinaryKind::Add,
                },
                &[out, d],
            )?;
        }
        Ok(out)
    }

    fn use_macro(&mut self, name: &str, m: &Map<String, J>) -> Result<(), String> {
        let name = self.interp(name)?;
        let mac = self.spec.macro_def(&name)?;
        self.depth += 1;
        if self.depth > 64 {
            return Err(format!("macro `{name}`: recursion too deep"));
        }
        // Operands resolve in the caller's scope.
        let mut params = HashMap::new();
        let ins = m
            .get("in")
            .and_then(J::as_array)
            .cloned()
            .unwrap_or_default();
        if ins.len() > mac.params.len() {
            return Err(format!(
                "macro `{name}` takes {} operands, got {}",
                mac.params.len(),
                ins.len()
            ));
        }
        for (p, r) in mac.params.iter().zip(&ins) {
            if let Some(v) = self.operand(r.as_str().ok_or("operand refs are strings")?)? {
                params.insert(p.clone(), v);
            }
        }
        let mut frame = Vec::new();
        if let Some(a) = m.get("args") {
            for (k, e) in a.as_object().ok_or("`args` must be an object")? {
                frame.push((k.clone(), self.eval(e)?));
            }
        }
        let out = m
            .get("out")
            .map(|o| self.eval_out(o))
            .transpose()?
            .unwrap_or_default();
        let call_name = match m.get("name").and_then(J::as_str) {
            Some(n) => self.interp(n)?,
            None => out.first().cloned().unwrap_or_else(|| name.clone()),
        };
        let pfx = match (
            self.prefix.last().filter(|p| !p.is_empty()),
            call_name.is_empty(),
        ) {
            (Some(p), false) => format!("{p}.{call_name}"),
            (Some(p), true) => p.clone(),
            (None, false) => call_name,
            (None, true) => String::new(),
        };
        self.frames.push(frame);
        self.values.push(params);
        self.prefix.push(pfx);
        let r = self.body(&mac.body);
        let returns: Vec<String> = match &mac.returns {
            J::Null => vec!["out".into()],
            J::String(s) => vec![s.clone()],
            J::Array(a) => a
                .iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect(),
            other => return Err(format!("macro `{name}`: bad `returns` {other}")),
        };
        let ret: Result<Vec<(bool, Option<ValueId>)>, String> = r.and_then(|_| {
            returns
                .iter()
                .map(|n| {
                    let (opt, n) = match n.strip_prefix("v?:") {
                        Some(rest) => (true, rest),
                        None => (false, n.as_str()),
                    };
                    let n = self.interp(n)?;
                    let v = n
                        .split('|')
                        .find_map(|k| self.values.last().unwrap().get(k).copied());
                    Ok((opt, v))
                })
                .collect()
        });
        self.prefix.pop();
        self.values.pop();
        self.frames.pop();
        self.depth -= 1;
        let ret = ret.map_err(|e| format!("{e}\n  in macro `{name}`"))?;
        if out.len() > ret.len() {
            return Err(format!(
                "macro `{name}` returns {} values, {} requested",
                ret.len(),
                out.len()
            ));
        }
        for (o, (opt, v)) in out.into_iter().zip(ret) {
            match v {
                Some(v) => self.bind(o, v),
                None if opt => {}
                None => {
                    return Err(format!(
                        "macro `{name}` did not bind its return value for `{o}`"
                    ))
                }
            }
        }
        Ok(())
    }

    fn eval_out(&self, o: &J) -> Result<Vec<String>, String> {
        match o {
            J::String(s) => Ok(vec![self.interp(s)?]),
            J::Array(a) => a
                .iter()
                .map(|x| self.interp(x.as_str().ok_or("out names are strings")?))
                .collect(),
            _ => Err(format!("bad `out` {o}")),
        }
    }
}

/// `graph::Op` from a variant name and its evaluated fields.
pub fn parse_op(op: &str, args: J) -> Result<Op, String> {
    let empty = args.as_object().is_some_and(Map::is_empty);
    if empty {
        if let Ok(o) = serde_json::from_value::<Op>(J::String(op.to_string())) {
            return Ok(o);
        }
    }
    let mut m = Map::new();
    m.insert(op.to_string(), args);
    serde_json::from_value::<Op>(J::Object(m)).map_err(|e| format!("op `{op}`: {e}"))
}

/// One cached graph program per component, run for every instance shape: the first instance
/// compiles; others reuse its weights/session via `GraphProgram::run_reshaped`. Built instances
/// are kept per var tuple (8, least recently used evicted).
pub struct ShapeCache {
    pub component: String,
    entries: parking_lot::Mutex<Vec<(String, std::sync::Arc<Instance>)>>,
}

impl ShapeCache {
    pub const CAP: usize = crate::graph::program::INSTANCE_SHAPES;

    pub fn new(component: &str) -> Self {
        Self {
            component: component.to_string(),
            entries: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// Key of an instance var tuple.
    pub fn key(vars: &[(&str, Val)]) -> String {
        vars.iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// The instance for `vars`, building it with `make` on a miss.
    pub fn get(
        &self,
        vars: &[(&str, Val)],
        make: impl FnOnce() -> Result<Instance, String>,
    ) -> Result<std::sync::Arc<Instance>, String> {
        let key = Self::key(vars);
        let mut e = self.entries.lock();
        if let Some(i) = e.iter().position(|(k, _)| *k == key) {
            let hit = e.remove(i);
            let inst = hit.1.clone();
            e.push(hit);
            return Ok(inst);
        }
        drop(e);
        let inst = std::sync::Arc::new(make()?);
        let mut e = self.entries.lock();
        if e.len() >= Self::CAP {
            e.remove(0);
        }
        e.push((key, inst.clone()));
        Ok(inst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{Lora, Registry, Signals};
    use crate::graph::DType;

    /// Weights: `emb [16, 8]`, per layer `l{i}.w [8, 8]`, `norm [8]`; LoRA on `l0.w`.
    struct Src;
    impl Source for Src {
        fn tensor(&self, name: &str) -> Option<(Vec<usize>, DType)> {
            match name {
                "emb" => Some((vec![16, 8], DType::F32)),
                "norm" => Some((vec![8], DType::F32)),
                "l0.w" | "l1.w" | "l2.w" => Some((vec![8, 8], DType::F32)),
                "lora.a" => Some((vec![2, 8], DType::F32)),
                "lora.b" => Some((vec![8, 2], DType::F32)),
                _ => None,
            }
        }
        fn lora(&self, w: &str) -> Option<Lora> {
            (w == "l0.w").then(|| Lora {
                a: "lora.a".into(),
                b: "lora.b".into(),
                scale: 0.5,
                bias_delta: None,
            })
        }
    }
    struct Cfg;
    impl Env for Cfg {
        fn lookup(&self, root: &str, path: &[Val]) -> Option<Val> {
            let v = match root {
                "layers" => Val::Num(3.0),
                "eps" => Val::Num(1e-5),
                "cfg" => Val::from_json(&serde_json::json!({"skip": [1]})),
                _ => return None,
            };
            walk(v, path)
        }
    }

    const SPEC: &str = r#"{
      family: "toy", pipeline: "causal_lm",
      macros: {
        norm_lin: {params: ["x"], returns: "y", body: [
          {op: "RmsNorm", args: {eps: "eps"}, in: ["x", "w:norm"], out: "n"},
          {op: "Linear", in: ["n", "w:{w}", "w?:{w}.bias"], out: "y"},
        ]},
      },
      components: {text: {
        vars: ["rows"],
        inputs: [
          {name: "ids", dtype: "u32", shape: ["rows"]},
          {name: "mask", dtype: "f32", shape: ["rows", "rows"], when: "rows > 1", gen: "causal_key_mask", args: {n: "rows"}},
        ],
        body: [
          {op: "Embedding", in: ["w:emb", "i:ids"], out: "x"},
          {state: "kv", shape: [4, 8], recurrent: true},
          {repeat: "i", count: "layers", body: [
            {block: [
              {let: {wname: "fmt('l{i}.w')"}},
              {use: "norm_lin", args: {w: "wname"}, in: ["x"], out: "h", name: "layer{i}"},
              {op: "Binary", args: {kind: "Add"}, in: ["x", "h"], out: "x"},
              {tap: "layer.{i}", in: "x"},
            ], when: "!contains(cfg.skip, i)"},
          ]},
          {repeat: "j", over: "[0, 2]", body: [
            {op: "Unary", args: {kind: {Scale: 2.0}}, in: ["x"], out: "x", name: "scale{j}"},
          ]},
          {op: "Linear", in: ["x", "w:emb"], out: "logits"},
        ],
        outputs: ["logits", {name: "x", when: "rows == 99"}, {name: "rep.{k}", value: "logits", repeat: "k", count: "layers - 1", when: "rows == 3"}],
      }},
    }"#;

    fn spec() -> Spec {
        let r = Registry::from_files(&[("families/toy.json5".into(), SPEC.into())], &[]).unwrap();
        r.spec("toy", &Signals::default()).unwrap()
    }

    #[test]
    fn every_node_kind() {
        let taps = Val::List(vec![Val::Str("layer.2".into())]);
        let inst = build(
            &spec(),
            "text",
            &Cfg,
            &Src,
            &[("rows", Val::Num(4.0)), ("taps", taps)],
        )
        .unwrap();
        let g = &inst.graph;
        // Inputs: ids, then the generated mask (rows > 1).
        assert_eq!(inst.inputs.len(), 2);
        assert_eq!(inst.inputs[1].gen.as_ref().unwrap().0, "causal_key_mask");
        assert_eq!(inst.inputs[1].gen.as_ref().unwrap().1["n"], 4);
        // Layer 1 is skipped by the block's `when`; layer 0 has a LoRA branch.
        let names: Vec<&str> = g.nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(
            names.contains(&"layer0.n")
                && names.contains(&"layer2.y")
                && !names.contains(&"layer1.y")
        );
        assert!(names.contains(&"layer0.y.lora_add"));
        assert!(names.contains(&"scale0") && names.contains(&"scale2"));
        // Outputs: logits [4, 16] then the requested tap; `x` output's `when` is false.
        assert_eq!(
            inst.outputs,
            vec!["logits".to_string(), "layer.2".to_string()]
        );
        assert_eq!(g.value(g.outputs[0]).shape, vec![4, 16]);
        assert_eq!(g.recurrent_states(), vec!["kv".to_string()]);
        // Weights are declared once each: emb is used twice.
        let mut w = inst.weights.clone();
        w.sort();
        assert_eq!(w, ["emb", "l0.w", "l2.w", "lora.a", "lora.b", "norm"]);
        let decls = g
            .values
            .iter()
            .filter(|v| v.name == "emb" && v.kind == crate::graph::ValueKind::Weight)
            .count();
        assert_eq!(decls, 1);
        // Single row: the mask input is not declared.
        let one = build(&spec(), "text", &Cfg, &Src, &[("rows", Val::Num(1.0))]).unwrap();
        assert_eq!(one.inputs.len(), 1);
        assert_eq!(one.outputs, vec!["logits".to_string()]);
    }

    #[test]
    fn repeated_outputs() {
        let i = build(&spec(), "text", &Cfg, &Src, &[("rows", Val::Num(3.0))]).unwrap();
        assert_eq!(i.outputs, ["logits", "rep.0", "rep.1"]);
        assert_eq!(i.graph.outputs.len(), 3);
    }

    #[test]
    fn errors_name_the_node() {
        let mut s = spec();
        s.components["text"]["body"][0]["in"][0] = serde_json::json!("w:missing");
        let e = build(&s, "text", &Cfg, &Src, &[("rows", Val::Num(2.0))]).unwrap_err();
        assert!(
            e.contains("weight `missing` is missing") && e.contains("op \"Embedding\""),
            "{e}"
        );
        let e = build(&spec(), "text", &Cfg, &Src, &[]).unwrap_err();
        assert!(e.contains("`rows` not given"), "{e}");
        let taps = Val::List(vec![Val::Str("layer.1".into())]);
        let e = build(
            &spec(),
            "text",
            &Cfg,
            &Src,
            &[("rows", Val::Num(2.0)), ("taps", taps)],
        )
        .unwrap_err();
        assert!(e.contains("tap `layer.1`"), "{e}");
    }

    #[test]
    fn shape_cache_reuses_and_evicts() {
        let c = ShapeCache::new("text");
        let s = spec();
        let s = &s;
        let mk = |n: usize| move || build(s, "text", &Cfg, &Src, &[("rows", Val::Num(n as f64))]);
        let a = c.get(&[("rows", Val::Num(1.0))], mk(1)).unwrap();
        let b = c
            .get(&[("rows", Val::Num(1.0))], || Err("must hit".into()))
            .unwrap();
        assert!(std::sync::Arc::ptr_eq(&a, &b));
        for n in 2..=ShapeCache::CAP + 1 {
            c.get(&[("rows", Val::Num(n as f64))], mk(n)).unwrap();
        }
        assert!(c
            .get(&[("rows", Val::Num(1.0))], || Err("evicted".into()))
            .is_err());
    }

    /// `act_quant` (transformer lib, BitNet BitLinear) on the CPU graph: ties round half to even
    /// like torch.round, -127 stays in [-128, 127], an all-zero row keeps its zeros (absmax clamps
    /// at 1e-5 instead of dividing by zero).
    #[test]
    fn act_quant_rounds_half_to_even_and_clamps_like_torch() {
        struct Qp;
        impl Source for Qp {
            fn tensor(&self, name: &str) -> Option<(Vec<usize>, DType)> {
                (name == "act_quant.qp").then(|| (vec![1], DType::F32))
            }
        }
        let spec = r#"{family: "aq", pipeline: "causal_lm", use_lib: ["transformer"], components: {t: {
          vars: ["rows"], inputs: [{name: "x", dtype: "f32", shape: ["rows", 5]}],
          body: [{use: "act_quant", args: {bits: 8}, in: ["i:x"], out: "y"}], outputs: ["y"]}}}"#;
        let files: Vec<(String, String)> = crate::arch::EMBEDDED
            .iter()
            .filter(|(p, _)| p.starts_with("lib/"))
            .map(|(p, s)| (p.to_string(), s.to_string()))
            .chain([("families/aq.json5".into(), spec.into())])
            .collect();
        let s = Registry::from_files(&files, &[])
            .unwrap()
            .spec("aq", &Signals::default())
            .unwrap();
        let inst = build(&s, "t", &Cfg, &Qp, &[("rows", Val::Num(2.0))]).unwrap();
        let mut w = crate::graph::Weights::new();
        w.insert("act_quant.qp", crate::graph::Tensor::F32(vec![127.0]));
        let x = vec![127.0, 0.5, 2.5, -1.5, -127.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let out =
            crate::graph::cpu::run_stateless(&inst.graph, &w, vec![crate::graph::Tensor::F32(x)])
                .unwrap();
        let crate::graph::Tensor::F32(y) = &out[0] else {
            panic!("dtype")
        };
        assert_eq!(y, &[127.0, 0.0, 2.0, -2.0, -127.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }
}
