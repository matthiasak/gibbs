//! Lowers a Gibbs [`Graph`] to one StableHLO MLIR module (text) with StableHLO ops only.
//!
//! Every float value is computed in f32 and every U32 value as i32 (same bits). Parameters are
//! flat: graph inputs in order, then the weights the graph reads, then its states; results are
//! the graph outputs, then every state the graph writes (each aliased to its parameter with
//! `tf.aliasing_output`, so the plugin may update it in place). Semantics follow the CPU
//! executor (`graph::cpu`) op by op.

use crate::graph::{
    BinaryKind, DType, Graph, MegakernelKind, Node, Op, PairLayout, ReduceKind, RopeVariant,
    UnaryKind, ValueId, ValueKind,
};
use std::collections::HashMap;
use std::fmt::Write;

/// Element type of an emitted value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    F32,
    BF16,
    I32,
    I1,
}

impl Ty {
    fn mlir(self) -> &'static str {
        match self {
            Ty::F32 => "f32",
            Ty::BF16 => "bf16",
            Ty::I32 => "i32",
            Ty::I1 => "i1",
        }
    }
}

fn tstr(shape: &[usize], ty: Ty) -> String {
    let mut s = String::from("tensor<");
    for d in shape {
        let _ = write!(s, "{d}x");
    }
    s + ty.mlir() + ">"
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn list(v: &[usize]) -> String {
    let parts: Vec<String> = v.iter().map(usize::to_string).collect();
    format!("[{}]", parts.join(", "))
}

fn hex(x: f32) -> String {
    format!("0x{:08X}", x.to_bits())
}

/// An SSA value with its static shape and element type.
#[derive(Clone, Debug)]
struct V {
    name: String,
    shape: Vec<usize>,
    ty: Ty,
}

impl V {
    fn t(&self) -> String {
        tstr(&self.shape, self.ty)
    }
}

/// The emitted module and its parameter/result layout.
pub struct Module {
    pub text: String,
    /// Weight names bound after the graph inputs, in parameter order.
    pub weights: Vec<String>,
    /// State names bound after the weights, in parameter order.
    pub states: Vec<String>,
    /// States returned after the graph outputs, in result order.
    pub written: Vec<String>,
    /// Element count and type of each graph output.
    pub outputs: Vec<(usize, Ty)>,
    /// Element count and type of every parameter, in order.
    pub params: Vec<(usize, Ty)>,
}

/// One page class of the graph's paged KV states, bound as page windows: each named state's
/// parameter is `[rows, w]`, window slot `j` holding page row `j` (`slot_row[j] = j`, `-1` past
/// the class's pages), and the class adds the parameters `slot_row [rows]`, `write_slot [nq]`
/// (the slot each written query row lands in) and `base [1]` (`kv_base`: stored row of the run's
/// first position). `ring`: rows of a ring class (slot = stored row mod `ring`), 0 for full.
pub struct PagedClass {
    pub names: Vec<String>,
    pub rows: usize,
    pub nq: usize,
    pub ring: usize,
}

/// Emits `g`. `weight(name)` is the bound weight's element count and type (`None`: missing);
/// `states[name]` is the bound shape of each state the graph reads (f32); `paged` lists the
/// page classes whose states are bound as windows.
pub fn emit(
    g: &Graph,
    weight: &dyn Fn(&str) -> Option<(usize, Ty)>,
    states: &HashMap<String, Vec<usize>>,
    paged: &[PagedClass],
) -> Result<Module, String> {
    let mut e = Em {
        g,
        body: String::with_capacity(256 + g.nodes.len() * 600),
        n: 0,
        vals: vec![None; g.values.len()],
        cur: HashMap::new(),
        written: Vec::new(),
        paged: HashMap::new(),
        classes: Vec::new(),
    };
    let mut params: Vec<(String, String)> = Vec::new();
    let mut layout: Vec<(usize, Ty)> = Vec::new();
    for (i, &id) in g.inputs.iter().enumerate() {
        let v = g.value(id);
        let ty = match v.dtype {
            DType::U32 => Ty::I32,
            DType::F32 | DType::F16 | DType::BF16 => Ty::F32,
            DType::Quant(f) => return Err(format!("input '{}' is quantized ({f:?})", v.name)),
        };
        if v.shape.is_empty() {
            return Err(format!("input '{}' has no static shape", v.name));
        }
        let p = V {
            name: format!("%in{i}"),
            shape: vec![numel(&v.shape)],
            ty,
        };
        params.push((p.name.clone(), p.t()));
        layout.push((numel(&p.shape), p.ty));
        e.vals[id.0 as usize] = Some(e.reshape(&p, &v.shape));
    }
    // Weights and states in first-use order.
    let mut weights: Vec<String> = Vec::new();
    let mut state_names: Vec<String> = Vec::new();
    let mut weight_ids: HashMap<String, usize> = HashMap::new();
    for node in &g.nodes {
        for &id in &node.inputs {
            let v = g.value(id);
            match &v.kind {
                ValueKind::Weight if !weight_ids.contains_key(&v.name) => {
                    weight_ids.insert(v.name.clone(), weights.len());
                    weights.push(v.name.clone());
                }
                ValueKind::State { name, .. } if !state_names.contains(name) => {
                    state_names.push(name.clone())
                }
                _ => {}
            }
        }
    }
    for (i, name) in weights.iter().enumerate() {
        let (n, ty) = weight(name).ok_or_else(|| format!("weight '{name}' is missing"))?;
        let p = V {
            name: format!("%w{i}"),
            shape: vec![n],
            ty,
        };
        params.push((p.name.clone(), p.t()));
        layout.push((n, ty));
    }
    let first_state_param = params.len();
    for (i, name) in state_names.iter().enumerate() {
        let shape = states
            .get(name)
            .ok_or_else(|| format!("state '{name}' is not bound"))?;
        let p = V {
            name: format!("%s{i}"),
            shape: vec![numel(shape)],
            ty: Ty::F32,
        };
        params.push((p.name.clone(), p.t()));
        layout.push((numel(shape), Ty::F32));
        let v = e.reshape(&p, shape);
        e.cur.insert(name.clone(), v);
    }
    for (c, pc) in paged.iter().enumerate() {
        let mut vals = Vec::with_capacity(3);
        for (tag, n) in [("slot", pc.rows), ("wslot", pc.nq), ("base", 1)] {
            let p = V {
                name: format!("%{tag}{c}"),
                shape: vec![n],
                ty: Ty::I32,
            };
            params.push((p.name.clone(), p.t()));
            layout.push((n, Ty::I32));
            vals.push(p);
        }
        let base = e.reshape(&vals[2], &[]);
        e.classes.push((vals[0].clone(), vals[1].clone(), base, pc.ring));
        for name in &pc.names {
            e.paged.insert(name.clone(), c);
        }
    }
    // Weight values: flat param -> declared shape, f32 unless integer.
    for (i, v) in g.values.iter().enumerate() {
        if let (ValueKind::Weight, Some(&w)) = (&v.kind, weight_ids.get(&v.name)) {
            let (n, ty) = weight(&v.name).expect("checked above");
            let shape = if v.shape.is_empty() { vec![n] } else { v.shape.clone() };
            if numel(&shape) != n {
                return Err(format!(
                    "weight '{}' holds {n} elements, graph declares {:?}",
                    v.name, v.shape
                ));
            }
            let p = V {
                name: format!("%w{w}"),
                shape: vec![n],
                ty,
            };
            let p = if ty == Ty::BF16 { e.convert(&p, Ty::F32) } else { p };
            e.vals[i] = Some(e.reshape(&p, &shape));
        }
    }
    for node in &g.nodes {
        let out = node.outputs[0];
        let decl = g.value(out);
        let v = e
            .node(node)
            .map_err(|err| format!("node '{}' ({}): {err}", node.name, node.op.name()))?;
        let ty = if decl.dtype == DType::U32 { Ty::I32 } else { Ty::F32 };
        let v = e.cast(&v, ty);
        let v = if decl.shape.is_empty() {
            v
        } else if numel(&decl.shape) == numel(&v.shape) {
            e.reshape(&v, &decl.shape)
        } else {
            return Err(format!(
                "node '{}' ({}): computed {:?}, graph declares {:?}",
                node.name,
                node.op.name(),
                v.shape,
                decl.shape
            ));
        };
        e.vals[out.0 as usize] = Some(v);
    }
    let mut rets: Vec<V> = Vec::new();
    let mut outputs = Vec::new();
    for &o in &g.outputs {
        let v = e.val(o)?;
        let flat = e.reshape(&v, &[numel(&v.shape)]);
        outputs.push((numel(&v.shape), v.ty));
        rets.push(flat);
    }
    let written = e.written.clone();
    for (j, name) in written.iter().enumerate() {
        let v = e.cur[name].clone();
        let flat = e.reshape(&v, &[numel(&v.shape)]);
        let p = first_state_param
            + state_names
                .iter()
                .position(|s| s == name)
                .expect("written states are read states");
        params[p].1 += &format!(" {{tf.aliasing_output = {} : i32}}", outputs.len() + j);
        rets.push(flat);
    }
    let mut text = String::with_capacity(e.body.len() + 4096);
    let sig: Vec<String> = params.iter().map(|(n, t)| format!("{n}: {t}")).collect();
    let rty: Vec<String> = rets.iter().map(V::t).collect();
    let _ = writeln!(
        text,
        "module @gibbs {{\n  func.func public @main({}) -> ({}) {{",
        sig.join(", "),
        rty.join(", ")
    );
    text += &e.body;
    let rn: Vec<&str> = rets.iter().map(|v| v.name.as_str()).collect();
    let _ = writeln!(text, "    return {} : {}\n  }}\n}}", rn.join(", "), rty.join(", "));
    Ok(Module {
        text,
        weights,
        states: state_names,
        written,
        outputs,
        params: layout,
    })
}

struct Em<'g> {
    g: &'g Graph,
    body: String,
    n: usize,
    vals: Vec<Option<V>>,
    /// The current value of each state (a write replaces it for later readers).
    cur: HashMap<String, V>,
    written: Vec<String>,
    /// Paged state name → its class in `classes`.
    paged: HashMap<String, usize>,
    /// Per paged class: `slot_row`, `write_slot`, scalar `base`, ring rows (0 = full).
    classes: Vec<(V, V, V, usize)>,
}

#[derive(Clone, Copy)]
enum Red {
    Add,
    Max,
    Min,
}

impl<'g> Em<'g> {
    // ---- primitives ----

    fn op(&mut self, shape: Vec<usize>, ty: Ty, rhs: impl FnOnce(&str) -> String) -> V {
        self.n += 1;
        let name = format!("%t{}", self.n);
        let t = tstr(&shape, ty);
        let _ = writeln!(self.body, "    {name} = {}", rhs(&t));
        V { name, shape, ty }
    }

    fn kf(&mut self, x: f32, shape: &[usize]) -> V {
        self.op(shape.to_vec(), Ty::F32, |t| {
            format!("stablehlo.constant dense<{}> : {t}", hex(x))
        })
    }

    fn ki(&mut self, x: i64, shape: &[usize]) -> V {
        self.op(shape.to_vec(), Ty::I32, |t| {
            format!("stablehlo.constant dense<{x}> : {t}")
        })
    }

    fn dense(&mut self, vals: &[f32], shape: &[usize]) -> V {
        let items: Vec<String> = vals.iter().map(|&x| hex(x)).collect();
        let flat = self.op(vec![vals.len()], Ty::F32, |t| {
            format!("stablehlo.constant dense<[{}]> : {t}", items.join(", "))
        });
        self.reshape(&flat, shape)
    }

    /// Splat constant shaped like `like` (f32 or i32 by `like`'s type).
    fn k(&mut self, x: f64, like: &V) -> V {
        match like.ty {
            Ty::I32 => self.ki(x as i64, &like.shape.clone()),
            _ => self.kf(x as f32, &like.shape.clone()),
        }
    }

    fn un(&mut self, name: &str, a: &V) -> V {
        let at = a.t();
        self.op(a.shape.clone(), a.ty, |_| {
            format!("stablehlo.{name} {} : {at}", a.name)
        })
    }

    fn bin(&mut self, name: &str, a: &V, b: &V) -> V {
        debug_assert_eq!(a.shape, b.shape, "{name}");
        let at = a.t();
        self.op(a.shape.clone(), a.ty, |_| {
            format!("stablehlo.{name} {}, {} : {at}", a.name, b.name)
        })
    }

    fn add(&mut self, a: &V, b: &V) -> V {
        self.bin("add", a, b)
    }
    fn sub(&mut self, a: &V, b: &V) -> V {
        self.bin("subtract", a, b)
    }
    fn mul(&mut self, a: &V, b: &V) -> V {
        self.bin("multiply", a, b)
    }
    fn div(&mut self, a: &V, b: &V) -> V {
        self.bin("divide", a, b)
    }
    fn addc(&mut self, a: &V, c: f64) -> V {
        let k = self.k(c, a);
        self.add(a, &k)
    }
    fn mulc(&mut self, a: &V, c: f64) -> V {
        let k = self.k(c, a);
        self.mul(a, &k)
    }

    fn cmp(&mut self, dir: &str, a: &V, b: &V) -> V {
        let kind = if a.ty == Ty::I32 { "SIGNED" } else { "FLOAT" };
        let at = a.t();
        self.op(a.shape.clone(), Ty::I1, |t| {
            format!(
                "stablehlo.compare {dir}, {}, {}, {kind} : ({at}, {at}) -> {t}",
                a.name, b.name
            )
        })
    }

    fn cmpc(&mut self, dir: &str, a: &V, c: f64) -> V {
        let k = self.k(c, a);
        self.cmp(dir, a, &k)
    }

    fn and(&mut self, a: &V, b: &V) -> V {
        self.bin("and", a, b)
    }

    fn sel(&mut self, p: &V, a: &V, b: &V) -> V {
        let (pt, at) = (p.t(), a.t());
        self.op(a.shape.clone(), a.ty, |_| {
            format!("stablehlo.select {}, {}, {} : {pt}, {at}", p.name, a.name, b.name)
        })
    }

    fn reshape(&mut self, a: &V, shape: &[usize]) -> V {
        if a.shape == shape {
            return a.clone();
        }
        debug_assert_eq!(numel(&a.shape), numel(shape));
        let at = a.t();
        self.op(shape.to_vec(), a.ty, |t| {
            format!("stablehlo.reshape {} : ({at}) -> {t}", a.name)
        })
    }

    fn convert(&mut self, a: &V, ty: Ty) -> V {
        let at = a.t();
        self.op(a.shape.clone(), ty, |t| {
            format!("stablehlo.convert {} : ({at}) -> {t}", a.name)
        })
    }

    /// `a` as `ty` (I1 widens to 0/1).
    fn cast(&mut self, a: &V, ty: Ty) -> V {
        if a.ty == ty {
            a.clone()
        } else {
            self.convert(a, ty)
        }
    }

    fn bcast(&mut self, a: &V, shape: &[usize], dims: &[usize]) -> V {
        if a.shape == shape {
            return a.clone();
        }
        let at = a.t();
        self.op(shape.to_vec(), a.ty, |t| {
            format!(
                "stablehlo.broadcast_in_dim {}, dims = {} : ({at}) -> {t}",
                a.name,
                list(dims)
            )
        })
    }

    /// numpy (right-aligned) broadcast of `a` to `shape`.
    fn bto(&mut self, a: &V, shape: &[usize]) -> Result<V, String> {
        if a.shape == shape {
            return Ok(a.clone());
        }
        let (r, ar) = (shape.len(), a.shape.len());
        if ar > r {
            return Err(format!("cannot broadcast {:?} to {shape:?}", a.shape));
        }
        let (mut kept, mut dims) = (Vec::new(), Vec::new());
        for (j, &d) in a.shape.iter().enumerate() {
            let o = r - ar + j;
            if d == shape[o] {
                kept.push(d);
                dims.push(o);
            } else if d != 1 {
                return Err(format!("cannot broadcast {:?} to {shape:?}", a.shape));
            }
        }
        let a = self.reshape(a, &kept);
        Ok(self.bcast(&a, shape, &dims))
    }

    fn transpose(&mut self, a: &V, perm: &[usize]) -> V {
        if perm.iter().enumerate().all(|(i, &p)| i == p) {
            return a.clone();
        }
        let shape: Vec<usize> = perm.iter().map(|&p| a.shape[p]).collect();
        let at = a.t();
        self.op(shape, a.ty, |t| {
            format!(
                "stablehlo.transpose {}, dims = {} : ({at}) -> {t}",
                a.name,
                list(perm)
            )
        })
    }

    fn slice(&mut self, a: &V, start: &[usize], limit: &[usize]) -> V {
        if start.iter().all(|&s| s == 0) && limit == a.shape.as_slice() {
            return a.clone();
        }
        let shape: Vec<usize> = start.iter().zip(limit).map(|(s, l)| l - s).collect();
        let ranges: Vec<String> = start
            .iter()
            .zip(limit)
            .map(|(s, l)| format!("{s}:{l}"))
            .collect();
        let at = a.t();
        self.op(shape, a.ty, |t| {
            format!(
                "stablehlo.slice {} [{}] : ({at}) -> {t}",
                a.name,
                ranges.join(", ")
            )
        })
    }

    /// `a[.., lo..hi, ..]` along `axis`.
    fn slice_axis(&mut self, a: &V, axis: usize, lo: usize, hi: usize) -> V {
        let mut s = vec![0; a.shape.len()];
        let mut l = a.shape.clone();
        (s[axis], l[axis]) = (lo, hi);
        self.slice(a, &s, &l)
    }

    fn concat(&mut self, parts: &[V], dim: usize) -> V {
        let parts: Vec<&V> = parts.iter().filter(|p| p.shape[dim] > 0).collect();
        if parts.len() == 1 {
            return parts[0].clone();
        }
        let mut shape = parts[0].shape.clone();
        shape[dim] = parts.iter().map(|p| p.shape[dim]).sum();
        let names: Vec<&str> = parts.iter().map(|p| p.name.as_str()).collect();
        let tys: Vec<String> = parts.iter().map(|p| p.t()).collect();
        self.op(shape, parts[0].ty, |t| {
            format!(
                "stablehlo.concatenate {}, dim = {dim} : ({}) -> {t}",
                names.join(", "),
                tys.join(", ")
            )
        })
    }

    fn iota(&mut self, shape: &[usize], dim: usize) -> V {
        self.op(shape.to_vec(), Ty::I32, |t| {
            format!("stablehlo.iota dim = {dim} : {t}")
        })
    }

    fn reduce(&mut self, a: &V, dims: &[usize], how: Red) -> V {
        let (init, name) = match how {
            Red::Add => (0.0, "add"),
            Red::Max => (f64::NEG_INFINITY, "maximum"),
            Red::Min => (f64::INFINITY, "minimum"),
        };
        let init = match a.ty {
            Ty::I32 => self.ki(
                match how {
                    Red::Add => 0,
                    Red::Max => i32::MIN as i64,
                    Red::Min => i32::MAX as i64,
                },
                &[],
            ),
            _ => self.kf(init as f32, &[]),
        };
        let shape: Vec<usize> = (0..a.shape.len())
            .filter(|d| !dims.contains(d))
            .map(|d| a.shape[d])
            .collect();
        let (at, it) = (a.t(), init.t());
        self.op(shape, a.ty, |t| {
            format!(
                "stablehlo.reduce({} init: {}) applies stablehlo.{name} across dimensions = {} : ({at}, {it}) -> {t}",
                a.name,
                init.name,
                list(dims)
            )
        })
    }

    /// `reduce(a, [axis])` broadcast back to `a`'s shape.
    fn reduce_keep(&mut self, a: &V, axis: usize, how: Red) -> V {
        let r = self.reduce(a, &[axis], how);
        let dims: Vec<usize> = (0..a.shape.len()).filter(|&d| d != axis).collect();
        self.bcast(&r, &a.shape.clone(), &dims)
    }

    /// `dot_general`: result dims are batch, lhs free, rhs free.
    fn dot(&mut self, a: &V, b: &V, batch: (&[usize], &[usize]), contract: (&[usize], &[usize])) -> V {
        let mut shape: Vec<usize> = batch.0.iter().map(|&d| a.shape[d]).collect();
        for (d, &n) in a.shape.iter().enumerate() {
            if !batch.0.contains(&d) && !contract.0.contains(&d) {
                shape.push(n);
            }
        }
        for (d, &n) in b.shape.iter().enumerate() {
            if !batch.1.contains(&d) && !contract.1.contains(&d) {
                shape.push(n);
            }
        }
        let bd = if batch.0.is_empty() {
            String::new()
        } else {
            format!("batching_dims = {} x {}, ", list(batch.0), list(batch.1))
        };
        let (at, bt) = (a.t(), b.t());
        self.op(shape, Ty::F32, |t| {
            format!(
                "stablehlo.dot_general {}, {}, {bd}contracting_dims = {} x {} : ({at}, {bt}) -> {t}",
                a.name,
                b.name,
                list(contract.0),
                list(contract.1)
            )
        })
    }

    /// Scalar i32 `a[i]` of a rank-1 value.
    fn elem(&mut self, a: &V, i: usize) -> V {
        let s = self.slice(a, &[i], &[i + 1]);
        self.reshape(&s, &[])
    }

    fn dus(&mut self, a: &V, upd: &V, starts: &[V]) -> V {
        let names: Vec<&str> = starts.iter().map(|s| s.name.as_str()).collect();
        let tys: Vec<String> = starts.iter().map(|s| s.t()).collect();
        let (at, ut) = (a.t(), upd.t());
        self.op(a.shape.clone(), a.ty, |t| {
            format!(
                "stablehlo.dynamic_update_slice {}, {}, {} : ({at}, {ut}, {}) -> {t}",
                a.name,
                upd.name,
                names.join(", "),
                tys.join(", ")
            )
        })
    }

    fn dslice(&mut self, a: &V, starts: &[V], sizes: &[usize]) -> V {
        let names: Vec<&str> = starts.iter().map(|s| s.name.as_str()).collect();
        let tys: Vec<String> = starts.iter().map(|s| s.t()).collect();
        let at = a.t();
        self.op(sizes.to_vec(), a.ty, |t| {
            format!(
                "stablehlo.dynamic_slice {}, {}, sizes = {} : ({at}, {}) -> {t}",
                a.name,
                names.join(", "),
                list(sizes),
                tys.join(", ")
            )
        })
    }

    /// `table[ids[i], :]` for `table [R, D]`, `ids [N]` i32 → `[N, D]` (ids clamped by XLA).
    fn gather_rows(&mut self, table: &V, ids: &V) -> V {
        let (n, d) = (numel(&ids.shape), table.shape[1]);
        let ids = self.reshape(ids, &[n, 1]);
        let (tt, it) = (table.t(), ids.t());
        self.op(vec![n, d], table.ty, |t| {
            format!(
                "\"stablehlo.gather\"({}, {}) {{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, {d}>, indices_are_sorted = false}} : ({tt}, {it}) -> {t}",
                table.name, ids.name
            )
        })
    }

    /// `base` with rows `ids[i]` replaced by `upd[i]` (`base [R, D]`, `upd [N, D]`).
    fn scatter_rows(&mut self, base: &V, ids: &V, upd: &V) -> V {
        let n = numel(&ids.shape);
        let ids = self.reshape(ids, &[n, 1]);
        let (bt, it, ut) = (base.t(), ids.t(), upd.t());
        let st = tstr(&[], base.ty);
        self.n += 1;
        let (a, b) = (format!("%r{}a", self.n), format!("%r{}b", self.n));
        self.op(base.shape.clone(), base.ty, |t| {
            format!(
                "\"stablehlo.scatter\"({}, {}, {}) ({{\n    ^bb0({a}: {st}, {b}: {st}):\n      stablehlo.return {b} : {st}\n    }}) {{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false}} : ({bt}, {it}, {ut}) -> {t}",
                base.name, ids.name, upd.name
            )
        })
    }

    /// Stable sort of `keys` along `dim` carrying `vals`; `dir` `GT` sorts descending. Returns the
    /// permuted `vals`.
    fn sort_by(&mut self, keys: &V, vals: &V, dim: usize, dir: &str) -> V {
        self.n += 1;
        let r = format!("%t{}", self.n);
        let tag = self.n;
        let (kt, vt) = (keys.t(), vals.t());
        let (ks, vs) = (tstr(&[], keys.ty), tstr(&[], vals.ty));
        let kind = if keys.ty == Ty::I32 { "SIGNED" } else { "FLOAT" };
        let _ = write!(
            self.body,
            "    {r}:2 = \"stablehlo.sort\"({}, {}) ({{\n    ^bb0(%q{tag}a: {ks}, %q{tag}b: {ks}, %q{tag}c: {vs}, %q{tag}d: {vs}):\n      %q{tag}p = stablehlo.compare {dir}, %q{tag}a, %q{tag}b, {kind} : ({ks}, {ks}) -> tensor<i1>\n      stablehlo.return %q{tag}p : tensor<i1>\n    }}) {{dimension = {dim} : i64, is_stable = true}} : ({kt}, {vt}) -> ({kt}, {vt})\n",
            keys.name, vals.name
        );
        V {
            name: format!("{r}#1"),
            shape: vals.shape.clone(),
            ty: vals.ty,
        }
    }

    // ---- operands ----

    fn val(&mut self, id: ValueId) -> Result<V, String> {
        let v = self.g.value(id);
        if let ValueKind::State { name, .. } = &v.kind {
            return self
                .cur
                .get(name)
                .cloned()
                .ok_or_else(|| format!("state '{name}' is not bound"));
        }
        self.vals[id.0 as usize]
            .clone()
            .ok_or_else(|| format!("value '{}' is not computed", v.name))
    }

    fn flt(&mut self, id: ValueId) -> Result<V, String> {
        let v = self.val(id)?;
        Ok(self.cast(&v, Ty::F32))
    }

    fn idx(&mut self, id: ValueId) -> Result<V, String> {
        let v = self.val(id)?;
        Ok(self.cast(&v, Ty::I32))
    }

    /// `x` viewed as `[len / last, last]`.
    fn rows(&mut self, x: &V, last: usize) -> Result<V, String> {
        let n = numel(&x.shape);
        if last == 0 || n % last != 0 {
            return Err(format!("{:?} is not rows of {last}", x.shape));
        }
        Ok(self.reshape(x, &[n / last, last]))
    }

    // ---- composite math ----

    fn sigmoid(&mut self, x: &V) -> V {
        let nx = self.un("negate", x);
        let ex = self.un("exponential", &nx);
        let den = self.addc(&ex, 1.0);
        let one = self.k(1.0, x);
        self.div(&one, &den)
    }

    fn gelu_tanh(&mut self, x: &V) -> V {
        let x2 = self.mul(x, x);
        let x3 = self.mul(&x2, x);
        let c = self.mulc(&x3, 0.044715);
        let s = self.add(x, &c);
        let s = self.mulc(&s, 0.797_884_560_8);
        let th = self.un("tanh", &s);
        let th = self.addc(&th, 1.0);
        let h = self.mulc(x, 0.5);
        self.mul(&h, &th)
    }

    /// `nn::gelu`'s erf (Numerical Recipes erfc fit).
    fn gelu_erf(&mut self, x: &V) -> V {
        let z0 = self.mulc(x, std::f64::consts::FRAC_1_SQRT_2);
        let z = self.un("abs", &z0);
        let hz = self.mulc(&z, 0.5);
        let den = self.addc(&hz, 1.0);
        let one = self.k(1.0, x);
        let t = self.div(&one, &den);
        const C: [f64; 9] = [
            1.00002368,
            0.37409196,
            0.09678418,
            -0.18628806,
            0.27886807,
            -1.13520398,
            1.48851587,
            -0.82215223,
            0.17087277,
        ];
        let mut acc = self.k(C[8], x);
        for &c in C[..8].iter().rev() {
            let m = self.mul(&t, &acc);
            acc = self.addc(&m, c);
        }
        let ta = self.mul(&t, &acc);
        let zz = self.mul(&z, &z);
        let poly = self.sub(&ta, &zz);
        let poly = self.addc(&poly, -1.26551223);
        let ex = self.un("exponential", &poly);
        let erfc = self.mul(&t, &ex);
        let pos = self.k(1.0, x);
        let pos = self.sub(&pos, &erfc);
        let neg = self.addc(&erfc, -1.0);
        let ge = self.cmpc("GE", &z0, 0.0);
        let erf = self.sel(&ge, &pos, &neg);
        let erf = self.addc(&erf, 1.0);
        let h = self.mulc(x, 0.5);
        self.mul(&h, &erf)
    }

    fn unary(&mut self, kind: &UnaryKind, x: &V) -> V {
        match kind {
            UnaryKind::Exp => self.un("exponential", x),
            UnaryKind::Sigmoid => self.sigmoid(x),
            UnaryKind::Tanh => self.un("tanh", x),
            UnaryKind::Softplus => {
                let e = self.un("exponential", x);
                let l = self.un("log_plus_one", &e);
                let big = self.cmpc("GT", x, 20.0);
                self.sel(&big, x, &l)
            }
            UnaryKind::Neg => self.un("negate", x),
            UnaryKind::Sqrt => self.un("sqrt", x),
            UnaryKind::Rsqrt => {
                let s = self.un("sqrt", x);
                let one = self.k(1.0, x);
                self.div(&one, &s)
            }
            UnaryKind::Abs => self.un("abs", x),
            UnaryKind::Relu => {
                let z = self.k(0.0, x);
                self.bin("maximum", x, &z)
            }
            UnaryKind::Relu2 => {
                let z = self.k(0.0, x);
                let r = self.bin("maximum", x, &z);
                self.mul(&r, &r)
            }
            UnaryKind::Silu => {
                let s = self.sigmoid(x);
                self.mul(x, &s)
            }
            UnaryKind::GeluErf => self.gelu_erf(x),
            UnaryKind::GeluTanh => self.gelu_tanh(x),
            UnaryKind::Scale(s) => self.mulc(x, *s as f64),
            UnaryKind::Clamp(lo, hi) => {
                let (lo, hi) = (self.kf(*lo, &[]), self.kf(*hi, &[]));
                let xt = x.t();
                self.op(x.shape.clone(), x.ty, |_| {
                    format!(
                        "stablehlo.clamp {}, {}, {} : (tensor<f32>, {xt}, tensor<f32>) -> {xt}",
                        lo.name, x.name, hi.name
                    )
                })
            }
            UnaryKind::SoftCap(cap) => self.softcap(x, *cap),
            UnaryKind::Ln => self.un("log", x),
            UnaryKind::RoundEven => self.un("round_nearest_even", x),
            UnaryKind::Sign => {
                let (one, mone, zero) = (self.k(1.0, x), self.k(-1.0, x), self.k(0.0, x));
                let lt = self.cmpc("LT", x, 0.0);
                let gt = self.cmpc("GT", x, 0.0);
                let neg = self.sel(&lt, &mone, &zero);
                self.sel(&gt, &one, &neg)
            }
        }
    }

    fn softcap(&mut self, x: &V, cap: f32) -> V {
        let c = self.k(cap as f64, x);
        let d = self.div(x, &c);
        let t = self.un("tanh", &d);
        self.mulc(&t, cap as f64)
    }

    fn rms(&mut self, x: &V, g: Option<&V>, eps: f32) -> V {
        let d = x.shape[1];
        let sq = self.mul(x, x);
        let ss = self.reduce(&sq, &[1], Red::Add);
        let dk = self.k(d as f64, &ss);
        let ms = self.div(&ss, &dk);
        let ms = self.addc(&ms, eps as f64);
        let s = self.un("sqrt", &ms);
        let one = self.k(1.0, &s);
        let inv = self.div(&one, &s);
        let inv = self.bcast(&inv, &x.shape.clone(), &[0]);
        let y = self.mul(x, &inv);
        match g {
            Some(g) => {
                let g = self.bcast(g, &x.shape.clone(), &[1]);
                self.mul(&y, &g)
            }
            None => y,
        }
    }

    /// `x [rows, in] · w [out, in]ᵀ (+ bias)`.
    fn linear(&mut self, x: &V, w: &V, bias: Option<&V>) -> Result<V, String> {
        if w.shape.len() != 2 || w.shape[1] != x.shape[1] {
            return Err(format!("x {:?} does not match weight {:?}", x.shape, w.shape));
        }
        let y = self.dot(x, w, (&[], &[]), (&[1], &[1]));
        Ok(match bias {
            Some(b) => {
                let b = self.reshape(b, &[w.shape[0]]);
                let b = self.bcast(&b, &y.shape.clone(), &[1]);
                self.add(&y, &b)
            }
            None => y,
        })
    }

    /// Masked softmax attention. `q [nq, h, d]`, `k [nk, kvh, d]`, `v [nk, kvh, dv]`, `mask [nq, nk]`
    /// (true = visible), `bias [h, nq, nk]` added after the soft cap → `[nq, h * dv]`. A query
    /// with no visible key yields zeros.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &mut self,
        q: &V,
        k: &V,
        v: &V,
        mask: &V,
        scale: f32,
        soft_cap: Option<f32>,
        bias: Option<&V>,
    ) -> V {
        let (nq, h, d) = (q.shape[0], q.shape[1], q.shape[2]);
        let (nk, kvh, dv) = (k.shape[0], k.shape[1], v.shape[2]);
        let rep = h / kvh;
        let q = self.reshape(q, &[nq, kvh, rep, d]);
        let q = self.transpose(&q, &[1, 0, 2, 3]);
        let q = self.reshape(&q, &[kvh, nq * rep, d]);
        let kt = self.transpose(k, &[1, 0, 2]);
        let s = self.dot(&q, &kt, (&[0], &[0]), (&[2], &[2]));
        let s = self.mulc(&s, scale as f64);
        let s = match soft_cap {
            Some(c) => self.softcap(&s, c),
            None => s,
        };
        let shape4 = [kvh, nq, rep, nk];
        let mut s = self.reshape(&s, &shape4);
        if let Some(b) = bias {
            let b = self.reshape(b, &[kvh, rep, nq, nk]);
            let b = self.transpose(&b, &[0, 2, 1, 3]);
            s = self.add(&s, &b);
        }
        let m = self.bcast(mask, &shape4, &[1, 3]);
        let ninf = self.k(f64::NEG_INFINITY, &s);
        let s = self.sel(&m, &s, &ninf);
        let mx = self.reduce(&s, &[3], Red::Max);
        let isinf = self.cmpc("EQ", &mx, f64::NEG_INFINITY);
        let zero = self.k(0.0, &mx);
        let mx = self.sel(&isinf, &zero, &mx);
        let mx = self.bcast(&mx, &shape4, &[0, 1, 2]);
        let e = self.sub(&s, &mx);
        let e = self.un("exponential", &e);
        let sum = self.reduce(&e, &[3], Red::Add);
        let z = self.cmpc("EQ", &sum, 0.0);
        let one = self.k(1.0, &sum);
        let sum = self.sel(&z, &one, &sum);
        let sum = self.bcast(&sum, &shape4, &[0, 1, 2]);
        let w = self.div(&e, &sum);
        let w = self.reshape(&w, &[kvh, nq * rep, nk]);
        let vt = self.transpose(v, &[1, 0, 2]);
        let o = self.dot(&w, &vt, (&[0], &[0]), (&[2], &[1]));
        let o = self.reshape(&o, &[kvh, nq, rep, dv]);
        let o = self.transpose(&o, &[1, 0, 2, 3]);
        self.reshape(&o, &[nq, h * dv])
    }

    fn set_state(&mut self, id: ValueId, v: V) -> Result<(), String> {
        let ValueKind::State { name, .. } = &self.g.value(id).kind else {
            return Ok(());
        };
        let bound = self.cur[name].shape.clone();
        if numel(&bound) != numel(&v.shape) {
            return Err(format!(
                "writes {:?} into state '{name}' bound as {bound:?}",
                v.shape
            ));
        }
        let v = self.reshape(&v, &bound);
        self.cur.insert(name.clone(), v);
        if !self.written.contains(name) {
            self.written.push(name.clone());
        }
        Ok(())
    }

    fn norm_axis(axis: isize, rank: usize) -> Result<usize, String> {
        let a = if axis < 0 { axis + rank as isize } else { axis };
        if a < 0 || a as usize >= rank {
            return Err(format!("axis {axis} out of range for rank {rank}"));
        }
        Ok(a as usize)
    }

    // ---- ops ----

    fn node(&mut self, node: &Node) -> Result<V, String> {
        let ins = &node.inputs;
        let g = self.g;
        let decl = g.value(node.outputs[0]).shape.clone();
        let shape_of = |k: usize| g.value(ins[k]).shape.clone();
        match &node.op {
            Op::Embedding => {
                let t = self.flt(ins[0])?;
                let d = t.shape.last().copied().unwrap_or(0);
                let t = self.rows(&t, d)?;
                let ids = self.idx(ins[1])?;
                let n = numel(&ids.shape);
                let ids = self.reshape(&ids, &[n]);
                Ok(self.gather_rows(&t, &ids))
            }
            Op::Linear => {
                let w = self.flt(ins[1])?;
                let x = self.flt(ins[0])?;
                let x = self.rows(&x, *w.shape.get(1).unwrap_or(&0))?;
                let b = match ins.get(2) {
                    Some(&b) => Some(self.flt(b)?),
                    None => None,
                };
                self.linear(&x, &w, b.as_ref())
            }
            Op::MatMul {
                transpose_a,
                transpose_b,
            } => {
                let (a, b) = (self.flt(ins[0])?, self.flt(ins[1])?);
                let (sa, sb) = (a.shape.clone(), b.shape.clone());
                if sa.len() < 2 || sb.len() < 2 {
                    return Err("MatMul operands must be rank >= 2".into());
                }
                let (ar, ac) = (sa[sa.len() - 2], sa[sa.len() - 1]);
                let (br, bc) = (sb[sb.len() - 2], sb[sb.len() - 1]);
                let nd = (sa.len() - 2).max(sb.len() - 2);
                let pad = |d: &[usize]| [vec![1; nd - d.len()], d.to_vec()].concat();
                let (pa, pb) = (pad(&sa[..sa.len() - 2]), pad(&sb[..sb.len() - 2]));
                let mut ob = Vec::with_capacity(nd);
                for i in 0..nd {
                    match (pa[i], pb[i]) {
                        (x, y) if x == y || y == 1 => ob.push(x),
                        (1, y) => ob.push(y),
                        _ => return Err(format!("MatMul batch dims {sa:?} and {sb:?} do not broadcast")),
                    }
                }
                let nb = numel(&ob);
                let a = self.bto(&a, &[ob.clone(), vec![ar, ac]].concat())?;
                let b = self.bto(&b, &[ob.clone(), vec![br, bc]].concat())?;
                let a = self.reshape(&a, &[nb, ar, ac]);
                let b = self.reshape(&b, &[nb, br, bc]);
                let ca = if *transpose_a { 1 } else { 2 };
                let cb = if *transpose_b { 2 } else { 1 };
                let o = self.dot(&a, &b, (&[0], &[0]), (&[ca], &[cb]));
                let (m, n) = (o.shape[1], o.shape[2]);
                Ok(self.reshape(&o, &[ob, vec![m, n]].concat()))
            }
            Op::Megakernel(desc) => {
                let MegakernelKind::Mlp {
                    hidden_dim,
                    intermediate_dim: _,
                    eps,
                    is_geglu,
                    has_post_norm,
                } = &desc.kind;
                let x = self.flt(ins[0])?;
                let x = self.rows(&x, *hidden_dim)?;
                let nw = self.flt(ins[1])?;
                let nw = self.reshape(&nw, &[*hidden_dim]);
                let (gw, uw, dw) = (self.flt(ins[2])?, self.flt(ins[3])?, self.flt(ins[4])?);
                let xn = self.rms(&x, Some(&nw), *eps);
                let gate = self.linear(&xn, &gw, None)?;
                let up = self.linear(&xn, &uw, None)?;
                let act = if *is_geglu {
                    self.gelu_tanh(&gate)
                } else {
                    let s = self.sigmoid(&gate);
                    self.mul(&gate, &s)
                };
                let act = self.mul(&act, &up);
                let mut down = self.linear(&act, &dw, None)?;
                if *has_post_norm {
                    let pw = self.flt(ins[5])?;
                    let pw = self.reshape(&pw, &[*hidden_dim]);
                    down = self.rms(&down, Some(&pw), *eps);
                }
                Ok(self.add(&x, &down))
            }
            Op::RmsNorm { eps } => {
                let x = self.flt(ins[0])?;
                let gamma = match ins.get(1) {
                    Some(&gi) => Some(self.flt(gi)?),
                    None => None,
                };
                let d = match &gamma {
                    Some(gv) => numel(&gv.shape),
                    None => *x.shape.last().unwrap_or(&0),
                };
                let x = self.rows(&x, d)?;
                let gamma = gamma.map(|gv| self.reshape(&gv, &[d]));
                Ok(self.rms(&x, gamma.as_ref(), *eps))
            }
            Op::SwiGlu | Op::GeGlu => {
                let (gt, up) = (self.flt(ins[0])?, self.flt(ins[1])?);
                let n = numel(&gt.shape);
                if n != numel(&up.shape) {
                    return Err("gate and up lengths differ".into());
                }
                let gt = self.reshape(&gt, &[n]);
                let up = self.reshape(&up, &[n]);
                let act = if matches!(node.op, Op::GeGlu) {
                    self.gelu_tanh(&gt)
                } else {
                    let s = self.sigmoid(&gt);
                    self.mul(&gt, &s)
                };
                Ok(self.mul(&act, &up))
            }
            Op::Unary { kind } => {
                let x = self.flt(ins[0])?;
                Ok(self.unary(kind, &x))
            }
            Op::Binary { kind } => {
                let (a, b) = (self.flt(ins[0])?, self.flt(ins[1])?);
                let name = match kind {
                    BinaryKind::Add => "add",
                    BinaryKind::Sub => "subtract",
                    BinaryKind::Mul => "multiply",
                    BinaryKind::Div => "divide",
                    BinaryKind::Max => "maximum",
                    BinaryKind::Min => "minimum",
                };
                let (sa, sb) = (a.shape.clone(), b.shape.clone());
                // numpy broadcast to the declared element count, as `cpu::broadcast_np`.
                if sa != sb {
                    let r = sa.len().max(sb.len());
                    let dim = |s: &[usize], i: usize| {
                        if i + s.len() >= r {
                            s[i + s.len() - r]
                        } else {
                            1
                        }
                    };
                    let shape: Option<Vec<usize>> = (0..r)
                        .map(|i| {
                            let (x, y) = (dim(&sa, i), dim(&sb, i));
                            (x == y || x == 1 || y == 1).then_some(if x == 1 { y } else { x })
                        })
                        .collect();
                    if let Some(shape) = shape.filter(|s| numel(s) == numel(&decl)) {
                        let a = self.bto(&a, &shape)?;
                        let b = self.bto(&b, &shape)?;
                        return Ok(self.bin(name, &a, &b));
                    }
                    // `cpu::row_scale`: `[m, ..] * [m, 1]` scales row r by the other's r.
                    if *kind == BinaryKind::Mul {
                        let fits = |x: &[usize], s: &[usize]| {
                            x.len() >= 2 && s.len() >= 2 && s.last() == Some(&1) && x[0] == s[0]
                        };
                        for (x, s) in [(&a, &b), (&b, &a)] {
                            if fits(&x.shape, &s.shape) {
                                let m = x.shape[0];
                                let x2 = self.rows(x, numel(&x.shape) / m)?;
                                let s1 = self.reshape(s, &[m]);
                                let s2 = self.bcast(&s1, &x2.shape.clone(), &[0]);
                                return Ok(self.mul(&x2, &s2));
                            }
                        }
                    }
                }
                // `cpu::tile_binary`: flat zip, or the shorter operand tiled over the longer.
                let (na, nb) = (numel(&sa), numel(&sb));
                if na == nb {
                    let a = self.reshape(&a, &[na]);
                    let b = self.reshape(&b, &[na]);
                    Ok(self.bin(name, &a, &b))
                } else if nb > 0 && na % nb == 0 {
                    let a2 = self.reshape(&a, &[na / nb, nb]);
                    let b1 = self.reshape(&b, &[nb]);
                    let b2 = self.bcast(&b1, &[na / nb, nb], &[1]);
                    Ok(self.bin(name, &a2, &b2))
                } else if na > 0 && nb % na == 0 {
                    let b2 = self.reshape(&b, &[nb / na, na]);
                    let a1 = self.reshape(&a, &[na]);
                    let a2 = self.bcast(&a1, &[nb / na, na], &[1]);
                    Ok(self.bin(name, &a2, &b2))
                } else {
                    Err(format!("shape mismatch {sa:?} vs {sb:?}"))
                }
            }
            Op::Softmax { axis } => {
                let x = self.flt(ins[0])?;
                let ax = Self::norm_axis(*axis, x.shape.len())?;
                let m = self.reduce_keep(&x, ax, Red::Max);
                let e = self.sub(&x, &m);
                let e = self.un("exponential", &e);
                let s = self.reduce_keep(&e, ax, Red::Add);
                let lo = self.k(1e-12, &s);
                let s = self.bin("maximum", &s, &lo);
                Ok(self.div(&e, &s))
            }
            Op::TopK { k } => {
                let x = self.flt(ins[0])?;
                let cols = *x.shape.last().unwrap_or(&0);
                if *k == 0 || *k > cols {
                    return Err(format!("TopK needs 1 <= k <= {cols}, got {k}"));
                }
                let x = self.rows(&x, cols)?;
                let nan = self.cmp("NE", &x, &x);
                let ninf = self.k(f64::NEG_INFINITY, &x);
                let x = self.sel(&nan, &ninf, &x);
                let iota = self.iota(&x.shape.clone(), 1);
                let order = self.sort_by(&x, &iota, 1, "GT");
                let rows = x.shape[0];
                Ok(self.slice(&order, &[0, 0], &[rows, *k]))
            }
            Op::GatherCols => {
                let x = self.flt(ins[0])?;
                let idx = self.idx(ins[1])?;
                let (cols, k) = (
                    *x.shape.last().unwrap_or(&0),
                    *idx.shape.last().unwrap_or(&0),
                );
                let x = self.rows(&x, cols)?;
                let idx = self.rows(&idx, k)?;
                let rows = idx.shape[0];
                let flat = if x.shape[0] == 1 {
                    idx
                } else {
                    let r = self.iota(&[rows, k], 0);
                    let off = self.k(cols as f64, &r);
                    let r = self.mul(&r, &off);
                    self.add(&r, &idx)
                };
                let xf = self.reshape(&x, &[numel(&x.shape), 1]);
                let flat = self.reshape(&flat, &[rows * k]);
                Ok(self.gather_rows(&xf, &flat))
            }
            Op::MoeLinear { experts, top_k } => {
                let bank = self.flt(ins[1])?;
                let [bank_rows, in_f] = bank.shape[..] else {
                    return Err("the expert bank is not [experts * out, in]".into());
                };
                let out_f = bank_rows / experts;
                let x = self.flt(ins[0])?;
                let x = self.rows(&x, in_f)?;
                let ids = self.idx(ins[2])?;
                let slots = numel(&ids.shape);
                let xr = x.shape[0];
                let per_slot = xr == slots;
                // ponytail: every expert for every row ([rows, experts * out]), then the routed
                // rows; a per-slot weight gather wins when experts / top_k is large.
                let all = self.dot(&x, &bank, (&[], &[]), (&[1], &[1]));
                let all = self.reshape(&all, &[xr * experts, out_f]);
                let ids = self.reshape(&ids, &[slots]);
                let s = self.iota(&[slots], 0);
                let r = if per_slot {
                    s
                } else {
                    let tk = self.k(*top_k as f64, &s);
                    self.div(&s, &tk)
                };
                let e = self.k(*experts as f64, &r);
                let r = self.mul(&r, &e);
                let at = self.add(&r, &ids);
                Ok(self.gather_rows(&all, &at))
            }
            Op::Rope {
                head_dim,
                rotary_dim,
                variant,
                mscale,
            } => {
                let x = self.flt(ins[0])?;
                let pos = self.flt(ins[1])?;
                let inv = self.flt(ins[2])?;
                let half = rotary_dim / 2;
                if numel(&inv.shape) != half {
                    return Err(format!("inv_freq has {:?}, expected {half}", inv.shape));
                }
                let inv = self.reshape(&inv, &[half]);
                let np = numel(&pos.shape);
                let n = match variant {
                    RopeVariant::MRope { .. } => np / 3,
                    _ => np,
                }
                .max(1);
                let total = numel(&x.shape) / n;
                let (hd, heads) = (*head_dim, total / head_dim);
                let p = match variant {
                    RopeVariant::MRope {
                        sections,
                        interleaved,
                        ..
                    } => {
                        let p3 = self.reshape(&pos, &[n, 3]);
                        let mut acc: Option<V> = None;
                        for which in 0..3 {
                            let mask: Vec<f32> = (0..half)
                                .map(|j| {
                                    let w = if *interleaved {
                                        if j % 3 == 1 && (j as u32) < 3 * sections[1] {
                                            1
                                        } else if j % 3 == 2 && (j as u32) < 3 * sections[2] {
                                            2
                                        } else {
                                            0
                                        }
                                    } else {
                                        let (s0, s1) = (sections[0] as usize, sections[1] as usize);
                                        if j < s0 {
                                            0
                                        } else if j < s0 + s1 {
                                            1
                                        } else {
                                            2
                                        }
                                    };
                                    (w == which) as u8 as f32
                                })
                                .collect();
                            let col = self.slice(&p3, &[0, which], &[n, which + 1]);
                            let col = self.reshape(&col, &[n]);
                            let col = self.bcast(&col, &[n, half], &[0]);
                            let m = self.dense(&mask, &[half]);
                            let m = self.bcast(&m, &[n, half], &[1]);
                            let term = self.mul(&col, &m);
                            acc = Some(match acc {
                                Some(a) => self.add(&a, &term),
                                None => term,
                            });
                        }
                        acc.expect("three sections")
                    }
                    _ => {
                        let p = self.reshape(&pos, &[n]);
                        self.bcast(&p, &[n, half], &[0])
                    }
                };
                let f = self.bcast(&inv, &[n, half], &[1]);
                let theta = self.mul(&p, &f);
                let cos = self.un("cosine", &theta);
                let cos = self.mulc(&cos, *mscale as f64);
                let sin = self.un("sine", &theta);
                let sin = self.mulc(&sin, *mscale as f64);
                let cos = self.bcast(&cos, &[n, heads, half], &[0, 2]);
                let sin = self.bcast(&sin, &[n, heads, half], &[0, 2]);
                let x3 = self.reshape(&x, &[n, heads, hd]);
                let adjacent = match variant {
                    RopeVariant::Interleaved => true,
                    RopeVariant::HalfSplit => false,
                    RopeVariant::MRope { pairs, .. } => *pairs == PairLayout::Adjacent,
                };
                let rest = self.slice(&x3, &[0, 0, 2 * half], &[n, heads, hd]);
                let (x0, x1) = if adjacent {
                    let r = self.slice(&x3, &[0, 0, 0], &[n, heads, 2 * half]);
                    let r = self.reshape(&r, &[n, heads, half, 2]);
                    let a = self.slice(&r, &[0, 0, 0, 0], &[n, heads, half, 1]);
                    let b = self.slice(&r, &[0, 0, 0, 1], &[n, heads, half, 2]);
                    (self.reshape(&a, &[n, heads, half]), self.reshape(&b, &[n, heads, half]))
                } else {
                    (
                        self.slice(&x3, &[0, 0, 0], &[n, heads, half]),
                        self.slice(&x3, &[0, 0, half], &[n, heads, 2 * half]),
                    )
                };
                let (ac, bs) = (self.mul(&x0, &cos), self.mul(&x1, &sin));
                let o0 = self.sub(&ac, &bs);
                let (as_, bc) = (self.mul(&x0, &sin), self.mul(&x1, &cos));
                let o1 = self.add(&as_, &bc);
                let rot = if adjacent {
                    let a = self.reshape(&o0, &[n, heads, half, 1]);
                    let b = self.reshape(&o1, &[n, heads, half, 1]);
                    let c = self.concat(&[a, b], 3);
                    self.reshape(&c, &[n, heads, 2 * half])
                } else {
                    self.concat(&[o0, o1], 2)
                };
                let out = self.concat(&[rot, rest], 2);
                Ok(self.reshape(&out, &[n, total]))
            }
            Op::RelPosBias {
                buckets,
                max_distance,
                bidirectional,
                heads,
            } => {
                let table = self.flt(ins[0])?;
                let (qp, kp) = (self.idx(ins[1])?, self.idx(ins[2])?);
                let (n, m, h) = (numel(&qp.shape), numel(&kp.shape), *heads);
                let nb = if *bidirectional { buckets / 2 } else { *buckets };
                if nb < 2 || *max_distance <= nb / 2 || numel(&table.shape) != buckets * h {
                    return Err("RelPosBias table/buckets/max_distance mismatch".into());
                }
                let qp = self.reshape(&qp, &[n]);
                let kp = self.reshape(&kp, &[m]);
                let qb = self.bcast(&qp, &[n, m], &[0]);
                let kb = self.bcast(&kp, &[n, m], &[1]);
                let rel = self.sub(&kb, &qb);
                let zero = self.k(0.0, &rel);
                let (ret, dist) = if *bidirectional {
                    let gt = self.cmpc("GT", &rel, 0.0);
                    let nbk = self.k(nb as f64, &rel);
                    (self.sel(&gt, &nbk, &zero), self.un("abs", &rel))
                } else {
                    let nr = self.un("negate", &rel);
                    (zero.clone(), self.bin("maximum", &nr, &zero))
                };
                let exact = nb / 2;
                let scale = (*max_distance as f64 / exact as f64).ln() as f32;
                let df = self.convert(&dist, Ty::F32);
                let ek = self.k(exact as f64, &df);
                let l = self.div(&df, &ek);
                let l = self.un("log", &l);
                let sk = self.k(scale as f64, &l);
                let l = self.div(&l, &sk);
                let l = self.mulc(&l, (nb - exact) as f64);
                let large = self.convert(&l, Ty::I32);
                let large = self.addc(&large, exact as f64);
                let cap = self.k((nb - 1) as f64, &large);
                let large = self.bin("minimum", &large, &cap);
                let small = self.cmpc("LT", &dist, exact as f64);
                let pick = self.sel(&small, &dist, &large);
                let b = self.add(&ret, &pick);
                // out[head, q, k] = table[b * heads + head]
                let b3 = self.bcast(&b, &[h, n, m], &[1, 2]);
                let hk = self.k(h as f64, &b3);
                let b3 = self.mul(&b3, &hk);
                let hi = self.iota(&[h, n, m], 0);
                let at = self.add(&b3, &hi);
                let at = self.reshape(&at, &[h * n * m]);
                let tf = self.reshape(&table, &[buckets * h, 1]);
                Ok(self.gather_rows(&tf, &at))
            }
            Op::StateReorder => {
                let st = self.flt(ins[0])?;
                let parent = self.idx(ins[1])?;
                let batch = numel(&parent.shape);
                let st = self.rows(&st, numel(&st.shape) / batch.max(1))?;
                let parent = self.reshape(&parent, &[batch]);
                let rowsv = self.gather_rows(&st, &parent);
                let ok = self.cmpc("LT", &parent, batch as f64);
                let ok = self.bcast(&ok, &rowsv.shape.clone(), &[0]);
                let z = self.k(0.0, &rowsv);
                Ok(self.sel(&ok, &rowsv, &z))
            }
            Op::StateWrite => {
                let st = self.flt(ins[0])?;
                let w = *shape_of(0).last().unwrap_or(&0);
                let st = self.rows(&st, w)?;
                let rowsv = self.flt(ins[1])?;
                let rowsv = self.rows(&rowsv, w)?;
                let p = self.idx(ins[2])?;
                let p = self.reshape(&p, &[numel(&p.shape)]);
                let p0 = self.elem(&p, 0);
                let z = self.ki(0, &[]);
                let new = self.dus(&st, &rowsv, &[p0.clone(), z]);
                self.set_state(ins[0], new)?;
                let m = self.k(rowsv.shape[0] as f64, &p0);
                let end = self.add(&p0, &m);
                Ok(self.reshape(&end, &[1]))
            }
            Op::QsaSelect { ratio, top_blocks } | Op::QsaBias { ratio, top_blocks } => {
                let score = self.flt(ins[0])?;
                let pos = self.idx(ins[1])?;
                let n = numel(&pos.shape);
                let nb = numel(&score.shape) / n.max(1);
                let score = self.reshape(&score, &[n, nb]);
                let pos = self.reshape(&pos, &[n]);
                let (ratio, tb) = (*ratio, *top_blocks);
                let kk = tb * ratio + ratio - 1;
                let vis = self.addc(&pos, 1.0);
                let rk = self.k(ratio as f64, &vis);
                let blocks = self.div(&vis, &rk);
                let vis2 = self.bcast(&vis, &[n, kk], &[0]);
                let j = self.iota(&[n, kk], 1);
                // Fewer complete blocks than top_blocks: every visible key in order.
                let simple_ok = self.cmp("LT", &j, &vis2);
                let (ids, ok) = if tb >= nb {
                    (j, simple_ok)
                } else {
                    let bi = self.iota(&[n, nb], 1);
                    let bl = self.bcast(&blocks, &[n, nb], &[0]);
                    let live = self.cmp("LT", &bi, &bl);
                    let ninf = self.k(f64::NEG_INFINITY, &score);
                    let ms = self.sel(&live, &score, &ninf);
                    let order = self.sort_by(&ms, &bi, 1, "GT");
                    let best = self.slice(&order, &[0, 0], &[n, tb]);
                    let best = self.sort_by(&best, &best, 1, "LT");
                    let b3 = self.bcast(&best, &[n, tb, ratio], &[0, 1]);
                    let rk3 = self.k(ratio as f64, &b3);
                    let b3 = self.mul(&b3, &rk3);
                    let off = self.iota(&[n, tb, ratio], 2);
                    let sel_keys = self.add(&b3, &off);
                    let sel_keys = self.reshape(&sel_keys, &[n, tb * ratio]);
                    let tail = ratio - 1;
                    let (keys, valid) = if tail > 0 {
                        let bs = self.mul(&blocks, &rk);
                        let bs = self.bcast(&bs, &[n, tail], &[0]);
                        let t = self.iota(&[n, tail], 1);
                        let tk = self.add(&bs, &t);
                        let v2 = self.bcast(&vis, &[n, tail], &[0]);
                        let tv = self.cmp("LT", &tk, &v2);
                        let keys = self.concat(&[sel_keys.clone(), tk], 1);
                        let one = self.k(1.0, &sel_keys);
                        let sv = self.cmp("EQ", &one, &one);
                        (keys, self.concat(&[sv, tv], 1))
                    } else {
                        let one = self.k(1.0, &sel_keys);
                        let sv = self.cmp("EQ", &one, &one);
                        (sel_keys, sv)
                    };
                    let tbk = self.k(tb as f64, &blocks);
                    let few = self.cmp("LE", &blocks, &tbk);
                    let few = self.bcast(&few, &[n, kk], &[0]);
                    (self.sel(&few, &j, &keys), self.sel(&few, &simple_ok, &valid))
                };
                if matches!(node.op, Op::QsaSelect { .. }) {
                    let z = self.k(0.0, &ids);
                    Ok(self.sel(&ok, &ids, &z))
                } else {
                    let z = self.kf(0.0, &[n, kk]);
                    let ninf = self.kf(f32::NEG_INFINITY, &[n, kk]);
                    Ok(self.sel(&ok, &z, &ninf))
                }
            }
            Op::CachedAttention {
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
                let (h, kvh, hd, vhd) = (*heads, *kv_heads, *head_dim, *v_head_dim);
                let q = self.flt(ins[0])?;
                let nq = numel(&q.shape) / (h * hd).max(1);
                let q = self.reshape(&q, &[nq, h, hd]);
                let kn = self.flt(ins[1])?;
                let vn = self.flt(ins[2])?;
                let class = match ins.get(3).map(|&v| &g.value(v).kind) {
                    Some(ValueKind::State { name, .. }) if ins.len() >= 6 => {
                        self.paged.get(name).copied()
                    }
                    _ => None,
                };
                if let Some(c) = class {
                    // Page windows in stored-row space (`crate::kv`): written rows scatter into
                    // their slots; every slot holds its page row, masked by the stored row it
                    // stands for.
                    let (slot, wslot, base, ring) = self.classes[c].clone();
                    let p = self.idx(ins[5])?;
                    let p = self.reshape(&p, &[numel(&p.shape)]);
                    let p0 = self.elem(&p, 0);
                    let ks = self.flt(ins[3])?;
                    let vs = self.flt(ins[4])?;
                    let mut ks = self.rows(&ks, kvh * hd)?;
                    let mut vs = self.rows(&vs, kvh * vhd)?;
                    if *write {
                        let kn = self.rows(&kn, kvh * hd)?;
                        let vn = self.rows(&vn, kvh * vhd)?;
                        ks = self.scatter_rows(&ks, &wslot, &kn);
                        vs = self.scatter_rows(&vs, &wslot, &vn);
                        self.set_state(ins[3], ks.clone())?;
                        self.set_state(ins[4], vs.clone())?;
                    }
                    let rows = ks.shape[0];
                    let slot_b = self.bcast(&slot, &[nq, rows], &[1]);
                    let qi = self.iota(&[nq, rows], 0);
                    let bb = self.bcast(&base, &[nq, rows], &[]);
                    let cur = self.add(&bb, &qi);
                    // The stored row slot `j` stands for: `j` itself, or in a ring of `R` rows
                    // (slot = row mod R, as the host addresses it) the one row `≡ j` in the
                    // query's `[lo, lo + R)`, `lo` the window start.
                    let t = if ring == 0 {
                        slot_b.clone()
                    } else {
                        let lo = match window {
                            Some(w) => {
                                let l = self.addc(&cur, 1.0 - *w as f64);
                                let z = self.k(0.0, &l);
                                self.bin("maximum", &l, &z)
                            }
                            None => self.k(0.0, &cur),
                        };
                        let rk = self.k(ring as f64, &lo);
                        let lq = self.div(&lo, &rk);
                        let lq = self.mul(&lq, &rk);
                        let lr = self.sub(&lo, &lq);
                        let d = self.sub(&slot_b, &lr);
                        let neg = self.cmpc("LT", &d, 0.0);
                        let dr = self.add(&d, &rk);
                        let d = self.sel(&neg, &dr, &d);
                        self.add(&lo, &d)
                    };
                    let shift = self.sub(&p0, &base);
                    let vis = match ins.get(6) {
                        Some(&vi) => {
                            let v = self.idx(vi)?;
                            let v = self.reshape(&v, &[nq]);
                            let sh = self.bcast(&shift, &[nq], &[]);
                            Some(self.sub(&v, &sh))
                        }
                        None => None,
                    };
                    let hi = match (*causal, &vis) {
                        (true, Some(v)) => self.bcast(v, &[nq, rows], &[0]),
                        (true, None) => self.addc(&cur, 1.0),
                        (false, _) => {
                            let mut all = self.addc(&base, nq as f64);
                            if let Some(v) = &vis {
                                let vm = self.reduce(v, &[0], Red::Max);
                                all = self.bin("maximum", &all, &vm);
                            }
                            self.bcast(&all, &[nq, rows], &[])
                        }
                    };
                    let valid = self.cmpc("GE", &slot_b, 0.0);
                    let below = self.cmp("LT", &t, &hi);
                    let mut mask = self.and(&valid, &below);
                    if let Some(w) = window {
                        let tw = self.addc(&t, *w as f64);
                        let c = self.cmp("GT", &tw, &cur);
                        mask = self.and(&mask, &c);
                    }
                    let k3 = self.reshape(&ks, &[rows, kvh, hd]);
                    let v3 = self.reshape(&vs, &[rows, kvh, vhd]);
                    return Ok(self.attend(&q, &k3, &v3, &mask, *scale, *soft_cap, None));
                }
                let (k_all, v_all, pos) = if ins.len() >= 6 {
                    let p = self.idx(ins[5])?;
                    let p = self.reshape(&p, &[numel(&p.shape)]);
                    let p0 = self.elem(&p, 0);
                    let ks = self.flt(ins[3])?;
                    let vs = self.flt(ins[4])?;
                    let ks = self.rows(&ks, kvh * hd)?;
                    let vs = self.rows(&vs, kvh * vhd)?;
                    if *write {
                        let kn = self.rows(&kn, kvh * hd)?;
                        let vn = self.rows(&vn, kvh * vhd)?;
                        let z = self.ki(0, &[]);
                        let ks2 = self.dus(&ks, &kn, &[p0.clone(), z.clone()]);
                        let vs2 = self.dus(&vs, &vn, &[p0.clone(), z]);
                        self.set_state(ins[3], ks2.clone())?;
                        self.set_state(ins[4], vs2.clone())?;
                        (ks2, vs2, p0)
                    } else {
                        (ks, vs, p0)
                    }
                } else {
                    let kn = self.rows(&kn, kvh * hd)?;
                    let vn = self.rows(&vn, kvh * vhd)?;
                    (kn, vn, self.ki(0, &[]))
                };
                let rows = k_all.shape[0];
                let qi = self.iota(&[nq, rows], 0);
                let ki = self.iota(&[nq, rows], 1);
                let pb = self.bcast(&pos, &[nq, rows], &[]);
                let cur = self.add(&pb, &qi);
                // nk = min(max(pos + nq, max(visible_end)), rows)
                let mut nk = self.addc(&pos, nq as f64);
                let vis = match ins.get(6) {
                    Some(&vi) => {
                        let v = self.idx(vi)?;
                        let v = self.reshape(&v, &[nq]);
                        let vm = self.reduce(&v, &[0], Red::Max);
                        nk = self.bin("maximum", &nk, &vm);
                        Some(v)
                    }
                    None => None,
                };
                let rk = self.k(rows as f64, &nk);
                let nk = self.bin("minimum", &nk, &rk);
                let nkb = self.bcast(&nk, &[nq, rows], &[]);
                let mut mask = self.cmp("LT", &ki, &nkb);
                if *causal {
                    let lim = match &vis {
                        Some(v) => self.bcast(v, &[nq, rows], &[0]),
                        None => self.addc(&cur, 1.0),
                    };
                    let c = self.cmp("LT", &ki, &lim);
                    mask = self.and(&mask, &c);
                }
                if let Some(w) = window {
                    let kw = self.addc(&ki, *w as f64);
                    let c = self.cmp("GT", &kw, &cur);
                    mask = self.and(&mask, &c);
                }
                let k3 = self.reshape(&k_all, &[rows, kvh, hd]);
                let v3 = self.reshape(&v_all, &[rows, kvh, vhd]);
                Ok(self.attend(&q, &k3, &v3, &mask, *scale, *soft_cap, None))
            }
            Op::Attention {
                heads,
                kv_heads,
                causal,
                window,
                soft_cap,
                scale,
            } => {
                let q = self.flt(ins[0])?;
                let hd = *shape_of(0).last().unwrap_or(&0);
                let (h, kvh) = (*heads, *kv_heads);
                if h == 0 || kvh == 0 || h % kvh != 0 || hd % h != 0 {
                    return Err(format!("q/k/v do not fit {h} heads over {kvh} kv heads"));
                }
                let d = hd / h;
                let nq = numel(&q.shape) / hd;
                let q = self.reshape(&q, &[nq, h, d]);
                let (k, v) = (self.flt(ins[1])?, self.flt(ins[2])?);
                let nk = numel(&k.shape) / (kvh * d);
                let k = self.reshape(&k, &[nk, kvh, d]);
                let v = self.reshape(&v, &[nk, kvh, d]);
                let bias = match ins.get(3) {
                    Some(&bi) => {
                        let b = self.flt(bi)?;
                        let bn = numel(&b.shape);
                        Some(if bn == nk {
                            let b = self.reshape(&b, &[nk]);
                            self.bcast(&b, &[h, nq, nk], &[2])
                        } else if bn == nq * nk {
                            let b = self.reshape(&b, &[nq, nk]);
                            self.bcast(&b, &[h, nq, nk], &[1, 2])
                        } else if bn == h * nq * nk {
                            self.reshape(&b, &[h, nq, nk])
                        } else {
                            return Err(format!("bias {:?} does not fit [{h}, {nq}, {nk}]", b.shape));
                        })
                    }
                    None => None,
                };
                let qi = self.iota(&[nq, nk], 0);
                let ki = self.iota(&[nq, nk], 1);
                let mask = match (causal, window) {
                    (true, w) => {
                        let q1 = self.addc(&qi, 1.0);
                        let m = self.cmp("LT", &ki, &q1);
                        match w {
                            Some(w) => {
                                let kw = self.addc(&ki, *w as f64);
                                let c = self.cmp("GT", &kw, &qi);
                                self.and(&m, &c)
                            }
                            None => m,
                        }
                    }
                    (false, None) => self.cmp("EQ", &ki, &ki),
                    (false, Some(w)) => {
                        let lo = self.addc(&qi, -((w / 2) as f64));
                        let hi = self.addc(&qi, (w / 2 + 1) as f64);
                        let a = self.cmp("GE", &ki, &lo);
                        let b = self.cmp("LT", &ki, &hi);
                        self.and(&a, &b)
                    }
                };
                let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
                Ok(self.attend(&q, &k, &v, &mask, scale, *soft_cap, bias.as_ref()))
            }
            Op::CausalConv1dCached { kernel, dilation } => {
                let (k, d) = (*kernel, *dilation);
                let w = self.flt(ins[1])?;
                let c = numel(&w.shape) / k.max(1);
                let w = self.reshape(&w, &[c, k]);
                let x = self.flt(ins[0])?;
                let x = self.rows(&x, c)?;
                let n = x.shape[0];
                let hist = (k - 1) * d;
                let window = if hist == 0 {
                    x.clone()
                } else {
                    let st = match ins.get(2) {
                        Some(&si) => {
                            let s = self.flt(si)?;
                            (numel(&s.shape) == hist * c).then(|| self.reshape(&s, &[hist, c]))
                        }
                        None => None,
                    };
                    let st = match st {
                        Some(s) => s,
                        None => self.kf(0.0, &[hist, c]),
                    };
                    self.concat(&[st, x.clone()], 0)
                };
                let mut acc: Option<V> = None;
                for tap in 0..k {
                    let lo = hist - (k - 1 - tap) * d;
                    let xs = self.slice(&window, &[lo, 0], &[lo + n, c]);
                    let wc = self.slice(&w, &[0, tap], &[c, tap + 1]);
                    let wc = self.reshape(&wc, &[c]);
                    let wc = self.bcast(&wc, &[n, c], &[1]);
                    let t = self.mul(&xs, &wc);
                    acc = Some(match acc {
                        Some(a) => self.add(&a, &t),
                        None => t,
                    });
                }
                if hist > 0 {
                    if let Some(&si) = ins.get(2) {
                        let ns = self.slice(&window, &[n, 0], &[n + hist, c]);
                        self.set_state(si, ns)?;
                    }
                }
                Ok(acc.expect("kernel >= 1"))
            }
            Op::DeltaRule {
                key_heads,
                val_heads,
                key_dim,
                val_dim,
            } => self.delta_rule(node, *key_heads, *val_heads, *key_dim, *val_dim),
            Op::Gather { axis } => {
                let x = self.flt(ins[0])?;
                let ids = self.idx(ins[1])?;
                let ni = numel(&ids.shape);
                let ids = self.reshape(&ids, &[ni]);
                let xs = x.shape.clone();
                let ax = Self::norm_axis(*axis, xs.len())?;
                let (outer, dim, inner) = (
                    numel(&xs[..ax]),
                    xs[ax],
                    numel(&xs[ax + 1..]),
                );
                let x3 = self.reshape(&x, &[outer, dim, inner]);
                let xt = self.transpose(&x3, &[1, 0, 2]);
                let xt = self.reshape(&xt, &[dim, outer * inner]);
                let gth = self.gather_rows(&xt, &ids);
                let gth = self.reshape(&gth, &[ni, outer, inner]);
                Ok(self.transpose(&gth, &[1, 0, 2]))
            }
            Op::Scatter { axis } => {
                let base = self.flt(ins[0])?;
                let ids = self.idx(ins[1])?;
                let upd = self.flt(ins[2])?;
                let ni = numel(&ids.shape);
                let ids = self.reshape(&ids, &[ni]);
                let bs = base.shape.clone();
                let ax = Self::norm_axis(*axis, bs.len())?;
                let (outer, dim, inner) = (numel(&bs[..ax]), bs[ax], numel(&bs[ax + 1..]));
                let b3 = self.reshape(&base, &[outer, dim, inner]);
                let bt = self.transpose(&b3, &[1, 0, 2]);
                let bt = self.reshape(&bt, &[dim, outer * inner]);
                let u3 = self.reshape(&upd, &[outer, ni, inner]);
                let ut = self.transpose(&u3, &[1, 0, 2]);
                let ut = self.reshape(&ut, &[ni, outer * inner]);
                let sc = self.scatter_rows(&bt, &ids, &ut);
                let sc = self.reshape(&sc, &[dim, outer, inner]);
                Ok(self.transpose(&sc, &[1, 0, 2]))
            }
            Op::Slice { axis, start, end } => {
                let x = self.val(ins[0])?;
                let xs = x.shape.clone();
                let ax = Self::norm_axis(*axis, xs.len())?;
                let dim = xs[ax] as i64;
                let clamp = |v: i64| if v < 0 { (dim + v).max(0) } else { v.min(dim) } as usize;
                let (s, e) = (clamp(*start), clamp(*end));
                Ok(self.slice_axis(&x, ax, s, e.max(s)))
            }
            Op::Pad { padding, value } => {
                let x = self.flt(ins[0])?;
                let (r, k) = (x.shape.len(), padding.len() / 2);
                if padding.len() % 2 != 0 || k > r {
                    return Err(format!("Pad {padding:?} does not fit {:?}", x.shape));
                }
                let lo: Vec<usize> = (0..r)
                    .map(|d| if d + k >= r { padding[2 * (d + k - r)] } else { 0 })
                    .collect();
                let hi: Vec<usize> = (0..r)
                    .map(|d| if d + k >= r { padding[2 * (d + k - r) + 1] } else { 0 })
                    .collect();
                let shape: Vec<usize> = (0..r).map(|d| x.shape[d] + lo[d] + hi[d]).collect();
                let pv = self.kf(*value, &[]);
                let xt = x.t();
                Ok(self.op(shape, Ty::F32, |t| {
                    format!(
                        "stablehlo.pad {}, {}, low = {}, high = {}, interior = {} : ({xt}, tensor<f32>) -> {t}",
                        x.name,
                        pv.name,
                        list(&lo),
                        list(&hi),
                        list(&vec![0; r])
                    )
                }))
            }
            Op::AvgPool2d { kernel, stride } => {
                let x = self.flt(ins[0])?;
                let s = x.shape.clone();
                let (c, h, w) = if s.len() >= 3 {
                    (numel(&s[..s.len() - 2]), s[s.len() - 2], s[s.len() - 1])
                } else {
                    (1, 1, numel(&s))
                };
                let x = self.reshape(&x, &[c, h, w]);
                let (oh, ow) = ((h - kernel[0]) / stride[0] + 1, (w - kernel[1]) / stride[1] + 1);
                let init = self.kf(0.0, &[]);
                self.n += 1;
                let tag = self.n;
                let xt = x.t();
                let sum = self.op(vec![c, oh, ow], Ty::F32, |t| {
                    format!(
                        "\"stablehlo.reduce_window\"({}, {}) ({{\n    ^bb0(%u{tag}a: tensor<f32>, %u{tag}b: tensor<f32>):\n      %u{tag}s = stablehlo.add %u{tag}a, %u{tag}b : tensor<f32>\n      stablehlo.return %u{tag}s : tensor<f32>\n    }}) {{window_dimensions = array<i64: 1, {}, {}>, window_strides = array<i64: 1, {}, {}>}} : ({xt}, tensor<f32>) -> {t}",
                        x.name, init.name, kernel[0], kernel[1], stride[0], stride[1]
                    )
                });
                Ok(self.mulc(&sum, 1.0 / (kernel[0] * kernel[1]) as f64))
            }
            Op::Upsample { scale, mode } => {
                let x = self.flt(ins[0])?;
                let inp = x.shape.clone();
                let nsp = crate::graph::cpu::upsample_nsp(&inp, &decl, *scale)?;
                let r = inp.len();
                let mut cur = x;
                // Separable: each scaled axis is a contraction with its [out, in] tap matrix.
                for axis in r - nsp..r {
                    let n = inp[axis];
                    let o = n * scale;
                    let mut m = vec![0.0f32; o * n];
                    for (oi, row) in m.chunks_exact_mut(n).enumerate() {
                        for (i, wt) in crate::graph::cpu::upsample_taps(oi, n, *scale, *mode) {
                            row[i] += wt;
                        }
                    }
                    let mat = self.dense(&m, &[o, n]);
                    // `cur` as [a, n, b]; the last axis is a plain row contraction, an inner one a
                    // batched [a, o, n] · [a, n, b] (no transposes: jax-metal mis-infers them here).
                    let (a, b) = (numel(&cur.shape[..axis]), numel(&cur.shape[axis + 1..]));
                    let mut out = cur.shape.clone();
                    out[axis] = o;
                    let y = if b == 1 {
                        let x2 = self.reshape(&cur, &[a, n]);
                        self.dot(&x2, &mat, (&[], &[]), (&[1], &[1]))
                    } else {
                        let x3 = self.reshape(&cur, &[a, n, b]);
                        let m3 = self.bcast(&mat, &[a, o, n], &[1, 2]);
                        self.dot(&m3, &x3, (&[0], &[0]), (&[2], &[1]))
                    };
                    cur = self.reshape(&y, &out);
                }
                Ok(cur)
            }
            Op::Conv1d {
                stride,
                padding,
                dilation,
                groups,
            } => self.conv(node, 1, &[*stride], &[[padding[0], padding[1]]], &[*dilation], *groups),
            Op::Conv2d {
                stride,
                padding: p,
                dilation,
                groups,
            } => self.conv(node, 2, stride, &[[p[0], p[2]], [p[1], p[3]]], dilation, *groups),
            Op::Conv3d {
                stride,
                padding: p,
                dilation,
                groups,
            } => self.conv(
                node,
                3,
                stride,
                &[[p[0], p[3]], [p[1], p[4]], [p[2], p[5]]],
                dilation,
                *groups,
            ),
            Op::LayerNorm { eps } => {
                let gamma = self.flt(ins[1])?;
                let d = numel(&gamma.shape);
                let gamma = self.reshape(&gamma, &[d]);
                let x = self.flt(ins[0])?;
                let x = self.rows(&x, d)?;
                let shape = x.shape.clone();
                let s = self.reduce(&x, &[1], Red::Add);
                let dk = self.k(d as f64, &s);
                let mean = self.div(&s, &dk);
                let mean = self.bcast(&mean, &shape, &[0]);
                let xc = self.sub(&x, &mean);
                let sq = self.mul(&xc, &xc);
                let v = self.reduce(&sq, &[1], Red::Add);
                let v = self.div(&v, &dk);
                let v = self.addc(&v, *eps as f64);
                let sd = self.un("sqrt", &v);
                let one = self.k(1.0, &sd);
                let inv = self.div(&one, &sd);
                let inv = self.bcast(&inv, &shape, &[0]);
                let y = self.mul(&xc, &inv);
                let gb = self.bcast(&gamma, &shape, &[1]);
                let y = self.mul(&y, &gb);
                match ins.get(2) {
                    Some(&bi) => {
                        let b = self.flt(bi)?;
                        let b = self.reshape(&b, &[d]);
                        let b = self.bcast(&b, &shape, &[1]);
                        Ok(self.add(&y, &b))
                    }
                    None => Ok(y),
                }
            }
            Op::GroupNorm { .. } | Op::InstanceNorm { .. } => {
                let (gm, bt) = (self.flt(ins[1])?, self.flt(ins[2])?);
                let c = numel(&gm.shape);
                let (groups, eps) = match node.op {
                    Op::GroupNorm { groups, eps } => (groups, eps),
                    Op::InstanceNorm { eps } => (c, eps),
                    _ => unreachable!(),
                };
                if c == 0 || numel(&bt.shape) != c || groups == 0 || c % groups != 0 {
                    return Err(format!("gamma/beta of {c} do not fit {groups} groups"));
                }
                let x = self.flt(ins[0])?;
                let (n, hw) = crate::graph::cpu::channel_layout(&x.shape, c)?;
                let len = c / groups * hw;
                let xg = self.reshape(&x, &[n, groups, len]);
                let shape = xg.shape.clone();
                let s = self.reduce(&xg, &[2], Red::Add);
                let lk = self.k(len as f64, &s);
                let mean = self.div(&s, &lk);
                let mean = self.bcast(&mean, &shape, &[0, 1]);
                let xc = self.sub(&xg, &mean);
                let sq = self.mul(&xc, &xc);
                let v = self.reduce(&sq, &[2], Red::Add);
                let v = self.div(&v, &lk);
                let v = self.addc(&v, eps as f64);
                let sd = self.un("sqrt", &v);
                let one = self.k(1.0, &sd);
                let inv = self.div(&one, &sd);
                let inv = self.bcast(&inv, &shape, &[0, 1]);
                let y = self.mul(&xc, &inv);
                let y = self.reshape(&y, &[n, c, hw]);
                let gm = self.reshape(&gm, &[c]);
                let bt = self.reshape(&bt, &[c]);
                let gb = self.bcast(&gm, &[n, c, hw], &[1]);
                let bb = self.bcast(&bt, &[n, c, hw], &[1]);
                let y = self.mul(&y, &gb);
                Ok(self.add(&y, &bb))
            }
            Op::Reduce { kind, axes, .. } => {
                let x = self.flt(ins[0])?;
                let r = x.shape.len();
                if axes.is_empty() {
                    return Err("Reduce with no axes is ambiguous".into());
                }
                let mut dims: Vec<usize> = Vec::new();
                for &a in axes {
                    let d = Self::norm_axis(a, r)?;
                    if dims.contains(&d) {
                        return Err(format!("axis {a} repeats"));
                    }
                    dims.push(d);
                }
                dims.sort_unstable();
                let how = match kind {
                    ReduceKind::Max => Red::Max,
                    ReduceKind::Min => Red::Min,
                    _ => Red::Add,
                };
                let out = self.reduce(&x, &dims, how);
                Ok(if *kind == ReduceKind::Mean {
                    let cnt: usize = dims.iter().map(|&d| x.shape[d]).product();
                    let ck = self.k(cnt as f64, &out);
                    self.div(&out, &ck)
                } else {
                    out
                })
            }
            Op::MeanPool => {
                let x = self.flt(ins[0])?;
                let d = *x.shape.last().unwrap_or(&0);
                let x = self.rows(&x, d)?;
                let n = x.shape[0];
                let m = self.idx(ins[1])?;
                let m = self.reshape(&m, &[n]);
                let on = self.cmpc("NE", &m, 0.0);
                let mf = self.convert(&on, Ty::F32);
                let mb = self.bcast(&mf, &[n, d], &[0]);
                let xm = self.mul(&x, &mb);
                let s = self.reduce(&xm, &[0], Red::Add);
                let cnt = self.reduce(&mf, &[0], Red::Add);
                let one = self.k(1.0, &cnt);
                let cnt = self.bin("maximum", &cnt, &one);
                let cnt = self.bcast(&cnt, &[d], &[]);
                Ok(self.div(&s, &cnt))
            }
            Op::L2Normalize => {
                let x = self.flt(ins[0])?;
                let all: Vec<usize> = (0..x.shape.len()).collect();
                let sq = self.mul(&x, &x);
                let s = self.reduce(&sq, &all, Red::Add);
                let s = self.un("sqrt", &s);
                let lo = self.k(1e-12, &s);
                let s = self.bin("maximum", &s, &lo);
                let s = self.bcast(&s, &x.shape.clone(), &[]);
                Ok(self.div(&x, &s))
            }
            Op::Reshape { .. } | Op::Squeeze { .. } | Op::Unsqueeze { .. } => {
                let x = self.val(ins[0])?;
                if numel(&x.shape) != numel(&decl) {
                    return Err("changes the element count".into());
                }
                Ok(x)
            }
            Op::Transpose { perm } => {
                let x = self.val(ins[0])?;
                if perm.len() != x.shape.len() {
                    return Err(format!("{perm:?} is not a permutation of rank {}", x.shape.len()));
                }
                Ok(self.transpose(&x, perm))
            }
            Op::Tile { reps } => {
                let x = self.val(ins[0])?;
                let r = x.shape.len().max(reps.len());
                let pad = |v: &[usize]| -> Vec<usize> {
                    std::iter::repeat_n(1, r - v.len()).chain(v.iter().copied()).collect()
                };
                let (ish, rp) = (pad(&x.shape), pad(reps));
                let x = self.reshape(&x, &ish);
                let mut big = Vec::with_capacity(2 * r);
                let mut dims = Vec::with_capacity(r);
                for d in 0..r {
                    big.push(rp[d]);
                    big.push(ish[d]);
                    dims.push(2 * d + 1);
                }
                let b = self.bcast(&x, &big, &dims);
                let out: Vec<usize> = (0..r).map(|d| rp[d] * ish[d]).collect();
                Ok(self.reshape(&b, &out))
            }
            Op::Concat { axis } => {
                let mut parts = Vec::with_capacity(ins.len());
                for &i in ins {
                    parts.push(self.val(i)?);
                }
                let ty = if parts.iter().all(|p| p.ty == Ty::I32) { Ty::I32 } else { Ty::F32 };
                let parts: Vec<V> = parts.iter().map(|p| self.cast(p, ty)).collect();
                let rank = parts[0].shape.len();
                let ax = Self::norm_axis(*axis, rank)?;
                Ok(self.concat(&parts, ax))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        node: &Node,
        nd: usize,
        stride: &[usize],
        pad: &[[usize; 2]],
        dil: &[usize],
        groups: usize,
    ) -> Result<V, String> {
        let ins = &node.inputs;
        let w = self.flt(ins[1])?;
        let x = self.flt(ins[0])?;
        if w.shape.len() != nd + 2 || x.shape.len() < nd + 1 {
            return Err(format!("conv shapes {:?} / {:?} do not fit {nd}-d", x.shape, w.shape));
        }
        let xs = x.shape.clone();
        let c_in = xs[xs.len() - nd - 1];
        let batch = numel(&xs[..xs.len() - nd - 1]);
        let spatial = &xs[xs.len() - nd..];
        let c_out = w.shape[0];
        let x = self.reshape(&x, &[vec![batch, c_in], spatial.to_vec()].concat());
        let mut out = vec![batch, c_out];
        for a in 0..nd {
            let span = dil[a] * (w.shape[2 + a] - 1) + 1;
            let padded = spatial[a] + pad[a][0] + pad[a][1];
            if padded < span {
                return Err("conv kernel larger than padded input".into());
            }
            out.push((padded - span) / stride[a] + 1);
        }
        let sp: Vec<String> = (0..nd).map(|i| i.to_string()).collect();
        let sp = sp.join(", ");
        let pads: Vec<String> = pad.iter().map(|p| format!("[{}, {}]", p[0], p[1])).collect();
        let (xt, wt) = (x.t(), w.t());
        let y = self.op(out.clone(), Ty::F32, |t| {
            format!(
                "stablehlo.convolution({}, {}) dim_numbers = [b, f, {sp}]x[o, i, {sp}]->[b, f, {sp}], window = {{stride = {}, pad = [{}], rhs_dilate = {}}} {{batch_group_count = 1 : i64, feature_group_count = {groups} : i64}} : ({xt}, {wt}) -> {t}",
                x.name,
                w.name,
                list(stride),
                pads.join(", "),
                list(dil)
            )
        });
        match ins.get(2) {
            Some(&bi) => {
                let b = self.flt(bi)?;
                let b = self.reshape(&b, &[c_out]);
                let b = self.bcast(&b, &out, &[1]);
                Ok(self.add(&y, &b))
            }
            None => Ok(y),
        }
    }

    /// Gated DeltaNet recurrence: precomputed normalized q/k, then a `while` over rows carrying the
    /// state and the output.
    fn delta_rule(
        &mut self,
        node: &Node,
        kh: usize,
        vh: usize,
        kd: usize,
        vd: usize,
    ) -> Result<V, String> {
        let ins = &node.inputs;
        let rep = vh / kh.max(1);
        let q = self.flt(ins[0])?;
        let n = numel(&q.shape) / (kh * kd).max(1);
        let (k, v) = (self.flt(ins[1])?, self.flt(ins[2])?);
        let (beta, ld) = (self.flt(ins[3])?, self.flt(ins[4])?);
        if numel(&beta.shape) != n * vh || numel(&ld.shape) != n * vh {
            return Err(format!("beta/log_decay must be [{n}, {vh}]"));
        }
        let beta = self.reshape(&beta, &[n, vh]);
        let decay = self.reshape(&ld, &[n, vh]);
        let decay = self.un("exponential", &decay);
        // L2-normalize each head row, expand key heads to value heads.
        let norm = |e: &mut Self, x: &V, s: f64| -> V {
            let x = e.reshape(x, &[n, kh, kd]);
            let sq = e.mul(&x, &x);
            let ss = e.reduce(&sq, &[2], Red::Add);
            let nn = e.un("sqrt", &ss);
            let lo = e.k(1e-12, &nn);
            let nn = e.bin("maximum", &nn, &lo);
            let sk = e.k(s, &nn);
            let f = e.div(&sk, &nn);
            let f = e.bcast(&f, &[n, kh, kd], &[0, 1]);
            let x = e.mul(&x, &f);
            let x = e.bcast(&x, &[n, kh, rep, kd], &[0, 1, 3]);
            e.reshape(&x, &[n, vh, kd])
        };
        let qn = norm(self, &q, 1.0 / (kd as f64).sqrt());
        let kn = norm(self, &k, 1.0);
        let v = self.reshape(&v, &[n, vh, vd]);
        let st_shape = [vh, kd, vd];
        let s0 = match ins.get(5) {
            Some(&si) => {
                let s = self.flt(si)?;
                if numel(&s.shape) == vh * kd * vd {
                    self.reshape(&s, &st_shape)
                } else {
                    self.kf(0.0, &st_shape)
                }
            }
            None => self.kf(0.0, &st_shape),
        };
        let o0 = self.kf(0.0, &[n, vh, vd]);
        let t0 = self.ki(0, &[]);
        // while (t < n): carry (t, S, out, q, k, v, beta, decay)
        let carry = [t0, s0, o0, qn, kn, v, beta, decay];
        self.n += 1;
        let tag = self.n;
        let r = format!("%t{tag}");
        let names: Vec<String> = (0..carry.len()).map(|i| format!("%c{tag}_{i}")).collect();
        let inits: Vec<String> = names
            .iter()
            .zip(&carry)
            .map(|(nm, c)| format!("{nm} = {}", c.name))
            .collect();
        let tys: Vec<String> = carry.iter().map(V::t).collect();
        let _ = writeln!(
            self.body,
            "    {r}:{} = stablehlo.while({}) : {}\n    cond {{",
            carry.len(),
            inits.join(", "),
            tys.join(", ")
        );
        let arg = |i: usize| V {
            name: names[i].clone(),
            shape: carry[i].shape.clone(),
            ty: carry[i].ty,
        };
        let nk = self.ki(n as i64, &[]);
        let c = self.cmp("LT", &arg(0), &nk);
        let _ = writeln!(self.body, "      stablehlo.return {} : tensor<i1>\n    }} do {{", c.name);
        let (t, s) = (arg(0), arg(1));
        let z = self.ki(0, &[]);
        let row = |e: &mut Self, x: &V, w: usize| -> V {
            let r = e.dslice(x, &[t.clone(), z.clone(), z.clone()], &[1, vh, w]);
            e.reshape(&r, &[vh, w])
        };
        let (qt, kt, vt) = (row(self, &arg(3), kd), row(self, &arg(4), kd), row(self, &arg(5), vd));
        let bt = self.dslice(&arg(6), &[t.clone(), z.clone()], &[1, vh]);
        let bt = self.reshape(&bt, &[vh]);
        let dt = self.dslice(&arg(7), &[t.clone(), z.clone()], &[1, vh]);
        let dt = self.reshape(&dt, &[vh]);
        let kv = self.dot(&s, &kt, (&[0], &[0]), (&[1], &[1])); // [vh, vd]
        let db = self.bcast(&dt, &[vh, vd], &[0]);
        let dkv = self.mul(&db, &kv);
        let diff = self.sub(&vt, &dkv);
        let bb = self.bcast(&bt, &[vh, vd], &[0]);
        let delta = self.mul(&bb, &diff);
        let ds = self.bcast(&dt, &st_shape, &[0]);
        let s_dec = self.mul(&ds, &s);
        let kb = self.bcast(&kt, &st_shape, &[0, 1]);
        let dlb = self.bcast(&delta, &st_shape, &[0, 2]);
        let outer = self.mul(&kb, &dlb);
        let s1 = self.add(&s_dec, &outer);
        let o = self.dot(&s1, &qt, (&[0], &[0]), (&[1], &[1])); // [vh, vd]
        let o = self.reshape(&o, &[1, vh, vd]);
        let out = self.dus(&arg(2), &o, &[t.clone(), z.clone(), z.clone()]);
        let t1 = self.addc(&t, 1.0);
        let mut next: Vec<&V> = vec![&t1, &s1, &out];
        let rest: Vec<V> = (3..carry.len()).map(arg).collect();
        next.extend(rest.iter());
        let nn: Vec<&str> = next.iter().map(|v| v.name.as_str()).collect();
        let _ = writeln!(
            self.body,
            "      stablehlo.return {} : {}\n    }}",
            nn.join(", "),
            tys.join(", ")
        );
        let res = |i: usize| V {
            name: format!("{r}#{i}"),
            shape: carry[i].shape.clone(),
            ty: carry[i].ty,
        };
        if let Some(&si) = ins.get(5) {
            self.set_state(si, res(1))?;
        }
        Ok(self.reshape(&res(2), &[n, vh * vd]))
    }
}
