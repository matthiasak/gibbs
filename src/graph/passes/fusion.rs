//! Subgraph fusion optimization passes for Graph IR.

use crate::graph::pattern::{GraphRewriter, MatchContext};
use crate::graph::{DType, Graph, Op, Tensor, ValueKind, Weights};
use crate::model::weights::Mat;

/// Concatenates 2D weight tensors along axis 0 (rows).
/// Supports F32, BF16, F16, and Mat.
pub fn concat_weight_tensors(tensors: &[&Tensor]) -> Option<Tensor> {
    match tensors.first()? {
        Tensor::Mat(_) => {
            let ms: Vec<&Mat> = tensors
                .iter()
                .map(|t| match t {
                    Tensor::Mat(m) => Some(m),
                    _ => None,
                })
                .collect::<Option<_>>()?;
            concat_mats(&ms)
        }
        Tensor::F32(_) => {
            let mut data = Vec::new();
            for t in tensors {
                let Tensor::F32(v) = t else { return None };
                data.extend_from_slice(v);
            }
            Some(Tensor::F32(data))
        }
        Tensor::BF16(_) => {
            let mut data = Vec::new();
            for t in tensors {
                let Tensor::BF16(v) = t else { return None };
                data.extend_from_slice(v);
            }
            Some(Tensor::BF16(data))
        }
        Tensor::F16(_) => {
            let mut data = Vec::new();
            for t in tensors {
                let Tensor::F16(v) = t else { return None };
                data.extend_from_slice(v);
            }
            Some(Tensor::F16(data))
        }
        Tensor::U32(_) => None,
    }
}

/// Row-concatenates loader matrices of one storage (all bf16, or all one packed format, whose
/// rows are whole blocks so rows concatenate as bytes); `None` when the storages differ.
fn concat_mats(ms: &[&Mat]) -> Option<Tensor> {
    let cols = ms[0].cols;
    let rows = ms.iter().map(|m| m.rows).sum();
    if ms.iter().any(|m| m.cols != cols) {
        return None;
    }
    let Some(format) = ms[0].packed().map(|p| p.format) else {
        let mut v = Vec::with_capacity(rows * cols);
        for m in ms {
            if m.packed().is_some() {
                return None;
            }
            v.extend_from_slice(m.data());
        }
        return Some(Tensor::Mat(Mat::from_bf16(v, rows, cols)));
    };
    let mut bytes = Vec::with_capacity(rows * format.tensor_bytes(cols));
    for m in ms {
        let p = m.packed().filter(|p| p.format == format)?;
        bytes.extend_from_slice(
            p.map
                .get(p.off..p.off + m.rows * format.tensor_bytes(cols))?,
        );
    }
    Some(Tensor::Mat(Mat::from_packed_bytes(format, bytes, rows, cols)))
}


/// Name of a fused weight: `<node>_fused_<kind>_w`, or `<weight>_fused_<kind>_w` when another
/// fusion already took that (nodes need not have unique names; weights do).
fn fused_name(g: &Graph, node: &str, weight: &str, kind: &str) -> String {
    let n = format!("{node}_fused_{kind}_w");
    if g.values.iter().any(|v| v.name == n) {
        format!("{weight}_fused_{kind}_w")
    } else {
        n
    }
}

/// What the fusion rewriters do with the tensors behind the weights they fuse.
pub enum FuseWeights<'a> {
    /// Concatenate each fused tensor into the weights (unpacked originals are dropped).
    Materialize(&'a mut Weights),
    /// Never concatenate: fuse only onto a weight the compiled program (a graph of the same
    /// network, fused with [`FuseWeights::Materialize`]) already binds under the fused name with
    /// the fused dtype and shape. Reshaped instances of a compiled program fuse this way.
    ReuseOnly(&'a Graph),
    /// Rewrite the graph only.
    None,
}

impl FuseWeights<'_> {
    pub(crate) fn reborrow(&mut self) -> FuseWeights<'_> {
        match self {
            FuseWeights::Materialize(w) => FuseWeights::Materialize(w),
            FuseWeights::ReuseOnly(g) => FuseWeights::ReuseOnly(g),
            FuseWeights::None => FuseWeights::None,
        }
    }

    /// The tensors a [`FuseWeights::Materialize`] fusion concatenates.
    pub(crate) fn tensors(&self) -> Option<&Weights> {
        match self {
            FuseWeights::Materialize(w) => Some(w),
            _ => None,
        }
    }
}

/// Whether `program` binds weight `name` as `dtype` with `shape` ([`FuseWeights::ReuseOnly`]).
fn binds(program: &Graph, name: &str, dtype: DType, shape: [usize; 2]) -> bool {
    program.values.iter().any(|v| {
        v.kind == ValueKind::Weight && v.name == name && v.dtype == dtype && v.shape == shape
    })
}

/// Validates that `tensor` matches the declared graph `dtype`.
pub(crate) fn matches_dtype(dtype: DType, tensor: &Tensor) -> bool {
    match (dtype, tensor) {
        (DType::F32, Tensor::F32(_)) => true,
        (DType::BF16, Tensor::BF16(_)) => true,
        (DType::F16, Tensor::F16(_)) => true,
        (DType::BF16, Tensor::Mat(m)) => m.packed().is_none(),
        (DType::Quant(fmt), Tensor::Mat(m)) => m.packed().is_some_and(|p| p.format == fmt),
        _ => false,
    }
}

/// Fuses parallel Q, K, V linear projections of one input into one projection and slices (streams
/// the activations once instead of 3 times): materializes the e-graph's chosen `matches` (captures
/// `q`, `k`, `v`; `q` first in the fused weight).
pub(crate) fn rewrite_qkv(
    graph: &mut Graph,
    matches: &[MatchContext],
    mut weights: FuseWeights<'_>,
) -> usize {
    if matches.is_empty() {
        return 0;
    }
    let mut weight_updates: Vec<(String, Tensor, Vec<String>)> = Vec::new();

    let rewrites = GraphRewriter::replace(graph, matches, |g, m| {
        let q_id = m.get("q")?;
        let k_id = m.get("k")?;
        let v_id = m.get("v")?;

        // Bias not supported for fusion yet: require exactly 2 inputs (x, w)
        if g.node(q_id).inputs.len() != 2
            || g.node(k_id).inputs.len() != 2
            || g.node(v_id).inputs.len() != 2
        {
            return None;
        }

        let in_val = g.node(q_id).inputs.first().copied()?;
        if g.node(k_id).inputs.first().copied()? != in_val
            || g.node(v_id).inputs.first().copied()? != in_val
        {
            return None;
        }

        let q_w_id = g.node(q_id).inputs.get(1).copied()?;
        let k_w_id = g.node(k_id).inputs.get(1).copied()?;
        let v_w_id = g.node(v_id).inputs.get(1).copied()?;

        // Verify identical dtypes
        let dtype = g.value(q_w_id).dtype;
        if g.value(k_w_id).dtype != dtype || g.value(v_w_id).dtype != dtype {
            return None;
        }

        let q_w_shape = g.value(q_w_id).shape.clone();
        let k_w_shape = g.value(k_w_id).shape.clone();
        let v_w_shape = g.value(v_w_id).shape.clone();

        if q_w_shape.len() != 2 || k_w_shape.len() != 2 || v_w_shape.len() != 2 {
            return None;
        }
        if q_w_shape[1] != k_w_shape[1] || q_w_shape[1] != v_w_shape[1] {
            return None;
        }

        let in_features = q_w_shape[1];
        let q_out = q_w_shape[0];
        let k_out = k_w_shape[0];
        let v_out = v_w_shape[0];
        let total_out = q_out + k_out + v_out;

        let q_name = g.node(q_id).name.clone();
        let k_name = g.node(k_id).name.clone();
        let v_name = g.node(v_id).name.clone();

        let q_out_shape = g.value(g.node(q_id).outputs[0]).shape.clone();
        let k_out_shape = g.value(g.node(k_id).outputs[0]).shape.clone();
        let v_out_shape = g.value(g.node(v_id).outputs[0]).shape.clone();

        let fused_w_name = fused_name(g, &q_name, &g.value(q_w_id).name, "qkv");

        match &weights {
            // The weight tensors MUST fuse.
            FuseWeights::Materialize(w_store) => {
                let q_t = w_store.get(&g.value(q_w_id).name)?;
                let k_t = w_store.get(&g.value(k_w_id).name)?;
                let v_t = w_store.get(&g.value(v_w_id).name)?;
                let fused_tensor = concat_weight_tensors(&[q_t, k_t, v_t])?;
                if !matches_dtype(dtype, &fused_tensor) {
                    return None;
                }
                // Loader Mats stay bound: a reshaped instance of the same program whose fusion
                // is rejected still reads them, and mapped ones cost no memory.
                let old_names = match q_t {
                    Tensor::Mat(_) => Vec::new(),
                    _ => vec![
                        g.value(q_w_id).name.clone(),
                        g.value(k_w_id).name.clone(),
                        g.value(v_w_id).name.clone(),
                    ],
                };
                weight_updates.push((fused_w_name.clone(), fused_tensor, old_names));
            }
            FuseWeights::ReuseOnly(p) if !binds(p, &fused_w_name, dtype, [total_out, in_features]) => {
                return None
            }
            FuseWeights::ReuseOnly(_) | FuseWeights::None => {}
        }

        let fused_w = g.weight(fused_w_name, vec![total_out, in_features], dtype);
        let in_shape = g.value(in_val).shape.clone();
        let mut fused_out_shape = in_shape;
        if let Some(last) = fused_out_shape.last_mut() {
            *last = total_out;
        }

        let fused_lin = g.add(
            format!("{}.fused_qkv", q_name),
            Op::Linear,
            &[in_val, fused_w],
            fused_out_shape,
            DType::F32,
        );

        let q_slice = g.add(
            format!("{}.q_slice", q_name),
            Op::Slice {
                axis: -1,
                start: 0,
                end: q_out as i64,
            },
            &[fused_lin],
            q_out_shape,
            DType::F32,
        );

        let k_slice = g.add(
            format!("{}.k_slice", k_name),
            Op::Slice {
                axis: -1,
                start: q_out as i64,
                end: (q_out + k_out) as i64,
            },
            &[fused_lin],
            k_out_shape,
            DType::F32,
        );

        let v_slice = g.add(
            format!("{}.v_slice", v_name),
            Op::Slice {
                axis: -1,
                start: (q_out + k_out) as i64,
                end: total_out as i64,
            },
            &[fused_lin],
            v_out_shape,
            DType::F32,
        );

        Some((vec![q_id, k_id, v_id], vec![q_slice, k_slice, v_slice]))
    });

    if let FuseWeights::Materialize(w_store) = &mut weights {
        for (name, tensor, old_names) in weight_updates {
            w_store.insert(name, tensor);
            for old in old_names {
                w_store.remove(&old);
            }
        }
    }

    rewrites
}

/// Fuses the gate and up projections of a SwiGLU / GeGLU into one projection (streams the
/// activations once instead of twice): materializes the e-graph's chosen `matches` (captures
/// `gate`, `up`; roles come from the GLU).
pub(crate) fn rewrite_gate_up(
    graph: &mut Graph,
    matches: &[MatchContext],
    mut weights: FuseWeights<'_>,
) -> usize {
    if matches.is_empty() {
        return 0;
    }
    let mut weight_updates: Vec<(String, Tensor, Vec<String>)> = Vec::new();

    let rewrites = GraphRewriter::replace(graph, matches, |g, m| {
        let a_id = m.get("gate")?;
        let b_id = m.get("up")?;

        // Bias not supported for fusion yet: require exactly 2 inputs (x, w)
        if g.node(a_id).inputs.len() != 2 || g.node(b_id).inputs.len() != 2 {
            return None;
        }

        let a_out = g.node(a_id).outputs[0];
        let b_out = g.node(b_id).outputs[0];

        // Consumers of a and b must feed into SwiGlu or GeGlu
        let glu_node = g.nodes.iter().find(|n| {
            matches!(n.op, Op::SwiGlu | Op::GeGlu)
                && (n.inputs.contains(&a_out) && n.inputs.contains(&b_out))
        })?;

        // Determine gate and up strictly by GLU inputs order: inputs[0] is gate, inputs[1] is up
        let (gate_id, up_id) = if glu_node.inputs[0] == a_out && glu_node.inputs[1] == b_out {
            (a_id, b_id)
        } else if glu_node.inputs[0] == b_out && glu_node.inputs[1] == a_out {
            (b_id, a_id)
        } else {
            return None;
        };

        let in_val = g.node(gate_id).inputs.first().copied()?;
        if g.node(up_id).inputs.first().copied()? != in_val {
            return None;
        }

        let gate_w_id = g.node(gate_id).inputs.get(1).copied()?;
        let up_w_id = g.node(up_id).inputs.get(1).copied()?;

        let dtype = g.value(gate_w_id).dtype;
        if g.value(up_w_id).dtype != dtype {
            return None;
        }

        let gate_w_shape = g.value(gate_w_id).shape.clone();
        let up_w_shape = g.value(up_w_id).shape.clone();

        if gate_w_shape.len() != 2 || up_w_shape.len() != 2 || gate_w_shape[1] != up_w_shape[1] {
            return None;
        }

        let in_features = gate_w_shape[1];
        let gate_out_dim = gate_w_shape[0];
        let up_out_dim = up_w_shape[0];
        let total_out = gate_out_dim + up_out_dim;

        let gate_name = g.node(gate_id).name.clone();
        let up_name = g.node(up_id).name.clone();
        let gate_out_shape = g.value(g.node(gate_id).outputs[0]).shape.clone();
        let up_out_shape = g.value(g.node(up_id).outputs[0]).shape.clone();

        let fused_w_name = fused_name(g, &gate_name, &g.value(gate_w_id).name, "gate_up");

        match &weights {
            // The weight tensors MUST fuse.
            FuseWeights::Materialize(w_store) => {
                let gate_t = w_store.get(&g.value(gate_w_id).name)?;
                let up_t = w_store.get(&g.value(up_w_id).name)?;
                let fused_tensor = concat_weight_tensors(&[gate_t, up_t])?;
                if !matches_dtype(dtype, &fused_tensor) {
                    return None;
                }
                let old_names = match gate_t {
                    Tensor::Mat(_) => Vec::new(),
                    _ => vec![
                        g.value(gate_w_id).name.clone(),
                        g.value(up_w_id).name.clone(),
                    ],
                };
                weight_updates.push((fused_w_name.clone(), fused_tensor, old_names));
            }
            FuseWeights::ReuseOnly(p) if !binds(p, &fused_w_name, dtype, [total_out, in_features]) => {
                return None
            }
            FuseWeights::ReuseOnly(_) | FuseWeights::None => {}
        }

        let fused_w = g.weight(fused_w_name, vec![total_out, in_features], dtype);
        let in_shape = g.value(in_val).shape.clone();
        let mut fused_out_shape = in_shape;
        if let Some(last) = fused_out_shape.last_mut() {
            *last = total_out;
        }

        let fused_lin = g.add(
            format!("{}.fused_gate_up", gate_name),
            Op::Linear,
            &[in_val, fused_w],
            fused_out_shape,
            DType::F32,
        );

        let gate_slice = g.add(
            format!("{}.gate_slice", gate_name),
            Op::Slice {
                axis: -1,
                start: 0,
                end: gate_out_dim as i64,
            },
            &[fused_lin],
            gate_out_shape,
            DType::F32,
        );

        let up_slice = g.add(
            format!("{}.up_slice", up_name),
            Op::Slice {
                axis: -1,
                start: gate_out_dim as i64,
                end: total_out as i64,
            },
            &[fused_lin],
            up_out_shape,
            DType::F32,
        );

        Some((vec![gate_id, up_id], vec![gate_slice, up_slice]))
    });

    if let FuseWeights::Materialize(w_store) = &mut weights {
        for (name, tensor, old_names) in weight_updates {
            w_store.insert(name, tensor);
            for old in old_names {
                w_store.remove(&old);
            }
        }
    }

    rewrites
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::egraph::{fuse_rule, Rule};
    use crate::graph::pattern::validate_ssa;

    #[test]
    fn test_fuse_qkv_pass() {
        let mut g = Graph::new();
        let in_x = g.input("x", vec![1, 64], DType::F32);
        let gamma = g.weight("gamma", vec![64], DType::F32);
        let norm = g.add(
            "norm",
            Op::RmsNorm { eps: 1e-5 },
            &[in_x, gamma],
            vec![1, 64],
            DType::F32,
        );

        let w_q = g.weight("w_q", vec![64, 64], DType::F32);
        let w_k = g.weight("w_k", vec![32, 64], DType::F32);
        let w_v = g.weight("w_v", vec![32, 64], DType::F32);

        let q = g.add("q", Op::Linear, &[norm, w_q], vec![1, 64], DType::F32);
        let k = g.add("k", Op::Linear, &[norm, w_k], vec![1, 32], DType::F32);
        let v = g.add("v", Op::Linear, &[norm, w_v], vec![1, 32], DType::F32);

        let out = g.add("out_add", Op::Binary { kind: crate::graph::BinaryKind::Add }, &[k, v], vec![1, 32], DType::F32);
        g.output(q);
        g.output(out);

        let mut weights = Weights::new();
        weights.insert("w_q", Tensor::F32(vec![1.0; 64 * 64]));
        weights.insert("w_k", Tensor::F32(vec![2.0; 32 * 64]));
        weights.insert("w_v", Tensor::F32(vec![3.0; 32 * 64]));

        let rewrites = fuse_rule(&mut g, FuseWeights::Materialize(&mut weights), Rule::Qkv);
        assert_eq!(rewrites, 1);
        assert!(validate_ssa(&g).is_ok());

        assert!(weights.get("q_fused_qkv_w").is_some());
        let fused_w = weights.get("q_fused_qkv_w").unwrap().as_f32().unwrap();
        assert_eq!(fused_w.len(), (64 + 32 + 32) * 64);
        assert_eq!(fused_w[0], 1.0);
        assert_eq!(fused_w[64 * 64], 2.0);
        assert_eq!(fused_w[(64 + 32) * 64], 3.0);
    }

    #[test]
    fn test_fuse_gate_up_pass() {
        let mut g = Graph::new();
        let in_x = g.input("x", vec![1, 64], DType::F32);
        let w_g = g.weight("w_g", vec![128, 64], DType::F32);
        let w_u = g.weight("w_u", vec![128, 64], DType::F32);

        let gate = g.add("gate", Op::Linear, &[in_x, w_g], vec![1, 128], DType::F32);
        let up = g.add("up", Op::Linear, &[in_x, w_u], vec![1, 128], DType::F32);
        let mlp = g.add("swiglu", Op::SwiGlu, &[gate, up], vec![1, 128], DType::F32);
        g.output(mlp);

        let mut weights = Weights::new();
        weights.insert("w_g", Tensor::F32(vec![0.5; 128 * 64]));
        weights.insert("w_u", Tensor::F32(vec![0.75; 128 * 64]));

        let rewrites = fuse_rule(&mut g, FuseWeights::Materialize(&mut weights), Rule::GateUp);
        assert_eq!(rewrites, 1);
        assert!(validate_ssa(&g).is_ok());
        assert!(weights.get("gate_fused_gate_up_w").is_some());
    }

    #[test]
    fn test_cpu_parity_before_and_after_fusion() {
        use crate::graph::cpu::run_stateless;

        // Build a graph with distinct weights
        let mut g = Graph::new();
        let in_x = g.input("x", vec![1, 4], DType::F32);
        let gamma = g.weight("gamma", vec![4], DType::F32);
        let norm = g.add(
            "norm",
            Op::RmsNorm { eps: 1e-5 },
            &[in_x, gamma],
            vec![1, 4],
            DType::F32,
        );

        let w_q = g.weight("w_q", vec![4, 4], DType::F32);
        let w_k = g.weight("w_k", vec![2, 4], DType::F32);
        let w_v = g.weight("w_v", vec![2, 4], DType::F32);
        let q = g.add("q", Op::Linear, &[norm, w_q], vec![1, 4], DType::F32);
        let k = g.add("k", Op::Linear, &[norm, w_k], vec![1, 2], DType::F32);
        let v = g.add("v", Op::Linear, &[norm, w_v], vec![1, 2], DType::F32);

        let w_g = g.weight("w_g", vec![8, 4], DType::F32);
        let w_u = g.weight("w_u", vec![8, 4], DType::F32);
        let gate = g.add("gate", Op::Linear, &[norm, w_g], vec![1, 8], DType::F32);
        let up = g.add("up", Op::Linear, &[norm, w_u], vec![1, 8], DType::F32);
        let mlp = g.add("swiglu", Op::SwiGlu, &[gate, up], vec![1, 8], DType::F32);

        g.output(q);
        g.output(k);
        g.output(v);
        g.output(mlp);

        let mut weights = Weights::new();
        weights.insert("gamma", Tensor::F32(vec![1.1, 0.9, 1.05, 0.95]));
        weights.insert(
            "w_q",
            Tensor::F32((0..16).map(|i| (i as f32 * 0.1) - 0.7).collect()),
        );
        weights.insert(
            "w_k",
            Tensor::F32((0..8).map(|i| (i as f32 * 0.15) - 0.4).collect()),
        );
        weights.insert(
            "w_v",
            Tensor::F32((0..8).map(|i| (i as f32 * 0.2) - 0.5).collect()),
        );
        weights.insert(
            "w_g",
            Tensor::F32((0..32).map(|i| (i as f32 * 0.05) - 0.3).collect()),
        );
        weights.insert(
            "w_u",
            Tensor::F32((0..32).map(|i| (i as f32 * 0.07) - 0.6).collect()),
        );

        let input_tensor = vec![Tensor::F32(vec![0.5, -1.2, 2.3, -0.4])];

        // 1. Run baseline on unoptimized graph
        let baseline_res = run_stateless(&g, &weights, input_tensor.clone()).unwrap();

        // 2. Run passes
        let mut opt_g = g.clone();
        let mut opt_w = weights.clone();
        let qkv_rewrites = fuse_rule(&mut opt_g, FuseWeights::Materialize(&mut opt_w), Rule::Qkv);
        let glu_rewrites =
            fuse_rule(&mut opt_g, FuseWeights::Materialize(&mut opt_w), Rule::GateUp);
        assert_eq!(qkv_rewrites, 1);
        assert_eq!(glu_rewrites, 1);
        assert!(validate_ssa(&opt_g).is_ok());

        // 3. Run optimized graph
        let opt_res = run_stateless(&opt_g, &opt_w, input_tensor).unwrap();

        // 4. Assert exact numerical equality across all 4 outputs
        assert_eq!(baseline_res.len(), opt_res.len());
        for (out_idx, (b_t, o_t)) in baseline_res.iter().zip(&opt_res).enumerate() {
            let b_slice = b_t.as_f32().unwrap();
            let o_slice = o_t.as_f32().unwrap();
            assert_eq!(b_slice.len(), o_slice.len());
            for (i, (&b_val, &o_val)) in b_slice.iter().zip(o_slice).enumerate() {
                let diff = (b_val - o_val).abs();
                assert!(
                    diff < 1e-5,
                    "Parity failed at output {out_idx} element {i}: baseline={b_val}, opt={o_val}, diff={diff}"
                );
            }
        }
    }

    /// x -> norm -> {q,k,v} Linear -> Attention with per-projection weight dtypes. F16 weights
    /// stand in for a second plain codec; `Quant(f)` weights are packed GGUF `Mat`s in an
    /// anonymous mapping, as the GGUF loaders produce them. Returns the graph, its weights and the
    /// input.
    fn qkv_graph(dts: [DType; 3]) -> (Graph, Weights, Vec<Tensor>) {
        let (h, rows) = (256usize, [8usize, 4, 4]);
        let mut g = Graph::new();
        let x = g.input("x", vec![1, h], DType::F32);
        let gamma = g.weight("gamma", vec![h], DType::F32);
        let norm = g.add(
            "norm",
            Op::RmsNorm { eps: 1e-5 },
            &[x, gamma],
            vec![1, h],
            DType::F32,
        );
        let mut weights = Weights::new();
        weights.insert("gamma", Tensor::F32(vec![1.0; h]));
        let mut outs = Vec::new();
        for (i, name) in ["q", "k", "v"].into_iter().enumerate() {
            let n = rows[i] * h;
            let wn = format!("w_{name}");
            let w = g.weight(&wn, vec![rows[i], h], dts[i]);
            // Exact halves of the form ±(j%5)/4, so F16 and F32 weights both decode exactly.
            let vals = (0..n).map(|j| ((j * (i + 3)) % 5) as f32 / 4.0 - 0.5);
            weights.insert(
                wn,
                match dts[i] {
                    DType::F16 => Tensor::F16(
                        vals.map(|f| match (f * 4.0) as i32 {
                            -2 => 0xB800,
                            -1 => 0xB400,
                            0 => 0,
                            1 => 0x3400,
                            _ => 0x3800,
                        })
                        .collect(),
                    ),
                    DType::Quant(fmt) => packed_mat(fmt, rows[i], h, i),
                    _ => Tensor::F32(vals.collect()),
                },
            );
            let out = g.add(name, Op::Linear, &[norm, w], vec![1, rows[i]], DType::F32);
            g.output(out);
            outs.push(out);
        }
        let op = Op::Attention {
            heads: 2,
            kv_heads: 1,
            causal: true,
            window: None,
            soft_cap: None,
            scale: None,
        };
        let a = g.add("attn", op, &outs, vec![1, rows[0]], DType::F32);
        g.output(a);
        let input = vec![Tensor::F32((0..h).map(|i| i as f32 * 0.3 - 1.0).collect())];
        (g, weights, input)
    }

    /// A `[rows, cols]` packed GGUF `Mat` of patterned blocks whose f16 scales (Q4_0: `d`;
    /// Q4_K: `d`, `dmin`) are a finite 0.0625.
    fn packed_mat(
        fmt: crate::hardware::QuantFormat,
        rows: usize,
        cols: usize,
        seed: usize,
    ) -> Tensor {
        use crate::model::weights::{Mat, Packed};
        let (block, header) = match fmt {
            crate::hardware::QuantFormat::GGUF_Q4_0 => (18, 2),
            crate::hardware::QuantFormat::GGUF_Q4_K_M => (144, 4),
            f => panic!("no fixture for {f:?}"),
        };
        let n = rows * fmt.tensor_bytes(cols);
        let mut map = memmap2::MmapMut::map_anon(n).unwrap();
        for (j, b) in map.iter_mut().enumerate() {
            *b = (j * 31 + seed * 7) as u8;
        }
        for blk in map.chunks_exact_mut(block) {
            for c in blk[..header].chunks_exact_mut(2) {
                c.copy_from_slice(&0x2C00u16.to_le_bytes());
            }
        }
        let p = Packed {
            map: std::sync::Arc::new(crate::model::loaders::gguf::GgufData::Map(
                map.make_read_only().unwrap(),
            )),
            off: 0,
            format: fmt,
            shape: vec![rows, cols],
        };
        Tensor::Mat(Mat::from_packed(p, rows, cols))
    }

    /// Runs the QKV rule, checks CPU parity with the unfused graph, and returns the names of
    /// the remaining Linear nodes.
    fn fuse_and_check(dts: [DType; 3]) -> Vec<String> {
        use crate::graph::cpu::run_stateless;
        let (g, w, input) = qkv_graph(dts);
        let base = run_stateless(&g, &w, input.clone()).unwrap();
        let (mut fg, mut fw) = (g.clone(), w.clone());
        assert_eq!(fuse_rule(&mut fg, FuseWeights::Materialize(&mut fw), Rule::Qkv), 1);
        assert!(validate_ssa(&fg).is_ok());
        let fused = run_stateless(&fg, &fw, input).unwrap();
        assert_eq!(base.len(), fused.len());
        for (b, f) in base.iter().zip(&fused) {
            assert_eq!(b.as_f32().unwrap(), f.as_f32().unwrap());
        }
        let mut names: Vec<String> = fg
            .nodes
            .iter()
            .filter(|n| matches!(n.op, Op::Linear))
            .map(|n| n.name.clone())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn fuse_qkv_pass_leaves_mixed_formats_unfused() {
        let (h, f) = (DType::F16, DType::F32);
        for dts in [[h, h, f], [f, h, h]] {
            let (mut g, mut w, _) = qkv_graph(dts);
            assert_eq!(
                fuse_rule(&mut g, FuseWeights::Materialize(&mut w), Rule::Qkv),
                0,
                "{dts:?}"
            );
        }
    }

    #[test]
    fn fuse_qkv_pass_concatenates_packed_mats() {
        let q4 = DType::Quant(crate::hardware::QuantFormat::GGUF_Q4_0);
        assert_eq!(fuse_and_check([q4, q4, q4]), ["q.fused_qkv"]);
    }

    #[test]
    fn fuse_qkv_pass_same_format_still_fuses_all_three() {
        let f = DType::F32;
        assert_eq!(fuse_and_check([f, f, f]), ["q.fused_qkv"]);
    }
}
