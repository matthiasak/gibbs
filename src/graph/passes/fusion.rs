//! Subgraph fusion optimization passes for Graph IR.

use crate::graph::pattern::{GraphRewriter, MatchContext};
use crate::graph::{DType, Graph, Node, NodeId, Op, Tensor, ValueId, ValueKind, Weights};
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

/// Concatenates optional 1D bias tensors along axis 0, filling `0.0` for `None` siblings
/// whenever at least one sibling carries a bias (`None` if no sibling carries a bias).
pub fn concat_bias_tensors(biases: &[Option<(&Tensor, usize)>]) -> Option<Tensor> {
    let default_rows = biases.iter().flatten().next()?.1;
    let slots: Vec<(Option<&Tensor>, usize)> = biases
        .iter()
        .map(|b| match b {
            Some((t, n)) => (Some(*t), *n),
            None => (None, default_rows),
        })
        .collect();
    concat_bias_slots(&slots).map(|(t, _)| t)
}

/// Concatenates optional 1D biases with explicit per-slot row counts (`(Option<&Tensor>, rows)`),
/// zero-filling `None` siblings when at least one sibling carries a bias.
pub fn concat_bias_slots(slots: &[(Option<&Tensor>, usize)]) -> Option<(Tensor, DType)> {
    if slots.iter().all(|(t, _)| t.is_none()) {
        return None;
    }
    for &(t, n) in slots {
        if let Some(t) = t {
            let ok = match t {
                Tensor::F32(v) => v.len() == n,
                Tensor::F16(v) => v.len() == n,
                Tensor::BF16(v) => v.len() == n,
                _ => false,
            };
            if !ok {
                return None;
            }
        }
    }
    let total: usize = slots.iter().map(|&(_, n)| n).sum();
    let all_f16 = slots
        .iter()
        .filter_map(|&(t, _)| t)
        .all(|t| matches!(t, Tensor::F16(_)));
    let all_bf16 = slots
        .iter()
        .filter_map(|&(t, _)| t)
        .all(|t| matches!(t, Tensor::BF16(_)));
    if all_f16 {
        let mut out = Vec::with_capacity(total);
        for &(t, n) in slots {
            match t {
                Some(Tensor::F16(v)) => out.extend_from_slice(v),
                None => out.extend(std::iter::repeat_n(0u16, n)),
                _ => unreachable!(),
            }
        }
        return Some((Tensor::F16(out), DType::F16));
    }
    if all_bf16 {
        let mut out = Vec::with_capacity(total);
        for &(t, n) in slots {
            match t {
                Some(Tensor::BF16(v)) => out.extend_from_slice(v),
                None => out.extend(std::iter::repeat_n(0u16, n)),
                _ => unreachable!(),
            }
        }
        return Some((Tensor::BF16(out), DType::BF16));
    }
    let mut out = Vec::with_capacity(total);
    for &(t, n) in slots {
        match t {
            Some(t) => out.extend_from_slice(&t.to_f32()),
            None => out.extend(std::iter::repeat_n(0.0f32, n)),
        }
    }
    Some((Tensor::F32(out), DType::F32))
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
    Some(Tensor::Mat(Mat::from_packed_bytes(
        format, bytes, rows, cols,
    )))
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
fn binds_1d(program: &Graph, name: &str, dtype: DType, len: usize) -> bool {
    program.values.iter().any(|v| {
        v.kind == ValueKind::Weight && v.name == name && v.dtype == dtype && v.shape == [len]
    })
}

/// Checks whether a Linear node's optional 3rd input is a fusible static 1D float bias weight.
pub(crate) fn valid_linear_bias(g: &Graph, n: &Node) -> bool {
    match n.inputs.as_slice() {
        [_, _] => true,
        [_, w, b] => {
            let wv = g.value(*w);
            let bv = g.value(*b);
            wv.shape.len() == 2
                && bv.kind == ValueKind::Weight
                && bv.shape == [wv.shape[0]]
                && matches!(bv.dtype, DType::F32 | DType::F16 | DType::BF16)
        }
        _ => false,
    }
}

fn infer_bias_dtype(g: &Graph, bias_ids: &[Option<ValueId>]) -> Option<DType> {
    let present: Vec<DType> = bias_ids
        .iter()
        .flatten()
        .map(|&b| g.value(b).dtype)
        .collect();
    if present.is_empty() {
        return None;
    }
    if present.iter().all(|&d| d == DType::F16) {
        Some(DType::F16)
    } else if present.iter().all(|&d| d == DType::BF16) {
        Some(DType::BF16)
    } else {
        Some(DType::F32)
    }
}

fn fuse_linear_nodes_in_rewriter(
    g: &mut Graph,
    ids: &[NodeId],
    kind: &str,
    slice_suffixes: &[&str],
    weights: &FuseWeights<'_>,
    weight_updates: &mut Vec<(String, Tensor, Vec<String>)>,
) -> Option<(Vec<NodeId>, Vec<ValueId>)> {
    if ids.len() < 2 || ids.len() != slice_suffixes.len() {
        return None;
    }
    for &id in ids {
        if !matches!(g.node(id).op, Op::Linear) || !valid_linear_bias(g, g.node(id)) {
            return None;
        }
    }
    let in_val = g.node(ids[0]).inputs[0];
    if ids.iter().any(|&id| g.node(id).inputs[0] != in_val) {
        return None;
    }
    let w_ids: Vec<ValueId> = ids.iter().map(|&id| g.node(id).inputs[1]).collect();
    let b_ids: Vec<Option<ValueId>> = ids
        .iter()
        .map(|&id| g.node(id).inputs.get(2).copied())
        .collect();
    let dtype = g.value(w_ids[0]).dtype;
    let w0_shape = g.value(w_ids[0]).shape.clone();
    if w0_shape.len() != 2 {
        return None;
    }
    let in_features = w0_shape[1];
    let mut out_dims = Vec::with_capacity(ids.len());
    for &wid in &w_ids {
        let ws = &g.value(wid).shape;
        if g.value(wid).dtype != dtype || ws.len() != 2 || ws[1] != in_features {
            return None;
        }
        out_dims.push(ws[0]);
    }
    let total_out: usize = out_dims.iter().sum();
    let first_name = g.node(ids[0]).name.clone();
    let fused_w_name = fused_name(g, &first_name, &g.value(w_ids[0]).name, kind);
    let fused_b_name = format!("{fused_w_name}_b");
    let mut fused_b_dtype = infer_bias_dtype(g, &b_ids);

    // Count external readers so shared weights/biases are never removed from `Weights`.
    let in_group: std::collections::HashSet<u32> = ids.iter().map(|n| n.0).collect();
    let has_other_reader = |vid: ValueId| -> bool {
        g.nodes
            .iter()
            .enumerate()
            .any(|(ni, n)| !in_group.contains(&(ni as u32)) && n.inputs.contains(&vid))
    };

    match weights {
        FuseWeights::Materialize(w_store) => {
            let mut w_tensors = Vec::with_capacity(w_ids.len());
            for &wid in &w_ids {
                w_tensors.push(w_store.get(&g.value(wid).name)?);
            }
            let fused_tensor = concat_weight_tensors(&w_tensors)?;
            if !matches_dtype(dtype, &fused_tensor) {
                return None;
            }
            let old_names = match w_tensors[0] {
                Tensor::Mat(_) => Vec::new(),
                _ => w_ids
                    .iter()
                    .filter(|&&wid| !has_other_reader(wid))
                    .map(|&wid| g.value(wid).name.clone())
                    .collect(),
            };
            let mut pending_b = None;
            if fused_b_dtype.is_some() {
                let mut b_slots = Vec::with_capacity(b_ids.len());
                for (&bid, &rows) in b_ids.iter().zip(&out_dims) {
                    let bt = match bid {
                        Some(b) => Some(w_store.get(&g.value(b).name)?),
                        None => None,
                    };
                    b_slots.push((bt, rows));
                }
                let (fb_t, fb_dt) = concat_bias_slots(&b_slots)?;
                fused_b_dtype = Some(fb_dt);
                let old_b: Vec<String> = b_ids
                    .iter()
                    .flatten()
                    .filter(|&&bid| !has_other_reader(bid))
                    .map(|&bid| g.value(bid).name.clone())
                    .collect();
                pending_b = Some((fused_b_name.clone(), fb_t, old_b));
            }
            weight_updates.push((fused_w_name.clone(), fused_tensor, old_names));
            if let Some(pb) = pending_b {
                weight_updates.push(pb);
            }
        }
        FuseWeights::ReuseOnly(p) => {
            if !binds(p, &fused_w_name, dtype, [total_out, in_features]) {
                return None;
            }
            if let Some(bdt) = fused_b_dtype {
                if !binds_1d(p, &fused_b_name, bdt, total_out) {
                    return None;
                }
            }
        }
        FuseWeights::None => {}
    }

    let node_names: Vec<String> = ids.iter().map(|&id| g.node(id).name.clone()).collect();
    let out_shapes: Vec<Vec<usize>> = ids
        .iter()
        .map(|&id| g.value(g.node(id).outputs[0]).shape.clone())
        .collect();
    let out_dtypes: Vec<DType> = ids
        .iter()
        .map(|&id| g.value(g.node(id).outputs[0]).dtype)
        .collect();

    let fused_w = g.weight(fused_w_name, vec![total_out, in_features], dtype);
    let mut lin_inputs = vec![in_val, fused_w];
    if let Some(bdt) = fused_b_dtype {
        let fused_b = g.weight(fused_b_name, vec![total_out], bdt);
        lin_inputs.push(fused_b);
    }
    let mut fused_out_shape = g.value(in_val).shape.clone();
    if let Some(last) = fused_out_shape.last_mut() {
        *last = total_out;
    }
    let fused_lin = g.add(
        format!("{first_name}.fused_{kind}"),
        Op::Linear,
        &lin_inputs,
        fused_out_shape,
        out_dtypes[0],
    );

    let mut slices = Vec::with_capacity(ids.len());
    let mut cursor = 0usize;
    for i in 0..ids.len() {
        let next = cursor + out_dims[i];
        let s = g.add(
            format!("{}.{}", node_names[i], slice_suffixes[i]),
            Op::Slice {
                axis: -1,
                start: cursor as i64,
                end: next as i64,
            },
            &[fused_lin],
            out_shapes[i].clone(),
            out_dtypes[i],
        );
        slices.push(s);
        cursor = next;
    }
    Some((ids.to_vec(), slices))
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
        fuse_linear_nodes_in_rewriter(
            g,
            &[q_id, k_id, v_id],
            "qkv",
            &["q_slice", "k_slice", "v_slice"],
            &weights,
            &mut weight_updates,
        )
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

/// Fuses parallel Q and K linear projections of one input into one projection and 2 slices when V
/// has a different weight format (e.g. Q4_K Q/K + Q6_K V) or is absent (keys reused as values).
pub(crate) fn rewrite_qk(
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
        fuse_linear_nodes_in_rewriter(
            g,
            &[q_id, k_id],
            "qk",
            &["q_slice", "k_slice"],
            &weights,
            &mut weight_updates,
        )
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

        if !valid_linear_bias(g, g.node(a_id)) || !valid_linear_bias(g, g.node(b_id)) {
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

        fuse_linear_nodes_in_rewriter(
            g,
            &[gate_id, up_id],
            "gate_up",
            &["gate_slice", "up_slice"],
            &weights,
            &mut weight_updates,
        )
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

/// Fuses any remaining ≥2 `Op::Linear` nodes reading the exact same `x: ValueId` and sharing
/// concatenable `(DType, cols)` weights into one `Op::Linear` + `Op::Slice`s (e.g. a linear-attention
/// mixer's parallel input projections, or per-layer-embedding sibling projections).
/// Also flattens direct `Slice(axis=-1)` children of an already-fused sibling so 4 parallel
/// linears always materialize as 1 `Op::Linear` + 4 `Op::Slice`s.
pub fn fuse_shared_input_linears(graph: &mut Graph, mut weights: FuseWeights<'_>) -> usize {
    // Weight reader counts so tied weights (e.g. `lm_head` == `embed_tokens`) stay untouched.
    let mut w_readers = vec![0usize; graph.values.len()];
    let mut glu_branches: std::collections::HashSet<ValueId> = std::collections::HashSet::new();
    for n in &graph.nodes {
        for &inp in &n.inputs {
            w_readers[inp.0 as usize] += 1;
        }
        if matches!(n.op, Op::SwiGlu | Op::GeGlu) {
            for &inp in &n.inputs {
                glu_branches.insert(inp);
            }
        }
    }

    // Group candidate Linear node indices by `(x, dtype, cols)` in topological order.
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut group_idx: std::collections::HashMap<(ValueId, String, usize), usize> =
        std::collections::HashMap::new();
    for (ni, n) in graph.nodes.iter().enumerate() {
        if !matches!(n.op, Op::Linear) || !valid_linear_bias(graph, n) {
            continue;
        }
        let (x, w) = (n.inputs[0], n.inputs[1]);
        if w_readers[w.0 as usize] != 1 {
            continue;
        }
        let wv = graph.value(w);
        if wv.kind != ValueKind::Weight || wv.shape.len() != 2 {
            continue;
        }
        let out_v = n.outputs[0];
        // Never fuse GLU gate/up branches on packed/Mat weights (Metal folds GLU into up epilogue).
        let is_mat = matches!(wv.dtype, DType::Quant(_))
            || weights
                .tensors()
                .and_then(|ws| ws.get(&wv.name))
                .is_some_and(|t| matches!(t, Tensor::Mat(_)));
        if is_mat && glu_branches.contains(&out_v) {
            continue;
        }
        if let Some(ws) = weights.tensors() {
            let Some(t) = ws.get(&wv.name) else {
                continue;
            };
            if !matches_dtype(wv.dtype, t) {
                continue;
            }
        }
        let key = (x, format!("{:?}", wv.dtype), wv.shape[1]);
        let gi = *group_idx.entry(key).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[gi].push(ni);
    }

    let candidate_groups: Vec<Vec<usize>> = groups.into_iter().filter(|g| g.len() >= 2).collect();
    if candidate_groups.is_empty() {
        return 0;
    }

    let mut weight_updates: Vec<(String, Tensor, Vec<String>)> = Vec::new();
    // Keyed by first-member node index: `(all_member_indices, fused_lin_node, slice_nodes, old_out -> new_slice_out)`.
    let mut plans: std::collections::HashMap<
        usize,
        (Vec<usize>, Node, Vec<Node>, Vec<(ValueId, ValueId)>),
    > = std::collections::HashMap::new();
    let mut removed_members: std::collections::HashSet<usize> = std::collections::HashSet::new();

    let in_all_groups: std::collections::HashSet<usize> =
        candidate_groups.iter().flatten().copied().collect();

    for grp in candidate_groups {
        let in_val = graph.nodes[grp[0]].inputs[0];
        let w_ids: Vec<ValueId> = grp.iter().map(|&ni| graph.nodes[ni].inputs[1]).collect();
        let b_ids: Vec<Option<ValueId>> = grp
            .iter()
            .map(|&ni| graph.nodes[ni].inputs.get(2).copied())
            .collect();
        let dtype = graph.value(w_ids[0]).dtype;
        let in_features = graph.value(w_ids[0]).shape[1];
        let out_dims: Vec<usize> = w_ids.iter().map(|&w| graph.value(w).shape[0]).collect();
        let total_out: usize = out_dims.iter().sum();
        let first_name = graph.nodes[grp[0]].name.clone();
        let first_clean = first_name
            .strip_suffix(".fused_qkv")
            .or_else(|| first_name.strip_suffix(".fused_qk"))
            .unwrap_or(&first_name)
            .to_string();
        let fused_w_name = fused_name(graph, &first_clean, &graph.value(w_ids[0]).name, "shared");
        let fused_b_name = format!("{fused_w_name}_b");
        let mut fused_b_dtype = infer_bias_dtype(graph, &b_ids);

        let has_ext_reader = |vid: ValueId| -> bool {
            graph
                .nodes
                .iter()
                .enumerate()
                .any(|(ni, n)| !in_all_groups.contains(&ni) && n.inputs.contains(&vid))
        };

        match &weights {
            FuseWeights::Materialize(w_store) => {
                let mut w_tensors = Vec::with_capacity(w_ids.len());
                let mut ok = true;
                for &wid in &w_ids {
                    if let Some(t) = w_store.get(&graph.value(wid).name) {
                        w_tensors.push(t);
                    } else {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    continue;
                }
                let Some(fused_tensor) = concat_weight_tensors(&w_tensors) else {
                    continue;
                };
                if !matches_dtype(dtype, &fused_tensor) {
                    continue;
                }
                let old_names = match w_tensors[0] {
                    Tensor::Mat(_) => Vec::new(),
                    _ => w_ids
                        .iter()
                        .filter(|&&wid| !has_ext_reader(wid))
                        .map(|&wid| graph.value(wid).name.clone())
                        .collect(),
                };
                let mut pending_b = None;
                if fused_b_dtype.is_some() {
                    let mut b_slots = Vec::with_capacity(b_ids.len());
                    let mut bok = true;
                    for (&bid, &rows) in b_ids.iter().zip(&out_dims) {
                        match bid {
                            Some(b) => {
                                if let Some(bt) = w_store.get(&graph.value(b).name) {
                                    b_slots.push((Some(bt), rows));
                                } else {
                                    bok = false;
                                    break;
                                }
                            }
                            None => b_slots.push((None, rows)),
                        }
                    }
                    if !bok {
                        continue;
                    }
                    let Some((fb_t, fb_dt)) = concat_bias_slots(&b_slots) else {
                        continue;
                    };
                    fused_b_dtype = Some(fb_dt);
                    let old_b: Vec<String> = b_ids
                        .iter()
                        .flatten()
                        .filter(|&&bid| !has_ext_reader(bid))
                        .map(|&bid| graph.value(bid).name.clone())
                        .collect();
                    pending_b = Some((fused_b_name.clone(), fb_t, old_b));
                }
                weight_updates.push((fused_w_name.clone(), fused_tensor, old_names));
                if let Some(pb) = pending_b {
                    weight_updates.push(pb);
                }
            }
            FuseWeights::ReuseOnly(p) => {
                if !binds(p, &fused_w_name, dtype, [total_out, in_features]) {
                    continue;
                }
                if let Some(bdt) = fused_b_dtype {
                    if !binds_1d(p, &fused_b_name, bdt, total_out) {
                        continue;
                    }
                }
            }
            FuseWeights::None => {}
        }

        let member_names: Vec<String> =
            grp.iter().map(|&ni| graph.nodes[ni].name.clone()).collect();
        let member_outs: Vec<ValueId> = grp.iter().map(|&ni| graph.nodes[ni].outputs[0]).collect();
        let member_shapes: Vec<Vec<usize>> = member_outs
            .iter()
            .map(|&v| graph.value(v).shape.clone())
            .collect();
        let member_dtypes: Vec<DType> = member_outs.iter().map(|&v| graph.value(v).dtype).collect();

        let fused_w = graph.weight(fused_w_name, vec![total_out, in_features], dtype);
        let mut lin_inputs = vec![in_val, fused_w];
        if let Some(bdt) = fused_b_dtype {
            let fused_b = graph.weight(fused_b_name, vec![total_out], bdt);
            lin_inputs.push(fused_b);
        }
        let mut fused_out_shape = graph.value(in_val).shape.clone();
        if let Some(last) = fused_out_shape.last_mut() {
            *last = total_out;
        }
        let placeholder = NodeId(0);
        let fused_out = graph.push_value(
            format!("{first_clean}.fused_shared"),
            fused_out_shape,
            member_dtypes[0],
            ValueKind::Node(placeholder),
        );
        let first_dev = graph.nodes[grp[0]].device;
        let fused_node = Node {
            name: format!("{first_clean}.fused_shared"),
            op: Op::Linear,
            inputs: lin_inputs,
            outputs: vec![fused_out],
            device: first_dev,
        };
        let mut slice_nodes = Vec::with_capacity(grp.len());
        let mut remaps = Vec::with_capacity(grp.len());
        let mut cursor = 0usize;
        for i in 0..grp.len() {
            let next = cursor + out_dims[i];
            let s_name = format!("{}.shared_slice_{i}", member_names[i]);
            let s_out = graph.push_value(
                s_name.clone(),
                member_shapes[i].clone(),
                member_dtypes[i],
                ValueKind::Node(placeholder),
            );
            slice_nodes.push(Node {
                name: s_name,
                op: Op::Slice {
                    axis: -1,
                    start: cursor as i64,
                    end: next as i64,
                },
                inputs: vec![fused_out],
                outputs: vec![s_out],
                device: first_dev,
            });
            remaps.push((member_outs[i], s_out));
            cursor = next;
        }
        for &ni in &grp[1..] {
            removed_members.insert(ni);
        }
        plans.insert(grp[0], (grp, fused_node, slice_nodes, remaps));
    }

    if plans.is_empty() {
        return 0;
    }
    let fused_groups = plans.len();

    let mut remap: std::collections::HashMap<ValueId, ValueId> = std::collections::HashMap::new();
    for (_, _, _, rs) in plans.values() {
        for &(old, new) in rs {
            remap.insert(old, new);
        }
    }

    let old_nodes = std::mem::take(&mut graph.nodes);
    let mut new_nodes = Vec::with_capacity(old_nodes.len() + fused_groups * 3);
    for (ni, mut node) in old_nodes.into_iter().enumerate() {
        if removed_members.contains(&ni) {
            continue;
        }
        if let Some((_, fused_node, slice_nodes, _)) = plans.remove(&ni) {
            new_nodes.push(fused_node);
            new_nodes.extend(slice_nodes);
            continue;
        }
        for inp in &mut node.inputs {
            if let Some(&new_v) = remap.get(inp) {
                *inp = new_v;
            }
        }
        new_nodes.push(node);
    }
    for out in &mut graph.outputs {
        if let Some(&new_v) = remap.get(out) {
            *out = new_v;
        }
    }

    // Fold nested `Slice(axis=-1)` -> `Slice(axis=-1)` so egraph QKV + 4th sibling produces
    // 1 Linear + 4 direct Slices with zero extra copies.
    let mut slice_src: std::collections::HashMap<ValueId, (ValueId, i64, i64)> =
        std::collections::HashMap::new();
    for n in &mut new_nodes {
        if let Op::Slice {
            axis: -1,
            start,
            end,
        } = &mut n.op
        {
            if let Some(&(root, base_start, base_end)) = slice_src.get(&n.inputs[0]) {
                if *start >= 0 && *end >= *start && base_start + *end <= base_end {
                    n.inputs[0] = root;
                    *start += base_start;
                    *end += base_start;
                }
            }
            slice_src.insert(n.outputs[0], (n.inputs[0], *start, *end));
        }
    }

    // Rebind `ValueKind::Node(NodeId)` producer indices and prune dead intermediate slices.
    graph.nodes = new_nodes;
    for (idx, n) in graph.nodes.iter().enumerate() {
        let nid = NodeId(idx as u32);
        for &out in &n.outputs {
            graph.values[out.0 as usize].kind = ValueKind::Node(nid);
        }
    }
    super::canonical::eliminate_dead_nodes(graph);

    if let FuseWeights::Materialize(w_store) = &mut weights {
        for (name, tensor, old_names) in weight_updates {
            w_store.insert(name, tensor);
            for old in old_names {
                w_store.remove(&old);
            }
        }
    }

    fused_groups
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

        let out = g.add(
            "out_add",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[k, v],
            vec![1, 32],
            DType::F32,
        );
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
        let glu_rewrites = fuse_rule(
            &mut opt_g,
            FuseWeights::Materialize(&mut opt_w),
            Rule::GateUp,
        );
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
        assert_eq!(
            fuse_rule(&mut fg, FuseWeights::Materialize(&mut fw), Rule::Qkv),
            1
        );
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

    #[test]
    fn qk_mixed_quant_and_biased_qkv_and_sibling_linears_fuse() {
        use crate::graph::cpu::run_stateless;

        // 1. Mixed quant: Q4_K Q + Q4_K K + Q4_0 V -> Rule::Qk fuses Q+K, leaves V separate, CPU parity.
        let q4k = DType::Quant(crate::hardware::QuantFormat::GGUF_Q4_K_M);
        let q40 = DType::Quant(crate::hardware::QuantFormat::GGUF_Q4_0);
        let (g_qk, w_qk, inp_qk) = qkv_graph([q4k, q4k, q40]);
        let base_qk = run_stateless(&g_qk, &w_qk, inp_qk.clone()).unwrap();
        let (mut fg_qk, mut fw_qk) = (g_qk.clone(), w_qk.clone());
        crate::graph::passes::fuse(&mut fg_qk, FuseWeights::Materialize(&mut fw_qk)).unwrap();
        assert!(validate_ssa(&fg_qk).is_ok());
        let lin_names_qk: Vec<&str> = fg_qk
            .nodes
            .iter()
            .filter(|n| matches!(n.op, Op::Linear))
            .map(|n| n.name.as_str())
            .collect();
        assert_eq!(lin_names_qk, vec!["q.fused_qk", "v"]);
        let got_qk = run_stateless(&fg_qk, &fw_qk, inp_qk).unwrap();
        for (b, f) in base_qk.iter().zip(&got_qk) {
            assert_eq!(b.as_f32().unwrap(), f.as_f32().unwrap());
        }

        // 2. Biased QKV (Q biased, K unbiased, V biased, as in Qwen2): concatenates biases (0-filling K) and matches CPU.
        let mut g_b = Graph::new();
        let x = g_b.input("x", vec![2, 8], DType::F32);
        let gamma = g_b.weight("gamma", vec![8], DType::F32);
        let norm = g_b.add(
            "norm",
            Op::RmsNorm { eps: 1e-5 },
            &[x, gamma],
            vec![2, 8],
            DType::F32,
        );
        let wq = g_b.weight("wq", vec![8, 8], DType::F32);
        let bq = g_b.weight("bq", vec![8], DType::F32);
        let wk = g_b.weight("wk", vec![4, 8], DType::F32);
        let wv = g_b.weight("wv", vec![4, 8], DType::F32);
        let bv = g_b.weight("bv", vec![4], DType::F32);
        let q = g_b.add("q", Op::Linear, &[norm, wq, bq], vec![2, 8], DType::F32);
        let k = g_b.add("k", Op::Linear, &[norm, wk], vec![2, 4], DType::F32);
        let v = g_b.add("v", Op::Linear, &[norm, wv, bv], vec![2, 4], DType::F32);
        for o in [q, k, v] {
            g_b.output(o);
        }
        let mut w_b = Weights::new();
        w_b.insert("gamma", Tensor::F32(vec![1.0; 8]));
        w_b.insert(
            "wq",
            Tensor::F32((0..64).map(|i| (i as f32 - 20.0) * 0.03).collect()),
        );
        w_b.insert(
            "bq",
            Tensor::F32((0..8).map(|i| 0.25 + i as f32 * 0.1).collect()),
        );
        w_b.insert(
            "wk",
            Tensor::F32((0..32).map(|i| (i as f32 - 10.0) * 0.04).collect()),
        );
        w_b.insert(
            "wv",
            Tensor::F32((0..32).map(|i| (i as f32 - 15.0) * -0.02).collect()),
        );
        w_b.insert("bv", Tensor::F32(vec![-0.5, 0.25, -0.125, 0.75]));
        let inp_b = vec![Tensor::F32(
            (0..16).map(|i| (i as f32 - 7.0) * 0.2).collect(),
        )];
        let base_b = run_stateless(&g_b, &w_b, inp_b.clone()).unwrap();
        let (mut fg_b, mut fw_b) = (g_b.clone(), w_b.clone());
        assert_eq!(
            fuse_rule(&mut fg_b, FuseWeights::Materialize(&mut fw_b), Rule::Qkv),
            1
        );
        assert!(validate_ssa(&fg_b).is_ok());
        assert!(fw_b.get("q_fused_qkv_w_b").is_some());
        let got_b = run_stateless(&fg_b, &fw_b, inp_b).unwrap();
        for (b, f) in base_b.iter().zip(&got_b) {
            for (&bv, &fv) in b.as_f32().unwrap().iter().zip(f.as_f32().unwrap()) {
                assert!((bv - fv).abs() < 1e-5, "{bv} vs {fv}");
            }
        }

        // 3. 4 parallel DeltaNet sibling linears (`in_proj_{qkv,z,b,a}`) fuse into 1 Linear + 4 Slices!
        let mut g_d = Graph::new();
        let xd = g_d.input("x", vec![2, 16], DType::F32);
        let gd = g_d.weight("gd", vec![16], DType::F32);
        let nd = g_d.add(
            "nd",
            Op::RmsNorm { eps: 1e-5 },
            &[xd, gd],
            vec![2, 16],
            DType::F32,
        );
        let mut w_d = Weights::new();
        w_d.insert("gd", Tensor::F32(vec![1.0; 16]));
        for (idx, (nm, rows)) in [
            ("in_proj_qkv", 24usize),
            ("in_proj_z", 16),
            ("in_proj_b", 4),
            ("in_proj_a", 4),
        ]
        .into_iter()
        .enumerate()
        {
            let wn = format!("{nm}.w");
            let wid = g_d.weight(&wn, vec![rows, 16], DType::F32);
            w_d.insert(
                wn,
                Tensor::F32(
                    (0..rows * 16)
                        .map(|j| ((j + idx * 17) as f32 % 11.0 - 5.0) * 0.05)
                        .collect(),
                ),
            );
            let out = g_d.add(nm, Op::Linear, &[nd, wid], vec![2, rows], DType::F32);
            g_d.output(out);
        }
        let inp_d = vec![Tensor::F32(
            (0..32).map(|i| (i as f32 - 15.0) * 0.1).collect(),
        )];
        let base_d = run_stateless(&g_d, &w_d, inp_d.clone()).unwrap();
        let (mut fg_d, mut fw_d) = (g_d.clone(), w_d.clone());
        crate::graph::passes::fuse(&mut fg_d, FuseWeights::Materialize(&mut fw_d)).unwrap();
        assert!(validate_ssa(&fg_d).is_ok());
        let lin_cnt = fg_d
            .nodes
            .iter()
            .filter(|n| matches!(n.op, Op::Linear))
            .count();
        let slice_cnt = fg_d
            .nodes
            .iter()
            .filter(|n| matches!(n.op, Op::Slice { .. }))
            .count();
        assert_eq!((lin_cnt, slice_cnt), (1, 4));
        let got_d = run_stateless(&fg_d, &fw_d, inp_d.clone()).unwrap();
        for (b, f) in base_d.iter().zip(&got_d) {
            for (&bv, &fv) in b.as_f32().unwrap().iter().zip(f.as_f32().unwrap()) {
                assert!((bv - fv).abs() < 1e-5, "{bv} vs {fv}");
            }
        }
        // Reshaped ReuseOnly instance reuses the compiled program's fused weight without concatenating.
        let mut rg_d = g_d.clone();
        crate::graph::passes::fuse(&mut rg_d, FuseWeights::ReuseOnly(&fg_d)).unwrap();
        assert_eq!(
            rg_d.nodes
                .iter()
                .filter(|n| matches!(n.op, Op::Linear))
                .count(),
            1
        );
    }
}
