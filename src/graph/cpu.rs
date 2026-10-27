//! CPU executor for the graph IR, built on the `crate::model::nn` kernels.

use super::{Graph, Op, StateStore, Tensor, ValueId, ValueKind, Weights};
use crate::model::nn;

thread_local! {
    static CPU_ARENA: std::cell::RefCell<Vec<Vec<f32>>> = const { std::cell::RefCell::new(Vec::new()) };
}
/// Runs `graph` with stateless temporary buffers (discarded after completion).
pub fn run_stateless(
    graph: &Graph,
    weights: &Weights,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>, String> {
    crate::hardware::cpu_compute::enter("graph::cpu::run_stateless");
    let mut state = StateStore::new();
    run(graph, weights, &mut state, inputs)
}

/// Returns whether the CPU executor implements `op` for the given data type.
pub fn supports_op(op: &Op, _dtype: crate::graph::DType) -> bool {
    match op {
        Op::Embedding
        | Op::Linear
        | Op::RmsNorm { .. }
        | Op::LayerNorm { .. }
        | Op::SwiGlu
        | Op::GroupNorm { .. }
        | Op::InstanceNorm { .. }
        | Op::Reduce { .. }
        | Op::GeGlu
        | Op::Unary { .. }
        | Op::Binary { .. }
        | Op::Rope { .. }
        | Op::CachedAttention { .. }
        | Op::CausalConv1dCached { .. }
        | Op::DeltaRule { .. }
        | Op::Gather { .. }
        | Op::Scatter { .. }
        | Op::Pad { .. }
        | Op::AvgPool2d { .. }
        | Op::MeanPool
        | Op::L2Normalize
        | Op::Softmax { .. }
        | Op::TopK { .. }
        | Op::GatherCols
        | Op::MoeLinear { .. }
        | Op::Slice { .. }
        | Op::Conv2d { .. }
        | Op::Conv1d { .. }
        | Op::Conv3d { .. }
        | Op::RelPosBias { .. }
        | Op::StateReorder
        | Op::StateWrite
        | Op::QsaSelect { .. }
        | Op::QsaBias { .. }
        | Op::Attention { .. }
        | Op::Megakernel(_) => true,
        Op::Reshape { .. }
        | Op::Squeeze { .. }
        | Op::Unsqueeze { .. }
        | Op::Transpose { .. }
        | Op::Concat { .. }
        | Op::Tile { .. }
        | Op::Upsample { .. } => true,
        Op::MatMul { .. } => true,
    }
}

/// Runs `graph` with `inputs` bound positionally to `graph.inputs` and persistent `state` passed by reference.
pub fn run(
    graph: &Graph,
    weights: &Weights,
    state: &mut StateStore,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>, String> {
    crate::hardware::cpu_compute::enter("graph::cpu::run");
    if inputs.len() != graph.inputs.len() {
        return Err(format!(
            "graph has {} inputs, got {}",
            graph.inputs.len(),
            inputs.len()
        ));
    }
    let nv = graph.values.len();
    let mut slots: Vec<Option<Tensor>> = vec![None; nv];
    let mut bound: Vec<Option<&Tensor>> = vec![None; nv];
    for (&id, t) in graph.inputs.iter().zip(inputs) {
        let v = graph.value(id);
        if !v.shape.is_empty() && v.shape.iter().product::<usize>() != t.len() {
            return Err(format!(
                "input '{}' expects shape {:?}, got {} elements",
                v.name,
                v.shape,
                t.len()
            ));
        }
        slots[id.0 as usize] = Some(t);
    }
    let mut used_indices = vec![false; nv];
    for node in &graph.nodes {
        for &inp in &node.inputs {
            if (inp.0 as usize) < nv {
                used_indices[inp.0 as usize] = true;
            }
        }
    }
    for (i, v) in graph.values.iter().enumerate() {
        if !used_indices[i] {
            continue;
        }
        match &v.kind {
            ValueKind::Weight => {
                bound[i] = Some(
                    weights
                        .get(&v.name)
                        .ok_or_else(|| format!("weight '{}' is missing", v.name))?,
                );
            }
            ValueKind::State {
                name: state_name, ..
            } => {
                if let Some(t) = state.get(state_name) {
                    slots[i] = Some(t.clone());
                } else {
                    let cached_attn_only = graph.nodes.iter().all(|n| {
                        !n.inputs.contains(&ValueId(i as u32))
                            || matches!(n.op, Op::CachedAttention { .. })
                    });
                    let numel = if v.shape.is_empty() {
                        0
                    } else if cached_attn_only && v.shape.len() == 2 && v.shape[0] > 4096 {
                        4096 * v.shape[1]
                    } else {
                        v.shape.iter().product::<usize>()
                    };
                    slots[i] = Some(Tensor::F32(vec![0.0f32; numel]));
                }
            }
            _ => {}
        }
    }
    // Last node index reading each value; graph outputs are never dropped.
    let mut last_use = vec![0usize; nv];
    for (i, node) in graph.nodes.iter().enumerate() {
        for v in &node.inputs {
            last_use[v.0 as usize] = i;
        }
    }
    for v in &graph.outputs {
        last_use[v.0 as usize] = usize::MAX;
    }

    for (i, node) in graph.nodes.iter().enumerate() {
        let name = &node.name;
        let ins = &node.inputs;
        let last_dim = |k: usize| graph.value(ins[k]).shape.last().copied().unwrap_or(0);

        let mut out_u32: Option<Vec<u32>> = None;
        let out = match &node.op {
            Op::Embedding => {
                let table_slot = slot(&bound, &slots, ins, name, 0)?;
                let ids = operand_u32(&bound, &slots, ins, name, 1)?;
                let d = last_dim(0);
                if d == 0 {
                    return Err(format!("node '{name}': embedding table has unknown width"));
                }
                let mut out = Vec::with_capacity(ids.len() * d);
                match table_slot {
                    Tensor::F32(table) => {
                        let rows = table.len() / d;
                        for (r, &id) in ids.iter().enumerate() {
                            let id = id as usize;
                            if id >= rows {
                                return Err(format!(
                                    "node '{name}': row {r} id {id} out of range for table of {rows} rows"
                                ));
                            }
                            out.extend_from_slice(&table[id * d..(id + 1) * d]);
                        }
                    }
                    Tensor::Mat(m) => {
                        if m.cols != d {
                            return Err(format!(
                                "node '{name}': width {d} != table cols {}",
                                m.cols
                            ));
                        }
                        out.resize(ids.len() * d, 0.0);
                        for (r, &id) in ids.iter().enumerate() {
                            let id = id as usize;
                            if id >= m.rows {
                                return Err(format!(
                                    "node '{name}': row {r} id {id} out of range for table of {} rows",
                                    m.rows
                                ));
                            }
                            m.decode_row_f32(id, &mut out[r * d..(r + 1) * d]);
                        }
                    }
                    Tensor::BF16(v) => {
                        let rows = v.len() / d;
                        for (r, &id) in ids.iter().enumerate() {
                            let id = id as usize;
                            if id >= rows {
                                return Err(format!(
                                    "node '{name}': row {r} id {id} out of range for table of {rows} rows"
                                ));
                            }
                            out.extend(
                                v[id * d..(id + 1) * d]
                                    .iter()
                                    .map(|&x| crate::model::bf16_to_f32(x)),
                            );
                        }
                    }
                    Tensor::F16(v) => {
                        let rows = v.len() / d;
                        for (r, &id) in ids.iter().enumerate() {
                            let id = id as usize;
                            if id >= rows {
                                return Err(format!(
                                    "node '{name}': row {r} id {id} out of range for table of {rows} rows"
                                ));
                            }
                            out.extend(
                                v[id * d..(id + 1) * d]
                                    .iter()
                                    .map(|&x| crate::model::f16_to_f32(x)),
                            );
                        }
                    }
                    other => {
                        let full = other.to_f32();
                        let rows = full.len() / d;
                        for (r, &id) in ids.iter().enumerate() {
                            let id = id as usize;
                            if id >= rows {
                                return Err(format!(
                                    "node '{name}': row {r} id {id} out of range for table of {rows} rows"
                                ));
                            }
                            out.extend_from_slice(&full[id * d..(id + 1) * d]);
                        }
                    }
                }
                out
            }
            Op::Linear => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let w_slot = slot(&bound, &slots, ins, name, 1)?;
                let shape = &graph.value(ins[1]).shape;
                let [out_f, in_f] = shape[..] else {
                    return Err(format!(
                        "node '{name}': weight shape {shape:?} is not [out, in]"
                    ));
                };
                if in_f == 0 || x.len() % in_f != 0 {
                    return Err(format!(
                        "node '{name}': x of {} elements does not match weight {shape:?}",
                        x.len()
                    ));
                }
                let bias = if ins.len() > 2 {
                    Some(operand(&bound, &slots, ins, name, 2)?)
                } else {
                    None
                };
                if bias.is_some_and(|b| b.len() != out_f) {
                    return Err(format!("node '{name}': bias is not [{out_f}]"));
                }
                let rows = x.len() / in_f;
                dispatch_linear(x, w_slot, bias, in_f, out_f, rows, name)?
            }
            Op::MatMul {
                transpose_a,
                transpose_b,
            } => {
                let (a, b) = (
                    operand(&bound, &slots, ins, name, 0)?,
                    operand(&bound, &slots, ins, name, 1)?,
                );
                matmul_ref(graph, ins, a, b, *transpose_a, *transpose_b, name)?
            }
            Op::Megakernel(desc) => {
                match &desc.kind {
                    super::MegakernelKind::Mlp {
                        hidden_dim,
                        intermediate_dim,
                        eps,
                        is_geglu,
                        has_post_norm,
                    } => {
                        let x = operand(&bound, &slots, ins, name, 0)?;
                        let norm_w = operand(&bound, &slots, ins, name, 1)?;
                        let gate_slot = slot(&bound, &slots, ins, name, 2)?;
                        let up_slot = slot(&bound, &slots, ins, name, 3)?;
                        let down_slot = slot(&bound, &slots, ins, name, 4)?;
                        let post_norm_w = if *has_post_norm {
                            Some(operand(&bound, &slots, ins, name, 5)?)
                        } else {
                            None
                        };
                        let d = *hidden_dim;
                        let inter = *intermediate_dim;
                        let rows = x.len() / d.max(1);

                        // 1. RMSNorm
                        let mut normed_x = vec![0.0f32; x.len()];
                        for (x_chunk, out_chunk) in x.chunks(d).zip(normed_x.chunks_mut(d)) {
                            crate::model::simd::vector_rmsnorm(x_chunk, norm_w, out_chunk, *eps);
                        }

                        // 2. Gate & Up projections
                        let gate_act =
                            dispatch_linear(&normed_x, gate_slot, None, d, inter, rows, name)?;
                        let up_act =
                            dispatch_linear(&normed_x, up_slot, None, d, inter, rows, name)?;

                        // 3. SwiGLU / GeGLU
                        let mut act = vec![0.0f32; gate_act.len()];
                        if *is_geglu {
                            act.copy_from_slice(&gate_act);
                            nn::gelu_tanh(&mut act);
                            for (o, &u) in act.iter_mut().zip(&up_act) {
                                *o *= u;
                            }
                        } else {
                            for (o, (&g, &u)) in act.iter_mut().zip(gate_act.iter().zip(&up_act)) {
                                let sig = 1.0 / (1.0 + (-g).exp());
                                *o = (g * sig) * u;
                            }
                        }

                        // 4. Down projection
                        let mut down_act =
                            dispatch_linear(&act, down_slot, None, inter, d, rows, name)?;

                        if let Some(post_w) = post_norm_w {
                            let mut normed_down = vec![0.0f32; down_act.len()];
                            for (d_chunk, o_chunk) in
                                down_act.chunks(d).zip(normed_down.chunks_mut(d))
                            {
                                crate::model::simd::vector_rmsnorm(d_chunk, post_w, o_chunk, *eps);
                            }
                            down_act = normed_down;
                        }

                        // 5. In-place residual add
                        let mut out = x.to_vec();
                        for (o, &d_val) in out.iter_mut().zip(&down_act) {
                            *o += d_val;
                        }
                        out
                    }
                }
            }
            Op::RmsNorm { eps } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                // Without gamma the unit weight spans the last dim.
                let ones;
                let g = if ins.len() > 1 {
                    operand(&bound, &slots, ins, name, 1)?
                } else {
                    ones = vec![1.0f32; last_dim(0)];
                    &ones[..]
                };
                let d = g.len();
                if d == 0 || x.len() % d != 0 {
                    return Err(format!("node '{name}': RmsNorm shapes mismatch"));
                }
                let mut out = vec![0.0f32; x.len()];
                for (x_chunk, out_chunk) in x.chunks(d).zip(out.chunks_mut(d)) {
                    crate::model::simd::vector_rmsnorm(x_chunk, g, out_chunk, *eps);
                }
                out
            }
            Op::SwiGlu => {
                let gate = operand(&bound, &slots, ins, name, 0)?;
                let up = operand(&bound, &slots, ins, name, 1)?;
                if gate.len() != up.len() {
                    return Err(format!("node '{name}': SwiGlu lengths mismatch"));
                }
                let mut out = vec![0.0f32; gate.len()];
                for (o, (&g, &u)) in out.iter_mut().zip(gate.iter().zip(up)) {
                    let sig = 1.0 / (1.0 + (-g).exp());
                    *o = (g * sig) * u;
                }
                out
            }
            Op::GeGlu => {
                let gate = operand(&bound, &slots, ins, name, 0)?;
                let up = operand(&bound, &slots, ins, name, 1)?;
                if gate.len() != up.len() {
                    return Err(format!("node '{name}': GeGlu lengths mismatch"));
                }
                let mut out = gate.to_vec();
                nn::gelu_tanh(&mut out);
                for (o, &u) in out.iter_mut().zip(up) {
                    *o *= u;
                }
                out
            }
            Op::Unary { kind } => {
                let mut x = take0(&bound, &mut slots, ins, name, &last_use, i)?;
                match kind {
                    super::UnaryKind::Exp => x.iter_mut().for_each(|v| *v = v.exp()),
                    super::UnaryKind::Sigmoid => {
                        x.iter_mut().for_each(|v| *v = 1.0 / (1.0 + (-*v).exp()))
                    }
                    super::UnaryKind::Tanh => x.iter_mut().for_each(|v| *v = v.tanh()),
                    super::UnaryKind::Softplus => {
                        x.iter_mut().for_each(|v| {
                            *v = if *v > 20.0 { *v } else { (*v).exp().ln_1p() };
                        });
                    }
                    super::UnaryKind::Neg => x.iter_mut().for_each(|v| *v = -*v),
                    super::UnaryKind::Sqrt => x.iter_mut().for_each(|v| *v = v.sqrt()),
                    super::UnaryKind::Rsqrt => x.iter_mut().for_each(|v| *v = 1.0 / v.sqrt()),
                    super::UnaryKind::Abs => x.iter_mut().for_each(|v| *v = v.abs()),
                    super::UnaryKind::Relu => x.iter_mut().for_each(|v| *v = v.max(0.0)),
                    super::UnaryKind::Relu2 => x.iter_mut().for_each(|v| {
                        let r = v.max(0.0);
                        *v = r * r;
                    }),
                    super::UnaryKind::Silu => {
                        x.iter_mut().for_each(|v| *v = *v / (1.0 + (-*v).exp()))
                    }
                    super::UnaryKind::GeluErf => nn::gelu(&mut x),
                    super::UnaryKind::GeluTanh => {
                        const SQRT_2_OVER_PI: f32 = 0.7978845608;
                        x.iter_mut().for_each(|v| {
                            let cube = 0.044715 * *v * *v * *v;
                            *v = 0.5 * *v * (1.0 + (SQRT_2_OVER_PI * (*v + cube)).tanh());
                        });
                    }
                    super::UnaryKind::Scale(s) => x.iter_mut().for_each(|v| *v *= s),
                    super::UnaryKind::Clamp(min, max) => {
                        x.iter_mut().for_each(|v| *v = v.clamp(*min, *max))
                    }
                    super::UnaryKind::SoftCap(cap) => {
                        x.iter_mut().for_each(|v| *v = cap * (*v / cap).tanh())
                    }
                    super::UnaryKind::Ln => x.iter_mut().for_each(|v| *v = v.ln()),
                    super::UnaryKind::RoundEven => {
                        x.iter_mut().for_each(|v| *v = v.round_ties_even())
                    }
                    super::UnaryKind::Sign => x.iter_mut().for_each(|v| {
                        *v = if *v > 0.0 {
                            1.0
                        } else if *v < 0.0 {
                            -1.0
                        } else {
                            0.0
                        }
                    }),
                }
                x
            }
            Op::Binary { kind } => {
                let a = operand(&bound, &slots, ins, name, 0)?;
                let b = operand(&bound, &slots, ins, name, 1)?;
                let op_fn: fn(f32, f32) -> f32 = match kind {
                    super::BinaryKind::Add => |x, y| x + y,
                    super::BinaryKind::Sub => |x, y| x - y,
                    super::BinaryKind::Mul => |x, y| x * y,
                    super::BinaryKind::Div => |x, y| x / y,
                    super::BinaryKind::Max => |x, y| x.max(y),
                    super::BinaryKind::Min => |x, y| x.min(y),
                };
                let (a_shape, b_shape) = (&graph.value(ins[0]).shape, &graph.value(ins[1]).shape);
                broadcast_np(a, a_shape, b, b_shape, declared_len(graph, node), op_fn)
                    .or_else(|| {
                        (*kind == super::BinaryKind::Mul)
                            .then(|| row_scale(a, a_shape, b, b_shape))
                            .flatten()
                    })
                    .or_else(|| tile_binary(a, b, op_fn))
                    .ok_or_else(|| {
                        format!(
                            "node '{name}': Binary shape mismatch {} vs {}",
                            a.len(),
                            b.len()
                        )
                    })?
            }
            Op::Softmax { axis } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let shape = &graph.value(ins[0]).shape;
                let shape = if shape.is_empty() {
                    vec![x.len()]
                } else {
                    shape.clone()
                };
                let ax = if *axis < 0 {
                    *axis + shape.len() as isize
                } else {
                    *axis
                };
                if ax < 0 || ax >= shape.len() as isize {
                    return Err(format!("node '{name}': bad Softmax axis {axis}"));
                }
                let ax = ax as usize;
                let (outer, n, inner) = (
                    shape[..ax].iter().product::<usize>(),
                    shape[ax],
                    shape[ax + 1..].iter().product::<usize>(),
                );
                let mut out = vec![0.0f32; x.len()];
                for o in 0..outer {
                    for i in 0..inner {
                        let at = |k: usize| (o * n + k) * inner + i;
                        let max_val = (0..n).map(|k| x[at(k)]).fold(f32::NEG_INFINITY, f32::max);
                        let mut sum_e = 0.0f32;
                        for k in 0..n {
                            let e = (x[at(k)] - max_val).exp();
                            out[at(k)] = e;
                            sum_e += e;
                        }
                        let inv = 1.0 / sum_e.max(1e-12);
                        for k in 0..n {
                            out[at(k)] *= inv;
                        }
                    }
                }
                out
            }
            Op::TopK { k } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let cols = last_dim(0);
                if *k == 0 || *k > cols || x.len() % cols != 0 {
                    return Err(format!(
                        "node '{name}': TopK needs 1 <= k <= row width, got k={k} over {cols} columns of {} values",
                        x.len()
                    ));
                }
                out_u32 = Some(
                    x.chunks_exact(cols)
                        .flat_map(|row| topk_row(row, *k))
                        .collect(),
                );
                Vec::new()
            }
            Op::GatherCols => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let idx = operand_u32(&bound, &slots, ins, name, 1)?;
                let (cols, k) = (last_dim(0), last_dim(1));
                if cols == 0 || k == 0 || x.len() % cols != 0 || idx.len() % k != 0 {
                    return Err(format!(
                        "node '{name}': GatherCols needs x [rows, n] and idx [rows, k]"
                    ));
                }
                // One x row serves every idx row (a shared table such as per-expert scales).
                let shared = x.len() == cols;
                if !shared && x.len() / cols != idx.len() / k {
                    return Err(format!(
                        "node '{name}': GatherCols needs x and idx over the same rows, or a single x row"
                    ));
                }
                let mut out = Vec::with_capacity(idx.len());
                for (r, row) in idx.chunks_exact(k).enumerate() {
                    for &c in row {
                        let v = x[if shared { 0 } else { r * cols }..][..cols]
                            .get(c as usize)
                            .ok_or_else(|| {
                                format!("node '{name}': column {c} out of range for {cols} columns")
                            })?;
                        out.push(*v);
                    }
                }
                out
            }
            Op::MoeLinear { experts, top_k } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let bank = slot(&bound, &slots, ins, name, 1)?;
                let ids = operand_u32(&bound, &slots, ins, name, 2)?;
                let [bank_rows, in_f] = graph.value(ins[1]).shape[..] else {
                    return Err(format!(
                        "node '{name}': the expert bank is not [experts * out, in]"
                    ));
                };
                if *experts == 0
                    || *top_k == 0
                    || in_f == 0
                    || bank_rows % experts != 0
                    || ids.len() % top_k != 0
                {
                    return Err(format!(
                        "node '{name}': MoeLinear needs experts, top_k >= 1 and a [experts * out, in] bank with [m, top_k] ids"
                    ));
                }
                let (out_f, slots_n) = (bank_rows / experts, ids.len());
                let x_rows = x.len() / in_f;
                if x.len() % in_f != 0 || (x_rows != slots_n && x_rows * top_k != slots_n) {
                    return Err(format!(
                        "node '{name}': x of {} values is neither [m, {in_f}] nor [m * {top_k}, {in_f}] for {slots_n} slots",
                        x.len()
                    ));
                }
                moe_linear_cpu(x, bank, ids, [*experts, *top_k, in_f, out_f], name)?
            }
            Op::Rope {
                head_dim,
                rotary_dim,
                variant,
                mscale,
            } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let pos = operand_u32(&bound, &slots, ins, name, 1)?;
                let inv_freq = operand(&bound, &slots, ins, name, 2)?;
                let half = rotary_dim / 2;
                if inv_freq.len() != half {
                    return Err(format!(
                        "node '{name}': inv_freq has {} entries, expected {half}",
                        inv_freq.len()
                    ));
                }

                let mut out = x.to_vec();
                let n = if !pos.is_empty() {
                    match variant {
                        super::RopeVariant::MRope { .. } => pos.len() / 3,
                        _ => pos.len(),
                    }
                } else {
                    1
                };
                let total_dim = x.len() / n.max(1);
                let hd = *head_dim;
                let heads = if hd > 0 { total_dim / hd } else { 1 };

                for seq_idx in 0..n {
                    let row_offset = seq_idx * total_dim;
                    for h in 0..heads {
                        let head_offset = row_offset + h * hd;
                        for j in 0..half {
                            let freq = inv_freq[j];
                            if freq == 0.0 && *mscale == 1.0 {
                                continue;
                            }
                            let p = match variant {
                                super::RopeVariant::MRope {
                                    sections,
                                    interleaved,
                                    ..
                                } => {
                                    let p_t = pos.get(seq_idx * 3).copied().unwrap_or(0);
                                    let p_h = pos.get(seq_idx * 3 + 1).copied().unwrap_or(0);
                                    let p_w = pos.get(seq_idx * 3 + 2).copied().unwrap_or(0);
                                    if *interleaved {
                                        if j % 3 == 1 && (j as u32) < 3 * sections[1] {
                                            p_h
                                        } else if j % 3 == 2 && (j as u32) < 3 * sections[2] {
                                            p_w
                                        } else {
                                            p_t
                                        }
                                    } else {
                                        let s0 = sections[0] as usize;
                                        let s1 = sections[1] as usize;
                                        if j < s0 {
                                            p_t
                                        } else if j < s0 + s1 {
                                            p_h
                                        } else {
                                            p_w
                                        }
                                    }
                                }
                                _ => pos.get(seq_idx).copied().unwrap_or(0),
                            };

                            let theta = (p as f32) * freq;
                            let cos = theta.cos() * *mscale;
                            let sin = theta.sin() * *mscale;

                            match variant {
                                super::RopeVariant::Interleaved => {
                                    let i0 = head_offset + 2 * j;
                                    let i1 = i0 + 1;
                                    if i1 < out.len() {
                                        let x0 = out[i0];
                                        let x1 = out[i1];
                                        out[i0] = x0 * cos - x1 * sin;
                                        out[i1] = x0 * sin + x1 * cos;
                                    }
                                }
                                super::RopeVariant::HalfSplit => {
                                    let i0 = head_offset + j;
                                    let i1 = head_offset + j + half;
                                    if i1 < out.len() {
                                        let x0 = out[i0];
                                        let x1 = out[i1];
                                        out[i0] = x0 * cos - x1 * sin;
                                        out[i1] = x0 * sin + x1 * cos;
                                    }
                                }
                                super::RopeVariant::MRope { pairs, .. } => {
                                    let (i0, i1) = match pairs {
                                        super::PairLayout::Adjacent => {
                                            (head_offset + 2 * j, head_offset + 2 * j + 1)
                                        }
                                        super::PairLayout::Half => {
                                            (head_offset + j, head_offset + j + half)
                                        }
                                    };
                                    if i1 < out.len() {
                                        let x0 = out[i0];
                                        let x1 = out[i1];
                                        out[i0] = x0 * cos - x1 * sin;
                                        out[i1] = x0 * sin + x1 * cos;
                                    }
                                }
                            }
                        }
                    }
                }
                out
            }
            Op::RelPosBias {
                buckets,
                max_distance,
                bidirectional,
                heads,
            } => {
                let table = operand(&bound, &slots, ins, name, 0)?;
                let q_pos = operand_u32(&bound, &slots, ins, name, 1)?;
                let k_pos = operand_u32(&bound, &slots, ins, name, 2)?;
                let (n, m, h) = (q_pos.len(), k_pos.len(), *heads);
                let nb = if *bidirectional {
                    buckets / 2
                } else {
                    *buckets
                };
                if nb < 2 || *max_distance <= nb / 2 || table.len() != buckets * h {
                    return Err(format!(
                        "node '{name}': RelPosBias needs table [{buckets}, {h}] ({} elements), \
                         at least 2 buckets per direction and max_distance > {}",
                        table.len(),
                        nb / 2
                    ));
                }
                let mut out = vec![0.0f32; h * n * m];
                for (qi, &qp) in q_pos.iter().enumerate() {
                    for (ki, &kp) in k_pos.iter().enumerate() {
                        let b = rel_pos_bucket(
                            kp as i64 - qp as i64,
                            *bidirectional,
                            *buckets,
                            *max_distance,
                        );
                        for head in 0..h {
                            out[head * n * m + qi * m + ki] = table[b * h + head];
                        }
                    }
                }
                out
            }
            Op::StateReorder => {
                let state_tensor = operand(&bound, &slots, ins, name, 0)?;
                let parent = operand_u32(&bound, &slots, ins, name, 1)?;
                let batch = parent.len();
                let row_len = state_tensor.len() / batch.max(1);
                let mut out = vec![0.0f32; state_tensor.len()];
                for (b, &p) in parent.iter().enumerate() {
                    let p_idx = p as usize;
                    if p_idx < batch {
                        out[b * row_len..(b + 1) * row_len]
                            .copy_from_slice(&state_tensor[p_idx * row_len..(p_idx + 1) * row_len]);
                    }
                }
                out
            }
            Op::StateWrite => {
                let ValueKind::State { name: sname, .. } = &graph.value(ins[0]).kind else {
                    return Err(format!(
                        "node '{name}': StateWrite operand 0 must be a graph state"
                    ));
                };
                let w = last_dim(0);
                let mut buf = operand(&bound, &slots, ins, name, 0)?.to_vec();
                let rows = operand(&bound, &slots, ins, name, 1)?;
                let pos0 = operand_u32(&bound, &slots, ins, name, 2)?
                    .first()
                    .copied()
                    .ok_or_else(|| format!("node '{name}': StateWrite pos0 is empty"))?
                    as usize;
                if w == 0 || last_dim(1) != w || rows.len() % w != 0 {
                    return Err(format!(
                        "node '{name}': StateWrite rows are not [m, {w}] like the state"
                    ));
                }
                let m = rows.len() / w;
                if (pos0 + m) * w > buf.len() {
                    return Err(format!(
                        "node '{name}': writing rows {pos0}..{} past the state's {} rows",
                        pos0 + m,
                        buf.len() / w
                    ));
                }
                buf[pos0 * w..(pos0 + m) * w].copy_from_slice(rows);
                // Later nodes reading this state (any value id) see the write.
                for (j, v) in graph.values.iter().enumerate() {
                    if matches!(&v.kind, ValueKind::State { name: n, .. } if n == sname)
                        && slots[j].is_some()
                    {
                        slots[j] = Some(Tensor::F32(buf.clone()));
                    }
                }
                state.insert(sname, Tensor::F32(buf));
                out_u32 = Some(vec![(pos0 + m) as u32]);
                Vec::new()
            }
            Op::QsaSelect { ratio, top_blocks } | Op::QsaBias { ratio, top_blocks } => {
                let score = operand(&bound, &slots, ins, name, 0)?;
                let pos = operand_u32(&bound, &slots, ins, name, 1)?;
                let (ids, bias) = qsa_select(score, last_dim(0), pos, *ratio, *top_blocks)
                    .map_err(|e| format!("node '{name}': {e}"))?;
                if matches!(node.op, Op::QsaSelect { .. }) {
                    out_u32 = Some(ids);
                    Vec::new()
                } else {
                    bias
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
                let q = operand(&bound, &slots, ins, name, 0)?;
                let k_in = operand(&bound, &slots, ins, name, 1)?;
                let v_in = operand(&bound, &slots, ins, name, 2)?;
                let hd = *head_dim;
                let vhd = *v_head_dim;
                let h = *heads;
                let kvh = *kv_heads;
                let rep = h / kvh.max(1);
                let nq = q.len() / (h * hd).max(1);

                let (k_buf, v_buf, pos_offset) = if ins.len() >= 6 {
                    let pos = operand_u32(&bound, &slots, ins, name, 5)
                        .map(|p| p.first().copied().unwrap_or(0) as usize)
                        .unwrap_or(0);
                    let k_state = operand(&bound, &slots, ins, name, 3)?;
                    let v_state = operand(&bound, &slots, ins, name, 4)?;
                    let mut k_all = k_state.to_vec();
                    let mut v_all = v_state.to_vec();
                    // A `write: false` layer reads another layer's cache, which already holds this
                    // position; its `k`/`v` operands are placeholders and must not overwrite it.
                    if *write {
                        let end_k = (pos * kvh * hd) + k_in.len();
                        if end_k > k_all.len() {
                            k_all.resize(end_k, 0.0);
                        }
                        k_all[end_k - k_in.len()..end_k].copy_from_slice(k_in);
                        let end_v = (pos * kvh * vhd) + v_in.len();
                        if end_v > v_all.len() {
                            v_all.resize(end_v, 0.0);
                        }
                        v_all[end_v - v_in.len()..end_v].copy_from_slice(v_in);
                    }
                    (k_all, v_all, pos)
                } else {
                    (k_in.to_vec(), v_in.to_vec(), 0)
                };
                let vis = if ins.len() >= 7 {
                    Some(operand_u32(&bound, &slots, ins, name, 6)?)
                } else {
                    None
                };
                let nk = (pos_offset + nq)
                    .max(vis.map_or(0, |v| v.iter().copied().max().unwrap_or(0) as usize))
                    .min(k_buf.len() / (kvh * hd).max(1));
                let mut out = vec![0.0f32; nq * h * vhd];
                for qi in 0..nq {
                    let current_pos = pos_offset + qi;
                    for head in 0..h {
                        let kv_head = head / rep;
                        let q_vec = &q[qi * h * hd + head * hd..][..hd];
                        let mut scores = Vec::with_capacity(nk);
                        for ki in 0..nk {
                            if *causal && ki >= vis.map_or(current_pos + 1, |v| v[qi] as usize) {
                                scores.push(f32::NEG_INFINITY);
                                continue;
                            }
                            if let Some(w) = window {
                                if ki + *w <= current_pos {
                                    scores.push(f32::NEG_INFINITY);
                                    continue;
                                }
                            }
                            let k_vec = &k_buf[ki * kvh * hd + kv_head * hd..][..hd];
                            let dot: f32 = q_vec.iter().zip(k_vec).map(|(&a, &b)| a * b).sum();
                            let mut s = dot * scale;
                            if let Some(cap) = soft_cap {
                                s = cap * (s / cap).tanh();
                            }
                            scores.push(s);
                        }
                        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum_exp = 0.0f32;
                        let mut weights = Vec::with_capacity(nk);
                        for &s in &scores {
                            if s == f32::NEG_INFINITY {
                                weights.push(0.0f32);
                            } else {
                                let exp = (s - max_score).exp();
                                sum_exp += exp;
                                weights.push(exp);
                            }
                        }
                        if sum_exp > 0.0 {
                            weights.iter_mut().for_each(|w| *w /= sum_exp);
                        }
                        let out_slice = &mut out[qi * h * vhd + head * vhd..][..vhd];
                        for (ki, &w) in weights.iter().enumerate() {
                            if w == 0.0 {
                                continue;
                            }
                            let v_vec = &v_buf[ki * kvh * vhd + kv_head * vhd..][..vhd];
                            for (o, &val) in out_slice.iter_mut().zip(v_vec) {
                                *o += w * val;
                            }
                        }
                    }
                }
                if *write && ins.len() >= 6 {
                    for (slot_id, buf) in [(ins[3], &k_buf), (ins[4], &v_buf)] {
                        if let ValueKind::State { name: sname, .. } = &graph.value(slot_id).kind {
                            state.insert(sname, Tensor::F32(buf.clone()));
                            // Later readers of this state (an `Attention` over the whole cache,
                            // `write: false` layers with their own value ids) see this step's write.
                            for (j, v) in graph.values.iter().enumerate() {
                                if matches!(&v.kind, ValueKind::State { name: n, .. } if n == sname)
                                    && slots[j].is_some()
                                {
                                    slots[j] = Some(Tensor::F32(buf.clone()));
                                }
                            }
                        }
                    }
                }
                out
            }
            Op::CausalConv1dCached { kernel, dilation } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let w = operand(&bound, &slots, ins, name, 1)?;
                let k = *kernel;
                let d = *dilation;
                let channels = w.len() / k.max(1);
                let n = x.len() / channels.max(1);
                // a dilated causal conv reaches (k-1)*d samples back
                let hist = (k - 1) * d;
                let mut window = if ins.len() >= 3 {
                    operand(&bound, &slots, ins, name, 2)
                        .map(<[f32]>::to_vec)
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                if window.len() != hist * channels {
                    window = vec![0.0f32; hist * channels];
                }
                window.extend_from_slice(x);
                let mut out = vec![0.0f32; n * channels];
                for i in 0..n {
                    for ch in 0..channels {
                        let mut acc = 0.0f32;
                        let w_ch = &w[ch * k..(ch + 1) * k];
                        for (tap, &weight) in w_ch.iter().enumerate() {
                            let offset_step = (k - 1 - tap) * d;
                            let sample_idx = hist + i;
                            if sample_idx >= offset_step {
                                acc += weight * window[(sample_idx - offset_step) * channels + ch];
                            }
                        }
                        out[i * channels + ch] = acc;
                    }
                }
                let new_state = window[n * channels..].to_vec();
                if ins.len() >= 3 {
                    if let ValueKind::State { name: sname, .. } = &graph.value(ins[2]).kind {
                        state.insert(sname, Tensor::F32(new_state));
                    }
                }
                out
            }
            Op::DeltaRule {
                key_heads,
                val_heads,
                key_dim,
                val_dim,
            } => {
                let q = operand(&bound, &slots, ins, name, 0)?;
                let k = operand(&bound, &slots, ins, name, 1)?;
                let v = operand(&bound, &slots, ins, name, 2)?;
                let beta = operand(&bound, &slots, ins, name, 3)?;
                let log_decay = operand(&bound, &slots, ins, name, 4)?;
                let kh = *key_heads;
                let vh = *val_heads;
                let kd = *key_dim;
                let vd = *val_dim;
                let rep = vh / kh.max(1);
                let n = q.len() / (kh * kd).max(1);
                let mut out = vec![0.0f32; n * vh * vd];

                let mut ssm_state = if ins.len() > 5 {
                    operand(&bound, &slots, ins, name, 5)
                        .map(<[f32]>::to_vec)
                        .unwrap_or_else(|_| vec![0.0f32; vh * kd * vd])
                } else {
                    vec![0.0f32; vh * kd * vd]
                };
                if ssm_state.len() != vh * kd * vd {
                    ssm_state = vec![0.0f32; vh * kd * vd];
                }

                for h in 0..vh {
                    let k_h = h / rep;
                    let state_slice = &mut ssm_state[h * kd * vd..(h + 1) * kd * vd];
                    for t in 0..n {
                        let mut q_vec = q[t * kh * kd + k_h * kd..][..kd].to_vec();
                        let mut k_vec = k[t * kh * kd + k_h * kd..][..kd].to_vec();
                        let v_t = &v[t * vh * vd + h * vd..][..vd];
                        let b = beta.get(t * vh + h).copied().unwrap_or(1.0);
                        let ld = log_decay.get(t * vh + h).copied().unwrap_or(0.0);
                        let decay = ld.exp();

                        let q_norm = q_vec.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
                        let q_scale = (1.0 / (kd as f32).sqrt()) / q_norm;
                        q_vec.iter_mut().for_each(|x| *x *= q_scale);

                        let k_norm = k_vec.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
                        let k_scale = 1.0 / k_norm;
                        k_vec.iter_mut().for_each(|x| *x *= k_scale);

                        let mut kv = vec![0.0f32; vd];
                        for i in 0..kd {
                            let k_val = k_vec[i];
                            let row = &state_slice[i * vd..(i + 1) * vd];
                            for (kv_elem, &s) in kv.iter_mut().zip(row) {
                                *kv_elem += s * k_val;
                            }
                        }
                        let mut delta = vec![0.0f32; vd];
                        for (d, (&vt, &kvt)) in delta.iter_mut().zip(v_t.iter().zip(&kv)) {
                            *d = b * (vt - decay * kvt);
                        }
                        for i in 0..kd {
                            let k_val = k_vec[i];
                            let row = &mut state_slice[i * vd..(i + 1) * vd];
                            for (s, &d) in row.iter_mut().zip(&delta) {
                                *s = decay * *s + k_val * d;
                            }
                        }
                        let out_slice = &mut out[t * vh * vd + h * vd..][..vd];
                        for i in 0..kd {
                            let q_val = q_vec[i];
                            let row = &state_slice[i * vd..(i + 1) * vd];
                            for (o, &s) in out_slice.iter_mut().zip(row) {
                                *o += s * q_val;
                            }
                        }
                    }
                }
                if ins.len() > 5 {
                    if let ValueKind::State { name: sname, .. } = &graph.value(ins[5]).kind {
                        state.insert(sname, Tensor::F32(ssm_state));
                    }
                }
                out
            }
            Op::Gather { axis } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let indices = operand_u32(&bound, &slots, ins, name, 1)?;
                let x_shape = &graph.value(ins[0]).shape;
                let ax = if *axis < 0 {
                    (x_shape.len() as isize + *axis) as usize
                } else {
                    *axis as usize
                };
                let outer: usize = x_shape[..ax.min(x_shape.len())].iter().product();
                let dim_size = x_shape.get(ax).copied().unwrap_or(x.len());
                let inner: usize = if ax + 1 < x_shape.len() {
                    x_shape[ax + 1..].iter().product()
                } else {
                    1
                };
                let mut out = Vec::with_capacity(outer * indices.len() * inner);
                for o in 0..outer.max(1) {
                    for &idx in indices {
                        let i = idx as usize;
                        if i < dim_size {
                            let start = o * dim_size * inner + i * inner;
                            if start + inner <= x.len() {
                                out.extend_from_slice(&x[start..start + inner]);
                            }
                        }
                    }
                }
                out
            }
            Op::Slice { axis, start, end } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let x_shape = &graph.value(ins[0]).shape;
                if x_shape.is_empty() {
                    return Err(format!(
                        "node '{name}': cannot slice operand with empty shape"
                    ));
                }
                let ax = if *axis < 0 {
                    let adjusted = x_shape.len() as isize + *axis;
                    if adjusted < 0 {
                        return Err(format!(
                            "node '{name}': slice axis {axis} out of bounds for shape {x_shape:?}"
                        ));
                    }
                    adjusted as usize
                } else {
                    *axis as usize
                };
                if ax >= x_shape.len() {
                    return Err(format!(
                        "node '{name}': slice axis {axis} out of bounds for shape {x_shape:?}"
                    ));
                }
                let outer: usize = x_shape[..ax].iter().product();
                let dim_size = x_shape[ax];
                let inner: usize = if ax + 1 < x_shape.len() {
                    x_shape[ax + 1..].iter().product()
                } else {
                    1
                };
                let s = if *start < 0 {
                    (dim_size as i64 + *start).max(0) as usize
                } else {
                    (*start as usize).min(dim_size)
                };
                let e = if *end < 0 {
                    (dim_size as i64 + *end).max(0) as usize
                } else {
                    (*end as usize).min(dim_size)
                };
                let slice_len = e.saturating_sub(s);
                let mut out = Vec::with_capacity(outer * slice_len * inner);
                for o in 0..outer.max(1) {
                    let row_base = o * dim_size * inner;
                    let start_idx = row_base + s * inner;
                    let end_idx = row_base + e * inner;
                    if end_idx <= x.len() {
                        out.extend_from_slice(&x[start_idx..end_idx]);
                    }
                }
                out
            }
            Op::Scatter { axis } => {
                let mut base = operand(&bound, &slots, ins, name, 0)?.to_vec();
                let indices = operand_u32(&bound, &slots, ins, name, 1)?;
                let updates = operand(&bound, &slots, ins, name, 2)?;
                let base_shape = &graph.value(ins[0]).shape;
                let ax = if *axis < 0 {
                    (base_shape.len() as isize + *axis) as usize
                } else {
                    *axis as usize
                };
                let outer: usize = base_shape[..ax.min(base_shape.len())].iter().product();
                let dim_size = base_shape.get(ax).copied().unwrap_or(base.len());
                let inner: usize = if ax + 1 < base_shape.len() {
                    base_shape[ax + 1..].iter().product()
                } else {
                    1
                };
                for o in 0..outer.max(1) {
                    for (j, &idx) in indices.iter().enumerate() {
                        let i = idx as usize;
                        let up_idx = (o * indices.len() + j) * inner;
                        if i < dim_size && up_idx + inner <= updates.len() {
                            let start = o * dim_size * inner + i * inner;
                            if start + inner <= base.len() {
                                base[start..start + inner]
                                    .copy_from_slice(&updates[up_idx..up_idx + inner]);
                            }
                        }
                    }
                }
                base
            }
            Op::Pad { padding, value } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                pad_cpu(x, &graph.value(ins[0]).shape, padding, *value, name)?
            }
            Op::AvgPool2d { kernel, stride } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let shape = &graph.value(ins[0]).shape;
                // Leading dims are independent planes (batch × channels).
                let (c, h, w) = if shape.len() >= 3 {
                    (
                        shape[..shape.len() - 2].iter().product(),
                        shape[shape.len() - 2],
                        shape[shape.len() - 1],
                    )
                } else {
                    (1, 1, x.len())
                };
                let (kh, kw) = (kernel[0], kernel[1]);
                let (sh, sw) = (stride[0], stride[1]);
                let out_h = if h >= kh { (h - kh) / sh + 1 } else { 0 };
                let out_w = if w >= kw { (w - kw) / sw + 1 } else { 0 };
                let mut out = Vec::with_capacity(c * out_h * out_w);
                for ch in 0..c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut sum = 0.0f32;
                            for r in 0..kh {
                                for col in 0..kw {
                                    let ih = oh * sh + r;
                                    let iw = ow * sw + col;
                                    sum += x[ch * h * w + ih * w + iw];
                                }
                            }
                            out.push(sum / (kh * kw) as f32);
                        }
                    }
                }
                out
            }
            Op::Upsample { scale, mode } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let (inp, outp) = (
                    &graph.value(ins[0]).shape,
                    &graph.value(node.outputs[0]).shape,
                );
                let nsp =
                    upsample_nsp(inp, outp, *scale).map_err(|e| format!("node '{name}': {e}"))?;
                upsample_cpu(x, inp, *scale, *mode, nsp)
            }
            Op::Conv3d {
                stride,
                padding: p,
                dilation,
                groups,
            } => conv_nd(
                name,
                operand(&bound, &slots, ins, name, 0)?,
                operand(&bound, &slots, ins, name, 1)?,
                if ins.len() > 2 {
                    Some(operand(&bound, &slots, ins, name, 2)?)
                } else {
                    None
                },
                (&graph.value(ins[0]).shape, &graph.value(ins[1]).shape, 3),
                (
                    *stride,
                    [p[0], p[1], p[2]],
                    [p[3], p[4], p[5]],
                    *dilation,
                    *groups,
                ),
            )?,
            Op::Conv1d {
                stride,
                padding: p,
                dilation,
                groups,
            } => conv_nd(
                name,
                operand(&bound, &slots, ins, name, 0)?,
                operand(&bound, &slots, ins, name, 1)?,
                if ins.len() > 2 {
                    Some(operand(&bound, &slots, ins, name, 2)?)
                } else {
                    None
                },
                (&graph.value(ins[0]).shape, &graph.value(ins[1]).shape, 1),
                (
                    [1, 1, *stride],
                    [0, 0, p[0]],
                    [0, 0, p[1]],
                    [1, 1, *dilation],
                    *groups,
                ),
            )?,
            Op::LayerNorm { eps } => {
                if ins.len() < 2 {
                    return Err(format!("node '{name}': LayerNorm needs x and gamma"));
                }
                let (x, g) = (
                    operand(&bound, &slots, ins, name, 0)?,
                    operand(&bound, &slots, ins, name, 1)?,
                );
                let zeros;
                let b = if ins.len() > 2 {
                    operand(&bound, &slots, ins, name, 2)?
                } else {
                    zeros = vec![0.0f32; g.len()];
                    &zeros[..]
                };
                let d = g.len();
                if d == 0 || b.len() != d || x.len() % d != 0 {
                    return Err(format!(
                        "node '{name}': x/gamma/beta lengths {}/{}/{} disagree",
                        x.len(),
                        d,
                        b.len()
                    ));
                }
                nn::layer_norm(x, x.len() / d, d, g, b, *eps)
            }
            Op::GroupNorm { .. } | Op::InstanceNorm { .. } => {
                let (x, g, b) = (
                    operand(&bound, &slots, ins, name, 0)?,
                    operand(&bound, &slots, ins, name, 1)?,
                    operand(&bound, &slots, ins, name, 2)?,
                );
                let c = g.len();
                let (groups, eps) = match node.op {
                    Op::GroupNorm { groups, eps } => (groups, eps),
                    Op::InstanceNorm { eps } => (c, eps),
                    _ => unreachable!(),
                };
                if c == 0 || b.len() != c || groups == 0 || c % groups != 0 {
                    return Err(format!(
                        "node '{name}': gamma/beta lengths {c}/{} do not fit {groups} groups",
                        b.len()
                    ));
                }
                let (n, hw) = channel_layout(&graph.value(ins[0]).shape, c)
                    .map_err(|e| format!("node '{name}': {e}"))?;
                if x.len() != n * c * hw {
                    return Err(format!(
                        "node '{name}': x length {} is not {n}*{c}*{hw}",
                        x.len()
                    ));
                }
                let mut out = x.to_vec();
                for sample in out.chunks_mut(c * hw) {
                    nn::group_norm(sample, c, hw, groups, eps, g, b);
                }
                out
            }
            Op::Reduce { kind, axes, .. } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let shape = &graph.value(ins[0]).shape;
                let rank = shape.len();
                let mut red = vec![false; rank];
                if axes.is_empty() {
                    return Err(format!("node '{name}': Reduce with no axes is ambiguous"));
                }
                for &a in axes {
                    let ax = if a < 0 { a + rank as isize } else { a };
                    if ax < 0 || ax >= rank as isize || red[ax as usize] {
                        return Err(format!(
                            "node '{name}': bad Reduce axis {a} for rank {rank}"
                        ));
                    }
                    red[ax as usize] = true;
                }
                // Row-major strides over the kept dims only.
                let mut ostride = vec![0usize; rank];
                let mut n_out = 1usize;
                for i in (0..rank).rev() {
                    if !red[i] {
                        ostride[i] = n_out;
                        n_out *= shape[i];
                    }
                }
                let rcount = (x.len() / n_out.max(1)) as f64;
                let init = match kind {
                    super::ReduceKind::Max => f64::NEG_INFINITY,
                    super::ReduceKind::Min => f64::INFINITY,
                    _ => 0.0,
                };
                let mut acc = vec![init; n_out];
                let mut coord = vec![0usize; rank];
                for &v in x {
                    let o: usize = coord.iter().zip(&ostride).map(|(c, s)| c * s).sum();
                    let v = v as f64;
                    acc[o] = match kind {
                        super::ReduceKind::Max => acc[o].max(v),
                        super::ReduceKind::Min => acc[o].min(v),
                        _ => acc[o] + v,
                    };
                    for i in (0..rank).rev() {
                        coord[i] += 1;
                        if coord[i] < shape[i] {
                            break;
                        }
                        coord[i] = 0;
                    }
                }
                let mean = matches!(kind, super::ReduceKind::Mean);
                acc.into_iter()
                    .map(|a| (if mean { a / rcount } else { a }) as f32)
                    .collect()
            }
            Op::Attention {
                heads,
                kv_heads,
                causal,
                window,
                soft_cap,
                scale,
            } => {
                let (q, k, v) = (
                    operand(&bound, &slots, ins, name, 0)?,
                    operand(&bound, &slots, ins, name, 1)?,
                    operand(&bound, &slots, ins, name, 2)?,
                );
                let bias = if ins.len() > 3 {
                    Some(operand(&bound, &slots, ins, name, 3)?)
                } else {
                    None
                };
                let hd = last_dim(0);
                if *heads == 0
                    || *kv_heads == 0
                    || heads % kv_heads != 0
                    || hd == 0
                    || hd % heads != 0
                    || q.len() % hd != 0
                    || k.is_empty()
                    || k.len() % (kv_heads * (hd / heads)) != 0
                    || k.len() != v.len()
                {
                    return Err(format!(
                        "node '{name}': q/k/v shapes do not fit {heads} heads over {kv_heads} kv heads"
                    ));
                }
                let d = hd / heads;
                let (nq, nk) = (q.len() / hd, k.len() / (kv_heads * d));
                if let Some(b) = bias {
                    if ![nk, nq * nk, heads * nq * nk].contains(&b.len()) {
                        return Err(format!(
                            "node '{name}': bias of {} elements is not [{heads}, {nq}, {nk}], [{nq}, {nk}] or [{nk}]",
                            b.len()
                        ));
                    }
                }
                let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
                if kv_heads == heads
                    && window.is_none()
                    && soft_cap.is_none()
                    && bias.is_none()
                    && scale == 1.0 / (d as f32).sqrt()
                {
                    nn::attention(q, k, v, nq, nk, *heads, d, *causal)
                } else {
                    attention_general(
                        q, k, v, bias, nq, nk, *heads, *kv_heads, d, *causal, *window, *soft_cap,
                        scale,
                    )
                }
            }
            Op::MeanPool => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                let mask = operand_u32(&bound, &slots, ins, name, 1)?;
                let d = last_dim(0);
                if d == 0 || x.len() != mask.len() * d {
                    return Err(format!(
                        "node '{name}': x of {} elements does not match mask of {}",
                        x.len(),
                        mask.len()
                    ));
                }
                nn::mean_pool(x, mask, d)
            }
            Op::L2Normalize => {
                let mut x = take0(&bound, &mut slots, ins, name, &last_use, i)?;
                nn::l2_normalize(&mut x);
                x
            }
            Op::Conv2d {
                stride,
                padding: p,
                dilation,
                groups,
            } => conv_nd(
                name,
                operand(&bound, &slots, ins, name, 0)?,
                operand(&bound, &slots, ins, name, 1)?,
                if ins.len() > 2 {
                    Some(operand(&bound, &slots, ins, name, 2)?)
                } else {
                    None
                },
                (&graph.value(ins[0]).shape, &graph.value(ins[1]).shape, 2),
                (
                    [1, stride[0], stride[1]],
                    [0, p[0], p[1]],
                    [0, p[2], p[3]],
                    [1, dilation[0], dilation[1]],
                    *groups,
                ),
            )?,
            Op::Reshape { .. } | Op::Squeeze { .. } | Op::Unsqueeze { .. } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                if x.len() != graph.value(node.outputs[0]).shape.iter().product::<usize>() {
                    return Err(format!(
                        "node '{name}': {} changes the element count",
                        node.op.name()
                    ));
                }
                x.to_vec()
            }
            Op::Transpose { perm } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                transpose_cpu(x, &graph.value(ins[0]).shape, perm, name)?
            }
            Op::Tile { reps } => {
                let x = operand(&bound, &slots, ins, name, 0)?;
                tile_cpu(x, &graph.value(ins[0]).shape, reps)
            }
            Op::Concat { axis } => {
                let mut parts = Vec::with_capacity(ins.len());
                for k in 0..ins.len() {
                    parts.push((
                        operand(&bound, &slots, ins, name, k)?,
                        graph.value(ins[k]).shape.as_slice(),
                    ));
                }
                concat_cpu(&parts, *axis, name)?
            }
        };

        for v in ins {
            if last_use[v.0 as usize] == i {
                if let Some(Tensor::F32(recycled)) = slots[v.0 as usize].take() {
                    CPU_ARENA.with(|a| {
                        let mut pool = a.borrow_mut();
                        if pool.len() < 64 && recycled.capacity() >= 64 {
                            pool.push(recycled);
                        }
                    });
                }
            }
        }
        slots[node.outputs[0].0 as usize] = Some(match out_u32 {
            Some(idx) => Tensor::U32(idx),
            None => Tensor::F32(out),
        });
    }

    graph
        .outputs
        .iter()
        .map(|v| {
            let id = v.0 as usize;
            bound[id]
                .cloned()
                .or_else(|| slots[id].clone())
                .ok_or_else(|| format!("output '{}' was never computed", graph.value(*v).name))
        })
        .collect()
}

/// `op(a)·op(b)` over the last two dims; leading batch dims broadcast numpy-style (right-aligned).
fn matmul_ref(
    graph: &Graph,
    ins: &[ValueId],
    a: &[f32],
    b: &[f32],
    ta: bool,
    tb: bool,
    name: &str,
) -> Result<Vec<f32>, String> {
    let (sa, sb) = (&graph.value(ins[0]).shape, &graph.value(ins[1]).shape);
    if sa.len() < 2
        || sb.len() < 2
        || a.len() != sa.iter().product::<usize>()
        || b.len() != sb.iter().product::<usize>()
    {
        return Err(format!(
            "node '{name}': MatMul operands {sa:?} x {sb:?} must be rank >= 2 and match their data"
        ));
    }
    let (ar, ac, br, bc) = (
        sa[sa.len() - 2],
        sa[sa.len() - 1],
        sb[sb.len() - 2],
        sb[sb.len() - 1],
    );
    let (m, k) = if ta { (ac, ar) } else { (ar, ac) };
    let (k2, n) = if tb { (bc, br) } else { (br, bc) };
    if k != k2 {
        return Err(format!(
            "node '{name}': MatMul inner dims {k} != {k2} ({sa:?} x {sb:?})"
        ));
    }
    let nd = (sa.len() - 2).max(sb.len() - 2);
    let pad = |d: &[usize]| [vec![1; nd - d.len()], d.to_vec()].concat();
    let (pa, pb) = (pad(&sa[..sa.len() - 2]), pad(&sb[..sb.len() - 2]));
    let mut ob = Vec::with_capacity(nd);
    for i in 0..nd {
        match (pa[i], pb[i]) {
            (x, y) if x == y || y == 1 => ob.push(x),
            (1, y) => ob.push(y),
            _ => {
                return Err(format!(
                    "node '{name}': MatMul batch dims {sa:?} and {sb:?} do not broadcast"
                ))
            }
        }
    }
    // Element offset of batch index `flat` in an operand with padded batch dims `p` and matrix size `sz`.
    let off = |p: &[usize], mut flat: usize, sz: usize| {
        let (mut o, mut stride) = (0, sz);
        for i in (0..nd).rev() {
            let idx = flat % ob[i];
            flat /= ob[i];
            if p[i] != 1 {
                o += idx * stride;
            }
            stride *= p[i];
        }
        o
    };
    let mut out = vec![0.0f32; ob.iter().product::<usize>() * m * n];
    for bi in 0..out.len() / (m * n).max(1) {
        let (oa, obb) = (off(&pa, bi, ar * ac), off(&pb, bi, br * bc));
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for kk in 0..k {
                    let av = if ta {
                        a[oa + kk * m + i]
                    } else {
                        a[oa + i * k + kk]
                    };
                    let bv = if tb {
                        b[obb + j * k + kk]
                    } else {
                        b[obb + kk * n + j]
                    };
                    acc += av as f64 * bv as f64;
                }
                out[bi * m * n + i * n + j] = acc as f32;
            }
        }
    }
    Ok(out)
}

fn slot<'a>(
    bound: &[Option<&'a Tensor>],
    slots: &'a [Option<Tensor>],
    ins: &[ValueId],
    name: &str,
    k: usize,
) -> Result<&'a Tensor, String> {
    let id = ins
        .get(k)
        .ok_or_else(|| format!("node '{name}' is missing operand {k}"))?
        .0 as usize;
    bound[id]
        .or(slots[id].as_ref())
        .ok_or_else(|| format!("node '{name}': operand {k} has no value"))
}

/// `(batch, hw)` of a channel-first tensor with `c` channels. A single axis of size `c` (0 or 1)
/// is the channel axis; when both are `c`, rank 2 is `[C, HW]` (axis 0) and rank >= 3 is
/// `[N, C, ...]` (axis 1). `batch` is the product of the axes before the channel axis.
/// Shared with the Metal lowering.
// ponytail: an unbatched `[C, C, w]` tensor reads as batched `[N=C, C, w]`; add a batch attribute if that matters.
pub(crate) fn channel_layout(shape: &[usize], c: usize) -> Result<(usize, usize), String> {
    let at0 = shape.first() == Some(&c);
    let at1 = shape.get(1) == Some(&c);
    let ax = match (at0, at1) {
        (true, true) if shape.len() >= 3 => 1,
        (true, _) => 0,
        (false, true) => 1,
        _ => {
            return Err(format!(
                "shape {shape:?} has no channel axis of size {c} at axis 0 or 1"
            ))
        }
    };
    Ok((
        shape[..ax].iter().product(),
        shape[ax + 1..].iter().product(),
    ))
}

fn operand<'a>(
    bound: &[Option<&'a Tensor>],
    slots: &'a [Option<Tensor>],
    ins: &[ValueId],
    name: &str,
    k: usize,
) -> Result<&'a [f32], String> {
    match slot(bound, slots, ins, name, k)? {
        Tensor::F32(v) => Ok(v),
        _ => Err(format!("node '{name}': operand {k} must be f32")),
    }
}

/// Direct reference convolution. `x [.., c_in, spatial..]`, `w [c_out, c_in/groups, k..]` with
/// `nd` spatial dims (leading dims of `x` are batch); `geom` = (stride, pad begin, pad end,
/// dilation, groups) right-aligned into 3 spatial dims (unused dims neutral: 1/0/0/1).
#[allow(clippy::type_complexity)]
fn conv_nd(
    name: &str,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    (xs, ws, nd): (&[usize], &[usize], usize),
    (stride, pb, pe, dil, groups): ([usize; 3], [usize; 3], [usize; 3], [usize; 3], usize),
) -> Result<Vec<f32>, String> {
    if xs.len() < nd + 1 || ws.len() != nd + 2 {
        return Err(format!(
            "node '{name}': conv shapes {xs:?} / {ws:?} do not fit {nd}-d conv"
        ));
    }
    let (mut inp, mut k) = ([1usize; 3], [1usize; 3]);
    inp[3 - nd..].copy_from_slice(&xs[xs.len() - nd..]);
    k[3 - nd..].copy_from_slice(&ws[2..]);
    let c_in = xs[xs.len() - nd - 1];
    let batch: usize = xs[..xs.len() - nd - 1].iter().product();
    let (c_out, cpg) = (ws[0], ws[1]);
    if groups == 0
        || c_in != cpg * groups
        || c_out % groups != 0
        || stride.contains(&0)
        || dil.contains(&0)
    {
        return Err(format!(
            "node '{name}': conv c_in {c_in}, weight {ws:?}, groups {groups} are inconsistent"
        ));
    }
    if x.len() != batch * c_in * inp.iter().product::<usize>()
        || w.len() != ws.iter().product::<usize>()
    {
        return Err(format!(
            "node '{name}': conv operand lengths do not match shapes"
        ));
    }
    if bias.is_some_and(|b| b.len() != c_out) {
        return Err(format!("node '{name}': conv bias is not c_out"));
    }
    let mut out_d = [1usize; 3];
    for a in 0..3 {
        let span = dil[a] * (k[a] - 1) + 1;
        if inp[a] + pb[a] + pe[a] < span {
            return Err(format!(
                "node '{name}': conv kernel larger than padded input"
            ));
        }
        out_d[a] = (inp[a] + pb[a] + pe[a] - span) / stride[a] + 1;
    }
    let mut out = Vec::with_capacity(batch * c_out * out_d.iter().product::<usize>());
    let ocg = c_out / groups;
    for b in 0..batch {
        for oc in 0..c_out {
            let g = oc / ocg;
            for od in 0..out_d[0] {
                for oh in 0..out_d[1] {
                    for ow in 0..out_d[2] {
                        let mut sum = bias.map_or(0.0, |bb| bb[oc]);
                        for ic in 0..cpg {
                            let xc = (b * c_in + g * cpg + ic) * inp.iter().product::<usize>();
                            let wc = (oc * cpg + ic) * k.iter().product::<usize>();
                            for kd in 0..k[0] {
                                let id = (od * stride[0] + kd * dil[0]) as isize - pb[0] as isize;
                                if id < 0 || id >= inp[0] as isize {
                                    continue;
                                }
                                for kh in 0..k[1] {
                                    let ih =
                                        (oh * stride[1] + kh * dil[1]) as isize - pb[1] as isize;
                                    if ih < 0 || ih >= inp[1] as isize {
                                        continue;
                                    }
                                    for kw in 0..k[2] {
                                        let iw = (ow * stride[2] + kw * dil[2]) as isize
                                            - pb[2] as isize;
                                        if iw < 0 || iw >= inp[2] as isize {
                                            continue;
                                        }
                                        let xi = xc
                                            + (id as usize * inp[1] + ih as usize) * inp[2]
                                            + iw as usize;
                                        sum += x[xi] * w[wc + (kd * k[1] + kh) * k[2] + kw];
                                    }
                                }
                            }
                        }
                        out.push(sum);
                    }
                }
            }
        }
    }
    Ok(out)
}

fn operand_u32<'a>(
    bound: &[Option<&'a Tensor>],
    slots: &'a [Option<Tensor>],
    ins: &[ValueId],
    name: &str,
    k: usize,
) -> Result<&'a [u32], String> {
    match slot(bound, slots, ins, name, k)? {
        Tensor::U32(v) => Ok(v),
        _ => Err(format!("node '{name}': operand {k} must be u32")),
    }
}

// ---- data-movement references (Transpose, Concat, Tile, Pad, Upsample) ----

fn dense_strides(shape: &[usize]) -> Vec<usize> {
    let mut st = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        st[d] = st[d + 1] * shape[d + 1];
    }
    st
}

fn norm_axis(axis: isize, rank: usize, name: &str) -> Result<usize, String> {
    let a = if axis < 0 { axis + rank as isize } else { axis };
    if a < 0 || a as usize >= rank {
        return Err(format!(
            "node '{name}': axis {axis} out of range for rank {rank}"
        ));
    }
    Ok(a as usize)
}

/// Visits every multi-index of `dims` in row-major order.
fn for_each_index(dims: &[usize], mut f: impl FnMut(&[usize])) {
    if dims.iter().any(|&d| d == 0) {
        return;
    }
    let mut idx = vec![0usize; dims.len()];
    loop {
        f(&idx);
        let mut d = dims.len();
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            idx[d] += 1;
            if idx[d] < dims[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

fn transpose_cpu(
    x: &[f32],
    shape: &[usize],
    perm: &[usize],
    name: &str,
) -> Result<Vec<f32>, String> {
    let r = shape.len();
    let mut seen = vec![false; r];
    for &p in perm {
        if p >= r || seen[p] {
            return Err(format!(
                "node '{name}': {perm:?} is not a permutation of rank {r}"
            ));
        }
        seen[p] = true;
    }
    if perm.len() != r {
        return Err(format!(
            "node '{name}': {perm:?} is not a permutation of rank {r}"
        ));
    }
    let in_st = dense_strides(shape);
    let odims: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let mut out = Vec::with_capacity(x.len());
    for_each_index(&odims, |idx| {
        out.push(
            x[idx
                .iter()
                .zip(perm)
                .map(|(&i, &p)| i * in_st[p])
                .sum::<usize>()],
        );
    });
    Ok(out)
}

fn tile_cpu(x: &[f32], shape: &[usize], reps: &[usize]) -> Vec<f32> {
    let r = shape.len().max(reps.len());
    let pad = |v: &[usize]| -> Vec<usize> {
        std::iter::repeat_n(1, r - v.len())
            .chain(v.iter().copied())
            .collect()
    };
    let (ish, rp) = (pad(shape), pad(reps));
    let odims: Vec<usize> = ish.iter().zip(&rp).map(|(a, b)| a * b).collect();
    let in_st = dense_strides(&ish);
    let mut out = Vec::with_capacity(odims.iter().product());
    for_each_index(&odims, |idx| {
        out.push(x[(0..r).map(|d| (idx[d] % ish[d]) * in_st[d]).sum::<usize>()]);
    });
    out
}

fn concat_cpu(parts: &[(&[f32], &[usize])], axis: isize, name: &str) -> Result<Vec<f32>, String> {
    let rank = parts.first().map_or(0, |p| p.1.len());
    let ax = norm_axis(axis, rank, name)?;
    for (_, sh) in parts {
        if sh.len() != rank || (0..rank).any(|d| d != ax && sh[d] != parts[0].1[d]) {
            return Err(format!(
                "node '{name}': concat operands disagree off axis {ax}"
            ));
        }
    }
    let outer: usize = parts[0].1[..ax].iter().product();
    let inner: usize = parts[0].1[ax + 1..].iter().product();
    let mut out = Vec::new();
    for o in 0..outer {
        for (x, sh) in parts {
            let chunk = sh[ax] * inner;
            out.extend_from_slice(&x[o * chunk..(o + 1) * chunk]);
        }
    }
    Ok(out)
}

/// `padding` holds `(before, after)` pairs for the last `padding.len() / 2` dims, outermost first.
fn pad_cpu(
    x: &[f32],
    shape: &[usize],
    padding: &[usize],
    value: f32,
    name: &str,
) -> Result<Vec<f32>, String> {
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
    let out_st = dense_strides(&odims);
    let mut out = vec![value; odims.iter().product()];
    let mut i = 0;
    for_each_index(shape, |idx| {
        out[(0..r)
            .map(|d| (idx[d] + before(d)) * out_st[d])
            .sum::<usize>()] = x[i];
        i += 1;
    });
    Ok(out)
}

/// Number of trailing spatial dims (2 or 3) an `Upsample` scales, from the declared shapes:
/// 3 only when the third-from-last dim also grows by `scale`.
pub(crate) fn upsample_nsp(inp: &[usize], out: &[usize], scale: usize) -> Result<usize, String> {
    let r = inp.len();
    if r < 2 || out.len() != r || scale == 0 {
        return Err(format!(
            "Upsample {inp:?} -> {out:?} by {scale} is not a spatial resize"
        ));
    }
    let nsp = if r >= 3 && scale > 1 && out[r - 3] == inp[r - 3] * scale {
        3
    } else {
        2
    };
    for d in 0..r {
        let want = if d + nsp >= r { inp[d] * scale } else { inp[d] };
        if out[d] != want {
            return Err(format!(
                "Upsample {inp:?} -> {out:?} by {scale}: dim {d} should be {want}"
            ));
        }
    }
    Ok(nsp)
}

/// Source taps `(index, weight)` of output coordinate `o` along one axis of size `n`
/// (half-pixel centres, `align_corners = false`, border clamp).
fn upsample_taps(o: usize, n: usize, scale: usize, mode: super::UpsampleMode) -> Vec<(usize, f32)> {
    use super::UpsampleMode::*;
    let src = (o as f32 + 0.5) / scale as f32 - 0.5;
    match mode {
        Nearest => vec![((o / scale).min(n - 1), 1.0)],
        Bilinear => {
            let src = src.max(0.0);
            let i0 = src.floor();
            let l = src - i0;
            let i0 = (i0 as usize).min(n - 1);
            vec![(i0, 1.0 - l), ((i0 + 1).min(n - 1), l)]
        }
        Bicubic => {
            const A: f32 = -0.75;
            let near = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
            let far = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
            let i0 = src.floor();
            let t = src - i0;
            let w = [far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)];
            (0..4)
                .map(|k| {
                    (
                        (i0 as i64 + k as i64 - 1).clamp(0, n as i64 - 1) as usize,
                        w[k],
                    )
                })
                .collect()
        }
    }
}

fn upsample_cpu(
    x: &[f32],
    shape: &[usize],
    scale: usize,
    mode: super::UpsampleMode,
    nsp: usize,
) -> Vec<f32> {
    let r = shape.len();
    let lead: usize = shape[..r - nsp].iter().product();
    let (d, h, w) = if nsp == 3 {
        (shape[r - 3], shape[r - 2], shape[r - 1])
    } else {
        (1, shape[r - 2], shape[r - 1])
    };
    let sd = if nsp == 3 { scale } else { 1 };
    let (od, oh, ow) = (d * sd, h * scale, w * scale);
    let mut out = Vec::with_capacity(lead * od * oh * ow);
    for l in 0..lead {
        let plane = &x[l * d * h * w..];
        for z in 0..od {
            let tz = upsample_taps(z, d, sd, mode);
            for y in 0..oh {
                let ty = upsample_taps(y, h, scale, mode);
                for xx in 0..ow {
                    let tx = upsample_taps(xx, w, scale, mode);
                    let mut acc = 0.0f32;
                    for &(iz, wz) in &tz {
                        for &(iy, wy) in &ty {
                            for &(ix, wx) in &tx {
                                acc += wz * wy * wx * plane[(iz * h + iy) * w + ix];
                            }
                        }
                    }
                    out.push(acc);
                }
            }
        }
    }
    out
}

/// Element count of a node's declared output; 0 when it has no static shape.
fn declared_len(graph: &Graph, node: &super::Node) -> usize {
    node.outputs
        .first()
        .map_or(0, |&o| match graph.value(o).shape.as_slice() {
            [] => 0,
            s => s.iter().product(),
        })
}

/// numpy-style broadcast of two f32 operands over their declared shapes. `None` when a shape is
/// unknown or disagrees with its data, the shapes are identical (callers zip flat), are not
/// right-aligned compatible, or broadcast to other than `out_len` elements (callers tile by length).
fn broadcast_np(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    out_len: usize,
    f: impl Fn(f32, f32) -> f32,
) -> Option<Vec<f32>> {
    let numel = |s: &[usize]| s.iter().product::<usize>();
    if a_shape == b_shape
        || a_shape.is_empty()
        || b_shape.is_empty()
        || numel(a_shape) != a.len()
        || numel(b_shape) != b.len()
    {
        return None;
    }
    let r = a_shape.len().max(b_shape.len());
    let dim = |s: &[usize], i: usize| {
        if i + s.len() >= r {
            s[i + s.len() - r]
        } else {
            1
        }
    };
    let mut shape = Vec::with_capacity(r);
    for i in 0..r {
        let (x, y) = (dim(a_shape, i), dim(b_shape, i));
        if x != y && x != 1 && y != 1 {
            return None;
        }
        shape.push(if x == 1 { y } else { x });
    }
    if numel(&shape) != out_len {
        return None;
    }
    // Element stride per output dim, 0 where the operand is broadcast.
    let strides = |s: &[usize]| {
        let mut st = vec![0usize; r];
        let mut acc = 1;
        for i in (0..r).rev() {
            let d = dim(s, i);
            st[i] = if d == 1 { 0 } else { acc };
            acc *= d;
        }
        st
    };
    let (sa, sb) = (strides(a_shape), strides(b_shape));
    let (inner, la, lb) = (shape[r - 1], sa[r - 1], sb[r - 1]);
    let mut out = Vec::with_capacity(out_len);
    let mut idx = vec![0usize; r - 1];
    let (mut ia, mut ib) = (0usize, 0usize);
    for _ in 0..out_len / inner.max(1) {
        out.extend((0..inner).map(|j| f(a[ia + j * la], b[ib + j * lb])));
        for i in (0..r - 1).rev() {
            idx[i] += 1;
            ia += sa[i];
            ib += sb[i];
            if idx[i] < shape[i] {
                break;
            }
            ia -= sa[i] * shape[i];
            ib -= sb[i] * shape[i];
            idx[i] = 0;
        }
    }
    Some(out)
}

/// numpy-incompatible `[m, ..] * [m, 1]` (either order): row `r` scaled by the `[m, 1]` operand's `r`.
fn row_scale(a: &[f32], sa: &[usize], b: &[f32], sb: &[usize]) -> Option<Vec<f32>> {
    let rows = |x: &[f32], sx: &[usize], s: &[f32], ss: &[usize]| {
        (sx.len() >= 2 && ss.len() >= 2 && ss.last() == Some(&1) && sx[0] == ss[0] && !s.is_empty())
            .then(|| {
                let d = (x.len() / s.len()).max(1);
                x.chunks(d)
                    .zip(s)
                    .flat_map(|(row, &k)| row.iter().map(move |&v| v * k))
                    .collect()
            })
    };
    rows(a, sa, b, sb).or_else(|| rows(b, sb, a, sa))
}

/// `f` over equal-length operands, or with the shorter operand tiled over the longer one.
fn tile_binary(a: &[f32], b: &[f32], f: impl Fn(f32, f32) -> f32) -> Option<Vec<f32>> {
    let mut out = Vec::with_capacity(a.len().max(b.len()));
    if a.len() == b.len() {
        out.extend(a.iter().zip(b).map(|(&x, &y)| f(x, y)));
    } else if !b.is_empty() && a.len() % b.len() == 0 {
        for chunk in a.chunks(b.len()) {
            out.extend(chunk.iter().zip(b).map(|(&x, &y)| f(x, y)));
        }
    } else if !a.is_empty() && b.len() % a.len() == 0 {
        for chunk in b.chunks(a.len()) {
            out.extend(a.iter().zip(chunk).map(|(&x, &y)| f(x, y)));
        }
    } else {
        return None;
    }
    Some(out)
}

/// Indices of the `k` largest values of `row`: value descending, equal values lower index first, NaN
/// ranking below every number (the Metal kernel orders the same way).
pub(crate) fn topk_row(row: &[f32], k: usize) -> Vec<u32> {
    let key = |i: u32| match row[i as usize] {
        v if v.is_nan() => f32::NEG_INFINITY,
        v => v,
    };
    let before = |a: &u32, b: &u32| {
        key(*b)
            .partial_cmp(&key(*a))
            .expect("NaN mapped away")
            .then(a.cmp(b))
    };
    let mut order: Vec<u32> = (0..row.len() as u32).collect();
    order.select_nth_unstable_by(k - 1, before);
    order.truncate(k);
    order.sort_unstable_by(before);
    order
}

/// `Op::QsaSelect` ids and `Op::QsaBias` bias (`[n, K]` each, `K = top_blocks * ratio + ratio - 1`)
/// of `score` (`[n, b]`) at `positions` (`[n]`); semantics on [`Op::QsaSelect`].
pub(crate) fn qsa_select(
    score: &[f32],
    b: usize,
    positions: &[u32],
    ratio: usize,
    top_blocks: usize,
) -> Result<(Vec<u32>, Vec<f32>), String> {
    let n = positions.len();
    if ratio == 0 || score.len() != n * b {
        return Err(format!(
            "QSA needs ratio >= 1 and score [{n}, B], got ratio {ratio} over {} scores",
            score.len()
        ));
    }
    let k = top_blocks * ratio + ratio - 1;
    let (mut ids, mut bias) = (vec![0u32; n * k], vec![f32::NEG_INFINITY; n * k]);
    for (t, &p) in positions.iter().enumerate() {
        let visible = p as usize + 1;
        let blocks = visible / ratio;
        if blocks > b {
            return Err(format!(
                "row {t} at position {p} sees {blocks} complete blocks, score has {b}"
            ));
        }
        let keys: Vec<u32> = if blocks <= top_blocks {
            (0..visible as u32).collect()
        } else {
            let row = &score[t * b..][..blocks];
            let mut best: Vec<(f32, u32)> =
                (0..blocks as u32).map(|j| (row[j as usize], j)).collect();
            best.select_nth_unstable_by(top_blocks.max(1) - 1, |x, y| {
                y.0.total_cmp(&x.0).then(x.1.cmp(&y.1))
            });
            best.truncate(top_blocks);
            best.sort_unstable_by_key(|s| s.1);
            best.iter()
                .flat_map(|&(_, j)| j * ratio as u32..(j + 1) * ratio as u32)
                .chain((blocks * ratio) as u32..visible as u32)
                .collect()
        };
        ids[t * k..][..keys.len()].copy_from_slice(&keys);
        bias[t * k..][..keys.len()].fill(0.0);
    }
    Ok((ids, bias))
}

/// Row `row` (of `out.len()` weights) of the expert bank `bank` as f32.
fn bank_row(bank: &Tensor, row: usize, out: &mut [f32]) -> Result<(), String> {
    let cols = out.len();
    let short = || format!("expert bank has no row {row}");
    match bank {
        Tensor::F32(v) => {
            out.copy_from_slice(v.get(row * cols..(row + 1) * cols).ok_or_else(short)?)
        }
        Tensor::BF16(v) => {
            let r = v.get(row * cols..(row + 1) * cols).ok_or_else(short)?;
            out.iter_mut()
                .zip(r)
                .for_each(|(o, &b)| *o = crate::model::bf16_to_f32(b));
        }
        Tensor::F16(v) => {
            let r = v.get(row * cols..(row + 1) * cols).ok_or_else(short)?;
            out.iter_mut()
                .zip(r)
                .for_each(|(o, &b)| *o = crate::model::f16_to_f32(b));
        }
        Tensor::Mat(m) if row < m.rows && m.cols == cols => m.decode_row_f32(row, out),
        _ => return Err("expert banks must be f32/f16/bf16 or GGUF blocks".into()),
    }
    Ok(())
}

/// `Op::MoeLinear` reference: slot `s` of `ids` multiplies its x row by expert `ids[s]`'s weight.
fn moe_linear_cpu(
    x: &[f32],
    bank: &Tensor,
    ids: &[u32],
    [experts, top_k, in_f, out_f]: [usize; 4],
    name: &str,
) -> Result<Vec<f32>, String> {
    if let Some((s, e)) = ids
        .iter()
        .enumerate()
        .find(|&(_, &e)| e as usize >= experts)
    {
        return Err(format!(
            "node '{name}': slot {s} routes to expert {e} of {experts}"
        ));
    }
    let per_slot = x.len() / in_f == ids.len();
    let x_row = |s: usize| if per_slot { s } else { s / top_k };
    let mut out = vec![0.0f32; ids.len() * out_f];
    if let Tensor::Mat(m) = bank {
        if m.rows == experts * out_f && m.cols == in_f {
            let slots: Vec<(usize, usize)> = (0..ids.len())
                .map(|s| (x_row(s), ids[s] as usize * out_f))
                .collect();
            crate::model::qdot::gather_linear(m, x, &slots, out_f, &mut out);
            return Ok(out);
        }
    }
    thread_local! {
        static BANK_ROW: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    const ROW_BLOCK: usize = 16;
    let blocks = out_f.div_ceil(ROW_BLOCK);
    let dst = crate::model::par::OutPtr::new(&mut out);
    let failure = parking_lot::Mutex::new(None::<String>);
    crate::model::par::par_for(ids.len() * blocks, &|t| {
        let (s, b) = (t / blocks, t % blocks);
        let xr = &x[x_row(s) * in_f..][..in_f];
        BANK_ROW.with_borrow_mut(|w| {
            w.resize(in_f, 0.0);
            for r in b * ROW_BLOCK..((b + 1) * ROW_BLOCK).min(out_f) {
                if let Err(e) = bank_row(bank, ids[s] as usize * out_f + r, w) {
                    *failure.lock() = Some(e);
                    return;
                }
                let v = crate::model::simd::vector_dot_product(w, xr);
                // SAFETY: cell (s, r) belongs to this task alone.
                unsafe { dst.slice(s * out_f + r, 1)[0] = v };
            }
        });
    });
    match failure.into_inner() {
        Some(e) => Err(format!("node '{name}': {e}")),
        None => Ok(out),
    }
}

fn alloc_cpu_f32(len: usize) -> Vec<f32> {
    let mut v = CPU_ARENA
        .with(|a| {
            let mut pool = a.borrow_mut();
            pool.iter()
                .position(|b| b.capacity() >= len && b.capacity() <= len.saturating_mul(4).max(256))
                .map(|idx| pool.swap_remove(idx))
        })
        .unwrap_or_default();
    v.resize(len, 0.0);
    v
}

fn dispatch_linear(
    x: &[f32],
    w_slot: &Tensor,
    bias: Option<&[f32]>,
    in_f: usize,
    out_f: usize,
    rows: usize,
    name: &str,
) -> Result<Vec<f32>, String> {
    match w_slot {
        Tensor::F32(w) => {
            if w.len() != out_f * in_f {
                return Err(format!(
                    "node '{name}': weight elements {} != {out_f} * {in_f}",
                    w.len()
                ));
            }
            Ok(nn::linear(x, rows, w, bias, in_f, out_f))
        }
        Tensor::BF16(w) => {
            if w.len() != out_f * in_f {
                return Err(format!(
                    "node '{name}': weight elements {} != {out_f} * {in_f}",
                    w.len()
                ));
            }
            let mut out = alloc_cpu_f32(rows * out_f);
            nn::matmul_bf16(x, w, bias, rows, in_f, out_f, &mut out);
            Ok(out)
        }
        Tensor::F16(w) => {
            if w.len() != out_f * in_f {
                return Err(format!(
                    "node '{name}': weight elements {} != {out_f} * {in_f}",
                    w.len()
                ));
            }
            let mut out = alloc_cpu_f32(rows * out_f);
            nn::matmul_rows(x, rows, in_f, out_f, bias, &mut out, &|j, p0, d| {
                for (dst, &s) in d.iter_mut().zip(&w[j * in_f + p0..]) {
                    *dst = crate::model::f16_to_f32(s);
                }
            });
            Ok(out)
        }
        Tensor::Mat(m) => {
            if m.rows != out_f || m.cols != in_f {
                return Err(format!(
                    "node '{name}': weight elements {} != {out_f} * {in_f}",
                    m.rows * m.cols
                ));
            }
            let mut out = alloc_cpu_f32(rows * out_f);
            match m.packed() {
                None => nn::matmul_bf16(x, m.data(), bias, rows, in_f, out_f, &mut out),
                // Decode and small batches dot the stored blocks directly; larger batches keep the
                // f32 GEMM unless an `smmla` kernel covers the format.
                Some(p) if rows <= 8 || crate::model::qdot::i8mm_format(p.format) => {
                    crate::model::qdot::linear_q(m, x, rows, bias, &mut out)
                }
                Some(_) => nn::matmul_rows(x, rows, in_f, out_f, bias, &mut out, &|j, p0, d| {
                    m.decode_slice_f32(j, p0, d);
                }),
            }
            Ok(out)
        }
        other => {
            let w_f32 = other.to_f32();
            if w_f32.len() != out_f * in_f {
                return Err(format!(
                    "node '{name}': weight elements {} != {out_f} * {in_f}",
                    w_f32.len()
                ));
            }
            Ok(nn::linear(x, rows, &w_f32, bias, in_f, out_f))
        }
    }
}
/// top-left aligned like `nn::attention`; windows follow sprint-2 S2.2: causal sees `w` keys
/// `pos - w < k <= pos`, non-causal sees `|pos - k| <= w / 2`.
pub(crate) fn attention_key_range(
    pos: usize,
    nk: usize,
    causal: bool,
    window: Option<usize>,
) -> (usize, usize) {
    match (causal, window) {
        (true, None) => (0, nk.min(pos + 1)),
        (true, Some(w)) => ((pos + 1).saturating_sub(w), nk.min(pos + 1)),
        (false, None) => (0, nk),
        (false, Some(w)) => (
            pos.saturating_sub(w / 2),
            nk.min(pos.saturating_add(w / 2 + 1)),
        ),
    }
}

/// `Op::Attention` with grouped kv heads, window, soft cap and/or an additive `bias`
/// (`[heads, nq, nk]`, `[nq, nk]` or `[nk]`, added after scaling and the soft cap; `-inf` masks).
/// A query with no visible key yields zeros. Scalar reference; the plain form keeps the GEMM-based
/// `nn::attention`.
// ponytail: O(nq*nk*d) without blocking; block it if a CPU-pinned model needs windowed attention.
#[allow(clippy::too_many_arguments)]
fn attention_general(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bias: Option<&[f32]>,
    nq: usize,
    nk: usize,
    heads: usize,
    kv_heads: usize,
    d: usize,
    causal: bool,
    window: Option<usize>,
    soft_cap: Option<f32>,
    scale: f32,
) -> Vec<f32> {
    let (qw, kw, rep) = (heads * d, kv_heads * d, heads / kv_heads);
    let mut out = vec![0.0f32; nq * qw];
    let mut s = vec![0.0f32; nk];
    for qi in 0..nq {
        let (lo, hi) = attention_key_range(qi, nk, causal, window);
        for head in 0..heads {
            let kvh = head / rep;
            let qv = &q[qi * qw + head * d..][..d];
            let mut mx = f32::NEG_INFINITY;
            for ki in lo..hi {
                let kv = &k[ki * kw + kvh * d..][..d];
                let mut x = qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * scale;
                if let Some(cap) = soft_cap {
                    x = cap * (x / cap).tanh();
                }
                if let Some(b) = bias {
                    x += match b.len() {
                        l if l == nk => b[ki],
                        l if l == nq * nk => b[qi * nk + ki],
                        _ => b[(head * nq + qi) * nk + ki],
                    };
                }
                s[ki] = x;
                mx = mx.max(x);
            }
            let o = &mut out[qi * qw + head * d..][..d];
            let mut sum = 0.0f32;
            for ki in lo..hi {
                if s[ki] == f32::NEG_INFINITY {
                    continue;
                }
                let e = (s[ki] - mx).exp();
                sum += e;
                for (o, v) in o.iter_mut().zip(&v[ki * kw + kvh * d..][..d]) {
                    *o += e * v;
                }
            }
            if sum > 0.0 {
                o.iter_mut().for_each(|x| *x /= sum);
            }
        }
    }
    out
}

/// HF `_relative_position_bucket` of `rel = key_pos - query_pos` (f32 log with an f64 scale,
/// truncated), as transformers' `_relative_position_bucket`. Needs `buckets / 2 >= 2` per direction and
/// `max_distance > exact` (checked by the caller).
pub(crate) fn rel_pos_bucket(
    rel: i64,
    bidirectional: bool,
    buckets: usize,
    max_distance: usize,
) -> usize {
    let (mut nb, mut ret) = (buckets as i64, 0i64);
    let n = if bidirectional {
        nb /= 2;
        if rel > 0 {
            ret += nb;
        }
        rel.abs()
    } else {
        (-rel).max(0)
    };
    let exact = nb / 2;
    if n < exact {
        return (ret + n) as usize;
    }
    let scale = (max_distance as f64 / exact as f64).ln() as f32;
    let large = exact + ((n as f32 / exact as f32).ln() / scale * (nb - exact) as f32) as i64;
    (ret + large.min(nb - 1)) as usize
}

/// Owns operand 0, moving it out when node `i` is its last (and only-once) reader.
fn take0(
    bound: &[Option<&Tensor>],
    slots: &mut [Option<Tensor>],
    ins: &[ValueId],
    name: &str,
    last_use: &[usize],
    i: usize,
) -> Result<Vec<f32>, String> {
    let id = ins
        .first()
        .ok_or_else(|| format!("node '{name}' is missing operand 0"))?
        .0 as usize;
    if bound[id].is_none() && last_use[id] == i && !ins[1..].iter().any(|v| v.0 as usize == id) {
        if let Some(t) = slots[id].take() {
            return t.into_f32().map_err(|e| format!("node '{name}': {e}"));
        }
    }
    operand(bound, slots, ins, name, 0).map(<[f32]>::to_vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_rel_pos_bias_and_state_reorder() {
        let mut g = Graph::new();
        let table = g.weight("table", vec![32, 4], DType::F32);
        let q_pos = g.input("q_pos", vec![2], DType::U32);
        let k_pos = g.input("k_pos", vec![4], DType::U32);
        let rpb = g.add(
            "rpb",
            Op::RelPosBias {
                buckets: 32,
                max_distance: 128,
                bidirectional: true,
                heads: 4,
            },
            &[table, q_pos, k_pos],
            vec![4, 2, 4],
            DType::F32,
        );
        g.output(rpb);

        let mut w = Weights::new();
        w.insert("table", Tensor::F32(vec![0.5; 32 * 4]));
        let inputs = vec![Tensor::U32(vec![0, 1]), Tensor::U32(vec![0, 1, 2, 3])];
        let out = run_stateless(&g, &w, inputs).unwrap();
        assert_eq!(out[0].len(), 4 * 2 * 4);
    }

    use crate::graph::{BinaryKind, DType, Op, RopeVariant, UnaryKind, UpsampleMode};

    #[test]
    fn test_cpu_rmsnorm() {
        let mut g = Graph::new();
        let x = g.input("x", vec![2, 4], DType::F32);
        let gamma = g.weight("gamma", vec![4], DType::F32);
        let out = g.add(
            "norm",
            Op::RmsNorm { eps: 1e-5 },
            &[x, gamma],
            vec![2, 4],
            DType::F32,
        );
        g.output(out);

        let mut w = Weights::new();
        w.insert("gamma", Tensor::F32(vec![1.0, 1.0, 1.0, 1.0]));
        let inputs = vec![Tensor::F32(vec![
            1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0,
        ])];
        let res = run_stateless(&g, &w, inputs).unwrap();
        let out_slice = res[0].as_f32().unwrap();
        assert_eq!(out_slice.len(), 8);
        let rms = ((1.0 + 4.0 + 9.0 + 16.0) / 4.0f32 + 1e-5).sqrt();
        assert!((out_slice[0] - 1.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_cpu_swiglu_and_geglu() {
        let mut g = Graph::new();
        let gate = g.input("gate", vec![2], DType::F32);
        let up = g.input("up", vec![2], DType::F32);
        let s = g.add("swiglu", Op::SwiGlu, &[gate, up], vec![2], DType::F32);
        let ge = g.add("geglu", Op::GeGlu, &[gate, up], vec![2], DType::F32);
        g.output(s);
        g.output(ge);

        let inputs = vec![Tensor::F32(vec![0.0, 2.0]), Tensor::F32(vec![1.0, 3.0])];
        let res = run_stateless(&g, &Weights::new(), inputs).unwrap();
        let s_out = res[0].as_f32().unwrap();
        let ge_out = res[1].as_f32().unwrap();
        assert!((s_out[0] - 0.0).abs() < 1e-5);
        let silu_2 = 2.0 / (1.0 + (-2.0f32).exp());
        assert!((s_out[1] - silu_2 * 3.0).abs() < 1e-4);
        assert!((ge_out[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_unary_and_binary() {
        let mut g = Graph::new();
        let a = g.input("a", vec![3], DType::F32);
        let b = g.input("b", vec![1], DType::F32);
        let softplus = g.add(
            "sp",
            Op::Unary {
                kind: UnaryKind::Softplus,
            },
            &[a],
            vec![3],
            DType::F32,
        );
        let clamp = g.add(
            "cl",
            Op::Unary {
                kind: UnaryKind::Clamp(0.0, 10.0),
            },
            &[a],
            vec![3],
            DType::F32,
        );
        let add = g.add(
            "add",
            Op::Binary {
                kind: BinaryKind::Add,
            },
            &[a, b],
            vec![3],
            DType::F32,
        );
        let mul = g.add(
            "mul",
            Op::Binary {
                kind: BinaryKind::Mul,
            },
            &[a, b],
            vec![3],
            DType::F32,
        );
        g.output(softplus);
        g.output(clamp);
        g.output(add);
        g.output(mul);

        let inputs = vec![Tensor::F32(vec![-5.0, 0.0, 30.0]), Tensor::F32(vec![2.0])];
        let res = run_stateless(&g, &Weights::new(), inputs).unwrap();
        let sp = res[0].as_f32().unwrap();
        assert!(sp[0] > 0.0 && sp[0] < 0.01);
        assert!((sp[1] - (2.0f32).ln()).abs() < 1e-4);
        assert!((sp[2] - 30.0).abs() < 1e-4); // softplus numerical stability for > 20

        let cl = res[1].as_f32().unwrap();
        assert_eq!(cl, &[0.0, 0.0, 10.0]);

        let ad = res[2].as_f32().unwrap();
        assert_eq!(ad, &[-3.0, 2.0, 32.0]);

        let mu = res[3].as_f32().unwrap();
        assert_eq!(mu, &[-10.0, 0.0, 60.0]);
    }

    #[test]
    fn test_cpu_rope() {
        let mut g = Graph::new();
        let mut w = Weights::new();
        let x = g.input("x", vec![1, 4], DType::F32);
        let pos = g.input("pos", vec![1], DType::U32);
        let inv_freq = g.weight("rope.inv_freq.default", vec![2], DType::F32);
        w.insert(
            "rope.inv_freq.default".to_string(),
            Tensor::F32(vec![1.0, 0.01]),
        );

        let r_inter = g.add(
            "rope_inter",
            Op::Rope {
                head_dim: 4,
                rotary_dim: 4,
                variant: RopeVariant::Interleaved,
                mscale: 1.0,
            },
            &[x, pos, inv_freq],
            vec![1, 4],
            DType::F32,
        );
        let r_half = g.add(
            "rope_half",
            Op::Rope {
                head_dim: 4,
                rotary_dim: 4,
                variant: RopeVariant::HalfSplit,
                mscale: 1.0,
            },
            &[x, pos, inv_freq],
            vec![1, 4],
            DType::F32,
        );
        g.output(r_inter);
        g.output(r_half);

        let inputs = vec![Tensor::F32(vec![1.0, 0.0, 0.0, 1.0]), Tensor::U32(vec![0])];
        let res = run_stateless(&g, &w, inputs).unwrap();
        assert_eq!(res[0].as_f32().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(res[1].as_f32().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_attention_window_exact_w_keys() {
        // window = 4, query at pos 10 over keys 0..=10.
        // Exactly keys 7..=10 have non-zero weight (4 keys).
        let w = 4usize;
        let pos = 10usize;
        let mut visible = Vec::new();
        for ki in 0..=pos {
            if !(ki + w <= pos) {
                visible.push(ki);
            }
        }
        assert_eq!(visible, vec![7, 8, 9, 10]);
    }

    #[test]
    fn test_cpu_cached_attention() {
        let mut g = Graph::new();
        let q = g.input("q", vec![1, 4], DType::F32);
        let k = g.input("k", vec![1, 4], DType::F32);
        let v = g.input("v", vec![1, 4], DType::F32);
        let out = g.add(
            "attn",
            Op::CachedAttention {
                heads: 1,
                kv_heads: 1,
                head_dim: 4,
                v_head_dim: 4,
                scale: 0.5,
                causal: true,
                window: None,
                soft_cap: None,
                write: false,
            },
            &[q, k, v],
            vec![1, 4],
            DType::F32,
        );
        g.output(out);

        let inputs = vec![
            Tensor::F32(vec![1.0, 0.0, 0.0, 0.0]),
            Tensor::F32(vec![1.0, 0.0, 0.0, 0.0]),
            Tensor::F32(vec![0.5, 1.5, 2.5, 3.5]),
        ];
        let res = run_stateless(&g, &Weights::new(), inputs).unwrap();
        let out = res[0].as_f32().unwrap();
        assert_eq!(out, &[0.5, 1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_cpu_causal_conv1d_cached() {
        let mut g = Graph::new();
        let x = g.input("x", vec![3, 1], DType::F32); // 3 steps, 1 channel
        let w = g.weight("w", vec![2], DType::F32); // kernel 2
        let out = g.add(
            "conv",
            Op::CausalConv1dCached {
                kernel: 2,
                dilation: 1,
            },
            &[x, w],
            vec![3, 1],
            DType::F32,
        );
        g.output(out);

        let mut weights = Weights::new();
        weights.insert("w", Tensor::F32(vec![0.5, 1.0])); // tap 0 is earlier, tap 1 is current
        let inputs = vec![Tensor::F32(vec![2.0, 4.0, 6.0])];
        let res = run_stateless(&g, &weights, inputs).unwrap();
        let out = res[0].as_f32().unwrap();
        // step 0: 1.0 * 2.0 = 2.0
        // step 1: 0.5 * 2.0 + 1.0 * 4.0 = 5.0
        // step 2: 0.5 * 4.0 + 1.0 * 6.0 = 8.0
        assert_eq!(out, &[2.0, 5.0, 8.0]);
    }

    #[test]
    fn test_cpu_delta_rule() {
        let mut g = Graph::new();
        let q = g.input("q", vec![1, 2], DType::F32);
        let k = g.input("k", vec![1, 2], DType::F32);
        let v = g.input("v", vec![1, 2], DType::F32);
        let beta = g.input("beta", vec![1], DType::F32);
        let log_decay = g.input("log_decay", vec![1], DType::F32);
        let out = g.add(
            "delta",
            Op::DeltaRule {
                key_heads: 1,
                val_heads: 1,
                key_dim: 2,
                val_dim: 2,
            },
            &[q, k, v, beta, log_decay],
            vec![1, 2],
            DType::F32,
        );
        g.output(out);

        let inputs = vec![
            Tensor::F32(vec![1.0, 0.0]),
            Tensor::F32(vec![1.0, 0.0]),
            Tensor::F32(vec![3.0, 4.0]),
            Tensor::F32(vec![1.0]),
            Tensor::F32(vec![0.0]),
        ];
        let res = run_stateless(&g, &Weights::new(), inputs).unwrap();
        let out = res[0].as_f32().unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 3.0 / (2.0f32).sqrt()).abs() < 1e-4);
        assert!((out[1] - 4.0 / (2.0f32).sqrt()).abs() < 1e-4);
    }

    #[test]
    fn test_cpu_gather_scatter_pad_pool_upsample() {
        let mut g = Graph::new();
        let x = g.input("x", vec![2, 2], DType::F32);
        let idx = g.input("idx", vec![1], DType::U32);
        let gat = g.add(
            "gather",
            Op::Gather { axis: 0 },
            &[x, idx],
            vec![1, 2],
            DType::F32,
        );
        let pad = g.add(
            "pad",
            Op::Pad {
                padding: vec![1, 1],
                value: 0.0,
            },
            &[gat],
            vec![1, 4],
            DType::F32,
        );
        let up = g.add(
            "up",
            Op::Upsample {
                scale: 2,
                mode: UpsampleMode::Nearest,
            },
            &[x],
            vec![4, 4],
            DType::F32,
        );
        g.output(gat);
        g.output(pad);
        g.output(up);

        let inputs = vec![Tensor::F32(vec![1.0, 2.0, 3.0, 4.0]), Tensor::U32(vec![1])];
        let res = run_stateless(&g, &Weights::new(), inputs).unwrap();
        assert_eq!(res[0].as_f32().unwrap(), &[3.0, 4.0]);
        assert_eq!(res[1].as_f32().unwrap(), &[0.0, 3.0, 4.0, 0.0]);
        assert_eq!(res[2].as_f32().unwrap().len(), 16);
    }

    #[test]
    fn test_cpu_cached_attention_step_by_step_matches_prefill() {
        let mut g = Graph::new();
        let q = g.input("q", vec![1, 4], DType::F32);
        let k = g.input("k", vec![1, 4], DType::F32);
        let v = g.input("v", vec![1, 4], DType::F32);
        let ks = g.state("k_state", vec![16, 4], DType::F32, false);
        let vs = g.state("v_state", vec![16, 4], DType::F32, false);
        let pos = g.input("pos", vec![1], DType::U32);
        let out = g.add(
            "attn",
            Op::CachedAttention {
                heads: 1,
                kv_heads: 1,
                head_dim: 4,
                v_head_dim: 4,
                scale: 0.5,
                causal: true,
                window: None,
                soft_cap: None,
                write: true,
            },
            &[q, k, v, ks, vs, pos],
            vec![1, 4],
            DType::F32,
        );
        g.output(out);

        let mut state = StateStore::new();
        // Step 0
        let r0 = run(
            &g,
            &Weights::new(),
            &mut state,
            vec![
                Tensor::F32(vec![1.0, 0.0, 0.0, 0.0]),
                Tensor::F32(vec![1.0, 0.0, 0.0, 0.0]),
                Tensor::F32(vec![1.0, 2.0, 3.0, 4.0]),
                Tensor::U32(vec![0]),
            ],
        )
        .unwrap();
        assert_eq!(r0[0].as_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

        // Step 1
        let r1 = run(
            &g,
            &Weights::new(),
            &mut state,
            vec![
                Tensor::F32(vec![0.0, 1.0, 0.0, 0.0]),
                Tensor::F32(vec![0.0, 1.0, 0.0, 0.0]),
                Tensor::F32(vec![10.0, 20.0, 30.0, 40.0]),
                Tensor::U32(vec![1]),
            ],
        )
        .unwrap();
        let o1 = r1[0].as_f32().unwrap();
        assert_eq!(o1.len(), 4);
        // Step 1 q=[0,1,0,0] attends over k0=[1,0,0,0] and k1=[0,1,0,0]
        assert!(o1[0] > 1.0);
    }

    /// A `write: false` reader of the same state value, after the writer, sees the writer's row.
    #[test]
    fn test_cpu_shared_kv_reader_sees_the_write() {
        let mut g = Graph::new();
        let q = g.input("q", vec![1, 4], DType::F32);
        let k = g.input("k", vec![1, 4], DType::F32);
        let v = g.input("v", vec![1, 4], DType::F32);
        let ks = g.state("k_state", vec![16, 4], DType::F32, false);
        let vs = g.state("v_state", vec![16, 4], DType::F32, false);
        let pos = g.input("pos", vec![1], DType::U32);
        let attn = |write| Op::CachedAttention {
            heads: 1,
            kv_heads: 1,
            head_dim: 4,
            v_head_dim: 4,
            scale: 0.5,
            causal: true,
            window: None,
            soft_cap: None,
            write,
        };
        let a = g.add(
            "writer",
            attn(true),
            &[q, k, v, ks, vs, pos],
            vec![1, 4],
            DType::F32,
        );
        let b = g.add(
            "reader",
            attn(false),
            &[q, q, q, ks, vs, pos],
            vec![1, 4],
            DType::F32,
        );
        g.output(a);
        g.output(b);
        let mut state = StateStore::new();
        let r = run(
            &g,
            &Weights::new(),
            &mut state,
            vec![
                Tensor::F32(vec![1.0, 0.0, 0.0, 0.0]),
                Tensor::F32(vec![1.0, 0.0, 0.0, 0.0]),
                Tensor::F32(vec![1.0, 2.0, 3.0, 4.0]),
                Tensor::U32(vec![0]),
            ],
        )
        .unwrap();
        assert_eq!(r[1].as_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(r[0].as_f32().unwrap(), r[1].as_f32().unwrap());
    }

    #[test]
    fn test_cpu_causal_conv1d_step_by_step_matches_prefill() {
        let mut g_prefill = Graph::new();
        let x_pre = g_prefill.input("x", vec![2, 1], DType::F32);
        let w_pre = g_prefill.weight("w", vec![2], DType::F32);
        let out_pre = g_prefill.add(
            "conv",
            Op::CausalConv1dCached {
                kernel: 2,
                dilation: 1,
            },
            &[x_pre, w_pre],
            vec![2, 1],
            DType::F32,
        );
        g_prefill.output(out_pre);

        let mut weights = Weights::new();
        weights.insert("w", Tensor::F32(vec![0.3, 0.7]));

        let pre_res =
            run_stateless(&g_prefill, &weights, vec![Tensor::F32(vec![1.0, 5.0])]).unwrap();
        let expected = pre_res[0].as_f32().unwrap();

        // Step by step with state
        let mut g_step = Graph::new();
        let x_step = g_step.input("x", vec![1, 1], DType::F32);
        let w_step = g_step.weight("w", vec![2], DType::F32);
        let s_step = g_step.state("conv_state", vec![1, 1], DType::F32, true);
        let out_step = g_step.add(
            "conv",
            Op::CausalConv1dCached {
                kernel: 2,
                dilation: 1,
            },
            &[x_step, w_step, s_step],
            vec![1, 1],
            DType::F32,
        );
        g_step.output(out_step);

        let mut state = StateStore::new();
        let step0 = run(&g_step, &weights, &mut state, vec![Tensor::F32(vec![1.0])]).unwrap();
        let step1 = run(&g_step, &weights, &mut state, vec![Tensor::F32(vec![5.0])]).unwrap();

        assert_eq!(step0[0].as_f32().unwrap()[0], expected[0]);
        assert!((step1[0].as_f32().unwrap()[0] - expected[1]).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_delta_rule_step_by_step_matches_prefill() {
        let mut g_prefill = Graph::new();
        let q = g_prefill.input("q", vec![2, 2], DType::F32);
        let k = g_prefill.input("k", vec![2, 2], DType::F32);
        let v = g_prefill.input("v", vec![2, 2], DType::F32);
        let beta = g_prefill.input("beta", vec![2], DType::F32);
        let log_decay = g_prefill.input("log_decay", vec![2], DType::F32);
        let out = g_prefill.add(
            "delta",
            Op::DeltaRule {
                key_heads: 1,
                val_heads: 1,
                key_dim: 2,
                val_dim: 2,
            },
            &[q, k, v, beta, log_decay],
            vec![2, 2],
            DType::F32,
        );
        g_prefill.output(out);

        let q_data = vec![1.0, 0.0, 0.0, 1.0];
        let k_data = vec![1.0, 0.0, 0.0, 1.0];
        let v_data = vec![2.0, 3.0, 4.0, 5.0];
        let beta_data = vec![1.0, 1.0];
        let ld_data = vec![0.0, 0.0];

        let pre_res = run_stateless(
            &g_prefill,
            &Weights::new(),
            vec![
                Tensor::F32(q_data.clone()),
                Tensor::F32(k_data.clone()),
                Tensor::F32(v_data.clone()),
                Tensor::F32(beta_data.clone()),
                Tensor::F32(ld_data.clone()),
            ],
        )
        .unwrap();
        let expected = pre_res[0].as_f32().unwrap();

        // Now step by step with persistent ssm_state
        let mut g_step = Graph::new();
        let q_s = g_step.input("q", vec![1, 2], DType::F32);
        let k_s = g_step.input("k", vec![1, 2], DType::F32);
        let v_s = g_step.input("v", vec![1, 2], DType::F32);
        let beta_s = g_step.input("beta", vec![1], DType::F32);
        let ld_s = g_step.input("log_decay", vec![1], DType::F32);
        let ssm_state = g_step.state("ssm_state", vec![1, 2, 2], DType::F32, true);
        let out_s = g_step.add(
            "delta",
            Op::DeltaRule {
                key_heads: 1,
                val_heads: 1,
                key_dim: 2,
                val_dim: 2,
            },
            &[q_s, k_s, v_s, beta_s, ld_s, ssm_state],
            vec![1, 2],
            DType::F32,
        );
        g_step.output(out_s);

        let mut state = StateStore::new();
        let step0 = run(
            &g_step,
            &Weights::new(),
            &mut state,
            vec![
                Tensor::F32(vec![1.0, 0.0]),
                Tensor::F32(vec![1.0, 0.0]),
                Tensor::F32(vec![2.0, 3.0]),
                Tensor::F32(vec![1.0]),
                Tensor::F32(vec![0.0]),
            ],
        )
        .unwrap();
        let step1 = run(
            &g_step,
            &Weights::new(),
            &mut state,
            vec![
                Tensor::F32(vec![0.0, 1.0]),
                Tensor::F32(vec![0.0, 1.0]),
                Tensor::F32(vec![4.0, 5.0]),
                Tensor::F32(vec![1.0]),
                Tensor::F32(vec![0.0]),
            ],
        )
        .unwrap();

        let s0 = step0[0].as_f32().unwrap();
        let s1 = step1[0].as_f32().unwrap();

        assert!((s0[0] - expected[0]).abs() < 1e-5);
        assert!((s0[1] - expected[1]).abs() < 1e-5);
        assert!((s1[0] - expected[2]).abs() < 1e-5);
        assert!((s1[1] - expected[3]).abs() < 1e-5);
    }
}
