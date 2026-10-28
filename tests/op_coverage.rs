//! Every graph op at every GPU activation dtype is lowered by some avenue, or is listed in
//! `KNOWN_UNLOWERED`. The list may only shrink: lowering an entry makes this test demand its removal.

use gibbs_engine::graph::avenues::{explore_node, Verdict};
use gibbs_engine::graph::{
    BinaryKind, DType, MegakernelDescriptor, MegakernelKind, Op, PairLayout, ReduceKind,
    RopeVariant, UnaryKind, UpsampleMode,
};

/// `(op key, dtype, reason)`; the reason is the avenue verdict text of the first rejected avenue.
const KNOWN_UNLOWERED: &[(&str, &str, &str)] = &[];

const DTYPES: [(DType, &str); 3] = [(DType::F32, "F32"), (DType::F16, "F16"), (DType::BF16, "BF16")];

/// One index per `Op` variant. The match has no wildcard arm: adding a variant breaks the build here.
fn ordinal(op: &Op) -> usize {
    match op {
        Op::Embedding => 0,
        Op::Linear => 1,
        Op::MatMul { .. } => 2,
        Op::LayerNorm { .. } => 3,
        Op::RmsNorm { .. } => 4,
        Op::GroupNorm { .. } => 5,
        Op::InstanceNorm { .. } => 6,
        Op::Rope { .. } => 7,
        Op::Attention { .. } => 8,
        Op::SwiGlu => 9,
        Op::GeGlu => 10,
        Op::Unary { .. } => 11,
        Op::Binary { .. } => 12,
        Op::CachedAttention { .. } => 13,
        Op::CausalConv1dCached { .. } => 15,
        Op::DeltaRule { .. } => 16,
        Op::Gather { .. } => 17,
        Op::Scatter { .. } => 18,
        Op::Pad { .. } => 19,
        Op::AvgPool2d { .. } => 20,
        Op::Upsample { .. } => 21,
        Op::Conv3d { .. } => 22,
        Op::MoeLinear { .. } => 23,
        Op::Softmax { .. } => 24,
        Op::TopK { .. } => 25,
        Op::GatherCols => 41,
        Op::RelPosBias { .. } => 26,
        Op::StateReorder => 27,
        Op::Conv2d { .. } => 28,
        Op::Conv1d { .. } => 29,
        Op::Reshape { .. } => 30,
        Op::Transpose { .. } => 31,
        Op::Concat { .. } => 32,
        Op::Slice { .. } => 33,
        Op::Squeeze { .. } => 34,
        Op::Unsqueeze { .. } => 35,
        Op::Tile { .. } => 36,
        Op::Reduce { .. } => 37,
        Op::MeanPool => 38,
        Op::L2Normalize => 39,
        Op::Megakernel(_) => 40,
        Op::StateWrite => 42,
        Op::QsaSelect { .. } => 43,
        Op::QsaBias { .. } => 14,
    }
}
const VARIANTS: usize = 44;

/// Key used in `KNOWN_UNLOWERED`: the op name, plus the kind for ops whose support varies by kind.
fn key(op: &Op) -> String {
    match op {
        Op::Unary { kind } => format!("Unary::{kind:?}"),
        Op::Binary { kind } => format!("Binary::{kind:?}"),
        Op::Reduce { kind, .. } => format!("Reduce::{kind:?}"),
        Op::Megakernel(d) => match d.kind {
            MegakernelKind::Mlp { .. } => "Megakernel::Mlp".to_string(),
        },
        other => other.name().to_string(),
    }
}

fn all_ops() -> Vec<Op> {
    let mut ops = vec![
        Op::Embedding,
        Op::Linear,
        Op::MatMul { transpose_a: false, transpose_b: true },
        Op::LayerNorm { eps: 1e-5 },
        Op::RmsNorm { eps: 1e-6 },
        Op::GroupNorm { groups: 8, eps: 1e-5 },
        Op::InstanceNorm { eps: 1e-5 },
        Op::Rope { head_dim: 16, rotary_dim: 16, variant: RopeVariant::HalfSplit, mscale: 1.0 },
        Op::Rope { head_dim: 16, rotary_dim: 16, variant: RopeVariant::Interleaved, mscale: 1.0 },
        Op::Rope {
            head_dim: 16,
            rotary_dim: 16,
            variant: RopeVariant::MRope { sections: [8, 4, 4], interleaved: true, pairs: PairLayout::Half },
            mscale: 1.0,
        },
        Op::Attention { heads: 2, kv_heads: 2, causal: true, window: None, soft_cap: None, scale: None },
        Op::SwiGlu,
        Op::GeGlu,
        Op::CachedAttention {
            heads: 2, kv_heads: 1, head_dim: 16, v_head_dim: 16, scale: 1.0,
            causal: true, window: Some(4), soft_cap: Some(30.0), write: true,
        },
        Op::CausalConv1dCached { kernel: 4, dilation: 1 },
        Op::DeltaRule { key_heads: 2, val_heads: 2, key_dim: 16, val_dim: 16 },
        Op::Gather { axis: 0 },
        Op::Scatter { axis: 0 },
        Op::Pad { padding: vec![0, 1], value: 0.0 },
        Op::AvgPool2d { kernel: [2, 2], stride: [2, 2] },
        Op::Upsample { scale: 2, mode: UpsampleMode::Nearest },
        Op::Conv3d { stride: [1, 1, 1], padding: [0, 0, 1, 1, 1, 1], dilation: [1, 1, 1], groups: 1 },
        Op::MoeLinear { experts: 4, top_k: 2 },
        Op::GatherCols,
        Op::Softmax { axis: -1 },
        Op::TopK { k: 2 },
        Op::RelPosBias { buckets: 32, max_distance: 128, bidirectional: true, heads: 2 },
        Op::StateReorder,
        Op::StateWrite,
        Op::QsaSelect { ratio: 4, top_blocks: 2 },
        Op::QsaBias { ratio: 4, top_blocks: 2 },
        Op::Conv2d { stride: [1, 1], padding: [1, 1, 1, 1], dilation: [1, 1], groups: 1 },
        Op::Conv1d { stride: 1, padding: [1, 1], dilation: 1, groups: 1 },
        Op::Reshape { shape: Some(vec![2, 2]) },
        Op::Transpose { perm: vec![1, 0] },
        Op::Concat { axis: 0 },
        Op::Slice { axis: 0, start: 0, end: 1 },
        Op::Squeeze { axes: vec![0] },
        Op::Unsqueeze { axes: vec![0] },
        Op::Tile { reps: vec![2] },
        Op::MeanPool,
        Op::L2Normalize,
        Op::Megakernel(MegakernelDescriptor {
            kind: MegakernelKind::Mlp {
                hidden_dim: 64, intermediate_dim: 128, eps: 1e-6, is_geglu: false, has_post_norm: false,
            },
            name: "mlp".into(),
        }),
    ];
    for kind in [
        UnaryKind::Exp, UnaryKind::Sigmoid, UnaryKind::Tanh, UnaryKind::Softplus, UnaryKind::Neg,
        UnaryKind::Sqrt, UnaryKind::Rsqrt, UnaryKind::Abs, UnaryKind::Relu, UnaryKind::Relu2,
        UnaryKind::Silu, UnaryKind::GeluErf, UnaryKind::GeluTanh, UnaryKind::Scale(0.5),
        UnaryKind::Clamp(-1.0, 1.0), UnaryKind::SoftCap(30.0), UnaryKind::Ln, UnaryKind::RoundEven,
        UnaryKind::Sign,
    ] {
        ops.push(Op::Unary { kind });
    }
    for kind in [BinaryKind::Add, BinaryKind::Sub, BinaryKind::Mul, BinaryKind::Div, BinaryKind::Max, BinaryKind::Min] {
        ops.push(Op::Binary { kind });
    }
    for kind in [ReduceKind::Sum, ReduceKind::Mean, ReduceKind::Max, ReduceKind::Min] {
        ops.push(Op::Reduce { kind, axes: vec![-1], keep_dims: true });
    }
    ops
}

#[test]
fn every_op_is_lowered_or_known_unlowered() {
    let ops = all_ops();
    let mut seen = [false; VARIANTS];
    for op in &ops {
        seen[ordinal(op)] = true;
    }
    assert!(seen.iter().all(|&s| s), "all_ops() misses Op variant(s): {:?}", seen);

    let mut unexpected = Vec::new();
    let mut stale = Vec::new();
    let mut actual_unlowered = Vec::new();
    for op in &ops {
        for (dtype, dname) in DTYPES {
            let report = explore_node("sample", op, dtype, gibbs_engine::hardware::Device::Metal);
            let k = key(op);
            let listed = KNOWN_UNLOWERED.iter().any(|(o, d, _)| *o == k && *d == dname);
            match report.lowered_by() {
                Some(_) if listed => stale.push(format!("remove {k} {dname} from KNOWN_UNLOWERED")),
                Some(_) => {}
                None => {
                    let reason = match &report.attempts[0].verdict {
                        Verdict::Rejected(r) => r.clone(),
                        Verdict::Lowered(_) => unreachable!(),
                    };
                    actual_unlowered.push(format!("    (\"{k}\", \"{dname}\", \"{reason}\"),"));
                    if !listed {
                        unexpected.push(format!("{k} {dname}: {reason}"));
                    }
                }
            }
        }
    }
    println!("unlowered pairs: {}\n{}", actual_unlowered.len(), actual_unlowered.join("\n"));
    assert!(stale.is_empty(), "{}", stale.join("\n"));
    assert!(
        unexpected.is_empty(),
        "ops with no lowering and no KNOWN_UNLOWERED entry:\n{}",
        unexpected.join("\n")
    );
}
