//! One small graph per op kind ([`op_key`]): compiled once per plugin to learn which ops it
//! accepts ([`super::supports_op`]), and runnable against the CPU executor.

use crate::graph::{
    BinaryKind, DType, Graph, MegakernelDescriptor, MegakernelKind, Op, PairLayout, ReduceKind,
    RopeVariant, Tensor, UnaryKind, UpsampleMode, Weights,
};

/// The probe key of `op`: its name, plus the sub-kind for ops whose kinds lower differently.
pub fn op_key(op: &Op) -> String {
    let sub = match op {
        Op::Unary { kind } => format!("{kind:?}"),
        Op::Binary { kind } => format!("{kind:?}"),
        Op::Reduce { kind, .. } => format!("{kind:?}"),
        Op::Upsample { mode, .. } => format!("{mode:?}"),
        Op::Rope { variant, .. } => format!("{variant:?}"),
        Op::Conv1d { groups, .. } | Op::Conv2d { groups, .. } | Op::Conv3d { groups, .. }
            if *groups > 1 =>
        {
            "grouped".into()
        }
        _ => return op.name().to_string(),
    };
    let sub = sub.split(['(', ' ', '{']).next().unwrap_or_default();
    format!("{}.{sub}", op.name())
}

/// A probe graph with its weights and inputs.
pub struct Probe {
    pub key: String,
    pub graph: Graph,
    pub weights: Weights,
    pub inputs: Vec<Tensor>,
}

/// Deterministic values in `[lo, hi)`.
fn vals(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (hi - lo) * ((s >> 40) as f32 / (1u64 << 24) as f32)
        })
        .collect()
}

struct B {
    g: Graph,
    w: Weights,
    inputs: Vec<Tensor>,
    seed: u64,
}

impl B {
    fn new() -> Self {
        B {
            g: Graph::new(),
            w: Weights::new(),
            inputs: Vec::new(),
            seed: 1,
        }
    }

    fn x(&mut self, shape: &[usize]) -> crate::graph::ValueId {
        self.xr(shape, -1.0, 1.0)
    }

    fn xr(&mut self, shape: &[usize], lo: f32, hi: f32) -> crate::graph::ValueId {
        self.seed += 1;
        let n = shape.iter().product();
        self.inputs.push(Tensor::F32(vals(n, self.seed, lo, hi)));
        let name = format!("x{}", self.inputs.len());
        self.g.input(name, shape.to_vec(), DType::F32)
    }

    fn ids(&mut self, shape: &[usize], data: Vec<u32>) -> crate::graph::ValueId {
        self.inputs.push(Tensor::U32(data));
        let name = format!("i{}", self.inputs.len());
        self.g.input(name, shape.to_vec(), DType::U32)
    }

    fn w(&mut self, shape: &[usize]) -> crate::graph::ValueId {
        self.seed += 1;
        let n = shape.iter().product();
        let name = format!("w{}", self.seed);
        self.w.insert(&name, Tensor::F32(vals(n, self.seed, -0.5, 0.5)));
        self.g.weight(name, shape.to_vec(), DType::F32)
    }

    fn wv(&mut self, shape: &[usize], data: Vec<f32>) -> crate::graph::ValueId {
        self.seed += 1;
        let name = format!("w{}", self.seed);
        self.w.insert(&name, Tensor::F32(data));
        self.g.weight(name, shape.to_vec(), DType::F32)
    }

    fn state(&mut self, shape: &[usize], recurrent: bool) -> crate::graph::ValueId {
        self.seed += 1;
        self.g
            .state(format!("s{}", self.seed), shape.to_vec(), DType::F32, recurrent)
    }

    fn op(&mut self, op: Op, ins: &[crate::graph::ValueId], shape: &[usize], dt: DType) -> Probe {
        let key = op_key(&op);
        let out = self.g.add("probe", op, ins, shape.to_vec(), dt);
        self.g.output(out);
        Probe {
            key,
            graph: std::mem::take(&mut self.g),
            weights: std::mem::take(&mut self.w),
            inputs: std::mem::take(&mut self.inputs),
        }
    }
}

/// Every probe, one per [`op_key`] value the emitter lowers.
pub fn probe_graphs() -> Vec<Probe> {
    use DType::{F32, U32};
    let mut p = Vec::new();
    let mut b = B::new();
    let ids = |n: usize, m: u32| (0..n as u32).map(|i| (i * 7 + 3) % m).collect::<Vec<_>>();

    let t = b.w(&[16, 8]);
    let i = b.ids(&[3], ids(3, 16));
    p.push(b.op(Op::Embedding, &[t, i], &[3, 8], F32));

    let (x, w, bias) = (b.x(&[3, 8]), b.w(&[5, 8]), b.w(&[5]));
    p.push(b.op(Op::Linear, &[x, w, bias], &[3, 5], F32));

    let (x, y) = (b.x(&[2, 3, 4]), b.x(&[5, 4]));
    let mm = Op::MatMul {
        transpose_a: false,
        transpose_b: true,
    };
    p.push(b.op(mm, &[x, y], &[2, 3, 5], F32));

    let (x, gm, bt) = (b.x(&[3, 8]), b.w(&[8]), b.w(&[8]));
    p.push(b.op(Op::LayerNorm { eps: 1e-5 }, &[x, gm, bt], &[3, 8], F32));
    let (x, gm) = (b.x(&[3, 8]), b.w(&[8]));
    p.push(b.op(Op::RmsNorm { eps: 1e-6 }, &[x, gm], &[3, 8], F32));
    let (x, gm, bt) = (b.x(&[4, 6]), b.w(&[4]), b.w(&[4]));
    let gn = Op::GroupNorm {
        groups: 2,
        eps: 1e-5,
    };
    p.push(b.op(gn, &[x, gm, bt], &[4, 6], F32));
    let (x, gm, bt) = (b.x(&[4, 6]), b.w(&[4]), b.w(&[4]));
    p.push(b.op(Op::InstanceNorm { eps: 1e-5 }, &[x, gm, bt], &[4, 6], F32));

    for variant in [
        RopeVariant::Interleaved,
        RopeVariant::HalfSplit,
        RopeVariant::MRope {
            sections: [2, 1, 1],
            interleaved: false,
            pairs: PairLayout::Half,
        },
    ] {
        let x = b.x(&[3, 16]);
        let np = if matches!(variant, RopeVariant::MRope { .. }) { 9 } else { 3 };
        let pos = b.ids(&[np], ids(np, 11));
        let inv = b.wv(&[4], (0..4).map(|j| 10000f32.powf(-(j as f32) / 4.0)).collect());
        let op = Op::Rope {
            head_dim: 8,
            rotary_dim: 8,
            variant,
            mscale: 1.0,
        };
        p.push(b.op(op, &[x, pos, inv], &[3, 16], F32));
    }

    let (q, k, v, bias) = (b.x(&[3, 8]), b.x(&[5, 4]), b.x(&[5, 4]), b.x(&[5]));
    let at = Op::Attention {
        heads: 2,
        kv_heads: 1,
        causal: true,
        window: Some(4),
        soft_cap: Some(20.0),
        scale: None,
    };
    p.push(b.op(at, &[q, k, v, bias], &[3, 8], F32));

    for op in [Op::SwiGlu, Op::GeGlu] {
        let (g, u) = (b.x(&[3, 8]), b.x(&[3, 8]));
        p.push(b.op(op, &[g, u], &[3, 8], F32));
    }

    for kind in [
        UnaryKind::Exp,
        UnaryKind::Sigmoid,
        UnaryKind::Tanh,
        UnaryKind::Softplus,
        UnaryKind::Neg,
        UnaryKind::Sqrt,
        UnaryKind::Rsqrt,
        UnaryKind::Abs,
        UnaryKind::Relu,
        UnaryKind::Relu2,
        UnaryKind::Silu,
        UnaryKind::GeluErf,
        UnaryKind::GeluTanh,
        UnaryKind::Scale(0.5),
        UnaryKind::Clamp(-0.5, 0.5),
        UnaryKind::SoftCap(0.7),
        UnaryKind::Ln,
        UnaryKind::RoundEven,
        UnaryKind::Sign,
    ] {
        let positive = matches!(kind, UnaryKind::Sqrt | UnaryKind::Rsqrt | UnaryKind::Ln);
        let x = if positive { b.xr(&[3, 8], 0.5, 2.0) } else { b.xr(&[3, 8], -3.0, 3.0) };
        p.push(b.op(Op::Unary { kind }, &[x], &[3, 8], F32));
    }
    for kind in [
        BinaryKind::Add,
        BinaryKind::Sub,
        BinaryKind::Mul,
        BinaryKind::Div,
        BinaryKind::Max,
        BinaryKind::Min,
    ] {
        let (x, y) = (b.x(&[3, 8]), b.xr(&[8], 0.5, 2.0));
        p.push(b.op(Op::Binary { kind }, &[x, y], &[3, 8], F32));
    }

    let (q, kn, vn) = (b.x(&[2, 8]), b.x(&[2, 4]), b.x(&[2, 4]));
    let (ks, vs) = (b.state(&[16, 4], false), b.state(&[16, 4], false));
    let pos = b.ids(&[2], vec![5, 6]);
    let ca = Op::CachedAttention {
        heads: 2,
        kv_heads: 1,
        head_dim: 4,
        v_head_dim: 4,
        scale: 0.5,
        causal: true,
        window: Some(6),
        soft_cap: None,
        write: true,
    };
    p.push(b.op(ca, &[q, kn, vn, ks, vs, pos], &[2, 8], F32));

    let (x, w, s) = (b.x(&[2, 4]), b.w(&[4, 3]), b.state(&[2, 4], true));
    let cc = Op::CausalConv1dCached {
        kernel: 3,
        dilation: 1,
    };
    p.push(b.op(cc, &[x, w, s], &[2, 4], F32));

    let (q, k, v) = (b.x(&[3, 4]), b.x(&[3, 4]), b.x(&[3, 8]));
    let (beta, ld) = (b.xr(&[3, 2], 0.1, 0.9), b.xr(&[3, 2], -1.0, -0.1));
    let s = b.state(&[2, 4, 4], true);
    let dr = Op::DeltaRule {
        key_heads: 1,
        val_heads: 2,
        key_dim: 4,
        val_dim: 4,
    };
    p.push(b.op(dr, &[q, k, v, beta, ld, s], &[3, 8], F32));

    let (x, i) = (b.x(&[3, 5]), b.ids(&[2], vec![4, 1]));
    p.push(b.op(Op::Gather { axis: 1 }, &[x, i], &[3, 2], F32));
    let (x, i, u) = (b.x(&[5, 3]), b.ids(&[2], vec![3, 0]), b.x(&[2, 3]));
    p.push(b.op(Op::Scatter { axis: 0 }, &[x, i, u], &[5, 3], F32));
    let x = b.x(&[3, 4]);
    let pad = Op::Pad {
        padding: vec![1, 2],
        value: 0.25,
    };
    p.push(b.op(pad, &[x], &[3, 7], F32));
    let x = b.x(&[2, 4, 4]);
    let ap = Op::AvgPool2d {
        kernel: [2, 2],
        stride: [2, 2],
    };
    p.push(b.op(ap, &[x], &[2, 2, 2], F32));
    for mode in [
        UpsampleMode::Nearest,
        UpsampleMode::Bilinear,
        UpsampleMode::Bicubic,
    ] {
        let x = b.x(&[2, 3, 3]);
        p.push(b.op(Op::Upsample { scale: 2, mode }, &[x], &[2, 6, 6], F32));
    }
    for groups in [1, 2] {
        let (x, w) = (b.x(&[2, 3, 4, 4]), b.w(&[4, 2 / groups, 2, 2, 2]));
        let c3 = Op::Conv3d {
            stride: [1, 1, 1],
            padding: [0, 1, 1, 0, 1, 1],
            dilation: [1, 1, 1],
            groups,
        };
        p.push(b.op(c3, &[x, w], &[4, 2, 5, 5], F32));
    }
    let x = b.x(&[2, 8]);
    p.push(b.op(Op::TopK { k: 3 }, &[x], &[2, 3], U32));
    let (x, i) = (b.x(&[2, 8]), b.ids(&[2, 3], ids(6, 8)));
    p.push(b.op(Op::GatherCols, &[x, i], &[2, 3], F32));
    let (x, bank, i) = (b.x(&[3, 8]), b.w(&[20, 8]), b.ids(&[3, 2], ids(6, 4)));
    let moe = Op::MoeLinear {
        experts: 4,
        top_k: 2,
    };
    p.push(b.op(moe, &[x, bank, i], &[6, 5], F32));
    let x = b.x(&[3, 8]);
    p.push(b.op(Op::Softmax { axis: -1 }, &[x], &[3, 8], F32));
    let (t, qp, kp) = (b.w(&[8, 2]), b.ids(&[3], vec![0, 1, 2]), b.ids(&[4], vec![0, 1, 2, 7]));
    let rp = Op::RelPosBias {
        buckets: 8,
        max_distance: 16,
        bidirectional: true,
        heads: 2,
    };
    p.push(b.op(rp, &[t, qp, kp], &[2, 3, 4], F32));
    let (s, i) = (b.state(&[3, 4], false), b.ids(&[3], vec![2, 0, 1]));
    p.push(b.op(Op::StateReorder, &[s, i], &[3, 4], F32));
    let (s, r, p0) = (b.state(&[8, 4], false), b.x(&[2, 4]), b.ids(&[1], vec![3]));
    p.push(b.op(Op::StateWrite, &[s, r, p0], &[1], U32));
    for bias in [false, true] {
        let (sc, pos) = (b.x(&[3, 4]), b.ids(&[3], vec![2, 6, 7]));
        let op = if bias {
            Op::QsaBias {
                ratio: 2,
                top_blocks: 2,
            }
        } else {
            Op::QsaSelect {
                ratio: 2,
                top_blocks: 2,
            }
        };
        p.push(b.op(op, &[sc, pos], &[3, 5], if bias { F32 } else { U32 }));
    }
    for groups in [1, 2] {
        let (x, w, bi) = (b.x(&[4, 6, 6]), b.w(&[4, 4 / groups, 3, 3]), b.w(&[4]));
        let c2 = Op::Conv2d {
            stride: [1, 1],
            padding: [1, 1, 1, 1],
            dilation: [1, 1],
            groups,
        };
        p.push(b.op(c2, &[x, w, bi], &[4, 6, 6], F32));
        let (x, w) = (b.x(&[4, 8]), b.w(&[4, 4 / groups, 3]));
        let c1 = Op::Conv1d {
            stride: 1,
            padding: [1, 1],
            dilation: 1,
            groups,
        };
        p.push(b.op(c1, &[x, w], &[4, 8], F32));
    }
    let x = b.x(&[3, 8]);
    p.push(b.op(Op::Reshape { shape: Some(vec![4, 6]) }, &[x], &[4, 6], F32));
    let x = b.x(&[2, 3, 4]);
    let tr = Op::Transpose {
        perm: vec![2, 0, 1],
    };
    p.push(b.op(tr, &[x], &[4, 2, 3], F32));
    let (x, y) = (b.x(&[2, 3]), b.x(&[2, 5]));
    p.push(b.op(Op::Concat { axis: 1 }, &[x, y], &[2, 8], F32));
    let x = b.x(&[4, 6]);
    let sl = Op::Slice {
        axis: 1,
        start: 1,
        end: -1,
    };
    p.push(b.op(sl, &[x], &[4, 4], F32));
    let x = b.x(&[3, 1, 4]);
    p.push(b.op(Op::Squeeze { axes: vec![1] }, &[x], &[3, 4], F32));
    let x = b.x(&[3, 4]);
    p.push(b.op(Op::Unsqueeze { axes: vec![0] }, &[x], &[1, 3, 4], F32));
    let x = b.x(&[2, 3]);
    p.push(b.op(Op::Tile { reps: vec![2, 2] }, &[x], &[4, 6], F32));
    for kind in [
        ReduceKind::Sum,
        ReduceKind::Mean,
        ReduceKind::Max,
        ReduceKind::Min,
    ] {
        let x = b.x(&[3, 4, 5]);
        let op = Op::Reduce {
            kind,
            axes: vec![1],
            keep_dims: true,
        };
        p.push(b.op(op, &[x], &[3, 1, 5], F32));
    }
    let (x, m) = (b.x(&[4, 6]), b.ids(&[4], vec![1, 0, 1, 1]));
    p.push(b.op(Op::MeanPool, &[x, m], &[6], F32));
    let x = b.x(&[3, 8]);
    p.push(b.op(Op::L2Normalize, &[x], &[3, 8], F32));
    let (x, nw, gw, uw, dw, pw) = (
        b.x(&[3, 8]),
        b.w(&[8]),
        b.w(&[16, 8]),
        b.w(&[16, 8]),
        b.w(&[8, 16]),
        b.w(&[8]),
    );
    let mk = Op::Megakernel(MegakernelDescriptor {
        kind: MegakernelKind::Mlp {
            hidden_dim: 8,
            intermediate_dim: 16,
            eps: 1e-6,
            is_geglu: false,
            has_post_norm: true,
        },
        name: "mlp".into(),
    });
    p.push(b.op(mk, &[x, nw, gw, uw, dw, pw], &[3, 8], F32));
    p
}
