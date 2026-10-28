use super::*;
use crate::graph::{
    BinaryKind, MegakernelDescriptor, MegakernelKind, ReduceKind, RopeVariant, StateStore,
    UnaryKind, UpsampleMode, ValueId,
};

struct Rng(u64);

impl Rng {
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale
            })
            .collect()
    }
}

/// A graph under construction with its weights and input data.
struct Case {
    name: &'static str,
    g: Graph,
    w: Weights,
    inputs: Vec<Tensor>,
    rng: Rng,
}

impl Case {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            g: Graph::new(),
            w: Weights::new(),
            inputs: Vec::new(),
            rng: Rng(name.len() as u64 * 7919 + 17),
        }
    }

    fn input(&mut self, name: &str, shape: Vec<usize>, scale: f32) -> ValueId {
        let n = shape.iter().product();
        self.inputs.push(Tensor::F32(self.rng.vec(n, scale)));
        self.g.input(name, shape, DType::F32)
    }

    fn input_u32(&mut self, name: &str, data: Vec<u32>) -> ValueId {
        let n = data.len();
        self.inputs.push(Tensor::U32(data));
        self.g.input(name, vec![n], DType::U32)
    }

    fn weight(&mut self, name: &str, shape: Vec<usize>, scale: f32) -> ValueId {
        let n = shape.iter().product();
        let data = self.rng.vec(n, scale);
        self.weight_data(name, shape, data)
    }

    fn weight_data(&mut self, name: &str, shape: Vec<usize>, data: Vec<f32>) -> ValueId {
        self.w.insert(name, Tensor::F32(data));
        self.g.weight(name, shape, DType::F32)
    }

    /// An `[out, in]` Linear weight stored in GGUF `format`.
    fn quant(&mut self, name: &str, out: usize, inp: usize, format: QuantFormat) -> ValueId {
        let data = self.rng.vec(out * inp, 0.1);
        let bytes = crate::model::layers::encode_rows(&data, inp, format).unwrap();
        let mat = crate::model::weights::Mat::from_packed_bytes(format, bytes, out, inp);
        self.w.insert(name, Tensor::Mat(mat));
        self.g.weight(name, vec![out, inp], DType::Quant(format))
    }

    fn op(&mut self, name: &str, op: Op, ins: &[ValueId]) -> ValueId {
        self.g.push(name, op, ins).unwrap()
    }

    fn out(mut self, outs: &[ValueId]) -> Self {
        outs.iter().for_each(|&v| self.g.output(v));
        self
    }
}

/// Lowers, compiles and runs `c` on the ANE; the worst relative RMS error over its outputs
/// against `cpu::run`.
fn ane_error(c: &Case, info: &AneInfo) -> Result<f32, String> {
    let (text, blob, io) = lower(&c.g, &c.w, info)?;
    let mut prog = AneProgram::compile(
        &text,
        &blob,
        io.inputs.len(),
        io.in_bytes(),
        io.outputs.len(),
        io.out_bytes(),
    )?;
    let got = run(&mut prog, &io, &c.g, &c.w, &c.inputs)?;
    let want = cpu_reference(c)?;
    Ok(got
        .iter()
        .zip(&want)
        .map(|(g, w)| rel_rms(&g.to_f32(), &w.to_f32()))
        .fold(0.0, f32::max))
}

/// The CPU executor's result for `c` (the reference every case compares against).
fn cpu_reference(c: &Case) -> Result<Vec<Tensor>, String> {
    let _exact = crate::model::qdot::exact();
    crate::graph::cpu::run(&c.g, &c.w, &mut StateStore::new(), c.inputs.clone())
}

fn linear_case(name: &'static str, format: Option<QuantFormat>, inp: usize) -> Case {
    let mut c = Case::new(name);
    let x = c.input("x", vec![7, inp], 1.0);
    let w = match format {
        Some(f) => c.quant("w", 48, inp, f),
        None => c.weight("w", vec![48, inp], 0.1),
    };
    let b = c.weight("b", vec![48], 0.5);
    let y = c.op("y", Op::Linear, &[x, w, b]);
    c.out(&[y])
}

fn rope_case(name: &'static str, rotary_dim: usize, variant: RopeVariant) -> Case {
    let mut c = Case::new(name);
    let x = c.input("x", vec![5, 64], 1.0);
    let pos = c.input_u32("pos", vec![3, 4, 5, 6, 900]);
    let inv: Vec<f32> = (0..rotary_dim / 2)
        .map(|i| 10000f32.powf(-2.0 * i as f32 / rotary_dim as f32))
        .collect();
    let f = c.weight_data("inv_freq", vec![rotary_dim / 2], inv);
    let op = Op::Rope {
        head_dim: 16,
        rotary_dim,
        variant,
        mscale: 1.0,
    };
    let y = c.op("y", op, &[x, pos, f]);
    c.out(&[y])
}

fn mlp_case(name: &'static str, format: Option<QuantFormat>, geglu: bool, post: bool) -> Case {
    let mut c = Case::new(name);
    let x = c.input("x", vec![5, 64], 1.0);
    let n = c.weight("norm", vec![64], 1.0);
    let w = |c: &mut Case, name: &str, o: usize, i: usize| match format {
        Some(f) => c.quant(name, o, i, f),
        None => c.weight(name, vec![o, i], 0.1),
    };
    let (gate, up, down) = (
        w(&mut c, "gate", 96, 64),
        w(&mut c, "up", 96, 64),
        w(&mut c, "down", 64, 96),
    );
    let mut ins = vec![x, n, gate, up, down];
    if post {
        ins.push(c.weight("post", vec![64], 1.0));
    }
    let op = Op::Megakernel(MegakernelDescriptor {
        kind: MegakernelKind::Mlp {
            hidden_dim: 64,
            intermediate_dim: 96,
            eps: 1e-6,
            is_geglu: geglu,
            has_post_norm: post,
        },
        name: "mlp".into(),
    });
    let y = c.g.add("y", op, &ins, vec![5, 64], DType::F32);
    c.out(&[y])
}

/// A one-row Linear past the 65536 output channels one conv may have (an LM head), and a
/// last-axis `Concat` (drafter steps).
fn wide_case() -> Case {
    let mut c = Case::new("linear 131072 outputs + concat");
    let x = c.input("x", vec![1, 64], 1.0);
    let w = c.quant("w", 131072, 64, QuantFormat::GGUF_Q8_0);
    let y = c.op("y", Op::Linear, &[x, w]);
    let z = c.op("z", Op::Concat { axis: -1 }, &[x, x]);
    c.out(&[y, z])
}

fn cases() -> Vec<Case> {
    use QuantFormat as Q;
    let mut v = vec![
        wide_case(),
        linear_case("linear fp16", None, 64),
        linear_case("linear int4 (Q4_0 blockwise)", Some(Q::GGUF_Q4_0), 64),
        linear_case("linear int8 (Q8_0 blockwise)", Some(Q::GGUF_Q8_0), 64),
        linear_case("linear int8 (Q4_K per-channel)", Some(Q::GGUF_Q4_K_M), 256),
        rope_case("rope half-split", 16, RopeVariant::HalfSplit),
        rope_case("rope interleaved partial", 8, RopeVariant::Interleaved),
        mlp_case("megakernel mlp swiglu", None, false, false),
        mlp_case(
            "megakernel mlp geglu post-norm Q8_0",
            Some(Q::GGUF_Q8_0),
            true,
            true,
        ),
    ];
    {
        let mut c = Case::new("layer_norm + rms_norm");
        let x = c.input("x", vec![9, 64], 3.0);
        let (g, b) = (c.weight("g", vec![64], 1.0), c.weight("b", vec![64], 1.0));
        let ln = c.op("ln", Op::LayerNorm { eps: 1e-5 }, &[x, g, b]);
        let rn = c.op("rn", Op::RmsNorm { eps: 1e-6 }, &[x, g]);
        v.push(c.out(&[ln, rn]));
    }
    {
        let mut c = Case::new("attention causal gqa with mask");
        let q = c.input("q", vec![9, 64], 1.0);
        let k = c.input("k", vec![9, 32], 1.0);
        let vv = c.input("v", vec![9, 32], 1.0);
        let m = c.input("mask", vec![9, 9], 2.0);
        let op = Op::Attention {
            heads: 4,
            kv_heads: 2,
            causal: true,
            window: None,
            soft_cap: None,
            scale: None,
        };
        let y = c.op("y", op, &[q, k, vv, m]);
        v.push(c.out(&[y]));
    }
    {
        let mut c = Case::new("attention window soft-cap");
        let q = c.input("q", vec![6, 32], 2.0);
        let k = c.input("k", vec![10, 32], 2.0);
        let vv = c.input("v", vec![10, 32], 1.0);
        let op = Op::Attention {
            heads: 2,
            kv_heads: 2,
            causal: false,
            window: Some(4),
            soft_cap: Some(5.0),
            scale: Some(0.3),
        };
        let y = c.op("y", op, &[q, k, vv]);
        v.push(c.out(&[y]));
    }
    {
        let mut c = Case::new("swiglu + geglu + softmax");
        let (g, u) = (
            c.input("g", vec![7, 32], 2.0),
            c.input("u", vec![7, 32], 1.0),
        );
        let s = c.op("s", Op::SwiGlu, &[g, u]);
        let e = c.op("e", Op::GeGlu, &[g, u]);
        let x = c.input("x", vec![6, 40], 3.0);
        let s1 = c.op("s1", Op::Softmax { axis: -1 }, &[x]);
        let s0 = c.op("s0", Op::Softmax { axis: 0 }, &[x]);
        v.push(c.out(&[s, e, s1, s0]));
    }
    {
        let mut c = Case::new("conv1d + conv2d");
        let x1 = c.input("x1", vec![16, 20], 1.0);
        let w1 = c.weight("w1", vec![8, 8, 3], 0.3);
        let b1 = c.weight("b1", vec![8], 0.5);
        let op1 = Op::Conv1d {
            stride: 1,
            padding: [1, 1],
            dilation: 1,
            groups: 2,
        };
        let y1 = c.op("y1", op1, &[x1, w1, b1]);
        let x2 = c.input("x2", vec![8, 10, 12], 1.0);
        let w2 = c.weight("w2", vec![16, 8, 3, 3], 0.2);
        let b2 = c.weight("b2", vec![16], 0.5);
        let op2 = Op::Conv2d {
            stride: [2, 2],
            padding: [1, 1, 1, 1],
            dilation: [1, 1],
            groups: 1,
        };
        let y2 = c.op("y2", op2, &[x2, w2, b2]);
        v.push(c.out(&[y1, y2]));
    }
    {
        let mut c = Case::new("upsample + avg_pool");
        let x = c.input("x", vec![4, 6, 8], 1.0);
        let n = c.op(
            "n",
            Op::Upsample {
                scale: 2,
                mode: UpsampleMode::Nearest,
            },
            &[x],
        );
        let b = c.op(
            "b",
            Op::Upsample {
                scale: 2,
                mode: UpsampleMode::Bilinear,
            },
            &[x],
        );
        let p = c.op(
            "p",
            Op::AvgPool2d {
                kernel: [2, 2],
                stride: [2, 2],
            },
            &[x],
        );
        v.push(c.out(&[n, b, p]));
    }
    {
        let mut c = Case::new("mean_pool + l2_normalize");
        let x = c.input("x", vec![7, 32], 4.0);
        let m = c.input_u32("mask", vec![1, 1, 0, 1, 0, 1, 1]);
        let p = c.op("p", Op::MeanPool, &[x, m]);
        let n = c.op("n", Op::L2Normalize, &[p]);
        let x2 = c.input("x2", vec![33], 50.0);
        let n2 = c.op("n2", Op::L2Normalize, &[x2]);
        v.push(c.out(&[p, n, n2]));
    }
    {
        let mut c = Case::new("group_norm + instance_norm");
        let x = c.input("x", vec![8, 6, 5], 2.0);
        let (g, b) = (c.weight("g", vec![8], 1.0), c.weight("b", vec![8], 1.0));
        let gn = c.op(
            "gn",
            Op::GroupNorm {
                groups: 4,
                eps: 1e-5,
            },
            &[x, g, b],
        );
        let x2 = c.input("x2", vec![8, 30], 2.0);
        let inorm = c.op("in", Op::InstanceNorm { eps: 1e-5 }, &[x2, g, b]);
        v.push(c.out(&[gn, inorm]));
    }
    {
        let mut c = Case::new("unary");
        let x = c.input("x", vec![6, 32], 2.0);
        let half = c.weight_data("half", vec![1], vec![0.5]);
        let a = c.op(
            "a",
            Op::Unary {
                kind: UnaryKind::Abs,
            },
            &[x],
        );
        let pos = c.op(
            "pos",
            Op::Binary {
                kind: BinaryKind::Add,
            },
            &[a, half],
        );
        let mut outs = vec![a, pos];
        for (i, kind) in [
            UnaryKind::Exp,
            UnaryKind::Sigmoid,
            UnaryKind::Tanh,
            UnaryKind::Softplus,
            UnaryKind::Neg,
            UnaryKind::Relu,
            UnaryKind::Relu2,
            UnaryKind::Silu,
            UnaryKind::GeluErf,
            UnaryKind::GeluTanh,
            UnaryKind::Scale(1.5),
            UnaryKind::Clamp(-0.3, 0.4),
            UnaryKind::SoftCap(0.5),
        ]
        .into_iter()
        .enumerate()
        {
            let name = format!("u{i}");
            outs.push(c.g.push(name, Op::Unary { kind }, &[x]).unwrap());
        }
        outs.push(c.op(
            "sqrt",
            Op::Unary {
                kind: UnaryKind::Sqrt,
            },
            &[pos],
        ));
        outs.push(c.op(
            "rsqrt",
            Op::Unary {
                kind: UnaryKind::Rsqrt,
            },
            &[pos],
        ));
        v.push(c.out(&outs));
    }
    {
        let mut c = Case::new("binary broadcasts");
        let x = c.input("x", vec![6, 32], 1.0);
        let y = c.input("y", vec![6, 32], 1.0);
        let col = c.input("col", vec![6, 1], 1.0);
        let b = c.weight("b", vec![32], 1.0);
        let x3 = c.input("x3", vec![2, 3, 8], 1.0);
        let w31 = c.weight("w31", vec![3, 1], 1.0);
        let bin = |c: &mut Case, n: &str, kind, a, b| c.op(n, Op::Binary { kind }, &[a, b]);
        let add = bin(&mut c, "add", BinaryKind::Add, x, y);
        let sub = bin(&mut c, "sub", BinaryKind::Sub, x, b);
        let mul = bin(&mut c, "mul", BinaryKind::Mul, x, col);
        let mx = bin(&mut c, "max", BinaryKind::Max, x, y);
        let mn = bin(&mut c, "min", BinaryKind::Min, b, x);
        let ab = c.op(
            "ab",
            Op::Unary {
                kind: UnaryKind::Abs,
            },
            &[y],
        );
        let one = c.weight_data("one", vec![1], vec![1.0]);
        let den = bin(&mut c, "den", BinaryKind::Add, ab, one);
        let div = bin(&mut c, "div", BinaryKind::Div, x, den);
        let r3 = bin(&mut c, "r3", BinaryKind::Mul, x3, w31);
        v.push(c.out(&[add, sub, mul, mx, mn, div, r3]));
    }
    {
        let mut c = Case::new("reduce + shape ops + matmul");
        let x = c.input("x", vec![4, 6, 8], 1.0);
        let red = |c: &mut Case, n: &str, kind, axes: Vec<isize>, keep_dims| {
            c.op(
                n,
                Op::Reduce {
                    kind,
                    axes,
                    keep_dims,
                },
                &[x],
            )
        };
        let r0 = red(&mut c, "r0", ReduceKind::Sum, vec![2], false);
        let r1 = red(&mut c, "r1", ReduceKind::Mean, vec![0, 2], false);
        let r2 = red(&mut c, "r2", ReduceKind::Max, vec![1], true);
        let r3 = red(&mut c, "r3", ReduceKind::Min, vec![-1], false);
        let t = c.input("t", vec![6, 32], 1.0);
        let tr = c.op("tr", Op::Transpose { perm: vec![1, 0] }, &[t]);
        let rs = c.op(
            "rs",
            Op::Reshape {
                shape: Some(vec![3, 64]),
            },
            &[t],
        );
        let sl = c.op(
            "sl",
            Op::Slice {
                axis: -1,
                start: 4,
                end: 20,
            },
            &[t],
        );
        let s0 = c.op(
            "s0",
            Op::Slice {
                axis: 0,
                start: 1,
                end: 4,
            },
            &[t],
        );
        let ti = c.op("ti", Op::Tile { reps: vec![2, 1] }, &[t]);
        let pd = c.op(
            "pd",
            Op::Pad {
                padding: vec![1, 2],
                value: 0.5,
            },
            &[t],
        );
        let un = c.op("un", Op::Unsqueeze { axes: vec![0] }, &[t]);
        let sq = c.op("sq", Op::Squeeze { axes: vec![0] }, &[un]);
        let x4 = c.input("x4", vec![2, 3, 4, 5], 1.0);
        let t4 = c.op(
            "t4",
            Op::Transpose {
                perm: vec![0, 2, 1, 3],
            },
            &[x4],
        );
        let a = c.input("a", vec![2, 5, 16], 1.0);
        let b = c.input("b", vec![2, 16, 7], 1.0);
        let b2 = c.input("b2", vec![2, 7, 16], 1.0);
        let mm = c.op(
            "mm",
            Op::MatMul {
                transpose_a: false,
                transpose_b: false,
            },
            &[a, b],
        );
        let mt = c.op(
            "mt",
            Op::MatMul {
                transpose_a: false,
                transpose_b: true,
            },
            &[a, b2],
        );
        v.push(c.out(&[r0, r1, r2, r3, tr, rs, sl, s0, ti, pd, sq, t4, mm, mt]));
    }
    {
        let mut c = Case::new("per-head rms_norm + layer_norm");
        let x = c.input("x", vec![5, 64], 3.0);
        let w = c.weight("w", vec![64, 64], 0.2);
        let h = c.op("h", Op::Linear, &[x, w]);
        let (g, b) = (c.weight("g", vec![16], 1.0), c.weight("b", vec![16], 1.0));
        let rn = c.op("rn", Op::RmsNorm { eps: 1e-6 }, &[h, g]);
        let ln = c.op("ln", Op::LayerNorm { eps: 1e-5 }, &[x, g, b]);
        let r0 = c.op("r0", Op::RmsNorm { eps: 1e-6 }, &[x]);
        v.push(c.out(&[rn, ln, r0]));
    }
    {
        let mut c = Case::new("binary on token-form operands");
        let x = c.input("x", vec![6, 32], 1.0);
        let w = c.weight("w", vec![32, 32], 0.2);
        let t = c.op("t", Op::Linear, &[x, w]);
        let x3 = c.input("x3", vec![2, 3, 32], 1.0);
        let t3 = c.op("t3", Op::Linear, &[x3, w]);
        let col = c.input("col", vec![6, 1], 1.0);
        let row = c.input("row", vec![1, 32], 1.0);
        let bias = c.weight("bias", vec![32], 1.0);
        let k = c.weight_data("k", vec![1], vec![1.5]);
        let col3 = c.input("col3", vec![2, 3, 1], 1.0);
        let mid = c.input("mid", vec![3, 32], 1.0);
        let bin = |c: &mut Case, n: &str, kind, a, b| c.op(n, Op::Binary { kind }, &[a, b]);
        let outs = vec![
            bin(&mut c, "tcol", BinaryKind::Mul, t, col),
            bin(&mut c, "colt", BinaryKind::Sub, col, t),
            bin(&mut c, "trow", BinaryKind::Add, t, row),
            bin(&mut c, "tbias", BinaryKind::Add, bias, t),
            bin(&mut c, "tk", BinaryKind::Mul, t, k),
            bin(&mut c, "tt", BinaryKind::Max, t, t),
            bin(&mut c, "t3b", BinaryKind::Mul, t3, bias),
            bin(&mut c, "t3c", BinaryKind::Mul, t3, col3),
            bin(&mut c, "t3m", BinaryKind::Sub, t3, mid),
            bin(&mut c, "t3x", BinaryKind::Add, t3, x3),
        ];
        // Non-numpy forms the CPU executor accepts: row scale and flat tiling.
        let rs_x = c.input("rs_x", vec![6, 4, 8], 1.0);
        let rs = c.g.add(
            "rs",
            Op::Binary {
                kind: BinaryKind::Mul,
            },
            &[rs_x, col],
            vec![6, 4, 8],
            DType::F32,
        );
        let rs2 = c.g.add(
            "rs2",
            Op::Binary {
                kind: BinaryKind::Mul,
            },
            &[col, rs_x],
            vec![6, 4, 8],
            DType::F32,
        );
        let ta = c.input("ta", vec![24], 1.0);
        let tb = c.input("tb", vec![6], 1.0);
        let tl = c.g.add(
            "tl",
            Op::Binary {
                kind: BinaryKind::Sub,
            },
            &[ta, tb],
            vec![24],
            DType::F32,
        );
        let tc = c.input("tc", vec![2, 12], 1.0);
        let td = c.input("td", vec![3, 4], 1.0);
        let tm = c.g.add(
            "tm",
            Op::Binary {
                kind: BinaryKind::Div,
            },
            &[td, tc],
            vec![2, 12],
            DType::F32,
        );
        let mut outs = outs;
        outs.extend([rs, rs2, tl, tm]);
        v.push(c.out(&outs));
    }
    {
        let mut c = Case::new("reduce keep_dims=false consumers + pad + batched matmul");
        let x = c.input("x", vec![4, 6, 8], 1.0);
        let r = c.op(
            "r",
            Op::Reduce {
                kind: ReduceKind::Sum,
                axes: vec![1],
                keep_dims: false,
            },
            &[x],
        );
        let rt = c.op("rt", Op::Transpose { perm: vec![1, 0] }, &[r]);
        let u = c.input("u", vec![4, 8], 1.0);
        let sg = c.op("sg", Op::SwiGlu, &[r, u]);
        let un = c.op(
            "un",
            Op::Unary {
                kind: UnaryKind::Tanh,
            },
            &[r],
        );
        let ad = c.op(
            "ad",
            Op::Binary {
                kind: BinaryKind::Add,
            },
            &[un, u],
        );
        let p = c.input("p", vec![3, 5, 6], 1.0);
        let pd = c.op(
            "pd",
            Op::Pad {
                padding: vec![1, 0, 2, 1],
                value: -0.5,
            },
            &[p],
        );
        let a = c.input("a", vec![2, 5, 16], 1.0);
        let b = c.input("b", vec![16, 7], 1.0);
        let mm = c.op(
            "mm",
            Op::MatMul {
                transpose_a: false,
                transpose_b: false,
            },
            &[a, b],
        );
        v.push(c.out(&[r, rt, sg, ad, pd, mm]));
    }
    v
}

/// Nodes [`lower`] refuses, each built so `cpu::run` could run it: `supports_op` must refuse
/// them too.
fn rejected() -> Vec<Case> {
    let mut v = Vec::new();
    {
        let mut c = Case::new("batched conv2d");
        let x = c.input("x", vec![2, 4, 6, 6], 1.0);
        let w = c.weight("w", vec![4, 4, 3, 3], 0.2);
        let op = Op::Conv2d {
            stride: [1, 1],
            padding: [1, 1, 1, 1],
            dilation: [1, 1],
            groups: 1,
        };
        let y = c.g.add("y", op, &[x, w], vec![2, 4, 6, 6], DType::F32);
        v.push(c.out(&[y]));
    }
    {
        let mut c = Case::new("3-D upsample");
        let x = c.input("x", vec![2, 3, 4, 5], 1.0);
        let y = c.g.add(
            "y",
            Op::Upsample {
                scale: 2,
                mode: UpsampleMode::Nearest,
            },
            &[x],
            vec![2, 6, 8, 10],
            DType::F32,
        );
        v.push(c.out(&[y]));
    }
    {
        let mut c = Case::new("matmul batch mismatch");
        let a = c.input("a", vec![2, 5, 16], 1.0);
        let b = c.input("b", vec![3, 16, 7], 1.0);
        let y = c.g.add(
            "y",
            Op::MatMul {
                transpose_a: false,
                transpose_b: false,
            },
            &[a, b],
            vec![2, 5, 7],
            DType::F32,
        );
        v.push(c.out(&[y]));
    }
    {
        let mut c = Case::new("attention row without keys");
        let q = c.input("q", vec![9, 32], 1.0);
        let k = c.input("k", vec![3, 32], 1.0);
        let vv = c.input("v", vec![3, 32], 1.0);
        let op = Op::Attention {
            heads: 2,
            kv_heads: 2,
            causal: false,
            window: Some(2),
            soft_cap: None,
            scale: None,
        };
        let y = c.op("y", op, &[q, k, vv]);
        v.push(c.out(&[y]));
    }
    {
        let mut c = Case::new("ln");
        let x = c.input("x", vec![4, 8], 1.0);
        let y = c.op(
            "y",
            Op::Unary {
                kind: UnaryKind::Ln,
            },
            &[x],
        );
        v.push(c.out(&[y]));
    }
    {
        let mut c = Case::new("softmax axis out of range");
        let x = c.input("x", vec![4, 8], 1.0);
        let y =
            c.g.add("y", Op::Softmax { axis: 2 }, &[x], vec![4, 8], DType::F32);
        v.push(c.out(&[y]));
    }
    v
}

/// Device-bound: probes this host's ANE and reports what it supports.
#[test]
#[ignore]
fn probe_matches_cpu() {
    let info = measure().unwrap();
    eprintln!("ANE probe: {info:#?}");
    let fp16 = info.fp16.clone().unwrap();
    assert!(fp16 <= PROBE_TOL, "fp16 relative RMS {fp16}");
    for (form, r) in [("int8", &info.int8), ("int4", &info.int4)] {
        if let Ok(e) = r {
            assert!(*e <= PROBE_TOL, "{form} relative RMS {e}");
        }
    }
}

/// Device-bound: one small graph per op kind (Linear in every weight form) on the ANE against
/// the CPU executor.
#[test]
#[ignore]
fn lowering_matches_cpu_per_op() {
    let info = probe().unwrap().clone();
    let per_channel = AneInfo {
        int4: Err("forced per-channel int8".into()),
        ..info.clone()
    };
    let mut cases: Vec<(Case, AneInfo)> = cases().into_iter().map(|c| (c, info.clone())).collect();
    cases.push((
        linear_case(
            "linear int8 (Q8_0 per-channel)",
            Some(QuantFormat::GGUF_Q8_0),
            64,
        ),
        per_channel,
    ));
    let mut failed = Vec::new();
    for c in rejected() {
        let node = &c.g.nodes[c.g.nodes.len() - 1];
        if supports_op(&c.g, node) || lower(&c.g, &c.w, &info).is_ok() {
            failed.push(format!("{}: accepted", c.name));
        }
    }
    for (c, _) in &cases {
        // Concat lowers in drafter steps only; placement keeps it on Metal.
        for node in &c.g.nodes {
            if !supports_op(&c.g, node) && !matches!(node.op, Op::Concat { .. }) {
                failed.push(format!(
                    "{}: supports_op rejects '{}': {:?}",
                    c.name,
                    node.name,
                    check(&c.g, node)
                ));
            }
        }
    }
    for (c, info) in &cases {
        match ane_error(c, info) {
            Ok(e) if e <= 2e-2 => eprintln!("{:<40} relative RMS {e:.2e}", c.name),
            Ok(e) => failed.push(format!("{}: relative RMS {e:.2e}", c.name)),
            Err(e) => failed.push(format!("{}: {e}", c.name)),
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// A two-layer decode instance (full and 3-key windowed cached attention) run as a drafter step
/// on the ANE ([`draft::lower_drafter_step`]) matches the CPU's cached decode position by
/// position, and after a rewind ([`draft::Ring::truncate`]) continues like a fresh decode of
/// the new tokens.
#[test]
#[ignore]
fn drafter_step_matches_cached_decode() {
    let info = probe().expect("ANE probe");
    let (vocab, d, hd, heads, kvh) = (40, 64, 16, 4, 2);
    let f = DType::F32;
    let mut c = Case::new("drafter step");
    let ids = c.g.input("token_ids", vec![1], DType::U32);
    let pos = c.g.input("positions", vec![1], DType::U32);
    let table = c.weight("embed", vec![vocab, d], 1.0);
    let mut x = c.op("emb", Op::Embedding, &[table, ids]);
    let inv = (0..hd / 2).map(|i| 1e4f32.powf(-2.0 * i as f32 / hd as f32)).collect();
    let inv = c.weight_data("inv", vec![hd / 2], inv);
    for (l, window) in [None, Some(3)].into_iter().enumerate() {
        let lin = |c: &mut Case, n: &str, x: ValueId, o: usize| {
            let w = c.weight(&format!("{n}{l}"), vec![o, d], 0.2);
            c.op(&format!("{n}{l}.y"), Op::Linear, &[x, w])
        };
        let rope = Op::Rope {
            head_dim: hd,
            rotary_dim: hd,
            variant: RopeVariant::HalfSplit,
            mscale: 1.0,
        };
        let q = lin(&mut c, "q", x, heads * hd);
        let k = lin(&mut c, "k", x, kvh * hd);
        let v = lin(&mut c, "v", x, kvh * hd);
        let q = c.op(&format!("qr{l}"), rope.clone(), &[q, pos, inv]);
        let k = c.op(&format!("kr{l}"), rope, &[k, pos, inv]);
        let ks = c.g.state(format!("k{l}"), vec![16, kvh * hd], f, false);
        let vs = c.g.state(format!("v{l}"), vec![16, kvh * hd], f, false);
        let att = Op::CachedAttention {
            heads,
            kv_heads: kvh,
            head_dim: hd,
            v_head_dim: hd,
            scale: 0.25,
            causal: true,
            window,
            soft_cap: None,
            write: true,
        };
        let a = c.op(&format!("att{l}"), att, &[q, k, v, ks, vs, pos]);
        let wo = c.weight(&format!("o{l}"), vec![d, heads * hd], 0.2);
        let o = c.op(&format!("o{l}.y"), Op::Linear, &[a, wo]);
        x = c.op(&format!("res{l}"), Op::Binary { kind: BinaryKind::Add }, &[x, o]);
    }
    let head = c.weight("head", vec![vocab, d], 0.2);
    let logits = c.op("logits", Op::Linear, &[x, head]);
    c.g.output(logits);

    let sg = draft::lower_drafter_step(&c.g).unwrap();
    assert_eq!(sg.embed, ("embed".to_string(), d));
    let step = draft::DraftStep::compile(sg, &c.w, info).unwrap();
    let mut lane = draft::AneDraft::new(vec![step]);
    let mut ring = draft::Ring::new();
    assert!(lane.claim(&mut ring));
    let rows = c.w.get("embed").unwrap().to_f32();
    let cpu = |toks: &[u32]| -> Vec<Vec<f32>> {
        let _exact = crate::model::qdot::exact();
        let mut st = StateStore::new();
        (0..toks.len())
            .map(|p| {
                let ins = vec![Tensor::U32(vec![toks[p]]), Tensor::U32(vec![p as u32])];
                crate::graph::cpu::run(&c.g, &c.w, &mut st, ins).unwrap()[0].to_f32()
            })
            .collect()
    };
    let check = |lane: &mut draft::AneDraft, ring: &mut draft::Ring, toks: &[u32], from: usize| {
        let want = cpu(toks);
        for p in from..toks.len() {
            let t = toks[p] as usize;
            let got = lane.run(0, ring, p, &[("embed", &rows[t * d..][..d])]).unwrap();
            ring.ids.push(toks[p]);
            let e = rel_rms(&got[0], &want[p]);
            eprintln!("position {p}: relative RMS {e:.2e}");
            assert!(e <= 2e-2, "position {p}: relative RMS {e:.2e}");
        }
    };
    let a = [3u32, 17, 5, 9, 22, 1, 30, 8];
    check(&mut lane, &mut ring, &a, 0);
    ring.truncate(5);
    check(&mut lane, &mut ring, &[&a[..5], &[11, 4, 2]].concat(), 5);
    // A second sequence waits for the windows until the first lets go.
    let mut other = draft::Ring::new();
    assert!(!lane.claim(&mut other));
    drop(ring);
    assert!(lane.claim(&mut other) && other.ids.is_empty());
}

