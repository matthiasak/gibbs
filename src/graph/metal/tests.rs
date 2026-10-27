use super::*;
use crate::graph::cpu;
use crate::graph::StateStore;
use std::sync::LazyLock;

/// One runtime (one compile and host-profile run) shared by the tests.
static RT: LazyLock<Arc<Runtime>> =
    LazyLock::new(|| Runtime::new(tempfile::tempdir().unwrap().path()).unwrap());

fn rt() -> Arc<Runtime> {
    RT.clone()
}

fn kinds() -> Vec<GemmKernel> {
    let mut k = vec![GemmKernel::Simdgroup, GemmKernel::Rows];
    if rt().profile().mpp.is_ok() {
        k.push(GemmKernel::Mpp);
    }
    k
}

fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Runs `g` on the CPU executor (`nn.rs` kernels) and on Metal; returns the first outputs.
fn both(
    g: &Graph,
    w: &Weights,
    inputs: Vec<Tensor>,
    gemm: Option<GemmKernel>,
) -> (Vec<f32>, Vec<f32>) {
    let want = cpu::run_stateless(g, w, inputs.clone())
        .unwrap()
        .remove(0)
        .into_f32()
        .unwrap();
    let mut s = Session::new(rt(), w).unwrap();
    if let Some(k) = gemm {
        s.force_gemm(k).unwrap();
    }
    let got = s.run(g, inputs).unwrap().remove(0).into_f32().unwrap();
    assert_eq!(got.len(), want.len());
    (want, got)
}

/// max |a - b| relative to max(1, max |a|).
fn rel_err(want: &[f32], got: &[f32]) -> f32 {
    let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    max_abs_diff(want, got) / scale
}

#[test]
fn linear_and_epilogues_match_cpu_on_every_gemm_kernel() {
    let f = DType::F32;
    for kind in kinds() {
        for (m, n, k) in [
            (1, 1, 1),
            (17, 50, 37),
            (70, 33, 129),
            (5, 384, 384),
            (130, 96, 64),
        ] {
            // 0: no bias, 1: bias, 2: fused GELU, 3: fused residual, 4: two readers (standalone GELU + Add).
            for case in 0..5 {
                let mut g = Graph::new();
                let mut w = Weights::new();
                let x = g.input("x", vec![m, k], f);
                let r = g.input("r", vec![m, n], f);
                let wv = g.weight("w", vec![n, k], f);
                let bv = g.weight("b", vec![n], f);
                let ins = if case == 0 {
                    vec![x, wv]
                } else {
                    vec![x, wv, bv]
                };
                let lin = g.add("lin", Op::Linear, &ins, vec![m, n], f);
                let out = match case {
                    0 | 1 => lin,
                    2 => g.add(
                        "gelu",
                        Op::Unary {
                            kind: crate::graph::UnaryKind::GeluErf,
                        },
                        &[lin],
                        vec![m, n],
                        f,
                    ),
                    3 => g.add(
                        "res",
                        Op::Binary {
                            kind: crate::graph::BinaryKind::Add,
                        },
                        &[r, lin],
                        vec![m, n],
                        f,
                    ),
                    _ => {
                        let a = g.add(
                            "gelu",
                            Op::Unary {
                                kind: crate::graph::UnaryKind::GeluErf,
                            },
                            &[lin],
                            vec![m, n],
                            f,
                        );
                        g.add(
                            "sum",
                            Op::Binary {
                                kind: crate::graph::BinaryKind::Add,
                            },
                            &[a, lin],
                            vec![m, n],
                            f,
                        )
                    }
                };
                g.output(out);
                w.insert("w", Tensor::F32(rand(n * k, 1)));
                w.insert("b", Tensor::F32(rand(n, 2)));
                let inputs = vec![Tensor::F32(rand(m * k, 3)), Tensor::F32(rand(m * n, 4))];
                let (want, got) = both(&g, &w, inputs, Some(kind));
                let e = rel_err(&want, &got);
                assert!(e < 1e-5, "{kind:?} {m}x{n}x{k} case {case}: rel err {e:e}");
            }
        }
    }
}

#[test]
fn test_graph_fused_epilogue_matches_cpu() {
    let f = DType::F32;
    let bf = DType::BF16;
    let (m, n, k) = (1, 64, 32);
    let bf_weights: Vec<u16> = (0..n * k)
        .map(|i| crate::model::f32_to_bf16(0.01 * ((i % 7) as f32 - 3.0)))
        .collect();
    // 1. Linear -> Silu
    {
        let mut g = Graph::new();
        let mut w = Weights::new();
        let x = g.input("x", vec![m, k], f);
        let wv = g.weight("w", vec![n, k], bf);
        let lin = g.add("lin", Op::Linear, &[x, wv], vec![m, n], f);
        let silu = g.add(
            "silu",
            Op::Unary {
                kind: crate::graph::UnaryKind::Silu,
            },
            &[lin],
            vec![m, n],
            f,
        );
        g.output(silu);

        w.insert("w", Tensor::BF16(bf_weights.clone()));
        let inputs = vec![Tensor::F32(vec![0.1f32; m * k])];
        let (want, got) = both(&g, &w, inputs, None);
        assert!(rel_err(&want, &got) < 1e-4, "Fused Silu parity failed");
    }
    // 2. Linear -> Gelu
    {
        let mut g = Graph::new();
        let mut w = Weights::new();
        let x = g.input("x", vec![m, k], f);
        let wv = g.weight("w", vec![n, k], bf);
        let lin = g.add("lin", Op::Linear, &[x, wv], vec![m, n], f);
        let gelu = g.add(
            "gelu",
            Op::Unary {
                kind: crate::graph::UnaryKind::GeluErf,
            },
            &[lin],
            vec![m, n],
            f,
        );
        g.output(gelu);

        w.insert("w", Tensor::BF16(bf_weights.clone()));
        let inputs = vec![Tensor::F32(vec![0.1f32; m * k])];
        let (want, got) = both(&g, &w, inputs, None);
        assert!(rel_err(&want, &got) < 1e-4, "Fused Gelu parity failed");
    }
    // 3. Linear -> SwiGlu
    {
        let mut g = Graph::new();
        let mut w = Weights::new();
        let x = g.input("x", vec![m, k], f);
        let up_val = g.input("up", vec![m, n], f);
        let wv = g.weight("w", vec![n, k], bf);
        let lin = g.add("lin", Op::Linear, &[x, wv], vec![m, n], f);
        let swiglu = g.add("swiglu", Op::SwiGlu, &[lin, up_val], vec![m, n], f);
        g.output(swiglu);

        w.insert("w", Tensor::BF16(bf_weights));
        let inputs = vec![
            Tensor::F32(vec![0.1f32; m * k]),
            Tensor::F32(vec![0.5f32; m * n]),
        ];
        let (want, got) = both(&g, &w, inputs, None);
        assert!(rel_err(&want, &got) < 1e-4, "Fused SwiGlu parity failed");
    }
}

#[test]
fn test_generic_buffer_lifetime_arena_zero_reallocations() {
    let f = DType::F32;
    let (m, n, k) = (1, 64, 64);
    let mut g = Graph::new();
    let mut w = Weights::new();
    let x = g.input("x", vec![m, k], f);
    let w1 = g.weight("w1", vec![n, k], f);
    let w2 = g.weight("w2", vec![k, n], f);
    let l1 = g.add("l1", Op::Linear, &[x, w1], vec![m, n], f);
    let a1 = g.add(
        "a1",
        Op::Unary {
            kind: crate::graph::UnaryKind::GeluErf,
        },
        &[l1],
        vec![m, n],
        f,
    );
    let l2 = g.add("l2", Op::Linear, &[a1, w2], vec![m, k], f);
    g.output(l2);

    w.insert("w1", Tensor::F32(vec![0.02f32; n * k]));
    w.insert("w2", Tensor::F32(vec![0.02f32; k * n]));

    let s = Session::new(rt(), &w).unwrap();
    let plan = s.prepare_plan(&g).unwrap();
    let initial_ptr = s
        .arena_buffer_ptr()
        .expect("arena buffer pre-allocated during prepare_plan");

    for step in 0..100 {
        let inputs = vec![Tensor::F32(vec![0.05f32 * (step as f32); m * k])];
        let out = s
            .run_prepared(&g, &plan, None, inputs)
            .expect("decode step succeeds");
        assert_eq!(out.len(), 1);
        let cur_ptr = s
            .arena_buffer_ptr()
            .expect("arena buffer remains allocated");
        assert_eq!(
            cur_ptr, initial_ptr,
            "Arena buffer must never be reallocated across decode steps (step {step})"
        );
    }
}

#[test]
fn layer_norm_and_gelu_match_cpu() {
    let f = DType::F32;
    for (rows, d) in [(7, 384), (3, 33), (1, 1)] {
        let mut g = Graph::new();
        let x = g.input("x", vec![rows, d], f);
        let (gv, bv) = (g.weight("g", vec![d], f), g.weight("b", vec![d], f));
        let ln = g.add(
            "ln",
            Op::LayerNorm { eps: 1e-12 },
            &[x, gv, bv],
            vec![rows, d],
            f,
        );
        g.output(ln);
        let mut w = Weights::new();
        w.insert("g", Tensor::F32(rand(d, 5)));
        w.insert("b", Tensor::F32(rand(d, 6)));
        let xs: Vec<f32> = rand(rows * d, 7).iter().map(|v| v * 10.0 + 3.0).collect();
        let (want, got) = both(&g, &w, vec![Tensor::F32(xs)], None);
        assert!(rel_err(&want, &got) < 1e-5, "{rows}x{d}");
    }
    // GELU over [-8, 8]: the f32 erf fit tracks the CPU's f64 one.
    let mut g = Graph::new();
    let x = g.input("x", vec![1601], f);
    let y = g.add(
        "gelu",
        Op::Unary {
            kind: crate::graph::UnaryKind::GeluErf,
        },
        &[x],
        vec![1601],
        f,
    );
    g.output(y);
    let xs: Vec<f32> = (0..1601).map(|i| i as f32 / 100.0 - 8.0).collect();
    let (want, got) = both(&g, &Weights::new(), vec![Tensor::F32(xs)], None);
    let e = max_abs_diff(&want, &got);
    assert!(e < 2e-6, "gelu max abs err {e:e}");
}

#[test]
fn attention_matches_cpu() {
    let f = DType::F32;
    for (nq, nk, heads, hd, causal) in [
        (37, 37, 3, 24, false),
        (37, 37, 3, 24, true),
        (5, 5, 12, 32, false),
        (2, 9, 2, 64, false),
        (1, 1, 1, 256, false),
    ] {
        let w = heads * hd;
        let mut g = Graph::new();
        let q = g.input("q", vec![nq, w], f);
        let k = g.input("k", vec![nk, w], f);
        let v = g.input("v", vec![nk, w], f);
        let op = Op::Attention {
            heads,
            kv_heads: heads,
            causal,
            window: None,
            soft_cap: None,
            scale: None,
        };
        let a = g.add("attn", op, &[q, k, v], vec![nq, w], f);
        g.output(a);
        let inputs = vec![
            Tensor::F32(rand(nq * w, 8)),
            Tensor::F32(rand(nk * w, 9)),
            Tensor::F32(rand(nk * w, 10)),
        ];
        let (want, got) = both(&g, &Weights::new(), inputs, None);
        let e = rel_err(&want, &got);
        assert!(
            e < 1e-5,
            "nq {nq} nk {nk} heads {heads} hd {hd} causal {causal}: {e:e}"
        );
    }
}

#[test]
fn embedding_add_pool_normalize_match_cpu() {
    let (f, u) = (DType::F32, DType::U32);
    for mask in [vec![1, 1, 0, 1, 0], vec![0, 0, 0, 0, 0]] {
        let mut g = Graph::new();
        let ids = g.input("ids", vec![5], u);
        let pos = g.input("pos", vec![5], u);
        let mv = g.input("mask", vec![5], u);
        let (t1, t2) = (
            g.weight("t1", vec![10, 6], f),
            g.weight("t2", vec![5, 6], f),
        );
        let e1 = g.add("e1", Op::Embedding, &[t1, ids], vec![5, 6], f);
        let e2 = g.add("e2", Op::Embedding, &[t2, pos], vec![5, 6], f);
        let s = g.add(
            "sum",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[e1, e2],
            vec![5, 6],
            f,
        );
        let p = g.add("pool", Op::MeanPool, &[s, mv], vec![6], f);
        let o = g.add("l2", Op::L2Normalize, &[p], vec![6], f);
        g.output(o);
        let mut w = Weights::new();
        w.insert("t1", Tensor::F32(rand(60, 11)));
        w.insert("t2", Tensor::F32(rand(30, 12)));
        let inputs = vec![
            Tensor::U32(vec![9, 0, 3, 3, 7]),
            Tensor::U32((0..5).collect()),
            Tensor::U32(mask),
        ];
        let (want, got) = both(&g, &w, inputs, None);
        assert!(rel_err(&want, &got) < 1e-6);
    }
}

/// One Gemma-4-E2B sliding-layer decode block (f16 dense weights, hidden 1536, 8 heads, 1 KV head,
/// head_dim 256, intermediate 6144, 128-key cache): the fixture `scripts/fusion_vs_xla.py` compares
/// with XLA. Prints `FUSION_BLOCK {"us": <median GPU us>, "dispatches": <n>}`; run once per
/// `GIBBS_FUSION` value (`cargo test --release --lib fusion_block_bench -- --ignored --nocapture`).
#[test]
#[ignore = "measurement"]
fn fusion_block_bench() {
    let (h, nh, hd, ff, t) = (1536usize, 8usize, 256usize, 6144usize, 128usize);
    let f = DType::F32;
    let (mut g, mut w) = (Graph::new(), Weights::new());
    let mut dense = |g: &mut Graph, name: &str, rows: usize, cols: usize, seed: u64| {
        let scale = 1.0 / (cols as f32).sqrt();
        let data: Vec<u16> = rand(rows * cols, seed)
            .into_iter()
            .map(|v| crate::model::f32_to_f16(v * scale))
            .collect();
        w.insert(name.to_string(), Tensor::F16(data));
        g.weight(name, vec![rows, cols], DType::F16)
    };
    let x0 = g.input("x", vec![1, h], f);
    let pos = g.input("pos", vec![1], DType::U32);
    let ks = g.state("k_state", vec![t, hd], f, false);
    let vs = g.state("v_state", vec![t, hd], f, false);
    let (wq, wk, wv) = (
        dense(&mut g, "wq", nh * hd, h, 1),
        dense(&mut g, "wk", hd, h, 2),
        dense(&mut g, "wv", hd, h, 3),
    );
    let (wo, wg, wu, wd) = (
        dense(&mut g, "wo", h, nh * hd, 4),
        dense(&mut g, "wg", ff, h, 5),
        dense(&mut g, "wu", ff, h, 6),
        dense(&mut g, "wd", h, ff, 7),
    );
    let mut ones = |g: &mut Graph, name: &str, n: usize| {
        w.insert(name.to_string(), Tensor::F32(vec![1.0; n]));
        g.weight(name, vec![n], f)
    };
    let (g1, g2) = (ones(&mut g, "g1", h), ones(&mut g, "g2", h));
    let (qn, kn) = (ones(&mut g, "qn", hd), ones(&mut g, "kn", hd));
    w.insert(
        "inv_freq".to_string(),
        Tensor::F32(
            (0..hd / 2)
                .map(|i| 10000f32.powf(-(i as f32) / (hd / 2) as f32))
                .collect(),
        ),
    );
    let inv = g.weight("inv_freq", vec![hd / 2], f);
    let eps = Op::RmsNorm { eps: 1e-6 };
    let rope = Op::Rope {
        head_dim: hd,
        rotary_dim: hd,
        variant: crate::graph::RopeVariant::HalfSplit,
        mscale: 1.0,
    };
    let a = g.add("a", eps.clone(), &[x0, g1], vec![1, h], f);
    let q = g.add("q", Op::Linear, &[a, wq], vec![1, nh * hd], f);
    let k = g.add("k", Op::Linear, &[a, wk], vec![1, hd], f);
    let v = g.add("v", Op::Linear, &[a, wv], vec![1, hd], f);
    let q = g.add("qn", eps.clone(), &[q, qn], vec![1, nh * hd], f);
    let k = g.add("kn", eps.clone(), &[k, kn], vec![1, hd], f);
    let q = g.add("qr", rope.clone(), &[q, pos, inv], vec![1, nh * hd], f);
    let k = g.add("kr", rope, &[k, pos, inv], vec![1, hd], f);
    let att = Op::CachedAttention {
        heads: nh,
        kv_heads: 1,
        head_dim: hd,
        v_head_dim: hd,
        scale: 1.0 / (hd as f32).sqrt(),
        causal: true,
        window: None,
        soft_cap: None,
        write: true,
    };
    let o = g.add("attn", att, &[q, k, v, ks, vs, pos], vec![1, nh * hd], f);
    let o = g.add("o", Op::Linear, &[o, wo], vec![1, h], f);
    let x1 = g.add(
        "res1",
        Op::Binary {
            kind: crate::graph::BinaryKind::Add,
        },
        &[x0, o],
        vec![1, h],
        f,
    );
    let b = g.add("b", eps, &[x1, g2], vec![1, h], f);
    let gate = g.add("gate", Op::Linear, &[b, wg], vec![1, ff], f);
    let up = g.add("up", Op::Linear, &[b, wu], vec![1, ff], f);
    let act = g.add("act", Op::GeGlu, &[gate, up], vec![1, ff], f);
    let d = g.add("down", Op::Linear, &[act, wd], vec![1, h], f);
    let out = g.add(
        "res2",
        Op::Binary {
            kind: crate::graph::BinaryKind::Add,
        },
        &[x1, d],
        vec![1, h],
        f,
    );
    g.output(out);

    let s = Session::new(rt(), &w).unwrap();
    let plan = s.prepare_plan(&g).unwrap();
    let mut state = StateStore::new();
    let mut run = || {
        let inputs = vec![Tensor::F32(rand(h, 9)), Tensor::U32(vec![100])];
        s.run_prepared(&g, &plan, Some(&mut state), inputs).unwrap();
    };
    for _ in 0..20 {
        run();
    }
    take_run_samples();
    for _ in 0..200 {
        run();
    }
    let mut samples = take_run_samples();
    assert_eq!(samples.len(), 200);
    let mut run: Vec<u64> = samples.iter().map(|s| s.run_ns).collect();
    run.sort_unstable();
    samples.sort_by_key(|s| s.gpu_ns);
    // `run_us` (host wall per call, dispatch and sync included) is the basis comparable with XLA's timing.
    println!(
        "FUSION_BLOCK {{\"us\": {:.1}, \"run_us\": {:.1}, \"dispatches\": {}}}",
        samples[100].gpu_ns as f64 / 1e3,
        run[100] as f64 / 1e3,
        samples[100].dispatches
    );
}

/// A two-output step's outputs are both live until their own last reader: later steps never
/// get arena space that overlaps a still-live extra output.
#[test]
fn allocate_gives_extra_outs_distinct_live_ranges() {
    let f = DType::F32;
    let mut g = Graph::new();
    let x = g.input("x", vec![2, 4], f);
    let gv = g.weight("g", vec![4], f);
    let a = g.add("a", Op::RmsNorm { eps: 1e-5 }, &[x, gv], vec![2, 4], f);
    let b = g.add(
        "b",
        Op::Unary {
            kind: crate::graph::UnaryKind::Silu,
        },
        &[x],
        vec![2, 4],
        f,
    );
    let c = g.add("c", Op::RmsNorm { eps: 1e-5 }, &[a, gv], vec![2, 4], f);
    let d = g.add(
        "d",
        Op::Binary {
            kind: crate::graph::BinaryKind::Add,
        },
        &[b, c],
        vec![2, 4],
        f,
    );
    g.output(d);
    let norm = |x, out, extra_outs: Vec<ValueId>, reads| Planned {
        name: String::new(),
        op: Step::RmsNorm {
            x,
            g: gv,
            res: None,
            scale: None,
            chain: None,
            rope: None,
            rows: 2,
            d: 4,
            eps: 1e-5,
        },
        out,
        extra_outs,
        reads,
    };
    let steps = vec![
        norm(x, a, vec![b], vec![x, gv]),
        norm(a, c, vec![], vec![a, gv]),
        norm(b, d, vec![], vec![b, c]),
    ];
    let (offsets, _) = allocate(&g, &steps, &HashMap::new()).unwrap();
    let range = |v: ValueId| (offsets[&v], offsets[&v] + 2 * 4 * 4);
    let disjoint =
        |p: ValueId, q: ValueId| range(p).1 <= range(q).0 || range(q).1 <= range(p).0;
    // a is read by step 1, b by step 2, c by step 2, d is the graph output.
    for (p, q) in [(a, b), (a, c), (b, c), (b, d), (c, d)] {
        assert!(
            disjoint(p, q),
            "{p:?} and {q:?} overlap: {:?} {:?}",
            range(p),
            range(q)
        );
    }
}

/// Single-reader Rope after RmsNorm (and RmsNormBatch) and single-reader Q/K Rope feeding
/// decode CachedAttention fuse into their producer/consumer steps and match the CPU executor.
#[test]
fn rmsnorm_and_cached_attention_fuse_rope_and_match_cpu() {
    let f = DType::F32;
    let (heads, kv_heads, hd) = (4usize, 2usize, 8usize);
    let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
    let rope = Op::Rope {
        head_dim: hd,
        rotary_dim: hd,
        variant: crate::graph::RopeVariant::HalfSplit,
        mscale: 1.0,
    };
    // 1. Q/K/V RmsNorms + Q/K Ropes batch into one RmsNormBatch with rope_mask.
    let mut g1 = Graph::new();
    let q_in = g1.input("q", vec![1, q_dim], f);
    let k_in = g1.input("k", vec![1, kv_dim], f);
    let v_in = g1.input("v", vec![1, kv_dim], f);
    let pos = g1.input("pos", vec![1], DType::U32);
    let gw = g1.weight("gw", vec![hd], f);
    let inv = g1.weight("inv", vec![hd / 2], f);
    let qn = g1.add(
        "qn",
        Op::RmsNorm { eps: 1e-5 },
        &[q_in, gw],
        vec![1, q_dim],
        f,
    );
    let kn = g1.add(
        "kn",
        Op::RmsNorm { eps: 1e-5 },
        &[k_in, gw],
        vec![1, kv_dim],
        f,
    );
    let vn = g1.add(
        "vn",
        Op::RmsNorm { eps: 1e-5 },
        &[v_in, gw],
        vec![1, kv_dim],
        f,
    );
    let qr = g1.add("qr", rope.clone(), &[qn, pos, inv], vec![1, q_dim], f);
    let kr = g1.add("kr", rope.clone(), &[kn, pos, inv], vec![1, kv_dim], f);
    let ks = g1.state("ks", vec![16, kv_dim], f, false);
    let vs = g1.state("vs", vec![16, kv_dim], f, false);
    let att = Op::CachedAttention {
        heads,
        kv_heads,
        head_dim: hd,
        v_head_dim: hd,
        scale: 0.25,
        causal: true,
        window: None,
        soft_cap: None,
        write: true,
    };
    let out1 = g1.add(
        "att",
        att.clone(),
        &[qr, kr, vn, ks, vs, pos],
        vec![1, q_dim],
        f,
    );
    g1.output(out1);
    let mut w = Weights::new();
    w.insert("gw", Tensor::F32(rand(hd, 1)));
    w.insert("inv", Tensor::F32(rand(hd / 2, 2)));
    let mut sess1 = Session::new(rt(), &w).unwrap();
    let prep1 = sess1.prepare_plan(&g1).unwrap();
    assert_eq!(
        prep1.steps.len(),
        2,
        "RmsNormBatch(with Q/K Rope) + CachedAttention"
    );
    assert!(matches!(
        prep1.steps[0].op,
        Step::RmsNormBatch { rope: Some(_), .. }
    ));

    // 2. Without Q/K RmsNorms, Q/K Ropes fuse directly into decode CachedAttention.
    let mut g2 = Graph::new();
    let q2 = g2.input("q", vec![1, q_dim], f);
    let k2 = g2.input("k", vec![1, kv_dim], f);
    let v2 = g2.input("v", vec![1, kv_dim], f);
    let pos2 = g2.input("pos", vec![1], DType::U32);
    let inv2 = g2.weight("inv", vec![hd / 2], f);
    let qr2 = g2.add("qr", rope.clone(), &[q2, pos2, inv2], vec![1, q_dim], f);
    let kr2 = g2.add("kr", rope, &[k2, pos2, inv2], vec![1, kv_dim], f);
    let ks2 = g2.state("ks", vec![16, kv_dim], f, false);
    let vs2 = g2.state("vs", vec![16, kv_dim], f, false);
    let out2 = g2.add(
        "att",
        att,
        &[qr2, kr2, v2, ks2, vs2, pos2],
        vec![1, q_dim],
        f,
    );
    g2.output(out2);
    let mut sess2 = Session::new(rt(), &w).unwrap();
    let prep2 = sess2.prepare_plan(&g2).unwrap();
    assert_eq!(prep2.steps.len(), 1, "Q/K Rope fused into CachedAttention");
    assert!(matches!(
        prep2.steps[0].op,
        Step::CachedAttention { rope: Some(_), .. }
    ));

    // f16 KV covers the RoPE-fused f16 store; its rounding (<= 2^-11 of each K/V) sets the bound.
    for (kv_f16, tol) in [(false, 1e-5), (true, 1e-3)] {
        sess1.set_kv_f16(kv_f16);
        sess2.set_kv_f16(kv_f16);
        for (g, sess) in [(&g1, &sess1), (&g2, &sess2)] {
            let (mut cs, mut ms) = (StateStore::new(), StateStore::new());
            for step in 0..3u32 {
                let inp = vec![
                    Tensor::F32(rand(q_dim, 10 + step as u64)),
                    Tensor::F32(rand(kv_dim, 20 + step as u64)),
                    Tensor::F32(rand(kv_dim, 30 + step as u64)),
                    Tensor::U32(vec![step]),
                ];
                let want = cpu::run(g, &w, &mut cs, inp.clone()).unwrap();
                let got = sess.run_with_state(g, Some(&mut ms), inp).unwrap();
                let e = rel_err(want[0].as_f32().unwrap(), got[0].as_f32().unwrap());
                assert!(e < tol, "f16 KV {kv_f16} step {step}: {e}");
            }
        }
    }
}

/// A residual RmsNorm (`norm(x0) * g0 + res0`) feeding a MegakernelMlp fuses into the MLP's
/// `pre_res` (`rmsnorm_fused_f32`) while keeping `x` live for the MLP down-projection residual.
#[test]
fn megakernel_mlp_fuses_pre_residual_norm_and_matches_cpu() {
    let f = DType::F32;
    let (h, inter) = (96usize, 192usize);
    let mut g = Graph::new();
    let x0 = g.input("x0", vec![1, h], f);
    let res0 = g.input("res0", vec![1, h], f);
    let g0 = g.weight("g0", vec![h], f);
    let norm_w = g.weight("norm_w", vec![h], f);
    let gate_w = g.weight("gate_w", vec![inter, h], f);
    let up_w = g.weight("up_w", vec![inter, h], f);
    let down_w = g.weight("down_w", vec![h, inter], f);
    let post_w = g.weight("post_w", vec![h], f);

    let attn_norm = g.add(
        "attn_norm",
        Op::RmsNorm { eps: 1e-5 },
        &[x0, g0],
        vec![1, h],
        f,
    );
    let x = g.add(
        "res_attn",
        Op::Binary {
            kind: crate::graph::BinaryKind::Add,
        },
        &[res0, attn_norm],
        vec![1, h],
        f,
    );
    let ff_in = g.add(
        "pre_ff_ln",
        Op::RmsNorm { eps: 1e-6 },
        &[x, norm_w],
        vec![1, h],
        f,
    );
    let gate = g.add("gate", Op::Linear, &[ff_in, gate_w], vec![1, inter], f);
    let up = g.add("up", Op::Linear, &[ff_in, up_w], vec![1, inter], f);
    let act = g.add("act", Op::GeGlu, &[gate, up], vec![1, inter], f);
    let down = g.add("down", Op::Linear, &[act, down_w], vec![1, h], f);
    let down_norm = g.add(
        "post_ff_ln",
        Op::RmsNorm { eps: 1e-5 },
        &[down, post_w],
        vec![1, h],
        f,
    );
    let out = g.add(
        "res_ff",
        Op::Binary {
            kind: crate::graph::BinaryKind::Add,
        },
        &[x, down_norm],
        vec![1, h],
        f,
    );
    g.output(out);
    crate::graph::egraph::fuse_rule(&mut g, None, crate::graph::egraph::Rule::MegakernelMlp);

    let mut w = Weights::new();
    for (k, s) in [("g0", 1u64), ("norm_w", 2), ("post_w", 3)] {
        w.insert(k, Tensor::F32(rand(h, s)));
    }
    for (k, n, s) in [
        ("gate_w", inter * h, 4u64),
        ("up_w", inter * h, 5),
        ("down_w", h * inter, 6),
    ] {
        w.insert(k, Tensor::F32(rand(n, s)));
    }
    let sess = Session::new(rt(), &w).unwrap();
    let prep = sess.prepare_plan(&g).unwrap();
    assert_eq!(prep.steps.len(), 1);
    assert!(matches!(
        prep.steps[0].op,
        Step::MegakernelMlp {
            pre_res: Some(_),
            ..
        }
    ));
    let inp = vec![Tensor::F32(rand(h, 7)), Tensor::F32(rand(h, 8))];
    let (want, got) = both(&g, &w, inp, None);
    assert!(rel_err(&want, &got) < 1e-4);
}

/// Adjacent independent RmsNorms (different rows and eps) and Ropes (different head counts) become
/// one batch step each; a norm reading a batch member's output stays outside it. All outputs match
/// the CPU executor.
#[test]
fn horizontal_batches_match_cpu() {
    let f = DType::F32;
    let mut g = Graph::new();
    let xs = [(3usize, 1e-5f32), (1, 1e-6), (2, 1e-3)]
        .map(|(rows, eps)| (g.input("x", vec![rows, 4], f), rows, eps));
    let gv = g.weight("g", vec![4], f);
    let mut outs: Vec<ValueId> = xs
        .iter()
        .map(|&(x, rows, eps)| g.add("n", Op::RmsNorm { eps }, &[x, gv], vec![rows, 4], f))
        .collect();
    // Depends on the first member: cannot join the run.
    outs.push(g.add(
        "dep",
        Op::RmsNorm { eps: 1e-5 },
        &[outs[0], gv],
        vec![3, 4],
        f,
    ));
    let pos = g.input("pos", vec![1], DType::U32);
    let inv = g.weight("inv", vec![2], f);
    let rope = Op::Rope {
        head_dim: 4,
        rotary_dim: 4,
        variant: crate::graph::RopeVariant::HalfSplit,
        mscale: 1.0,
    };
    let (xa, xb) = (g.input("xa", vec![1, 8], f), g.input("xb", vec![1, 4], f));
    outs.push(g.add("ra", rope.clone(), &[xa, pos, inv], vec![1, 8], f));
    outs.push(g.add("rb", rope, &[xb, pos, inv], vec![1, 4], f));
    for &o in &outs {
        g.output(o);
    }
    let (steps, aliases) = plan(&g).unwrap();
    let steps = batch_horizontal(steps, &aliases).unwrap();
    let kinds: Vec<_> = steps
        .iter()
        .map(|s| kernel_kind(&format!("{:?}", s.op)).to_string())
        .collect();
    assert_eq!(kinds, ["RmsNormBatch", "RmsNorm", "RopeBatch"]);

    let mut w = Weights::new();
    w.insert("g", Tensor::F32(vec![0.5, 1.2, -0.8, 2.1]));
    w.insert("inv", Tensor::F32(vec![1.0, 0.1]));
    let inputs = vec![
        Tensor::F32(rand(12, 1)),
        Tensor::F32(rand(4, 2)),
        Tensor::F32(rand(8, 3)),
        Tensor::U32(vec![5]),
        Tensor::F32(rand(8, 4)),
        Tensor::F32(rand(4, 5)),
    ];
    let want = cpu::run_stateless(&g, &w, inputs.clone()).unwrap();
    let got = Session::new(rt(), &w).unwrap().run(&g, inputs).unwrap();
    for (i, (a, b)) in want.iter().zip(&got).enumerate() {
        let e = rel_err(a.as_f32().unwrap(), b.as_f32().unwrap());
        assert!(e < 1e-5, "output {i}: {e}");
    }
}

#[test]
fn rmsnorm_swiglu_and_geglu_match_cpu() {
    let f = DType::F32;
    let mut g = Graph::new();
    let x = g.input("x", vec![2, 4], f);
    let u_val = g.input("u", vec![2, 4], f);
    let gv = g.weight("g", vec![4], f);
    let norm = g.add("norm", Op::RmsNorm { eps: 1e-5 }, &[x, gv], vec![2, 4], f);
    let swiglu = g.add("swiglu", Op::SwiGlu, &[norm, u_val], vec![2, 4], f);
    let geglu = g.add("geglu", Op::GeGlu, &[norm, u_val], vec![2, 4], f);
    let out = g.add(
        "sum",
        Op::Binary {
            kind: crate::graph::BinaryKind::Add,
        },
        &[swiglu, geglu],
        vec![2, 4],
        f,
    );
    g.output(out);

    let mut w = Weights::new();
    w.insert("g", Tensor::F32(vec![0.5, 1.2, -0.8, 2.1]));
    let inputs = vec![
        Tensor::F32(vec![1.0, 2.0, 3.0, 4.0, 0.5, -0.5, 1.5, -1.5]),
        Tensor::F32(vec![0.2, -0.4, 1.1, -0.7, 0.9, -1.2, 0.3, 0.8]),
    ];
    let (want, got) = both(&g, &w, inputs, None);
    assert!(rel_err(&want, &got) < 1e-4);
}

/// `norm(x) * g + r` with the Add as the norm's only reader is one fused step; a norm that
/// is also read elsewhere stays a plain norm. Both match the CPU executor.
#[test]
fn rmsnorm_residual_fuses_and_matches_cpu() {
    let f = DType::F32;
    let mut g = Graph::new();
    let x = g.input("x", vec![3, 4], f);
    let r = g.input("r", vec![3, 4], f);
    let gv = g.weight("g", vec![4], f);
    let n1 = g.add("n1", Op::RmsNorm { eps: 1e-5 }, &[x, gv], vec![3, 4], f);
    let s1 = g.add(
        "s1",
        Op::Binary {
            kind: crate::graph::BinaryKind::Add,
        },
        &[r, n1],
        vec![3, 4],
        f,
    );
    let n2 = g.add("n2", Op::RmsNorm { eps: 1e-5 }, &[s1, gv], vec![3, 4], f);
    let s2 = g.add(
        "s2",
        Op::Binary {
            kind: crate::graph::BinaryKind::Add,
        },
        &[n2, s1],
        vec![3, 4],
        f,
    );
    let o = g.add(
        "o",
        Op::Binary {
            kind: crate::graph::BinaryKind::Mul,
        },
        &[n2, s2],
        vec![3, 4],
        f,
    );
    g.output(o);
    let (steps, _) = plan(&g).unwrap();
    // n1+s1 fuse; n2 has two readers (s2 and o), so it and s2 stay separate: norm, norm, add, mul.
    assert_eq!(steps.len(), 4, "norm+add, norm, add, mul");
    assert!(matches!(steps[0].op, Step::RmsNorm { res: Some(_), .. }));
    assert!(matches!(steps[1].op, Step::RmsNorm { res: None, .. }));

    let mut w = Weights::new();
    w.insert("g", Tensor::F32(vec![0.5, 1.2, -0.8, 2.1]));
    let inputs = vec![
        Tensor::F32((0..12).map(|i| (i as f32 * 0.7).sin() * 2.0).collect()),
        Tensor::F32((0..12).map(|i| (i as f32 * 1.3).cos()).collect()),
    ];
    let (want, got) = both(&g, &w, inputs, None);
    assert!(rel_err(&want, &got) < 1e-5);
}

/// `(norm(x) * g + r) * s` followed by `norm(out) * g2` fuses into one `rmsnorm_fused_f32`
/// dispatch writing both the scaled residual stream and the pre-normalised activation.
#[test]
fn rmsnorm_scale_and_chain_fuse_and_match_cpu() {
    let f = DType::F32;
    let mut g = Graph::new();
    let x = g.input("x", vec![2, 8], f);
    let r = g.input("r", vec![2, 8], f);
    let g1 = g.weight("g1", vec![8], f);
    let g2 = g.weight("g2", vec![8], f);
    let sc = g.weight("sc", vec![1], f);
    let n1 = g.add("n1", Op::RmsNorm { eps: 1e-5 }, &[x, g1], vec![2, 8], f);
    let a1 = g.add(
        "a1",
        Op::Binary {
            kind: crate::graph::BinaryKind::Add,
        },
        &[r, n1],
        vec![2, 8],
        f,
    );
    let m1 = g.add(
        "m1",
        Op::Binary {
            kind: crate::graph::BinaryKind::Mul,
        },
        &[a1, sc],
        vec![2, 8],
        f,
    );
    let n2 = g.add("n2", Op::RmsNorm { eps: 1e-6 }, &[m1, g2], vec![2, 8], f);
    let out = g.add(
        "out",
        Op::Binary {
            kind: crate::graph::BinaryKind::Mul,
        },
        &[m1, n2],
        vec![2, 8],
        f,
    );
    g.output(out);

    let mut w = Weights::new();
    w.insert(
        "g1",
        Tensor::F32((0..8).map(|i| 0.5 + i as f32 * 0.1).collect()),
    );
    w.insert(
        "g2",
        Tensor::F32((0..8).map(|i| 1.2 - i as f32 * 0.08).collect()),
    );
    w.insert("sc", Tensor::F32(vec![0.70710677]));
    let sess = Session::new(rt(), &w).unwrap();
    let prep = sess.prepare_plan(&g).unwrap();
    assert_eq!(
        prep.steps.len(),
        2,
        "fused norm+add+scale+prenorm, then add"
    );
    assert!(matches!(
        prep.steps[0].op,
        Step::RmsNorm {
            res: Some(_),
            scale: Some(_),
            chain: Some(_),
            ..
        }
    ));
    let inputs = vec![
        Tensor::F32((0..16).map(|i| (i as f32 * 0.4).sin()).collect()),
        Tensor::F32((0..16).map(|i| (i as f32 * 0.9).cos()).collect()),
    ];
    let (want, got) = both(&g, &w, inputs, None);
    assert!(rel_err(&want, &got) < 1e-5);
}

/// Every kernel source this build emits compiles to a pipeline: the hand library as MSL 4.0
/// with MPP (each hand, group and MPP kernel as a pipeline with SIMD width 32 and at least the
/// threads its encoder dispatches), each generated GEMV (`gen` and `gen64`) and dequant variant,
/// each expert-GEMV grid binding, and the fused GeGlu→GEMV chain. Compile only: nothing is
/// dispatched and no runtime (host-profile measurement) is created.
#[test]
fn every_kernel_source_compiles() {
    use crate::graph::codegen::{self, GemvParams};
    let device = Device::system_default().expect("Metal device");
    let (lib, mpp) = compile(&device).expect("kernel library");
    mpp.expect("MSL 4.0 kernel library");
    // `Runtime::new` checks only the `TG` floor; the fused delta-rule step dispatches more.
    let need = |name: &str| -> u64 {
        if name.starts_with("deltanet_step_") {
            ext::deltanet::THREADS as u64
        } else if name.starts_with("gemmgen_")
            || name.starts_with("flash_attn_")
            || name.starts_with("gemm_mpp_")
        {
            TG4
        } else {
            TG
        }
    };
    let check = |name: &str, pso: &ComputePipelineState| {
        assert_eq!(pso.thread_execution_width(), 32, "{name}");
        assert!(
            pso.max_total_threads_per_threadgroup() >= need(name),
            "{name}: at most {} threads per threadgroup",
            pso.max_total_threads_per_threadgroup()
        );
    };
    for name in BASE_KERNELS
        .iter()
        .copied()
        .chain(ext::kernel_names())
        .chain(MPP_KERNELS)
    {
        let f = lib
            .get_function(name, None)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let pso = device
            .new_compute_pipeline_state_with_function(&f)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        check(name, &pso);
    }
    let build = |src: &str, name: &str| {
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        // SAFETY: `setLanguageVersion:` takes an NSUInteger enum; 0x40000 is MTLLanguageVersion4_0.
        let () = unsafe { msg_send![&*opts, setLanguageVersion: MSL_4_0] };
        let lib = device
            .new_library_with_source(src, &opts)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let f = lib
            .get_function(name, None)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let pso = device
            .new_compute_pipeline_state_with_function(&f)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        check(name, &pso);
    };
    let head = "#include <metal_stdlib>\nusing namespace metal;\n";
    let common = codegen::gemv_common(&KERNELS_MSL);
    let head_mpp = "#define GIBBS_MPP 1\n#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\n";
    let gemm_common = codegen::gemm_common(&KERNELS_MSL);
    let mut codecs: Vec<&codec::Codec> = Vec::new();
    for f in [
        QuantFormat::GGUF_Q4_0,
        QuantFormat::GGUF_Q4_1,
        QuantFormat::GGUF_Q5_0,
        QuantFormat::GGUF_Q5_1,
        QuantFormat::GGUF_Q8_0,
        QuantFormat::GGUF_Q2_K,
        QuantFormat::GGUF_Q3_K,
        QuantFormat::GGUF_Q4_K_M,
        QuantFormat::GGUF_Q5_K_M,
        QuantFormat::GGUF_Q6_K,
        QuantFormat::GGUF_IQ4_NL,
        QuantFormat::GGUF_IQ4_XS,
        QuantFormat::GGUF_IQ3_XXS,
        QuantFormat::GGUF_IQ3_S,
        QuantFormat::GGUF_IQ2_XXS,
        QuantFormat::GGUF_IQ2_XS,
        QuantFormat::GGUF_IQ2_S,
        QuantFormat::GGUF_IQ2_M,
        QuantFormat::GGUF_IQ1_S,
        QuantFormat::GGUF_TQ1_0,
        QuantFormat::GGUF_TQ2_0,
        QuantFormat::GGUF_NVFP4,
    ] {
        if let Some(c) = codec::codec(f) {
            if !codecs.iter().any(|d| d.format == c.format) {
                codecs.push(c);
            }
        }
    }
    for &c in &codecs {
        let variants = tune::GEN_ROWS.map(tune::GemvVariant::generated);
        for v in variants {
            let p = v.gen_params();
            build(
                &format!("{head}{common}{}", codegen::gemv_packed(c, p)),
                &codegen::gemv_packed_name(c, p),
            );
        }
        build(
            &format!("{head}{}", codegen::dequant_packed(c)),
            &codegen::dequant_packed_name(c),
        );
        for r in [2, 4] {
            let p = GemvParams {
                rows_per_sg: r,
                ..Default::default()
            };
            build(
                &format!("{head}{}", codegen::gemv_indexed(c, p)),
                &codegen::gemv_indexed_name(c, p),
            );
        }
        for epi in [0, 1, 2] {
            build(
                &format!("{head_mpp}{gemm_common}{}", codegen::gemm_packed(c, epi, false)),
                &codegen::gemm_packed_name(c, epi, false),
            );
        }
        for epi in [0, 1, 3] {
            build(
                &format!("{head_mpp}{gemm_common}{}", codegen::gemm_packed(c, epi, true)),
                &codegen::gemm_packed_name(c, epi, true),
            );
        }
    }
    for c in [&codegen::CODEC_F16, &codegen::CODEC_BF16] {
        for epi in [0, 1, 2] {
            build(
                &format!("{head_mpp}{gemm_common}{}", codegen::gemm_packed(c, epi, false)),
                &codegen::gemm_packed_name(c, epi, false),
            );
        }
        for epi in [0, 1, 3] {
            build(
                &format!("{head_mpp}{gemm_common}{}", codegen::gemm_packed(c, epi, true)),
                &codegen::gemm_packed_name(c, epi, true),
            );
        }
    }
}

/// Every packed format (generated and hand kernels): the GPU GEMV equals the CPU executor
/// (which dequantizes through `dequantize_*_block`), with and without a bias epilogue.
#[test]
fn gemv_matches_cpu_for_every_packed_format() {
    use crate::model::layers::gguf_block_codec;
    let formats = [
        QuantFormat::GGUF_Q4_1,
        QuantFormat::GGUF_Q5_0,
        QuantFormat::GGUF_Q5_1,
        QuantFormat::GGUF_Q3_K,
        QuantFormat::GGUF_Q5_K_M,
        QuantFormat::GGUF_Q6_K,
        QuantFormat::GGUF_IQ4_NL,
        QuantFormat::GGUF_IQ3_XXS,
        QuantFormat::GGUF_IQ2_XXS,
        QuantFormat::GGUF_IQ2_XS,
        QuantFormat::GGUF_IQ1_S,
        QuantFormat::GGUF_TQ1_0,
        QuantFormat::GGUF_TQ2_0,
        // hand kernels
        QuantFormat::GGUF_Q4_0,
        QuantFormat::GGUF_Q8_0,
        QuantFormat::GGUF_IQ4_XS,
        QuantFormat::GGUF_IQ2_M,
        QuantFormat::GGUF_IQ2_S,
        QuantFormat::GGUF_Q4_K_M,
        QuantFormat::GGUF_Q2_K,
        QuantFormat::GGUF_IQ3_S,
        QuantFormat::GGUF_NVFP4,
    ];
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    // Finite f16 scale (|x| in 2^-6..2^1) so decoded values fit in half and never meet NaN/inf.
    let f16 = |next: &mut dyn FnMut() -> u32| -> [u8; 2] {
        let exp = 9 + (next() % 8) as u16;
        (((next() & 1) as u16) << 15 | exp << 10 | (next() & 0x3FF) as u16).to_le_bytes()
    };
    // m = 3 runs the GEMV kernels; m = 20 (>= 16) the batched GEMM (hand kernels, or codec expansion + f32 GEMM).
    for (f, m) in formats.into_iter().flat_map(|f| [(f, 3usize), (f, 20)]) {
        let (be, bb, _) = gguf_block_codec(f).expect("CPU dequantizer");
        let scale_at: &[usize] = match f {
            QuantFormat::GGUF_Q4_1
            | QuantFormat::GGUF_Q5_1
            | QuantFormat::GGUF_Q5_K_M
            | QuantFormat::GGUF_Q4_K_M => &[0, 2],
            QuantFormat::GGUF_Q2_K => &[80, 82],
            QuantFormat::GGUF_Q6_K => &[208],
            QuantFormat::GGUF_Q3_K => &[108],
            QuantFormat::GGUF_TQ1_0 => &[52],
            QuantFormat::GGUF_TQ2_0 => &[64],
            _ => &[0],
        };
        let (n, k) = (37usize, be * 3);
        let mut raw: Vec<u8> = (0..n * (k / be) * bb)
            .map(|_| (next() >> 4) as u8)
            .collect();
        for blk in raw.chunks_exact_mut(bb) {
            for &o in scale_at {
                blk[o..o + 2].copy_from_slice(&f16(&mut next));
            }
        }
        let mut g = Graph::new();
        let x = g.input("x", vec![m, k], DType::F32);
        let w_val = g.weight("w", vec![n, k], DType::Quant(f));
        let b_val = g.weight("b", vec![n], DType::F32);
        let y = g.add(
            "lin",
            Op::Linear,
            &[x, w_val, b_val],
            vec![m, n],
            DType::F32,
        );
        g.output(y);
        let mut w = Weights::new();
        w.insert(
            "w",
            Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(f, raw, n, k)),
        );
        w.insert(
            "b",
            Tensor::F32((0..n).map(|i| i as f32 * 0.01 - 0.2).collect()),
        );
        let xs: Vec<f32> = (0..m * k)
            .map(|_| (next() % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let want = cpu::run_stateless(&g, &w, vec![Tensor::F32(xs.clone())])
            .unwrap()
            .remove(0)
            .into_f32()
            .unwrap();
        // Every variant of the format (hand and generated at each rows/simdgroup); formats with
        // no variant choice (NVFP4) run their one kernel.
        let variants = WeightFormat::of_quant(f).gemv_variants();
        let variants: Vec<Option<GemvVariant>> = if variants.is_empty() {
            vec![None]
        } else {
            variants.into_iter().map(Some).collect()
        };
        for v in variants {
            let mut s = Session::new(rt(), &w).unwrap();
            s.set_variant_override(v);
            let got = s
                .run(&g, vec![Tensor::F32(xs.clone())])
                .unwrap()
                .remove(0)
                .into_f32()
                .unwrap();
            let err = rel_err(&want, &got);
            let tol = if m >= 16 { 1e-3 } else { 1e-4 };
            assert!(
                err < tol,
                "{f:?} m={m} {v:?}: packed linear differs from CPU by {err}"
            );
        }
    }
}

#[test]
fn megakernel_mlp_matches_cpu_for_every_variant() {
    use crate::model::layers::gguf_block_codec;
    for f in [
        QuantFormat::GGUF_Q4_0,
        QuantFormat::GGUF_Q4_K_M,
        QuantFormat::GGUF_Q8_0,
    ] {
        for post_norm in [false, true] {
            let (hidden, inter) = (256usize, 512usize);
            let mut seed = 0x1234_5678_9ABC_DEF1u64 ^ f as u64;
            let mut next = move || {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (seed >> 33) as u32
            };
            let (be, bb, _) = gguf_block_codec(f).expect("CPU dequantizer");
            let scale_at: &[usize] = match f {
                QuantFormat::GGUF_Q4_K_M => &[0, 2],
                _ => &[0],
            };
            let mut w = Weights::new();
            let mut g = Graph::new();
            let mut quant = |g: &mut Graph, w: &mut Weights, name: &str, n: usize, k: usize| {
                let mut raw: Vec<u8> = (0..n * (k / be) * bb)
                    .map(|_| (next() >> 4) as u8)
                    .collect();
                for blk in raw.chunks_exact_mut(bb) {
                    for &o in scale_at {
                        let exp = 9 + (next() % 14) as u16;
                        let d =
                            ((next() & 1) as u16) << 15 | exp << 10 | (next() & 0x3FF) as u16;
                        blk[o..o + 2].copy_from_slice(&d.to_le_bytes());
                    }
                }
                w.insert(
                    name,
                    Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(f, raw, n, k)),
                );
                g.weight(name, vec![n, k], DType::Quant(f))
            };
            let x = g.input("x", vec![1, hidden], DType::F32);
            let norm_w = g.weight("norm.weight", vec![hidden], DType::F32);
            w.insert("norm.weight", Tensor::F32(vec![1.0; hidden]));
            let normed = g.add(
                "norm",
                Op::RmsNorm { eps: 1e-5 },
                &[x, norm_w],
                vec![1, hidden],
                DType::F32,
            );
            let gw = quant(&mut g, &mut w, "gate.weight", inter, hidden);
            let uw = quant(&mut g, &mut w, "up.weight", inter, hidden);
            let dw = quant(&mut g, &mut w, "down.weight", hidden, inter);
            let gate = g.add(
                "gate",
                Op::Linear,
                &[normed, gw],
                vec![1, inter],
                DType::F32,
            );
            let up = g.add("up", Op::Linear, &[normed, uw], vec![1, inter], DType::F32);
            let act = g.add(
                "swiglu",
                Op::SwiGlu,
                &[gate, up],
                vec![1, inter],
                DType::F32,
            );
            let mut down = g.add("down", Op::Linear, &[act, dw], vec![1, hidden], DType::F32);
            if post_norm {
                let pn = g.weight("post_norm.weight", vec![hidden], DType::F32);
                w.insert("post_norm.weight", Tensor::F32(vec![0.9; hidden]));
                down = g.add(
                    "post_norm",
                    Op::RmsNorm { eps: 1e-5 },
                    &[down, pn],
                    vec![1, hidden],
                    DType::F32,
                );
            }
            let out = g.add(
                "add",
                Op::Binary {
                    kind: crate::graph::BinaryKind::Add,
                },
                &[x, down],
                vec![1, hidden],
                DType::F32,
            );
            g.output(out);
            let unfused = g.clone();
            assert_eq!(
                crate::graph::egraph::fuse_rule(
                    &mut g,
                    None,
                    crate::graph::egraph::Rule::MegakernelMlp
                ),
                1
            );
            let xs: Vec<f32> = (0..256).map(|i| (i as f32 * 0.37).sin()).collect();
            let want = cpu::run_stateless(&unfused, &w, vec![Tensor::F32(xs.clone())])
                .unwrap()
                .remove(0)
                .into_f32()
                .unwrap();
            for v in WeightFormat::of_quant(f).gemv_variants() {
                let mut s = Session::new(rt(), &w).unwrap();
                s.set_variant_override(Some(v));
                let got = s
                    .run(&g, vec![Tensor::F32(xs.clone())])
                    .unwrap()
                    .remove(0)
                    .into_f32()
                    .unwrap();
                let err = rel_err(&want, &got);
                assert!(
                    err < 1e-3,
                    "{f:?} post_norm={post_norm} {v:?}: megakernel MLP differs from CPU by {err}"
                );
            }
        }
    }
}

/// The cooperative decode MLP megakernel (norm + gate + up + GLU in one dispatch) equals the
/// multi-dispatch path (the same MLP unfused: RmsNorm, GEMVs, GLU, GEMV, Add) to 1e-5 relative RMS.
#[test]
fn cooperative_mlp_matches_multi_dispatch() {
    use crate::model::layers::gguf_block_codec;
    for f in [
        QuantFormat::GGUF_Q4_0,
        QuantFormat::GGUF_Q4_K_M,
        QuantFormat::GGUF_Q8_0,
    ] {
        for is_geglu in [false, true] {
            let (hidden, inter) = (1024usize, 768usize);
            let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ f as u64;
            let mut next = move || {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (seed >> 33) as u32
            };
            let (be, bb, _) = gguf_block_codec(f).expect("CPU dequantizer");
            let scale_at: &[usize] = match f {
                QuantFormat::GGUF_Q4_K_M => &[0, 2],
                _ => &[0],
            };
            let mut w = Weights::new();
            let mut g = Graph::new();
            let mut quant = |g: &mut Graph, w: &mut Weights, name: &str, n: usize, k: usize| {
                let mut raw: Vec<u8> = (0..n * (k / be) * bb)
                    .map(|_| (next() >> 4) as u8)
                    .collect();
                for blk in raw.chunks_exact_mut(bb) {
                    for &o in scale_at {
                        let exp = 9 + (next() % 6) as u16;
                        let d =
                            ((next() & 1) as u16) << 15 | exp << 10 | (next() & 0x3FF) as u16;
                        blk[o..o + 2].copy_from_slice(&d.to_le_bytes());
                    }
                }
                w.insert(
                    name,
                    Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(f, raw, n, k)),
                );
                g.weight(name, vec![n, k], DType::Quant(f))
            };
            let x = g.input("x", vec![1, hidden], DType::F32);
            let norm_w = g.weight("norm.weight", vec![hidden], DType::F32);
            w.insert("norm.weight", Tensor::F32(rand(hidden, 3)));
            let normed = g.add(
                "norm",
                Op::RmsNorm { eps: 1e-6 },
                &[x, norm_w],
                vec![1, hidden],
                DType::F32,
            );
            let gw = quant(&mut g, &mut w, "gate.weight", inter, hidden);
            let uw = quant(&mut g, &mut w, "up.weight", inter, hidden);
            let dw = quant(&mut g, &mut w, "down.weight", hidden, inter);
            let gate = g.add("gate", Op::Linear, &[normed, gw], vec![1, inter], DType::F32);
            let up = g.add("up", Op::Linear, &[normed, uw], vec![1, inter], DType::F32);
            let glu = if is_geglu { Op::GeGlu } else { Op::SwiGlu };
            let act = g.add("glu", glu, &[gate, up], vec![1, inter], DType::F32);
            let down = g.add("down", Op::Linear, &[act, dw], vec![1, hidden], DType::F32);
            let out = g.add(
                "add",
                Op::Binary {
                    kind: crate::graph::BinaryKind::Add,
                },
                &[x, down],
                vec![1, hidden],
                DType::F32,
            );
            g.output(out);
            let unfused = g.clone();
            assert_eq!(
                crate::graph::egraph::fuse_rule(
                    &mut g,
                    None,
                    crate::graph::egraph::Rule::MegakernelMlp
                ),
                1
            );
            let xs = rand(hidden, 7);
            let run = |graph: &Graph| {
                Session::new(rt(), &w)
                    .unwrap()
                    .run(graph, vec![Tensor::F32(xs.clone())])
                    .unwrap()
                    .remove(0)
                    .into_f32()
                    .unwrap()
            };
            let (want, got) = (run(&unfused), run(&g));
            let num: f32 = want.iter().zip(&got).map(|(a, b)| (a - b) * (a - b)).sum();
            let den: f32 = want.iter().map(|a| a * a).sum();
            let rms = (num / den).sqrt();
            assert!(rms <= 1e-5, "{f:?} geglu={is_geglu}: relative RMS {rms}");
        }
    }
}

/// TQ2_0 lives in the planar device layout (groups of 8 blocks): GEMV (m = 1, every variant), the
/// dequantizing batched path (m = 20) equal the CPU executor at shapes where rows are not whole groups
/// (5, 9 blocks per row), n is not a multiple of the rows per threadgroup and the total block count is
/// not a multiple of 8.
#[test]
fn tq2_0_planar_layout_matches_cpu() {
    let f = QuantFormat::GGUF_TQ2_0;
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    for (n, k) in [(37usize, 1280usize), (101, 2304), (19, 2048), (5, 768)] {
        let mut raw: Vec<u8> = (0..n * (k / 256) * 66)
            .map(|_| (next() >> 4) as u8)
            .collect();
        for blk in raw.chunks_exact_mut(66) {
            // Finite f16 scale in 2^-6..2^4.
            let exp = 9 + (next() % 14) as u16;
            let d = ((next() & 1) as u16) << 15 | exp << 10 | (next() & 0x3FF) as u16;
            blk[64..].copy_from_slice(&d.to_le_bytes());
        }
        let mut w = Weights::new();
        w.insert(
            "w",
            Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(f, raw, n, k)),
        );
        for m in [1usize, 20] {
            let mut g = Graph::new();
            let x = g.input("x", vec![m, k], DType::F32);
            let w_val = g.weight("w", vec![n, k], DType::Quant(f));
            let y = g.add("lin", Op::Linear, &[x, w_val], vec![m, n], DType::F32);
            g.output(y);
            let xs: Vec<f32> = (0..m * k)
                .map(|_| (next() % 2001) as f32 / 1000.0 - 1.0)
                .collect();
            let want = cpu::run_stateless(&g, &w, vec![Tensor::F32(xs.clone())])
                .unwrap()
                .remove(0)
                .into_f32()
                .unwrap();
            for v in WeightFormat::of_quant(f).gemv_variants() {
                let mut s = Session::new(rt(), &w).unwrap();
                s.set_variant_override(Some(v));
                let got = s
                    .run(&g, vec![Tensor::F32(xs.clone())])
                    .unwrap()
                    .remove(0)
                    .into_f32()
                    .unwrap();
                let err = rel_err(&want, &got);
                let tol = if m >= 16 { 1e-3 } else { 1e-4 };
                assert!(
                    err < tol,
                    "TQ2_0 {n}x{k} m={m} {v:?}: differs from CPU by {err}"
                );
            }
        }
    }
}

/// Generated-GEMV bandwidth at lm_head-like shapes (`cargo test --release -- --ignored --nocapture`).
#[test]
#[ignore = "bandwidth measurement"]
fn generated_gemv_bandwidth() {
    use crate::model::layers::gguf_block_codec;
    for f in [
        QuantFormat::GGUF_TQ2_0,
        QuantFormat::GGUF_Q4_0,
        QuantFormat::GGUF_Q8_0,
        QuantFormat::GGUF_Q6_K,
        QuantFormat::GGUF_Q5_K_M,
    ] {
        let (be, bb, _) = gguf_block_codec(f).unwrap();
        for (n, k) in [
            (262144usize, 1536usize),
            (6144, 1536),
            (1536, 6144),
            (14336, 4096),
        ] {
            let raw = vec![0u8; n * (k / be) * bb];
            let bytes = raw.len();
            let mut g = Graph::new();
            let x = g.input("x", vec![1, k], DType::F32);
            let wv = g.weight("w", vec![n, k], DType::Quant(f));
            let y = g.add("lin", Op::Linear, &[x, wv], vec![1, n], DType::F32);
            g.output(y);
            let mut w = Weights::new();
            w.insert(
                "w",
                Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(f, raw, n, k)),
            );
            let s = Session::new(rt(), &w).unwrap();
            let xs = Tensor::F32(vec![0.5; k]);
            s.run(&g, vec![xs.clone()]).unwrap();
            // GPU-side time of the command buffer (host copy-in/out excluded).
            let mut dt = 0.0;
            for _ in 0..20 {
                s.run(&g, vec![xs.clone()]).unwrap();
                dt += s.last_gpu_seconds() / 20.0;
            }
            println!(
                "{f:?} {n}x{k}: {:.0} us, {:.1} GB/s",
                dt * 1e6,
                bytes as f64 / dt / 1e9
            );
        }
    }
}

/// Best GEMV rate per format at the projection shapes of a ~2B model, DRAM-streaming conditions
/// (`cargo test --release --lib shape_rates -- --ignored --nocapture`; `SHAPE_RATES_FMT=TQ2_0`
/// keeps only the formats whose name contains the text).
#[test]
#[ignore = "bandwidth measurement"]
fn shape_rates() {
    for f in [
        QuantFormat::GGUF_TQ2_0,
        QuantFormat::GGUF_Q4_0,
        QuantFormat::GGUF_Q4_K_M,
        QuantFormat::GGUF_Q6_K,
        QuantFormat::GGUF_Q8_0,
        QuantFormat::GGUF_IQ4_XS,
        QuantFormat::GGUF_IQ2_XXS,
        QuantFormat::GGUF_Q5_K_M,
        QuantFormat::GGUF_Q3_K,
        QuantFormat::GGUF_IQ4_NL,
    ] {
        if std::env::var("SHAPE_RATES_FMT").is_ok_and(|s| !format!("{f:?}").contains(&s)) {
            continue;
        }
        let fmt = WeightFormat::of_quant(f);
        for shape in [
            (6144usize, 1536usize),
            (1536, 6144),
            (2048, 1536),
            (1536, 2048),
            (1536, 2816),
            (2816, 768),
            (262144, 1536),
            (262144, 2816),
        ] {
            let t = rt().bench_gemv(f, &[shape], &fmt.gemv_variants()).unwrap();
            let best = t.iter().map(|x| x.1).fold(0.0, f64::max);
            println!(
                "{f:?} {shape:?}: best {best:.0} GB/s, {} variants {:?}",
                t.len(),
                t.iter()
                    .map(|x| (x.0.label(), x.1 as u32))
                    .collect::<Vec<_>>()
            );
        }
    }
}

/// Linear -> Silu/SwiGlu/GeGlu on a BF16 weight at m below and above the batched-GEMM cut (16):
/// the activation must never be dropped.
#[test]
fn packed_linear_activation_epilogues_match_cpu_at_every_m() {
    let (n, k) = (40usize, 64usize);
    let bf: Vec<u16> = (0..n * k)
        .map(|i| crate::model::f32_to_bf16(0.05 * ((i % 11) as f32 - 5.0)))
        .collect();
    for m in [1usize, 15, 16, 64] {
        for act in 0..3 {
            let mut g = Graph::new();
            let x = g.input("x", vec![m, k], DType::F32);
            let up = g.input("up", vec![m, n], DType::F32);
            let wv = g.weight("w", vec![n, k], DType::BF16);
            let lin = g.add("lin", Op::Linear, &[x, wv], vec![m, n], DType::F32);
            let y = match act {
                0 => g.add(
                    "a",
                    Op::Unary {
                        kind: crate::graph::UnaryKind::Silu,
                    },
                    &[lin],
                    vec![m, n],
                    DType::F32,
                ),
                1 => g.add("a", Op::SwiGlu, &[lin, up], vec![m, n], DType::F32),
                _ => g.add("a", Op::GeGlu, &[lin, up], vec![m, n], DType::F32),
            };
            g.output(y);
            let mut w = Weights::new();
            w.insert("w", Tensor::BF16(bf.clone()));
            let inputs = vec![Tensor::F32(rand(m * k, 7)), Tensor::F32(rand(m * n, 8))];
            let (want, got) = both(&g, &w, inputs, None);
            let err = rel_err(&want, &got);
            assert!(err < 1e-3, "m={m} act={act}: rel err {err}");
        }
    }
}

#[test]
fn gemv_q8_0_matches_cpu() {
    let (m, n, k) = (3, 37, 32 * 40);
    let mut g = Graph::new();
    let x = g.input("x", vec![m, k], DType::F32);
    let w_val = g.weight("w", vec![n, k], DType::Quant(QuantFormat::GGUF_Q8_0));
    let b_val = g.weight("b", vec![n], DType::F32);
    let y = g.add(
        "lin",
        Op::Linear,
        &[x, w_val, b_val],
        vec![m, n],
        DType::F32,
    );
    g.output(y);

    let n_blocks = n * (k / 32);
    let mut raw = Vec::with_capacity(n_blocks * 34);
    for b in 0..n_blocks {
        let d = crate::model::f32_to_f16(0.125 * ((b % 5) as f32 + 1.0));
        raw.extend_from_slice(&d.to_le_bytes());
        for j in 0..32 {
            let q = (((j as i32 * 7 + b as i32 * 3) % 255) - 127) as i8;
            raw.push(q as u8);
        }
    }
    let mut w = Weights::new();
    w.insert(
        "w",
        Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(
            QuantFormat::GGUF_Q8_0,
            raw,
            n,
            k,
        )),
    );
    w.insert("b", Tensor::F32(rand(n, 17)));
    let inputs = vec![Tensor::F32(rand(m * k, 42))];
    let (want, got) = both(&g, &w, inputs, None);
    assert!(
        rel_err(&want, &got) < 1e-4,
        "want {:?} vs got {:?}",
        &want[..4],
        &got[..4]
    );
}

#[test]
fn gemv_q4_0_matches_cpu() {
    let (m, n, k) = (2, 37, 32 * 40);
    let mut g = Graph::new();
    let x = g.input("x", vec![m, k], DType::F32);
    let w_val = g.weight("w", vec![n, k], DType::Quant(QuantFormat::GGUF_Q4_0));
    let b_val = g.weight("b", vec![n], DType::F32);
    let y = g.add(
        "lin",
        Op::Linear,
        &[x, w_val, b_val],
        vec![m, n],
        DType::F32,
    );
    g.output(y);

    let n_blocks = n * (k / 32);
    let mut raw = Vec::with_capacity(n_blocks * 18);
    for b in 0..n_blocks {
        let d = crate::model::f32_to_f16(0.25 * ((b % 3) as f32 + 1.0));
        raw.extend_from_slice(&d.to_le_bytes());
        for j in 0..16 {
            let low = (j * 3 + b) % 16;
            let high = (j * 5 + b * 2) % 16;
            raw.push((low | (high << 4)) as u8);
        }
    }
    let mut w = Weights::new();
    w.insert(
        "w",
        Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(
            QuantFormat::GGUF_Q4_0,
            raw,
            n,
            k,
        )),
    );
    w.insert("b", Tensor::F32(rand(n, 23)));
    let inputs = vec![Tensor::F32(rand(m * k, 99))];
    let (want, got) = both(&g, &w, inputs, None);
    assert!(
        rel_err(&want, &got) < 1e-4,
        "want {:?} vs got {:?}",
        &want[..4],
        &got[..4]
    );
}
#[test]
fn gemv_q4_0_residual_matches_cpu() {
    let (m, n, k) = (1, 37, 32 * 40);
    let mut g = Graph::new();
    let x = g.input("x", vec![m, k], DType::F32);
    let res_in = g.input("res", vec![m, n], DType::F32);
    let w_val = g.weight("w", vec![n, k], DType::Quant(QuantFormat::GGUF_Q4_0));
    let b_val = g.weight("b", vec![n], DType::F32);
    let y = g.add(
        "lin",
        Op::Linear,
        &[x, w_val, b_val],
        vec![m, n],
        DType::F32,
    );
    let y_res = g.add(
        "res_add",
        Op::Binary {
            kind: crate::graph::BinaryKind::Add,
        },
        &[y, res_in],
        vec![m, n],
        DType::F32,
    );
    g.output(y_res);

    let n_blocks = n * (k / 32);
    let mut raw = Vec::with_capacity(n_blocks * 18);
    for b in 0..n_blocks {
        let d = crate::model::f32_to_f16(0.25 * ((b % 3) as f32 + 1.0));
        raw.extend_from_slice(&d.to_le_bytes());
        for j in 0..16 {
            let low = (j * 3 + b) % 16;
            let high = (j * 5 + b * 2) % 16;
            raw.push((low | (high << 4)) as u8);
        }
    }
    let mut w = Weights::new();
    w.insert(
        "w",
        Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(
            QuantFormat::GGUF_Q4_0,
            raw,
            n,
            k,
        )),
    );
    w.insert("b", Tensor::F32(rand(n, 23)));
    let inputs = vec![Tensor::F32(rand(m * k, 99)), Tensor::F32(rand(m * n, 77))];
    let (want, got) = both(&g, &w, inputs, None);
    assert!(
        rel_err(&want, &got) < 1e-4,
        "want {:?} vs got {:?}",
        &want[..4],
        &got[..4]
    );
}


#[test]
fn gemv_bf16_matches_cpu() {
    let (m, n, k) = (4, 37, 1000);
    let mut g = Graph::new();
    let x = g.input("x", vec![m, k], DType::F32);
    let w_val = g.weight("w", vec![n, k], DType::BF16);
    let b_val = g.weight("b", vec![n], DType::F32);
    let y = g.add(
        "lin",
        Op::Linear,
        &[x, w_val, b_val],
        vec![m, n],
        DType::F32,
    );
    g.output(y);

    let f32_vals = rand(n * k, 77);
    let bf16_vals: Vec<u16> = f32_vals
        .iter()
        .map(|&v| (v.to_bits() >> 16) as u16)
        .collect();
    let mut w = Weights::new();
    w.insert("w", Tensor::BF16(bf16_vals));
    w.insert("b", Tensor::F32(rand(n, 31)));
    let inputs = vec![Tensor::F32(rand(m * k, 123))];
    let (want, got) = both(&g, &w, inputs, None);
    assert!(
        rel_err(&want, &got) < 1e-4,
        "want {:?} vs got {:?}",
        &want[..4],
        &got[..4]
    );
}
#[test]
fn gemv_f16_matches_cpu() {
    let (m, n, k) = (3, 37, 1000);
    let mut g = Graph::new();
    let x = g.input("x", vec![m, k], DType::F32);
    let w_val = g.weight("w", vec![n, k], DType::F16);
    let b_val = g.weight("b", vec![n], DType::F32);
    let y = g.add(
        "lin",
        Op::Linear,
        &[x, w_val, b_val],
        vec![m, n],
        DType::F32,
    );
    g.output(y);

    let f32_vals = rand(n * k, 77);
    let f16_vals: Vec<u16> = f32_vals
        .iter()
        .map(|&v| crate::model::f32_to_f16(v))
        .collect();
    let mut w = Weights::new();
    w.insert("w", Tensor::F16(f16_vals));
    w.insert("b", Tensor::F32(rand(n, 31)));
    let inputs = vec![Tensor::F32(rand(m * k, 123))];
    let (want, got) = both(&g, &w, inputs, None);
    assert!(
        rel_err(&want, &got) < 1e-4,
        "want {:?} vs got {:?}",
        &want[..4],
        &got[..4]
    );
}
#[test]
#[ignore = "benchmark: measures hardware roofline saturation, run via cargo test --lib test_tiled_gemv_roofline_saturation -- --ignored --test-threads=1"]
fn test_tiled_gemv_roofline_saturation() {
    use metal::MTLResourceOptions;
    let rt = Runtime::new(&crate::model::hub::HuggingFaceHub::cache_dir()).unwrap();
    let roofline_gbps = rt.benchmark_memory_bandwidth().unwrap_or(298.0);
    println!(
        "Hardware streaming read roofline (memory_sweep): {:.2} GB/s",
        roofline_gbps
    );

    #[repr(C)]
    struct PackedGemvParams(u32, u32, u32, u32, u32);

    // 1. Q4_0: rows = 16384, cols = 8192 (~75.5 MB)
    {
        let (rows, cols) = (16384usize, 8192usize);
        let n_blocks = rows * (cols / 32);
        let bytes = n_blocks * 18;
        let w_buf = rt
            .device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
        let x_buf = rt
            .device
            .new_buffer((cols * 4) as u64, MTLResourceOptions::StorageModeShared);
        let y_buf = rt
            .device
            .new_buffer((rows * 4) as u64, MTLResourceOptions::StorageModeShared);
        let pso = rt.pso("gemv_q4_0").unwrap();
        let p = PackedGemvParams(rows as u32, cols as u32, 1, 0, 0);

        let us = rt
            .timed(20, |enc| {
                enc.set_compute_pipeline_state(pso);
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&y_buf), 0);
                set_bytes(enc, 3, &p);
                enc.set_buffer(4, Some(&y_buf), 0);
                enc.set_buffer(5, Some(&y_buf), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new((rows as u64).div_ceil(16), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                Ok(())
            })
            .unwrap();

        let gb_per_sec = (bytes as f64 / 1e9) / (us * 1e-6);
        println!(
            "Tiled GEMV Q4_0: {:.2} GB/s ({:.1}% roofline)",
            gb_per_sec,
            (gb_per_sec / roofline_gbps) * 100.0
        );
        let target_pct = 0.85;
        assert!(
            gb_per_sec > target_pct * roofline_gbps,
            "Q4_0 bandwidth {:.2} GB/s should reach at least {:.0}% of measured roofline ({:.2} GB/s)",
            gb_per_sec, target_pct * 100.0, target_pct * roofline_gbps
        );
    }

    // 2. Q8_0: rows = 8192, cols = 8192 (~71.3 MB)
    {
        let (rows, cols) = (8192usize, 8192usize);
        let n_blocks = rows * (cols / 32);
        let bytes = n_blocks * 34;
        let w_buf = rt
            .device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
        let x_buf = rt
            .device
            .new_buffer((cols * 4) as u64, MTLResourceOptions::StorageModeShared);
        let y_buf = rt
            .device
            .new_buffer((rows * 4) as u64, MTLResourceOptions::StorageModeShared);
        let pso = rt.pso("gemv_q8_0").unwrap();
        let p = PackedGemvParams(rows as u32, cols as u32, 1, 0, 0);

        let us = rt
            .timed(20, |enc| {
                enc.set_compute_pipeline_state(pso);
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&y_buf), 0);
                set_bytes(enc, 3, &p);
                enc.set_buffer(4, Some(&y_buf), 0);
                enc.set_buffer(5, Some(&y_buf), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new((rows as u64).div_ceil(8), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                Ok(())
            })
            .unwrap();

        let gb_per_sec = (bytes as f64 / 1e9) / (us * 1e-6);
        println!(
            "Tiled GEMV Q8_0: {:.2} GB/s ({:.1}% roofline)",
            gb_per_sec,
            (gb_per_sec / roofline_gbps) * 100.0
        );
        let target_pct = 0.85;
        assert!(
            gb_per_sec > target_pct * roofline_gbps,
            "Q8_0 bandwidth {:.2} GB/s should reach at least {:.0}% of measured roofline ({:.2} GB/s)",
            gb_per_sec, target_pct * 100.0, target_pct * roofline_gbps
        );
    }

    // 3. BF16: rows = 8192, cols = 4096 (~67.1 MB)
    {
        let (rows, cols) = (8192usize, 4096usize);
        let bytes = rows * cols * 2;
        let w_buf = rt
            .device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
        let x_buf = rt
            .device
            .new_buffer((cols * 4) as u64, MTLResourceOptions::StorageModeShared);
        let y_buf = rt
            .device
            .new_buffer((rows * 4) as u64, MTLResourceOptions::StorageModeShared);
        let pso = rt.pso("gemv_bf16").unwrap();
        let p = PackedGemvParams(rows as u32, cols as u32, 1, 0, 0);

        let us = rt
            .timed(20, |enc| {
                enc.set_compute_pipeline_state(pso);
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&y_buf), 0);
                set_bytes(enc, 3, &p);
                enc.set_buffer(4, Some(&y_buf), 0);
                enc.set_buffer(5, Some(&y_buf), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new((rows as u64).div_ceil(16), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                Ok(())
            })
            .unwrap();

        let gb_per_sec = (bytes as f64 / 1e9) / (us * 1e-6);
        println!(
            "Tiled GEMV BF16: {:.2} GB/s ({:.1}% roofline)",
            gb_per_sec,
            (gb_per_sec / roofline_gbps) * 100.0
        );
        let target_pct = 0.85;
        assert!(
            gb_per_sec > target_pct * roofline_gbps,
            "BF16 bandwidth {:.2} GB/s should reach at least {:.0}% of measured roofline ({:.2} GB/s)",
            gb_per_sec, target_pct * 100.0, target_pct * roofline_gbps
        );
    }
}

#[test]
fn gemm_batched_packed_matches_cpu() {
    let (m, n, k) = (32, 64, 64);
    // 1. BF16
    {
        let mut g = Graph::new();
        let x = g.input("x", vec![m, k], DType::F32);
        let w_val = g.weight("w", vec![n, k], DType::BF16);
        let b_val = g.weight("b", vec![n], DType::F32);
        let y = g.add(
            "lin",
            Op::Linear,
            &[x, w_val, b_val],
            vec![m, n],
            DType::F32,
        );
        g.output(y);
        let mut w = Weights::new();
        let raw_w: Vec<u16> = rand(n * k, 123)
            .into_iter()
            .map(crate::model::f32_to_bf16)
            .collect();
        w.insert("w", Tensor::BF16(raw_w));
        w.insert("b", Tensor::F32(rand(n, 456)));
        let inputs = vec![Tensor::F32(rand(m * k, 789))];
        let (want, got) = both(&g, &w, inputs, None);
        assert!(rel_err(&want, &got) < 1e-3, "BF16 batched mismatch");
    }
    // 2. Q8_0
    {
        let mut g = Graph::new();
        let x = g.input("x", vec![m, k], DType::F32);
        let w_val = g.weight("w", vec![n, k], DType::Quant(QuantFormat::GGUF_Q8_0));
        let y = g.add("lin", Op::Linear, &[x, w_val], vec![m, n], DType::F32);
        g.output(y);
        let n_blocks = n * (k / 32);
        let mut raw = Vec::with_capacity(n_blocks * 34);
        for b in 0..n_blocks {
            let d = crate::model::f32_to_f16(0.125 * ((b % 5) as f32 + 1.0));
            raw.extend_from_slice(&d.to_le_bytes());
            for j in 0..32 {
                let q = (((j as i32 * 7 + b as i32 * 3) % 255) - 127) as i8;
                raw.push(q as u8);
            }
        }
        let mut w = Weights::new();
        w.insert(
            "w",
            Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(
                QuantFormat::GGUF_Q8_0,
                raw,
                n,
                k,
            )),
        );
        let inputs = vec![Tensor::F32(rand(m * k, 999))];
        let (want, got) = both(&g, &w, inputs, None);
        assert!(rel_err(&want, &got) < 1e-3, "Q8_0 batched mismatch");
    }
}

#[test]
fn gemm_packed_in_tile_dequant_matches_cpu() {
    use crate::model::layers::{encode_rows, gguf_block_codec};
    let rt = rt();
    assert!(rt.profile.mpp.is_ok(), "MPP required: {:?}", rt.profile.mpp);
    let formats = [
        QuantFormat::BF16,
        QuantFormat::FP16,
        QuantFormat::GGUF_Q4_0,
        QuantFormat::GGUF_Q8_0,
        QuantFormat::GGUF_Q4_K_M,
        QuantFormat::GGUF_Q5_K_M,
        QuantFormat::GGUF_Q6_K,
        QuantFormat::GGUF_TQ2_0,
        QuantFormat::GGUF_IQ2_XXS,
        QuantFormat::GGUF_IQ4_NL,
    ];
    let mut seed = 0x517CC1B727220A95u64;
    let mut next_u32 = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    let f16_scale = |next: &mut dyn FnMut() -> u32| -> [u8; 2] {
        let exp = 10 + (next() % 6) as u16;
        (((next() & 1) as u16) << 15 | exp << 10 | (next() & 0x3FF) as u16).to_le_bytes()
    };
    for fmt in formats {
        for (n, k) in [(64usize, 256usize), (200, 1536)] {
            let mut w = Weights::new();
            let w_f32 = rand(n * k, 100 + fmt as u64);
            let w_dtype = match fmt {
                QuantFormat::BF16 => {
                    let raw_w: Vec<u16> = w_f32.iter().copied().map(crate::model::f32_to_bf16).collect();
                    w.insert("w", Tensor::BF16(raw_w));
                    DType::BF16
                }
                QuantFormat::FP16 => {
                    let raw_w: Vec<u16> = w_f32.iter().copied().map(crate::model::f32_to_f16).collect();
                    w.insert("w", Tensor::F16(raw_w));
                    DType::F16
                }
                QuantFormat::GGUF_Q4_0
                | QuantFormat::GGUF_Q8_0
                | QuantFormat::GGUF_Q4_K_M
                | QuantFormat::GGUF_Q5_K_M
                | QuantFormat::GGUF_Q6_K => {
                    let raw = encode_rows(&w_f32, k, fmt).expect("encode_rows");
                    w.insert(
                        "w",
                        Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(fmt, raw, n, k)),
                    );
                    DType::Quant(fmt)
                }
                _ => {
                    let (be, bb, dequant) = gguf_block_codec(fmt).expect("CPU dequantizer");
                    let scale_off = if fmt == QuantFormat::GGUF_TQ2_0 { 64 } else { 0 };
                    let mut raw: Vec<u8> = (0..n * (k / be) * bb)
                        .map(|_| (next_u32() >> 4) as u8)
                        .collect();
                    let mut decoded = vec![0.0f32; be];
                    for blk in raw.chunks_exact_mut(bb) {
                        blk[scale_off..scale_off + 2].copy_from_slice(&f16_scale(&mut next_u32));
                        dequant(blk, &mut decoded);
                        if !decoded.iter().all(|v| v.is_finite()) {
                            blk.fill(0);
                        }
                    }
                    w.insert(
                        "w",
                        Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(fmt, raw, n, k)),
                    );
                    DType::Quant(fmt)
                }
            };
            w.insert("b", Tensor::F32(rand(n, 200)));
            let session = Session::new(rt.clone(), &w).unwrap();
            let x_512 = rand(512 * k, 300);
            let r_512 = rand(512 * n, 400);
            let build_graph = |m: usize, case: usize| -> Graph {
                let mut g = Graph::new();
                let x = g.input("x", vec![m, k], DType::F32);
                let r = g.input("r", vec![m, n], DType::F32);
                let wv = g.weight("w", vec![n, k], w_dtype);
                let bv = g.weight("b", vec![n], DType::F32);
                let ins = if case == 0 {
                    vec![x, wv]
                } else {
                    vec![x, wv, bv]
                };
                let lin = g.add("lin", Op::Linear, &ins, vec![m, n], DType::F32);
                let out = match case {
                    0 | 1 => lin,
                    2 => g.add(
                        "gelu",
                        Op::Unary {
                            kind: crate::graph::UnaryKind::GeluErf,
                        },
                        &[lin],
                        vec![m, n],
                        DType::F32,
                    ),
                    _ => g.add(
                        "res",
                        Op::Binary {
                            kind: crate::graph::BinaryKind::Add,
                        },
                        &[r, lin],
                        vec![m, n],
                        DType::F32,
                    ),
                };
                g.output(out);
                g
            };
            // Compute CPU reference once for m=512; row-independence gives m=16 and m=77 as prefixes.
            for case in 0..4 {
                let g_512 = build_graph(512, case);
                let want_512 = cpu::run_stateless(
                    &g_512,
                    &w,
                    vec![Tensor::F32(x_512.clone()), Tensor::F32(r_512.clone())],
                )
                .unwrap()
                .remove(0)
                .into_f32()
                .unwrap();
                for m in [16usize, 77, 512] {
                    let g = build_graph(m, case);
                    let inp = vec![
                        Tensor::F32(x_512[..m * k].to_vec()),
                        Tensor::F32(r_512[..m * n].to_vec()),
                    ];
                    let got = session
                        .run(&g, inp)
                        .unwrap()
                        .remove(0)
                        .into_f32()
                        .unwrap();
                    let want = &want_512[..m * n];
                    let err = rel_err(want, &got);
                    assert!(
                        err <= 1e-2,
                        "{fmt:?} {m}x{n}x{k} case {case}: rel_err {err:e} > 1e-2"
                    );
                }
            }
            if rt.profile.mpp.is_ok() {
                assert!(
                    session.dequant_scratch.lock().is_none(),
                    "{fmt:?}: dequant_scratch should not be allocated when MPP is active"
                );
            }
        }
    }
}
#[test]
fn test_metal_batched_prefill_throughput() {
    let (n, k) = (576, 576);
    let runtime = rt();

    for &fmt in &[WeightFormat::BF16, WeightFormat::Q4_0] {
        let mut w = Weights::new();
        let dtype = match fmt {
            WeightFormat::BF16 => {
                let raw_w: Vec<u16> = rand(n * k, 42)
                    .into_iter()
                    .map(crate::model::f32_to_bf16)
                    .collect();
                w.insert("w", Tensor::BF16(raw_w));
                DType::BF16
            }
            WeightFormat::Q4_0 => {
                let n_blocks = n * (k / 32);
                let mut raw = Vec::with_capacity(n_blocks * 18);
                for b in 0..n_blocks {
                    let d = crate::model::f32_to_f16(0.125 * ((b % 5) as f32 + 1.0));
                    raw.extend_from_slice(&d.to_le_bytes());
                    for _ in 0..16 {
                        raw.push(0x88);
                    }
                }
                w.insert(
                    "w",
                    Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(
                        QuantFormat::GGUF_Q4_0,
                        raw,
                        n,
                        k,
                    )),
                );
                DType::Quant(QuantFormat::GGUF_Q4_0)
            }
            _ => unreachable!(),
        };
        let session = Session::new(runtime.clone(), &w).unwrap();

        for &m in &[512, 1024, 2048] {
            let mut g = Graph::new();
            let x = g.input("x", vec![m, k], DType::F32);
            let w_val = g.weight("w", vec![n, k], dtype);
            let y = g.add("lin", Op::Linear, &[x, w_val], vec![m, n], DType::F32);
            g.output(y);

            let inp = vec![Tensor::F32(vec![0.1f32; m * k])];

            // Warmup
            let _ = session.run(&g, inp.clone()).unwrap();

            // Benchmark
            let iters = 5;
            let t0 = Instant::now();
            for _ in 0..iters {
                let _ = session.run(&g, inp.clone()).unwrap();
            }
            let elapsed = t0.elapsed().as_secs_f64() / iters as f64;
            let throughput = m as f64 / elapsed;

            println!(
                "Batched Prefill [{:?}] M={:<4} N={:<4} K={:<4}: {:.2} ms | Throughput: {:>8.1} tok/s",
                fmt, m, n, k, elapsed * 1000.0, throughput
            );
            assert!(
                throughput > 100.0,
                "Throughput {:.1} tok/s must be > 100 tok/s for M={m} ({:?})",
                throughput,
                fmt
            );
        }
    }
}

#[test]
fn unsupported_ops_bad_ids_and_missing_weights_are_errors() {
    let f = DType::F32;
    let s = Session::new(rt(), &Weights::new()).unwrap();
    // Relu lowers at F32 only; a u32 value has no kernel.
    for op in [Op::Unary {
        kind: crate::graph::UnaryKind::Relu,
    }] {
        let mut g = Graph::new();
        let x = g.input("x", vec![4], DType::U32);
        let y = g.add("act", op, &[x], vec![4], DType::U32);
        g.output(y);
        let e = s.run(&g, vec![Tensor::F32(vec![0.0; 4])]).unwrap_err();
        assert!(e.contains("the Metal kernels take f32"), "{e}");
    }
    let mut g = Graph::new();
    let x = g.input("x", vec![1, 2], f);
    let wv = g.weight("absent", vec![3, 2], f);
    let y = g.add("lin", Op::Linear, &[x, wv], vec![1, 3], f);
    g.output(y);
    assert!(s
        .run(&g, vec![Tensor::F32(vec![0.0; 2])])
        .unwrap_err()
        .contains("missing weight 'absent'"));

    let mut w = Weights::new();
    w.insert("t", Tensor::F32(vec![0.0; 6]));
    let s = Session::new(rt(), &w).unwrap();
    let mut g = Graph::new();
    let ids = g.input("ids", vec![2], DType::U32);
    let t = g.weight("t", vec![3, 2], f);
    let e = g.add("emb", Op::Embedding, &[t, ids], vec![2, 2], f);
    g.output(e);
    let err = s.run(&g, vec![Tensor::U32(vec![0, 3])]).unwrap_err();
    assert!(err.contains("id 3 out of range"), "{err}");

    let mut g = Graph::new();
    let a = g.input("a", vec![2], DType::U32);
    let b = g.input("b", vec![2], f);
    let y = g.add(
        "sum",
        Op::Binary {
            kind: crate::graph::BinaryKind::Add,
        },
        &[a, b],
        vec![2],
        f,
    );
    g.output(y);
    let err = s
        .run(&g, vec![Tensor::U32(vec![1, 2]), Tensor::F32(vec![0.0; 2])])
        .unwrap_err();
    assert!(err.contains("the Metal kernels take f32"), "{err}");
}

#[test]
fn second_load_restores_pipelines_and_profile_from_cache() {
    let dir = tempfile::tempdir().unwrap();
    let first = Runtime::new(dir.path()).unwrap();
    assert!(!first.archive_hit, "{}", first.report);
    let second = Runtime::new(dir.path()).unwrap();
    assert!(second.archive_hit, "{}", second.report);
    assert!(
        second.report.contains("host profile from"),
        "{}",
        second.report
    );
}

#[test]
fn test_dynamic_roofline_scan() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = Runtime::new(dir.path()).unwrap();
    let profile = runtime.profile();
    assert!(
        profile.memory_bandwidth_gbps > 0.0,
        "measured bandwidth must be > 0: got {}",
        profile.memory_bandwidth_gbps
    );

    // For Gemma 4 E2B Q4_0 (~1.44 GB = 1_440_000_000 bytes):
    let gemma4_e2b_bytes = 1_440_000_000u64;
    let roofline = profile.roofline_tok_s(gemma4_e2b_bytes);
    println!("Host Metal GPU: {}", profile.device);
    println!(
        "Measured streaming bandwidth: {:.2} GB/s",
        profile.memory_bandwidth_gbps
    );
    println!("Gemma 4 E2B Q4_0 Roofline: {:.1} tok/s", roofline);
    assert!(
        roofline > 0.0,
        "roofline rate must be dynamic and > 0, got {roofline}"
    );
    let expected_roofline = profile.memory_bandwidth_gbps / 1.44;
    assert!((roofline - expected_roofline).abs() < 1e-4);
}
#[test]
fn test_attention_and_state_ops_match_cpu() {
    // 1. Slice
    {
        let mut g = Graph::new();
        let x = g.input("x", vec![2, 8], DType::F32);
        let y = g.add(
            "sl",
            Op::Slice {
                axis: -1,
                start: 2,
                end: 6,
            },
            &[x],
            vec![2, 4],
            DType::F32,
        );
        g.output(y);
        let inp = vec![Tensor::F32(vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0,
        ])];
        let (want, got) = both(&g, &Weights::new(), inp, None);
        assert_eq!(want, got);
    }
    // 2. Rope (Interleaved and HalfSplit)
    for variant in [
        crate::graph::RopeVariant::Interleaved,
        crate::graph::RopeVariant::HalfSplit,
    ] {
        let mut g = Graph::new();
        let mut w = Weights::new();
        let x = g.input("x", vec![1, 8], DType::F32);
        let pos = g.input("pos", vec![1], DType::U32);
        let inv_freq = g.weight("rope.inv_freq.default", vec![4], DType::F32);
        w.insert(
            "rope.inv_freq.default".to_string(),
            Tensor::F32(vec![1.0, 0.1, 0.01, 0.001]),
        );
        let y = g.add(
            "rope",
            Op::Rope {
                head_dim: 8,
                rotary_dim: 8,
                variant,
                mscale: 1.0,
            },
            &[x, pos, inv_freq],
            vec![1, 8],
            DType::F32,
        );
        g.output(y);
        let inp = vec![
            Tensor::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            Tensor::U32(vec![3]),
        ];
        let (want, got) = both(&g, &w, inp, None);
        assert!(
            rel_err(&want, &got) < 1e-4,
            "Rope mismatch: {want:?} vs {got:?}"
        );
    }

    // 3. CachedAttention
    {
        let mut g = Graph::new();
        let q = g.input("q", vec![1, 4], DType::F32);
        let k = g.input("k", vec![1, 4], DType::F32);
        let v = g.input("v", vec![1, 4], DType::F32);
        let y = g.add(
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
        g.output(y);
        let inp = vec![
            Tensor::F32(vec![1.0, 0.0, 0.0, 0.0]),
            Tensor::F32(vec![1.0, 0.0, 0.0, 0.0]),
            Tensor::F32(vec![0.5, 1.5, 2.5, 3.5]),
        ];
        let (want, got) = both(&g, &Weights::new(), inp, None);
        assert!(rel_err(&want, &got) < 1e-4, "CachedAttention mismatch");
    }


    // 4. CausalConv1dCached
    {
        let mut g = Graph::new();
        let x = g.input("x", vec![3, 1], DType::F32);
        let w_val = g.weight("w", vec![2], DType::F32);
        let y = g.add(
            "conv",
            Op::CausalConv1dCached {
                kernel: 2,
                dilation: 1,
            },
            &[x, w_val],
            vec![3, 1],
            DType::F32,
        );
        g.output(y);
        let mut w = Weights::new();
        w.insert("w", Tensor::F32(vec![0.5, 1.0]));
        let inp = vec![Tensor::F32(vec![2.0, 4.0, 6.0])];
        let (want, got) = both(&g, &w, inp, None);
        assert!(
            rel_err(&want, &got) < 1e-4,
            "Conv1d mismatch: {want:?} vs {got:?}"
        );
    }

    // 5. DeltaRule
    {
        let mut g = Graph::new();
        let q = g.input("q", vec![1, 2], DType::F32);
        let k = g.input("k", vec![1, 2], DType::F32);
        let v = g.input("v", vec![1, 2], DType::F32);
        let beta = g.input("beta", vec![1, 1], DType::F32);
        let log_decay = g.input("log_decay", vec![1], DType::F32);
        let y = g.add(
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
        g.output(y);
        let inp = vec![
            Tensor::F32(vec![1.0, 0.5]),
            Tensor::F32(vec![0.2, 0.8]),
            Tensor::F32(vec![0.3, -0.4]),
            Tensor::F32(vec![0.9]),
            Tensor::F32(vec![-0.1]),
        ];
        let (want, got) = both(&g, &Weights::new(), inp, None);
        assert!(
            rel_err(&want, &got) < 1e-4,
            "DeltaRule mismatch: {want:?} vs {got:?}"
        );
    }
}
#[test]
fn test_zero_copy_slice_aliasing_m1_matches_cpu() {
    let mut g = Graph::new();
    let x = g.input("x", vec![1, 64], DType::F32);
    let w_large = g.weight("w_large", vec![192, 64], DType::F32);
    let lin = g.add("lin", Op::Linear, &[x, w_large], vec![1, 192], DType::F32);

    // 3-way slice
    let v0 = g.add(
        "sl0",
        Op::Slice {
            axis: -1,
            start: 0,
            end: 64,
        },
        &[lin],
        vec![1, 64],
        DType::F32,
    );
    let v1 = g.add(
        "sl1",
        Op::Slice {
            axis: -1,
            start: 64,
            end: 128,
        },
        &[lin],
        vec![1, 64],
        DType::F32,
    );
    let v2 = g.add(
        "sl2",
        Op::Slice {
            axis: -1,
            start: 128,
            end: 192,
        },
        &[lin],
        vec![1, 64],
        DType::F32,
    );

    // Slice of a slice: check transitive composition
    let v0_sub = g.add(
        "sl0_sub",
        Op::Slice {
            axis: -1,
            start: 16,
            end: 48,
        },
        &[v0],
        vec![1, 32],
        DType::F32,
    );

    // Downstream steps consuming each slice view
    let w1 = g.weight("w1", vec![64, 64], DType::F32);
    let step1 = g.add("step1", Op::Linear, &[v1, w1], vec![1, 64], DType::F32);
    let step2 = g.add(
        "step2",
        Op::Unary {
            kind: crate::graph::UnaryKind::GeluErf,
        },
        &[v2],
        vec![1, 64],
        DType::F32,
    );
    let step3 = g.add(
        "step3",
        Op::Binary {
            kind: crate::graph::BinaryKind::Add,
        },
        &[step1, step2],
        vec![1, 64],
        DType::F32,
    );

    // Later large allocation to exercise arena reuse
    let w_huge = g.weight("w_huge", vec![1024, 64], DType::F32);
    let huge_step = g.add(
        "huge_step",
        Op::Linear,
        &[step3, w_huge],
        vec![1, 1024],
        DType::F32,
    );

    // Second parent: NOT a graph output! Tests the alias-read lifetime extension directly
    let w_lin2 = g.weight("w_lin2", vec![1024, 64], DType::F32);
    let lin2 = g.add("lin2", Op::Linear, &[x, w_lin2], vec![1, 1024], DType::F32);
    let a = g.add(
        "a",
        Op::Slice {
            axis: -1,
            start: 0,
            end: 64,
        },
        &[lin2],
        vec![1, 64],
        DType::F32,
    );
    let b = g.add(
        "b",
        Op::Slice {
            axis: -1,
            start: 64,
            end: 128,
        },
        &[lin2],
        vec![1, 64],
        DType::F32,
    );

    // Intermediate allocation smaller than lin2: fits in lin2's arena space if lin2 is freed!
    let w_inter = g.weight("w_inter", vec![512, 64], DType::F32);
    let inter = g.add("inter", Op::Linear, &[x, w_inter], vec![1, 512], DType::F32);
    let inter_sub = g.add(
        "inter_sub",
        Op::Slice {
            axis: -1,
            start: 0,
            end: 64,
        },
        &[inter],
        vec![1, 64],
        DType::F32,
    );

    // Consume a and b AFTER the inter allocation
    let add_a = g.add(
        "add_a",
        Op::Binary {
            kind: crate::graph::BinaryKind::Add,
        },
        &[a, inter_sub],
        vec![1, 64],
        DType::F32,
    );
    let gelu_b = g.add(
        "gelu_b",
        Op::Unary {
            kind: crate::graph::UnaryKind::GeluErf,
        },
        &[b],
        vec![1, 64],
        DType::F32,
    );

    g.output(v0);
    g.output(v0_sub);
    g.output(huge_step);
    g.output(add_a);
    g.output(gelu_b);

    let mut w = Weights::new();
    w.insert("w_large", Tensor::F32(rand(192 * 64, 11)));
    w.insert("w1", Tensor::F32(rand(64 * 64, 22)));
    w.insert("w_huge", Tensor::F32(rand(1024 * 64, 33)));
    w.insert("w_lin2", Tensor::F32(rand(1024 * 64, 55)));
    w.insert("w_inter", Tensor::F32(rand(512 * 64, 66)));
    let inp = vec![Tensor::F32(rand(64, 44))];
    let want = cpu::run_stateless(&g, &w, inp.clone()).unwrap();
    let s = Session::new(rt(), &w).unwrap();
    let got = s.run(&g, inp).unwrap();
    assert_eq!(want.len(), got.len());
    for (i, (w_t, g_t)) in want.iter().zip(&got).enumerate() {
        let w_slice = w_t.as_f32().unwrap();
        let g_slice = g_t.as_f32().unwrap();
        let diff = rel_err(w_slice, g_slice);
        assert!(
            diff < 1e-4,
            "Output {i} mismatch: want {:?} vs got {:?}, diff = {diff}",
            &w_slice[..4],
            &g_slice[..4]
        );
    }
}
#[test]
fn test_measure_llama_decode_dispatches_per_layer() {
    use crate::arch::expr::Val;
    // The llama spec's decode instance on a synthetic 4-layer checkpoint.
    let num_layers = 4;
    let dir = tempfile::tempdir().unwrap();
    let a = crate::arch::testing::synthetic_llama(
        dir.path(),
        serde_json::json!({"num_hidden_layers": num_layers}),
    );
    let mut r = crate::arch::testing::Runner::from_artifacts(
        &a,
        Some("llama"),
        crate::hardware::Device::Metal,
        512,
    )
    .unwrap();
    let vars = [("rows", Val::Num(1.0)), ("head", Val::Str("last".into()))];
    let inst = r.instance("text", &vars).unwrap();
    let weights = r.model.store.materialize(&inst.weights).unwrap();
    let decode_g = inst.graph;

    let s = Session::new(rt(), &weights).unwrap();
    let plan = s.prepare_plan(&decode_g).unwrap();

    let total_steps = plan.steps.len();
    let non_layer_steps = 3;
    let layer_steps = total_steps.saturating_sub(non_layer_steps);
    let steps_per_layer = layer_steps as f64 / num_layers as f64;

    println!("=== llama spec decode dispatch count ===");
    println!("Total graph nodes: {}", decode_g.nodes.len());
    println!("Total GPU dispatches: {}", total_steps);
    println!(
        "Dispatches per layer: {:.1} ({} layers)",
        steps_per_layer, num_layers
    );
    for (i, step) in plan.steps.iter().enumerate() {
        println!("  Step {i:02}: {}", step.name);
    }
    assert!(
        steps_per_layer <= 10.0,
        "Dispatches per layer must be <= 10, got {steps_per_layer}"
    );
}

#[test]
fn test_stateful_cached_attention_parity() {
    for (kv_f16, tol) in KV_DTYPES {
        stateful_cached_attention_parity(kv_f16, tol);
    }
}

fn stateful_cached_attention_parity(kv_f16: bool, tol: f32) {
    let (heads, kv_heads, hd) = (4, 2, 8);
    let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
    let max_seq = 32;
    let scale = 1.0 / (hd as f32).sqrt();
    let window = Some(3);
    let soft_cap = Some(15.0);

    let make_graph = |m: usize| {
        let mut g = Graph::new();
        let q = g.input("q", vec![m, q_dim], DType::F32);
        let k = g.input("k", vec![m, kv_dim], DType::F32);
        let v = g.input("v", vec![m, kv_dim], DType::F32);
        let k_state = g.state("k_state", vec![max_seq, kv_dim], DType::F32, false);
        let v_state = g.state("v_state", vec![max_seq, kv_dim], DType::F32, false);
        let pos = g.input("pos", vec![m], DType::U32);
        let out = g.add(
            "cached_attn",
            Op::CachedAttention {
                heads,
                kv_heads,
                head_dim: hd,
                v_head_dim: hd,
                scale,
                causal: true,
                window,
                soft_cap,
                write: true,
            },
            &[q, k, v, k_state, v_state, pos],
            vec![m, q_dim],
            DType::F32,
        );
        g.output(out);
        g
    };

    let prefill_g = make_graph(5);
    let decode_g = make_graph(1);

    let mut cpu_state = StateStore::new();
    let mut metal_state = StateStore::new();
    let mut session = Session::new(rt(), &Weights::new()).unwrap();
    session.set_kv_f16(kv_f16);

    // 1. Prefill chunk of m = 5 (positions 0..5)
    {
        let q_data = rand(5 * q_dim, 101);
        let k_data = rand(5 * kv_dim, 202);
        let v_data = rand(5 * kv_dim, 303);
        let pos_data = vec![0u32, 1, 2, 3, 4];

        let inputs = vec![
            Tensor::F32(q_data),
            Tensor::F32(k_data),
            Tensor::F32(v_data),
            Tensor::U32(pos_data),
        ];

        let cpu_out =
            cpu::run(&prefill_g, &Weights::new(), &mut cpu_state, inputs.clone()).unwrap();
        let metal_out = session
            .run_with_state(&prefill_g, Some(&mut metal_state), inputs)
            .unwrap();

        let cpu_out_f32 = cpu_out[0].as_f32().unwrap();
        let metal_out_f32 = metal_out[0].as_f32().unwrap();
        assert!(
            rel_err(cpu_out_f32, metal_out_f32) < tol,
            "Prefill output mismatch: max rel err {}",
            rel_err(cpu_out_f32, metal_out_f32)
        );

        metal_state.sync_metal_to_cpu();
        let cpu_k = cpu_state.get("k_state").unwrap().as_f32().unwrap();
        let metal_k = metal_state.get("k_state").unwrap().as_f32().unwrap();
        assert!(
            rel_err(cpu_k, metal_k) < tol,
            "Prefill k_state mismatch: max rel err {}",
            rel_err(cpu_k, metal_k)
        );

        let cpu_v = cpu_state.get("v_state").unwrap().as_f32().unwrap();
        let metal_v = metal_state.get("v_state").unwrap().as_f32().unwrap();
        assert!(
            rel_err(cpu_v, metal_v) < tol,
            "Prefill v_state mismatch: max rel err {}",
            rel_err(cpu_v, metal_v)
        );
    }

    // 2. 3 decode steps (m = 1, positions 5, 6, 7)
    for step in 0..3 {
        let cur_pos = (5 + step) as u32;
        let q_data = rand(q_dim, 400 + step as u64);
        let k_data = rand(kv_dim, 500 + step as u64);
        let v_data = rand(kv_dim, 600 + step as u64);
        let pos_data = vec![cur_pos];

        let inputs = vec![
            Tensor::F32(q_data),
            Tensor::F32(k_data),
            Tensor::F32(v_data),
            Tensor::U32(pos_data),
        ];

        let cpu_out =
            cpu::run(&decode_g, &Weights::new(), &mut cpu_state, inputs.clone()).unwrap();
        let metal_out = session
            .run_with_state(&decode_g, Some(&mut metal_state), inputs)
            .unwrap();

        let cpu_out_f32 = cpu_out[0].as_f32().unwrap();
        let metal_out_f32 = metal_out[0].as_f32().unwrap();
        assert!(
            rel_err(cpu_out_f32, metal_out_f32) < tol,
            "Decode step {step} output mismatch: max rel err {}",
            rel_err(cpu_out_f32, metal_out_f32)
        );
        metal_state.sync_metal_to_cpu();

        let cpu_k = cpu_state.get("k_state").unwrap().as_f32().unwrap();
        let metal_k = metal_state.get("k_state").unwrap().as_f32().unwrap();
        assert!(
            rel_err(cpu_k, metal_k) < tol,
            "Decode step {step} k_state mismatch: max rel err {}",
            rel_err(cpu_k, metal_k)
        );

        let cpu_v = cpu_state.get("v_state").unwrap().as_f32().unwrap();
        let metal_v = metal_state.get("v_state").unwrap().as_f32().unwrap();
        assert!(
            rel_err(cpu_v, metal_v) < tol,
            "Decode step {step} v_state mismatch: max rel err {}",
            rel_err(cpu_v, metal_v)
        );
    }
}

/// (KV dtype is f16?, rel_err bound against the f32 CPU executor): f16 rounds each stored K/V to 11 bits
/// (<= 2^-11 relative, 2.4e-4 absolute on |x| <= 1 inputs).
const KV_DTYPES: [(bool, f32); 2] = [(false, 1e-4), (true, 1e-3)];

/// Smallest cosine between matching `hd`-wide head rows of `want` and `got`.
fn min_head_cosine(want: &[f32], got: &[f32], hd: usize) -> f32 {
    want.chunks(hd)
        .zip(got.chunks(hd))
        .map(|(a, b)| {
            let dot = |x: &[f32], y: &[f32]| {
                x.iter().zip(y).map(|(&p, &q)| (p * q) as f64).sum::<f64>()
            };
            (dot(a, b) / (dot(a, a).sqrt() * dot(b, b).sqrt()).max(1e-30)) as f32
        })
        .fold(1.0, f32::min)
}

/// Decode over a cache long enough to split across threadgroups (full and sliding window, soft cap,
/// grouped KV heads): the split + merge path equals the CPU executor, with f32 and f16 KV caches
/// (f16: every head row at cosine >= 0.9999).
#[test]
fn split_decode_attention_matches_cpu() {
    for (kv_f16, tol) in KV_DTYPES {
        split_decode_attention_matches_cpu_for(kv_f16, tol);
    }
}

fn split_decode_attention_matches_cpu_for(kv_f16: bool, tol: f32) {
    // (cache size, prefill length) pairs are looped below.
    // Query heads per KV head: 2 (a partial head tile), 8 (two full tiles), 1, 6 (tiles of 4 and 2),
    // 3 (a single tile of 3); head dims below 512 exercise the q-tile zero fill.
    for (heads, kv_heads, hd) in [
        (4usize, 2usize, 32usize),
        (8, 1, 64),
        (3, 3, 32),
        (6, 1, 32),
        (6, 2, 128),
        // E2B's sliding (hd 256) and global (hd 512) layers: the 8- and 16-value kernel variants.
        (8, 1, 256),
        (8, 1, 512),
    ] {
        // Short caches for every shape; for the two widest, a cache past the 4096-row minimum of the
        // physical state buffers: allocated large at prefill (4200), and grown with a copy mid-decode
        // (4060 prefill, the decode at +40 crosses 4096).
        for (max_seq, prefill) in [(1024usize, 700usize), (4400, 4200), (4400, 4060)] {
            if prefill > 1000 && !matches!(hd, 64 | 128) {
                continue;
            }
            let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
            for window in [None, Some(300), Some(90)] {
                let make = |m: usize| {
                    let mut g = Graph::new();
                    let q = g.input("q", vec![m, q_dim], DType::F32);
                    let k = g.input("k", vec![m, kv_dim], DType::F32);
                    let v = g.input("v", vec![m, kv_dim], DType::F32);
                    let ks = g.state("k_state", vec![max_seq, kv_dim], DType::F32, false);
                    let vs = g.state("v_state", vec![max_seq, kv_dim], DType::F32, false);
                    let pos = g.input("pos", vec![m], DType::U32);
                    let out = g.add(
                        "attn",
                        Op::CachedAttention {
                            heads,
                            kv_heads,
                            head_dim: hd,
                            v_head_dim: hd,
                            scale: 1.0 / (hd as f32).sqrt(),
                            causal: true,
                            window,
                            soft_cap: Some(30.0),
                            write: true,
                        },
                        &[q, k, v, ks, vs, pos],
                        vec![m, q_dim],
                        DType::F32,
                    );
                    g.output(out);
                    g
                };
                let (mut cpu_state, mut metal_state) = (StateStore::new(), StateStore::new());
                let mut session = Session::new(rt(), &Weights::new()).unwrap();
                session.set_kv_f16(kv_f16);
                let step = |g: &Graph,
                            m: usize,
                            at: usize,
                            seed: u64,
                            cs: &mut StateStore,
                            ms: &mut StateStore| {
                    let inputs = vec![
                        Tensor::F32(rand(m * q_dim, seed)),
                        Tensor::F32(rand(m * kv_dim, seed + 1)),
                        Tensor::F32(rand(m * kv_dim, seed + 2)),
                        Tensor::U32((at..at + m).map(|p| p as u32).collect()),
                    ];
                    let want = cpu::run(g, &Weights::new(), cs, inputs.clone()).unwrap();
                    let got = session.run_with_state(g, Some(ms), inputs).unwrap();
                    let (want, got) = (want[0].as_f32().unwrap(), got[0].as_f32().unwrap());
                    let cos = min_head_cosine(want, got, hd);
                    assert!(
                        cos >= 0.9999,
                        "f16 KV {kv_f16} at {at} {window:?}: min cosine {cos}"
                    );
                    rel_err(want, got)
                };
                let e = step(
                    &make(prefill),
                    prefill,
                    0,
                    1,
                    &mut cpu_state,
                    &mut metal_state,
                );
                let prefill_tol = if hd > 32 { tol.max(2e-3) } else { tol };
                assert!(e <= prefill_tol, "f16 KV {kv_f16} prefill {window:?}: {e}");
                let (kbuf, n) = &metal_state.metal_buffers["k_state"];
                assert_eq!(metal_state.metal_f16.contains("k_state"), kv_f16);
                assert_eq!(
                    kbuf.length() as usize,
                    (n * if kv_f16 { 2 } else { 4 }).max(ARENA_ALIGN)
                );
                let decode = make(1);
                // 700..: more than 128 keys visible, so the sliding windows of 300 and 90 straddle the cut-off.
                for (i, at) in [prefill, prefill + 1, prefill + 40].into_iter().enumerate() {
                    let e = step(
                        &decode,
                        1,
                        at,
                        10 + 3 * i as u64,
                        &mut cpu_state,
                        &mut metal_state,
                    );
                    assert!(e < tol, "f16 KV {kv_f16} decode at {at} {window:?}: {e}");
                }
            }
        }
    }
}

/// E1 probe: a chain of dependent f16 GEMVs (each reads 1536x1536 weights, feeding the next) as
/// separate dispatches versus one persistent kernel with an atomic grid barrier between steps.
/// Prints per-step time for both and checks the results agree
/// (`cargo test --release --lib persistent_chain_probe -- --ignored --nocapture`).
#[test]
#[ignore = "measurement"]
fn persistent_chain_probe() {
    const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
inline float row_dot(device const half* w, device const float* x, uint cols, uint lane) {
float acc = 0.0f;
for (uint c = lane * 4; c < cols; c += 128) {
    acc += dot(float4(*(device const half4*)(w + c)), *(device const float4*)(x + c));
}
return simd_sum(acc);
}
kernel void sep(device const half* w [[buffer(0)]], device const float* x [[buffer(1)]],
            device float* y [[buffer(2)]], constant uint2& p [[buffer(3)]],
            uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
            uint lane [[thread_index_in_simdgroup]]) {
uint row = tg * 8 + sg;
if (row >= p.x) return;
float s = row_dot(w + (ulong)row * p.y, x, p.y, lane);
if (lane == 0) y[row] = s;
}
kernel void persist(device const half* w [[buffer(0)]], device float* xa [[buffer(1)]],
                device float* xb [[buffer(2)]], device atomic_uint* ctr [[buffer(3)]],
                constant uint4& p [[buffer(4)]],
                uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
for (uint s = 0; s < p.z; s++) {
    device const half* ws = w + (ulong)s * p.x * p.y;
    device float* x = (s & 1) ? xb : xa;
    device float* y = (s & 1) ? xa : xb;
    for (uint row = tg * 8 + sg; row < p.x; row += p.w * 8) {
        float v = row_dot(ws + (ulong)row * p.y, x, p.y, lane);
        if (lane == 0) y[row] = v;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
        // Spin cap: a co-residency mistake ends the kernel (wrong results) instead of hanging the GPU.
        for (uint spin = 0; spin < 20000000u && atomic_load_explicit(ctr, memory_order_relaxed) < (s + 1) * p.w; spin++) {}
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_device);
}
}
"#;
    let rt = rt();
    let dev = rt.device.clone();
    let lib = dev
        .new_library_with_source(SRC, &metal::CompileOptions::new())
        .unwrap();
    let pso = |n: &str| {
        dev.new_compute_pipeline_state_with_function(&lib.get_function(n, None).unwrap())
            .unwrap()
    };
    let (sep, persist) = (pso("sep"), pso("persist"));
    let n = std::env::var("PROBE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1536usize);
    let steps = 64usize;
    let w = new_shared_buffer(&dev, steps * n * n * 2).unwrap();
    unsafe {
        let p = w.contents() as *mut u16;
        // Random f16 weights, variance 1/n: the chain keeps O(1) magnitudes and every row differs.
        let mut s = 0x9E3779B97F4A7C15u64;
        let amp = 3f32.sqrt() / (n as f32).sqrt();
        for i in 0..steps * n * n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0;
            *p.add(i) = crate::model::f32_to_f16(u * amp);
        }
    }
    let xa = new_shared_buffer(&dev, n * 4).unwrap();
    let xb = new_shared_buffer(&dev, n * 4).unwrap();
    let ctr = new_shared_buffer(&dev, 256).unwrap();
    let init = |b: &Buffer| unsafe {
        for i in 0..n {
            *(b.contents() as *mut f32).add(i) = ((i % 7) as f32 - 3.0) * 0.1;
        }
    };
    let q = dev.new_command_queue();
    let read = |b: &Buffer| -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n).to_vec() }
    };
    let run_sep = || -> f64 {
        init(&xa);
        let cb = q.new_command_buffer();
        let raw = cb.new_compute_command_encoder();
        let enc = &Counted::new(raw);
        enc.set_compute_pipeline_state(&sep);
        for s in 0..steps {
            let (x, y) = if s % 2 == 0 { (&xa, &xb) } else { (&xb, &xa) };
            enc.set_buffer(0, Some(&w), (s * n * n * 2) as u64);
            enc.set_buffer(1, Some(x), 0);
            enc.set_buffer(2, Some(y), 0);
            set_bytes(enc, 3, &[n as u32, n as u32]);
            enc.dispatch_thread_groups(
                MTLSize::new((n as u64).div_ceil(8), 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }
        enc.end_encoding();
        let t = Instant::now();
        commit_and_wait(cb).unwrap();
        t.elapsed().as_secs_f64()
    };
    let run_persist = |tgs: u64| -> f64 {
        init(&xa);
        unsafe { *(ctr.contents() as *mut u32) = 0 };
        let cb = q.new_command_buffer();
        let raw = cb.new_compute_command_encoder();
        let enc = &Counted::new(raw);
        enc.set_compute_pipeline_state(&persist);
        enc.set_buffer(0, Some(&w), 0);
        enc.set_buffer(1, Some(&xa), 0);
        enc.set_buffer(2, Some(&xb), 0);
        enc.set_buffer(3, Some(&ctr), 0);
        set_bytes(enc, 4, &[n as u32, n as u32, steps as u32, tgs as u32]);
        enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        let t = Instant::now();
        commit_and_wait(cb).unwrap();
        t.elapsed().as_secs_f64()
    };
    let bytes = (n * n * 2) as f64;
    run_sep();
    let t_sep = (0..5).map(|_| run_sep()).fold(f64::MAX, f64::min);
    let want = read(if steps % 2 == 0 { &xa } else { &xb });
    let mag = want.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(mag > 1e-2, "degenerate chain: max |y| {mag}");
    println!(
        "n {n}: separate {:.1} us/step = {:.0} GB/s (max |y| {mag:.2})",
        t_sep / steps as f64 * 1e6,
        bytes * steps as f64 / t_sep / 1e9
    );
    for tgs in [20u64, 40, 80, 120] {
        run_persist(tgs);
        let t = (0..5).map(|_| run_persist(tgs)).fold(f64::MAX, f64::min);
        let got = read(if steps % 2 == 0 { &xa } else { &xb });
        let err = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "persistent {tgs:>3} TGs: {:.1} us/step = {:.0} GB/s, max |diff| {err:e}",
            t / steps as f64 * 1e6,
            bytes * steps as f64 / t / 1e9
        );
    }
}

/// Decode attention GPU time over a long cache for the Gemma-4 layer shapes (8 query heads on one KV
/// head: full attention hd 512, sliding hd 256 with a 512 window)
/// (`cargo test --release --lib attention_decode_rates -- --ignored --nocapture`).
#[test]
#[ignore = "bandwidth measurement"]
fn attention_decode_rates() {
    for (heads, kv_heads, hd, window, at) in [
        (8usize, 1usize, 512usize, None, 4096usize),
        (8, 1, 256, Some(512usize), 4096),
        (8, 4, 128, None, 4096),
    ] {
        let (q_dim, kv_dim, max_seq) = (heads * hd, kv_heads * hd, 8192);
        let mut g = Graph::new();
        let q = g.input("q", vec![1, q_dim], DType::F32);
        let k = g.input("k", vec![1, kv_dim], DType::F32);
        let v = g.input("v", vec![1, kv_dim], DType::F32);
        let ks = g.state("k_state", vec![max_seq, kv_dim], DType::F32, false);
        let vs = g.state("v_state", vec![max_seq, kv_dim], DType::F32, false);
        let pos = g.input("pos", vec![1], DType::U32);
        let out = g.add(
            "attn",
            Op::CachedAttention {
                heads,
                kv_heads,
                head_dim: hd,
                v_head_dim: hd,
                scale: 1.0,
                causal: true,
                window,
                soft_cap: None,
                write: true,
            },
            &[q, k, v, ks, vs, pos],
            vec![1, q_dim],
            DType::F32,
        );
        g.output(out);
        let session = Session::new(rt(), &Weights::new()).unwrap();
        let mut state = StateStore::new();
        let mut best = f64::INFINITY;
        for i in 0..12 {
            let inputs = vec![
                Tensor::F32(rand(q_dim, 1 + i)),
                Tensor::F32(rand(kv_dim, 2)),
                Tensor::F32(rand(kv_dim, 3)),
                Tensor::U32(vec![at as u32]),
            ];
            session
                .run_with_state(&g, Some(&mut state), inputs)
                .unwrap();
            if i >= 2 {
                best = best.min(session.last_gpu_seconds());
            }
        }
        let keys = window.map_or(at + 1, |w| w.min(at + 1));
        let bytes = (keys * kv_dim * 2 * 4) as f64;
        println!(
            "attention heads {heads} kv {kv_heads} hd {hd} window {window:?} keys {keys}: {:.0} us, {:.0} GB/s of KV",
            best * 1e6,
            bytes / best / 1e9
        );
    }
}

/// A bidirectional block: rows inside it see keys up to the block's end (`visible_end`), the sliding
/// window still starting at the row's own position. Metal against the CPU executor.
#[test]
fn cached_attention_visible_end_matches_cpu() {
    for (heads, kv_heads, hd, m, win) in [
        (4, 2, 8, 6, Some(4)),
        (8, 1, 64, 8, Some(128)),
        (8, 1, 128, 33, Some(128)),
        (2, 2, 256, 33, None),
        (8, 1, 512, 8, None),
    ] {
        let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
        let mut g = Graph::new();
        let q = g.input("q", vec![m, q_dim], DType::F32);
        let k = g.input("k", vec![m, kv_dim], DType::F32);
        let v = g.input("v", vec![m, kv_dim], DType::F32);
        let ks = g.state("k_state", vec![m + 8, kv_dim], DType::F32, false);
        let vs = g.state("v_state", vec![m + 8, kv_dim], DType::F32, false);
        let pos = g.input("pos", vec![m], DType::U32);
        let vis = g.input("vis", vec![m], DType::U32);
        let out = g.add(
            "attn",
            Op::CachedAttention {
                heads,
                kv_heads,
                head_dim: hd,
                v_head_dim: hd,
                scale: 1.0 / (hd as f32).sqrt(),
                causal: true,
                window: win,
                soft_cap: Some(50.0),
                write: true,
            },
            &[q, k, v, ks, vs, pos, vis],
            vec![m, q_dim],
            DType::F32,
        );
        g.output(out);
        let vis_vec: Vec<u32> = (0..m)
            .map(|i| if i >= m / 2 { m as u32 } else { (i + 1) as u32 })
            .collect();
        let inputs = vec![
            Tensor::F32(rand(m * q_dim, 11)),
            Tensor::F32(rand(m * kv_dim, 22)),
            Tensor::F32(rand(m * kv_dim, 33)),
            Tensor::U32((0..m as u32).collect()),
            Tensor::U32(vis_vec),
        ];
        let want = cpu::run(&g, &Weights::new(), &mut StateStore::new(), inputs.clone()).unwrap();
        for (kv_f16, tol) in KV_DTYPES {
            let mut session = Session::new(rt(), &Weights::new()).unwrap();
            session.set_kv_f16(kv_f16);
            let got = session
                .run_with_state(&g, Some(&mut StateStore::new()), inputs.clone())
                .unwrap();
            let err = rel_err(want[0].as_f32().unwrap(), got[0].as_f32().unwrap());
            let max_tol = if m >= 8 && hd > 64 { tol.max(2e-3) } else { tol };
            assert!(
                err <= max_tol,
                "visible_end mismatch m={m} hd={hd} (f16 KV {kv_f16}): {err}"
            );
        }
    }

    // Full prefill matrix over m in {8, 33, 512} x hd in {64, 128, 256, 512}, GQA 8:1 and 1:1,
    // causal, window 128, soft-cap 50 and vis block.
    for m in [8usize, 33, 512] {
        for hd in [64usize, 128, 256, 512] {
            for (heads, kv_heads) in [(8usize, 1usize), (2, 2)] {
                for (window, soft_cap, has_vis) in [
                    (None, None, false),
                    (Some(128), None, false),
                    (None, Some(50.0), false),
                    (Some(128), Some(50.0), true),
                ] {
                    let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
                    let mut g = Graph::new();
                    let q = g.input("q", vec![m, q_dim], DType::F32);
                    let k = g.input("k", vec![m, kv_dim], DType::F32);
                    let v = g.input("v", vec![m, kv_dim], DType::F32);
                    let ks = g.state("k_state", vec![m + 8, kv_dim], DType::F32, false);
                    let vs = g.state("v_state", vec![m + 8, kv_dim], DType::F32, false);
                    let pos = g.input("pos", vec![m], DType::U32);
                    let mut ins = vec![q, k, v, ks, vs, pos];
                    if has_vis {
                        ins.push(g.input("vis", vec![m], DType::U32));
                    }
                    let out = g.add(
                        "attn",
                        Op::CachedAttention {
                            heads,
                            kv_heads,
                            head_dim: hd,
                            v_head_dim: hd,
                            scale: 1.0 / (hd as f32).sqrt(),
                            causal: true,
                            window,
                            soft_cap,
                            write: true,
                        },
                        &ins,
                        vec![m, q_dim],
                        DType::F32,
                    );
                    g.output(out);
                    let mut inputs = vec![
                        Tensor::F32(rand(m * q_dim, 41)),
                        Tensor::F32(rand(m * kv_dim, 42)),
                        Tensor::F32(rand(m * kv_dim, 43)),
                        Tensor::U32((0..m as u32).collect()),
                    ];
                    if has_vis {
                        let vis_vec: Vec<u32> = (0..m)
                            .map(|i| if i >= m / 2 { m as u32 } else { (i + 1) as u32 })
                            .collect();
                        inputs.push(Tensor::U32(vis_vec));
                    }
                    let want =
                        cpu::run(&g, &Weights::new(), &mut StateStore::new(), inputs.clone())
                            .unwrap();
                    let session = Session::new(rt(), &Weights::new()).unwrap();
                    let got = session
                        .run_with_state(&g, Some(&mut StateStore::new()), inputs)
                        .unwrap();
                    let err = rel_err(want[0].as_f32().unwrap(), got[0].as_f32().unwrap());
                    assert!(
                        err <= 2e-3,
                        "cached_attn_mm m={m} hd={hd} {heads}:{kv_heads} win={window:?} cap={soft_cap:?} vis={has_vis}: {err:e}"
                    );
                }
            }
        }
    }
}

/// `m` rows of cached attention over `[max_seq, kv_heads * hd]` k/v states named `k_state` / `v_state`.
fn kv_attn_graph(
    heads: usize,
    kv_heads: usize,
    hd: usize,
    window: Option<usize>,
    max_seq: usize,
    m: usize,
) -> Graph {
    let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
    let mut g = Graph::new();
    let q = g.input("q", vec![m, q_dim], DType::F32);
    let k = g.input("k", vec![m, kv_dim], DType::F32);
    let v = g.input("v", vec![m, kv_dim], DType::F32);
    let ks = g.state("k_state", vec![max_seq, kv_dim], DType::F32, false);
    let vs = g.state("v_state", vec![max_seq, kv_dim], DType::F32, false);
    let pos = g.input("pos", vec![m], DType::U32);
    let out = g.add(
        "attn",
        Op::CachedAttention {
            heads,
            kv_heads,
            head_dim: hd,
            v_head_dim: hd,
            scale: 1.0 / (hd as f32).sqrt(),
            causal: true,
            window,
            soft_cap: None,
            write: true,
        },
        &[q, k, v, ks, vs, pos],
        vec![m, q_dim],
        DType::F32,
    );
    g.output(out);
    g
}

/// f16 KV caches against the f32 CPU executor: prefills of 1, 63, 64, 65 and 4096 rows then two decode
/// steps (the last prefill's first decode crosses the 4096-row physical buffer: grown by a byte copy and
/// split across threadgroups), grouped / ungrouped KV heads and a sliding window. The stored caches are
/// exactly the round-to-nearest-even f16 of what the CPU executor stored (prefill via `kv_append`,
/// decode via the split kernel's fused write).
#[test]
fn f16_kv_attention_matches_cpu_across_lengths() {
    use crate::model::{f16_to_f32, f32_to_f16};
    let max_seq = 4400;
    for (heads, kv_heads, hd, window) in [
        (4usize, 2usize, 32usize, None),
        (4, 4, 32, None),
        (8, 1, 64, Some(40usize)),
    ] {
        let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
        for prefill in [1usize, 63, 64, 65, 4096] {
            let mut session = Session::new(rt(), &Weights::new()).unwrap();
            session.set_kv_f16(true);
            let (mut cs, mut ms) = (StateStore::new(), StateStore::new());
            let mut at = 0;
            for (i, m) in [prefill, 1, 1].into_iter().enumerate() {
                let g = kv_attn_graph(heads, kv_heads, hd, window, max_seq, m);
                let seed = 100 * i as u64 + 7;
                let inputs = vec![
                    Tensor::F32(rand(m * q_dim, seed)),
                    Tensor::F32(rand(m * kv_dim, seed + 1)),
                    Tensor::F32(rand(m * kv_dim, seed + 2)),
                    Tensor::U32((at..at + m).map(|p| p as u32).collect()),
                ];
                let want = cpu::run(&g, &Weights::new(), &mut cs, inputs.clone()).unwrap();
                let got = session.run_with_state(&g, Some(&mut ms), inputs).unwrap();
                let (want, got) = (want[0].as_f32().unwrap(), got[0].as_f32().unwrap());
                let (e, cos) = (rel_err(want, got), min_head_cosine(want, got, hd));
                assert!(
                    e < 1e-3 && cos >= 0.9999,
                    "heads {heads}/{kv_heads} hd {hd} {window:?} prefill {prefill} chunk {i}: {e}, cos {cos}"
                );
                at += m;
            }
            assert!(ms.metal_f16.contains("k_state") && ms.metal_f16.contains("v_state"));
            let n_buf = ms.metal_buffers["k_state"].1;
            assert_eq!(ms.metal_buffers["k_state"].0.length() as usize, n_buf * 2);
            ms.sync_metal_to_cpu();
            for name in ["k_state", "v_state"] {
                let want = &cs.get(name).unwrap().as_f32().unwrap()[..at * kv_dim];
                let got = &ms.get(name).unwrap().as_f32().unwrap()[..at * kv_dim];
                for (i, (&w, &g)) in want.iter().zip(got).enumerate() {
                    assert_eq!(
                        g,
                        f16_to_f32(f32_to_f16(w)),
                        "{name}[{i}] of {at} positions: stored {g}, f32 {w}"
                    );
                }
            }
        }
    }
}

/// Out-of-range K/V saturate at +-65504 instead of becoming inf; NaN stays NaN; tiny values flush to 0.
/// Covers `kv_append_batched_f16` (m = 2 prefill) and the split kernel's fused f16 write (m = 1).
#[test]
fn f16_kv_saturates_out_of_range_values() {
    let k = [1e6, -1e6, 70000.0, f32::NAN];
    let v = [65504.0, 65520.0, 1e-9, -2.5];
    for m in [1usize, 2] {
        let g = kv_attn_graph(1, 1, 4, None, 16, m);
        let mut session = Session::new(rt(), &Weights::new()).unwrap();
        session.set_kv_f16(true);
        let mut st = StateStore::new();
        let inputs = vec![
            Tensor::F32(vec![1.0; 4 * m]),
            Tensor::F32(k.repeat(m)),
            Tensor::F32(v.repeat(m)),
            Tensor::U32((0..m as u32).collect()),
        ];
        session.run_with_state(&g, Some(&mut st), inputs).unwrap();
        let (ks, vs) = (
            st.read_metal("k_state").unwrap(),
            st.read_metal("v_state").unwrap(),
        );
        for row in 0..m {
            let (kr, vr) = (&ks[row * 4..row * 4 + 4], &vs[row * 4..row * 4 + 4]);
            assert_eq!(kr[..3], [65504.0, -65504.0, 65504.0], "m {m} row {row}");
            assert!(kr[3].is_nan(), "m {m} row {row}");
            assert_eq!(vr, [65504.0, 65504.0, 0.0, -2.5], "m {m} row {row}");
        }
    }
}

/// CPU-only: a state is f16 only when every reader is a `CachedAttention` k/v operand. Layers sharing
/// a cache stay f16; a plain `Attention` over the cache keeps it f32.
#[test]
fn kv_dtype_follows_the_states_readers() {
    let (heads, hd, max_seq) = (2, 8, 16);
    let f = DType::F32;
    let attn = |write| Op::CachedAttention {
        heads,
        kv_heads: heads,
        head_dim: hd,
        v_head_dim: hd,
        scale: 0.3,
        causal: true,
        window: None,
        soft_cap: None,
        write,
    };
    let build = |shared: bool, plain_reader: bool| {
        let mut g = Graph::new();
        let q = g.input("q", vec![1, heads * hd], f);
        let k = g.input("k", vec![1, heads * hd], f);
        let v = g.input("v", vec![1, heads * hd], f);
        let pos = g.input("pos", vec![1], DType::U32);
        let ks = g.state("k_state", vec![max_seq, heads * hd], f, false);
        let vs = g.state("v_state", vec![max_seq, heads * hd], f, false);
        let a = g.add(
            "a",
            attn(true),
            &[q, k, v, ks, vs, pos],
            vec![1, heads * hd],
            f,
        );
        g.output(a);
        if shared {
            let (ks2, vs2) = (
                g.state("k_state", vec![max_seq, heads * hd], f, false),
                g.state("v_state", vec![max_seq, heads * hd], f, false),
            );
            let b = g.add(
                "b",
                attn(false),
                &[a, k, v, ks2, vs2, pos],
                vec![1, heads * hd],
                f,
            );
            g.output(b);
        }
        if plain_reader {
            let plain = Op::Attention {
                heads,
                kv_heads: heads,
                causal: false,
                window: None,
                soft_cap: None,
                scale: None,
            };
            let c = g.add("c", plain, &[q, ks, vs], vec![1, heads * hd], f);
            g.output(c);
        }
        g
    };
    let assign = |g: &Graph| {
        let (mut steps, _) = plan(g).unwrap();
        let f16 = assign_kv_dtypes(g, &mut steps);
        let flags: Vec<bool> = steps
            .iter()
            .filter_map(|s| match s.op {
                Step::CachedAttention { kv_f16, .. } => Some(kv_f16),
                _ => None,
            })
            .collect();
        (f16.len(), flags)
    };
    assert_eq!(assign(&build(false, false)), (2, vec![true]));
    assert_eq!(assign(&build(true, false)), (2, vec![true, true]));
    assert_eq!(assign(&build(false, true)), (0, vec![false]));
    assert_eq!(assign(&build(true, true)), (0, vec![false, false]));
}

#[test]
fn kernel_rows_group_steps_by_innermost_kind_slowest_first() {
    assert_eq!(kernel_kind("Ext(Norm(RmsNorm { eps: 1e-6 }))"), "RmsNorm");
    assert_eq!(kernel_kind("GemvPacked { rows: 8 }"), "GemvPacked");
    assert_eq!(kernel_kind("?"), "?");
    // Field values never name the kind: `ValueId(..)`, `Some(ValueId(..))` and `Gen(..)` follow it.
    assert_eq!(
        kernel_kind("GemmBatchedPacked { x: ValueId(1), res: Some(ValueId(4)), format: Gen(GGUF_Q4_K_M) }"),
        "GemmBatchedPacked"
    );
    assert_eq!(kernel_kind("Ext(Moe(Embed))"), "Embed");
    assert_eq!(kernel_kind("Softmax(3)"), "Softmax");
    let t = |name: &str, desc: &str, s: f64, bytes: u64| {
        (name.to_string(), desc.to_string(), s, bytes)
    };
    let rows = kernel_rows(
        &[
            t("a", "Ext(Norm(RmsNorm { eps: 1e-6 }))", 1e-6, 1_000),
            t("b", "GemvPacked { rows: 8 }", 8e-6, 2_000_000),
            t("c", "Ext(Norm(RmsNorm { eps: 1e-6 }))", 1e-6, 1_000),
        ],
        100.0,
    );
    let kinds: Vec<_> = rows.iter().map(|r| (r.kind.as_str(), r.steps)).collect();
    assert_eq!(kinds, [("GemvPacked", 1), ("RmsNorm", 2)]);
    assert!((rows[0].us - 8.0).abs() < 1e-9);
    // 2 MB in 8 µs = 250 GB/s = 250% of a 100 GB/s ceiling.
    assert!((rows[0].gbps - 250.0).abs() < 1e-6 && (rows[0].pct - 250.0).abs() < 1e-6);
}

/// A requested profile times one real decode token step by step: the token's output is the
/// one an unprofiled run gives, and the per-kind rows land in `last_kernel_profile`.
#[test]
fn requested_kernel_profile_samples_a_decode_token_and_keeps_its_output() {
    let (heads, kv_heads, hd) = (4, 2, 8);
    let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
    let mut g = Graph::new();
    let q = g.input("q", vec![1, q_dim], DType::F32);
    let k = g.input("k", vec![1, kv_dim], DType::F32);
    let v = g.input("v", vec![1, kv_dim], DType::F32);
    let ks = g.state("k_state", vec![16, kv_dim], DType::F32, false);
    let vs = g.state("v_state", vec![16, kv_dim], DType::F32, false);
    let pos = g.input("pos", vec![1], DType::U32);
    let out = g.add(
        "cached_attn",
        Op::CachedAttention {
            heads,
            kv_heads,
            head_dim: hd,
            v_head_dim: hd,
            scale: 1.0 / (hd as f32).sqrt(),
            causal: true,
            window: None,
            soft_cap: None,
            write: true,
        },
        &[q, k, v, ks, vs, pos],
        vec![1, q_dim],
        DType::F32,
    );
    g.output(out);
    let inputs = || {
        vec![
            Tensor::F32(rand(q_dim, 1)),
            Tensor::F32(rand(kv_dim, 2)),
            Tensor::F32(rand(kv_dim, 3)),
            Tensor::U32(vec![0]),
        ]
    };
    let session = Session::new(rt(), &Weights::new()).unwrap();
    let want = session
        .run_with_state(&g, Some(&mut StateStore::new()), inputs())
        .unwrap();
    // Another test's decode run may take the one-shot request: ask again until ours is sampled.
    let start = Instant::now();
    let mut got = None;
    for _ in 0..100 {
        request_kernel_profile();
        let out = session
            .run_with_state(&g, Some(&mut StateStore::new()), inputs())
            .unwrap();
        if last_kernel_profile().is_some_and(|(at, _)| at >= start) {
            got = Some(out);
            break;
        }
    }
    let got = got.expect("a requested decode token is profiled");
    assert!(rel_err(want[0].as_f32().unwrap(), got[0].as_f32().unwrap()) < 1e-6);
    let (_, rows) = last_kernel_profile().unwrap();
    assert!(rows.iter().map(|r| r.steps).sum::<usize>() >= 1, "{rows:?}");
    assert!(
        rows.iter().all(|r| r.us >= 0.0 && r.gbps >= 0.0),
        "{rows:?}"
    );
}


#[test]
fn moe_subgraph_fusions_match_cpu() {
    use crate::graph::ReduceKind;

    for m in [1usize, 2] {
        let (h, ff, experts, top_k) = (16, 16, 6, 3);
        let f = DType::F32;

        // 1. MoE subgraph starting from router logits (12 raw steps -> 5 fused steps).
        let mut g = Graph::new();
        let x = g.input("x", vec![m, h], f);
        let logits = g.input("logits", vec![m, experts], f);
        let scale_w = g.weight("scale", vec![1, experts], f);
        let gu_w = g.weight("gate_up", vec![experts * 2 * ff, h], f);
        let down_w = g.weight("down", vec![experts * h, ff], f);

        let ids = g.add(
            "topk",
            Op::TopK { k: top_k },
            &[logits],
            vec![m, top_k],
            DType::U32,
        );
        let picked = g.add("picked", Op::GatherCols, &[logits, ids], vec![m, top_k], f);
        let probs = g.add(
            "probs",
            Op::Softmax { axis: -1 },
            &[picked],
            vec![m, top_k],
            f,
        );
        let sc = g.add("sc", Op::GatherCols, &[scale_w, ids], vec![m, top_k], f);
        let weights_v = g.add(
            "weights",
            Op::Binary {
                kind: crate::graph::BinaryKind::Mul,
            },
            &[probs, sc],
            vec![m, top_k],
            f,
        );
        let gu = g.add(
            "gu",
            Op::MoeLinear { experts, top_k },
            &[x, gu_w, ids],
            vec![m * top_k, 2 * ff],
            f,
        );
        let gate = g.add(
            "gate",
            Op::Slice {
                axis: 1,
                start: 0,
                end: ff as i64,
            },
            &[gu],
            vec![m * top_k, ff],
            f,
        );
        let up = g.add(
            "up",
            Op::Slice {
                axis: 1,
                start: ff as i64,
                end: (2 * ff) as i64,
            },
            &[gu],
            vec![m * top_k, ff],
            f,
        );
        let act = g.add("act", Op::GeGlu, &[gate, up], vec![m * top_k, ff], f);
        let down = g.add(
            "down",
            Op::MoeLinear { experts, top_k },
            &[act, down_w, ids],
            vec![m * top_k, h],
            f,
        );
        let w_col = g.add(
            "w_col",
            Op::Reshape {
                shape: Some(vec![m * top_k, 1]),
            },
            &[weights_v],
            vec![m * top_k, 1],
            f,
        );
        let weighted = g.add(
            "weighted",
            Op::Binary {
                kind: crate::graph::BinaryKind::Mul,
            },
            &[down, w_col],
            vec![m * top_k, h],
            f,
        );
        let slots = g.add(
            "slots",
            Op::Reshape {
                shape: Some(vec![m, top_k, h]),
            },
            &[weighted],
            vec![m, top_k, h],
            f,
        );
        let out = g.add(
            "out",
            Op::Reduce {
                kind: ReduceKind::Sum,
                axes: vec![1],
                keep_dims: false,
            },
            &[slots],
            vec![m, h],
            f,
        );
        g.output(out);

        let mut w = Weights::new();
        w.insert("scale", Tensor::F32(rand(experts, 11)));
        w.insert("gate_up", Tensor::F32(rand(experts * 2 * ff * h, 22)));
        w.insert("down", Tensor::F32(rand(experts * h * ff, 33)));

        let sess = Session::new(rt(), &w).unwrap();
        let prep = sess.prepare_plan(&g).unwrap();
        assert_eq!(prep.steps.len(), 5, "12 raw MoE steps fuse to 5");
        assert!(matches!(
            prep.steps[0].op,
            Step::Ext(ext::ExtStep::Experts(ext::moe::Step::TopK {
                weights_out: Some(_),
                scale: Some(_),
                ..
            }))
        ));
        assert!(matches!(prep.steps[2].op, Step::SplitGlu { kind: 1, .. }));
        assert!(matches!(prep.steps[4].op, Step::MoeWeightedSum { .. }));

        let inputs = vec![
            Tensor::F32(rand(m * h, 44)),
            Tensor::F32(rand(m * experts, 55)),
        ];
        let want = cpu::run_stateless(&g, &w, inputs.clone())
            .unwrap()
            .remove(0)
            .into_f32()
            .unwrap();
        let got = sess
            .run_prepared(&g, &prep, None, inputs)
            .unwrap()
            .remove(0)
            .into_f32()
            .unwrap();
        assert!(
            rel_err(&want, &got) < 1e-5,
            "rel_err = {}",
            rel_err(&want, &got)
        );
    }
}

#[test]
fn gemv_packed_pair_small_rows_matches_cpu() {
    use crate::model::layers::gguf_block_codec;
    for f in [
        QuantFormat::GGUF_Q4_K_M,
        QuantFormat::GGUF_Q8_0,
        QuantFormat::GGUF_Q4_0,
    ] {
        for (rows0, rows1, cols) in [(256usize, 256usize, 512usize), (16, 16, 2048)] {
            let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ (f as u64) ^ (cols as u64);
            let mut next = move || {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (seed >> 33) as u32
            };
            let (be, bb, _) = gguf_block_codec(f).expect("CPU dequantizer");
            let scale_at: &[usize] = match f {
                QuantFormat::GGUF_Q4_K_M => &[0, 2],
                _ => &[0],
            };
            let mut w = Weights::new();
            let mut g = Graph::new();
            let mut quant = |g: &mut Graph, w: &mut Weights, name: &str, n: usize, k: usize| {
                let mut raw: Vec<u8> = (0..n * (k / be) * bb)
                    .map(|_| (next() >> 4) as u8)
                    .collect();
                for blk in raw.chunks_exact_mut(bb) {
                    for &o in scale_at {
                        let exp = 9 + (next() % 14) as u16;
                        let d =
                            ((next() & 1) as u16) << 15 | exp << 10 | (next() & 0x3FF) as u16;
                        blk[o..o + 2].copy_from_slice(&d.to_le_bytes());
                    }
                }
                w.insert(
                    name,
                    Tensor::Mat(crate::model::weights::Mat::from_packed_bytes(f, raw, n, k)),
                );
                g.weight(name, vec![n, k], DType::Quant(f))
            };
            let x = g.input("x", vec![1, cols], DType::F32);
            let kw = quant(&mut g, &mut w, "k_w", rows0, cols);
            let vw = quant(&mut g, &mut w, "v_w", rows1, cols);
            let kb = g.weight("k_b", vec![rows0], DType::F32);
            w.insert("k_b", Tensor::F32(rand(rows0, 11)));
            let k_out = g.add(
                "k_proj",
                Op::Linear,
                &[x, kw, kb],
                vec![1, rows0],
                DType::F32,
            );
            let v_out = g.add("v_proj", Op::Linear, &[x, vw], vec![1, rows1], DType::F32);
            g.output(k_out);
            g.output(v_out);

            let mut s = Session::new(rt(), &w).unwrap();
            s.set_variant_override(Some(GemvVariant::HAND));
            let plan = s.prepare_plan(&g).unwrap();
            assert_eq!(plan.steps.len(), 1);
            assert!(matches!(plan.steps[0].op, Step::GemvPackedPair { .. }));

            let inp = vec![Tensor::F32(rand(cols, 42))];
            let mut cs = StateStore::new();
            let want = cpu::run(&g, &w, &mut cs, inp.clone()).unwrap();
            let got = s.run_prepared(&g, &plan, None, inp).unwrap();
            for (i, (wt, gt)) in want.iter().zip(&got).enumerate() {
                let err = rel_err(wt.as_f32().unwrap(), gt.as_f32().unwrap());
                assert!(
                    err < 1e-5,
                    "{f:?} ({rows0},{rows1},{cols}) out {i}: err {err}"
                );
            }
            if f == QuantFormat::GGUF_Q4_K_M && (rows0, rows1, cols) == (16, 16, 2048) {
                s.set_variant_override(Some(GemvVariant::generated(1)));
                assert_eq!(s.prepare_plan(&g).unwrap().steps.len(), 2);
            }
        }
    }
    for (rows0, rows1, cols) in [(256usize, 256usize, 512usize), (16, 16, 2048)] {
        let mut w = Weights::new();
        let mut g = Graph::new();
        let x = g.input("x", vec![1, cols], DType::F32);
        let kw = g.weight("k_w", vec![rows0, cols], DType::BF16);
        let vw = g.weight("v_w", vec![rows1, cols], DType::BF16);
        let kb = g.weight("k_b", vec![rows0], DType::F32);
        let vb = g.weight("v_b", vec![rows1], DType::F32);
        let to_bf16 = |vals: Vec<f32>| -> Vec<u16> {
            vals.iter().map(|&v| (v.to_bits() >> 16) as u16).collect()
        };
        w.insert("k_w", Tensor::BF16(to_bf16(rand(rows0 * cols, 71))));
        w.insert("v_w", Tensor::BF16(to_bf16(rand(rows1 * cols, 73))));
        w.insert("k_b", Tensor::F32(rand(rows0, 11)));
        w.insert("v_b", Tensor::F32(rand(rows1, 13)));
        let k_out = g.add(
            "k_proj",
            Op::Linear,
            &[x, kw, kb],
            vec![1, rows0],
            DType::F32,
        );
        let v_raw = g.add("v_proj", Op::Linear, &[x, vw], vec![1, rows1], DType::F32);
        let v_out = g.add(
            "v_biased",
            Op::Binary {
                kind: crate::graph::BinaryKind::Add,
            },
            &[v_raw, vb],
            vec![1, rows1],
            DType::F32,
        );
        g.output(k_out);
        g.output(v_out);

        let s = Session::new(rt(), &w).unwrap();
        let plan = s.prepare_plan(&g).unwrap();
        assert_eq!(plan.steps.len(), 1);
        assert!(matches!(plan.steps[0].op, Step::GemvPackedPair { .. }));

        let inp = vec![Tensor::F32(rand(cols, 42))];
        let mut cs = StateStore::new();
        let want = cpu::run(&g, &w, &mut cs, inp.clone()).unwrap();
        let got = s.run_prepared(&g, &plan, None, inp).unwrap();
        for (i, (wt, gt)) in want.iter().zip(&got).enumerate() {
            let err = rel_err(wt.as_f32().unwrap(), gt.as_f32().unwrap());
            assert!(
                err < 1e-5,
                "BF16 ({rows0},{rows1},{cols}) out {i}: err {err}"
            );
        }
    }
}

#[test]
fn plan_value_locs_resolve_without_per_token_hashmap() {
    let f = DType::F32;
    let mut g = Graph::new();
    let ids = g.input("ids", vec![1], DType::U32);
    let pos = g.input("pos", vec![1], DType::U32);
    let emb_w = g.weight("emb", vec![8, 4], f);
    let x = g.add("emb_lookup", Op::Embedding, &[emb_w, ids], vec![1, 4], f);
    let wq = g.weight("wq", vec![4, 4], f);
    let wk = g.weight("wk", vec![4, 4], f);
    let wv = g.weight("wv", vec![4, 4], f);
    let q = g.add("q", Op::Linear, &[x, wq], vec![1, 4], f);
    let k = g.add("k", Op::Linear, &[x, wk], vec![1, 4], f);
    let v = g.add("v", Op::Linear, &[x, wv], vec![1, 4], f);
    let ks = g.state("k_cache", vec![16, 4], f, false);
    let vs = g.state("v_cache", vec![16, 4], f, false);
    let attn = g.add(
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
        f,
    );
    g.output(attn);

    let mut w = Weights::new();
    w.insert("emb", Tensor::F32(rand(8 * 4, 1)));
    w.insert("wq", Tensor::F32(rand(16, 2)));
    w.insert("wk", Tensor::F32(rand(16, 3)));
    w.insert("wv", Tensor::F32(rand(16, 4)));

    let mut s = Session::new(rt(), &w).unwrap();
    s.set_kv_f16(false);
    let plan = s.prepare_plan(&g).unwrap();

    assert_eq!(plan.value_locs.len(), g.values.len());
    assert_eq!(plan.embedding_checks.len(), 1);
    assert_eq!(plan.embedding_checks[0].0, 0);
    assert_eq!(plan.embedding_checks[0].1, 8);
    assert_eq!(plan.attn_pos_inputs, vec![(1, 1)]);
    assert_eq!(plan.state_vars.len(), 2);
    assert!(matches!(
        plan.value_locs[ids.0 as usize],
        PlanValueLoc::Arena(_)
    ));
    assert!(matches!(
        plan.value_locs[emb_w.0 as usize],
        PlanValueLoc::Weight(_, _)
    ));
    assert!(matches!(
        plan.value_locs[ks.0 as usize],
        PlanValueLoc::State(0)
    ));
    assert!(matches!(
        plan.value_locs[vs.0 as usize],
        PlanValueLoc::State(1)
    ));

    let mut m_state = StateStore::new();
    let mut c_state = StateStore::new();
    for step_pos in 0..3u32 {
        let inp = vec![Tensor::U32(vec![step_pos + 1]), Tensor::U32(vec![step_pos])];
        let want = cpu::run(&g, &w, &mut c_state, inp.clone()).unwrap();
        let got = s.run_prepared(&g, &plan, Some(&mut m_state), inp).unwrap();
        let err = rel_err(want[0].as_f32().unwrap(), got[0].as_f32().unwrap());
        assert!(err < 1e-4, "step {step_pos}: rel_err {err}");
    }
}


/// CPU only: `plan_dump` groups steps by layer and describes a materialized strided slice.
#[test]
fn plan_dump_describes_layers_and_slices() {
    let mut g = Graph::new();
    let x = g.input("x", vec![2, 8], DType::F32);
    let op = Op::Slice {
        axis: 1,
        start: 0,
        end: 4,
    };
    let s = g.add("blk.3.slice", op, &[x], vec![2, 4], DType::F32);
    let neg = Op::Unary {
        kind: crate::graph::UnaryKind::Neg,
    };
    let y = g.add("blk.3.neg", neg, &[s], vec![2, 4], DType::F32);
    g.output(y);
    let d = plan_dump(&g, 1).unwrap();
    assert!(d.contains("-- layer 3\n"), "{d}");
    assert!(
        d.contains("slice axis=1 [0..4] of [2, 8] contiguous=false"),
        "{d}"
    );
    assert!(
        d.contains("plan-dump: 2 steps, 2 dispatches at ctx 1"),
        "{d}"
    );
}

#[test]
fn batch_horizontal_commutes_interleaved_gemv_and_rmsnorm() {
    let f = DType::F32;
    let (heads, kv_heads, hd, h) = (4usize, 2usize, 8usize, 32usize);
    let (q_dim, kv_dim) = (heads * hd, kv_heads * hd);
    let rope = Op::Rope {
        head_dim: hd,
        rotary_dim: hd,
        variant: crate::graph::RopeVariant::HalfSplit,
        mscale: 1.0,
    };
    let mut g = Graph::new();
    let x = g.input("x", vec![1, h], f);
    let pos = g.input("pos", vec![1], DType::U32);
    let wq = g.weight("wq", vec![q_dim, h], DType::F16);
    let wk = g.weight("wk", vec![kv_dim, h], DType::F16);
    let wv = g.weight("wv", vec![kv_dim, h], DType::F16);
    let gw = g.weight("gw", vec![hd], f);
    let inv = g.weight("inv", vec![hd / 2], f);

    let q = g.add("q", Op::Linear, &[x, wq], vec![1, q_dim], f);
    let qn = g.add("qn", Op::RmsNorm { eps: 1e-5 }, &[q, gw], vec![1, q_dim], f);
    let qr = g.add("qr", rope.clone(), &[qn, pos, inv], vec![1, q_dim], f);

    let k = g.add("k", Op::Linear, &[x, wk], vec![1, kv_dim], f);
    let kn = g.add(
        "kn",
        Op::RmsNorm { eps: 1e-5 },
        &[k, gw],
        vec![1, kv_dim],
        f,
    );
    let kr = g.add("kr", rope, &[kn, pos, inv], vec![1, kv_dim], f);

    let v = g.add("v", Op::Linear, &[x, wv], vec![1, kv_dim], f);
    let vn = g.add(
        "vn",
        Op::RmsNorm { eps: 1e-6 },
        &[v, gw],
        vec![1, kv_dim],
        f,
    );

    let ks = g.state("ks", vec![16, kv_dim], f, false);
    let vs = g.state("vs", vec![16, kv_dim], f, false);
    let att = g.add(
        "att",
        Op::CachedAttention {
            heads,
            kv_heads,
            head_dim: hd,
            v_head_dim: hd,
            scale: 0.25,
            causal: true,
            window: None,
            soft_cap: None,
            write: true,
        },
        &[qr, kr, vn, ks, vs, pos],
        vec![1, q_dim],
        f,
    );
    g.output(att);

    let to_f16 = |vals: Vec<f32>| -> Vec<u16> {
        vals.into_iter()
            .map(|v| crate::model::f32_to_f16(v * 0.2))
            .collect()
    };
    let mut w = Weights::new();
    w.insert("wq", Tensor::F16(to_f16(rand(q_dim * h, 1))));
    w.insert("wk", Tensor::F16(to_f16(rand(kv_dim * h, 2))));
    w.insert("wv", Tensor::F16(to_f16(rand(kv_dim * h, 3))));
    w.insert("gw", Tensor::F32(rand(hd, 4)));
    w.insert("inv", Tensor::F32(rand(hd / 2, 5)));

    let mut sess = Session::new(rt(), &w).unwrap();
    sess.set_kv_f16(false);
    let plan = sess.prepare_plan(&g).unwrap();
    assert_eq!(plan.steps.len(), 5);
    assert!(matches!(plan.steps[0].op, Step::GemvPacked { .. }));
    assert!(matches!(plan.steps[1].op, Step::GemvPacked { .. }));
    assert!(matches!(plan.steps[2].op, Step::GemvPacked { .. }));
    match &plan.steps[3].op {
        Step::RmsNormBatch {
            rope: Some(_),
            segs,
        } => {
            assert_eq!(segs.len(), 3);
            assert_eq!(
                [segs[0].has_rope, segs[1].has_rope, segs[2].has_rope],
                [true, true, false]
            );
        }
        other => panic!("expected RmsNormBatch, got {other:?}"),
    }
    assert!(matches!(plan.steps[4].op, Step::CachedAttention { .. }));

    let (mut cs, mut ms) = (StateStore::new(), StateStore::new());
    for step in 0..3u32 {
        let inp = vec![
            Tensor::F32(rand(h, 10 + step as u64)),
            Tensor::U32(vec![step]),
        ];
        let want = cpu::run(&g, &w, &mut cs, inp.clone()).unwrap();
        let got = sess.run_prepared(&g, &plan, Some(&mut ms), inp).unwrap();
        let err = rel_err(want[0].as_f32().unwrap(), got[0].as_f32().unwrap());
        assert!(err < 1e-4, "step {step}: rel_err {err}");
    }
}

#[test]
fn readback_span_timing_probe() {
    let n = 262_144usize;
    let buf = new_shared_buffer(&rt().device, n * 4).unwrap();
    let ptr = buf.contents() as *mut f32;
    unsafe {
        for i in 0..n {
            *ptr.add(i) = (i as f32) * 0.001;
        }
    }
    let s = unsafe { std::slice::from_raw_parts(ptr as *const f32, n) };
    let mut to_vec_us = Vec::with_capacity(100);
    for _ in 0..100 {
        let t0 = Instant::now();
        let v = std::hint::black_box(s.to_vec());
        to_vec_us.push(t0.elapsed().as_secs_f64() * 1e6);
        drop(v);
    }
    let mut reuse = vec![0.0f32; n];
    let mut reuse_us = Vec::with_capacity(100);
    for _ in 0..100 {
        let t0 = Instant::now();
        reuse.copy_from_slice(std::hint::black_box(s));
        std::hint::black_box(&reuse);
        reuse_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    to_vec_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    reuse_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "readback_to_vec_med_us={:.2} readback_reuse_med_us={:.2}",
        to_vec_us[50], reuse_us[50]
    );
}

#[test]
fn prepared_plan_precomputes_mlp_scratch_size() {
    let f = DType::F32;
    let (h, inter) = (64usize, 128usize);
    let mut g = Graph::new();
    let x = g.input("x", vec![1, h], f);
    let norm_w = g.weight("norm_w", vec![h], f);
    let gate_w = g.weight("gate_w", vec![inter, h], f);
    let up_w = g.weight("up_w", vec![inter, h], f);
    let down_w = g.weight("down_w", vec![h, inter], f);
    let ff_in = g.add("ln", Op::RmsNorm { eps: 1e-5 }, &[x, norm_w], vec![1, h], f);
    let gate = g.add("gate", Op::Linear, &[ff_in, gate_w], vec![1, inter], f);
    let up = g.add("up", Op::Linear, &[ff_in, up_w], vec![1, inter], f);
    let act = g.add("act", Op::SwiGlu, &[gate, up], vec![1, inter], f);
    let down = g.add("down", Op::Linear, &[act, down_w], vec![1, h], f);
    let out = g.add(
        "res",
        Op::Binary {
            kind: crate::graph::BinaryKind::Add,
        },
        &[x, down],
        vec![1, h],
        f,
    );
    g.output(out);
    assert_eq!(
        crate::graph::egraph::fuse_rule(
            &mut g,
            None,
            crate::graph::egraph::Rule::MegakernelMlp
        ),
        1
    );
    let mut w = Weights::new();
    w.insert("norm_w", Tensor::F32(rand(h, 1)));
    w.insert("gate_w", Tensor::F32(rand(inter * h, 2)));
    w.insert("up_w", Tensor::F32(rand(inter * h, 3)));
    w.insert("down_w", Tensor::F32(rand(h * inter, 4)));
    let s = Session::new(rt(), &w).unwrap();
    let plan = s.prepare_plan(&g).unwrap();
    assert_eq!(plan.mlp_scratch_size, h * 4 + 2 * inter * 4);
}

/// Bits of a graph output: u32 ids as themselves, f32 as `to_bits`.
fn out_bits(t: &Tensor) -> Vec<u32> {
    match t {
        Tensor::U32(v) => v.clone(),
        Tensor::F32(v) => v.iter().map(|x| x.to_bits()).collect(),
        other => panic!("unexpected output {other:?}"),
    }
}

/// Runs `g` on the CPU executor and on Metal, each from the states `init`, once per input set in
/// `runs` (state persisting between runs); returns every run's (CPU, Metal) outputs and the final
/// (CPU, Metal) value of each state in `init`.
#[allow(clippy::type_complexity)]
fn both_stateful(
    g: &Graph,
    w: &Weights,
    init: &[(&str, Vec<f32>)],
    runs: Vec<Vec<Tensor>>,
) -> (Vec<(Vec<Tensor>, Vec<Tensor>)>, Vec<(Vec<f32>, Vec<f32>)>) {
    let (mut cs, mut gs) = (StateStore::new(), StateStore::new());
    for (name, v) in init {
        cs.insert(*name, Tensor::F32(v.clone()));
        gs.insert(*name, Tensor::F32(v.clone()));
    }
    let s = Session::new(rt(), w).unwrap();
    let outs = runs
        .into_iter()
        .map(|inputs| {
            let c = cpu::run(g, w, &mut cs, inputs.clone()).unwrap();
            let m = s.run_with_state(g, Some(&mut gs), inputs).unwrap();
            (c, m)
        })
        .collect();
    let states = init
        .iter()
        .map(|(name, _)| {
            let c = cs.get(name).unwrap().as_f32().unwrap().to_vec();
            (c, gs.read_metal(name).unwrap())
        })
        .collect();
    (outs, states)
}

/// `StateWrite` writes rows into a persistent state at `pos0` on both executors: the token is
/// `pos0 + m`, a later node of the same graph reads the new rows, and the state carries into the
/// next run.
#[test]
fn state_write_matches_cpu() {
    let (cap, w, m) = (6, 3, 2);
    let mut g = Graph::new();
    let st = g.state("cache", vec![cap, w], DType::F32, false);
    let rows = g.input("rows", vec![m, w], DType::F32);
    let pos0 = g.input("pos0", vec![1], DType::U32);
    let all = g.input("all", vec![cap], DType::U32);
    let token = g.push("write", Op::StateWrite, &[st, rows, pos0]).unwrap();
    let read = g.push("read", Op::Gather { axis: 0 }, &[st, all]).unwrap();
    g.output(token);
    g.output(read);
    let init = rand(cap * w, 3);
    let run = |p: u32, seed| {
        vec![
            Tensor::F32(rand(m * w, seed)),
            Tensor::U32(vec![p]),
            Tensor::U32((0..cap as u32).collect()),
        ]
    };
    let (outs, states) = both_stateful(
        &g,
        &Weights::new(),
        &[("cache", init.clone())],
        vec![run(3, 4), run(0, 5), run(4, 6)],
    );
    let mut want = init;
    for ((c, gpu), (p, seed)) in outs.iter().zip([(3usize, 4u64), (0, 5), (4, 6)]) {
        want[p * w..(p + m) * w].copy_from_slice(&rand(m * w, seed));
        assert_eq!(c[0], Tensor::U32(vec![(p + m) as u32]));
        assert_eq!(out_bits(&gpu[0]), [(p + m) as u32]);
        assert_eq!(c[1], Tensor::F32(want.clone()), "CPU reads the write");
        assert_eq!(gpu[1], Tensor::F32(want.clone()), "Metal reads the write");
    }
    assert_eq!(states[0].0, want);
    assert_eq!(states[0].1, want);
    // Dead-node elimination keeps the write even when nothing reads its token.
    let mut g2 = Graph::new();
    let st = g2.state("cache", vec![cap, w], DType::F32, false);
    let rows = g2.input("rows", vec![m, w], DType::F32);
    let pos0 = g2.input("pos0", vec![1], DType::U32);
    g2.push("write", Op::StateWrite, &[st, rows, pos0]).unwrap();
    let out = g2
        .push(
            "other",
            Op::Unary {
                kind: crate::graph::UnaryKind::Neg,
            },
            &[rows],
        )
        .unwrap();
    g2.output(out);
    crate::graph::passes::canonical::eliminate_dead_nodes(&mut g2);
    assert!(g2.nodes.iter().any(|n| n.op == Op::StateWrite));
    // Writing past the declared rows is the same error on both executors.
    let past = || {
        vec![
            Tensor::F32(rand(m * w, 1)),
            Tensor::U32(vec![5]),
            Tensor::U32((0..cap as u32).collect()),
        ]
    };
    let mut st = StateStore::new();
    let ce = cpu::run(&g, &Weights::new(), &mut st, past()).unwrap_err();
    let me = Session::new(rt(), &Weights::new())
        .unwrap()
        .run(&g, past())
        .unwrap_err();
    assert!(ce.contains("past the state's 6 rows"), "{ce}");
    assert_eq!(ce, me);
}

/// `QsaSelect` / `QsaBias` on Metal equal the CPU executor bit for bit: random scores with ties,
/// signed zeros, infinities and NaN, rows below and above `top_blocks` complete blocks, and a
/// score width past one threadgroup.
#[test]
fn qsa_select_and_bias_match_cpu() {
    for (n, b, ratio, top) in [
        (7usize, 9usize, 4usize, 2usize),
        (5, 700, 2, 37),
        (9, 300, 1, 1),
        (4, 64, 8, 0),
        (3, 33, 3, 32),
    ] {
        let mut score: Vec<f32> = rand(n * b, n as u64 + b as u64)
            .into_iter()
            .map(|v| (v * 4.0).round() * 0.5)
            .collect();
        for (i, special) in [
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
        ]
        .into_iter()
        .enumerate()
        {
            score[(i * 7) % (n * b)] = special;
        }
        let pos: Vec<u32> = rand(n, 9)
            .into_iter()
            .map(|v| ((v + 1.0) * 0.5 * (b * ratio) as f32) as u32 % (b * ratio) as u32)
            .collect();
        for select in [true, false] {
            let mut g = Graph::new();
            let sc = g.input("score", vec![n, b], DType::F32);
            let p = g.input("pos", vec![n], DType::U32);
            let op = if select {
                Op::QsaSelect {
                    ratio,
                    top_blocks: top,
                }
            } else {
                Op::QsaBias {
                    ratio,
                    top_blocks: top,
                }
            };
            let y = g.push("qsa", op, &[sc, p]).unwrap();
            g.output(y);
            let inputs = vec![Tensor::F32(score.clone()), Tensor::U32(pos.clone())];
            let want = cpu::run_stateless(&g, &Weights::new(), inputs.clone()).unwrap();
            let got = Session::new(rt(), &Weights::new())
                .unwrap()
                .run(&g, inputs)
                .unwrap();
            assert_eq!(
                out_bits(&want[0]),
                out_bits(&got[0]),
                "n {n} B {b} ratio {ratio} top {top} select {select}"
            );
        }
    }
}

/// `DeltaRule` and `CausalConv1dCached` with state over n = 5 rows in one call, starting from a
/// non-zero state: Metal matches the CPU executor in every output row and in the final state, over
/// two consecutive calls.
#[test]
fn multi_row_recurrent_ops_with_state_match_cpu() {
    let n = 5;
    for (kh, vh, kd, vd) in [
        (1usize, 1usize, 4usize, 4usize),
        (2, 4, 16, 32),
        (2, 2, 128, 128),
    ] {
        let mut g = Graph::new();
        let f = DType::F32;
        let q = g.input("q", vec![n, kh * kd], f);
        let k = g.input("k", vec![n, kh * kd], f);
        let v = g.input("v", vec![n, vh * vd], f);
        let beta = g.input("beta", vec![n, vh], f);
        let ld = g.input("log_decay", vec![n, vh], f);
        let st = g.state("ssm", vec![vh * kd * vd], f, true);
        let op = Op::DeltaRule {
            key_heads: kh,
            val_heads: vh,
            key_dim: kd,
            val_dim: vd,
        };
        let y = g.push("delta", op, &[q, k, v, beta, ld, st]).unwrap();
        g.output(y);
        let run = |seed: u64| {
            vec![
                Tensor::F32(rand(n * kh * kd, seed)),
                Tensor::F32(rand(n * kh * kd, seed + 1)),
                Tensor::F32(rand(n * vh * vd, seed + 2)),
                Tensor::F32(
                    rand(n * vh, seed + 3)
                        .iter()
                        .map(|x| 0.5 + 0.4 * x)
                        .collect(),
                ),
                Tensor::F32(
                    rand(n * vh, seed + 4)
                        .iter()
                        .map(|x| -0.3 + 0.2 * x)
                        .collect(),
                ),
            ]
        };
        let init = rand(vh * kd * vd, 7);
        let (outs, states) = both_stateful(
            &g,
            &Weights::new(),
            &[("ssm", init.clone())],
            vec![run(10), run(20)],
        );
        for (c, gpu) in &outs {
            let (c, gpu) = (c[0].as_f32().unwrap(), gpu[0].as_f32().unwrap());
            assert_eq!(c.len(), n * vh * vd);
            assert!(rel_err(c, gpu) < 1e-4, "DeltaRule {kh}/{vh}/{kd}/{vd} rows");
        }
        assert!(
            rel_err(&states[0].0, &states[0].1) < 1e-4,
            "DeltaRule state"
        );
        assert!(max_abs_diff(&states[0].0, &init) > 1e-3, "the state moved");
    }
    for (ch, kernel, dilation) in [(7usize, 4usize, 1usize), (33, 3, 2), (1, 2, 1)] {
        let mut g = Graph::new();
        let x = g.input("x", vec![n, ch], DType::F32);
        let wv = g.weight("w", vec![ch, kernel], DType::F32);
        let hist = (kernel - 1) * dilation;
        let st = g.state("conv", vec![hist, ch], DType::F32, true);
        let op = Op::CausalConv1dCached { kernel, dilation };
        let y = g.push("conv", op, &[x, wv, st]).unwrap();
        g.output(y);
        let mut w = Weights::new();
        w.insert("w", Tensor::F32(rand(ch * kernel, 3)));
        let init = rand(hist * ch, 5);
        let runs = vec![
            vec![Tensor::F32(rand(n * ch, 1))],
            vec![Tensor::F32(rand(n * ch, 2))],
        ];
        let (outs, states) = both_stateful(&g, &w, &[("conv", init)], runs);
        for (c, gpu) in &outs {
            let (c, gpu) = (c[0].as_f32().unwrap(), gpu[0].as_f32().unwrap());
            assert_eq!(c.len(), n * ch);
            assert!(rel_err(c, gpu) < 1e-5, "conv {ch}/{kernel}/{dilation} rows");
        }
        assert!(rel_err(&states[0].0, &states[0].1) < 1e-6, "conv state");
    }
}

#[test]
fn sample_topk_matches_cpu_nucleus() {
    let s = Session::new(rt(), &Weights::new()).unwrap();
    let vocab = 10_000;
    let mut logits: Vec<f32> = rand(vocab, 11).iter().map(|v| v * 4.0).collect();
    // The two largest logits share a 256-stride lane of the first span (ids 300 and 300 + 256).
    logits[300] = 9.0;
    logits[556] = 8.5;
    let (temp, top_p) = (0.8f32, 0.95f32);
    let buf = |bytes: usize| new_shared_buffer(&s.rt.device, bytes).unwrap();
    let (lb, ob, rb) = (buf(vocab * 4), buf(4), buf(64 * 4));
    // SAFETY: fresh shared buffers no command buffer uses yet.
    unsafe { std::ptr::copy_nonoverlapping(logits.as_ptr(), lb.contents() as *mut f32, vocab) };
    let p = SampleGpuParams {
        vocab: vocab as u32,
        top_k: crate::scheduler::SAMPLE_TOP_K as u32,
        inv_temp: 1.0 / temp,
        top_p,
        rep_penalty: 1.0,
        n_recent: 1,
        rng_seed: 42,
    };
    let cb = s.rt.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    s.encode_sample(&Counted::new(enc), (&lb, 0), (&ob, 0), &rb, &p)
        .unwrap();
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let k = crate::scheduler::SAMPLE_TOP_K;
    let scratch = s.argmax_scratch.lock().clone().unwrap();
    // SAFETY: the command buffer completed; the final kernel wrote `k` (id, prob bits) pairs.
    let kept = unsafe { std::slice::from_raw_parts(scratch.contents() as *const [u32; 2], k) };
    let got: Vec<(u32, f32)> = kept
        .iter()
        .take_while(|e| e[0] != u32::MAX)
        .map(|e| (e[0], f32::from_bits(e[1])))
        .collect();
    let (cands, probs, n) = crate::runtime::lm::nucleus(&logits, &[], temp, k, top_p, 1.0);
    assert!(n > 2, "the nucleus keeps more than the two planted logits");
    assert_eq!(
        got.iter().map(|e| e.0).collect::<Vec<_>>(),
        cands[..n].iter().map(|c| c.0 as u32).collect::<Vec<_>>()
    );
    for ((_, gp), &cp) in got.iter().zip(&probs[..n]) {
        assert!((gp - cp).abs() < 1e-5, "prob {gp} vs {cp}");
    }
    // SAFETY: the command buffer completed.
    let tok = unsafe { *(ob.contents() as *const u32) };
    assert!(got.iter().any(|e| e.0 == tok), "drawn token {tok} is kept");
}
