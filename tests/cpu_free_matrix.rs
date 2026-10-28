//! No family computes on the CPU unless it is gated and the user opted in with `--backend cpu`.
//!
//! Runs in its own process because it arms the global CPU tripwire.

use gibbs_engine::arch::{Pipeline, Signals};
use gibbs_engine::hardware::{available_devices, cpu_compute, Device, PlacementMode};
use std::sync::Once;

static INIT: Once = Once::new();

#[allow(dead_code)]
#[path = "../src/model/classifier/synth.rs"]
mod synth;

/// Loads a tiny synthetic classifier repo on Metal through the real loader and classifies a text and
/// a pair twice each; the tripwire is armed, so any host forward would panic.
fn classifier_on_gpu(name: &str, (cfg, ts): (serde_json::Value, synth::Ts)) {
    use gibbs_engine::model::classifier::SequenceClassifier;
    use gibbs_engine::model::hub::HuggingFaceHub;
    let hub = HuggingFaceHub::with_cache_dir(synth::write_repo(name, cfg, &ts));
    let c = SequenceClassifier::load(&hub, &format!("t/{name}"), Device::Metal)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(c.device(), Device::Metal, "{name}");
    let before = cpu_compute::entered();
    for pair in [None, Some(synth::PAIR)] {
        for _ in 0..2 {
            let (scores, _) = c
                .classify(synth::TEXT, pair)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(
                scores.iter().all(|s| s.score.is_finite()),
                "{name}: non-finite scores"
            );
            let sum: f32 = scores.iter().map(|s| s.score).sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "{name}: softmax scores sum to {sum}"
            );
        }
    }
    assert_eq!(
        cpu_compute::entered(),
        before,
        "{name}: entered CPU compute"
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    Ran,
    Exempt(&'static str),
}

fn find_fixture(rel: &str) -> std::path::PathBuf {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let local = root.join("target/golden-fixtures").join(rel);
    assert!(
        local.exists(),
        "golden fixture `{rel}` absent at {}: run `cargo test --release --test golden` first",
        local.display()
    );
    local
}

/// A `tests/golden` fixture checkpoint (`target/golden-fixtures/<fixture>`, written by the golden
/// tests) through the spec runtime on Metal: prepare `ids` with `media`, prefill, two decode
/// steps; the tripwire is armed, so any host compute panics.
fn spec_fixture_on_gpu(
    name: &str,
    fixture: &str,
    family: Option<&str>,
    ids: &[u32],
    media: &[gibbs_engine::model::native::MediaInput],
) -> Outcome {
    let dir = find_fixture(fixture);
    let a = gibbs_engine::model::hub::ModelArtifacts {
        model_id: format!("golden/{fixture}"),
        local_dir: Some(dir.clone()),
        config_path: Some(dir.join("config.json")),
        tokenizer_path: Some(dir.join("tokenizer.json")).filter(|p| p.exists()),
        weights_path: Some(dir.join("model.safetensors")),
        is_cached: true,
        ..Default::default()
    };
    let reg = gibbs_engine::arch::registry().unwrap();
    let lm = gibbs_engine::runtime::lm::load(&a, reg, family, PlacementMode::Pin(Device::Metal))
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(lm.device(), Device::Metal, "{name}");
    let (ids, mut st) = lm
        .prepare(ids.to_vec(), media)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let before = cpu_compute::entered();
    let mut l = lm
        .forward(&ids, 0, &mut st)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    for q in 0..2 {
        assert!(l.iter().all(|v| v.is_finite()), "{name}: non-finite logits");
        l = lm
            .forward(&[1], ids.len() + q, &mut st)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
    }
    assert_eq!(
        cpu_compute::entered(),
        before,
        "{name}: entered CPU compute"
    );
    Outcome::Ran
}

/// Runs component `net_name` of `family` on golden fixture `fixture` pinned to Metal (`Net::open`
/// fails if any node in the graph cannot lower to Metal).
fn net_on_gpu(
    family: &str,
    fixture: &str,
    net_name: &str,
    vars: &[(&str, f64)],
    step: &[(&str, f64)],
) -> Outcome {
    use gibbs_engine::arch::expr::Val;
    use gibbs_engine::graph::{DType, Tensor};
    let dir = find_fixture(fixture);
    let net = gibbs_engine::runtime::diffusion::Net::open(
        &dir,
        family,
        net_name,
        PlacementMode::Pin(Device::Metal),
    )
    .unwrap_or_else(|e| panic!("{family}/{net_name}: {e}"));
    let mut val_vars: Vec<(&str, Val)> = vars.iter().map(|&(k, v)| (k, Val::Num(v))).collect();
    val_vars.push(("head", Val::Str("hidden".into())));
    let inst = net
        .instance(&val_vars)
        .unwrap_or_else(|e| panic!("{family}/{net_name} instance: {e}"));
    let mut r = Lcg(13);
    let mut given: Vec<(&str, Tensor)> = Vec::new();
    for p in &inst.inputs {
        if p.gen.is_some() {
            continue;
        }
        let len: usize = p.shape.iter().product();
        let t = match p.dtype {
            DType::U32 if p.name == "positions" => Tensor::U32((0..len as u32).collect()),
            DType::U32 if p.name == "pos0" => Tensor::U32(vec![0; len]),
            DType::U32 => Tensor::U32(vec![1; len]),
            _ => Tensor::F32(r.vec(len, 0.1)),
        };
        given.push((p.name.as_str(), t));
    }
    let _ = step;
    let step = [
        ("t", 500.0),
        ("timestep", 500.0),
        ("sigma", 0.5),
        ("guidance", 3.5),
    ];
    let before = cpu_compute::entered();
    let out = net
        .run(&val_vars, &given, &step)
        .unwrap_or_else(|e| panic!("{family}/{net_name} run: {e}"));
    assert_eq!(
        cpu_compute::entered(),
        before,
        "{family}/{net_name}: entered CPU compute"
    );
    for (oname, t) in out {
        if let Ok(s) = t.as_f32() {
            assert!(
                s.iter().all(|v| v.is_finite()),
                "{family}/{net_name} {oname}: non-finite output"
            );
        }
    }
    Outcome::Ran
}

/// 16 kHz mono 16-bit WAV: 0.5 s of a 440 Hz tone.
fn tone_wav() -> Vec<u8> {
    let pcm: Vec<u8> = (0..8000)
        .flat_map(|i| {
            (((i as f32 / 16000.0 * 440.0 * std::f32::consts::TAU).sin() * 8000.0) as i16)
                .to_le_bytes()
        })
        .collect();
    let mut b = b"RIFF".to_vec();
    b.extend((36 + pcm.len() as u32).to_le_bytes());
    b.extend(b"WAVEfmt ");
    b.extend(16u32.to_le_bytes());
    b.extend([1, 0, 1, 0]);
    b.extend(16_000u32.to_le_bytes());
    b.extend(32_000u32.to_le_bytes());
    b.extend([2, 0, 16, 0]);
    b.extend(b"data");
    b.extend((pcm.len() as u32).to_le_bytes());
    b.extend(pcm);
    b
}

fn metal_present() -> bool {
    available_devices().contains(&Device::Metal)
}

/// The `<image>` id of the golden `enc_smolvlm_*` fixtures.
const IDEFICS3_IMG: u32 = 120;

/// Media soft tokens placed over prompt rows by the GPU `Scatter`.
fn scatter_fixture(name: &str) {
    let base: Vec<f32> = (0..4 * 8).map(|i| i as f32).collect();
    let out = gibbs_engine::model::vision_graph::scatter_rows(
        base.clone(),
        4,
        8,
        &[1, 3],
        vec![-1.0; 16],
        PlacementMode::Auto,
    )
    .unwrap_or_else(|e| panic!("{name} scatter: {e}"));
    assert_eq!(&out[..8], &base[..8], "{name}");
    assert!(
        out[8..16].iter().chain(&out[24..]).all(|&v| v == -1.0),
        "{name}"
    );
}

/// Deterministic f32 tensors written as a safetensors checkpoint.
struct Ckpt(u64, Vec<(String, Vec<usize>, Vec<f32>)>);
impl Ckpt {
    fn new(seed: u64) -> Self {
        Self(seed, Vec::new())
    }
    fn s(&mut self, n: impl Into<String>, shape: &[usize], scale: f32) -> &mut Self {
        let v = (0..shape.iter().product::<usize>())
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect();
        self.1.push((n.into(), shape.to_vec(), v));
        self
    }
    fn w(&mut self, n: impl Into<String>, shape: &[usize]) -> &mut Self {
        self.s(n, shape, 1.0 / (shape[1] as f32).sqrt())
    }
    fn norm(&mut self, n: impl Into<String>, d: usize, _c: f32) -> &mut Self {
        self.1.push((n.into(), vec![d], vec![1.0; d]));
        self
    }
    fn write_checkpoint(&self, dir: &std::path::Path, cfg: &serde_json::Value, vocab: usize) {
        use serde_json::json;
        std::fs::create_dir_all(dir).unwrap();
        let (mut h, mut data) = (serde_json::Map::new(), Vec::<u8>::new());
        for (n, shape, v) in &self.1 {
            let a = data.len();
            data.extend(v.iter().flat_map(|x| x.to_le_bytes()));
            h.insert(
                n.clone(),
                json!({"dtype": "F32", "shape": shape, "data_offsets": [a, data.len()]}),
            );
        }
        let mut hb = serde_json::to_vec(&h).unwrap();
        hb.resize(hb.len().next_multiple_of(8), b' ');
        let mut out = (hb.len() as u64).to_le_bytes().to_vec();
        out.extend(hb);
        out.extend(data);
        std::fs::write(dir.join("model.safetensors"), out).unwrap();
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        let vm: serde_json::Map<String, serde_json::Value> =
            (0..vocab).map(|i| (format!("t{i}"), json!(i))).collect();
        let tok = json!({"version": "1.0", "added_tokens": [], "normalizer": null,
            "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": vm, "unk_token": "t0"}});
        std::fs::write(dir.join("tokenizer.json"), tok.to_string()).unwrap();
    }
}

fn artifacts(dir: &std::path::Path) -> gibbs_engine::model::hub::ModelArtifacts {
    gibbs_engine::model::hub::ModelArtifacts {
        model_id: "test/cpu-free".into(),
        local_dir: Some(dir.to_path_buf()),
        config_path: Some(dir.join("config.json")),
        tokenizer_path: Some(dir.join("tokenizer.json")),
        weights_path: Some(dir.join("model.safetensors")),
        is_cached: true,
        ..Default::default()
    }
}

/// A tiny random T5 (gated) or BART (3-beam search, forking beam state) through
/// `runtime::encdec` on Metal: encoder, search and decoder steps all on the GPU.
fn encdec_on_gpu(name: &str, bart: bool) {
    use gibbs_engine::model::native::NativeLm;
    use gibbs_engine::runtime::encdec::SpecEncDec;
    use serde_json::json;
    let (v, d) = (40, 32);
    let dir = std::env::temp_dir().join(format!("gibbs-matrix-encdec-{name}"));
    let mut w = Ckpt::new(7);
    let cfg = if bart {
        w.s("model.shared.weight", &[v, d], 0.3);
        for side in ["encoder", "decoder"] {
            let p = format!("model.{side}");
            w.s(format!("{p}.embed_positions.weight"), &[66, d], 0.3);
            w.norm(format!("{p}.layernorm_embedding.weight"), d, 1.0).s(
                format!("{p}.layernorm_embedding.bias"),
                &[d],
                0.1,
            );
            let mut attns = vec!["self_attn"];
            if side == "decoder" {
                attns.push("encoder_attn");
            }
            for a in attns {
                for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                    w.w(format!("{p}.layers.0.{a}.{proj}.weight"), &[d, d]).s(
                        format!("{p}.layers.0.{a}.{proj}.bias"),
                        &[d],
                        0.1,
                    );
                }
                w.norm(format!("{p}.layers.0.{a}_layer_norm.weight"), d, 1.0)
                    .s(format!("{p}.layers.0.{a}_layer_norm.bias"), &[d], 0.1);
            }
            w.w(format!("{p}.layers.0.fc1.weight"), &[64, d]).s(
                format!("{p}.layers.0.fc1.bias"),
                &[64],
                0.1,
            );
            w.w(format!("{p}.layers.0.fc2.weight"), &[d, 64]).s(
                format!("{p}.layers.0.fc2.bias"),
                &[d],
                0.1,
            );
            w.norm(format!("{p}.layers.0.final_layer_norm.weight"), d, 1.0)
                .s(format!("{p}.layers.0.final_layer_norm.bias"), &[d], 0.1);
        }
        json!({"architectures": ["BartForConditionalGeneration"], "model_type": "bart", "vocab_size": v,
            "d_model": d, "max_position_embeddings": 64, "encoder_layers": 1, "decoder_layers": 1,
            "encoder_attention_heads": 4, "decoder_attention_heads": 4, "activation_function": "gelu",
            "bos_token_id": 0, "pad_token_id": 1, "eos_token_id": 2, "decoder_start_token_id": 2,
            "num_beams": 3, "max_length": 8})
    } else {
        w.s("shared.weight", &[v, d], 1.0);
        for stack in ["encoder", "decoder"] {
            let b = format!("{stack}.block.0.layer");
            let mut subs = vec![(0, "SelfAttention")];
            if stack == "decoder" {
                subs.push((1, "EncDecAttention"));
            }
            for (l, m) in subs {
                w.norm(format!("{b}.{l}.layer_norm.weight"), d, 1.0);
                for p in ["q", "k", "v", "o"] {
                    w.w(format!("{b}.{l}.{m}.{p}.weight"), &[d, d]);
                }
            }
            w.s(
                format!("{b}.0.SelfAttention.relative_attention_bias.weight"),
                &[8, 4],
                1.0,
            );
            let f = if stack == "decoder" { 2 } else { 1 };
            w.norm(format!("{b}.{f}.layer_norm.weight"), d, 1.0);
            w.w(format!("{b}.{f}.DenseReluDense.wi_0.weight"), &[48, d])
                .w(format!("{b}.{f}.DenseReluDense.wi_1.weight"), &[48, d]);
            w.w(format!("{b}.{f}.DenseReluDense.wo.weight"), &[d, 48]);
            w.norm(format!("{stack}.final_layer_norm.weight"), d, 1.0);
        }
        json!({"architectures": ["T5ForConditionalGeneration"], "model_type": "t5", "vocab_size": v,
            "d_model": d, "d_kv": 8, "num_heads": 4, "d_ff": 48, "num_layers": 1, "relative_attention_num_buckets": 8,
            "relative_attention_max_distance": 16, "feed_forward_proj": "gated-gelu", "n_positions": 64})
    };
    w.write_checkpoint(&dir, &cfg, v);
    let reg = gibbs_engine::arch::registry().unwrap();
    let m = gibbs_engine::arch::load::load(&artifacts(&dir), reg, None)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let lm = SpecEncDec::new(m, PlacementMode::Pin(Device::Metal))
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(lm.device(), Device::Metal, "{name}");
    let src = if bart {
        b"t3 t5 t7".iter().map(|&b| b as u32).collect()
    } else {
        vec![3u32, 5, 7]
    };
    let (prompt, mut st) = lm
        .prepare(src, &[])
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let mut l = lm
        .forward(&prompt, 0, &mut st)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    for pos in prompt.len()..prompt.len() + 2 {
        let t = (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as u32;
        l = lm
            .forward(&[t], pos, &mut st)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
    }
    assert_eq!(lm.cpu_node_count(), 0, "{name}: nodes run on the CPU");
}

/// A tiny Qwen3.5 checkpoint (DeltaNet, DeltaNet, gated attention; `moe`: 8 routed experts + a
/// shared one, else dense with an MTP head; `vl`: nested decoder + ViT) through the `qwen3_5`
/// spec on Metal: prefill, the shipped drafter's proposal, and two decode steps.
fn qwen35_on_gpu(name: &str, moe: bool, vl: bool) {
    use gibbs_engine::model::native::{MediaInput, NativeLm};
    use serde_json::json;
    const H: usize = 64;
    const IMAGE: u32 = 90;
    fn mlp(w: &mut Ckpt, p: &str, moe: bool) {
        if moe {
            w.s(format!("{p}.mlp.gate.weight"), &[8, H], 1.0)
                .w(format!("{p}.mlp.experts.gate_up_proj"), &[8, 64, H])
                .w(format!("{p}.mlp.experts.down_proj"), &[8, H, 32])
                .w(format!("{p}.mlp.shared_expert.gate_proj.weight"), &[32, H])
                .w(format!("{p}.mlp.shared_expert.up_proj.weight"), &[32, H])
                .w(format!("{p}.mlp.shared_expert.down_proj.weight"), &[H, 32])
                .w(format!("{p}.mlp.shared_expert_gate.weight"), &[1, H]);
        } else {
            w.w(format!("{p}.mlp.gate_proj.weight"), &[96, H])
                .w(format!("{p}.mlp.up_proj.weight"), &[96, H])
                .w(format!("{p}.mlp.down_proj.weight"), &[H, 96]);
        }
    }
    fn attn(w: &mut Ckpt, a: &str) {
        w.w(format!("{a}.q_proj.weight"), &[2 * 2 * 32, H])
            .w(format!("{a}.k_proj.weight"), &[32, H])
            .w(format!("{a}.v_proj.weight"), &[32, H])
            .w(format!("{a}.o_proj.weight"), &[H, 2 * 32])
            .norm(format!("{a}.q_norm.weight"), 32, 0.0)
            .norm(format!("{a}.k_norm.weight"), 32, 0.0);
    }
    let mut w = Ckpt::new(0x5157_454E_00F0);
    let p = if vl { "model.language_model" } else { "model" };
    w.w(format!("{p}.embed_tokens.weight"), &[96, H]);
    for l in 0..3 {
        let lp = format!("{p}.layers.{l}");
        w.norm(format!("{lp}.input_layernorm.weight"), H, 0.0).norm(
            format!("{lp}.post_attention_layernorm.weight"),
            H,
            0.0,
        );
        if l < 2 {
            let a = format!("{lp}.linear_attn");
            w.w(format!("{a}.in_proj_qkv.weight"), &[64, H])
                .w(format!("{a}.in_proj_z.weight"), &[32, H])
                .w(format!("{a}.in_proj_b.weight"), &[2, H])
                .w(format!("{a}.in_proj_a.weight"), &[2, H])
                .w(format!("{a}.conv1d.weight"), &[64, 1, 4])
                .s(format!("{a}.A_log"), &[2], 0.5)
                .s(format!("{a}.dt_bias"), &[2], 0.5)
                .norm(format!("{a}.norm.weight"), 16, 1.0)
                .w(format!("{a}.out_proj.weight"), &[H, 32]);
        } else {
            attn(&mut w, &format!("{lp}.self_attn"));
        }
        mlp(&mut w, &lp, moe);
    }
    w.norm(format!("{p}.norm.weight"), H, 0.0)
        .w("lm_head.weight", &[96, H]);
    let mtp = !moe && !vl;
    if mtp {
        w.w("mtp.fc.weight", &[H, 2 * H]);
        for n in [
            "pre_fc_norm_embedding",
            "pre_fc_norm_hidden",
            "norm",
            "layers.0.input_layernorm",
            "layers.0.post_attention_layernorm",
        ] {
            w.norm(format!("mtp.{n}.weight"), H, 0.0);
        }
        attn(&mut w, "mtp.layers.0.self_attn");
        mlp(&mut w, "mtp.layers.0", false);
    }
    let mut text = json!({
        "model_type": if moe { "qwen3_5_moe" } else { "qwen3_5" },
        "architectures": [if moe { "Qwen3_5MoeForCausalLM" } else { "Qwen3_5ForCausalLM" }],
        "hidden_size": H, "vocab_size": 96, "num_hidden_layers": 3,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
        "intermediate_size": 96, "max_position_embeddings": 128, "rms_norm_eps": 1e-6,
        "attn_output_gate": true, "tie_word_embeddings": false, "eos_token_id": 2,
        "layer_types": ["linear_attention", "linear_attention", "full_attention"],
        "linear_num_key_heads": 1, "linear_num_value_heads": 2,
        "linear_key_head_dim": 16, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
        "mtp_num_hidden_layers": mtp as u32,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
            "mrope_interleaved": true, "mrope_section": [2, 1, 1]}
    });
    if moe {
        let c = text.as_object_mut().unwrap();
        c.remove("intermediate_size");
        for (k, v) in [
            ("num_experts", 8),
            ("num_experts_per_tok", 2),
            ("moe_intermediate_size", 32),
            ("shared_expert_intermediate_size", 32),
        ] {
            c.insert(k.into(), json!(v));
        }
    }
    let cfg = if vl {
        // ViT: hidden 32, 2 heads, 2 blocks, 4x4 position grid, patch 2, temporal 2, merge 2, one
        // deepstack merger.
        let (d, v) = (32, "model.visual");
        w.w(format!("{v}.patch_embed.proj.weight"), &[d, 3, 2, 2, 2])
            .s(format!("{v}.patch_embed.proj.bias"), &[d], 0.1)
            .s(format!("{v}.pos_embed.weight"), &[16, d], 0.5);
        for b in 0..2 {
            let bp = format!("{v}.blocks.{b}");
            for n in ["norm1", "norm2"] {
                w.norm(format!("{bp}.{n}.weight"), d, 1.0)
                    .s(format!("{bp}.{n}.bias"), &[d], 0.1);
            }
            for (n, r, c) in [
                ("attn.qkv", 3 * d, d),
                ("attn.proj", d, d),
                ("mlp.linear_fc1", 64, d),
                ("mlp.linear_fc2", d, 64),
            ] {
                w.w(format!("{bp}.{n}.weight"), &[r, c])
                    .s(format!("{bp}.{n}.bias"), &[r], 0.1);
            }
        }
        for (m, wide) in [("merger", d), ("deepstack_merger_list.0", 4 * d)] {
            let mp = format!("{v}.{m}");
            w.norm(format!("{mp}.norm.weight"), wide, 1.0)
                .s(format!("{mp}.norm.bias"), &[wide], 0.1)
                .w(format!("{mp}.linear_fc1.weight"), &[4 * d, 4 * d])
                .s(format!("{mp}.linear_fc1.bias"), &[4 * d], 0.1)
                .w(format!("{mp}.linear_fc2.weight"), &[H, 4 * d])
                .s(format!("{mp}.linear_fc2.bias"), &[H], 0.1);
        }
        text.as_object_mut().unwrap().remove("architectures");
        text["model_type"] = "qwen3_5_text".into();
        json!({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
            "image_token_id": IMAGE, "text_config": text,
            "vision_config": {"hidden_size": 32, "num_heads": 2, "depth": 2, "num_position_embeddings": 16,
                "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2, "in_channels": 3,
                "intermediate_size": 64, "out_hidden_size": H, "hidden_act": "gelu_pytorch_tanh",
                "deepstack_visual_indexes": [0]}})
    } else {
        text
    };
    let dir = std::env::temp_dir().join(format!("gibbs_cpu_free_{name}_{}", std::process::id()));
    w.write_checkpoint(&dir, &cfg, 96);
    std::fs::write(
        dir.join("preprocessor_config.json"),
        r#"{"size":{"shortest_edge":64,"longest_edge":100000},"image_mean":[0.5,0.5,0.5],
            "image_std":[0.5,0.5,0.5],"patch_size":2,"temporal_patch_size":2,"merge_size":2}"#,
    )
    .unwrap();
    let m = gibbs_engine::arch::load::load(
        &artifacts(&dir),
        gibbs_engine::arch::registry().unwrap(),
        None,
    )
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(m.spec.family, "qwen3_5", "{name}");
    let lm = gibbs_engine::runtime::lm::SpecLm::new(m, PlacementMode::Pin(Device::Metal))
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(lm.device(), Device::Metal, "{name}");
    let argmax = |l: &[f32]| (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap() as u32;
    let media = match vl {
        true => {
            let mut img = gibbs_engine::model::image::ImageBuffer::new(16, 24);
            img.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, px)| *px = (i * 37 % 251) as u8);
            vec![MediaInput::Image(img.to_png())]
        }
        false => Vec::new(),
    };
    let prompt = if vl {
        vec![3, 7, IMAGE, 11, 5]
    } else {
        vec![4, 8, 15, 16, 23]
    };
    let (ids, mut st) = lm
        .prepare(prompt, &media)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let n = ids.len();
    let drafter = lm.drafter();
    assert_eq!(drafter.is_some(), mtp, "{name}: shipped MTP drafter");
    let mut l = match &drafter {
        Some(d) => {
            lm.set_taps(&mut st, &d.tap_layers()).unwrap();
            lm.forward(&ids[..n - 1], 0, &mut st)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let h = lm.take_taps(&mut st);
            let drafts = d
                .propose(&mut d.new_state(), &ids, &h, 3, &mut |l| match l {
                    gibbs_engine::scheduler::spec::DraftLogits::Dense(l) => argmax(l),
                    _ => unreachable!("model drafters pick dense rows"),
                })
                .unwrap_or_else(|e| panic!("{name} drafter: {e}"));
            assert!(!drafts.is_empty(), "{name}: no drafts");
            lm.forward(&ids[n - 1..], n - 1, &mut st)
        }
        None => lm.forward(&ids, 0, &mut st),
    }
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    for pos in n..n + 2 {
        assert!(l.iter().all(|x| x.is_finite()), "{name}: non-finite logits");
        l = lm
            .forward(&[argmax(&l)], pos, &mut st)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

type Specs = Vec<(String, Vec<usize>)>;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

fn lin(s: &mut Specs, p: &str, i: usize, o: usize, bias: bool) {
    s.push((format!("{p}.weight"), vec![o, i]));
    if bias {
        s.push((format!("{p}.bias"), vec![o]));
    }
}
fn conv(s: &mut Specs, p: &str, i: usize, o: usize, k: usize) {
    s.push((format!("{p}.weight"), vec![o, i, k, k]));
    s.push((format!("{p}.bias"), vec![o]));
}
fn nrm(s: &mut Specs, p: &str, c: usize) {
    s.push((format!("{p}.weight"), vec![c]));
    s.push((format!("{p}.bias"), vec![c]));
}

fn write_f32_safetensors(
    path: &std::path::Path,
    tensors: &[(String, Vec<usize>, Vec<f32>)],
) -> Result<(), String> {
    let (mut header, mut data) = (serde_json::Map::new(), Vec::<u8>::new());
    for (name, shape, v) in tensors {
        let start = data.len();
        data.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut h =
        serde_json::to_vec(&serde_json::Value::Object(header)).map_err(|e| e.to_string())?;
    h.resize(h.len().next_multiple_of(8), b' ');
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    std::fs::write(path, out).map_err(|e| format!("{}: {e}", path.display()))
}

fn values(specs: &Specs, seed: u64) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut r = Lcg(seed);
    specs
        .iter()
        .map(|(n, s)| {
            let len: usize = s.iter().product();
            let fan: usize = s[1..].iter().product::<usize>().max(1);
            let v = match s.len() {
                1 if n.ends_with("weight") => (0..len).map(|_| 1.0 + 0.1 * r.next()).collect(),
                1 => (0..len).map(|_| 0.1 * r.next()).collect(),
                _ => (0..len).map(|_| r.next() / (fan as f32).sqrt()).collect(),
            };
            (n.clone(), s.clone(), v)
        })
        .collect()
}

fn tiny_vae_on_gpu() -> Result<(), String> {
    use gibbs_engine::arch::expr::Val;
    use gibbs_engine::graph::Tensor;
    let latent = 4;
    let mut s = Specs::new();
    let conv = |s: &mut Specs, p: &str, i: usize, o: usize, k: usize| {
        s.push((format!("first_stage_model.{p}.weight"), vec![o, i, k, k]));
        s.push((format!("first_stage_model.{p}.bias"), vec![o]));
    };
    let norm = |s: &mut Specs, p: &str, c: usize| {
        s.push((format!("first_stage_model.{p}.weight"), vec![c]));
        s.push((format!("first_stage_model.{p}.bias"), vec![c]));
    };
    fn resnet(s: &mut Specs, p: &str, i: usize, o: usize) {
        for (n, c) in [("norm1", i), ("norm2", o)] {
            s.push((format!("first_stage_model.{p}.{n}.weight"), vec![c]));
            s.push((format!("first_stage_model.{p}.{n}.bias"), vec![c]));
        }
        for (n, ci, co, k) in [("conv1", i, o, 3), ("conv2", o, o, 3)] {
            s.push((
                format!("first_stage_model.{p}.{n}.weight"),
                vec![co, ci, k, k],
            ));
            s.push((format!("first_stage_model.{p}.{n}.bias"), vec![co]));
        }
        if i != o {
            s.push((
                format!("first_stage_model.{p}.nin_shortcut.weight"),
                vec![o, i, 1, 1],
            ));
            s.push((format!("first_stage_model.{p}.nin_shortcut.bias"), vec![o]));
        }
    }
    let mid = |s: &mut Specs, p: &str, c: usize| {
        resnet(s, &format!("{p}.mid.block_1"), c, c);
        resnet(s, &format!("{p}.mid.block_2"), c, c);
        s.push((
            format!("first_stage_model.{p}.mid.attn_1.norm.weight"),
            vec![c],
        ));
        s.push((
            format!("first_stage_model.{p}.mid.attn_1.norm.bias"),
            vec![c],
        ));
        for n in ["q", "k", "v", "proj_out"] {
            s.push((
                format!("first_stage_model.{p}.mid.attn_1.{n}.weight"),
                vec![c, c, 1, 1],
            ));
            s.push((
                format!("first_stage_model.{p}.mid.attn_1.{n}.bias"),
                vec![c],
            ));
        }
    };
    conv(&mut s, "encoder.conv_in", 3, 32, 3);
    resnet(&mut s, "encoder.down.0.block.0", 32, 32);
    conv(&mut s, "encoder.down.0.downsample.conv", 32, 32, 3);
    resnet(&mut s, "encoder.down.1.block.0", 32, 64);
    mid(&mut s, "encoder", 64);
    norm(&mut s, "encoder.norm_out", 64);
    conv(&mut s, "encoder.conv_out", 64, 2 * latent, 3);
    conv(&mut s, "quant_conv", 2 * latent, 2 * latent, 1);
    conv(&mut s, "post_quant_conv", latent, latent, 1);
    conv(&mut s, "decoder.conv_in", latent, 64, 3);
    mid(&mut s, "decoder", 64);
    resnet(&mut s, "decoder.up.1.block.0", 64, 64);
    resnet(&mut s, "decoder.up.1.block.1", 64, 64);
    conv(&mut s, "decoder.up.1.upsample.conv", 64, 64, 3);
    resnet(&mut s, "decoder.up.0.block.0", 64, 32);
    resnet(&mut s, "decoder.up.0.block.1", 32, 32);
    norm(&mut s, "decoder.norm_out", 32);
    conv(&mut s, "decoder.conv_out", 32, 3, 3);
    let d = std::env::temp_dir().join(format!("gibbs-tiny-vae-{}", std::process::id()));
    std::fs::create_dir_all(&d).map_err(|e| e.to_string())?;
    write_f32_safetensors(
        &d.join("diffusion_pytorch_model.safetensors"),
        &values(&s, 7),
    )?;
    let de = gibbs_engine::runtime::diffusion::Net::open(
        &d,
        "sd_vae",
        "decoder",
        PlacementMode::Pin(Device::Metal),
    )?;
    let (h, w) = (6, 5);
    let vars = [("h", Val::Num(h as f64)), ("w", Val::Num(w as f64))];
    let img = de.run_one(
        &vars,
        &[("z", Tensor::F32(Lcg(3).vec(4 * h * w, 1.0)))],
        &[],
        None,
    )?;
    std::fs::remove_dir_all(&d).ok();
    if img.len() != 3 * 2 * h * 2 * w {
        return Err(format!("sd vae: {} values", img.len()));
    }
    Ok(())
}

fn tiny_video_on_gpu() -> Result<(), String> {
    use gibbs_engine::arch::expr::Val;
    use gibbs_engine::graph::Tensor;
    use gibbs_engine::runtime::diffusion::Net;
    let tmp = tempfile::tempdir().map_err(|e| e.to_string())?;
    let num = |v: usize| Val::Num(v as f64);

    let (d, ti) = (64, 32);
    let mut s = Specs::new();
    for (p, o, i) in [
        ("proj_in", d, 128),
        ("time_embed.emb.timestep_embedder.linear_1", d, 256),
        ("time_embed.emb.timestep_embedder.linear_2", d, d),
        ("time_embed.linear", 6 * d, d),
        ("caption_projection.linear_1", d, ti),
        ("caption_projection.linear_2", d, d),
        ("proj_out", 128, d),
    ] {
        lin(&mut s, p, i, o, true);
    }
    let b = "transformer_blocks.0";
    for a in ["attn1", "attn2"] {
        for n in ["to_q", "to_k", "to_v", "to_out.0"] {
            lin(&mut s, &format!("{b}.{a}.{n}"), d, d, true);
        }
        s.push((format!("{b}.{a}.norm_q.weight"), vec![d]));
        s.push((format!("{b}.{a}.norm_k.weight"), vec![d]));
    }
    lin(&mut s, &format!("{b}.ff.net.0.proj"), d, 4 * d, true);
    lin(&mut s, &format!("{b}.ff.net.2"), 4 * d, d, true);
    s.push((format!("{b}.scale_shift_table"), vec![6, d]));
    s.push(("scale_shift_table".into(), vec![2, d]));
    let dir = tmp.path().join("ltx");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    write_f32_safetensors(
        &dir.join("diffusion_pytorch_model.safetensors"),
        &values(&s, 3),
    )?;
    let net = Net::open(&dir, "ltx", "denoiser", PlacementMode::Pin(Device::Metal))?;
    let (f, h, w, nt) = (2, 2, 2, 3);
    let vars = [
        ("frames", num(f)),
        ("h", num(h)),
        ("w", num(w)),
        ("nt", num(nt)),
        ("clean", num(h * w)),
    ];
    let given = [
        ("x", Tensor::F32(Lcg(4).vec(f * h * w * 128, 1.0))),
        ("context", Tensor::F32(Lcg(5).vec(nt * ti, 1.0))),
    ];
    let v = net.run_one(&vars, &given, &[("t", 500.0)], None)?;
    if v.iter().any(|x| !x.is_finite()) {
        return Err("non-finite LTX output".into());
    }

    let (ch, p, lat) = (8, 2, 8);
    let mut s = Specs::new();
    let c3 = |s: &mut Specs, n: &str, ci: usize, co: usize, k: usize| {
        s.push((format!("{n}.weight"), vec![co, ci, k, k, k]));
        s.push((format!("{n}.bias"), vec![co]));
    };
    let res = |s: &mut Specs, b: &str| {
        c3(s, &format!("{b}.conv1.conv"), ch, ch, 3);
        c3(s, &format!("{b}.conv2.conv"), ch, ch, 3);
    };
    c3(&mut s, "decoder.conv_in.conv", lat, ch, 3);
    res(&mut s, "decoder.mid_block.resnets.0");
    for i in 0..4 {
        if i == 3 {
            c3(
                &mut s,
                "decoder.up_blocks.3.upsamplers.0.conv.conv",
                ch,
                8 * ch,
                3,
            );
        }
        res(&mut s, &format!("decoder.up_blocks.{i}.resnets.0"));
    }
    c3(&mut s, "decoder.conv_out.conv", ch, 3 * p * p, 3);
    c3(&mut s, "encoder.conv_in.conv", 3 * p * p, ch, 3);
    for i in 0..4 {
        res(&mut s, &format!("encoder.down_blocks.{i}.resnets.0"));
        if i == 0 {
            c3(
                &mut s,
                "encoder.down_blocks.0.downsamplers.0.conv",
                ch,
                ch,
                3,
            );
        }
    }
    res(&mut s, "encoder.mid_block.resnets.0");
    c3(&mut s, "encoder.conv_out.conv", ch, lat + 1, 3);
    s.push(("latents_mean".into(), vec![lat]));
    s.push(("latents_std".into(), vec![lat]));
    let dir = tmp.path().join("ltx_vae");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    write_f32_safetensors(
        &dir.join("diffusion_pytorch_model.safetensors"),
        &values(&s, 9),
    )?;
    let cfg = serde_json::json!({"_class_name": "AutoencoderKLLTXVideo", "block_out_channels": [ch, ch, ch, ch],
        "layers_per_block": [1, 1, 1, 1, 1], "spatio_temporal_scaling": [true, false, false, false], "patch_size": p, "latent_channels": lat});
    std::fs::write(dir.join("config.json"), cfg.to_string()).map_err(|e| e.to_string())?;
    let mode = PlacementMode::Pin(Device::Metal);
    let dec = Net::open(&dir, "ltx_vae", "decoder", mode)?;
    let (f, h, w) = (2, 2, 3);
    let vars = [("frames", num(f)), ("h", num(h)), ("w", num(w))];
    let rgb = dec.run_one(
        &vars,
        &[("z", Tensor::F32(Lcg(10).vec(f * lat * h * w, 1.0)))],
        &[],
        None,
    )?;
    if rgb.len() != 3 * 3 * (2 * p * h) * (2 * p * w) || rgb.iter().any(|x| !x.is_finite()) {
        return Err(format!("LTX VAE decode: {} values", rgb.len()));
    }
    let enc = Net::open(&dir, "ltx_vae", "encoder", mode)?;
    let img = Lcg(11).vec(3 * (2 * p * h) * (2 * p * w), 1.0);
    let z = enc.run_one(
        &[("h", num(h)), ("w", num(w))],
        &[("image", Tensor::F32(img))],
        &[],
        None,
    )?;
    if z.len() != lat * h * w || z.iter().any(|x| !x.is_finite()) {
        return Err(format!("LTX VAE encode: {} values", z.len()));
    }

    let (c, ctx) = (32, 24);
    let mut s = Specs::new();
    let temporal = |s: &mut Specs, p: &str, c: usize, kv: usize| {
        nrm(s, &format!("{p}.norm"), c);
        lin(s, &format!("{p}.proj_in"), c, c, true);
        lin(s, &format!("{p}.proj_out"), c, c, true);
        let t = format!("{p}.transformer_blocks.0");
        for n in ["norm1", "norm2", "norm3"] {
            nrm(s, &format!("{t}.{n}"), c);
        }
        for (a, kv) in [("attn1", c), ("attn2", kv)] {
            lin(s, &format!("{t}.{a}.to_q"), c, c, false);
            lin(s, &format!("{t}.{a}.to_k"), kv, c, false);
            lin(s, &format!("{t}.{a}.to_v"), kv, c, false);
            lin(s, &format!("{t}.{a}.to_out.0"), c, c, true);
        }
        lin(s, &format!("{t}.ff.net.0.proj"), c, 8 * c, true);
        lin(s, &format!("{t}.ff.net.2"), 4 * c, c, true);
    };
    let layer = |s: &mut Specs, base: &str, j: usize, ci: usize, attn: bool| {
        let r = format!("{base}.resnets.{j}");
        nrm(s, &format!("{r}.norm1"), ci);
        conv(s, &format!("{r}.conv1"), ci, c, 3);
        lin(s, &format!("{r}.time_emb_proj"), 4 * c, c, true);
        nrm(s, &format!("{r}.norm2"), c);
        conv(s, &format!("{r}.conv2"), c, c, 3);
        if ci != c {
            conv(s, &format!("{r}.conv_shortcut"), ci, c, 1);
        }
        for i in 1..=4 {
            nrm(s, &format!("{base}.temp_convs.{j}.conv{i}.0"), c);
            let cw = format!(
                "{base}.temp_convs.{j}.conv{i}.{}",
                if i == 1 { 2 } else { 3 }
            );
            s.push((format!("{cw}.weight"), vec![c, c, 3, 1, 1]));
            s.push((format!("{cw}.bias"), vec![c]));
        }
        if attn {
            temporal(s, &format!("{base}.attentions.{j}"), c, ctx);
            temporal(s, &format!("{base}.temp_attentions.{j}"), c, c);
        }
    };
    lin(&mut s, "time_embedding.linear_1", c, 4 * c, true);
    lin(&mut s, "time_embedding.linear_2", 4 * c, 4 * c, true);
    conv(&mut s, "conv_in", 4, c, 3);
    temporal(&mut s, "transformer_in", c, c);
    for (i, attn) in [(0, true), (1, false)] {
        for j in 0..2 {
            layer(&mut s, &format!("down_blocks.{i}"), j, c, attn);
        }
    }
    conv(&mut s, "down_blocks.0.downsamplers.0.conv", c, c, 3);
    layer(&mut s, "mid_block", 0, c, false);
    temporal(&mut s, "mid_block.attentions.0", c, ctx);
    temporal(&mut s, "mid_block.temp_attentions.0", c, c);
    layer(&mut s, "mid_block", 1, c, false);
    for (i, attn) in [(0, false), (1, true)] {
        for j in 0..3 {
            layer(&mut s, &format!("up_blocks.{i}"), j, 2 * c, attn);
        }
    }
    conv(&mut s, "up_blocks.0.upsamplers.0.conv", c, c, 3);
    nrm(&mut s, "conv_norm_out", c);
    conv(&mut s, "conv_out", c, 4, 3);
    let dir = tmp.path().join("unet3d");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    write_f32_safetensors(
        &dir.join("diffusion_pytorch_model.safetensors"),
        &values(&s, 6),
    )?;
    let cfg = serde_json::json!({"_class_name": "UNet3DConditionModel", "in_channels": 4, "block_out_channels": [c, c],
        "cross_attention_dim": ctx, "attention_head_dim": 32, "layers_per_block": 2, "norm_num_groups": 32});
    std::fs::write(dir.join("config.json"), cfg.to_string()).map_err(|e| e.to_string())?;
    let net = Net::open(
        &dir,
        "modelscope_t2v",
        "denoiser",
        PlacementMode::Pin(Device::Metal),
    )?;
    let (f, h, w, nt) = (2, 8, 8, 3);
    let vars = [
        ("frames", num(f)),
        ("h", num(h)),
        ("w", num(w)),
        ("nt", num(nt)),
    ];
    let given = [
        ("x", Tensor::F32(Lcg(7).vec(f * 4 * h * w, 1.0))),
        ("context", Tensor::F32(Lcg(8).vec(nt * ctx, 1.0))),
    ];
    let e = net.run_one(&vars, &given, &[("t", 900.0)], None)?;
    if e.len() != f * 4 * h * w || e.iter().any(|x| !x.is_finite()) {
        return Err("bad UNet3D output".into());
    }
    Ok(())
}

/// Tiny BERT GGUF file run through `SequenceClassifier::from_gguf` on Metal.
fn bert_gguf_on_gpu() {
    use gibbs_engine::model::classifier::SequenceClassifier;
    const H: usize = 16;
    const INTER: usize = 32;
    const POS: usize = 16;
    const TOKENS: [&str; 8] = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "\u{2581}hello",
        "\u{2581}world",
        "\u{2581}the",
        "ing",
    ];
    let s = |v: &str| {
        let mut b = (v.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(v.as_bytes());
        b
    };
    let mut r = Lcg(42);
    let mut ts: Vec<(String, Vec<usize>, u32, Vec<u8>)> = Vec::new();
    let mut push = |name: String, shape: &[usize], gain: bool| {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|_| {
                if gain {
                    1.0 + 0.1 * r.next()
                } else {
                    0.3 * r.next()
                }
            })
            .collect();
        let dims: Vec<usize> = shape.iter().rev().copied().collect();
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        ts.push((name, dims, 0, bytes));
    };
    push("token_embd.weight".into(), &[TOKENS.len(), H], false);
    push("position_embd.weight".into(), &[POS, H], false);
    push("token_types.weight".into(), &[2, H], false);
    push("token_embd_norm.weight".into(), &[H], true);
    push("token_embd_norm.bias".into(), &[H], false);
    for (m, o, i) in [
        ("attn_q", H, H),
        ("attn_k", H, H),
        ("attn_v", H, H),
        ("attn_output", H, H),
        ("ffn_up", INTER, H),
        ("ffn_down", H, INTER),
    ] {
        push(format!("blk.0.{m}.weight"), &[o, i], false);
        push(format!("blk.0.{m}.bias"), &[o], false);
    }
    for m in ["attn_output_norm", "layer_output_norm"] {
        push(format!("blk.0.{m}.weight"), &[H], true);
        push(format!("blk.0.{m}.bias"), &[H], false);
    }
    push("cls.output.weight".into(), &[3, H], false);
    push("cls.output.bias".into(), &[3], false);
    let arr = |elem: u32, items: Vec<Vec<u8>>| {
        let mut b = elem.to_le_bytes().to_vec();
        b.extend_from_slice(&(items.len() as u64).to_le_bytes());
        items.into_iter().for_each(|i| b.extend(i));
        b
    };
    let u32b = |v: u32| v.to_le_bytes().to_vec();
    let kv = [
        ("general.architecture", 8u32, s("bert")),
        ("bert.embedding_length", 4, u32b(H as u32)),
        ("bert.block_count", 4, u32b(1)),
        ("bert.attention.head_count", 4, u32b(2)),
        ("bert.feed_forward_length", 4, u32b(INTER as u32)),
        (
            "bert.attention.layer_norm_epsilon",
            6,
            1e-12f32.to_le_bytes().to_vec(),
        ),
        ("tokenizer.ggml.model", 8, s("bert")),
        (
            "tokenizer.ggml.tokens",
            9,
            arr(8, TOKENS.iter().map(|t| s(t)).collect()),
        ),
        (
            "tokenizer.ggml.token_type",
            9,
            arr(
                5,
                (0..TOKENS.len())
                    .map(|i| u32b(if i < 4 { 3 } else { 1 }))
                    .collect(),
            ),
        ),
    ];
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&(ts.len() as u64).to_le_bytes());
    b.extend_from_slice(&(kv.len() as u64).to_le_bytes());
    for (k, t, payload) in &kv {
        b.extend_from_slice(&s(k));
        b.extend_from_slice(&t.to_le_bytes());
        b.extend_from_slice(payload);
    }
    let mut off = 0u64;
    for (n, dims, ttype, raw) in &ts {
        b.extend_from_slice(&s(n));
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims {
            b.extend_from_slice(&(d as u64).to_le_bytes());
        }
        b.extend_from_slice(&ttype.to_le_bytes());
        b.extend_from_slice(&off.to_le_bytes());
        off += (raw.len() as u64).div_ceil(32) * 32;
    }
    b.resize(b.len().div_ceil(32) * 32, 0);
    for (_, _, _, raw) in &ts {
        b.extend_from_slice(raw);
        b.resize(b.len().div_ceil(32) * 32, 0);
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, b).unwrap();
    let c = SequenceClassifier::from_gguf("t/bert_gguf", &path, Device::Metal)
        .unwrap_or_else(|e| panic!("bert_gguf: {e}"));
    assert_eq!(c.device(), Device::Metal, "bert_gguf");
    let before = cpu_compute::entered();
    let (scores, _) = c
        .classify("hello world", None)
        .unwrap_or_else(|e| panic!("bert_gguf: {e}"));
    assert!(
        scores.iter().all(|s| s.score.is_finite()),
        "bert_gguf: non-finite scores"
    );
    assert_eq!(
        cpu_compute::entered(),
        before,
        "bert_gguf: entered CPU compute"
    );
}

/// Families that are pure rename-rule or single-file wrappers over already-exercised components.
const EXEMPT: &[(&str, &str)] = &[
    (
        "sd1_file",
        "single-file wrapper over clip_text + sd + sd_vae",
    ),
    (
        "sd2_file",
        "single-file wrapper over clip_text + sd + sd_vae",
    ),
    (
        "sdxl_file",
        "single-file wrapper over clip_text + sd + sd_vae",
    ),
    ("sd2_gguf", "GGUF wrapper over clip_text + sd + sd_vae"),
    ("fsmt", "extends bart (FSMT config/rename rules)"),
];

fn gpu_fixture(family: &str) -> Outcome {
    if let Some(&(_, reason)) = EXEMPT.iter().find(|&&(f, _)| f == family) {
        return Outcome::Exempt(reason);
    }
    match family {
        "llama" => {
            classifier_on_gpu("llama-seqcls", synth::llama_seqcls());
            spec_fixture_on_gpu(
                "llama spec",
                "lm_graphlm_llama",
                Some("llama"),
                &[5, 17, 42, 99],
                &[],
            )
        }
        "glm" => spec_fixture_on_gpu(
            "glm",
            "lm_graphlm_qk_norm",
            Some("glm"),
            &[5, 17, 42, 99],
            &[],
        ),
        "granite" => spec_fixture_on_gpu(
            "granite",
            "lm_graphlm_granite",
            Some("granite"),
            &[5, 17, 42, 99],
            &[],
        ),
        "bitnet" => {
            spec_fixture_on_gpu("bitnet", "lm_bitnet", Some("bitnet"), &[5, 17, 42, 99], &[])
        }
        "gpt_neox" => {
            classifier_on_gpu("gpt-neox-seqcls", synth::gpt_neox_seqcls());
            Outcome::Ran
        }
        "gpt2" => spec_fixture_on_gpu(
            "gpt2",
            "enc_gpt2/models--gibbs-golden--enc_gpt2/snapshots/main",
            Some("gpt2"),
            &[2, 5, 7],
            &[],
        ),
        "opt" => spec_fixture_on_gpu(
            "opt",
            "enc_opt_post_ln/models--gibbs-golden--enc_opt_post_ln/snapshots/main",
            Some("opt"),
            &[2, 5, 7],
            &[],
        ),
        "lfm2" => spec_fixture_on_gpu(
            "lfm2",
            "lm_graphlm_lfm2",
            Some("lfm2"),
            &[5, 17, 42, 99],
            &[],
        ),
        "nano" => spec_fixture_on_gpu("nano", "lm_nano", Some("nano"), &[5, 17, 42, 99], &[]),
        "gemma2" => net_on_gpu(
            "gemma2",
            "diffa_sana/text_encoder",
            "text",
            &[("rows", 4.0)],
            &[],
        ),
        "gemma3" => spec_fixture_on_gpu(
            "gemma3",
            "gemma_gemma3_seqcls/models--gibbs-golden--gemma3-seqcls-tiny/snapshots/main",
            Some("gemma3"),
            &[2, 5, 7],
            &[],
        ),
        "gemma4" => {
            let r1 = spec_fixture_on_gpu(
                "Gemma4",
                "gemma_gemma4_ple",
                Some("gemma4"),
                &[2, 10, 11, 12],
                &[],
            );
            let _ = spec_fixture_on_gpu(
                "Gemma4 MoE",
                "gemma_gemma4_moe",
                Some("gemma4"),
                &[2, 10, 11, 12],
                &[],
            );
            let img = gibbs_engine::model::image::ImageBuffer::new(32, 24).to_png();
            let media = [gibbs_engine::model::native::MediaInput::Image(img)];
            let _ = spec_fixture_on_gpu(
                "Gemma4Media vision",
                "gemma_gemma4_image_prefill",
                Some("gemma4"),
                &[2, 10, 120, 12],
                &media,
            );
            let media = [gibbs_engine::model::native::MediaInput::Audio(tone_wav())];
            let _ = spec_fixture_on_gpu(
                "Audio conformer",
                "gemma_gemma4_audio",
                Some("gemma4"),
                &[2, 10, 121, 12],
                &media,
            );
            scatter_fixture("gemma4");
            r1
        }
        "gemma4_unified" => {
            let img = gibbs_engine::model::image::ImageBuffer::new(32, 24).to_png();
            let media = [gibbs_engine::model::native::MediaInput::Image(img)];
            spec_fixture_on_gpu(
                "gemma4_unified",
                "gemma_gemma4_unified_vision",
                Some("gemma4_unified"),
                &[2, 10, 120, 12],
                &media,
            )
        }
        "qwen3" => spec_fixture_on_gpu(
            "qwen3",
            "qwen_qwen3_decode",
            Some("qwen3"),
            &[5, 17, 42, 60],
            &[],
        ),
        "hunyuan" => spec_fixture_on_gpu(
            "hunyuan",
            "qwen_hunyuan_v1_dense_decode",
            Some("hunyuan"),
            &[5, 17, 42, 60],
            &[],
        ),
        "qwen3_5" => {
            qwen35_on_gpu("qwen3_5", false, false);
            qwen35_on_gpu("qwen3_5_moe", true, false);
            qwen35_on_gpu("qwen3_5_vl", false, true);
            Outcome::Ran
        }
        "qwen3_vl" => net_on_gpu(
            "krea2.text_encoder",
            "text_enc_qwen3_vl_spec",
            "text",
            &[("rows", 4.0)],
            &[],
        ),
        "qwen4_exp" => spec_fixture_on_gpu(
            "qwen4_exp",
            "qwen_qwen4_exp",
            Some("qwen4_exp"),
            &[5, 17, 42, 60],
            &[],
        ),
        "dflash" => {
            let dir = find_fixture("qwen_dflash");
            let d = gibbs_engine::runtime::draft::load_drafter(
                &artifacts(&dir.join("draft")),
                &artifacts(&dir.join("target")),
                PlacementMode::Pin(Device::Metal),
            )
            .unwrap_or_else(|e| panic!("dflash: {e}"));
            let taps = d.tap_layers();
            let hidden = Lcg(0x7A95).vec(5 * taps.len() * 64, 1.0);
            let toks = vec![5u32, 17, 33, 48, 71, 12];
            let mut st = d.new_state();
            let before = cpu_compute::entered();
            let ids = d
                .propose(&mut st, &toks, &hidden, d.max_k(), &mut |_| 5)
                .unwrap_or_else(|e| panic!("dflash propose: {e}"));
            assert!(!ids.is_empty(), "dflash produced no drafts");
            assert_eq!(
                cpu_compute::entered(),
                before,
                "dflash: entered CPU compute"
            );
            Outcome::Ran
        }
        "idefics3" => {
            let img = gibbs_engine::model::image::ImageBuffer::new(24, 24).to_png();
            let media = [gibbs_engine::model::native::MediaInput::Image(img)];
            let r = spec_fixture_on_gpu(
                "idefics3 spec",
                "enc_smolvlm_grid",
                Some("idefics3"),
                &[1, 10, 11, IDEFICS3_IMG, 12, 13],
                &media,
            );
            scatter_fixture("idefics3");
            r
        }
        "t5" => {
            encdec_on_gpu("t5", false);
            Outcome::Ran
        }
        "bart" => {
            encdec_on_gpu("bart", true);
            Outcome::Ran
        }
        "marian" => net_on_gpu("marian", "encdec_marian", "encoder", &[("rows", 4.0)], &[]),
        "m2m100" => net_on_gpu("m2m100", "encdec_m2m100", "encoder", &[("rows", 4.0)], &[]),
        "bert" => {
            classifier_on_gpu("bert", synth::bert(None));
            use gibbs_engine::model::hub::HuggingFaceHub;
            use gibbs_engine::runtime::encoder::system1::JevSystem1Model;
            let mut ts = synth::Ts::new(3);
            ts.w("embeddings.word_embeddings.weight", &[synth::VOCAB, 16]);
            ts.w("embeddings.position_embeddings.weight", &[32, 16]);
            ts.w("embeddings.token_type_embeddings.weight", &[2, 16]);
            ts.ln("embeddings.LayerNorm", 16);
            let p = "encoder.layer.0";
            for s in [
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
                "attention.output.dense",
            ] {
                ts.lin(&format!("{p}.{s}"), 16, 16);
            }
            ts.ln(&format!("{p}.attention.output.LayerNorm"), 16);
            ts.lin(&format!("{p}.intermediate.dense"), 32, 16);
            ts.lin(&format!("{p}.output.dense"), 16, 32);
            ts.ln(&format!("{p}.output.LayerNorm"), 16);
            let cfg = serde_json::json!({"architectures": ["BertModel"], "model_type": "bert", "hidden_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 2, "intermediate_size": 32,
                "max_position_embeddings": 32, "type_vocab_size": 2, "layer_norm_eps": 1e-5});
            let hub = HuggingFaceHub::with_cache_dir(synth::write_repo("system1", cfg, &ts));
            let m = JevSystem1Model::load(&hub, "t/system1", Device::Metal)
                .expect("System 1 loads on Metal");
            assert_eq!(m.encoder().map(|e| e.device()), Some(Device::Metal));
            for _ in 0..3 {
                assert_eq!(m.embed(synth::TEXT).expect("System 1 runs").len(), 16);
            }
            Outcome::Ran
        }
        "bert_gguf" => {
            bert_gguf_on_gpu();
            Outcome::Ran
        }
        "fasttext" => {
            let path = find_fixture("enc_fasttext/model.bin");
            let c = gibbs_engine::model::classifier::SequenceClassifier::from_fasttext(
                "golden/fasttext",
                &path,
                Device::Metal,
            )
            .unwrap_or_else(|e| panic!("fasttext: {e}"));
            assert_eq!(c.device(), Device::Metal, "fasttext");
            let before = cpu_compute::entered();
            let (scores, _) = c
                .classify("the cat sat", None)
                .unwrap_or_else(|e| panic!("fasttext: {e}"));
            assert!(
                scores.iter().all(|s| s.score.is_finite()),
                "fasttext: non-finite scores"
            );
            assert_eq!(
                cpu_compute::entered(),
                before,
                "fasttext: entered CPU compute"
            );
            Outcome::Ran
        }
        "modernbert" => {
            classifier_on_gpu("modernbert", synth::modernbert("mean", true));
            Outcome::Ran
        }
        "electra" => {
            classifier_on_gpu("electra", synth::electra());
            Outcome::Ran
        }
        "albert" => {
            classifier_on_gpu("albert", synth::albert());
            Outcome::Ran
        }
        "deberta" => {
            classifier_on_gpu("deberta-v2", synth::deberta_v2());
            classifier_on_gpu("deberta-v1", synth::deberta_v1());
            Outcome::Ran
        }
        "clip_text" => net_on_gpu(
            "clip_text",
            "diffa_clip_quick_gelu/text_encoder",
            "text",
            &[("rows", 77.0)],
            &[],
        ),
        "sd" => net_on_gpu(
            "sd",
            "diffa_unet_sd/unet",
            "denoiser",
            &[
                ("b", 1.0),
                ("h", 8.0),
                ("w", 8.0),
                ("nt", 4.0),
                ("adm", 0.0),
            ],
            &[("t", 500.0)],
        ),
        "sdxl" => net_on_gpu(
            "sdxl",
            "diffa_unet_xl_lcm/unet",
            "denoiser",
            &[
                ("b", 1.0),
                ("h", 8.0),
                ("w", 8.0),
                ("nt", 4.0),
                ("adm", 56.0),
            ],
            &[("t", 500.0)],
        ),
        "sd_vae" => {
            tiny_vae_on_gpu().expect("tiny VAEs run on Metal");
            Outcome::Ran
        }
        "dcae" => net_on_gpu(
            "dcae",
            "diffa_dcae/vae",
            "decoder",
            &[("h", 2.0), ("w", 2.0)],
            &[],
        ),
        "sana" => net_on_gpu(
            "sana",
            "diffa_sana/transformer",
            "denoiser",
            &[("h", 4.0), ("w", 4.0), ("nt", 5.0)],
            &[("t", 500.0)],
        ),
        "sd3" => net_on_gpu(
            "sd3",
            "diffa_sd3_plain/transformer",
            "denoiser",
            &[("h", 4.0), ("w", 4.0), ("nt", 5.0)],
            &[("t", 500.0)],
        ),
        "flux" => net_on_gpu(
            "flux",
            "diffa_flux1/transformer",
            "denoiser",
            &[("h", 2.0), ("w", 2.0), ("nt", 4.0)],
            &[("t", 500.0), ("g", 3500.0)],
        ),
        "chroma" => net_on_gpu(
            "chroma",
            "diffa_flux_chroma/transformer",
            "denoiser",
            &[("h", 2.0), ("w", 2.0), ("nt", 4.0)],
            &[("t", 500.0)],
        ),
        "flux2" => net_on_gpu(
            "flux2",
            "diffa_flux2/transformer",
            "denoiser",
            &[("h", 2.0), ("w", 2.0), ("nt", 4.0)],
            &[("t", 500.0), ("g", 3500.0)],
        ),
        "zimage" => net_on_gpu(
            "zimage",
            "diffb_zimage",
            "denoiser",
            &[("h", 2.0), ("w", 2.0), ("nt", 4.0)],
            &[("t", 500.0)],
        ),
        "qwen_image" => net_on_gpu(
            "qwen_image",
            "diffb_qwen_image_v1",
            "denoiser",
            &[("h", 2.0), ("w", 2.0), ("nt", 4.0)],
            &[("t", 500.0)],
        ),
        "qwen_image21" => net_on_gpu(
            "qwen_image21",
            "diffb_qwen_image21",
            "denoiser",
            &[("h", 2.0), ("w", 2.0), ("nt", 4.0)],
            &[("t", 500.0)],
        ),
        "qwen_vae" => net_on_gpu(
            "qwen_vae",
            "diffb_qwen_vae",
            "decoder",
            &[("h", 2.0), ("w", 2.0)],
            &[],
        ),
        "qwen_vae21" => net_on_gpu(
            "qwen_vae21",
            "diffb_qwen_vae21",
            "decoder",
            &[("h", 2.0), ("w", 2.0)],
            &[],
        ),
        "krea2" => net_on_gpu(
            "krea2",
            "diffb_krea2",
            "denoiser",
            &[("h", 2.0), ("w", 2.0), ("nt", 6.0)],
            &[("t", 500.0)],
        ),
        "ltx" | "ltx_vae" | "modelscope_t2v" => {
            tiny_video_on_gpu().expect("tiny video denoisers run on Metal");
            Outcome::Ran
        }
        other => panic!(
            "unhandled family `{other}` in cpu_free_matrix: add a GPU fixture or EXEMPT entry"
        ),
    }
}

#[test]
fn cpu_free_matrix() {
    INIT.call_once(|| cpu_compute::ban(true));
    let reg = gibbs_engine::arch::registry().unwrap();
    let mut rows = Vec::new();
    for family in reg.family_names() {
        let spec = reg.spec(family, &Signals::default()).unwrap();
        if spec.pipeline == Pipeline::Unsupported {
            continue;
        }
        let status = if metal_present() {
            let before_nodes = gibbs_engine::graph::program::compiled_cpu_nodes();
            let before_cpu = cpu_compute::entered();
            let outcome = gpu_fixture(family);
            assert_eq!(
                gibbs_engine::graph::program::compiled_cpu_nodes(),
                before_nodes,
                "{family}: compiled CPU nodes"
            );
            if !spec.components.contains_key("precompute") {
                assert_eq!(
                    cpu_compute::entered(),
                    before_cpu,
                    "{family}: entered CPU compute"
                );
            }
            match outcome {
                Outcome::Ran => "ran".to_string(),
                Outcome::Exempt(why) => format!("exempt: {why}"),
            }
        } else {
            "no Metal device".to_string()
        };
        rows.push(format!("| {family} | {status} |"));
    }
    println!("\n| family | status |\n|:---|:---|");
    for r in rows {
        println!("{r}");
    }
}
