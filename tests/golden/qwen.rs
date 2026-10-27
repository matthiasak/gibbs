//! Qwen family goldens: Qwen3.5 (DeltaNet + gated attention, dense and MoE), Qwen3, HunYuan dense,
//! the Qwen3.5 MTP head, vision tower and image prompts, Qwen4-Exp (hyper-connections, QSA, PLE)
//! and the DFlash / DFlash2 / DSpark block drafters, on tiny bf16 safetensors checkpoints.
//! Shapes follow the (deleted) native Qwen3.5 loader's test fixtures; the JSON is the oracle.

use crate::util::*;
use gibbs_engine::hardware::{Device, PlacementMode};
use gibbs_engine::model::image::ImageBuffer;
use gibbs_engine::model::native::{MediaInput, NativeLm, SeqState};
use gibbs_engine::scheduler::spec::{DraftLogits, Drafter};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;

const H: usize = 64;
const V: usize = 96;
const HD: usize = 32;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const INTER: usize = 96;
/// DeltaNet: 1 key head, 2 value heads, 16 wide, conv kernel 4 -> conv dim 64.
const CD: usize = 64;
const IMAGE_TOKEN: u32 = 90;
const PROMPT: [u32; 7] = [5, 17, 33, 48, 71, 12, 60];

const PRE: &str = r#"{"size":{"shortest_edge":64,"longest_edge":100000},
    "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
    "patch_size":2,"temporal_patch_size":2,"merge_size":2}"#;

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

// ---------------------------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Ffn {
    Dense,
    /// 8 routed experts (HF fused `gate_up_proj`), top 2, plus a gated shared expert.
    Moe,
}

fn mlp(w: &mut Weights, p: &str, ffn: Ffn) {
    match ffn {
        Ffn::Dense => {
            w.w(format!("{p}.mlp.gate_proj.weight"), &[INTER, H])
                .w(format!("{p}.mlp.up_proj.weight"), &[INTER, H])
                .w(format!("{p}.mlp.down_proj.weight"), &[H, INTER]);
        }
        Ffn::Moe => {
            let (e, ff) = (8, 32);
            w.s(format!("{p}.mlp.gate.weight"), &[e, H], 1.0)
                .w(format!("{p}.mlp.experts.gate_up_proj"), &[e, 2 * ff, H])
                .w(format!("{p}.mlp.experts.down_proj"), &[e, H, ff])
                .w(format!("{p}.mlp.shared_expert.gate_proj.weight"), &[ff, H])
                .w(format!("{p}.mlp.shared_expert.up_proj.weight"), &[ff, H])
                .w(format!("{p}.mlp.shared_expert.down_proj.weight"), &[H, ff])
                .w(format!("{p}.mlp.shared_expert_gate.weight"), &[1, H]);
        }
    }
}

/// Gated full attention (q_proj carries the output gate), Qwen3.5 `1 + w` q/k norms.
fn gated_attn(w: &mut Weights, a: &str) {
    w.w(format!("{a}.q_proj.weight"), &[HEADS * 2 * HD, H])
        .w(format!("{a}.k_proj.weight"), &[KV_HEADS * HD, H])
        .w(format!("{a}.v_proj.weight"), &[KV_HEADS * HD, H])
        .w(format!("{a}.o_proj.weight"), &[H, HEADS * HD])
        .norm(format!("{a}.q_norm.weight"), HD, 0.0)
        .norm(format!("{a}.k_norm.weight"), HD, 0.0);
}

fn delta(w: &mut Weights, a: &str) {
    w.w(format!("{a}.in_proj_qkv.weight"), &[CD, H])
        .w(format!("{a}.in_proj_z.weight"), &[32, H])
        .w(format!("{a}.in_proj_b.weight"), &[2, H])
        .w(format!("{a}.in_proj_a.weight"), &[2, H])
        .w(format!("{a}.conv1d.weight"), &[CD, 1, 4])
        .s(format!("{a}.A_log"), &[2], 0.5)
        .s(format!("{a}.dt_bias"), &[2], 0.5)
        .norm(format!("{a}.norm.weight"), 16, 1.0)
        .w(format!("{a}.out_proj.weight"), &[H, 32]);
}

/// A hyper-connection mixer (`hc_count` 2 streams, low rank 8).
fn hc_mix(w: &mut Weights, p: &str, inject: bool) {
    let wide = 2 * H;
    w.norm(format!("{p}.hc_norm.weight"), wide, 0.0)
        .w(format!("{p}.input_mix_weight_down.weight"), &[8, wide])
        .w(format!("{p}.input_mix_weight_up.weight"), &[wide, 8]);
    if inject {
        w.w(format!("{p}.block_inject_weight.weight"), &[2, wide]);
    }
}

/// Qwen3.5 decoder under prefix `p` (`model` or `model.language_model`): layers DeltaNet,
/// DeltaNet, gated attention. `qwen4`: Qwen4-Exp hyper-connections (no layer norms, final mixer
/// instead of `norm`), a QSA indexer on the attention layer and PLE on layer 0.
fn qwen35_text(w: &mut Weights, p: &str, ffn: Ffn, qwen4: bool) {
    w.w(format!("{p}.embed_tokens.weight"), &[V, H]);
    for l in 0..3 {
        let lp = format!("{p}.layers.{l}");
        if qwen4 {
            hc_mix(w, &format!("{lp}.attn_hyper_connection"), true);
            hc_mix(w, &format!("{lp}.mlp_hyper_connection"), true);
        } else {
            w.norm(format!("{lp}.input_layernorm.weight"), H, 0.0)
                .norm(format!("{lp}.post_attention_layernorm.weight"), H, 0.0);
        }
        if l < 2 {
            delta(w, &format!("{lp}.linear_attn"));
        } else {
            let a = format!("{lp}.self_attn");
            gated_attn(w, &a);
            if qwen4 {
                w.w(format!("{a}.indexer.index_qk_proj.weight"), &[3 * 16, H])
                    .norm(format!("{a}.indexer.q_layernorm.weight"), 16, 0.0)
                    .norm(format!("{a}.indexer.k_layernorm.weight"), 16, 0.0);
            }
        }
        if qwen4 && l == 0 {
            // 2 table parts x 32 rows x 8 dims; 4 hash heads (bigram, trigram x 2 per gram).
            let (pl, wide) = (format!("{lp}.ple"), 2 * H);
            w.w(format!("{pl}.ple_embedding.table"), &[2, 32, 8])
                .w(format!("{pl}.key_proj.weight"), &[wide, 32])
                .w(format!("{pl}.value_proj.weight"), &[H, 32])
                .norm(format!("{pl}.norm_key.weight"), wide, 0.0)
                .norm(format!("{pl}.norm_query.weight"), wide, 0.0)
                .norm(format!("{pl}.norm_conv.weight"), wide, 0.0)
                .w(format!("{pl}.conv1d.weight"), &[wide, 1, 2]);
        }
        mlp(w, &lp, ffn);
    }
    if qwen4 {
        hc_mix(w, &format!("{p}.hyper_connection_mixer"), false);
    } else {
        w.norm(format!("{p}.norm.weight"), H, 0.0);
    }
    w.w("lm_head.weight", &[V, H]);
}

fn mtp(w: &mut Weights) {
    w.w("mtp.fc.weight", &[H, 2 * H])
        .norm("mtp.pre_fc_norm_embedding.weight", H, 0.0)
        .norm("mtp.pre_fc_norm_hidden.weight", H, 0.0)
        .norm("mtp.norm.weight", H, 0.0)
        .norm("mtp.layers.0.input_layernorm.weight", H, 0.0)
        .norm("mtp.layers.0.post_attention_layernorm.weight", H, 0.0);
    gated_attn(w, "mtp.layers.0.self_attn");
    mlp(w, "mtp.layers.0", Ffn::Dense);
}

/// Vision tower: hidden 32, 2 heads, 2 blocks, 4x4 position grid, patch 2, temporal 2, merge 2,
/// one deepstack merger after block 0.
fn vision(w: &mut Weights) {
    let d = 32;
    let p = "model.visual";
    w.w(format!("{p}.patch_embed.proj.weight"), &[d, 3, 2, 2, 2])
        .s(format!("{p}.patch_embed.proj.bias"), &[d], 0.1)
        .s(format!("{p}.pos_embed.weight"), &[16, d], 0.5);
    for b in 0..2 {
        let bp = format!("{p}.blocks.{b}");
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
        let mp = format!("{p}.{m}");
        w.norm(format!("{mp}.norm.weight"), wide, 1.0)
            .s(format!("{mp}.norm.bias"), &[wide], 0.1)
            .w(format!("{mp}.linear_fc1.weight"), &[4 * d, 4 * d])
            .s(format!("{mp}.linear_fc1.bias"), &[4 * d], 0.1)
            .w(format!("{mp}.linear_fc2.weight"), &[H, 4 * d])
            .s(format!("{mp}.linear_fc2.bias"), &[H], 0.1);
    }
}

/// Qwen3.5 text config (flat, as text-only exports ship it).
fn qwen35_config(model_type: &str, arch: &str) -> Value {
    json!({
        "model_type": model_type, "architectures": [arch],
        "hidden_size": H, "vocab_size": V, "num_hidden_layers": 3,
        "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS, "head_dim": HD,
        "intermediate_size": INTER, "max_position_embeddings": 128, "rms_norm_eps": 1e-6,
        "attn_output_gate": true, "tie_word_embeddings": false, "eos_token_id": 2,
        "layer_types": ["linear_attention", "linear_attention", "full_attention"],
        "linear_num_key_heads": 1, "linear_num_value_heads": 2,
        "linear_key_head_dim": 16, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
        "mtp_num_hidden_layers": 0,
        "rope_parameters": {
            "rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
            "mrope_interleaved": true, "mrope_section": [2, 1, 1]
        }
    })
}

/// Writes `config.json`, bf16 `model.safetensors` and `tokenizer.json` into `dir`.
fn write_ckpt(dir: &Path, cfg: &Value, w: &Weights) {
    write_json(&dir.join("config.json"), cfg);
    write_st_as(&dir.join("model.safetensors"), &w.tensors, StDtype::BF16);
    write_tokenizer(dir, V);
}

/// The text-only Qwen3.5 checkpoint (`mtp`: with its MTP head) in `dir`.
fn qwen35_ckpt(dir: &Path, seed: u64, mtp_head: bool) {
    let mut w = Weights::new(seed);
    qwen35_text(&mut w, "model", Ffn::Dense, false);
    let mut cfg = qwen35_config("qwen3_5", "Qwen3_5ForCausalLM");
    if mtp_head {
        mtp(&mut w);
        cfg["mtp_num_hidden_layers"] = 1.into();
    }
    write_ckpt(dir, &cfg, &w);
}

/// The multimodal Qwen3.5 checkpoint: decoder under `model.language_model`, ViT under
/// `model.visual`, nested `text_config` / `vision_config` and `preprocessor_config.json`.
fn qwen35_vl_ckpt(dir: &Path) {
    let mut w = Weights::new(0x5157_454E_0003);
    qwen35_text(&mut w, "model.language_model", Ffn::Dense, false);
    vision(&mut w);
    let mut text = qwen35_config("qwen3_5_text", "Qwen3_5ForCausalLM");
    text.as_object_mut().unwrap().remove("architectures");
    let cfg = json!({
        "model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": IMAGE_TOKEN, "text_config": text,
        "vision_config": {
            "hidden_size": 32, "num_heads": 2, "depth": 2, "num_position_embeddings": 16,
            "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2,
            "in_channels": 3, "intermediate_size": 64, "out_hidden_size": H,
            "hidden_act": "gelu_pytorch_tanh", "deepstack_visual_indexes": [0]
        }
    });
    write_ckpt(dir, &cfg, &w);
    std::fs::write(dir.join("preprocessor_config.json"), PRE).unwrap();
}

/// Plain Qwen3-style decoder (2 full-attention layers, GQA 4/2, qk-norm `w` not `1 + w`).
fn dense_weights(seed: u64, q: &str, k: &str, lm_head: bool) -> Weights {
    let (heads, kv, hd) = (4, 2, 16);
    let mut w = Weights::new(seed);
    w.w("model.embed_tokens.weight", &[V, H]);
    for l in 0..2 {
        let p = format!("model.layers.{l}");
        w.norm(format!("{p}.input_layernorm.weight"), H, 1.0)
            .norm(format!("{p}.post_attention_layernorm.weight"), H, 1.0)
            .w(format!("{p}.self_attn.q_proj.weight"), &[heads * hd, H])
            .w(format!("{p}.self_attn.k_proj.weight"), &[kv * hd, H])
            .w(format!("{p}.self_attn.v_proj.weight"), &[kv * hd, H])
            .w(format!("{p}.self_attn.o_proj.weight"), &[H, heads * hd])
            .norm(format!("{p}.self_attn.{q}.weight"), hd, 1.0)
            .norm(format!("{p}.self_attn.{k}.weight"), hd, 1.0);
        mlp(&mut w, &p, Ffn::Dense);
    }
    w.norm("model.norm.weight", H, 1.0);
    if lm_head {
        w.w("lm_head.weight", &[V, H]);
    }
    w
}

fn dense_config(model_type: &str, arch: &str) -> Value {
    json!({
        "model_type": model_type, "architectures": [arch],
        "hidden_size": H, "vocab_size": V, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "intermediate_size": INTER, "max_position_embeddings": 128, "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "tie_word_embeddings": true, "attention_bias": false,
        "eos_token_id": 2
    })
}

// ---------------------------------------------------------------------------------------------
// Runners
// ---------------------------------------------------------------------------------------------

/// runner: arch::load::load_with (bit-exact) of the detected family spec -> runtime::lm::SpecLm.
fn load(dir: &Path, id: &str, dev: Device) -> Arc<dyn NativeLm> {
    use gibbs_engine::model::weight_format::{ReencodeConfig, ReencodePolicy};
    let re = ReencodeConfig { policy: ReencodePolicy::BitExact, ..Default::default() };
    let a = artifacts(dir, id, "model.safetensors");
    let m = gibbs_engine::arch::load::load_with(&a, gibbs_engine::arch::registry().unwrap(), None, re)
        .and_then(|s| gibbs_engine::runtime::lm::SpecLm::new(s, PlacementMode::Pin(dev)))
        .unwrap_or_else(|e| panic!("{id} on {dev:?}: {e}"));
    // Catches a silent CPU fallback of a Metal load.
    assert_eq!(m.device(), dev, "{id}");
    Arc::new(m)
}

fn fwd(m: &dyn NativeLm, toks: &[u32], pos: usize, st: &mut SeqState) -> Vec<f32> {
    m.forward(toks, pos, st).unwrap()
}

fn greedy(m: &dyn NativeLm, toks: &[u32], pos: usize, st: &mut SeqState) -> u32 {
    argmax(&m.forward(toks, pos, st).unwrap())
}

/// The decode protocol shared by the text decoders: 7-token prefill, 3 decode steps, verify 4
/// tokens with `forward_all`, rewind 2 and verify 2 more; and an 8-step greedy trajectory.
fn decode_case(m: &dyn NativeLm) -> Out {
    let mut out = Out::default();
    let (ids, mut st) = m.prepare(PROMPT.to_vec(), &[]).unwrap();
    assert_eq!(ids, PROMPT);
    let mut logits = fwd(m, &PROMPT, 0, &mut st);
    out.f("prefill", &logits);
    let mut pos = PROMPT.len();
    let mut steps = Vec::new();
    for i in 0..3 {
        let tok = argmax(&logits);
        steps.push(tok);
        logits = fwd(m, &[tok], pos, &mut st);
        out.f(format!("decode{i}"), &logits);
        pos += 1;
    }
    out.ids("decode_ids", &steps);
    out.f("verify4", &m.forward_all(&[30, 31, 50, 51], pos, &mut st).unwrap());
    m.truncate(&mut st, pos + 2).unwrap();
    out.f("after_rewind", &m.forward_all(&[32, 33], pos + 2, &mut st).unwrap());

    let (_, mut st) = m.prepare(PROMPT.to_vec(), &[]).unwrap();
    let mut traj = vec![greedy(m, &PROMPT, 0, &mut st)];
    for i in 0..7 {
        let t = greedy(m, &[traj[i]], PROMPT.len() + i, &mut st);
        traj.push(t);
    }
    out.ids("greedy8", &traj);
    out
}

fn png(w: usize, h: usize) -> Vec<u8> {
    let mut img = ImageBuffer::new(w, h);
    for (i, px) in img.data.iter_mut().enumerate() {
        *px = ((i * 37 + (i / 7) * 11) % 251) as u8;
    }
    img.to_png()
}

// ---------------------------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------------------------

#[test]
fn qwen3_5_decode() {
    let dir = fixture_dir("qwen_qwen3_5_decode");
    qwen35_ckpt(&dir, 0x5157_454E_0001, false);
    golden("qwen_qwen3_5_decode", TOL, |dev| {
        decode_case(&*load(&dir, "golden/qwen3_5", dev))
    });
}

#[test]
fn qwen3_5_moe_decode() {
    let dir = fixture_dir("qwen_qwen3_5_moe_decode");
    let mut w = Weights::new(0x5157_454E_0002);
    qwen35_text(&mut w, "model", Ffn::Moe, false);
    let mut cfg = qwen35_config("qwen3_5_moe", "Qwen3_5MoeForCausalLM");
    let c = cfg.as_object_mut().unwrap();
    c.remove("intermediate_size");
    for (k, v) in [
        ("num_experts", json!(8)),
        ("num_experts_per_tok", json!(2)),
        ("norm_topk_prob", json!(true)),
        ("moe_intermediate_size", json!(32)),
        ("shared_expert_intermediate_size", json!(32)),
    ] {
        c.insert(k.into(), v);
    }
    write_ckpt(&dir, &cfg, &w);
    golden("qwen_qwen3_5_moe_decode", TOL, |dev| {
        decode_case(&*load(&dir, "golden/qwen3_5_moe", dev))
    });
}

#[test]
fn qwen3_decode() {
    let dir = fixture_dir("qwen_qwen3_decode");
    let w = dense_weights(0x5157_454E_0004, "q_norm", "k_norm", false);
    write_ckpt(&dir, &dense_config("qwen3", "Qwen3ForCausalLM"), &w);
    golden("qwen_qwen3_decode", TOL, |dev| {
        decode_case(&*load(&dir, "golden/qwen3", dev))
    });
}

/// HunYuan dense v1: q/k norms after RoPE, dynamic-NTK alpha RoPE; ships an lm_head that the tied
/// embeddings override.
#[test]
fn hunyuan_v1_dense_decode() {
    let dir = fixture_dir("qwen_hunyuan_v1_dense_decode");
    let w = dense_weights(0x5157_454E_0005, "query_layernorm", "key_layernorm", true);
    let mut cfg = dense_config("hunyuan_v1_dense", "HunYuanDenseV1ForCausalLM");
    cfg["rope_scaling"] = json!({"type": "dynamic", "alpha": 1000.0, "factor": 1.0});
    cfg["use_qk_norm"] = true.into();
    cfg["use_cla"] = false.into();
    cfg["mlp_bias"] = false.into();
    write_ckpt(&dir, &cfg, &w);
    golden("qwen_hunyuan_v1_dense_decode", TOL, |dev| {
        decode_case(&*load(&dir, "golden/hunyuan", dev))
    });
}

/// The shipped MTP drafter: draft 3 after the prompt, verify, accept 2, draft 3 again.
#[test]
fn qwen3_5_mtp() {
    let dir = fixture_dir("qwen_qwen3_5_mtp");
    qwen35_ckpt(&dir, 0x5157_454E_0006, true);
    golden("qwen_qwen3_5_mtp", TOL, |dev| {
        // runner: arch::load::load_with (bit-exact) of the `qwen3_5` spec -> runtime::lm::SpecLm
        // on `dev`; its drafter is the spec's `mtp` component (runtime::draft).
        use gibbs_engine::model::weight_format::{ReencodeConfig, ReencodePolicy};
        let re = ReencodeConfig { policy: ReencodePolicy::BitExact, ..Default::default() };
        let a = artifacts(&dir, "golden/qwen3_5_mtp", "model.safetensors");
        let spec = gibbs_engine::arch::load::load_with(&a, gibbs_engine::arch::registry().unwrap(), None, re).unwrap();
        assert_eq!(spec.spec.family, "qwen3_5");
        let m = gibbs_engine::runtime::lm::SpecLm::new(spec, PlacementMode::Pin(dev)).unwrap();
        assert_eq!(m.device(), dev);
        let d = m.drafter().expect("shipped MTP drafter");
        let mut out = Out::default();
        out.ids("taps", &d.tap_layers().iter().map(|&t| t as u32).collect::<Vec<_>>());
        let prompt = vec![4u32, 8, 15, 16, 23, 42];
        let n = prompt.len();
        let (_, mut s) = m.prepare(prompt.clone(), &[]).unwrap();
        m.set_taps(&mut s, &d.tap_layers()).unwrap();
        let mut ds = d.new_state();
        m.forward(&prompt[..n - 1], 0, &mut s).unwrap();
        let h = m.take_taps(&mut s);
        let mut draw = |toks: &[u32], h: &[f32], tag: &str, out: &mut Out| {
            let mut rows = Vec::new();
            let ids = d
                .propose(&mut ds, toks, h, 3, &mut |l| {
                    let DraftLogits::Dense(l) = l else {
                        unreachable!("model drafters pick dense rows")
                    };
                    rows.extend_from_slice(l);
                    argmax(l)
                })
                .unwrap();
            out.ids(format!("draft_{tag}"), &ids).f(format!("logits_{tag}"), &rows);
            ids
        };
        let d0 = draw(&prompt, &h, "first", &mut out);
        let ver = [prompt[n - 1], d0[0], d0[1], d0[2]];
        out.f("verify", &m.forward_all(&ver, n - 1, &mut s).unwrap());
        m.truncate(&mut s, n + 2).unwrap();
        let h = m.take_taps(&mut s);
        let mut toks = prompt.clone();
        toks.extend([d0[0], d0[1], 77]);
        draw(&toks, &h[..3 * H], "catchup", &mut out);
        out
    });
}

/// An MTP head the `mtp` drafter does not run (here: a config claiming 2 MTP layers) leaves the
/// target loadable without a shipped drafter.
#[test]
fn qwen3_5_unsupported_mtp_head_is_skipped() {
    use gibbs_engine::model::weight_format::{ReencodeConfig, ReencodePolicy};
    let dir = fixture_dir("qwen_qwen3_5_mtp2");
    let mut w = Weights::new(0x5157_454E_0006);
    qwen35_text(&mut w, "model", Ffn::Dense, false);
    mtp(&mut w);
    let mut cfg = qwen35_config("qwen3_5", "Qwen3_5ForCausalLM");
    cfg["mtp_num_hidden_layers"] = 2.into();
    write_ckpt(&dir, &cfg, &w);
    let re = ReencodeConfig { policy: ReencodePolicy::BitExact, ..Default::default() };
    let a = artifacts(&dir, "golden/qwen3_5_mtp2", "model.safetensors");
    let spec = gibbs_engine::arch::load::load_with(&a, gibbs_engine::arch::registry().unwrap(), None, re).unwrap();
    let m = gibbs_engine::runtime::lm::SpecLm::new(spec, PlacementMode::Pin(Device::Cpu)).unwrap();
    assert!(m.drafter().is_none());
}

/// The ViT alone on two image sizes (the second runs reshaped on the first's program): merged
/// features and the deepstack feature. Qwen3-VL's spec runs the same tower.
#[test]
fn qwen3_5_vision_tower() {
    use gibbs_engine::arch::{load, Registry};
    use gibbs_engine::model::weight_format::{ReencodeConfig, ReencodePolicy};
    use gibbs_engine::runtime::media::{self, Towers};
    let dir = fixture_dir("qwen_qwen3_5_vision_tower");
    qwen35_vl_ckpt(&dir);
    // runner: arch::load::load_with (bit-exact) -> runtime::media::{preprocess, Towers::run}.
    let run = |family: &str, dev: Device| {
        let re = ReencodeConfig { policy: ReencodePolicy::BitExact, ..Default::default() };
        let a = artifacts(&dir, "golden/qwen3_5_vl", "model.safetensors");
        let m = load::load_with(&a, &Registry::load(None).unwrap(), Some(family), re).unwrap();
        load::check(&m).unwrap();
        let spec = media::resolve(&m.spec.runtime["media"], &m.env).unwrap().remove("image").unwrap();
        let towers = Towers::new(PlacementMode::Pin(dev));
        let mut out = Out::default();
        for (w, h) in [(16, 24), (24, 8)] {
            let p = media::preprocess(&spec.preprocess, &png(w, h)).unwrap();
            let o: std::collections::HashMap<_, _> = towers.run(&m, &spec.component, &p).unwrap().into_iter().collect();
            out.ids(format!("grid_{w}x{h}"), &[p.grid.rows as u32, p.grid.cols as u32])
                .f(format!("embeds_{w}x{h}"), &o["media"])
                .f(format!("deepstack0_{w}x{h}"), &o["deepstack.0"]);
        }
        out
    };
    golden("qwen_qwen3_5_vision_tower", TOL, |dev| run("qwen3_5", dev));
    golden("qwen_qwen3_5_vision_tower", TOL, |dev| run("qwen3_vl", dev));
}

/// Image prompt through `prepare` (placeholder expanded to the merged grid, M-RoPE positions,
/// deepstack rows), prefill, then 4 greedy decode steps.
#[test]
fn qwen3_5_image_prompt() {
    let dir = fixture_dir("qwen_qwen3_5_image_prompt");
    qwen35_vl_ckpt(&dir);
    golden("qwen_qwen3_5_image_prompt", TOL, |dev| {
        let m = load(&dir, "golden/qwen3_5_vl", dev);
        let mut out = Out::default();
        let media = [MediaInput::Image(png(16, 24))];
        let (ids, mut st) = m.prepare(vec![3, 7, IMAGE_TOKEN, 11, 5, 9], &media).unwrap();
        out.ids("prepared_ids", &ids);
        let mut logits = fwd(&*m, &ids, 0, &mut st);
        out.f("prefill", &logits);
        let (mut pos, mut steps) = (ids.len(), Vec::new());
        for i in 0..4 {
            let tok = argmax(&logits);
            steps.push(tok);
            logits = fwd(&*m, &[tok], pos, &mut st);
            out.f(format!("decode{i}"), &logits);
            pos += 1;
        }
        out.ids("decode_ids", &steps);
        out
    });
}

/// Qwen4-Exp: 2 hyper-connection streams, a QSA indexer (ratio 2, budget 2) on the attention
/// layer, bigram/trigram PLE on layer 0 (prompt holds the EOS that cuts n-grams). Chunked prefill
/// (3 + 4) and token-by-token prefill must agree with each other and with this golden.
#[test]
fn qwen4_exp() {
    let dir = fixture_dir("qwen_qwen4_exp");
    let mut w = Weights::new(0x5157_454E_0007);
    qwen35_text(&mut w, "model", Ffn::Dense, true);
    let mut cfg = qwen35_config("qwen4_exp", "Qwen4ExpForCausalLM");
    let c = cfg.as_object_mut().unwrap();
    for (k, v) in [
        (
            "layer_types",
            json!(["linear_attention", "linear_attention", "qwen_sparse_attention"]),
        ),
        ("output_gate_type", json!("sigmoid")),
        ("hc_count", json!(2)),
        ("hc_lowrank", json!(8)),
        ("indexer_n_heads", json!(2)),
        ("indexer_head_dim", json!(16)),
        ("indexer_kv_heads", json!(1)),
        ("indexer_compress_ratio", json!(2)),
        ("indexer_budget", json!(2)),
        ("ple_layer_ids", json!([1])),
        ("ngram_size", json!(3)),
        ("heads_per_ngram", json!(2)),
        ("ple_conv_kernel_size", json!(2)),
        // Hash constants as the GGUF path passes them (`ple_hash`), not as I64 buffers.
        (
            "ple_hash",
            json!({"multipliers": [3, 5, 7], "offsets": [0, 13, 29, 40],
                   "vocab_sizes": [13, 16, 11, 16]}),
        ),
    ] {
        c.insert(k.into(), v);
    }
    write_ckpt(&dir, &cfg, &w);
    let prompt = [5u32, 17, 2, 40, 63, 9, 88];
    golden("qwen_qwen4_exp", TOL, |dev| {
        let m = load(&dir, "golden/qwen4_exp", dev);
        let host: Vec<String> = m.placement_summary().unwrap().iter()
            .filter(|r| dev == Device::Metal && r.device == Device::Cpu).map(|r| r.op.clone()).collect();
        assert!(host.is_empty(), "CPU nodes in the Metal program: {host:?}");
        let mut out = Out::default();
        let (_, mut st) = m.prepare(prompt.to_vec(), &[]).unwrap();
        out.f("chunk3", &fwd(&*m, &prompt[..3], 0, &mut st));
        let mut logits = fwd(&*m, &prompt[3..], 3, &mut st);
        out.f("chunk7", &logits);
        let mut steps = Vec::new();
        for i in 0..4 {
            let tok = argmax(&logits);
            steps.push(tok);
            logits = fwd(&*m, &[tok], prompt.len() + i, &mut st);
        }
        out.ids("decode_ids", &steps).f("decode_last", &logits);
        // Rewind into a verified span: the PLE n-gram history and conv ring rewind with it.
        let at = prompt.len() + 4;
        let both = m.forward_all(&[30, 31], at, &mut st).unwrap();
        m.truncate(&mut st, at + 1).unwrap();
        let again = fwd(&*m, &[31], at + 1, &mut st);
        let d = again.iter().zip(&both[V..]).fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(d < 1e-3, "{dev:?}: rewound row differs by {d}");
        let (_, mut st) = m.prepare(prompt.to_vec(), &[]).unwrap();
        let mut inc = Vec::new();
        for (i, &t) in prompt.iter().enumerate() {
            inc.extend(fwd(&*m, &[t], i, &mut st));
        }
        out.f("incremental", &inc);
        out
    });
}

// ---------------------------------------------------------------------------------------------
// Block drafters
// ---------------------------------------------------------------------------------------------

/// One-or-two-layer z-lab style drafter (hidden 64, 2 heads x 32, 1 kv head, ffn 96) reading
/// target taps 0 and 2.
fn drafter_weights(seed: u64, layers: usize, conv: bool) -> Weights {
    let mut w = Weights::new(seed);
    w.w("fc.weight", &[H, 2 * H])
        .norm("hidden_norm.weight", H, 1.0)
        .norm("norm.weight", H, 1.0);
    for l in 0..layers {
        let p = format!("layers.{l}");
        w.norm(format!("{p}.input_layernorm.weight"), H, 1.0)
            .norm(format!("{p}.post_attention_layernorm.weight"), H, 1.0)
            .w(format!("{p}.self_attn.q_proj.weight"), &[2 * HD, H])
            .w(format!("{p}.self_attn.k_proj.weight"), &[HD, H])
            .w(format!("{p}.self_attn.v_proj.weight"), &[HD, H])
            .w(format!("{p}.self_attn.o_proj.weight"), &[H, 2 * HD])
            .norm(format!("{p}.self_attn.q_norm.weight"), HD, 1.0)
            .norm(format!("{p}.self_attn.k_norm.weight"), HD, 1.0);
        mlp(&mut w, &p, Ffn::Dense);
        if conv {
            // kernel 3, groups of 16 channels: [2, kernel, hidden] base, hidden -> 2*3*4 deltas.
            for c in ["attention_conv", "mlp_conv"] {
                w.s(format!("{p}.{c}.base_kernel"), &[2, 3, H], 0.5)
                    .s(format!("{p}.{c}.kernel_projection.weight"), &[2 * 3 * (H / 16), H], 0.05);
            }
        }
    }
    w
}

fn drafter_config(arch: &str, layers: usize) -> Value {
    json!({
        "architectures": [arch], "model_type": "qwen3",
        "hidden_size": H, "num_hidden_layers": layers, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": HD, "intermediate_size": INTER,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": 128,
        "vocab_size": V, "block_size": 4, "mask_token_id": 3, "target_layer_ids": [0, 2]
    })
}

/// Loads the drafter in `dir/draft` against the Qwen3.5 target in `dir/target` through
/// `runtime::draft::load_drafter`, feeds it 5 fixed tapped rows (6-token sequence), then 2 more
/// (catch-up after accepting one draft), recording drafted ids and every pick row's logits.
fn drafter_case(case: &str, cfg: Value, w: Weights) {
    let dir = fixture_dir(case);
    let (td, dd) = (dir.join("target"), dir.join("draft"));
    qwen35_ckpt(&td, 0x5157_454E_0010, false);
    write_json(&dd.join("config.json"), &cfg);
    write_st_as(&dd.join("model.safetensors"), &w.tensors, StDtype::BF16);
    golden(case, TOL, |dev| {
        let d: Arc<dyn Drafter> = gibbs_engine::runtime::draft::load_drafter(
            &artifacts(&dd, "golden/drafter", "model.safetensors"),
            &artifacts(&td, "golden/qwen3_5", "model.safetensors"),
            PlacementMode::Pin(dev),
        )
        .unwrap_or_else(|e| panic!("{case} on {dev:?}: {e}"));
        let taps = d.tap_layers();
        let mut out = Out::default();
        out.ids("taps", &taps.iter().map(|&t| t as u32).collect::<Vec<_>>());
        let mut rng = Rng::new(0x7A95);
        let mut st = d.new_state();
        let mut toks = vec![5u32, 17, 33, 48, 71, 12];
        for (call, rows) in [(0, 5), (1, 2)] {
            let hidden = rng.vec(rows * taps.len() * H, 1.0);
            let (mut vals, mut idx) = (Vec::new(), Vec::new());
            let ids = d
                .propose(&mut st, &toks, &hidden, d.max_k(), &mut |l| {
                    let DraftLogits::Dense(l) = l else {
                        unreachable!("model drafters pick dense rows")
                    };
                    // Selector / pruned-vocab rows are -inf off their candidates: keep the finite
                    // entries and their ids.
                    for (i, &x) in l.iter().enumerate().filter(|(_, x)| x.is_finite()) {
                        vals.push(x);
                        idx.push(i as u32);
                    }
                    argmax(l)
                })
                .unwrap();
            out.ids(format!("draft{call}"), &ids)
                .ids(format!("finite_ids{call}"), &idx)
                .f(format!("logits{call}"), &vals);
            toks.extend([ids[0], 60]);
        }
        out
    });
}

#[test]
fn dflash() {
    let mut cfg = drafter_config("DFlashDraftModel", 2);
    let c = cfg.as_object_mut().unwrap();
    for k in ["block_size", "mask_token_id", "target_layer_ids"] {
        c.remove(k);
    }
    c.insert("layer_types".into(), json!(["sliding_attention", "full_attention"]));
    c.insert("sliding_window".into(), json!(3));
    c.insert(
        "dflash_config".into(),
        json!({"block_size": 4, "mask_token_id": 3, "target_layer_ids": [0, 2]}),
    );
    drafter_case("qwen_dflash", cfg, drafter_weights(0x5157_454E_0011, 2, false));
}

/// DFlash2: grouped dynamic convs around both sublayers and the top-k candidate selector.
#[test]
fn dflash2() {
    let mut cfg = drafter_config("DFlash2DraftModel", 1);
    cfg["conv_group_size"] = 16.into();
    cfg["selector_top_k"] = 8.into();
    let mut w = drafter_weights(0x5157_454E_0012, 1, true);
    w.s("candidate_selector.predecessor_codebook", &[V, 8], 1.0)
        .s("candidate_selector.successor_codebook", &[V, 8], 1.0)
        .w("candidate_selector.hidden_projection.weight", &[8, H]);
    drafter_case("qwen_dflash2", cfg, w);
}

/// DSpark: blocks drafted from the anchor, own pruned 48-id head (`d2t`: draft id i -> 2i) and
/// the vanilla Markov bias.
#[test]
fn dspark() {
    let mut cfg = drafter_config("DSparkDraftModel", 1);
    cfg["markov_head_type"] = "vanilla".into();
    cfg["draft_vocab_size"] = 48.into();
    let mut w = drafter_weights(0x5157_454E_0013, 1, false);
    w.w("lm_head.weight", &[48, H])
        .put("d2t", &[48], (0..48).map(|i| i as f32).collect())
        .s("markov_head.markov_w1.weight", &[V, 4], 1.0)
        .s("markov_head.markov_w2.weight", &[48, 4], 1.0);
    drafter_case("qwen_dspark", cfg, w);
}
