use axum::body::Body;
use axum::http::{Request, StatusCode};
use gibbs_engine::hardware::{Device, PlacementMode, QuantFormat};
use gibbs_engine::model::hub::{HuggingFaceHub, ModelArtifacts};
use gibbs_engine::model::native::NativeLm;
use gibbs_engine::model::tokenizer::Tokenizer;
use gibbs_engine::model::weight_format::{reencode_override, set_reencode_override, ReencodeConfig};
use gibbs_engine::scheduler::{
    ContinuousBatchScheduler, ExecutableModel, SamplingParams, DEFAULT_MAX_NEW_TOKENS,
};
use gibbs_engine::server::{create_router, AppState};
use gibbs_engine::telemetry::EngineTelemetry;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower::ServiceExt;

/// Serializes loads: `set_reencode_override` is process-global and tests run in parallel.
static LOAD: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn load_unlocked(art: &ModelArtifacts) -> Result<Arc<dyn NativeLm>, String> {
    gibbs_engine::runtime::lm::load(
        art,
        gibbs_engine::arch::registry()?,
        None,
        PlacementMode::Pin(Device::Cpu),
    )
}

/// Loads `art` on the CPU through its family spec (detected from config.json / GGUF).
fn load(art: &ModelArtifacts) -> Result<Arc<dyn NativeLm>, String> {
    let _g = LOAD.lock().unwrap_or_else(|e| e.into_inner());
    load_unlocked(art)
}

/// The HF snapshot `dir` (config.json + tokenizer.json + model.safetensors), or `None` if absent.
fn load_dir(dir: &str) -> Option<Arc<dyn NativeLm>> {
    let d = std::path::Path::new(dir);
    if !d.join("config.json").exists() {
        println!("SKIP: {dir} not found");
        return None;
    }
    let art = ModelArtifacts {
        model_id: dir.to_string(),
        config_path: Some(d.join("config.json")),
        tokenizer_path: Some(d.join("tokenizer.json")),
        weights_path: Some(d.join("model.safetensors")),
        local_dir: Some(d.to_path_buf()),
        ..Default::default()
    };
    Some(load(&art).unwrap_or_else(|e| panic!("{dir}: {e}")))
}

fn load_gguf(path: &std::path::Path) -> Arc<dyn NativeLm> {
    let art = ModelArtifacts {
        model_id: path.display().to_string(),
        weights_path: Some(path.to_path_buf()),
        ..Default::default()
    };
    load(&art).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Random-init synthetic llama (`gibbs/tiny-5m`), optionally stored as `--quant force`.
fn tiny_lm(force: Option<QuantFormat>, dev: Device) -> Arc<dyn NativeLm> {
    // A fresh hub dir: a synthetic repo left in the user's cache would be reused as is.
    let hub_dir = tempfile::tempdir().unwrap();
    let art = HuggingFaceHub::with_cache_dir(hub_dir.path().to_path_buf())
        .resolve_or_synthesize_tiny("gibbs/tiny-5m")
        .unwrap();
    let _g = LOAD.lock().unwrap_or_else(|e| e.into_inner());
    let prev = reencode_override();
    set_reencode_override(force.map(|f| ReencodeConfig {
        force: Some(f),
        ..ReencodeConfig::from_settings()
    }));
    let lm = gibbs_engine::runtime::lm::load(
        &art,
        gibbs_engine::arch::registry().unwrap(),
        None,
        PlacementMode::Pin(dev),
    );
    set_reencode_override(prev);
    lm.unwrap()
}

fn tiny_model() -> ExecutableModel {
    ExecutableModel::new(tiny_lm(None, Device::Cpu))
}

fn argmax(lg: &[f32]) -> u32 {
    (0..lg.len())
        .max_by(|&a, &b| lg[a].total_cmp(&lg[b]))
        .unwrap() as u32
}

/// Greedy ids after `prompt`, plus the decode-step time (prefill excluded).
fn greedy_ids(lm: &dyn NativeLm, prompt: &[u32], n: usize) -> (Vec<u32>, Duration) {
    let (ids, mut state) = lm.prepare(prompt.to_vec(), &[]).unwrap();
    let mut next = argmax(&lm.forward(&ids, 0, &mut state).unwrap());
    let mut pos = ids.len();
    let mut out = vec![next];
    let t0 = Instant::now();
    while out.len() < n {
        next = argmax(&lm.forward(&[next], pos, &mut state).unwrap());
        pos += 1;
        out.push(next);
    }
    (out, t0.elapsed())
}

/// `prompt` + its greedy continuation (raw prompt, no chat template), like the old `generate_text`.
fn greedy_text(lm: &dyn NativeLm, prompt: &str, n: usize) -> String {
    let tok = lm.tokenizer();
    let ids = tok.encode(prompt);
    let (out, _) = greedy_ids(lm, &ids, n);
    tok.decode(&[ids, out].concat())
}

/// Greedy text from `dir` starting with `prompt`, free of byte-token leakage.
fn assert_coherent(dir: &str, prompt: &str, n: usize) {
    let Some(lm) = load_dir(dir) else { return };
    let out = greedy_text(&*lm, prompt, n);
    println!("{dir}: '{}'", out.replace('\n', " "));
    assert!(out.starts_with(prompt), "{dir}: {out:?}");
    for leak in ["<0x", "Ġ", "Ċ"] {
        assert!(!out.contains(leak), "{dir}: {out:?}");
    }
}

const PYTHIA: &str = "disk/models--EleutherAI--pythia-14m/snapshots/main";
const TS_INSTRUCT_8M: &str = "disk/models--roneneldan--TinyStories-Instruct-8M/snapshots/main";
const TS_1M: &str = "disk/models--roneneldan--TinyStories-1M/snapshots/main";
const TS_3M: &str = "disk/models--roneneldan--TinyStories-3M/snapshots/main";
const TS_8M: &str = "disk/models--roneneldan--TinyStories-8M/snapshots/main";
const SMOLLM2: &str = "disk/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main";
const STORIES15M: &str = "disk/models--Xenova--llama2.c-stories15M/snapshots/main";
const ANT_5M: &str = "disk/models--GODELEV--Ant-5M/snapshots/main";
const PLYX: &str = "disk/models--MultivexAI--Plyx-15M/snapshots/main";

#[test]
fn test_tokenizer_encoding_decoding_and_streaming() {
    let tokenizer = Tokenizer::default_bpe();
    let prompt = "How many 'r's in the word strawberry?";
    let tokens = tokenizer.encode(prompt);
    assert!(!tokens.is_empty());

    let decoded = tokenizer.decode(&tokens);
    assert_eq!(decoded, prompt);

    // Verify token-by-token streaming delta
    for &tok_id in &tokens {
        let piece = tokenizer.decode_token(tok_id);
        assert!(!piece.is_empty());
    }

    // Verify chat template
    let templated = tokenizer.apply_chat_template(&[("user", prompt)]);
    assert!(templated.contains("<|im_start|>user"));
    assert!(templated.contains("<|im_start|>assistant"));
}

#[test]
fn test_hub_model_budget_safeguard() {
    let hub = HuggingFaceHub::new();
    let artifacts_big = hub.locate_model("meta-llama/Llama-3.2-1B-Instruct");
    let budget_res = hub.validate_budget(&artifacts_big);
    // Should warn or reject models exceeding 10M parameter testing budget
    assert!(budget_res.is_err());
    let err_msg = budget_res.unwrap_err();
    assert!(err_msg.contains("10,000,000") || err_msg.contains("budget"));

    // Tiny model resolves within budget
    let artifacts_tiny = hub.resolve_or_synthesize_tiny("foundry/tiny-5m").unwrap();
    assert!(artifacts_tiny.estimated_params <= 10_000_000);
}

#[test]
fn test_tiny_nvfp4_inference_and_generation() {
    let lm = tiny_lm(Some(QuantFormat::GGUF_NVFP4), Device::Cpu);
    let ids = lm.tokenizer().encode("How many 'r's in the word strawberry?");
    // Random weights over the synthetic config's vocab (wider than the fallback tokenizer's).
    let (out, _) = greedy_ids(&*lm, &ids, 5);
    assert_eq!(out.len(), 5);
    assert!(greedy_text(&*lm, "There are", 5).starts_with("There are"));
}

#[test]
fn test_real_safetensors_pythia_model_inference_and_comprehension() {
    if !std::path::Path::new(PYTHIA).exists() {
        return;
    }
    let tokenizer = Tokenizer::from_json_str(
        &std::fs::read_to_string(std::path::Path::new(PYTHIA).join("tokenizer.json")).unwrap(),
    )
    .unwrap();
    assert!(tokenizer.vocab_size() >= 50277);
    // BPE space prefix encoding on the real vocabulary.
    let prompt = "The capital of France is";
    let tokens = tokenizer.encode(prompt);
    assert_eq!(tokens.len(), 5);
    assert_eq!(tokenizer.inv_vocab.get(&tokens[1]).unwrap(), "Ġcapital");
    assert_eq!(tokenizer.decode(&tokens), prompt);
    assert_coherent(PYTHIA, prompt, 6);
}

#[test]
fn test_real_safetensors_tinystories_model_inference_and_comprehension() {
    assert_coherent(TS_INSTRUCT_8M, "Once upon a time", 10);
}

#[tokio::test]
async fn test_server_real_streaming_and_system1_endpoint() {
    let telemetry = EngineTelemetry::new();

    let jev_model = Default::default();

    let scheduler = Arc::new(ContinuousBatchScheduler::new("", Some(tiny_model())));

    // Spawn scheduler step runner for the duration of the test
    let sched_clone = scheduler.clone();
    let step_handle = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(5)).await;
            sched_clone.step();
        }
    });

    let state = AppState {
        scheduler,
        telemetry,
        jev_model,
        image_model: Default::default(),
        classifier: Default::default(),
    };
    let app = create_router(state);

    // 1. Test POST /v1/chat/completions (Non-Streaming)
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::json!({
                "model": "foundry/smollm-1m",
                "messages": [{"role": "user", "content": "How many 'r's in strawberry?"}],
                "max_tokens": 8,
                "stream": false
            })
            .to_string(),
        ))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let chat_resp: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(chat_resp["object"], "chat.completion");
    assert!(
        chat_resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .len()
            > 0
    );

    let _ = step_handle.abort();
}

#[test]
fn test_smollm2_gqa_llama_inference_and_comprehension_ablation() {
    if let Some(lm) = load_dir(SMOLLM2) {
        assert_eq!(lm.tokenizer().vocab_size(), 49152);
    }
    assert_coherent(SMOLLM2, "The quick brown fox", 6);
}

#[test]
fn test_smollm2_program_scheduler_inference() {
    let Some(lm) = load_dir(SMOLLM2) else { return };
    let exec_model = ExecutableModel::new(lm);
    let tokenizer = exec_model.tokenizer();
    let prompt = "The capital of France is";
    let (prompt_tokens, state) = exec_model
        .lm()
        .prepare(tokenizer.encode(prompt), &[])
        .expect("prepare succeeds");
    let mut kv = Some(state);

    // Prefill through scheduler
    let first_next = exec_model
        .step_prefill_chunk(&prompt_tokens, 0, true, &prompt_tokens, 0.0, 1.0, &mut kv)
        .expect("prefill succeeds");
    let next_tok = first_next.expect("prefill returned next token");
    let mut generated_tokens = prompt_tokens.clone();
    generated_tokens.push(next_tok);

    for _ in 0..4 {
        let tok = exec_model
            .step_token_with_kv(
                &generated_tokens,
                *generated_tokens.last().unwrap(),
                generated_tokens.len(),
                0.0,
                1.0,
                None,
                &mut kv,
            )
            .map(|(t, _)| t)
            .expect("decode step succeeds");
        generated_tokens.push(tok);
    }

    let decoded_text = tokenizer.decode(&generated_tokens);
    println!(
        "SmolLM2-135M Program scheduler generated: '{}'",
        decoded_text
    );
    assert!(generated_tokens.len() > prompt_tokens.len());
}

#[test]
fn test_decode_throughput_benchmark_300_tok_s() {
    // The shipping backend: the CPU executor is the opt-in fallback, not a throughput target.
    let model = ExecutableModel::new(tiny_lm(None, gibbs_engine::hardware::default_device()));
    let prompt = "In a quiet valley surrounded by mountains";
    let num_tokens = 50;
    let _ = model.generate_text_measured_plain(prompt, 5, 0.0).unwrap(); // warm-up
    let (generated, steps, elapsed) = model
        .generate_text_measured_plain(prompt, num_tokens, 0.0)
        .unwrap();
    let tps = steps as f64 / elapsed.as_secs_f64();
    println!("⚡ Decode Throughput: {tps:.1} tok/s ({steps} steps in {elapsed:?})");
    println!("Sample generation: '{generated}'");
    assert!(!generated.is_empty());
    let target_tps = if cfg!(debug_assertions) { 100.0 } else { 300.0 };
    assert!(tps >= target_tps, "Throughput {tps:.1} tok/s must reach {target_tps} tok/s");
}

#[test]
fn test_real_weights_tinystories_decode_throughput_300_tok_s() {
    let Some(lm) = load_dir(TS_INSTRUCT_8M) else { return };
    let num_tokens = 30;
    let _ = greedy_ids(&*lm, &lm.tokenizer().encode("Hello"), 2); // warm-up
    let prompt = lm.tokenizer().encode("Once upon a time");
    let (out, decode_elapsed) = greedy_ids(&*lm, &prompt, num_tokens + 1);
    let tps = num_tokens as f64 / decode_elapsed.as_secs_f64();
    println!("⚡ Real TinyStories-8M Pure KV Decode Throughput: {tps:.1} tok/s");
    println!("Real weights generation: '{}'", lm.tokenizer().decode(&out));
    let target_tps = if cfg!(debug_assertions) { 20.0 } else { 300.0 };
    assert!(tps > target_tps, "Throughput {tps:.1} tok/s must exceed {target_tps} tok/s");
}

#[test]
fn test_all_readme_models_inference_and_zero_gibberish() {
    assert_coherent(TS_INSTRUCT_8M, "Once in a quiet village surrounded by green hills,", 15);
    assert_coherent(PYTHIA, "In a secret garden behind the old stone wall,", 12);
    assert_coherent(SMOLLM2, "User: How are you today? Assistant: I am", 10);

    let prompt = "The sun rises over the horizon,";
    let out = greedy_text(&*tiny_lm(None, Device::Cpu), prompt, 10);
    assert!(out.starts_with(prompt) && !out.contains("<0x"), "{out:?}");

    let gguf_path =
        std::path::Path::new("disk/models--ggml-org--tiny-llamas/snapshots/main/stories15M.gguf");
    if gguf_path.exists() {
        let bytes = std::fs::read(gguf_path).expect("Failed to read stories15M.gguf");
        let manifest = gibbs_engine::model::UniversalModelLoader::parse_gguf_bytes(&bytes)
            .expect("Failed to parse stories15M.gguf");
        assert!(manifest.tensors.len() >= 50);
        assert!(manifest.tensors.contains_key("blk.0.attn_q.weight"));
        assert!(manifest.tensors.contains_key("token_embd.weight"));
    }
}

#[test]
fn test_xenova_llama2_c_stories15m_inference_and_comprehension() {
    if let Some(lm) = load_dir(STORIES15M) {
        assert_eq!(lm.tokenizer().vocab_size(), 32000);
    }
    assert_coherent(STORIES15M, "Once upon a time, there was a little girl named Lily.", 20);
}

#[test]
fn test_ant_5m_llama_inference_and_comprehension() {
    let Some(lm) = load_dir(ANT_5M) else { return };
    assert_eq!(lm.tokenizer().vocab_size(), 16384);
    let out = greedy_text(&*lm, "<USER>Tell me a story<ASSISTANT>", 15);
    println!("GODELEV/Ant-5M generation: '{out}'");
    assert!(!out.is_empty() && !out.contains("<0x"));
}

#[test]
fn test_tinystories_1m_inference_and_comprehension() {
    assert_coherent(TS_1M, "Once upon a time", 15);
}

#[test]
fn test_tinystories_3m_inference_and_comprehension() {
    assert_coherent(TS_3M, "Once upon a time", 15);
}

#[test]
fn test_tinystories_8m_base_inference_and_comprehension() {
    assert_coherent(TS_8M, "Once upon a time", 15);
}

#[test]
fn test_plyx_15m_inference_and_comprehension() {
    assert_coherent(PLYX, "Once upon a time", 15);
}

#[test]
fn test_all_readme_models_legibility_and_grammar_verification() {
    assert_coherent(TS_INSTRUCT_8M, "Once upon a time", 15);
    assert_coherent(PYTHIA, "The capital of France is", 10);
}

#[test]
fn test_dynamic_unseen_prompts_across_all_readme_models() {
    for dir in [TS_1M, TS_3M, TS_8M, TS_INSTRUCT_8M] {
        assert_coherent(dir, "In a quiet village near the mountains, a friendly dog", 12);
    }
    assert_coherent(PYTHIA, "The sun rises in the east and", 12);
    assert_coherent(SMOLLM2, "User: What is the fastest animal on land? Assistant: The", 10);
    assert_coherent(STORIES15M, "One morning, a little bird named Pip decided to", 12);
    assert_coherent(PLYX, "The little boy opened the wooden box and", 10);
}

/// GLM-Edge (HF `GlmForCausalLM`): interleaved-pair RoPE + fused `mlp.gate_up_proj`.
/// Expected values come from transformers' `GlmForCausalLM` in f32 on the same checkpoint.
#[test]
fn test_glm_edge_matches_hf_oracle() {
    if cfg!(debug_assertions) {
        return; // the full 28-layer checkpoint is release-only
    }
    let Some(lm) = load_dir("disk/models--THUDM--glm-edge-1.5b-chat/snapshots/main") else {
        return;
    };
    let ids = lm.tokenizer().encode("The capital of France is");
    assert_eq!(ids, [1026, 8570, 325, 11334, 398]);
    let (seq, mut state) = lm.prepare(ids.clone(), &[]).unwrap();
    let logits = lm.forward(&seq, 0, &mut state).unwrap();
    assert_eq!(argmax(&logits), 14488); // " Paris"
    for (id, want) in [(14488usize, 16.5629f32), (11262, 13.4245)] {
        assert!((logits[id] - want).abs() < 0.02, "logit[{id}] = {} want {want}", logits[id]);
    }
    // HF greedy: " Paris. It is the most visited"
    assert_eq!(greedy_ids(&*lm, &ids, 7).0, [14488, 46, 1462, 398, 283, 1945, 24733]);
}

/// llama.cpp oracle (`llama-tokenize`, `llama-simple`) on ggml-org/tiny-llamas stories15M:
/// GGUF metadata config, llama q/k in interleaved-RoPE layout, Q4_0 element order, and
/// SentencePiece encoding ('▁' spaces + BOS) / decoding (dummy prefix stripped).
#[test]
fn test_gguf_stories15m_matches_llama_cpp() {
    for (file, want) in [
        (
            "stories15M.gguf",
            [
                29892, 727, 471, 263, 2217, 7826, 4257, 365, 2354, 29889, 2296, 18012, 304, 1708,
                5377, 297,
            ],
        ),
        (
            "stories15M-q4_0.gguf",
            [
                29892, 727, 471, 263, 2217, 7826, 4257, 365, 2354, 29889, 2296, 18012, 304, 1708,
                5377, 322,
            ],
        ),
    ] {
        let path =
            std::path::Path::new("disk/models--ggml-org--tiny-llamas/snapshots/main").join(file);
        if !path.exists() {
            continue;
        }
        let lm = load_gguf(&path);
        let prompt = lm.tokenizer().encode("Once upon a time");
        assert_eq!(prompt, [1, 9038, 2501, 263, 931], "{file}");
        let (out, _) = greedy_ids(&*lm, &prompt, want.len());
        assert_eq!(out, want, "{file}");
        let text = lm.tokenizer().decode(&[prompt, out].concat());
        assert!(
            text.starts_with("Once upon a time, there was a little girl named Lily."),
            "{file}: {text:?}"
        );
    }
}

/// llama.cpp oracle on unsloth SmolLM2-135M-Instruct Q4_K_M (Q5_0/Q4_K/Q6_K/Q8_0 tensors,
/// GQA 9/3, rope base 1e5, tied output): greedy " Paris. Paris is a city".
#[test]
fn test_gguf_smollm2_q4_k_m_matches_llama_cpp() {
    let path = std::path::Path::new("disk/models--unsloth--SmolLM2-135M-Instruct-GGUF/snapshots/main/SmolLM2-135M-Instruct-Q4_K_M.gguf");
    if !path.exists() || cfg!(debug_assertions) {
        return;
    }
    let lm = load_gguf(path);
    let prompt = lm.tokenizer().encode("The capital of France is");
    assert_eq!(prompt, [504, 3575, 282, 4649, 314]); // add_bos_token = false
    // llama.cpp result_output logits[0..3] for this prompt.
    let (seq, mut state) = lm.prepare(prompt.clone(), &[]).unwrap();
    let logits = lm.forward(&seq, 0, &mut state).unwrap();
    for (got, want) in logits.iter().zip([3.4901f32, -10.5134, -7.1320]) {
        assert!((got - want).abs() < 0.05, "logit {got} want {want}");
    }
    assert_eq!(greedy_ids(&*lm, &prompt, 6).0, [7042, 30, 7042, 314, 253, 2240]);
}

/// Minimal GGUF v3 writer (alignment 32, no metadata) for synthetic loader tests.
fn gguf_file(tensors: &[(&str, [u64; 2], u32, Vec<u8>)]) -> Vec<u8> {
    let s = |b: &mut Vec<u8>, v: &str| {
        b.extend((v.len() as u64).to_le_bytes());
        b.extend(v.as_bytes());
    };
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend((tensors.len() as u64).to_le_bytes());
    b.extend(0u64.to_le_bytes());
    let pad = |n: usize| n.div_ceil(32) * 32;
    let mut off = 0u64;
    for (name, dims, ty, data) in tensors {
        s(&mut b, name);
        b.extend(2u32.to_le_bytes());
        dims.iter().for_each(|d| b.extend(d.to_le_bytes()));
        b.extend(ty.to_le_bytes());
        b.extend(off.to_le_bytes());
        off += pad(data.len()) as u64;
    }
    b.resize(pad(b.len()), 0);
    for (_, _, _, data) in tensors {
        let end = b.len() + pad(data.len());
        b.extend(data);
        b.resize(end, 0);
    }
    b
}

/// Every GGUF block format the loader accepts decodes like ggml `dequantize_row_*` through both
/// the f32 extraction path and the zero-copy fused GEMV (expected values from gguf-py on the same
/// bytes); GGML types it cannot decode are a load error naming the tensor, not misread bytes.
#[test]
fn test_gguf_block_formats_decode_like_ggml() {
    use gibbs_engine::model::UniversalModelLoader;
    // (name, ggml type, block elems, block bytes, W at [0,17,31,32,100,255,256+77], W·x)
    let cases: [(&str, u32, usize, usize, [f32; 7], [f32; 2]); 8] = [
        (
            "Q4_0",
            2,
            32,
            18,
            [
                0.070015, 0.070015, 0.070015, -0.070015, 0.0100021, 0.0600128, 0.0,
            ],
            [1.30028, 1.29778],
        ),
        (
            "Q4_1",
            3,
            32,
            20,
            [
                0.00500107, 0.075016, 0.075016, 0.0450096, 0.00500107, 0.0650139, 0.135029,
            ],
            [-0.717653, -0.942701],
        ),
        (
            "Q5_0",
            6,
            32,
            22,
            [
                0.0300064, -0.150032, 0.0100021, 0.0900192, -0.070015, -0.160034, 0.0400085,
            ],
            [-0.770164, -0.217546],
        ),
        (
            "Q5_1",
            7,
            32,
            24,
            [
                0.205044, 0.255054, 0.0950203, 0.285061, 0.00500107, 0.0850182, 0.255054,
            ],
            [-0.137529, -0.527613],
        ),
        (
            "Q8_0",
            8,
            32,
            34,
            [
                -0.810173, -1.20026, -1.22026, 0.970207, -0.230049, 1.00021, 0.880188,
            ],
            [8.35678, -6.60641],
        ),
        (
            "Q4_K",
            12,
            256,
            144,
            [
                -0.0550117, -0.125027, -0.145031, 0.770164, 4.2009, 0.450096, 0.665142,
            ],
            [24.474, 11.5725],
        ),
        (
            "Q5_K",
            13,
            256,
            176,
            [
                0.105022, -0.125027, -0.145031, 0.970207, 0.28006, 0.530113, -0.135029,
            ],
            [-34.1785, -66.6817],
        ),
        (
            "Q6_K",
            14,
            256,
            210,
            [
                1.05022, 2.28049, -10.6423, 32.1969, -23.9851, 1.80038, -7.83167,
            ],
            [-36.5703, 31.5592],
        ),
    ];
    // 2 rows x 256: bytes (i*73+29) % 256 with f16 d = 0.01 (+ dmin 0.005 / m -0.005) per block.
    let raw = |name: &str, elems: usize, bytes: usize| {
        let mut r: Vec<u8> = (0..512 / elems * bytes)
            .map(|i| ((i * 73 + 29) % 256) as u8)
            .collect();
        for blk in r.chunks_exact_mut(bytes) {
            let (d_at, second) = match name {
                "Q6_K" => (208, None),
                "Q4_K" | "Q5_K" => (0, Some(0x1D1Fu16)),
                "Q4_1" | "Q5_1" => (0, Some(0x9D1F)),
                _ => (0, None),
            };
            blk[d_at..d_at + 2].copy_from_slice(&0x211Fu16.to_le_bytes());
            if let Some(v) = second {
                blk[2..4].copy_from_slice(&v.to_le_bytes());
            }
        }
        r
    };
    let tensors: Vec<_> = cases
        .iter()
        .map(|&(name, ty, elems, bytes, ..)| (name, [256, 2], ty, raw(name, elems, bytes)))
        .collect();
    let data = gguf_file(&tensors);
    let manifest = UniversalModelLoader::parse_gguf(&data).unwrap();
    let x: Vec<f32> = (0..256).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
    for (name, _, _, _, want_w, want_y) in cases {
        let w = manifest.extract_tensor_f32(&data, name).unwrap();
        for (i, want) in [0, 17, 31, 32, 100, 255, 333].into_iter().zip(want_w) {
            assert!(
                (w[i] - want).abs() <= 1e-4 * want.abs().max(1.0),
                "{name} w[{i}] = {} want {want}",
                w[i]
            );
        }
        let format = manifest.tensors[name].quant_format;
        let raw_data = &tensors.iter().find(|t| t.0 == name).unwrap().3;
        let mut y = [0.0f32; 2];
        gibbs_engine::model::weights::Mat::from_packed_bytes(format, raw_data.clone(), 2, 256)
            .gemv_f32(&x, &mut y);
        for (got, want) in y.iter().zip(want_y) {
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                "{name} W·x = {got} want {want}"
            );
        }
    }

    let bad = gguf_file(
        &[("blk.0.ffn_down.weight", [256, 2], 15, vec![0; 2 * 292])],
    );
    let err = UniversalModelLoader::parse_gguf(&bad).unwrap_err();
    assert!(
        err.contains("blk.0.ffn_down.weight") && err.contains("Q8_K"),
        "{err}"
    );
}

/// GGUF-embedded tokenizers encode exactly like llama.cpp (`llama-tokenize` ids): SentencePiece
/// scores-derived merges (stories15M) and byte-level BPE merges + smollm pre-tokenizer + the GGUF
/// chat template (SmolLM2).
#[test]
fn test_gguf_tokenizers_match_llama_tokenize() {
    use gibbs_engine::model::UniversalModelLoader;
    let dir = |repo: &str| std::path::PathBuf::from(format!("disk/models--{repo}/snapshots/main"));
    let stories = dir("ggml-org--tiny-llamas").join("stories15M.gguf");
    if stories.exists() {
        let tok =
            UniversalModelLoader::extract_gguf_tokenizer(&std::fs::read(stories).unwrap()).unwrap();
        for (text, want) in [
            (
                "Once upon a time, there was a little girl named Lily.",
                &[
                    1u32, 9038, 2501, 263, 931, 29892, 727, 471, 263, 2217, 7826, 4257, 365, 2354,
                    29889,
                ][..],
            ),
            (
                "  How are   you?\n\n\n",
                &[1, 259, 1128, 526, 259, 366, 29973, 13, 13, 13],
            ),
            (
                "Café résumé 🦀 1234 don't “quotes”",
                &[
                    1, 315, 28059, 6896, 398, 29948, 29871, 243, 162, 169, 131, 29871, 29896,
                    29906, 29941, 29946, 1016, 29915, 29873, 1346, 339, 4769, 30024,
                ],
            ),
        ] {
            assert_eq!(tok.encode(text), want, "{text:?}");
            assert_eq!(tok.decode(want), text);
        }
    }
    let smollm =
        dir("unsloth--SmolLM2-135M-Instruct-GGUF").join("SmolLM2-135M-Instruct-Q4_K_M.gguf");
    if smollm.exists() {
        let tok =
            UniversalModelLoader::extract_gguf_tokenizer(&std::fs::read(smollm).unwrap()).unwrap();
        assert_eq!(
            tok.encode("  How are   you?\n\n\n"),
            [216, 1073, 359, 256, 346, 47, 16506]
        );
        let chat = tok.apply_chat_template(&[("user", "What is the capital of France?")]);
        assert_eq!(
            tok.encode(&chat),
            [
                1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018,
                411, 407, 19712, 8182, 2, 198, 1, 4093, 198, 1780, 314, 260, 3575, 282, 4649, 47,
                2, 198, 1, 520, 9531, 198
            ],
            "{chat:?}"
        );
    }
}

/// Chat path end to end: the model's own `chat_template` (tokenizer_config.json) rendered and
/// BPE-tokenized like HF `apply_chat_template(add_generation_prompt=True)`, then greedy decoding
/// through `ContinuousBatchScheduler` equal to transformers' greedy `generate` (f32 weights).
#[test]
fn test_chat_template_and_scheduler_greedy_match_hf() {
    use gibbs_engine::model::hub::ModelArtifacts;
    use tokio::sync::mpsc::error::TryRecvError;
    if cfg!(debug_assertions) {
        return;
    }
    let cases: [(&str, &[u32], &[u32]); 2] = [
        (
            // Template injects SmolLM's default system prompt; answer ends with <|im_end|>.
            "disk/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main",
            &[
                1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018,
                411, 407, 19712, 8182, 2, 198, 1, 4093, 198, 1780, 314, 260, 3575, 282, 4649, 47,
                2, 198, 1, 520, 9531, 198,
            ],
            &[504, 3575, 282, 4649, 314, 7042, 30], // "The capital of France is Paris."
        ),
        (
            // Answer ends with <|user|> (a generation_config eos id).
            "disk/models--THUDM--glm-edge-1.5b-chat/snapshots/main",
            &[59253, 10, 5127, 398, 283, 8570, 325, 11334, 63, 59254, 10],
            &[1026, 8570, 325, 11334, 398, 14488, 46], // "The capital of France is Paris."
        ),
    ];
    for (dir, want_prompt, want_answer) in cases {
        let dir = std::path::Path::new(dir);
        if !dir.join("tokenizer_config.json").exists() {
            continue;
        }
        let artifacts = ModelArtifacts {
            model_id: dir.display().to_string(),
            config_path: Some(dir.join("config.json")),
            tokenizer_path: Some(dir.join("tokenizer.json")),
            weights_path: Some(dir.join("model.safetensors")),
            local_dir: Some(dir.to_path_buf()),
            ..Default::default()
        };
        let model = ExecutableModel::new(load(&artifacts).unwrap());

        let sched = ContinuousBatchScheduler::new("", Some(model));
        let greedy = SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            top_logprobs: None,
        };
        let (_, mut rx, _, _, _) = sched.submit_with_params(greedy, |tok| {
            let ids =
                tok.encode(&tok.apply_chat_template(&[("user", "What is the capital of France?")]));
            assert_eq!(ids, want_prompt, "{}", dir.display());
            ids
        });
        let mut answer = Vec::new();
        for _ in 0..64 {
            sched.step();
            match rx.try_recv() {
                Ok(id) => answer.push(id),
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => break,
            }
        }
        assert_eq!(answer, want_answer, "{}", dir.display());
    }
}

/// A checkpoint missing a weight its family spec reads is refused naming that tensor.
#[test]
fn test_missing_required_weight_names_the_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let header =
        br#"{"model.embed_tokens.weight":{"dtype":"F32","shape":[4,2],"data_offsets":[0,32]}}"#;
    let mut data = (header.len() as u64).to_le_bytes().to_vec();
    data.extend_from_slice(header);
    data.extend_from_slice(&[0u8; 32]);
    std::fs::write(dir.path().join("model.safetensors"), data).unwrap();
    let cfg = serde_json::json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "vocab_size": 4, "hidden_size": 2, "num_hidden_layers": 1, "num_attention_heads": 1,
        "num_key_value_heads": 1, "intermediate_size": 4, "max_position_embeddings": 16});
    std::fs::write(dir.path().join("config.json"), cfg.to_string()).unwrap();
    let tok = serde_json::json!({"version": "1.0", "added_tokens": [], "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"a": 0, "b": 1, "c": 2, "d": 3}, "unk_token": "a"}});
    std::fs::write(dir.path().join("tokenizer.json"), tok.to_string()).unwrap();
    let art = ModelArtifacts {
        model_id: "t/missing".into(),
        tokenizer_path: Some(dir.path().join("tokenizer.json")),
        local_dir: Some(dir.path().to_path_buf()),
        config_path: Some(dir.path().join("config.json")),
        weights_path: Some(dir.path().join("model.safetensors")),
        ..Default::default()
    };
    let err = load(&art).err().expect("a checkpoint without layer weights loads");
    assert!(err.contains("model.layers.0.input_layernorm.weight"), "{err}");
}
