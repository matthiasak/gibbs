use gibbs_engine::hardware::HardwareCapability;

#[test]
fn test_hardware_probe_reports_detected_facts() {
    let hw = HardwareCapability::probe_system();
    assert_eq!(hw.cpu_has_arm_neon, cfg!(target_arch = "aarch64"));
    if cfg!(target_os = "macos") {
        assert!(hw.cpu_name.is_some(), "machdep.cpu.brand_string not read");
        assert!(
            hw.total_memory_bytes.is_some_and(|b| b >= 1 << 30),
            "hw.memsize not read"
        );
        assert!(
            hw.gpu_max_working_set_bytes.is_some_and(|b| b >= 1 << 28),
            "recommendedMaxWorkingSetSize not read"
        );
        assert!(
            hw.gpu_max_buffer_bytes.is_some_and(|b| b >= 1 << 28),
            "maxBufferLength not read"
        );
    }
}

#[test]
fn test_codec_benchmark_efficiency() {
    #[cfg(target_os = "macos")]
    {
        let rt = gibbs_engine::graph::metal::Runtime::new(
            &gibbs_engine::model::hub::HuggingFaceHub::cache_dir(),
        )
        .unwrap();
        let e = rt.benchmark_codecs().unwrap();
        println!(
            "codec bench: ceiling {:.1} GB/s (stream {:.1})",
            e.ceiling_gbps,
            rt.profile().stream_gbps
        );
        for (fmt, g) in &e.gbps {
            println!("{fmt:<18} {g:>7.1} GB/s  eff {:.3}", e.eff[fmt]);
        }
        for fmt in ["GGUF_Q4_0", "GGUF_Q8_0", "BF16"] {
            let eff = e.eff[fmt];
            assert!(
                eff > 0.5 && eff <= 1.0,
                "{fmt}: eff {eff:.3} outside (0.5, 1.0]"
            );
        }
    }
}

#[test]
fn test_real_smollm2_135m_decode_throughput_on_disk() {
    use gibbs_engine::hardware::Device;
    use gibbs_engine::hardware::PlacementMode;
    use gibbs_engine::model::hub::HuggingFaceHub;
    use gibbs_engine::scheduler::ExecutableModel;
    use std::time::Instant;

    let repo = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let art = HuggingFaceHub::with_cache_dir("disk".into()).locate_model(repo);
    if art.weights_path.is_none() {
        println!("SKIP: SmolLM2-135M not found in disk/");
        return;
    }
    let reg = gibbs_engine::arch::registry().expect("registry");
    let exec_model = ExecutableModel::new(
        gibbs_engine::runtime::lm::load(&art, reg, None, PlacementMode::Pin(Device::Metal))
            .expect("loads SmolLM2"),
    );
    let tokenizer = exec_model.tokenizer();
    let prompt = "The capital of France is";
    let (prompt_tokens, state) = exec_model
        .lm()
        .prepare(tokenizer.encode(prompt), &[])
        .expect("prepare succeeds");
    let mut kv = Some(state);

    let first_next = exec_model
        .step_prefill_chunk(&prompt_tokens, 0, true, &prompt_tokens, 0.0, 1.0, &mut kv)
        .expect("prefill succeeds");
    let next_tok = first_next.expect("prefill returned next token");
    let mut generated = prompt_tokens.clone();
    generated.push(next_tok);

    let num_tokens = 32;
    let t0 = Instant::now();
    for _ in 0..num_tokens {
        let tok = exec_model
            .step_token_with_kv(
                &generated,
                *generated.last().unwrap(),
                generated.len(),
                0.0,
                1.0,
                None,
                &mut kv,
            )
            .map(|(t, _)| t)
            .expect("decode step succeeds");
        generated.push(tok);
    }
    let elapsed = t0.elapsed();
    let tps = (num_tokens as f64) / elapsed.as_secs_f64();
    let decoded = tokenizer.decode(&generated);
    println!(
        "SmolLM2-135M Real Disk Checkpoint: {tps:.2} tok/s ({num_tokens} tokens in {elapsed:?})"
    );
    println!("Decoded: {:?}", decoded);
    assert!(
        tps > 10.0,
        "SmolLM2 real decode throughput must exceed 10 tok/s"
    );
    assert!(
        decoded.to_lowercase().contains("paris"),
        "SmolLM2 must answer Paris"
    );
}

#[test]
fn test_real_gemma4_e2b_decode_and_bandwidth_on_disk() {
    use gibbs_engine::hardware::Device;
    use gibbs_engine::model::hub::HuggingFaceHub;
    let load_native = |a: &gibbs_engine::model::hub::ModelArtifacts, d: Device| {
        let reg = gibbs_engine::arch::registry().expect("registry");
        Some(gibbs_engine::runtime::lm::load(
            a,
            reg,
            None,
            gibbs_engine::hardware::PlacementMode::Pin(d),
        ))
    };
    use std::time::Instant;

    let repos = [
        "unsloth/gemma-4-E2B-it-qat-GGUF",
        "HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive",
    ];
    let hub = HuggingFaceHub::with_cache_dir("disk".into());
    let (repo, art) = match repos
        .iter()
        .map(|&r| (r, hub.locate_model(r)))
        .find(|(_, a)| a.weights_path.is_some())
    {
        Some(pair) => pair,
        None => {
            println!("SKIP: No Gemma 4 checkpoint found in disk/");
            return;
        }
    };

    let lm = match load_native(&art, Device::Metal) {
        Some(Ok(m)) => m,
        Some(Err(e)) if e.contains("gpu-only:") => {
            println!("SKIP: {repo} has no Metal program yet: {e}");
            return;
        }
        Some(Err(e)) => panic!("Failed to load {repo}: {e}"),
        None => panic!("load_native returned None for {repo}"),
    };

    let tok = lm.tokenizer();
    let prompt = "The capital of France is";
    let ids = tok.encode(prompt);
    let (seq_ids, mut state) = lm.prepare(ids, &[]).expect("prepare sequence");

    let logits = lm.forward(&seq_ids, 0, &mut state).expect("forward prompt");
    let mut cur_token = (0..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap() as u32;
    let mut pos = seq_ids.len();
    let mut generated = Vec::new();

    let num_tokens = 16;
    let t0 = Instant::now();
    for _ in 0..num_tokens {
        if tok.is_special(cur_token) {
            break;
        }
        generated.push(cur_token);
        let next_logits = lm
            .forward(&[cur_token], pos, &mut state)
            .expect("decode step");
        pos += 1;
        cur_token = (0..next_logits.len())
            .max_by(|&a, &b| next_logits[a].total_cmp(&next_logits[b]))
            .unwrap() as u32;
    }
    let elapsed = t0.elapsed();
    let tps = (generated.len() as f64) / elapsed.as_secs_f64();
    let model_bytes = art
        .weights_path
        .as_ref()
        .and_then(|p| std::fs::metadata(p).ok().map(|m| m.len() as f64))
        .unwrap_or(2.14e9f64);
    let eff_bw = (tps * model_bytes) / 1e9;
    println!("Gemma 4 E2B Real Disk Checkpoint: {tps:.2} tok/s ({eff_bw:.2} GB/s effective memory streaming)");
    assert!(
        tps > 5.0,
        "Gemma 4 E2B real decode throughput must exceed 5 tok/s"
    );
    let decoded = tok.decode(&generated);
    assert!(
        decoded.to_lowercase().contains("paris"),
        "Gemma 4 E2B must answer Paris, got: {decoded}"
    );
}

#[test]
fn test_real_qwen35_9b_decode_and_roofline_on_disk() {
    use gibbs_engine::hardware::{Device, PlacementMode};
    use gibbs_engine::model::hub::HuggingFaceHub;
    use std::time::Instant;

    let repo = "DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF";
    let art = HuggingFaceHub::with_cache_dir("disk".into()).locate_model(repo);
    if art.weights_path.is_none() {
        println!("SKIP: {repo} not found in disk/");
        return;
    }

    let reg = gibbs_engine::arch::registry().unwrap();
    let lm = gibbs_engine::runtime::lm::load(&art, reg, None, PlacementMode::Pin(Device::Metal))
        .unwrap_or_else(|e| panic!("Failed to load {repo}: {e}"));

    let tok = lm.tokenizer();
    let prompt = "The capital of France is";
    let ids = tok.encode(prompt);
    let (seq_ids, mut state) = lm.prepare(ids, &[]).expect("prepare sequence");

    let logits = lm.forward(&seq_ids, 0, &mut state).expect("forward prompt");
    let mut cur_token = (0..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap() as u32;
    let mut pos = seq_ids.len();
    let mut generated = Vec::new();

    let num_tokens = 8;
    let t0 = Instant::now();
    for _ in 0..num_tokens {
        if tok.is_special(cur_token) {
            break;
        }
        generated.push(cur_token);
        let next_logits = lm
            .forward(&[cur_token], pos, &mut state)
            .expect("decode step");
        pos += 1;
        cur_token = (0..next_logits.len())
            .max_by(|&a, &b| next_logits[a].total_cmp(&next_logits[b]))
            .unwrap() as u32;
    }
    let elapsed = t0.elapsed();
    let tps = (generated.len() as f64) / elapsed.as_secs_f64();
    let model_bytes = 4.71e9f64; // ~4.71 GB IQ2_M GGUF
    let eff_bw = (tps * model_bytes) / 1e9;
    let roofline_pct = (eff_bw / 298.83) * 100.0;
    println!("Qwen 3.5 9B Real Disk Checkpoint on Metal: {tps:.2} tok/s ({eff_bw:.2} GB/s, {roofline_pct:.1}% of 298.8 GB/s roofline)");
    assert!(
        tps > 20.0,
        "Qwen 3.5 9B real Metal decode throughput must exceed 20 tok/s"
    );
}
