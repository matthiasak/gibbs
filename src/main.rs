use clap::Parser;
use gibbs_engine::arch::dispatch::{LoadOpts, Loaded};
use gibbs_engine::hardware::{Device, HardwareCapability};
use gibbs_engine::kv::PagedKVPool;
use gibbs_engine::model::classifier::{self, ClassifierSlot};
use gibbs_engine::model::hub::{is_synthetic_model_id, HuggingFaceHub, StorageBackend};
use gibbs_engine::model::image::ImageBuffer;
use gibbs_engine::model::load_log;
use gibbs_engine::model::weight_format::ReencodeConfig;
use gibbs_engine::runtime::diffusion::{ImageGenerator, ImageRequest, ImageSlot};
use gibbs_engine::scheduler::{cpu_backend_label, ContinuousBatchScheduler};
use gibbs_engine::server::{create_router, AppState};
use gibbs_engine::telemetry::EngineTelemetry;
use gibbs_engine::tui::TuiApp;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(
    name = "gibbs",
    author = "Gibbs Systems <engineering@gibbs.ai>",
    version = "0.1.0",
    about = "Gibbs: local inference engine (CPU + Apple Metal)"
)]
struct Cli {
    #[arg(short, long, default_value = "HuggingFaceTB/SmolLM2-135M-Instruct")]
    model: String,

    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    #[arg(long, default_value_t = false)]
    tui: bool,

    /// Store every weight matrix in this format (uncompressed, f16, bf16, q8_0, q4_0, q4_k_m,
    /// q5_k_m, q6_k, nvfp4 = GGUF_NVFP4) instead of its checkpoint encoding. Without it weights
    /// keep their checkpoint encoding or follow the re-encode policy (`--reencode`).
    #[arg(long)]
    quant: Option<String>,

    /// Compute backend: auto (Metal on Apple Silicon, CPU only where no GPU exists), metal, or
    /// cpu. `cpu` is the explicit opt-in to the CPU executor.
    #[arg(long, default_value = "auto")]
    backend: String,

    #[arg(long, default_value = "disk")]
    storage: String,

    #[arg(long)]
    hf_token: Option<String>,

    #[arg(long)]
    prompt: Option<String>,

    /// Speculative decoding drafter for --model: a hidden-state drafter repo trained for it (a
    /// `pipeline: drafter` family spec), or a small causal LM with the same tokenizer.
    #[arg(long)]
    draft: Option<String>,

    /// Speculative draft mode: auto | noise | suffix | model | off.
    #[arg(long)]
    draft_mode: Option<String>,

    /// Maximum draft tokens per speculative step (0..=15).
    #[arg(long)]
    draft_max_k: Option<usize>,

    /// Target speculative acceptance rate (0.10..=0.98) for KTuner.
    #[arg(long)]
    draft_target_accept: Option<f32>,

    /// Verify compute cost weight (0.0..=1.0) in KTuner.
    #[arg(long)]
    draft_cost_weight: Option<f32>,

    /// Minimum suffix n-gram match length (1..=4) for SuffixLattice.
    #[arg(long)]
    draft_suffix_min: Option<usize>,

    /// Online learning rate for the forward-only `qlabs.sh/noise` block drafter.
    #[arg(long)]
    draft_noise_lr: Option<f32>,

    /// Session compaction mode: auto | light | aggressive | off.
    #[arg(long)]
    compact_mode: Option<String>,

    /// Context fraction (0.25..=0.98) that triggers automatic session compaction.
    #[arg(long)]
    compact_threshold: Option<f32>,

    /// Target context fraction (0.15..=0.85) after automatic session compaction.
    #[arg(long)]
    compact_target: Option<f32>,

    /// Number of recent turns kept verbatim during session compaction.
    #[arg(long)]
    compact_keep_turns: Option<usize>,

    #[arg(long)]
    image: Option<String>,

    /// Output file: a .png for text-to-image (default output.png), a required .mp4 for text-to-video.
    #[arg(long)]
    output: Option<String>,

    /// Denoising steps: default from the image model (e.g. 1 for turbo, 25 for SD), 25 for text-to-video.
    #[arg(long)]
    steps: Option<usize>,

    #[arg(long, default_value_t = 0.7)]
    strength: f32,

    /// Image/video seed; defaults to a clock-based seed that is printed so runs can be reproduced.
    #[arg(long)]
    seed: Option<u64>,

    /// Text-to-video: number of frames (16..=24 or 8k+1 depending on the video pipeline).
    #[arg(long)]
    frames: Option<usize>,

    /// Classifier-free guidance scale (CFG runs when > 1); default from the model (9 for text-to-video).
    #[arg(long)]
    guidance: Option<f32>,

    /// Text-to-image / text-to-video: negative prompt for classifier-free guidance.
    #[arg(long)]
    negative_prompt: Option<String>,

    /// Video pipelines with a separate text encoder: its weights (`org/repo` or
    /// `org/repo/file.gguf`); GGUF transformers default to a Q4_K_M encoder, diffusers repos to `text_encoder/`.
    #[arg(long)]
    text_encoder: Option<String>,

    /// Text-to-image width / height in pixels (default from the model).
    #[arg(long)]
    width: Option<usize>,

    #[arg(long)]
    height: Option<usize>,

    #[arg(long)]
    max_tokens: Option<usize>,

    #[arg(long)]
    ctx: Option<usize>,

    /// KV cache memory budget in GiB across every model's page pool (default: 25% of RAM).
    /// Past it, unused cached prefixes are evicted; then requests fail with `KV memory exhausted`.
    #[arg(long)]
    kv_mem: Option<f64>,

    /// GPU weight re-encode policy: max-speed | low-bit-native | bit-exact (default: settings.json, else max-speed).
    #[arg(long)]
    reencode: Option<String>,

    /// Mean-KL limit (nats) of the GPU re-encode gate; overrides the policy's own.
    #[arg(long)]
    reencode_max_kl: Option<f32>,

    /// Variant policy of the model picker and downloads: fastest | fastest-4bit | best-quality
    /// (default: settings.json, else fastest).
    #[arg(long)]
    variant: Option<String>,

    #[arg(long)]
    bench: Option<String>,

    #[arg(long)]
    bench_parity: Option<String>,

    /// Print the top-5 next-token logits (JSON) of --model for this raw prompt on Metal, then exit.
    #[arg(long)]
    probe_top5: Option<String>,

    /// Print where the graph nodes of --model run (op kind, device, avenue) on Metal, then exit.
    #[arg(long, default_value_t = false)]
    explain_placement: bool,

    #[arg(long)]
    tui_frame_log: Option<String>,

    /// Load --model as this family spec (`arch/families/<family>.json5`, or a user file in
    /// `~/.cache/gibbs/arch/families/`), skipping architecture detection.
    #[arg(long)]
    arch: Option<String>,

    /// Print --model's merged family spec, resolved config vars, raw -> canonical tensor names and
    /// unconsumed tensors (JSON), then exit.
    #[arg(long, default_value_t = false)]
    arch_dump: bool,
}

fn clock_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}

/// One CLI text-to-video (with `--image`, image-to-video) generation of a video pipeline plan.
fn run_video(
    cli: &Cli,
    hub: &HuggingFaceHub,
    plan: &gibbs_engine::model::video_repo::Plan,
    backend: Device,
) -> Result<(), String> {
    let model = &cli.model;
    let Some(prompt) = cli.prompt.as_deref() else {
        return Err(format!(
            "'{model}' is a text-to-video model; the HTTP server and daemon have no video endpoint. \
             Run one generation instead: gibbs --model {model} --prompt \"...\" --output clip.mp4"
        ));
    };
    let output = cli
        .output
        .as_deref()
        .filter(|o| o.ends_with(".mp4"))
        .ok_or("text-to-video needs --output <file.mp4>")?;
    let image = cli
        .image
        .as_deref()
        .map(ImageBuffer::from_base64_or_path)
        .transpose()?;
    let req = gibbs_engine::runtime::diffusion::VideoRequest {
        prompt: prompt.to_string(),
        negative_prompt: cli.negative_prompt.clone(),
        frames: cli.frames,
        width: cli.width,
        height: cli.height,
        steps: cli.steps,
        guidance: cli.guidance,
        seed: cli.seed.unwrap_or_else(clock_seed),
    };
    println!("Prompt: {prompt:?}");
    let start = Instant::now();
    let log = |s: &str| println!("{s}");
    let (frames, fps, req) = gibbs_engine::model::video_repo::generate(
        hub,
        plan,
        gibbs_engine::hardware::PlacementMode::Pin(backend),
        cli.text_encoder.as_deref(),
        &req,
        image.as_ref(),
        &log,
    )?;
    println!(
        "Seed: {}  Steps: {}  Frames: {}  Guidance: {}  Size: {}x{}",
        req.seed,
        req.steps.unwrap_or(0),
        frames.len(),
        req.guidance.unwrap_or(0.0),
        req.width.unwrap_or(0),
        req.height.unwrap_or(0)
    );
    gibbs_engine::model::image::write_mp4(&frames, fps, Path::new(output))?;
    let bytes = std::fs::metadata(output)
        .map_err(|e| format!("{output}: {e}"))?
        .len();
    println!(
        "Generated {} frames in {:.2}s",
        frames.len(),
        start.elapsed().as_secs_f64()
    );
    println!("Saved {output} ({bytes} bytes, {fps} fps)");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    if let Some(gib) = cli.kv_mem {
        if !(gib > 0.0) {
            return Err(format!("--kv-mem {gib}: needs a positive number of GiB").into());
        }
        gibbs_engine::kv::set_kv_mem_budget((gib * (1u64 << 30) as f64) as usize);
    }
    if let Some(v) = &cli.variant {
        gibbs_engine::model::variant::set_policy_override(Some(v.parse()?));
    }
    if cli.reencode.is_some() || cli.reencode_max_kl.is_some() {
        gibbs_engine::model::weight_format::set_reencode_override(Some(ReencodeConfig::resolve(
            cli.reencode.as_deref(),
            cli.reencode_max_kl,
        )?));
    }
    // The one-shot Metal tools (--bench, --bench-parity, --probe-top5, --explain-placement) exit before the
    // backend is resolved below: arm the GPU-only tripwire for them too, unless `--backend cpu` opts out.
    let one_shot_metal = cli.bench.is_some()
        || cli.bench_parity.is_some()
        || cli.probe_top5.is_some()
        || cli.explain_placement;
    if one_shot_metal
        && cli
            .backend
            .parse::<gibbs_engine::hardware::PlacementMode>()?
            != gibbs_engine::hardware::PlacementMode::Pin(Device::Cpu)
    {
        gibbs_engine::hardware::cpu_compute::ban(true);
    }
    if let Some(bench_path) = &cli.bench {
        let code = gibbs_engine::bench::run_bench_suite(bench_path)?;
        std::process::exit(code);
    }
    if let Some(parity_path) = &cli.bench_parity {
        let code = gibbs_engine::bench::run_bench_parity(parity_path)?;
        std::process::exit(code);
    }
    if let Some(prompt) = &cli.probe_top5 {
        gibbs_engine::bench::probe_top5(&cli.model, prompt)?;
        std::process::exit(0);
    }
    if cli.explain_placement {
        let hub = HuggingFaceHub::new().with_storage(StorageBackend::from_uri(&cli.storage)?);
        let code = gibbs_engine::bench::explain_placement(&hub, &cli.model)?;
        std::process::exit(code);
    }
    if cli.arch_dump {
        let hub = HuggingFaceHub::new()
            .with_storage(StorageBackend::from_uri(&cli.storage)?)
            .with_budget(usize::MAX);
        let token = cli
            .hf_token
            .clone()
            .or_else(|| std::env::var("HF_TOKEN").ok());
        let a = hub.auto_download_and_cache(&cli.model, token.as_deref())?;
        println!(
            "{}",
            gibbs_engine::arch::dispatch::dump(&a, cli.arch.as_deref())?
        );
        return Ok(());
    }
    let telemetry = EngineTelemetry::new();
    #[cfg(target_os = "macos")]
    let rt_warmup = {
        let warm = gibbs_engine::hardware::default_cache_dir()
            .ok()
            .and_then(|d| gibbs_engine::graph::metal::Runtime::publish_cached_rates(&d))
            .is_some();
        let telemetry = telemetry.clone();
        let spawn_bench = cli.prompt.is_none();
        let init = move || {
            if let Ok(rt) = gibbs_engine::graph::metal::Runtime::shared() {
                let prof = rt.profile();
                telemetry.set_hardware_profile(&prof.device, prof.memory_bandwidth_gbps as f32);
                // S3.6: measured codec efficiencies for the variant predictor; background, never blocks the load.
                // Only in the long-lived TUI: a one-shot `--prompt` run would share the GPU with it and exit
                // before it finishes (the result is saved only on completion).
                if spawn_bench {
                    rt.spawn_codec_bench();
                }
            }
        };
        if warm {
            Some(std::thread::spawn(init))
        } else {
            init();
            None
        }
    };

    // 1. Probe Hardware
    let hw = HardwareCapability::probe_system();
    let mut placement: gibbs_engine::hardware::PlacementMode = cli.backend.parse()?;
    // GPU-only by default: `auto` resolves to Metal wherever it exists; the CPU is an explicit `--backend cpu`.
    if placement == gibbs_engine::hardware::PlacementMode::Auto
        && gibbs_engine::hardware::default_device() == Device::Metal
    {
        placement = gibbs_engine::hardware::PlacementMode::Pin(Device::Metal);
    }
    let backend = match placement {
        gibbs_engine::hardware::PlacementMode::Pin(d) => d,
        gibbs_engine::hardware::PlacementMode::Auto => Device::Cpu,
    };
    let backend_label = match placement {
        gibbs_engine::hardware::PlacementMode::Auto => "auto (cost-modeled placement)".to_string(),
        gibbs_engine::hardware::PlacementMode::Pin(d) => match d {
            Device::Cpu => cpu_backend_label(),
            #[cfg(target_os = "macos")]
            Device::Metal => format!(
                "Metal ({})",
                gibbs_engine::backend::metal::system_device_name()?
            ),
            #[cfg(not(target_os = "macos"))]
            Device::Metal => return Err("--backend metal requires macOS".into()),
            Device::Cuda => gibbs_engine::graph::cuda::device_name(),
            Device::Rocm => gibbs_engine::graph::rocm::device_name(),
            Device::Xla => gibbs_engine::graph::xla::device_name(),
            Device::IntelXpu => gibbs_engine::graph::intel_xpu::device_name(),
            Device::Extropic => gibbs_engine::graph::extropic::device_name(),
        },
    };
    // GPU-only tripwire: from here on, any CPU compute outside `--backend cpu` panics.
    if backend == Device::Metal {
        gibbs_engine::hardware::cpu_compute::ban(true);
    }
    println!("═══════════════════════════════════════════════════════════════");
    println!("  GIBBS INFERENCE ENGINE // v0.1.0");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Compute backend: {backend_label}");
    for line in hw.summary() {
        println!("{line}");
    }

    let quant_override = cli
        .quant
        .as_deref()
        .map(gibbs_engine::model::weight_format::parse_quant)
        .transpose()?;

    // 2. Initialize Memory Hierarchy and KV Pool
    let kv_pool = Arc::new(PagedKVPool::new(
        gibbs_engine::scheduler::ONE_SHOT_POOL_BLOCKS,
    ));

    // 4. Model Resolution & Cache Inspection (Auto-Download & Cloud Storage)
    let storage_backend = StorageBackend::from_uri(&cli.storage)?;
    let hub = HuggingFaceHub::new()
        .with_storage(storage_backend)
        .with_budget(usize::MAX);

    let token = cli
        .hf_token
        .clone()
        .or_else(|| std::env::var("HF_TOKEN").ok());
    telemetry.log_info(&format!(
        "Checking model source: '{}' (storage: {:?})",
        cli.model, hub.storage
    ));
    let mut reencode = ReencodeConfig::resolve(cli.reencode.as_deref(), cli.reencode_max_kl)?;
    reencode.force = quant_override;
    let t = telemetry.clone();
    reencode.log = Some(Arc::new(move |m: &str| t.log_info(m)));
    let stage = parking_lot::Mutex::new("resolving");
    let t0 = Instant::now();
    let loaded = gibbs_engine::arch::load_model(
        &hub,
        &cli.model,
        &LoadOpts {
            placement,
            token: token.as_deref(),
            quant_name: cli.quant.as_deref(),
            hw: &hw,
            reencode,
            family: cli.arch.as_deref(),
            ctx: cli.ctx,
            stage: &|s| *stage.lock() = s,
            system1_first: true,
            admit: &|_| Ok(()),
        },
    )
    .map_err(|e| {
        let stage = *stage.lock();
        format!(
            "Cannot load '{}': {}",
            cli.model,
            load_log::record(&hub, &cli.model, stage, &e)
        )
    })?;
    #[cfg(target_os = "macos")]
    if let Some(h) = rt_warmup {
        let _ = h.join();
    }
    let is_image = matches!(loaded, Loaded::Image(_));
    if !matches!(loaded, Loaded::Video(_)) {
        if cli.image.is_some() && !is_image {
            return Err(format!(
                "--image requires a text-to-image model; '{}' is not one",
                cli.model
            )
            .into());
        }
        if cli.image.is_some() && cli.prompt.is_none() {
            return Err("--image requires --prompt".into());
        }
    }

    let mut image_model: Option<Arc<dyn ImageGenerator>> = None;
    let mut drafter = None;
    let classifier_slot = ClassifierSlot::default();
    let jev_slot = gibbs_engine::runtime::encoder::system1::System1Slot::default();
    let model_exec = match loaded {
        Loaded::Video(plan) => return run_video(&cli, &hub, &plan, backend).map_err(Into::into),
        Loaded::Image(g) => {
            let d = g.defaults();
            telemetry.log_info(&format!(
                "Image model: {} loaded in {:.2}s (default {}x{}, {} steps, guidance {})",
                g.summary(),
                t0.elapsed().as_secs_f64(),
                d.width,
                d.height,
                d.steps,
                d.guidance
            ));
            if let Some(prompt) = &cli.prompt {
                let req = ImageRequest {
                    prompt: prompt.clone(),
                    negative_prompt: cli.negative_prompt.clone(),
                    width: cli.width,
                    height: cli.height,
                    steps: cli.steps,
                    guidance: cli.guidance,
                    seed: cli.seed.unwrap_or_else(clock_seed),
                };
                let steps = req.steps.unwrap_or(d.steps);
                println!("Prompt: {prompt:?}");
                println!(
                    "Seed: {}  Steps: {steps}  Guidance: {}  Negative: {:?}",
                    req.seed,
                    req.guidance.unwrap_or(d.guidance),
                    req.negative_prompt.as_deref().unwrap_or("")
                );
                let start = Instant::now();
                let img = if let Some(init_path) = &cli.image {
                    let init = ImageBuffer::from_base64_or_path(init_path)
                        .map_err(|e| format!("Cannot read --image {init_path}: {e}"))?;
                    println!(
                        "Mode: image-to-image from {init_path} ({}x{}), strength {:.2}",
                        init.width, init.height, cli.strength
                    );
                    g.image_to_image(&req, &init, cli.strength)?
                } else {
                    println!(
                        "Mode: text-to-image {}x{}",
                        req.width.unwrap_or(d.width),
                        req.height.unwrap_or(d.height)
                    );
                    g.text_to_image(&req)?
                };
                let gen_secs = start.elapsed().as_secs_f64();
                let png_bytes = img.to_png();
                let output = cli.output.as_deref().unwrap_or("output.png");
                std::fs::write(output, &png_bytes)?;
                println!(
                    "Generated {}x{} in {gen_secs:.2}s ({:.2}s/step incl. text encoding and VAE)",
                    img.width,
                    img.height,
                    gen_secs / steps as f64
                );
                println!("Saved {output} ({} bytes)", png_bytes.len());
                return Ok(());
            }
            image_model = Some(g);
            None
        }
        Loaded::System1(jev) => {
            if jev.encoder().is_some() {
                telemetry.log_info(&format!(
                    "System 1 encoder '{}' ready, loaded in {:.2}s (zero-shot cosine decisions): {}",
                    jev.repo_id,
                    t0.elapsed().as_secs_f64(),
                    jev.summary()
                ));
            }
            if let Some(prompt) = &cli.prompt {
                if prompt.contains('|') {
                    let (query, opts) = prompt
                        .split_once('|')
                        .map(|(q, o)| (q.trim(), o))
                        .unwrap_or((prompt.as_str(), ""));
                    let options: Vec<gibbs_engine::runtime::encoder::system1::ChoiceOption> = opts
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(|s| {
                            gibbs_engine::runtime::encoder::system1::ChoiceOption::new(
                                s,
                                None::<String>,
                            )
                        })
                        .collect();
                    let decision = jev
                        .embed(query)
                        .and_then(|s| jev.evaluate_choice_embedded(&s, &options))?;
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "query": query,
                            "model": jev.repo_id,
                            "decision": decision,
                        }))?
                    );
                } else {
                    let v = jev.embed(prompt)?;
                    println!("{} dimensions · head={:?}", v.len(), &v[..v.len().min(8)]);
                }
                return Ok(());
            }
            *jev_slot.write() = Some(Arc::new(jev));
            None
        }
        Loaded::Classifier(c) => {
            telemetry.log_info(&format!(
                "Classifier '{}' ready: {}, loaded in {:.2}s",
                cli.model,
                c.summary(),
                t0.elapsed().as_secs_f64()
            ));
            if let Some(prompt) = &cli.prompt {
                let (text, pair) = classifier::split_pair(prompt);
                let (labels, tokens) = c.classify(text, pair)?;
                println!("{tokens} tokens");
                for l in labels {
                    println!("{:>10.6}  {}", l.score, l.label);
                }
                return Ok(());
            }
            *classifier_slot.write() = Some(Arc::new(c));
            None
        }
        Loaded::Lm {
            model: model_exec,
            artifacts,
        } => {
            let artifacts = &artifacts;
            telemetry.log_info(&format!(
                "Model Artifacts Resolved: cached={}, local_dir={:?}, weights={:?}",
                artifacts.is_cached, artifacts.local_dir, artifacts.weights_path
            ));
            if let Some(system_mem) = hw.total_memory_bytes {
                // Weights are read into f32 before any re-encoding.
                let estimated_weight_bytes = artifacts.estimated_params as u64 * 4;
                if estimated_weight_bytes > system_mem {
                    telemetry.log_warn(&format!(
                    "Model '{}' (~{} params, {:.2} GB in f32) exceeds physical system memory ({:.2} GB). Offloading will be active.",
                    cli.model,
                    artifacts.estimated_params,
                    estimated_weight_bytes as f64 / 1e9,
                    system_mem as f64 / 1e9
                ));
                } else {
                    telemetry.log_info(&format!(
                        "Model '{}' resolved (~{} params, {:.2} MB in f32, system memory {:.2} GB)",
                        cli.model,
                        artifacts.estimated_params,
                        estimated_weight_bytes as f64 / 1e6,
                        system_mem as f64 / 1e9
                    ));
                }
            }
            let model_size = artifacts.weights_path.as_ref().map_or(0, |p| {
                gibbs_engine::model::variant::decode_bytes_per_token(
                    p,
                    gibbs_engine::model::weight_format::ReencodeConfig::effective().policy,
                )
            });
            telemetry.set_model_size_bytes(model_size);
            telemetry.log_info(&format!("Model: {}", model_exec.lm().summary()));
            if is_synthetic_model_id(&cli.model) {
                telemetry.log_warn(&format!(
                "'{}' is a synthetic random-weight model for kernel benchmarking: its text output is noise",
                cli.model
            ));
            }
            // The roofline basis follows the policy the load applied (the gate may have fallen back).
            let applied = gibbs_engine::bench::applied_policy();
            if let (b, "streamed") = gibbs_engine::bench::roofline_bytes(artifacts, &applied) {
                telemetry.set_model_size_bytes(b);
            }
            telemetry.log_info(&format!("Compute backend: {}", model_exec.backend_label()));
            if let Some(d) = &cli.draft {
                let dr = if matches!(d.as_str(), "noise" | "online-noise" | "suffix") {
                    let mut k = gibbs_engine::tui::settings::load_settings().draft;
                    k.mode = if d == "suffix" {
                        gibbs_engine::runtime::noise_draft::DraftMode::Suffix
                    } else {
                        gibbs_engine::runtime::noise_draft::DraftMode::Noise
                    };
                    let live = Arc::new(gibbs_engine::runtime::noise_draft::LiveDraftKnobs::new(k));
                    gibbs_engine::runtime::draft::online_drafter_for_target(artifacts, live)
                        .map_err(|e| format!("Cannot initialize online drafter '{d}': {e}"))?
                } else {
                    let da = hub.auto_download_and_cache(d, token.as_deref())?;
                    gibbs_engine::runtime::draft::load_drafter(&da, artifacts, placement)
                        .map_err(|e| format!("Cannot load drafter '{d}': {e}"))?
                };
                telemetry.log_info(&format!("Speculative decoding drafter: {}", dr.label()));
                drafter = Some(dr);
            }

            Some(model_exec)
        }
    };

    // 5. Initialize Continuous Batch Scheduler with Real Streaming Channel & Telemetry
    let scheduler = Arc::new(ContinuousBatchScheduler::with_telemetry(
        cli.model.clone(),
        kv_pool.clone(),
        model_exec,
        telemetry.clone(),
    ));
    if cli.draft.is_some() && drafter.is_none() {
        return Err("--draft needs a text-generation --model".into());
    }
    if drafter.is_some() {
        if std::env::var("GIBBS_SPEC").as_deref() == Ok("off") {
            return Err("--draft: speculative decoding is disabled (GIBBS_SPEC=off)".into());
        }
        scheduler.set_drafter(drafter);
    }
    if cli.draft_mode.is_some()
        || cli.draft_max_k.is_some()
        || cli.draft_target_accept.is_some()
        || cli.draft_cost_weight.is_some()
        || cli.draft_suffix_min.is_some()
        || cli.draft_noise_lr.is_some()
    {
        let mut knobs = scheduler.draft_knobs();
        if let Some(m) = &cli.draft_mode {
            knobs.mode = m.parse()?;
        }
        if let Some(k) = cli.draft_max_k {
            knobs.max_k = k;
        }
        if let Some(a) = cli.draft_target_accept {
            knobs.target_acceptance = a;
        }
        if let Some(c) = cli.draft_cost_weight {
            knobs.cost_weight = c;
        }
        if let Some(s) = cli.draft_suffix_min {
            knobs.suffix_min_match = s;
        }
        if let Some(lr) = cli.draft_noise_lr {
            knobs.noise_lr = lr;
        }
        scheduler.set_draft_knobs(knobs);
    }
    if cli.compact_mode.is_some()
        || cli.compact_threshold.is_some()
        || cli.compact_target.is_some()
        || cli.compact_keep_turns.is_some()
    {
        let mut s = gibbs_engine::tui::settings::load_settings();
        if let Some(m) = &cli.compact_mode {
            s.compact.mode = m.parse()?;
        }
        if let Some(th) = cli.compact_threshold {
            s.compact.trigger_ratio = th;
        }
        if let Some(tg) = cli.compact_target {
            s.compact.target_ratio = tg;
        }
        if let Some(kt) = cli.compact_keep_turns {
            s.compact.keep_recent_turns = kt;
        }
        s.compact = s.compact.sanitized();
        let _ = gibbs_engine::tui::settings::save_settings(&s);
    }

    if let (Some(prompt), Some(model)) = (&cli.prompt, scheduler.model()) {
        println!("💬 Generating from prompt: \"{}\"", prompt);
        let cap = model.lm().default_max_new_tokens();
        let max_toks = cli.max_tokens.or(cap).unwrap_or(64);
        // Models that declare greedy decoding (encoder-decoders: transformers' default) are not sampled.
        let temperature = match scheduler.tokenizer().sampling_defaults {
            Some((t, _)) if t <= 1e-4 => 0.0,
            _ => 0.7,
        };
        let g = scheduler.generate(prompt, max_toks, temperature)?;
        println!("✨ Output:\n{}", g.text);
        let (decode, passes) = g.decode;
        let tok_s = passes as f64 / decode.as_secs_f64().max(1e-9);
        println!(
            "⚡ Decode: {passes} tokens in {:.2}ms ({:.1} tok/s) on {}",
            decode.as_secs_f64() * 1000.0,
            tok_s,
            model.backend_label()
        );

        // Roofline reporting
        let bytes_per_token = telemetry.bytes_per_token();
        let ceiling_gbps = telemetry.get_summary().memory_bandwidth_gbps as f64;
        if bytes_per_token > 0 && ceiling_gbps > 0.0 {
            let roofline_tok_s = ceiling_gbps / (bytes_per_token as f64 / 1e9);
            let pct = (tok_s / roofline_tok_s) * 100.0;
            println!(
                "📈 Roofline: {pct:.1}% of {roofline_tok_s:.1} tok/s ({bytes_per_token} B/token at {ceiling_gbps:.1} GB/s)"
            );
        }
        return Ok(());
    }

    // 6. Continuous batching loop on its own thread: a step is blocking compute, and on a tokio
    // worker it kept the streams it woke from running until it yielded (bursty SSE, stalls).
    // A panicking step fails the in-flight requests (their clients get an error) and the loop
    // keeps serving, instead of the thread dying and every later request hanging.
    let sched_clone = scheduler.clone();
    std::thread::spawn(move || loop {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sched_clone.step())) {
            Ok(out) if out.active_sequences_count == 0 => {
                sched_clone.wait_for_work(Duration::from_millis(50));
            }
            Ok(_) => {}
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                sched_clone.abort_all(&format!("engine step panicked: {msg}"));
            }
        }
    });

    // 7. HTTP/SSE server. Bound here so a busy port fails before the TUI takes the terminal.
    let image_slot = ImageSlot::default();
    *image_slot.write() = image_model;
    let state = AppState {
        scheduler: scheduler.clone(),
        telemetry: telemetry.clone(),
        jev_model: jev_slot.clone(),
        image_model: image_slot.clone(),
        classifier: classifier_slot.clone(),
    };
    let app = create_router(state);
    let addr = SocketAddr::from(([0, 0, 0, 0], cli.port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("Cannot listen on {addr}: {e}"))?;
    println!("🚀 Gibbs HTTP/SSE Server listening on http://{}", addr);
    let compat_client = gibbs_engine::arch::compat()["ndjson_api"]["client_name"]
        .as_str()
        .expect("compat.ndjson_api.client_name");
    let compat_env = gibbs_engine::arch::compat()["ndjson_api"]["host_env"]
        .as_str()
        .expect("compat.ndjson_api.host_env");
    println!("   Endpoints: /v1/chat/completions, /v1/completions, /v1/responses (OpenAI), /v1/messages (Anthropic), /api/chat, /api/generate ({compat_client}), /v1/images/generations, /v1/images/edits, /v1/embeddings, /v1/decisions, /v1/classify, /v1/rerank, /v1/models, /health");
    println!(
        "   Agents: {}  ·  {compat_env}={}  ·  TUI: /connect",
        gibbs_engine::server::connect::agent_command("claude", addr, &scheduler.model_id())?,
        gibbs_engine::server::connect::client_addr(addr)
    );
    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("HTTP server stopped: {e}");
        }
    });

    // 8. Interactive TUI or Headless Daemon Mode
    if cli.tui {
        let mut tui_app =
            TuiApp::new(telemetry, kv_pool).with_runtime(scheduler, hub, token, None, image_slot);
        tui_app.quant_override = quant_override;
        tui_app.frame_log_path = cli.tui_frame_log;
        tui_app.classifier = classifier_slot;
        tui_app.jev_model = jev_slot;
        tui_app.http_addr = Some(addr);
        tui_app.placement = placement;
        tokio::task::spawn_blocking(move || tui_app.run()).await??;
    } else {
        println!("Gibbs Engine running in daemon mode. Press Ctrl+C to stop.");
        tokio::signal::ctrl_c().await?;
        println!("Shutting down Gibbs Engine cleanly.");
    }

    Ok(())
}
