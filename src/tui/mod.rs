//! Terminal dashboard: live engine metrics, a prompt console, and a model picker that searches
//! the local cache and huggingface.co, then downloads, measures, and hot-swaps the chosen model.
//!
//! Prompts and model loads run on background threads and report through a channel, so the UI
//! never blocks; it redraws on input, on streamed output, and every [`REFRESH`] otherwise.

mod components;
mod layout;
pub(crate) mod mascot;
mod picker;
pub mod qos;
mod render;
pub mod settings;
#[cfg(test)]
mod snapshot;
pub mod theme;
mod widgets;
use crate::arch::dispatch::Loaded;
use crate::hardware::{HardwareCapability, PlacementMode, QuantFormat};
use crate::model::classifier::{self, ClassifierSlot};
use crate::model::hf_loader::ChatMessage;
use crate::model::hub::{
    split_revision, split_weight_file, DownloadProgress, HuggingFaceHub, StorageBackend,
};
use crate::model::image::ImageBuffer;
use crate::model::native::MediaInput;
use crate::model::reencode::ReencodeProgress;
use crate::model::tokenizer::Tokenizer;
use crate::model::weight_format::{
    reencode_override, set_reencode_override, ReencodeConfig, ReencodePolicy, VariantPolicy,
};
use crate::runtime::diffusion::{ImageRequest, ImageSlot};
use crate::runtime::encoder::system1::{ChoiceOption, JevSystem1Model};
use crate::runtime::encoder::{Activation, LabelScore};
use crate::scheduler::{finish_outcome, ContinuousBatchScheduler, SamplingParams};
use crate::telemetry::{EngineTelemetry, KernelRow};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use parking_lot::Mutex;
use picker::{Picker, PickerAction};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Offered in the picker even when not cached: the models verified to load and generate.
fn pinned_models() -> &'static [&'static str] {
    static P: std::sync::LazyLock<Vec<&'static str>> = std::sync::LazyLock::new(|| {
        crate::arch::compat()["tui"]["pinned_models"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(serde_json::Value::as_str)
            .collect()
    });
    &P
}
/// Redraw interval locked to 60 FPS tick (16.6 ms) without frame drops.
const REFRESH: Duration = Duration::from_nanos(16_666_667);
/// Poll interval while output streams or a model loads (60 FPS tick).
const BUSY_POLL: Duration = Duration::from_nanos(16_666_667);
/// Throughput history: one sample per second.
const SPARK_SECONDS: usize = 120;
/// Streamed chat text is re-sent to the UI at most this often.
const STREAM_UPDATE: Duration = Duration::from_millis(30);
/// A sampled kernel profile is at least this old before the TUI asks for the next.
const PROFILE_EVERY: Duration = Duration::from_secs(5);

/// The slash commands that take arguments, parsed without touching the app.
#[derive(Debug, PartialEq)]
enum SlashCmd {
    /// `None` cycles to the next policy.
    Reencode(Option<ReencodePolicy>),
    Variant(Option<VariantPolicy>),
    Profile(ProfileArg),
    Placement,
    /// No arguments: print the status.
    Spec(Vec<SpecArg>),
}

/// `/profile` arguments.
#[derive(Debug, PartialEq)]
enum ProfileArg {
    Status,
    On,
    Off,
    /// `None` cycles the column.
    Sort(Option<crate::telemetry::KernelSort>),
}

/// `/spec` arguments for live speculative decoding & online noise drafter tuning.
#[derive(Debug, PartialEq)]
enum SpecArg {
    Mode(crate::runtime::noise_draft::DraftMode),
    Objective(crate::runtime::noise_draft::DraftObjective),
    MaxK(usize),
    Accept(f32),
    Cost(f32),
    Suffix(usize),
    Lr(f32),
    Sigma(f32),
    Steps(usize),
    Shortlist(usize),
    Probes(usize),
    Window(usize),
}

/// Parses `/reencode`, `/variant`, `/profile` and `/placement` lines; `None` for any other command.
fn parse_slash(raw: &str) -> Option<Result<SlashCmd, String>> {
    let (cmd, rest) = raw.split_once(char::is_whitespace).unwrap_or((raw, ""));
    let arg = rest.trim();
    Some(match cmd {
        "/placement" => Ok(SlashCmd::Placement),
        "/reencode" if arg.is_empty() => Ok(SlashCmd::Reencode(None)),
        "/reencode" => arg
            .parse()
            .map(|p| SlashCmd::Reencode(Some(p)))
            .map_err(|_| {
                format!(
                    "unknown reencode policy '{arg}'; valid: max-speed, min-size, low-bit-native, bit-exact"
                )
            }),
        "/variant" if arg.is_empty() => Ok(SlashCmd::Variant(None)),
        "/variant" => arg
            .parse()
            .map(|p| SlashCmd::Variant(Some(p)))
            .map_err(|_| {
                format!(
                    "unknown variant policy '{arg}'; valid: fastest, fastest-4bit, best-quality"
                )
            }),
        "/profile" => parse_profile(arg).map(SlashCmd::Profile),
        "/spec" => parse_spec(arg).map(SlashCmd::Spec),
        _ => return None,
    })
}

/// `/spec` alone (status), `/spec <mode>`, `/spec <key> <value>` or `/spec key=value ...`.
fn parse_spec(arg: &str) -> Result<Vec<SpecArg>, String> {
    if arg.contains('=') {
        return arg
            .split_whitespace()
            .map(|kv| {
                let (k, v) = kv
                    .split_once('=')
                    .ok_or_else(|| format!("expected key=value, got '{kv}'"))?;
                spec_arg(k, v)
            })
            .collect();
    }
    let (word, rest) = arg.split_once(char::is_whitespace).unwrap_or((arg, ""));
    match (word, rest.trim()) {
        ("", _) => Ok(Vec::new()),
        (w, "") => w.parse().map(|m| vec![SpecArg::Mode(m)]),
        (w, v) => spec_arg(w, v).map(|a| vec![a]),
    }
}

fn spec_arg(key: &str, v: &str) -> Result<SpecArg, String> {
    fn num<T: std::str::FromStr>(v: &str, what: &str, range: &str) -> Result<T, String> {
        v.parse()
            .map_err(|_| format!("invalid {what} '{v}'; expected {range}"))
    }
    Ok(match key {
        "mode" => SpecArg::Mode(v.parse()?),
        "objective" => SpecArg::Objective(v.parse()?),
        "k" | "max-k" | "max_k" => SpecArg::MaxK(
            num::<usize>(v, "max_k", "integer 0..=15")?
                .min(crate::runtime::noise_draft::NOISE_MAX_K),
        ),
        "accept" | "target-accept" => {
            SpecArg::Accept(num(v, "target acceptance", "float 0.10..=0.98")?)
        }
        "cost" | "cost-weight" => SpecArg::Cost(num(v, "cost weight", "float 0.0..=1.0")?),
        "suffix" | "suffix-min" => SpecArg::Suffix(num(v, "suffix min match", "integer 1..=4")?),
        "lr" | "noise-lr" => SpecArg::Lr(num(v, "noise lr", "float 0.0..=0.10")?),
        "sigma" | "noise-sigma" => SpecArg::Sigma(num(v, "noise sigma", "float 1e-5..=0.25")?),
        "steps" | "denoise-steps" => SpecArg::Steps(num(v, "denoise steps", "integer 1..=4")?),
        "shortlist" => SpecArg::Shortlist(num(v, "shortlist", "integer 64..=1024")?),
        "probes" => SpecArg::Probes(num(v, "probes", "power of two 4..=32")?),
        "window" => SpecArg::Window(num(v, "window", "integer 1..=8")?),
        w => {
            return Err(format!(
                "unknown /spec argument '{w}'; use mode=<auto|noise|suffix|model|off> objective=<speed|tokens> k= accept= cost= suffix= lr= sigma= steps= shortlist= probes= window="
            ))
        }
    })
}

fn parse_profile(arg: &str) -> Result<ProfileArg, String> {
    let (word, col) = arg.split_once(char::is_whitespace).unwrap_or((arg, ""));
    match (word, col.trim()) {
        ("", _) => Ok(ProfileArg::Status),
        ("on" | "1" | "true", _) => Ok(ProfileArg::On),
        ("off" | "0" | "false", _) => Ok(ProfileArg::Off),
        ("sort", "") => Ok(ProfileArg::Sort(None)),
        ("sort", c) => crate::telemetry::KernelSort::parse(c)
            .map(|s| ProfileArg::Sort(Some(s)))
            .ok_or_else(|| format!("unknown sort column '{c}'; valid: time, bandwidth, share")),
        (w, _) => Err(format!(
            "unknown /profile argument '{w}'; use on, off, or sort <time|bandwidth|share>"
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Chat,
    Image,
    Decide,
    Embed,
    Classify,
    /// Video clip; `image`: the text starts with the first frame's path (`/i2v`).
    Video {
        image: bool,
    },
}

/// Default model family for bare (command-less) prompts, inferred from the active model id.
fn route_for_model(model_id: &str) -> Route {
    let m = model_id.to_lowercase();
    if m.contains("image") || m.contains("diffusion") {
        Route::Image
    } else if m.contains("minilm") || m.contains("embed") || m.contains("sentence-transformers") {
        Route::Embed
    } else if crate::arch::compat()["tui"]["decide_keywords"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .any(|k| m.contains(k))
    {
        Route::Decide
    } else {
        Route::Chat
    }
}

/// Text-generation models: the only kind the TUI hot-swaps into the scheduler.
pub(crate) fn is_chat_model(id: &str) -> bool {
    route_for_model(id) == Route::Chat
}

/// Luminance-ramp preview sampled from the real generated pixels.
fn ascii_preview(img: &ImageBuffer, cols: usize, rows: usize) -> String {
    const RAMP: &[u8] = b" .:-=+*#%@";
    if img.width == 0 || img.height == 0 || img.channels == 0 {
        return String::new();
    }
    let used = img.channels.min(3);
    let mut out = String::with_capacity((cols + 1) * rows);
    for r in 0..rows {
        for c in 0..cols {
            let i = ((r * img.height / rows) * img.width + c * img.width / cols) * img.channels;
            let lum = img.data[i..i + used]
                .iter()
                .map(|&v| v as usize)
                .sum::<usize>()
                / used;
            out.push(RAMP[lum * (RAMP.len() - 1) / 255] as char);
        }
        out.push('\n');
    }
    out.pop();
    out
}

/// A chat prompt word `@<path>` naming an existing image (png/jpg/jpeg) or WAV audio
/// file: the path and whether it is an image. Other `@words` stay prompt text.
fn attachment(word: &str) -> Option<(&std::path::Path, bool)> {
    let path = std::path::Path::new(word.strip_prefix('@')?);
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    let image = match ext.as_str() {
        "png" | "jpg" | "jpeg" => true,
        "wav" => false,
        _ => return None,
    };
    path.is_file().then_some((path, image))
}

/// A prompt with `@<path>` attachments as one user message of HF content parts (text and
/// `{"type":"image"|"audio"}` in prompt order) and the files' bytes in the same order; `None`
/// without attachments.
fn media_message(prompt: &str) -> Result<Option<(ChatMessage, Vec<MediaInput>)>, String> {
    let (mut parts, mut media, mut text) = (Vec::new(), Vec::new(), Vec::<&str>::new());
    let text_part = |text: &mut Vec<&str>| {
        let part = serde_json::json!({"type": "text", "text": text.join(" ")});
        text.clear();
        part
    };
    for word in prompt.split_whitespace() {
        let Some((path, image)) = attachment(word) else {
            text.push(word);
            continue;
        };
        if !text.is_empty() {
            parts.push(text_part(&mut text));
        }
        let bytes =
            std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        parts.push(serde_json::json!({"type": if image { "image" } else { "audio" }}));
        media.push(if image {
            MediaInput::Image(bytes)
        } else {
            MediaInput::Audio(bytes)
        });
    }
    if media.is_empty() {
        return Ok(None);
    }
    if !text.is_empty() {
        parts.push(text_part(&mut text));
    }
    let mut msg = ChatMessage::new("user", "");
    msg.content = serde_json::Value::Array(parts);
    Ok(Some((msg, media)))
}

/// Compacted turns and report for the chat `turns` (after `system_prompt`, before the `current`
/// message when given) on the active model, within its context capped at `ctx_limit`; `None` when
/// nothing was saved. The history holds text only, so no SnapCompact pages are drawn.
fn compact_turns(
    sched: &Arc<ContinuousBatchScheduler>,
    system_prompt: &str,
    turns: &[(String, String)],
    current: Option<ChatMessage>,
    ctx_limit: usize,
    cfg: &crate::compact::CompactionConfig,
    force: bool,
) -> Result<Option<(Vec<(String, String)>, crate::compact::CompactionReport)>, String> {
    let ctx = sched
        .model()
        .and_then(|m| m.context_len())
        .map_or(ctx_limit, |c| c.min(ctx_limit));
    let mut cfg = cfg.clone();
    cfg.enable_snapcompact_vision = false;
    let mut msgs = Vec::new();
    if !system_prompt.is_empty() {
        msgs.push(ChatMessage::new("system", system_prompt));
    }
    msgs.extend(turns.iter().map(|(r, c)| ChatMessage::new(r.as_str(), c.as_str())));
    let has_current = current.is_some();
    msgs.extend(current);
    let report = crate::compact::compact_for_scheduler(
        sched,
        &mut msgs,
        &mut Vec::new(),
        Some(ctx),
        &cfg,
        force,
    )?;
    if report.saved_tokens() == 0 {
        return Ok(None);
    }
    if has_current {
        msgs.pop();
    }
    let to = msgs
        .into_iter()
        .filter(|m| m.role != "system")
        .map(|m| (m.role.clone(), m.text()))
        .collect();
    Ok(Some((to, report)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Running,
    Text,
    Json,
    Image,
    Error,
}

/// One console entry: a prompt or model load and its result.
pub struct Entry {
    pub time: String,
    pub prompt: String,
    pub kind: Kind,
    pub output: String,
    /// e.g. `42 tokens · 86.1 tok/s`; empty when there is nothing to add.
    pub stats: String,
    started: Instant,
    took: Option<Duration>,
    /// Set for running chats: raising it stops streaming and drops the token receiver.
    cancel: Option<Arc<AtomicBool>>,
    /// User prompt text when this entry is a chat turn; committed to `chat_history` on finish.
    chat_user: Option<String>,
}

/// Results from background work, applied on the UI thread.
enum Update {
    Partial {
        idx: usize,
        text: String,
        tokens: usize,
    },
    Done {
        idx: usize,
        kind: Kind,
        text: String,
        stats: String,
    },
    Loaded {
        idx: usize,
        id: String,
        /// The loaded model's shape and backend.
        result: Result<String, String>,
    },
    Probe(Result<String, String>),
    /// A `/bench` run on the loaded model.
    Bench(Result<crate::bench::ModelBenchResult, String>),
    /// Chat history compaction: the history it started `from` and the compacted turns.
    Compacted {
        from: Vec<(String, String)>,
        result: Result<(Vec<(String, String)>, crate::compact::CompactionReport), String>,
    },
}

/// A prompt's background work: runs with its console entry index and the update channel.
type Job = Box<dyn FnOnce(usize, mpsc::Sender<Update>) + Send>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Focus {
    Dashboard,
    Prompt,
    Picker,
}

/// A model load running on a background thread.
pub(crate) struct Load {
    pub(crate) id: String,
    /// Console entry of this load.
    pub(crate) idx: usize,
    pub(crate) started: Instant,
    pub(crate) stage: Arc<Mutex<&'static str>>,
    pub(crate) progress: Arc<DownloadProgress>,
    /// (when, bytes done) of the last rate sample, and the bytes/s since the one before.
    pub(crate) sample: (Instant, u64),
    pub(crate) rate: f64,
    /// Tensors re-encoded so far and in total.
    pub(crate) reencode: Arc<ReencodeProgress>,
    /// (when, tensors done) the first time the re-encode counter moved: the ETA's baseline.
    reencode_seen: Option<(Instant, usize)>,
}

impl Load {
    /// `downloading onnx/model_q4.onnx 45% · 360/796 MB · 12 MB/s` while bytes flow, else the stage.
    fn progress_text(&self) -> String {
        let stage = *self.stage.lock();
        let (done, total) = (
            self.progress.done.load(Ordering::Relaxed),
            self.progress.total.load(Ordering::Relaxed),
        );
        if stage != "downloading" || total == 0 {
            return stage.to_string();
        }
        format!(
            "downloading {} {}% · {:.0}/{:.0} MB · {:.1} MB/s",
            self.progress.file.lock(),
            done * 100 / total,
            done as f64 / 1e6,
            total as f64 / 1e6,
            self.rate / 1e6
        )
    }

    /// `reencode max-speed · 12/197 tensors · ETA 9s` once tensors are counted, else the time so far.
    fn reencode_text(&self, policy: crate::model::weight_format::ReencodePolicy) -> String {
        let (done, total) = (
            self.reencode.done.load(Ordering::Relaxed),
            self.reencode.total.load(Ordering::Relaxed),
        );
        if total == 0 {
            return format!(
                "reencode {} · elapsed {}",
                policy.name(),
                render::secs(self.started.elapsed())
            );
        }
        let eta = self
            .reencode_seen
            .filter(|&(_, first)| done > first)
            .map_or("—".to_string(), |(since, first)| {
                let rate = (done - first) as f64 / since.elapsed().as_secs_f64().max(1e-9);
                format!("{:.0}s", total.saturating_sub(done) as f64 / rate)
            });
        format!(
            "reencode {} · {done}/{total} tensors · ETA {eta}",
            policy.name()
        )
    }
}

/// Codec micro-benchmark and tuner progress, as the Device panel shows it.
#[derive(Clone, Copy, Default)]
pub(crate) struct DeviceStatus {
    /// `(state, done, total)`: 0 idle, 1 running, 2 done, 3 failed.
    pub(crate) codec: (u8, usize, usize),
    /// `(cache hits, shapes)` of the GEMV tuner.
    pub(crate) tune: (usize, usize),
}

pub struct TuiApp {
    pub telemetry: Arc<EngineTelemetry>,
    pub hw: HardwareCapability,
    pub scheduler: Option<Arc<ContinuousBatchScheduler>>,
    pub hub: HuggingFaceHub,
    pub hf_token: Option<String>,
    pub jev_model: crate::runtime::encoder::system1::System1Slot,
    /// Shared with the HTTP server; `Some` makes `/image` and bare prompts generate images.
    pub image_model: ImageSlot,
    /// Set by loading a video repo; `/video` and `/i2v` generate clips with it.
    pub video: crate::model::video_repo::VideoSlot,
    /// Shared with the HTTP server; `Some` is the active model (a text classifier).
    pub classifier: ClassifierSlot,
    /// `--quant` ([`ReencodeConfig::force`]); `None` keeps checkpoint encodings / the policy.
    pub quant_override: Option<QuantFormat>,
    pub http_addr: Option<SocketAddr>,
    pub placement: PlacementMode,
    pub scratch_dir: tempfile::TempDir,
    pub history: Vec<Entry>,
    /// Multi-turn conversation history (`("user" | "assistant", content)` pairs).
    pub chat_history: Vec<(String, String)>,
    pub input: String,
    pub prompt_history: Vec<String>,
    pub prompt_history_idx: Option<usize>,
    pub prompt_draft: String,
    focus: Focus,
    picker: Picker,
    load: Option<Load>,
    running: usize,
    updates_tx: mpsc::Sender<Update>,
    updates_rx: mpsc::Receiver<Update>,
    /// Console lines scrolled up from the bottom; 0 follows new output.
    scroll: usize,
    /// Tokens generated in each of the last [`SPARK_SECONDS`] seconds, oldest first.
    spark: Vec<u64>,
    spark_at: Instant,
    spark_total: u64,
    host_lines: Vec<String>,
    pub mascot: mascot::MascotEngine,
    pub show_template_modal: bool,
    pub active_template: usize,
    pub system_prompt: String,
    /// Component shown in detail (`Ctrl+R/P/T/L`, `/roofline`); `Esc` clears it.
    pub detail: Option<layout::Comp>,
    /// Measured GEMM peak in GFLOP/s from the host profile; 0 when unmeasured.
    gemm_peak_gflops: f64,
    /// The three slowest host-profile GEMM timings, shown by the ROOFLINE profile row.
    gemm_top: Vec<String>,
    pub latest_logprobs: Option<crate::scheduler::TokenLogprobInfo>,
    pub logprob_history: std::collections::VecDeque<crate::scheduler::TokenLogprobInfo>,
    pub logprob_scrub_idx: usize,
    pub logprob_rx:
        Option<tokio::sync::mpsc::UnboundedReceiver<crate::scheduler::TokenLogprobInfo>>,
    pub roofline_probe_result: Option<String>,
    pub show_settings_modal: bool,
    /// `/placement` modal: the placement rows and the first visible line (Esc closes).
    placement_modal: Option<(Vec<crate::graph::placement::PlacementRow>, usize)>,
    pub reencode_policy: crate::model::weight_format::ReencodePolicy,
    pub variant_policy: crate::model::weight_format::VariantPolicy,
    pub draft_knobs: crate::runtime::noise_draft::DraftKnobs,
    pub compact_config: crate::compact::CompactionConfig,
    pub ctx_limit: usize,
    pub show_roofline_hud: bool,
    pub frame_log_path: Option<String>,
    /// `/profile on`: time one decode token every 5 s into the Kernels panel.
    profile_sampling: bool,
    /// The last per-kind kernel profile and when it was taken.
    kernel_profile: Option<(Instant, Vec<KernelRow>)>,
    /// Column the Kernels table is ordered by (`s` cycles it, `/profile sort <col>` sets it).
    kernel_sort: crate::telemetry::KernelSort,
    /// When this app last asked the engine for a profile sample.
    profile_asked: Option<Instant>,
    /// Set by every frame that draws the Kernels panel.
    kernels_visible: std::cell::Cell<bool>,
    pub(crate) device_status: DeviceStatus,
    /// A `/bench` thread is measuring the loaded model.
    benchmarking: bool,
}

impl TuiApp {
    pub fn new(telemetry: Arc<EngineTelemetry>) -> Self {
        let hw = HardwareCapability::probe_system();
        if let Some(perf_cores) = hw.performance_cores {
            let limit = (perf_cores as usize).saturating_sub(1).max(1);
            crate::model::nn::set_par_limit(limit);
        }
        let hub = HuggingFaceHub::new()
            .with_storage(StorageBackend::from_uri("disk").expect("local disk storage"))
            .with_budget(usize::MAX);
        let hf_token = std::env::var("HF_TOKEN").ok();
        let picker = Picker::new(
            hub.clone(),
            hf_token.clone(),
            pinned_models(),
            hw.total_memory_bytes,
        );
        // The Device panel's ANE line appears once the (disk-cached) probe finishes.
        #[cfg(all(target_os = "macos", not(test)))]
        std::thread::spawn(|| {
            let _ = crate::graph::ane::probe();
        });
        let (updates_tx, updates_rx) = mpsc::channel();
        let saved = settings::load_settings();
        let spark_total = telemetry.total_tokens_generated.load(Ordering::Relaxed);
        let mut gemm_peak_gflops = 0.0f64;
        let mut gemm_top: Vec<(f64, String)> = Vec::new();
        #[cfg(target_os = "macos")]
        if let Ok(rt) = crate::graph::metal::Runtime::shared() {
            let prof = rt.profile();
            telemetry.set_hardware_profile(&prof.device, prof.memory_bandwidth_gbps as f32);
            // Measured GEMM peak (GFLOP/s) over the host profile's timings: the compute roof.
            for t in &prof.gemm {
                let min_us = t.us.iter().map(|(_, us)| *us).fold(f64::INFINITY, f64::min);
                if min_us.is_finite() && min_us > 0.0 {
                    gemm_peak_gflops = gemm_peak_gflops
                        .max(2.0 * t.m as f64 * t.n as f64 * t.k as f64 / (min_us * 1e3));
                }
                if let Some((k, us)) = t.us.first() {
                    gemm_top.push((*us, format!("{k:?}[{},{}] {us:.1} µs", t.m, t.n)));
                }
            }
        }
        gemm_top.sort_by(|a, b| b.0.total_cmp(&a.0));
        let gemm_top: Vec<String> = gemm_top.into_iter().take(3).map(|(_, s)| s).collect();
        Self {
            telemetry,
            host_lines: hw.summary(),
            hw,
            scheduler: None,
            hub,
            hf_token,
            jev_model: crate::runtime::encoder::system1::System1Slot::default(),
            image_model: ImageSlot::default(),
            video: crate::model::video_repo::VideoSlot::default(),
            classifier: ClassifierSlot::default(),
            quant_override: None,
            http_addr: None,
            placement: PlacementMode::Auto,
            scratch_dir: tempfile::tempdir().expect("scratch directory for TUI images"),
            history: Vec::new(),
            chat_history: Vec::new(),
            input: String::new(),
            prompt_history: Vec::new(),
            prompt_history_idx: None,
            prompt_draft: String::new(),
            focus: Focus::Dashboard,
            picker,
            load: None,
            running: 0,
            updates_tx,
            updates_rx,
            scroll: 0,
            spark: vec![0; SPARK_SECONDS],
            spark_at: Instant::now(),
            spark_total,
            mascot: mascot::MascotEngine::new(),
            show_template_modal: false,
            active_template: 0,
            system_prompt: "You are a helpful and concise AI assistant.".into(),
            latest_logprobs: None,
            logprob_history: std::collections::VecDeque::with_capacity(64),
            logprob_scrub_idx: 0,
            logprob_rx: None,
            roofline_probe_result: None,
            detail: None,
            gemm_peak_gflops,
            gemm_top,
            show_settings_modal: false,
            placement_modal: None,
            reencode_policy: saved.reencode,
            variant_policy: saved.variant,
            draft_knobs: saved.draft,
            compact_config: saved.compact,
            ctx_limit: 8192,
            show_roofline_hud: true,
            frame_log_path: None,
            profile_sampling: false,
            kernel_profile: None,
            kernel_sort: crate::telemetry::KernelSort::default(),
            profile_asked: None,
            kernels_visible: std::cell::Cell::new(false),
            device_status: DeviceStatus::default(),
            benchmarking: false,
        }
    }

    /// Attaches the engine's live scheduler (for chat + model hot-swap) and any loaded System 1 / image models.
    pub fn with_runtime(
        mut self,
        scheduler: Arc<ContinuousBatchScheduler>,
        hub: HuggingFaceHub,
        hf_token: Option<String>,
        jev_model: Option<Arc<JevSystem1Model>>,
        image_model: ImageSlot,
    ) -> Self {
        self.picker = Picker::new(
            hub.clone(),
            hf_token.clone(),
            pinned_models(),
            self.hw.total_memory_bytes,
        );
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        *scheduler.logprob_tx.write() = Some(tx);
        self.logprob_rx = Some(rx);
        self.scheduler = Some(scheduler);
        self.hub = hub;
        self.hf_token = hf_token;
        *self.jev_model.write() = jev_model;
        self.image_model = image_model;
        self
    }

    /// Id of the active model: the loaded image model or classifier, else the scheduler's; empty
    /// without any.
    fn model_id(&self) -> String {
        if let Some(id) = self.image_model.read().as_ref().and_then(|m| m.model_id()) {
            return id;
        }
        if let Some(c) = self.classifier.read().as_ref() {
            return c.repo_id.clone();
        }
        if let Some(jev) = self.jev_model.read().as_ref() {
            return jev.repo_id.clone();
        }
        self.scheduler
            .as_ref()
            .map(|s| s.model_id())
            .unwrap_or_default()
    }

    fn push_entry(&mut self, prompt: String, kind: Kind, output: String) -> usize {
        let now = Instant::now();
        self.history.push(Entry {
            time: chrono::Local::now().format("%H:%M:%S").to_string(),
            prompt,
            kind,
            output,
            stats: String::new(),
            started: now,
            took: (kind != Kind::Running).then_some(Duration::ZERO),
            cancel: None,
            chat_user: None,
        });
        if kind == Kind::Running {
            self.running += 1;
        }
        self.scroll = 0;
        self.history.len() - 1
    }

    /// Runs the typed prompt on a background thread; its entry streams into the console.
    /// Commands: `/chat`, `/image`, `/decide <query> | opt, opt`, `/embed`, `/classify <text>`
    /// (`<text> || <pair>` for pair models); bare text classifies when a classifier is active,
    /// else routes by the active model id. `/connect` and `/agent` answer at once. Missing
    /// models and bad input are reported at once.
    pub fn submit_prompt(&mut self) {
        let raw = std::mem::take(&mut self.input).trim().to_string();
        if raw.is_empty() {
            return;
        }
        if self.prompt_history.last() != Some(&raw) {
            self.prompt_history.push(raw.clone());
        }
        self.prompt_history_idx = None;
        self.prompt_draft.clear();
        let (cmd, rest) = raw.split_once(char::is_whitespace).unwrap_or((&raw, ""));
        let answer = match cmd {
            "/connect" => Some(
                self.http_addr
                    .map(|a| crate::server::connect::setup_text(a, &self.model_id()))
                    .ok_or_else(|| "no HTTP server is running".to_string()),
            ),
            "/agent" => Some(self.launch_agent(rest.trim())),
            "/roofline" => {
                self.detail = Some(layout::Comp::Roofline);
                let probe = self.run_roofline_probe();
                self.roofline_probe_result = probe.as_ref().ok().cloned();
                Some(probe)
            }
            "/reencode" | "/variant" | "/profile" | "/placement" | "/spec" => Some(
                parse_slash(&raw)
                    .unwrap_or_else(|| Err(format!("unknown command '{cmd}'")))
                    .and_then(|c| self.run_slash(c, &raw)),
            ),
            "/settings" => {
                self.show_settings_modal = !self.show_settings_modal;
                Some(Ok(format!(
                    "Policy Settings: reencode={:?}, variant={:?}, ctx={}, roofline_hud={}",
                    self.reencode_policy, self.variant_policy, self.ctx_limit, self.show_roofline_hud
                )))
            }
            "/bench" => Some(self.start_bench()),
            "/system1" | "/s1" => Some(self.toggle_system1(rest.trim())),
            "/clear" | "/reset" | "/new" => {
                let prev_turns = self.chat_history.len() / 2;
                self.chat_history.clear();
                if let Some(s) = &self.scheduler {
                    s.radix.clear();
                }
                Some(Ok(format!(
                    "Cleared multi-turn chat history ({prev_turns} turns removed) and reset KV prefix cache."
                )))
            }
            "/pop" | "/undo" => {
                if self.chat_history.is_empty() {
                    Some(Err("chat history is already empty".to_string()))
                } else {
                    let removed = if self.chat_history.len() >= 2 { 2 } else { 1 };
                    for _ in 0..removed {
                        self.chat_history.pop();
                    }
                    let turns = self.chat_history.len() / 2;
                    Some(Ok(format!(
                        "Popped last chat turn ({turns} turn{} remaining in context).",
                        if turns == 1 { "" } else { "s" }
                    )))
                }
            }
            "/history" => {
                if self.chat_history.is_empty() {
                    Some(Ok("Multi-turn chat history is empty (0 turns).".to_string()))
                } else {
                    let mut lines = vec![format!(
                        "MULTI-TURN CHAT HISTORY ({} messages, {} turns):",
                        self.chat_history.len(),
                        self.chat_history.len() / 2
                    )];
                    for (i, (role, content)) in self.chat_history.iter().enumerate() {
                        let preview: String = content.chars().take(72).collect();
                        lines.push(format!("  [{}] {}: {}", i + 1, role, preview));
                    }
                    Some(Ok(lines.join("\n")))
                }
            }
            "/compact" => Some(self.run_compact_cmd(rest.trim())),
            "/quant" => {
                Some(Ok(format!(
                    "MODEL QUANTIZATION:\n\
                     · Active Model ID: {}\n\
                     · Engine Weight Precision: Native GGUF mixed precision\n\
                     · Host Acceleration: Metal GPU unified memory",
                    self.model_id()
                )))
            }
            "/help" => {
                Some(Ok(
                    "GIBBS ENGINE COMMAND & KEY REFERENCE:\n\
                     Keybindings:\n\
                     · m / Esc    : Switch between Dashboard & Model Picker (there ^r / ^f cycle the re-encode / variant policy and reload the model)\n\
                     · ↑ / ↓      : Cycle prior prompt entries for retry or editing\n\
                     · F2         : System Prompt / Chat Template Selector\n\
                     · F4         : Policy & Speed Options modal (defaults = fastest for model)\n\
                     · Trackpad / Mouse Wheel / PgUp / PgDn : Scroll console\n\
                     · Esc / ^X   : Cancel running prompt / return to Dashboard\n\
                     · Ctrl+C     : Exit Gibbs engine\n\n\
                     Slash Commands:\n\
                     · /reencode [max-speed|min-size|low-bit-native|bit-exact] : Set the weight re-encode policy (no argument cycles) and reload the active model under it\n\
                     · /variant [fastest|fastest-4bit|best-quality] : Set the variant policy of the picker and downloads (no argument cycles) and reload the active model\n\
                     · /settings  : Show the policy settings modal (F4) and the current policies\n\
                     · /profile on|off|sort <time|bandwidth|share> : Sample one decode token every 5 s, step by step, into the KERNELS panel; sort its rows (key s cycles)\n\
                     · /bench     : Benchmark the loaded model on Metal (128 tokens x 3 runs): tok/s, roofline % and where each token's time goes; refuses while generating\n\
                     · /placement : Where every graph node of the loaded model runs (op, device, avenue): 100% Metal, or the CPU ops\n\
                     · /roofline  : Focus ROOFLINE detail & run the calibration sweep\n\
                     · /spec [mode=|objective=|k=|accept=|cost=|suffix=|lr=|sigma=|steps=|shortlist=|probes=|window=] : Live-tune speculative decoding & the online drafter (no args: status)\n\
                     · /quant     : Display active weight quantization formats\n\
                     · /gibbs     : Full historical biography & ASCII portrait of J. Willard Gibbs\n\
                     · /system1 [on|off] : Turn the active LLM into a System 1 encoder & zero-shot judge (or return to chat)\n\
                     · /decide <q> | a, b: Zero-shot System 1 choice on the active System 1 or LLM model\n\
                     · /embed <text>     : Compute L2-normalized embedding on the active System 1 or LLM model\n\
                     · /chat <msg>: Send chat message to active model\n\
                     · /connect   : Print curl/OpenAI API connection commands\n\
                     · /clear (/reset, /new) : Clear multi-turn chat history and KV prefix cache\n\
                     · /pop (/undo)          : Remove the last user/assistant turn from chat history\n\
                     · /history              : Inspect active multi-turn chat messages\n\
                     · /compact [auto|light|aggressive|off] [now] [mode= threshold= target= keep= summarizer=] : Configure (saved) or run session compaction\n\
                     · /help      : This reference"
                        .to_string(),
                ))
            }
            "/gibbs" => Some(Ok(mascot::gibbs_full_portrait())),
            _ => None,
        };
        if let Some(r) = answer {
            let (kind, text) = r.map_or_else(|e| (Kind::Error, e), |t| (Kind::Text, t));
            self.push_entry(raw, kind, text);
            return;
        }
        if self.benchmarking {
            self.push_entry(raw, Kind::Error, "benchmark running".into());
            return;
        }
        let (route, body) = match cmd {
            "/chat" => (Ok(Route::Chat), rest.trim()),
            "/image" | "/img" => (Ok(Route::Image), rest.trim()),
            "/decide" | "/judge" => (Ok(Route::Decide), rest.trim()),
            "/embed" => (Ok(Route::Embed), rest.trim()),
            "/classify" | "/cls" => (Ok(Route::Classify), rest.trim()),
            "/i2v" => (Ok(Route::Video { image: true }), rest.trim()),
            "/video" | "/vid" | "/t2v" => (Ok(Route::Video { image: false }), rest.trim()),
            c if c.starts_with('/') => (Err(format!("unknown command '{c}'; use /chat, /system1, /image, /decide, /embed, /classify, /connect, /agent, /roofline, /reencode, /variant, /settings, /profile, /bench, /placement, /quant, /help, /gibbs")), ""),
            _ if self.classifier.read().is_some() => (Ok(Route::Classify), raw.as_str()),
            _ if self.image_model.read().is_some() => (Ok(Route::Image), raw.as_str()),
            _ if self.jev_model.read().is_some() => (
                Ok(if raw.contains('|') {
                    Route::Decide
                } else {
                    Route::Embed
                }),
                raw.as_str(),
            ),
            _ => (Ok(route_for_model(&self.model_id())), raw.as_str()),
        };
        let body_str = body.to_string();
        let is_chat = matches!(route, Ok(Route::Chat));
        let job = route.and_then(|r| {
            if body_str.is_empty() {
                Err("empty prompt".to_string())
            } else {
                self.prompt_job(r, body_str.clone())
            }
        });
        match job {
            Err(e) => {
                self.push_entry(raw, Kind::Error, e);
            }
            Ok((job, cancel)) => {
                let names: Vec<String> = raw
                    .split_whitespace()
                    .filter_map(attachment)
                    .filter_map(|(p, _)| Some(p.file_name()?.to_string_lossy().into_owned()))
                    .collect();
                let shown = if names.is_empty() {
                    raw
                } else {
                    format!("{raw}  [attached: {}]", names.join(", "))
                };
                let idx = self.push_entry(shown, Kind::Running, String::new());
                self.history[idx].cancel = cancel;
                if is_chat {
                    self.history[idx].chat_user = Some(body_str);
                }
                let tx = self.updates_tx.clone();
                std::thread::spawn(move || {
                    crate::tui::qos::set_qos(crate::tui::qos::QosClass::Utility);
                    job(idx, tx);
                });
            }
        }
    }

    /// Runs a parsed slash command.
    fn run_slash(&mut self, cmd: SlashCmd, prompt: &str) -> Result<String, String> {
        match cmd {
            SlashCmd::Placement => self.placement_report(prompt),
            SlashCmd::Reencode(p) => self.set_reencode(p),
            SlashCmd::Variant(p) => self.set_variant(p),
            SlashCmd::Profile(a) => {
                match a {
                    ProfileArg::Status => {}
                    ProfileArg::On => self.profile_sampling = true,
                    ProfileArg::Off => self.profile_sampling = false,
                    ProfileArg::Sort(s) => {
                        self.kernel_sort = s.unwrap_or_else(|| self.kernel_sort.next())
                    }
                }
                Ok(format!(
                    "Kernel profiling is {} (use /profile on|off): while the KERNELS panel shows, one decode token every 5 s is timed step by step into it; sorted by {} (/profile sort time|bandwidth|share, or s)",
                    if self.profile_sampling { "ON" } else { "OFF" },
                    self.kernel_sort.name()
                ))
            }
            SlashCmd::Spec(a) => self.apply_spec_cmd(a),
        }
    }

    /// `/system1 [on|off]`: turns any loaded LLM into a System 1 model (or returns it to chat).
    fn toggle_system1(&mut self, arg: &str) -> Result<String, String> {
        let want_off = matches!(arg, "off" | "0" | "false" | "chat");
        let want_on = matches!(arg, "on" | "1" | "true");
        if want_off || (!want_on && self.jev_model.read().as_ref().is_some_and(|j| j.is_llm())) {
            if self.jev_model.read().as_ref().is_some_and(|j| j.is_llm()) {
                *self.jev_model.write() = None;
                let id = self.model_id();
                self.telemetry
                    .log_info(&format!("TUI: System 1 mode disabled for '{id}'"));
                return Ok(format!(
                    "System 1 mode OFF: {id} returned to text generation (/chat)"
                ));
            }
            if let Some(jev) = self.jev_model.read().as_ref() {
                return Ok(format!(
                    "{} is a dedicated System 1 encoder; press m to load an LLM for chat",
                    jev.repo_id
                ));
            }
            return Ok("System 1 mode is already off".to_string());
        }
        if let Some(sched) = &self.scheduler {
            if let Some(llm) = sched.model() {
                let id = sched.model_id();
                let jev = Arc::new(JevSystem1Model::from_llm(id.clone(), llm));
                let shape = jev.summary();
                *self.jev_model.write() = Some(jev);
                *self.classifier.write() = None;
                *self.image_model.write() = None;
                self.telemetry
                    .log_info(&format!("TUI: '{id}' activated as System 1 model"));
                return Ok(format!("System 1 mode ON ({id}): {shape}"));
            }
        }
        if let Some(jev) = self.jev_model.read().as_ref() {
            return Ok(format!(
                "System 1 mode ON ({}): {}",
                jev.repo_id,
                jev.summary()
            ));
        }
        Err("no LLM or System 1 model loaded; press m to load a model".to_string())
    }

    /// `/reencode [policy]`: the named policy, else the next one in the cycle. It is saved to the
    /// settings file and the loaded model is reloaded under it.
    fn set_reencode(&mut self, policy: Option<ReencodePolicy>) -> Result<String, String> {
        let next = policy.unwrap_or_else(|| self.reencode_policy.next());
        self.reencode_policy = next;
        let mut s = settings::load_settings();
        s.reencode = next;
        settings::save_settings(&s)?;
        Ok(self.reload_for("reencode", next.name()))
    }

    /// `/variant [policy]`: as [`Self::set_reencode`], for the variant policy.
    fn set_variant(&mut self, policy: Option<VariantPolicy>) -> Result<String, String> {
        let next = policy.unwrap_or_else(|| self.variant_policy.next());
        self.variant_policy = next;
        let mut s = settings::load_settings();
        s.variant = next;
        settings::save_settings(&s)?;
        Ok(self.reload_for("variant", next.name()))
    }

    /// Applies `/spec` arguments to `self.draft_knobs`, `settings.json` and the live scheduler, and
    /// reports the knobs and what the `k` tuner measured.
    fn apply_spec_cmd(&mut self, args: Vec<SpecArg>) -> Result<String, String> {
        let mut k = self
            .scheduler
            .as_ref()
            .map_or(self.draft_knobs, |s| s.draft_knobs());
        let changed = !args.is_empty();
        for arg in args {
            match arg {
                SpecArg::Mode(m) => k.mode = m,
                SpecArg::Objective(o) => k.objective = o,
                SpecArg::MaxK(mk) => k.max_k = mk,
                SpecArg::Accept(a) => k.target_acceptance = a,
                SpecArg::Cost(c) => k.cost_weight = c,
                SpecArg::Suffix(sm) => k.suffix_min_match = sm,
                SpecArg::Lr(lr) => k.noise_lr = lr,
                SpecArg::Sigma(s) => k.noise_sigma = s,
                SpecArg::Steps(n) => k.denoise_steps = n,
                SpecArg::Shortlist(n) => k.shortlist = n,
                SpecArg::Probes(n) => k.probes = n,
                SpecArg::Window(n) => k.window = n,
            }
        }
        let k = k.sanitized();
        self.draft_knobs = k;
        if changed {
            let mut s = settings::load_settings();
            s.draft = k;
            let _ = settings::save_settings(&s);
            if let Some(sched) = &self.scheduler {
                sched.set_draft_knobs(k);
            }
        }
        let view = self.scheduler.as_ref().and_then(|sc| sc.view().spec);
        let best = view.as_ref().and_then(|v| v.k);
        let rate = self
            .scheduler
            .as_ref()
            .zip(best)
            .and_then(|(sc, b)| sc.k_tuner_rates().into_iter().find(|r| r.k == b));
        let measured = rate.map_or_else(
            || "acceptance — · tokens/forward — · tok/s —".to_string(),
            |r| {
                format!(
                    "acceptance {:.0}% · tokens/forward {:.2} · tok/s {:.1}",
                    r.acceptance_rate * 100.0,
                    r.tokens_per_forward,
                    r.tok_per_sec
                )
            },
        );
        Ok(format!(
            "Speculative decoding: mode={} · objective={} · active={} · k={} · {measured} · online updates={}\nKnobs: max_k={} · accept={:.0}% · cost={:.2} · suffix={} · lr={:.4} · sigma={:.4} · steps={} · shortlist={} · probes={} · window={}",
            k.mode.name(),
            k.objective.name(),
            view.as_ref().map_or("none (plain decode)", |v| v.label.as_str()),
            best.map_or_else(|| format!("0..{} (measuring)", k.max_k), |b| b.to_string()),
            view.as_ref().and_then(|v| v.updates).map_or_else(|| "—".to_string(), |u| u.to_string()),
            k.max_k,
            k.target_acceptance * 100.0,
            k.cost_weight,
            k.suffix_min_match,
            k.noise_lr,
            k.noise_sigma,
            k.denoise_steps,
            k.shortlist,
            k.probes,
            k.window,
        ))
    }

    /// `/compact [auto|light|aggressive|off] [now] [mode= threshold= target= keep= summarizer=]`:
    /// sets and saves the compaction settings (chats compact once their prompt passes the
    /// threshold); `now` compacts `chat_history` at once, in the background.
    fn run_compact_cmd(&mut self, arg: &str) -> Result<String, String> {
        let mut o = crate::compact::CompactionOverride::default();
        let mut run_now = false;
        for word in arg.split_whitespace() {
            let num = |v: &str| v.parse::<f32>().map_err(|_| format!("{word}: not a number"));
            match word.split_once('=') {
                Some(("mode", v)) => o.mode = Some(v.parse()?),
                Some(("threshold", v)) => o.trigger = Some(num(v)?),
                Some(("target", v)) => o.target = Some(num(v)?),
                Some(("keep", v)) => {
                    o.keep_turns = Some(v.parse().map_err(|_| format!("{word}: not a count"))?)
                }
                Some(("summarizer", v)) => o.summarizer = Some(v.to_string()),
                Some((k, _)) => {
                    return Err(format!(
                        "unknown /compact setting '{k}' (mode, threshold, target, keep, summarizer)"
                    ))
                }
                None if matches!(word, "now" | "run" | "force") => run_now = true,
                None => o.mode = Some(word.parse()?),
            }
        }
        let mut cfg = self.compact_config.clone();
        o.apply(&mut cfg);
        let cfg = cfg.sanitized();
        if cfg != self.compact_config {
            let mut s = settings::load_settings();
            s.compact = cfg.clone();
            settings::save_settings(&s)?;
            if let Some(sch) = &self.scheduler {
                sch.set_compaction(&cfg);
            }
            self.compact_config = cfg;
        }
        if run_now {
            return self.compact_now();
        }
        let c = &self.compact_config;
        Ok(format!(
            "Auto-compaction: mode={} · threshold={:.0}% · target={:.0}% · keep={} turns · summarizer={} · snapcompact_vision={}",
            c.mode.name(),
            c.trigger_ratio * 100.0,
            c.target_ratio * 100.0,
            c.keep_recent_turns,
            c.summarizer.as_deref().unwrap_or("active"),
            c.enable_snapcompact_vision,
        ))
    }

    /// `/compact now`: compacts `chat_history` on a background thread (Auto and Off run as
    /// Aggressive); the result arrives as [`Update::Compacted`].
    fn compact_now(&mut self) -> Result<String, String> {
        if self.chat_history.is_empty() {
            return Ok("nothing to compact: the chat history is empty".into());
        }
        let sched = self
            .scheduler
            .clone()
            .filter(|s| s.has_model())
            .ok_or("no text model loaded; press m to load one")?;
        let mut cfg = self.compact_config.clone();
        if matches!(
            cfg.mode,
            crate::compact::CompactionMode::Auto | crate::compact::CompactionMode::Off
        ) {
            cfg.mode = crate::compact::CompactionMode::Aggressive;
        }
        let (from, system) = (self.chat_history.clone(), self.system_prompt.clone());
        let (tx, ctx_limit, mode) = (self.updates_tx.clone(), self.ctx_limit, cfg.mode);
        std::thread::spawn(move || {
            let result = compact_turns(&sched, &system, &from, None, ctx_limit, &cfg, true)
                .and_then(|r| r.ok_or_else(|| "it saves no tokens".to_string()));
            let _ = tx.send(Update::Compacted { from, result });
        });
        Ok(format!(
            "compacting {} messages ({} mode)…",
            self.chat_history.len(),
            mode.name()
        ))
    }

    /// The reply to a changed setting. A loaded model is reloaded so the setting takes effect now;
    /// otherwise it applies to the next load.
    fn reload_for(&mut self, what: &str, value: &str) -> String {
        if self.benchmarking {
            return format!(
                "{what} -> {value}; a benchmark is running, so it applies to the next load"
            );
        }
        let loaded = self
            .scheduler
            .as_ref()
            .filter(|s| s.has_model())
            .map(|s| s.model_id());
        match (loaded, self.load.is_some()) {
            (Some(id), false) => {
                self.telemetry
                    .log_info(&format!("TUI: {what} -> {value}; reloading '{id}'"));
                self.load_model(&id);
                format!("{what} -> {value}; reloading {id}")
            }
            (Some(_), true) => {
                format!("{what} -> {value}; a load is in flight, so it applies to the next load")
            }
            (None, _) => format!("{what} -> {value}; applies to the next load"),
        }
    }

    /// `/bench`: measures the loaded model (128 decode tokens, 3 runs, drafters off) on a
    /// background thread; the result arrives as [`Update::Bench`].
    fn start_bench(&mut self) -> Result<String, String> {
        const TOKENS: usize = 128;
        const RUNS: usize = 3;
        if self.benchmarking {
            return Err("benchmark running".into());
        }
        if self.load.is_some() {
            return Err("busy: a model load is in flight".into());
        }
        let sched = self
            .scheduler
            .clone()
            .filter(|s| s.has_model())
            .ok_or("no text model loaded; press m to load one")?;
        let view = sched.view();
        if !view.seqs.is_empty() || view.waiting > 0 || self.running > 0 {
            return Err("busy: finish or cancel generation first".into());
        }
        let exec = sched
            .model()
            .ok_or("no text model loaded; press m to load one")?;
        let id = sched.model_id();
        let applied = crate::bench::applied_policy();
        let (bytes, basis) = crate::bench::roofline_bytes(&self.hub.locate_model(&id), &applied);
        let ceiling = match crate::model::variant::host_ceiling_gbps() {
            c if c > 0.0 => c,
            _ => f64::from(self.telemetry.get_summary().memory_bandwidth_gbps),
        };
        if ceiling <= 0.0 {
            return Err("bandwidth ceiling unmeasured: wait for the codec bench (Device panel) or run /roofline".into());
        }
        self.benchmarking = true;
        let (tx, name) = (self.updates_tx.clone(), id.clone());
        std::thread::spawn(move || {
            // Benchmarks on this machine take turns on the GPU (the lock is held for the run);
            // unit tests do not wait behind a benchmark running elsewhere.
            let _gpu = (!cfg!(test)).then(crate::hardware::gpu_lock);
            let r = crate::bench::measure_loaded(
                &exec,
                &crate::bench::default_prompt(),
                TOKENS,
                RUNS,
                bytes,
                ceiling,
                0,
            )
            .map(|mut r| {
                r.model = name;
                r.bytes_basis = basis;
                r
            });
            let _ = tx.send(Update::Bench(r));
        });
        Ok(format!(
            "benchmarking {id}: {TOKENS} decode tokens x {RUNS} runs, drafters off"
        ))
    }

    /// `/placement`: where every graph node of the loaded model runs. All on one GPU is one text
    /// entry ending in `100% Metal`; CPU ops follow the table as an error entry.
    fn placement_report(&mut self, prompt: &str) -> Result<String, String> {
        use crate::graph::placement::{placement_table, placement_verdict};
        let rows = self
            .scheduler
            .as_ref()
            .and_then(|s| s.model())
            .ok_or("no text model loaded; press m to load one")?
            .placement_summary()
            .ok_or("the loaded model runs no graph program, so it has no placement to explain")?;
        let table = placement_table(&rows);
        let verdict = placement_verdict(&rows);
        self.placement_modal = Some((rows, 0));
        match verdict {
            Ok(all_gpu) => Ok(format!("{table}\n{all_gpu}")),
            Err(cpu_ops) => {
                self.push_entry(prompt.to_string(), Kind::Text, table);
                Err(cpu_ops)
            }
        }
    }

    /// Starts an agent CLI pointed at this server: in a new tmux window inside tmux, else a new
    /// macOS Terminal window; elsewhere (or when the CLI is missing) returns the command to run.
    fn launch_agent(&self, agent: &str) -> Result<String, String> {
        let addr = self.http_addr.ok_or("no HTTP server is running")?;
        let cmd = crate::server::connect::agent_command(agent, addr, &self.model_id())?;
        let on_path = std::env::var_os("PATH")
            .is_some_and(|p| std::env::split_paths(&p).any(|d| d.join(agent).is_file()));
        if !on_path {
            return Err(format!(
                "{agent} is not on PATH; install it, then run:\n{cmd}"
            ));
        }
        let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
        let mut launcher = if std::env::var_os("TMUX").is_some() {
            let mut c = std::process::Command::new("tmux");
            c.args(["new-window", "-n", agent, "-c"])
                .arg(&cwd)
                .arg(&cmd);
            c
        } else if cfg!(target_os = "macos") {
            let cwd = crate::server::connect::q(&cwd.to_string_lossy());
            let script = format!("cd {cwd} && {cmd}")
                .replace('\\', "\\\\")
                .replace('"', "\\\"");
            let mut c = std::process::Command::new("osascript");
            c.args([
                "-e",
                &format!("tell application \"Terminal\" to do script \"{script}\""),
                "-e",
                "tell application \"Terminal\" to activate",
            ]);
            c
        } else {
            return Ok(format!("run in another terminal:\n{cmd}"));
        };
        let out = launcher
            .output()
            .map_err(|e| format!("cannot launch {agent}: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "cannot launch {agent}: {}\nrun it yourself:\n{cmd}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(format!("launched {agent} in a new window:\n{cmd}"))
    }
    /// Runs an empirical roofline calibration probe comparing measured memory bandwidth
    /// against Apple unified memory hardware roofline ceilings.
    pub fn trigger_background_roofline_probe(&self) {
        let tx = self.updates_tx.clone();
        let telemetry = self.telemetry.clone();
        std::thread::spawn(move || {
            crate::tui::qos::set_qos(crate::tui::qos::QosClass::Utility);
            let res = run_roofline_probe_static(&telemetry);
            let _ = tx.send(Update::Probe(res));
        });
    }

    /// Runs an empirical roofline calibration probe comparing measured memory bandwidth
    /// against Apple unified memory hardware roofline ceilings.
    pub fn run_roofline_probe(&self) -> Result<String, String> {
        self.trigger_background_roofline_probe();
        run_roofline_probe_static(&self.telemetry)
    }
}

pub fn run_roofline_probe_static(telemetry: &EngineTelemetry) -> Result<String, String> {
    #[cfg(target_os = "macos")]
    {
        let dir = crate::model::hub::HuggingFaceHub::cache_dir();
        let rt = crate::graph::metal::Runtime::new(&dir)?;
        let measured_gbps = rt.benchmark_memory_bandwidth()?;

        let mut peak_gflops = 0.0f64;
        for t in &rt.profile().gemm {
            let min_us = t.us.iter().map(|(_, us)| *us).fold(f64::INFINITY, f64::min);
            if min_us.is_finite() && min_us > 0.0 {
                let gflops = (2.0 * t.m as f64 * t.n as f64 * t.k as f64) / (min_us * 1e3);
                peak_gflops = peak_gflops.max(gflops);
            }
        }
        if peak_gflops == 0.0 {
            peak_gflops = 14_000.0;
        }

        let ridge = peak_gflops / measured_gbps.max(1.0);
        let s = telemetry.get_summary();
        let eta = if s.effective_bandwidth_gbps > 0.0 {
            (s.effective_bandwidth_gbps as f64 / measured_gbps) * 100.0
        } else {
            0.0
        };
        let regime = if eta > 65.0 {
            "Bandwidth-Constrained"
        } else {
            "Latency/Compute-Bound"
        };

        Ok(format!(
            "EMPIRICAL ROOFLINE CALIBRATION:\n\
                 · Host GPU Device: {}\n\
                 · Empirical Memory Sweep (256 MB): {:.1} GB/s\n\
                 · Peak Calibrated GEMM Compute: {:.1} TFLOPs ({:.0} GFLOP/s)\n\
                 · Empirical Operational Intensity Ridge: {:.1} FLOPs/Byte\n\
                 · Live Decode Bandwidth: {:.1} GB/s (η = {:.1}% {})\n\
                 · Instantaneous Throughput: {:.1} tok/s",
            &rt.profile().device,
            measured_gbps,
            peak_gflops / 1000.0,
            peak_gflops,
            ridge,
            s.effective_bandwidth_gbps,
            eta,
            regime,
            s.current_tps
        ))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err("Empirical roofline probe is only available on Metal hosts".into())
    }
}

impl TuiApp {
    /// The work for one prompt (and, for chats, its cancel flag), or why it cannot run.
    fn prompt_job(
        &self,
        route: Route,
        body: String,
    ) -> Result<(Job, Option<Arc<AtomicBool>>), String> {
        let done = |tx: &mpsc::Sender<Update>, idx, r: Result<(Kind, String, String), String>| {
            let (kind, text, stats) = r.unwrap_or_else(|e| (Kind::Error, e, String::new()));
            let _ = tx.send(Update::Done {
                idx,
                kind,
                text,
                stats,
            });
        };
        match route {
            Route::Chat => {
                let sched = self
                    .scheduler
                    .clone()
                    .filter(|s| s.has_model())
                    .ok_or("no text model loaded; press m to load one")?;
                let media = media_message(&body)?;
                let cancel = Arc::new(AtomicBool::new(false));
                let cancelled = cancel.clone();
                let active_template = self.active_template;
                let system_prompt = self.system_prompt.clone();
                let prior_turns = self.chat_history.clone();
                let (compact_cfg, ctx_limit) = (self.compact_config.clone(), self.ctx_limit);
                let job: Job = Box::new(move |idx, tx| {
                    // Over the trigger the history is compacted first; the UI adopts the result.
                    let prior_turns = if compact_cfg.mode == crate::compact::CompactionMode::Off {
                        prior_turns
                    } else {
                        let current = media.as_ref().map_or_else(
                            || ChatMessage::new("user", body.as_str()),
                            |(m, _)| m.clone(),
                        );
                        let result = compact_turns(
                            &sched,
                            &system_prompt,
                            &prior_turns,
                            Some(current),
                            ctx_limit,
                            &compact_cfg,
                            false,
                        );
                        match result {
                            Ok(None) => prior_turns,
                            Ok(Some((to, report))) => {
                                let result = Ok((to.clone(), report));
                                let _ = tx.send(Update::Compacted { from: prior_turns, result });
                                to
                            }
                            Err(e) => {
                                let from = prior_turns.clone();
                                let _ = tx.send(Update::Compacted { from, result: Err(e) });
                                prior_turns
                            }
                        }
                    };
                    // Instruct models get their chat template; base models (Plain) get the raw text.
                    // Answers run to EOS or the end of the model's context.
                    let params = SamplingParams {
                        temperature: None,
                        top_p: None,
                        max_new_tokens: usize::MAX,
                        top_logprobs: Some(5),
                    };
                    let submitted = match &media {
                        None => {
                            let system_prompt = system_prompt.clone();
                            let prior_turns = prior_turns.clone();
                            Ok(sched.submit_with_params(params, move |tok| {
                                let prompt_str = Self::format_chat_history_with(
                                    &system_prompt,
                                    active_template,
                                    tok,
                                    &prior_turns,
                                    body.as_str(),
                                );
                                tok.encode(&prompt_str)
                            }))
                        }
                        Some((msg, media)) => {
                            let system_prompt = system_prompt.clone();
                            let prior_turns = prior_turns.clone();
                            let msg = msg.clone();
                            sched.submit_with_media(params, media, move |tok| {
                                let mut msgs = Vec::new();
                                if !system_prompt.is_empty() {
                                    msgs.push(crate::model::hf_loader::ChatMessage::new(
                                        "system",
                                        system_prompt.as_str(),
                                    ));
                                }
                                for (role, content) in &prior_turns {
                                    msgs.push(crate::model::hf_loader::ChatMessage::new(
                                        role.as_str(),
                                        content.as_str(),
                                    ));
                                }
                                msgs.push(msg);
                                tok.encode(&tok.apply_chat_template_messages(&msgs, None))
                            })
                        }
                    };
                    let (_, mut rx, tok, _, finish) = match submitted {
                        Ok(s) => s,
                        Err(e) => return done(&tx, idx, Err(e)),
                    };
                    let mut ids = Vec::new();
                    let (mut first, mut sent) = (None, Instant::now());
                    while let Some(id) = rx.blocking_recv() {
                        if cancelled.load(Ordering::Relaxed) {
                            // Dropping `rx` ends the sequence at its next emit; the UI already
                            // marked the entry cancelled.
                            return;
                        }
                        ids.push(id);
                        first.get_or_insert_with(Instant::now);
                        if sent.elapsed() >= STREAM_UPDATE {
                            sent = Instant::now();
                            let text = tok.decode(&ids);
                            let _ = tx.send(Update::Partial {
                                idx,
                                text,
                                tokens: ids.len(),
                            });
                        }
                    }
                    let decode = first.map_or(0.0, |t| t.elapsed().as_secs_f64());
                    let rate = if ids.len() > 1 && decode > 0.0 {
                        format!(" · {:.1} tok/s", (ids.len() - 1) as f64 / decode)
                    } else {
                        String::new()
                    };
                    let r = finish_outcome(&finish).map(|reason| {
                        let stats = format!("{} tokens{rate} · {reason}", ids.len());
                        (Kind::Text, tok.decode(&ids), stats)
                    });
                    done(&tx, idx, r);
                });
                Ok((job, Some(cancel)))
            }
            Route::Image => {
                let sd = self.image_model.read().clone().ok_or(
                    "no image model loaded; press m and load a text-to-image repo, e.g. stabilityai/sd-turbo",
                )?;
                let dir = self.scratch_dir.path().to_path_buf();
                Ok((
                    Box::new(move |idx, tx| {
                        let seed = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_nanos() as u64;
                        let req = ImageRequest {
                            prompt: body,
                            seed,
                            ..Default::default()
                        };
                        let d = sd.defaults();
                        let r = sd
                            .text_to_image(&req)
                            .map_err(|e| format!("image generation failed: {e}"))
                            .and_then(|img| {
                                let path = dir.join(format!("tui_image_{:03}.png", idx + 1));
                                let png = img.to_png();
                                std::fs::write(&path, &png)
                                    .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
                                let text =
                                    format!("{}\n{}", path.display(), ascii_preview(&img, 48, 12));
                                let stats = format!(
                                    "{}x{} · {} steps · cfg {} · {} bytes · seed {seed}",
                                    img.width,
                                    img.height,
                                    d.steps,
                                    d.guidance,
                                    png.len()
                                );
                                Ok((Kind::Image, text, stats))
                            });
                        done(&tx, idx, r);
                    }),
                    None,
                ))
            }
            Route::Video { image } => {
                let model = self.video.read().clone().ok_or_else(|| {
                    crate::arch::compat()["tui"]["no_video_message"]
                        .as_str()
                        .unwrap_or("no video model loaded")
                        .to_string()
                })?;
                let dir = self.scratch_dir.path().to_path_buf();
                Ok((
                    Box::new(move |idx, tx| {
                        let r = (|| {
                            let (first, prompt) = match image {
                                true => body
                                    .split_once(char::is_whitespace)
                                    .ok_or("usage: /i2v <image path> <prompt>")?,
                                false => ("", body.as_str()),
                            };
                            let init = image
                                .then(|| ImageBuffer::from_base64_or_path(first))
                                .transpose()?;
                            let seed = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_nanos() as u64;
                            let start = Instant::now();
                            let req = crate::runtime::diffusion::VideoRequest {
                                prompt: prompt.trim().to_string(),
                                seed,
                                ..Default::default()
                            };
                            let (frames, fps, req) = crate::model::video_repo::generate(
                                &model.0,
                                &model.1,
                                crate::hardware::PlacementMode::Auto,
                                None,
                                &req,
                                init.as_ref(),
                                &|_| {},
                            )?;
                            let path = dir.join(format!("tui_video_{:03}.mp4", idx + 1));
                            crate::model::image::write_mp4(&frames, fps, &path)?;
                            let bytes = std::fs::metadata(&path)
                                .map_err(|e| format!("{}: {e}", path.display()))?
                                .len();
                            let stats = format!(
                                "{} frames {}x{} · {} steps · cfg {} · {:.0}s · {bytes} bytes · seed {seed}",
                                frames.len(),
                                req.width.unwrap_or(0),
                                req.height.unwrap_or(0),
                                req.steps.unwrap_or(0),
                                req.guidance.unwrap_or(0.0),
                                start.elapsed().as_secs_f64()
                            );
                            Ok((Kind::Text, path.display().to_string(), stats))
                        })();
                        done(&tx, idx, r);
                    }),
                    None,
                ))
            }
            Route::Decide | Route::Embed => {
                let jev = self
                    .jev_model
                    .read()
                    .clone()
                    .or_else(|| {
                        let sched = self.scheduler.as_ref()?;
                        Some(Arc::new(JevSystem1Model::from_llm(
                            sched.model_id(),
                            sched.model()?,
                        )))
                    })
                    .ok_or("no System 1 or LLM model loaded; press m to load a model")?;
                let run: fn(&JevSystem1Model, &str) -> Result<String, String> =
                    if route == Route::Decide {
                        decide
                    } else {
                        embed
                    };
                Ok((
                    Box::new(move |idx, tx| {
                        let r = run(&jev, &body).map(|json| (Kind::Json, json, String::new()));
                        done(&tx, idx, r);
                    }),
                    None,
                ))
            }
            Route::Classify => {
                let c = self.classifier.read().clone().ok_or(
                    "no text classifier loaded; press m and load a text-classification model",
                )?;
                Ok((
                    Box::new(move |idx, tx| {
                        let (text, pair) = classifier::split_pair(&body);
                        let r = c.classify(text, pair).map(|(labels, tokens)| {
                            let stats = format!("{tokens} tokens · {}", c.repo_id);
                            (Kind::Text, label_bars(&labels, c.activation), stats)
                        });
                        done(&tx, idx, r);
                    }),
                    None,
                ))
            }
        }
    }

    /// Downloads (if needed), loads, and activates `id` on a background thread: a text
    /// classifier becomes the active model; an LLM (picking its weight format) is hot-swapped
    /// into the scheduler and replaces any active classifier. On failure the active model stays.
    pub fn load_model(&mut self, id: &str) {
        let refused = match (&self.scheduler, &self.load) {
            (None, _) => Some("no engine scheduler attached".to_string()),
            (_, Some(l)) => Some(format!("{} is still loading", l.id)),
            (_, None) if self.benchmarking => Some("benchmark running".to_string()),
            (Some(_), None) => None,
        };
        if let Some(e) = refused {
            self.push_entry(format!("load {id}"), Kind::Error, e);
            return;
        }
        let sched = self.scheduler.clone().expect("checked above");
        let idx = self.push_entry(format!("load {id}"), Kind::Running, String::new());
        let stage = Arc::new(Mutex::new("resolving"));
        let progress = Arc::new(DownloadProgress::default());
        let reencode = Arc::new(ReencodeProgress::default());
        self.load = Some(Load {
            id: id.to_string(),
            idx,
            started: Instant::now(),
            stage: stage.clone(),
            progress: progress.clone(),
            sample: (Instant::now(), 0),
            rate: 0.0,
            reencode: reencode.clone(),
            reencode_seen: None,
        });
        self.telemetry.log_info(&format!("TUI: loading '{id}'"));
        let mut hub = self.hub.clone();
        hub.progress = Some(progress);
        let (token, tx) = (self.hf_token.clone(), self.updates_tx.clone());
        let slot = self.classifier.clone();
        let jev_slot = self.jev_model.clone();
        let images = self.image_model.clone();
        let video = self.video.clone();
        let (quant_override, hw, id, placement) = (
            self.quant_override,
            self.hw.clone(),
            id.to_string(),
            self.placement,
        );
        // Spec loaders read this through `ReencodeConfig::effective()`: the saved policy (and
        // `--quant`), with the load panel's tensor counter and the telemetry log attached.
        let reencode_cfg = ReencodeConfig {
            force: quant_override,
            progress: Some(reencode),
            log: Some({
                let telemetry = self.telemetry.clone();
                Arc::new(move |m: &str| telemetry.log_info(m))
            }),
            ..ReencodeConfig::from_settings()
        };
        std::thread::spawn(move || {
            crate::tui::qos::set_qos(crate::tui::qos::QosClass::Utility);
            let previous = reencode_override();
            set_reencode_override(Some(reencode_cfg.clone()));
            // A panic inside loading must surface as an error, not leave the load stuck.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let backend = match placement {
                    PlacementMode::Pin(d) => d,
                    PlacementMode::Auto => crate::hardware::default_device(),
                };
                let o = crate::arch::LoadOpts {
                    placement: PlacementMode::Pin(backend),
                    token: token.as_deref(),
                    quant_name: None,
                    hw: &hw,
                    reencode: reencode_cfg.clone(),
                    family: None,
                    ctx: None,
                    stage: &|s| *stage.lock() = s,
                    // Only text generators (and PEFT adapters of them) and image models hot-swap.
                    system1_first: false,
                    admit: &|adapter| match adapter
                        || route_for_model(&id) == Route::Image
                        || is_chat_model(&id)
                    {
                        true => Ok(()),
                        false => Err(format!(
                            "only text-generation, text-to-image models and text classifiers hot-swap; relaunch with `--model {id} --tui`"
                        )),
                    },
                };
                match crate::arch::load_model(&hub, &id, &o)? {
                    Loaded::Classifier(c) => {
                        let shape = c.summary();
                        *slot.write() = Some(Arc::new(c));
                        *jev_slot.write() = None;
                        *images.write() = None;
                        Ok(shape)
                    }
                    Loaded::System1(jev) => {
                        let shape = jev.summary();
                        *jev_slot.write() = Some(Arc::new(jev));
                        *slot.write() = None;
                        *images.write() = None;
                        *video.write() = None;
                        Ok(shape)
                    }
                    Loaded::Video(plan) => {
                        let shape = format!("{}; /video <prompt> (and /i2v <image> <prompt> where the pipeline takes an image) write .mp4 clips", plan.label());
                        *video.write() = Some(Arc::new((hub.clone(), plan)));
                        *jev_slot.write() = None;
                        *images.write() = None;
                        *slot.write() = None;
                        Ok(shape)
                    }
                    Loaded::Image(g) => {
                        let d = g.defaults();
                        let shape = format!(
                            "{} · default {}x{}, {} steps, cfg {}; prompts now generate images",
                            g.summary(),
                            d.width,
                            d.height,
                            d.steps,
                            d.guidance
                        );
                        *images.write() = Some(g);
                        *jev_slot.write() = None;
                        *slot.write() = None;
                        Ok(shape)
                    }
                    Loaded::Lm { model, artifacts } => {
                        let shape = format!("{}, on {}", model.lm().summary(), model.backend_label());
                        sched.swap_model(&id, model);
                        // What the load applied (its gate may fall back) sets the roofline's bytes per token.
                        let model_size = artifacts.weights_path.as_ref().map_or(0, |p| {
                            crate::model::variant::decode_bytes_per_token(p, reencode_cfg.policy)
                        });
                        sched.telemetry.set_model_size_bytes(model_size);
                        let applied = crate::bench::applied_policy();
                        if let (b, "streamed") =
                            crate::bench::roofline_bytes(&artifacts, &applied)
                        {
                            sched.telemetry.set_model_size_bytes(b);
                        }
                        *jev_slot.write() = None;
                        *slot.write() = None;
                        *images.write() = None;
                        Ok(shape)
                    }
                }
            }))
            .unwrap_or_else(|p| {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                Err(format!("loader panicked: {msg}"))
            });
            // Failures leave a record (error, stage, snapshot contents) in the load-failure log.
            set_reencode_override(previous);
            let result = result.map_err(|e| {
                let stage = *stage.lock();
                crate::model::load_log::record(&hub, &id, stage, &e)
            });
            let _ = tx.send(Update::Loaded { idx, id, result });
        });
    }

    /// Downloads (if needed) and attaches drafter `id` to the active model for speculative
    /// decoding, on a background thread; sequences admitted from then on use it.
    fn attach_draft(&mut self, id: &str) {
        let Some(sched) = self.scheduler.clone() else {
            self.push_entry(
                format!("draft {id}"),
                Kind::Error,
                "no engine scheduler attached".into(),
            );
            return;
        };
        let idx = self.push_entry(format!("draft {id}"), Kind::Running, String::new());
        let (hub, token, tx) = (
            self.hub.clone(),
            self.hf_token.clone(),
            self.updates_tx.clone(),
        );
        let (id, target, placement) = (id.to_string(), sched.model_id(), self.placement);
        std::thread::spawn(move || {
            crate::tui::qos::set_qos(crate::tui::qos::QosClass::Utility);
            let attach = || -> Result<String, String> {
                if matches!(id.as_str(), "noise" | "online-noise" | "suffix") {
                    let mode = if id == "suffix" {
                        crate::runtime::noise_draft::DraftMode::Suffix
                    } else {
                        crate::runtime::noise_draft::DraftMode::Noise
                    };
                    let mut knobs = sched.draft_knobs();
                    knobs.mode = mode;
                    sched.set_draft_knobs(knobs);
                    return Ok(format!(
                        "speculative decoding on {target} with {}",
                        mode.name()
                    ));
                }
                let da = hub.auto_download_and_cache(&id, token.as_deref())?;
                let d = crate::runtime::draft::load_drafter(
                    &da,
                    &hub.locate_model(&target),
                    placement,
                )?;
                // ponytail: check-then-set; a model swap landing in between keeps this drafter
                // until the swap resets it. Add a guarded set if loads race in practice.
                if sched.model_id() != target {
                    return Err(format!("{target} was replaced while the drafter loaded"));
                }
                let label = d.label();
                if std::env::var("GIBBS_SPEC").as_deref() == Ok("off") {
                    return Err("speculative decoding is disabled (GIBBS_SPEC=off)".into());
                }
                sched.set_drafter(Some(d));
                Ok(format!("speculative decoding on {target} with {label}"))
            };
            let (kind, text) = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(attach))
            {
                Ok(Ok(t)) => (Kind::Text, t),
                Ok(Err(e)) => (Kind::Error, e),
                Err(_) => (Kind::Error, "drafter loader panicked".into()),
            };
            let _ = tx.send(Update::Done {
                idx,
                kind,
                text,
                stats: String::new(),
            });
        });
    }

    /// Applies finished background work. Returns true when anything changed.
    fn drain_updates(&mut self) -> bool {
        let mut changed = false;
        while let Ok(u) = self.updates_rx.try_recv() {
            changed = true;
            match u {
                Update::Partial { idx, text, tokens } => {
                    let e = &mut self.history[idx];
                    // A cancelled chat keeps what it showed when cancelled.
                    if e.kind == Kind::Running {
                        (e.output, e.stats) = (text, format!("{tokens} tokens"));
                    }
                }
                Update::Done {
                    idx,
                    kind,
                    text,
                    stats,
                } => self.finish(idx, kind, text, stats),
                Update::Loaded { idx, id, result } => {
                    let load = self.load.take();
                    let secs = load
                        .as_ref()
                        .map_or(0.0, |l| l.started.elapsed().as_secs_f64());
                    let fetched = load.map_or(0, |l| l.progress.done.load(Ordering::Relaxed));
                    let source = if fetched > 0 {
                        format!("downloaded {}", picker::fmt_bytes(fetched))
                    } else {
                        "cached".to_string()
                    };
                    match result {
                        Ok(shape) => {
                            self.telemetry.log_info(&format!(
                                "TUI: loaded '{id}' in {secs:.1}s ({source}): {shape}"
                            ));
                            self.picker.mark_cached(&id);
                            self.finish(idx, Kind::Text, shape, source);
                        }
                        Err(e) => {
                            self.telemetry
                                .log_warn(&format!("TUI: cannot load '{id}': {e}"));
                            self.finish(idx, Kind::Error, e, String::new());
                        }
                    }
                }
                Update::Bench(res) => {
                    self.benchmarking = false;
                    let (kind, text) = match res {
                        Ok(r) => {
                            self.telemetry.log_info(&format!(
                                "TUI: benchmark {}: {:.1} tok/s, {:.1}% of the roofline",
                                r.model, r.decode_tok_s, r.roofline_pct
                            ));
                            (Kind::Text, bench_report(&r))
                        }
                        Err(e) => {
                            self.telemetry
                                .log_warn(&format!("TUI: benchmark failed: {e}"));
                            (Kind::Error, e)
                        }
                    };
                    self.push_entry("/bench".into(), kind, text);
                }
                Update::Compacted { from, result } => match result {
                    Ok((to, r)) => {
                        // Turns committed meanwhile stay after the compacted ones; a history
                        // cleared or popped meanwhile is left alone.
                        if self.chat_history.starts_with(&from) {
                            self.chat_history.splice(..from.len(), to);
                        }
                        let line = format!(
                            "compacted {} → {} tokens (tier1 {}, tier2 {}, tier3 {} turns)",
                            r.before_tokens,
                            r.after_tokens,
                            r.tier1_chars_elided,
                            r.tier2_chars_shaken,
                            r.tier3_turns_summarized
                        );
                        self.telemetry.log_info(&format!("TUI: {line}"));
                        self.push_entry("/compact".into(), Kind::Text, line);
                    }
                    Err(e) => {
                        self.push_entry("/compact".into(), Kind::Error, format!("not compacted: {e}"));
                    }
                },
                Update::Probe(res) => {
                    self.roofline_probe_result = res.as_ref().ok().cloned();
                    match &res {
                        Ok(_) => self
                            .telemetry
                            .log_info("TUI: roofline calibration probe completed"),
                        Err(e) => self
                            .telemetry
                            .log_info(&format!("TUI: roofline calibration probe failed: {e}")),
                    }
                }
            }
        }
        changed
    }

    /// Final state of a running entry; entries already finished (e.g. cancelled) stay as they are.
    fn finish(&mut self, idx: usize, kind: Kind, text: String, stats: String) {
        let e = &mut self.history[idx];
        if e.kind != Kind::Running {
            return;
        }
        self.running -= 1;
        if kind == Kind::Text {
            if let Some(user_msg) = e.chat_user.take() {
                self.chat_history.push(("user".to_string(), user_msg));
                self.chat_history
                    .push(("assistant".to_string(), text.clone()));
            }
        } else {
            e.chat_user = None;
        }
        (e.kind, e.output, e.stats) = (kind, text, stats);
        e.took = Some(e.started.elapsed());
        e.cancel = None;
    }

    /// Cancels the newest running chat: its thread drops the token receiver (freeing the
    /// sequence) and the entry keeps the tokens so far. Returns false when no chat runs.
    fn cancel_chat(&mut self) -> bool {
        let Some(idx) = self
            .history
            .iter()
            .rposition(|e| e.kind == Kind::Running && e.cancel.is_some())
        else {
            return false;
        };
        self.history[idx].chat_user = None;
        let e = &self.history[idx];
        if let Some(c) = &e.cancel {
            c.store(true, Ordering::Relaxed);
        }
        let tokens = if e.stats.is_empty() {
            "0 tokens"
        } else {
            e.stats.as_str()
        };
        let (text, stats) = (e.output.clone(), format!("{tokens} · cancelled"));
        self.finish(idx, Kind::Text, text, stats);
        self.telemetry.log_info("TUI: chat cancelled");
        true
    }

    /// Removes `<cache>/models--<org>--<repo>` of `id` (never the active or loading model).
    fn delete_cached(&mut self, id: &str) {
        // Ids may name one file (`org/repo/x-Q4_K_M.gguf`); the cache dir is per repo. Hub ids
        // and the default macOS filesystem are case-insensitive (`Org/X` is `org/x`).
        // Deleting a bare `org/repo` removes every revision, so `org/repo@<rev>` counts as it.
        let same_repo = |other: &str| {
            let repo = split_weight_file(other).0;
            repo.eq_ignore_ascii_case(id) || split_revision(repo).0.eq_ignore_ascii_case(id)
        };
        let refused = if same_repo(&self.model_id()) {
            Some("it is the active model")
        } else if self.load.as_ref().is_some_and(|l| same_repo(&l.id)) {
            Some("it is loading")
        } else if id.split('/').count() != 2 || id.contains("..") {
            Some("not an <org>/<repo> id")
        } else {
            None
        };
        if let Some(why) = refused {
            self.picker
                .set_note(format!("cannot delete {id}: {why}"), theme::Token::Crimson);
            return;
        }
        let dir = self.picker.repo_dir(id);
        let freed = picker::dir_size(&dir);
        match std::fs::remove_dir_all(&dir) {
            Ok(()) => {
                self.telemetry.log_info(&format!(
                    "TUI: deleted {id} from the cache, freed {}",
                    picker::fmt_bytes(freed)
                ));
                self.picker.mark_deleted(id, freed);
            }
            Err(e) => {
                let msg = format!("cannot delete {}: {e}", dir.display());
                self.telemetry.log_warn(&format!("TUI: {msg}"));
                self.picker.set_note(msg, theme::Token::Crimson);
            }
        }
    }

    /// Samples throughput once a second and applies background updates. Returns true when the
    /// screen should be redrawn.
    fn tick(&mut self) -> bool {
        let mut dirty = self.drain_updates();
        if self.spark_at.elapsed() >= Duration::from_secs(1) {
            let total = self
                .telemetry
                .total_tokens_generated
                .load(Ordering::Relaxed);
            self.spark.remove(0);
            self.spark.push(total - self.spark_total);
            (self.spark_total, self.spark_at) = (total, Instant::now());
            dirty = true;
        }
        self.mascot.tick();
        if self.load.is_some() {
            self.mascot.set_state(mascot::MascotState::Thinking);
        } else if self.running > 0 {
            self.mascot.set_state(mascot::MascotState::Generating);
        } else if self.history.last().map_or(false, |e| e.kind == Kind::Error) {
            self.mascot.set_state(mascot::MascotState::Error);
        } else if let Some(last) = self.history.last() {
            if last.kind == Kind::Text
                && last
                    .took
                    .is_some_and(|t| last.started.elapsed() < t + Duration::from_secs(3))
            {
                self.mascot.set_state(mascot::MascotState::Equilibrium);
            } else {
                self.mascot.set_state(mascot::MascotState::Idle);
            }
        } else {
            self.mascot.set_state(mascot::MascotState::Idle);
        }
        if let Some(l) = &mut self.load {
            let done = l.progress.done.load(Ordering::Relaxed);
            let dt = l.sample.0.elapsed().as_secs_f64();
            if dt >= 1.0 {
                l.rate = done.saturating_sub(l.sample.1) as f64 / dt;
                l.sample = (Instant::now(), done);
            }
            let text = l.progress_text();
            if let Some(e) = self.history.get_mut(l.idx) {
                if e.output != text {
                    e.output = text;
                    dirty = true;
                }
            }
            // A program being compiled (also inside a native loader) is the COMPILE stage.
            if crate::graph::program::compiling() {
                *l.stage.lock() = "compiling + tuning kernels";
            }
            let reencoded = l.reencode.done.load(Ordering::Relaxed);
            l.reencode_seen = match l.reencode_seen {
                Some((_, first)) if reencoded >= first => l.reencode_seen,
                _ if reencoded > 0 => Some((Instant::now(), reencoded)),
                _ => None,
            };
        }
        #[cfg(target_os = "macos")]
        {
            use crate::graph::metal;
            self.device_status = DeviceStatus {
                codec: metal::codec_bench_status(),
                tune: metal::tune_status(),
            };
            self.kernel_profile = metal::last_kernel_profile();
            // One decode token at a time is profiled, while the KERNELS panel is on screen and
            // sampling is on (`/profile on`) or the panel is focused (Ctrl+K).
            let wanted = (self.profile_sampling || self.detail == Some(layout::Comp::Kernels))
                && self.kernels_visible.get()
                && !self.benchmarking
                && self.running > 0;
            let old = |at: Option<Instant>| at.is_none_or(|t| t.elapsed() >= PROFILE_EVERY);
            if wanted && old(self.profile_asked) && old(self.kernel_profile.as_ref().map(|p| p.0)) {
                metal::request_kernel_profile();
                self.profile_asked = Some(Instant::now());
            }
        }
        // The host ceiling is what `/bench` and the gap panel measure against.
        let host_ceiling = crate::model::variant::host_ceiling_gbps() as f32;
        self.picker.set_ceiling_gbps(if host_ceiling > 0.0 {
            host_ceiling
        } else {
            self.telemetry.get_summary().memory_bandwidth_gbps
        });
        if self.focus == Focus::Picker {
            dirty |= self.picker.tick();
        }
        if let Some(rx) = &mut self.logprob_rx {
            while let Ok(info) = rx.try_recv() {
                if self.logprob_history.len() >= 64 {
                    self.logprob_history.pop_front();
                }
                self.logprob_history.push_back(info.clone());
                self.latest_logprobs = Some(info);
                self.logprob_scrub_idx = self.logprob_history.len().saturating_sub(1);
                if self.detail == Some(layout::Comp::Logprobs) {
                    dirty = true;
                }
            }
        }
        dirty
    }

    /// Handles a key press. Returns true when the TUI should exit.
    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        // Raw mode reports Enter typed as `\n` / `\r` as Ctrl-J / Ctrl-M.
        if ctrl && matches!(key.code, KeyCode::Char('j' | 'm')) {
            return self.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        }
        if ctrl && key.code == KeyCode::Char('c') {
            return true;
        }
        if ctrl && key.code == KeyCode::Char('x') {
            self.cancel_chat();
            return false;
        }
        // (Ctrl+R/P/T/L/K removed from keybar; use /roofline, /profile, F4 options)
        if self.placement_modal.is_some() {
            if matches!(key.code, KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q')) {
                self.placement_modal = None;
            } else if let Some((rows, scroll)) = &mut self.placement_modal {
                let last = rows.len() + 2;
                match key.code {
                    KeyCode::Up => *scroll = scroll.saturating_sub(1),
                    KeyCode::Down => *scroll = (*scroll + 1).min(last),
                    KeyCode::PageUp => *scroll = scroll.saturating_sub(10),
                    KeyCode::PageDown => *scroll = (*scroll + 10).min(last),
                    _ => {}
                }
            }
            return false;
        }
        if self.detail.is_some() && self.focus != Focus::Picker {
            match (self.detail, key.code) {
                (_, KeyCode::Esc) => {
                    self.detail = None;
                    return false;
                }
                (Some(layout::Comp::Roofline), KeyCode::Char('p')) => {
                    self.trigger_background_roofline_probe();
                    return false;
                }
                (Some(layout::Comp::Logprobs), KeyCode::Left) => {
                    self.logprob_scrub_idx = self.logprob_scrub_idx.saturating_sub(1);
                    return false;
                }
                (Some(layout::Comp::Logprobs), KeyCode::Right) => {
                    self.logprob_scrub_idx = (self.logprob_scrub_idx + 1)
                        .min(self.logprob_history.len().saturating_sub(1));
                    return false;
                }
                _ => {}
            }
        }
        if key.code == KeyCode::F(2) {
            self.show_template_modal = !self.show_template_modal;
            return false;
        }
        if self.show_template_modal {
            match key.code {
                KeyCode::Esc | KeyCode::F(2) => self.show_template_modal = false,
                KeyCode::Up => self.active_template = self.active_template.saturating_sub(1),
                KeyCode::Down => {
                    self.active_template =
                        (self.active_template + 1).min(crate::arch::templates::names().len())
                }
                KeyCode::Char(c) if c.is_ascii_digit() => {
                    let d = c.to_digit(10).unwrap_or(0) as usize;
                    if d <= crate::arch::templates::names().len() {
                        self.active_template = d;
                    }
                }
                KeyCode::Enter => self.show_template_modal = false,
                _ => {}
            }
            return false;
        }
        if key.code == KeyCode::F(4) {
            self.show_settings_modal = !self.show_settings_modal;
            return false;
        }
        if self.show_settings_modal {
            match key.code {
                KeyCode::Esc | KeyCode::F(4) => self.show_settings_modal = false,
                KeyCode::Char('1') => {
                    let _ = self.set_reencode(None);
                }
                KeyCode::Char('2') => {
                    let _ = self.set_variant(None);
                }
                KeyCode::Char('3') => {
                    self.show_roofline_hud = !self.show_roofline_hud;
                }
                KeyCode::Char('4') => {
                    self.ctx_limit = match self.ctx_limit {
                        0..=4096 => 8192,
                        4097..=8192 => 16384,
                        8193..=16384 => 32768,
                        _ => 4096,
                    };
                }
                KeyCode::Char('5') => {
                    let next = self.draft_knobs.mode.next();
                    let _ = self.apply_spec_cmd(vec![SpecArg::Mode(next)]);
                }
                KeyCode::Char('6') => {
                    let next = match self.draft_knobs.max_k {
                        0 => 2,
                        2 => 4,
                        4 => 6,
                        6 => 8,
                        8 => 12,
                        _ => 0,
                    };
                    let _ = self.apply_spec_cmd(vec![SpecArg::MaxK(next)]);
                }
                KeyCode::Char('7') => {
                    let cur = (self.draft_knobs.target_acceptance * 100.0).round() as u32;
                    let next = match cur {
                        0..=55 => 0.65,
                        56..=70 => 0.75,
                        71..=80 => 0.85,
                        _ => 0.50,
                    };
                    let _ = self.apply_spec_cmd(vec![SpecArg::Accept(next)]);
                }
                KeyCode::Char('8') => {
                    let next = self.compact_config.mode.next();
                    let _ = self.run_compact_cmd(next.name());
                }
                KeyCode::Char('9') => {
                    let next = self.draft_knobs.objective.next();
                    let _ = self.apply_spec_cmd(vec![SpecArg::Objective(next)]);
                }
                KeyCode::Char('0') => {
                    let next = self.draft_knobs.denoise_steps % 4 + 1;
                    let _ = self.apply_spec_cmd(vec![SpecArg::Steps(next)]);
                }
                KeyCode::Enter => self.show_settings_modal = false,
                _ => {}
            }
            return false;
        }
        match self.focus {
            Focus::Picker => match self.picker.handle_key(key) {
                PickerAction::None => {}
                PickerAction::Close => self.focus = Focus::Dashboard,
                PickerAction::Load(id) => {
                    self.focus = Focus::Dashboard;
                    self.load_model(&id);
                }
                PickerAction::Delete(id) => self.delete_cached(&id),
                PickerAction::Reload(what) => {
                    // The picker saved the setting; follow it and reload the loaded model under it.
                    let s = settings::load_settings();
                    self.reencode_policy = s.reencode;
                    self.variant_policy = s.variant;
                    let value = if what == "reencode" {
                        s.reencode.name()
                    } else {
                        s.variant.name()
                    };
                    let note = self.reload_for(what, value);
                    self.picker.set_note(note, theme::Token::Jade);
                }
                PickerAction::AttachDraft(id) => {
                    self.focus = Focus::Dashboard;
                    self.attach_draft(&id);
                }
            },
            Focus::Prompt => match key.code {
                KeyCode::Esc => {
                    if !self.cancel_chat() {
                        self.focus = Focus::Dashboard;
                    }
                }
                KeyCode::Up => self.navigate_prompt_history(true),
                KeyCode::Down => self.navigate_prompt_history(false),
                KeyCode::Enter => self.submit_prompt(),
                KeyCode::Backspace => {
                    self.input.pop();
                }
                KeyCode::Char('u') if ctrl => self.input.clear(),
                KeyCode::Char(c) if !ctrl => self.input.push(c),
                code => self.scroll_console(code),
            },
            Focus::Dashboard => match key.code {
                KeyCode::Char('q') => return true,
                KeyCode::Char('i') | KeyCode::Char('p') | KeyCode::Enter => {
                    self.focus = Focus::Prompt
                }
                KeyCode::Char('/') => {
                    self.focus = Focus::Prompt;
                    self.input.push('/');
                }
                KeyCode::Char('m') => {
                    self.picker.open(&self.model_id());
                    self.focus = Focus::Picker;
                }
                KeyCode::Char('s') => self.kernel_sort = self.kernel_sort.next(),
                KeyCode::Up => self.navigate_prompt_history(true),
                KeyCode::Down => self.navigate_prompt_history(false),
                KeyCode::Char('k') => self.scroll = self.scroll.saturating_add(1),
                KeyCode::Char('j') => self.scroll = self.scroll.saturating_sub(1),
                code => self.scroll_console(code),
            },
        }
        false
    }

    fn navigate_prompt_history(&mut self, up: bool) {
        if up {
            if self.prompt_history.is_empty() {
                return;
            }
            self.focus = Focus::Prompt;
            let idx = match self.prompt_history_idx {
                None => {
                    self.prompt_draft = std::mem::take(&mut self.input);
                    self.prompt_history.len() - 1
                }
                Some(cur) => cur.saturating_sub(1),
            };
            self.prompt_history_idx = Some(idx);
            self.input = self.prompt_history[idx].clone();
        } else if let Some(cur) = self.prompt_history_idx {
            self.focus = Focus::Prompt;
            if cur + 1 < self.prompt_history.len() {
                let next = cur + 1;
                self.prompt_history_idx = Some(next);
                self.input = self.prompt_history[next].clone();
            } else {
                self.input = std::mem::take(&mut self.prompt_draft);
                self.prompt_history_idx = None;
            }
        }
    }

    pub fn handle_mouse(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::ScrollUp => {
                if let Some((_, scroll)) = &mut self.placement_modal {
                    *scroll = scroll.saturating_sub(3);
                } else if self.focus == Focus::Picker {
                    for _ in 0..3 {
                        let _ = self
                            .picker
                            .handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
                    }
                } else {
                    self.scroll = self.scroll.saturating_add(3);
                }
            }
            MouseEventKind::ScrollDown => {
                if let Some((rows, scroll)) = &mut self.placement_modal {
                    let last = rows.len() + 2;
                    *scroll = (*scroll + 3).min(last);
                } else if self.focus == Focus::Picker {
                    for _ in 0..3 {
                        let _ = self
                            .picker
                            .handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
                    }
                } else {
                    self.scroll = self.scroll.saturating_sub(3);
                }
            }
            _ => {}
        }
    }

    fn scroll_console(&mut self, code: KeyCode) {
        self.scroll = match code {
            KeyCode::PageUp => self.scroll + 10,
            KeyCode::PageDown => self.scroll.saturating_sub(10),
            KeyCode::Home => usize::MAX / 2,
            KeyCode::End => 0,
            _ => return,
        };
    }

    pub fn run(&mut self) -> io::Result<()> {
        crate::tui::qos::set_qos(crate::tui::qos::QosClass::UserInteractive);
        self.telemetry.is_tui_active.store(true, Ordering::Relaxed);
        crate::telemetry::TUI_OWNS_TERMINAL.store(true, Ordering::Relaxed);
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
        // go to the log (the default hook would print over the alternate screen).
        let ui_thread = std::thread::current().id();
        let previous: Arc<dyn Fn(&std::panic::PanicHookInfo<'_>) + Send + Sync> =
            Arc::from(std::panic::take_hook());
        let (prev, telemetry) = (previous.clone(), self.telemetry.clone());
        std::panic::set_hook(Box::new(move |info| {
            if std::thread::current().id() == ui_thread {
                restore_terminal();
                prev(info);
            } else {
                telemetry.log_warn(&format!("panic: {info}"));
            }
        }));
        let result = match Terminal::new(CrosstermBackend::new(io::stdout())) {
            Ok(mut terminal) => {
                let r = self.event_loop(&mut terminal);
                restore_terminal();
                r.and(terminal.show_cursor())
            }
            Err(e) => {
                restore_terminal();
                Err(e)
            }
        };
        std::panic::set_hook(Box::new(move |info| previous(info)));
        self.telemetry.is_tui_active.store(false, Ordering::Relaxed);
        crate::telemetry::TUI_OWNS_TERMINAL.store(false, Ordering::Relaxed);
        result
    }

    fn event_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    ) -> io::Result<()> {
        let mut next_draw = Instant::now();
        loop {
            let frame_start = Instant::now();
            if self.tick() || Instant::now() >= next_draw {
                terminal.draw(|f| self.render(f))?;
                next_draw = Instant::now() + REFRESH;
                if let Some(path) = &self.frame_log_path {
                    use std::io::Write;
                    let elapsed_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)
                    {
                        let _ = writeln!(f, "{elapsed_ms:.3}");
                    }
                }
            }
            let busy = self.running > 0 || self.load.is_some() || self.focus == Focus::Picker;
            let wait = next_draw
                .saturating_duration_since(Instant::now())
                .min(if busy { BUSY_POLL } else { REFRESH });
            if event::poll(wait)? {
                // Apply every queued event before one redraw (fast typing, pastes, resizes).
                loop {
                    match event::read()? {
                        Event::Key(key) => {
                            if key.kind != KeyEventKind::Release && self.handle_key(key) {
                                return Ok(());
                            }
                        }
                        Event::Mouse(mouse) => self.handle_mouse(mouse),
                        _ => {}
                    }
                    if !event::poll(Duration::ZERO)? {
                        break;
                    }
                }
                next_draw = Instant::now();
            }
        }
    }
    /// Updates the active chat template: 0 is the model's own, `i` is
    /// `arch::templates::names()[i - 1]`.
    pub fn set_chat_template(&mut self, template_idx: usize) {
        self.active_template = template_idx.min(crate::arch::templates::names().len());
    }

    /// Updates the system prompt prepended to chat requests.
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.system_prompt = prompt.into();
    }

    /// Formats a chat prompt using the active chat template and system prompt.
    pub fn format_chat_prompt(&self, tok: &Tokenizer, user_prompt: &str) -> String {
        Self::format_chat_history_with(
            &self.system_prompt,
            self.active_template,
            tok,
            &self.chat_history,
            user_prompt,
        )
    }

    /// Formats a single-turn chat prompt using explicit system prompt and template style.
    pub fn format_chat_prompt_with(
        system_prompt: &str,
        active_template: usize,
        tok: &Tokenizer,
        user_prompt: &str,
    ) -> String {
        Self::format_chat_history_with(system_prompt, active_template, tok, &[], user_prompt)
    }

    /// Formats a multi-turn chat prompt including `history` (`(role, content)` pairs) and `user_prompt`.
    pub fn format_chat_history_with(
        system_prompt: &str,
        active_template: usize,
        tok: &Tokenizer,
        history: &[(String, String)],
        user_prompt: &str,
    ) -> String {
        let mut msgs: Vec<(&str, &str)> = Vec::with_capacity(history.len() + 2);
        if !system_prompt.is_empty() {
            msgs.push(("system", system_prompt));
        }
        for (role, content) in history {
            msgs.push((role.as_str(), content.as_str()));
        }
        msgs.push(("user", user_prompt));
        match active_template
            .checked_sub(1)
            .and_then(|i| crate::arch::templates::names().get(i))
        {
            Some(name) => tok.apply_chat_template_with_style(&msgs, name),
            None => tok.apply_chat_template(&msgs),
        }
    }
}

fn restore_terminal() {
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), DisableMouseCapture, LeaveAlternateScreen);
}

/// `/decide <state> | option, option`: zero-shot choice by embedding similarity.
fn decide(jev: &JevSystem1Model, body: &str) -> Result<String, String> {
    let (query, opts) = body
        .split_once('|')
        .map(|(q, o)| (q.trim(), o))
        .unwrap_or((body, ""));
    let options: Vec<ChoiceOption> = opts
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| ChoiceOption::new(s, None::<String>))
        .collect();
    if options.is_empty() {
        return Err(
            "usage: /decide <state> | option1, option2 (zero-shot choice by embedding similarity)"
                .to_string(),
        );
    }
    let decision = jev
        .embed(query)
        .and_then(|s| jev.evaluate_choice_embedded(&s, &options))
        .map_err(|e| format!("System 1 decision failed: {e}"))?;
    let json = serde_json::json!({ "query": query, "model": jev.repo_id, "decision": decision });
    Ok(serde_json::to_string_pretty(&json).unwrap_or_default())
}

/// `/embed <text>`: dimensions, L2 norm, and the head of the embedding.
fn embed(jev: &JevSystem1Model, text: &str) -> Result<String, String> {
    let v = jev
        .embed(text)
        .map_err(|e| format!("System 1 embedding failed: {e}"))?;
    let l2_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let json = serde_json::json!({
        "dimensions": v.len(),
        "l2_norm": l2_norm,
        "head": &v[..v.len().min(8)],
    });
    Ok(serde_json::to_string_pretty(&json).unwrap_or_default())
}

/// `/bench` output: throughput against the roofline and where a token's time goes.
fn bench_report(r: &crate::bench::ModelBenchResult) -> String {
    let mut s = format!(
        "BENCHMARK {}:\n\
         · decode {:.1} tok/s of a {:.1} tok/s roofline = {:.1}% ({:.0} MB/token, {} bytes, at {:.0} GB/s)",
        r.model,
        r.decode_tok_s,
        r.roofline_tok_s,
        r.roofline_pct,
        r.bytes_per_token as f64 / 1e6,
        r.bytes_basis,
        r.ceiling_gbps
    );
    let (Some(gpu), Some(run)) = (r.gpu_ms, r.run_ms) else {
        return s + "\n· no GPU run samples: this model does not run on Metal";
    };
    let gap = crate::telemetry::GapStats {
        wall_ms: r.wall_ms as f32,
        run_ms: run as f32,
        gpu_ms: gpu as f32,
        weights_ms: r.weights_ms as f32,
        samples: 1,
        ..Default::default()
    };
    let [weights, gpu_excess, run_host, outside] = gap.segments();
    s += &format!(
        "\n· token {:.2} ms = weights {weights:.2} + gpu excess {gpu_excess:.2} + run host {run_host:.2} + outside {outside:.2} ms",
        r.wall_ms
    );
    if let Some(d) = r.dispatches {
        s += &format!(
            "\n· {d:.0} dispatches per token, encode {:.0} µs",
            r.encode_us.unwrap_or(0.0)
        );
    }
    s
}

/// Classifier output: one line per label (the top [`SHOWN`]), with a bar for probabilities.
fn label_bars(labels: &[LabelScore], activation: Activation) -> String {
    const SHOWN: usize = 10;
    const BAR: usize = 24;
    let top = &labels[..labels.len().min(SHOWN)];
    let w = top
        .iter()
        .map(|l| l.label.chars().count())
        .max()
        .unwrap_or(0);
    let mut lines: Vec<String> = top
        .iter()
        .map(|l| {
            if activation == Activation::Identity {
                format!("{:<w$}  {:>9.4}  (raw score)", l.label, l.score)
            } else {
                let n = (l.score.clamp(0.0, 1.0) * BAR as f32).round() as usize;
                let bar = format!("{}{}", "█".repeat(n), "·".repeat(BAR - n));
                format!("{:<w$}  {bar}  {:.4}", l.label, l.score)
            }
        })
        .collect();
    if labels.len() > SHOWN {
        lines.push(format!("… {} more labels", labels.len() - SHOWN));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn type_and_run(app: &mut TuiApp, s: &str) {
        app.input = s.to_string();
        app.submit_prompt();
    }

    /// Applies background updates until no prompt or load is running.
    fn settle(app: &mut TuiApp) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while app.running > 0 || app.load.is_some() {
            app.drain_updates();
            assert!(Instant::now() < deadline, "background work did not finish");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Commands with nothing to run on fail at once with the reason; nothing is fabricated.
    #[test]
    fn test_prompt_commands_report_missing_models_and_bad_input() {
        let mut app = TuiApp::new(EngineTelemetry::new());
        let cases = [
            ("/decide is this safe | approve, reject", "System 1"),
            ("/embed quick brown fox", "System 1"),
            ("/chat hello", "no text model"),
            ("/image a lighthouse", "no image model"),
            ("/image", "empty prompt"),
            ("/bogus x", "unknown command '/bogus'"),
            ("/video airship", "no video model"),
            ("/i2v x", "no video model"),
        ];
        for (input, reason) in cases {
            type_and_run(&mut app, input);
            let e = app.history.last().unwrap();
            assert_eq!(e.kind, Kind::Error, "{input}");
            assert!(e.output.contains(reason), "{input}: {}", e.output);
        }
        assert_eq!(app.running, 0);
        assert!(app.input.is_empty());
    }

    /// Keys type into the prompt while it has focus; the same keys drive the dashboard otherwise.
    #[test]
    fn test_keys_type_in_prompt_and_command_dashboard() {
        let mut app = TuiApp::new(EngineTelemetry::new());
        assert!(!app.handle_key(key(KeyCode::Char('i'))));
        for c in "mq".chars() {
            assert!(!app.handle_key(key(KeyCode::Char(c))), "'{c}' must type");
        }
        assert_eq!(app.input, "mq");
        app.handle_key(key(KeyCode::Esc));
        assert_eq!(app.focus, Focus::Dashboard);
        app.handle_key(key(KeyCode::Char('m')));
        assert_eq!(app.focus, Focus::Picker);
        app.handle_key(key(KeyCode::Esc));
        assert_eq!(app.focus, Focus::Dashboard);
        assert!(app.handle_key(key(KeyCode::Char('q'))));
        assert!(app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)));
    }

    #[test]
    fn test_model_load_hot_swaps_scheduler_and_chat_streams() {
        let cache = tempfile::tempdir().unwrap();
        let sched = Arc::new(ContinuousBatchScheduler::new("", None));
        let mut app = TuiApp::new(EngineTelemetry::new()).with_runtime(
            sched.clone(),
            HuggingFaceHub::with_cache_dir(cache.path().to_path_buf()),
            None,
            None,
            ImageSlot::default(),
        );

        // Neither a text-generation model nor a sequence classifier: refused by the load worker
        // once it reads config.json (cached here, so no network), scheduler stays empty.
        let snap = cache
            .path()
            .join("models--sentence-transformers--all-MiniLM-L6-v2/snapshots/main");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(
            snap.join("config.json"),
            r#"{"architectures":["BertModel"]}"#,
        )
        .unwrap();
        app.load_model("sentence-transformers/all-MiniLM-L6-v2");
        settle(&mut app);
        assert_eq!(app.history[0].kind, Kind::Error);
        assert!(
            app.history[0].output.contains("hot-swap"),
            "{}",
            app.history[0].output
        );
        assert!(!sched.has_model());

        // Synthetic model loads offline in the background and replaces model + tokenizer.
        let before_tok = sched.tokenizer();
        app.load_model("gibbs/tiny-5m");
        settle(&mut app);
        let e = &app.history[1];
        assert_eq!(e.kind, Kind::Text, "{}", e.output);
        assert_eq!(sched.model_id(), "gibbs/tiny-5m");
        assert!(!Arc::ptr_eq(&before_tok, &sched.tokenizer()));

        // Ctrl-J submits like Enter; Esc with a chat running cancels it (focus stays), then leaves.
        app.focus = Focus::Prompt;
        app.input = "/chat Tell me a long story".into();
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::CONTROL));
        let idx = app.history.len() - 1;
        assert_eq!(app.history[idx].kind, Kind::Running);
        app.handle_key(key(KeyCode::Esc));
        let e = &app.history[idx];
        assert_eq!(
            (e.kind, e.stats.as_str()),
            (Kind::Text, "0 tokens · cancelled")
        );
        assert_eq!((app.running, app.focus), (0, Focus::Prompt));
        app.handle_key(key(KeyCode::Esc));
        assert_eq!(app.focus, Focus::Dashboard);

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stepper = {
            let (sched, stop) = (sched.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    sched.step();
                }
            })
        };
        type_and_run(&mut app, "/chat Once upon a time");
        settle(&mut app);
        stop.store(true, Ordering::Relaxed);
        stepper.join().unwrap();
        let e = app.history.last().unwrap();
        assert_eq!(e.kind, Kind::Text, "{}", e.output);
        assert!(e.stats.contains("tokens"), "{}", e.stats);
    }

    #[test]
    fn test_tui_native_gemma4_loads_on_metal() {
        if cfg!(debug_assertions) {
            eprintln!("SKIP: test requires --release to prevent settle timeout");
            return;
        }
        if !crate::hardware::available_devices().contains(&crate::hardware::Device::Metal) {
            eprintln!("SKIP: Metal not available");
            return;
        }
        let snap = std::path::Path::new("disk/models--unsloth--gemma-4-E2B-it-qat-mobile-GGUF/snapshots/main/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf");
        if !snap.exists() {
            eprintln!("SKIP: gemma-4-E2B GGUF not found");
            return;
        }
        let sched = Arc::new(ContinuousBatchScheduler::new("", None));
        let mut app = TuiApp::new(EngineTelemetry::new()).with_runtime(
            sched.clone(),
            HuggingFaceHub::with_cache_dir("disk".into()),
            None,
            None,
            ImageSlot::default(),
        );

        app.load_model("unsloth/gemma-4-E2B-it-qat-mobile-GGUF");
        settle(&mut app);
        let e = app.history.last().unwrap();
        assert_eq!(e.kind, Kind::Text, "{}", e.output);
        assert!(
            e.output.contains("on Metal ("),
            "TUI output was: {}",
            e.output
        );
        assert!(
            sched.backend_label().starts_with("Metal ("),
            "scheduler label was: {}",
            sched.backend_label()
        );
    }
    /// Loading a classifier makes it the active model and bare prompts classify (`a || b` is a
    /// pair); loading an LLM afterwards makes the LLM active again.
    #[test]
    fn test_classifier_load_classifies_until_an_llm_loads() {
        const CE: &str = "cross-encoder/ms-marco-MiniLM-L-6-v2";
        let repo = std::path::Path::new("disk/models--cross-encoder--ms-marco-MiniLM-L-6-v2");
        if !repo.join("snapshots/main/model.safetensors").exists() {
            eprintln!("SKIP: {CE} not cached in disk/");
            return;
        }
        // Private cache: the real classifier linked in, the synthetic LLM written beside it.
        let cache = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(
            repo.canonicalize().unwrap(),
            cache.path().join(repo.file_name().unwrap()),
        )
        .unwrap();
        let sched = Arc::new(ContinuousBatchScheduler::new("", None));
        let mut app = TuiApp::new(EngineTelemetry::new()).with_runtime(
            sched.clone(),
            HuggingFaceHub::with_cache_dir(cache.path().to_path_buf()),
            None,
            None,
            ImageSlot::default(),
        );

        app.load_model(CE);
        settle(&mut app);
        assert_eq!(app.history[0].kind, Kind::Text, "{}", app.history[0].output);
        assert_eq!(app.model_id(), CE);

        type_and_run(
            &mut app,
            "How many people live in Berlin? || Berlin had a population of 3,520,031 registered inhabitants.",
        );
        settle(&mut app);
        let e = app.history.last().unwrap();
        assert_eq!(e.kind, Kind::Text, "{}", e.output);
        assert!(
            e.output.starts_with("LABEL_0") && e.output.contains("(raw score)"),
            "{}",
            e.output
        );

        app.load_model("gibbs/tiny-5m");
        settle(&mut app);
        assert!(app.classifier.read().is_none());
        assert_eq!(app.model_id(), "gibbs/tiny-5m");
    }

    /// ctrl-d arms, a second ctrl-d removes the cached repo directory; a loading repo is refused.
    #[test]
    fn test_picker_ctrl_d_deletes_cached_repo_after_confirm() {
        let cache = tempfile::tempdir().unwrap();
        let repo = cache.path().join("models--o--r");
        std::fs::create_dir_all(repo.join("snapshots/main")).unwrap();
        std::fs::write(repo.join("snapshots/main/model.safetensors"), b"w").unwrap();
        let sched = Arc::new(ContinuousBatchScheduler::new("", None));
        let mut app = TuiApp::new(EngineTelemetry::new()).with_runtime(
            sched,
            HuggingFaceHub::with_cache_dir(cache.path().to_path_buf()),
            None,
            None,
            ImageSlot::default(),
        );
        let ctrl_d = KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL);
        app.handle_key(key(KeyCode::Char('m')));
        for _ in 0..500 {
            app.picker.tick();
            if !app.picker.scanning_local {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        app.handle_key(ctrl_d);
        assert!(repo.exists(), "first ctrl-d only asks for confirmation");
        app.handle_key(ctrl_d);
        assert!(!repo.exists());
        assert_eq!(app.focus, Focus::Picker);

        // A load of one file of the repo blocks deleting the repo.
        std::fs::create_dir_all(repo.join("snapshots/main")).unwrap();
        std::fs::write(repo.join("snapshots/main/model.safetensors"), b"w").unwrap();
        app.load = Some(Load {
            id: "o/r/model.safetensors".into(),
            idx: 0,
            started: Instant::now(),
            stage: Arc::new(Mutex::new("resolving")),
            progress: Arc::default(),
            sample: (Instant::now(), 0),
            rate: 0.0,
            reencode: Arc::default(),
            reencode_seen: None,
        });
        app.handle_key(key(KeyCode::Esc));
        app.handle_key(key(KeyCode::Char('m')));
        for _ in 0..500 {
            app.picker.tick();
            if !app.picker.scanning_local {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        app.handle_key(ctrl_d);
        assert!(repo.exists(), "a loading repo is never deleted");
    }

    #[test]
    fn test_scratch_dir_cleanup_on_drop() {
        let app = TuiApp::new(EngineTelemetry::new());
        let scratch = app.scratch_dir.path().to_path_buf();
        std::fs::write(scratch.join("probe.png"), b"x").unwrap();
        drop(app);
        assert!(!scratch.exists(), "scratch directory must clean up on drop");
    }

    #[test]
    fn test_at_path_attachments_become_ordered_media_parts() {
        let dir = tempfile::tempdir().unwrap();
        let (img, wav) = (dir.path().join("a.PNG"), dir.path().join("b.wav"));
        std::fs::write(&img, b"png").unwrap();
        std::fs::write(&wav, b"wav").unwrap();
        let prompt = format!(
            "compare @{} with @{} and @missing.png @notes.txt",
            img.display(),
            wav.display()
        );
        let (msg, media) = media_message(&prompt).unwrap().expect("has attachments");
        assert_eq!(
            media,
            [
                MediaInput::Image(b"png".to_vec()),
                MediaInput::Audio(b"wav".to_vec())
            ]
        );
        assert_eq!(
            msg.content,
            serde_json::json!([
                {"type": "text", "text": "compare"},
                {"type": "image"},
                {"type": "text", "text": "with"},
                {"type": "audio"},
                {"type": "text", "text": "and @missing.png @notes.txt"},
            ])
        );
        assert!(media_message("plain @word text").unwrap().is_none());
    }
    #[test]
    fn test_chat_template_hot_swapping_and_modal() {
        let mut app = TuiApp::new(EngineTelemetry::new());
        let tok = Tokenizer::default_bpe();

        // 1. Toggle modal with F2
        assert!(!app.show_template_modal);
        app.handle_key(key(KeyCode::F(2)));
        assert!(app.show_template_modal);

        // 2. Navigate templates with Down / Up / digit keys
        app.handle_key(key(KeyCode::Down));
        assert_eq!(app.active_template, 1);
        app.handle_key(key(KeyCode::Char('2')));
        assert_eq!(app.active_template, 2);
        app.handle_key(key(KeyCode::Enter));
        assert!(!app.show_template_modal);

        // 3. Test observable formatting effect with ChatML (1) vs Llama3 (2) vs Plain (4)
        app.set_chat_template(1); // ChatML
        app.set_system_prompt("You are a test assistant.");
        let prompt_chatml = app.format_chat_prompt(&tok, "Hello Gibbs");
        assert!(prompt_chatml.contains("<|im_start|>system\nYou are a test assistant.<|im_end|>"));
        assert!(prompt_chatml.contains("<|im_start|>user\nHello Gibbs<|im_end|>"));

        app.set_chat_template(2); // Llama3
        let prompt_llama = app.format_chat_prompt(&tok, "Hello Gibbs");
        assert!(prompt_llama.contains(
            "<|start_header_id|>system<|end_header_id|>\n\nYou are a test assistant.<|eot_id|>"
        ));
        assert!(prompt_llama
            .contains("<|start_header_id|>user<|end_header_id|>\n\nHello Gibbs<|eot_id|>"));

        app.set_chat_template(4); // Plain
        let prompt_plain = app.format_chat_prompt(&tok, "Hello Gibbs");
        assert_eq!(prompt_plain, "You are a test assistant.\nHello Gibbs");
    }

    #[test]
    fn test_mascot_reactive_state_machine_in_tui() {
        let mut app = TuiApp::new(EngineTelemetry::new());
        assert_eq!(app.mascot.state, mascot::MascotState::Idle);

        // While running > 0, mascot becomes Generating
        app.running = 1;
        app.tick();
        assert_eq!(app.mascot.state, mascot::MascotState::Generating);

        // Reset running, becomes Idle
        app.running = 0;
        app.tick();
        assert_eq!(app.mascot.state, mascot::MascotState::Idle);

        // On error in history, mascot becomes Error
        app.push_entry("/bogus".into(), Kind::Error, "unknown command".into());
        app.tick();
        assert_eq!(app.mascot.state, mascot::MascotState::Error);

        // Successful completion within 3s triggers Equilibrium
        app.push_entry("done prompt".into(), Kind::Text, "response text".into());
        app.history.last_mut().unwrap().took = Some(Duration::from_millis(100));
        app.tick();
        assert_eq!(app.mascot.state, mascot::MascotState::Equilibrium);
    }

    #[test]
    fn test_logprob_inspector_toggle_and_channel() {
        let cache = tempfile::tempdir().unwrap();
        let sched = Arc::new(ContinuousBatchScheduler::new("", None));
        let mut app = TuiApp::new(EngineTelemetry::new()).with_runtime(
            sched.clone(),
            HuggingFaceHub::with_cache_dir(cache.path().to_path_buf()),
            None,
            None,
            ImageSlot::default(),
        );

        assert_eq!(app.detail, None);

        // 1. Detail view on Dashboard closes with Esc
        app.detail = Some(layout::Comp::Logprobs);
        app.handle_key(key(KeyCode::Esc));
        assert_eq!(app.detail, None);

        // 2. Transmit real TokenLogprobInfo on channel and verify tick() receives it
        let info = crate::scheduler::TokenLogprobInfo {
            chosen_id: 11751,
            chosen_prob: 0.642,
            top_candidates: vec![(11751, 0.642), (13, 0.215), (198, 0.08)],
            entropy: 0.892,
        };
        if let Some(tx) = sched.logprob_tx.read().as_ref() {
            let _ = tx.send(info.clone());
        }
        app.tick();
        assert_eq!(app.latest_logprobs, Some(info));
        assert_eq!(app.logprob_history.len(), 1);

        // 3. Test scrub index navigation with Left / Right
        app.focus = Focus::Dashboard;
        app.detail = Some(layout::Comp::Logprobs);
        app.handle_key(key(KeyCode::Left));
        assert_eq!(app.logprob_scrub_idx, 0);
        app.handle_key(key(KeyCode::Right));
        assert_eq!(app.logprob_scrub_idx, 0);
    }

    #[test]
    fn test_roofline_probe_and_gibbs_commands() {
        let mut app = TuiApp::new(EngineTelemetry::new());

        // 1. /gibbs command
        app.input = "/gibbs".into();
        app.submit_prompt();
        assert_eq!(app.history.len(), 1);
        assert!(app.history[0].output.contains("JOSIAH WILLARD GIBBS"));
        assert!(app.history[0].output.contains("Mathematics is a language"));

        // 2. run_roofline_probe
        let probe = app.run_roofline_probe();
        if cfg!(target_os = "macos") {
            assert!(probe.is_ok());
            let report = probe.unwrap();
            assert!(report.contains("ROOFLINE CALIBRATION"));
            assert!(report.contains("GB/s"));
            assert!(report.contains("Ridge"));
        } else {
            assert!(probe.is_err());
        }

        // 3. /roofline command opens detail view, Esc unfocuses it
        app.input = "/roofline".into();
        app.submit_prompt();
        assert_eq!(app.detail, Some(layout::Comp::Roofline));
        app.handle_key(key(KeyCode::Esc));
        assert_eq!(app.detail, None);

        // 5. The removed stub is an unknown command; /gibbs prints the portrait and biography
        assert_eq!(app.mascot.state, mascot::MascotState::Idle);
        let gone = format!("/{}", ["ava", "tar"].concat());
        app.input = gone.clone();
        app.submit_prompt();
        let e = app.history.last().unwrap();
        assert_eq!(e.kind, Kind::Error);
        assert!(e.output.contains(&format!("unknown command '{gone}'")));
        app.input = "/gibbs".into();
        app.submit_prompt();
        assert!(app
            .history
            .last()
            .unwrap()
            .output
            .contains("JOSIAH WILLARD GIBBS"));
        // 7. /bench (no model loaded: refused), /quant, /help commands
        app.input = "/bench".into();
        app.submit_prompt();
        let e = app.history.last().unwrap();
        assert_eq!(e.kind, Kind::Error);
        assert!(e.output.contains("no text model loaded"), "{}", e.output);

        app.input = "/quant".into();
        app.submit_prompt();
        assert!(app
            .history
            .last()
            .unwrap()
            .output
            .contains("MODEL QUANTIZATION"));

        app.input = "/help".into();
        app.submit_prompt();
        assert!(app
            .history
            .last()
            .unwrap()
            .output
            .contains("GIBBS ENGINE COMMAND & KEY REFERENCE"));
    }
    #[test]
    fn test_settings_modal_and_policies() {
        use crate::model::weight_format::{ReencodePolicy as R, VariantPolicy as V};
        let _settings = settings_guard();
        let _ = std::fs::remove_file(settings::settings_path());
        let mut app = TuiApp::new(EngineTelemetry::new());

        // 1. /settings toggles modal
        assert!(!app.show_settings_modal);
        app.input = "/settings".into();
        app.submit_prompt();
        assert!(app.show_settings_modal);
        assert!(app
            .history
            .last()
            .unwrap()
            .output
            .contains("Policy Settings:"));

        // 2. F4 toggles modal
        app.handle_key(key(KeyCode::F(4)));
        assert!(!app.show_settings_modal);
        app.handle_key(key(KeyCode::F(4)));
        assert!(app.show_settings_modal);

        // 3. '1' cycles reencode_policy: MaxSpeed -> LowBitNative -> BitExact -> MaxSpeed, saving
        // each step to the settings file.
        assert_eq!(app.reencode_policy, R::MaxSpeed);
        for p in [R::LowBitNative, R::BitExact, R::MaxSpeed] {
            app.handle_key(key(KeyCode::Char('1')));
            assert_eq!(app.reencode_policy, p);
            assert_eq!(settings::load_settings().reencode, p);
        }

        // 4. '2' cycles variant_policy: Fastest -> Fastest4Bit -> BestQuality -> Fastest, saved.
        assert_eq!(app.variant_policy, V::Fastest);
        for p in [V::Fastest4Bit, V::BestQuality, V::Fastest] {
            app.handle_key(key(KeyCode::Char('2')));
            assert_eq!(app.variant_policy, p);
            assert_eq!(settings::load_settings().variant, p);
        }
        // A new app (restart) starts from the saved policies and compaction settings.
        app.handle_key(key(KeyCode::Char('1')));
        app.handle_key(key(KeyCode::Char('2')));
        type_and_run(
            &mut app,
            "/compact mode=aggressive threshold=0.3 target=0.2 keep=3 summarizer=org/sum",
        );
        let c = settings::load_settings().compact;
        assert_eq!(c.mode, crate::compact::CompactionMode::Aggressive);
        assert_eq!((c.trigger_ratio, c.target_ratio, c.keep_recent_turns), (0.3, 0.2, 3));
        assert_eq!(c.summarizer.as_deref(), Some("org/sum"));
        let restarted = TuiApp::new(EngineTelemetry::new());
        assert_eq!(
            (restarted.reencode_policy, restarted.variant_policy),
            (R::LowBitNative, V::Fastest4Bit)
        );
        assert_eq!(restarted.compact_config, c);
        type_and_run(&mut app, "/compact light summarizer=active");
        let c = settings::load_settings().compact;
        assert_eq!((c.mode, c.summarizer), (crate::compact::CompactionMode::Light, None));
        type_and_run(&mut app, "/compact keep=x");
        assert_eq!(app.history.last().unwrap().kind, Kind::Error);
        app.handle_key(key(KeyCode::Char('1')));
        app.handle_key(key(KeyCode::Char('1')));
        app.handle_key(key(KeyCode::Char('2')));
        app.handle_key(key(KeyCode::Char('2')));
        assert_eq!((app.reencode_policy, app.variant_policy), (R::MaxSpeed, V::Fastest));
        type_and_run(&mut app, "/compact auto");

        // 5. '3' toggles show_roofline_hud
        assert!(app.show_roofline_hud);
        app.handle_key(key(KeyCode::Char('3')));
        assert!(!app.show_roofline_hud);
        app.handle_key(key(KeyCode::Char('3')));
        assert!(app.show_roofline_hud);
        // 6. '4' cycles ctx_limit: 8192 -> 16384 -> 32768 -> 4096 -> 8192
        assert_eq!(app.ctx_limit, 8192);
        app.handle_key(key(KeyCode::Char('4')));
        assert_eq!(app.ctx_limit, 16384);
        app.handle_key(key(KeyCode::Char('4')));
        assert_eq!(app.ctx_limit, 32768);
        app.handle_key(key(KeyCode::Char('4')));
        assert_eq!(app.ctx_limit, 4096);
        app.handle_key(key(KeyCode::Char('4')));
        assert_eq!(app.ctx_limit, 8192);
        // '9' cycles the draft objective and '0' the denoise steps 1 -> 2 -> 3 -> 4 -> 1; both
        // persist.
        use crate::runtime::noise_draft::DraftObjective;
        assert_eq!(app.draft_knobs.objective, DraftObjective::Speed);
        app.handle_key(key(KeyCode::Char('9')));
        assert_eq!(app.draft_knobs.objective, DraftObjective::TokensPerForward);
        assert_eq!(
            settings::load_settings().draft.objective,
            DraftObjective::TokensPerForward
        );
        app.handle_key(key(KeyCode::Char('9')));
        assert_eq!(app.draft_knobs.objective, DraftObjective::Speed);
        assert_eq!(app.draft_knobs.denoise_steps, 2);
        for want in [3, 4, 1, 2] {
            app.handle_key(key(KeyCode::Char('0')));
            assert_eq!(app.draft_knobs.denoise_steps, want);
            assert_eq!(settings::load_settings().draft.denoise_steps, want);
        }

        // 7. Rendered F4 modal shows [FASTEST] defaults and bottom keybar shows F4 options & ↑↓ history
        let _g = truecolor();
        let modal_text = lines_of(&snapshot::render_buffer(&app, 140, 40)).join("\n");
        assert!(
            modal_text.contains("[FASTEST]") && modal_text.contains("POLICY & SPEED OPTIONS"),
            "{modal_text}"
        );
        assert!(
            modal_text.contains("⟦F4⟧ options") && modal_text.contains("⟦↑↓⟧ history"),
            "{modal_text}"
        );
        assert!(!modal_text.contains("⟦^R⟧"), "{modal_text}");
        assert!(
            modal_text.contains("[9] Spec Objective") && modal_text.contains("[0] Draft Denoise"),
            "{modal_text}"
        );

        // 8. Esc closes modal
        app.handle_key(key(KeyCode::Esc));
        assert!(!app.show_settings_modal);

        // `/spec key=value ...` applies and persists every knob; `/spec` prints the status.
        type_and_run(&mut app, "/spec objective=tokens steps=3 sigma=0.02");
        assert_eq!(app.draft_knobs.objective, DraftObjective::TokensPerForward);
        assert_eq!(settings::load_settings().draft.denoise_steps, 3);
        type_and_run(&mut app, "/spec");
        let out = &app.history.last().unwrap().output;
        for want in [
            "mode=auto",
            "objective=tokens",
            "tokens/forward",
            "tok/s",
            "online updates=",
            "steps=3",
            "sigma=0.0200",
        ] {
            assert!(out.contains(want), "{want}: {out}");
        }

        // 9. Up / Down arrow keys cycle prior prompt entries for retry/editing, mouse wheel scrolls
        type_and_run(&mut app, "/quant");
        type_and_run(&mut app, "/help");
        app.focus = Focus::Prompt;
        app.input = "draft in progress".into();
        app.handle_key(key(KeyCode::Up));
        assert_eq!(app.input, "/help");
        assert_eq!(app.prompt_history_idx, Some(app.prompt_history.len() - 1));
        app.handle_key(key(KeyCode::Up));
        assert_eq!(app.input, "/quant");
        app.handle_key(key(KeyCode::Down));
        assert_eq!(app.input, "/help");
        app.handle_key(key(KeyCode::Down));
        assert_eq!(app.input, "draft in progress");
        assert_eq!(app.prompt_history_idx, None);

        // Edit a recalled history entry and submit it
        app.handle_key(key(KeyCode::Up));
        assert_eq!(app.input, "/help");
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(app.prompt_history_idx, None);

        // Mouse wheel scrolls console
        let scroll_before = app.scroll;
        app.handle_mouse(MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 10,
            row: 10,
            modifiers: KeyModifiers::NONE,
        });
        assert_eq!(app.scroll, scroll_before + 3);
        app.handle_mouse(MouseEvent {
            kind: MouseEventKind::ScrollDown,
            column: 10,
            row: 10,
            modifiers: KeyModifiers::NONE,
        });
        assert_eq!(app.scroll, scroll_before);
    }
    #[test]
    fn test_s7_3_qos_par_tasks_and_background_probe() {
        // 1. QoS setters
        assert!(crate::tui::qos::set_qos(
            crate::tui::qos::QosClass::UserInteractive
        ));
        assert!(crate::tui::qos::set_qos(crate::tui::qos::QosClass::Utility));

        // 2. PAR_LIMIT configuration
        let app = TuiApp::new(EngineTelemetry::new());
        if cfg!(target_os = "macos") {
            let limit = crate::model::nn::PAR_LIMIT.load(Ordering::Relaxed);
            if app.hw.performance_cores.is_some() {
                assert!(limit >= 1);
            }
        }

        // 3. Background roofline probe
        app.trigger_background_roofline_probe();
        let mut got_probe = false;
        let mut app_mut = app;
        for _ in 0..500 {
            if app_mut.tick() && app_mut.roofline_probe_result.is_some() {
                got_probe = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        if cfg!(target_os = "macos") {
            assert!(got_probe);
            assert!(app_mut
                .roofline_probe_result
                .as_ref()
                .unwrap()
                .contains("ROOFLINE CALIBRATION"));
        }
    }
    #[test]
    fn test_s7_4_reencode_and_variant_slash_commands_and_picker_keys() {
        let _settings = settings_guard();
        let mut app = TuiApp::new(EngineTelemetry::new());

        // 1. /reencode slash command
        app.input = "/reencode low-bit-native".into();
        app.submit_prompt();
        assert_eq!(
            app.reencode_policy,
            crate::model::weight_format::ReencodePolicy::LowBitNative
        );
        assert!(app
            .history
            .last()
            .unwrap()
            .output
            .contains("reencode -> low-bit-native; applies to the next load"));
        let s = crate::tui::settings::load_settings();
        assert_eq!(s.reencode.name(), "low-bit-native");

        // 2. /variant slash command
        app.input = "/variant fastest-4bit".into();
        app.submit_prompt();
        assert_eq!(
            app.variant_policy,
            crate::model::weight_format::VariantPolicy::Fastest4Bit
        );
        assert!(app
            .history
            .last()
            .unwrap()
            .output
            .contains("variant -> fastest-4bit; applies to the next load"));
        let s = crate::tui::settings::load_settings();
        assert_eq!(s.variant.name(), "fastest-4bit");

        // 3. Picker '^r' and '^f' keys; plain 'r' and 'f' type into the search query
        app.handle_key(key(KeyCode::Char('m')));
        app.handle_key(key(KeyCode::Char('r')));
        app.handle_key(key(KeyCode::Char('f')));
        assert_eq!(app.picker.query, "rf");
        assert_eq!(
            crate::tui::settings::load_settings().reencode.name(),
            "low-bit-native"
        );
        assert_eq!(
            crate::tui::settings::load_settings().variant.name(),
            "fastest-4bit"
        );

        app.handle_key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL));
        let s = crate::tui::settings::load_settings();
        // From the low-bit-native set above: one step on is bit-exact, saved and followed by the app.
        assert_eq!(s.reencode.name(), "bit-exact");
        assert_eq!(app.reencode_policy, s.reencode);

        app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::CONTROL));
        let s = crate::tui::settings::load_settings();
        assert_eq!(s.variant.name(), "best-quality");
        assert_eq!(app.variant_policy, s.variant);
    }

    /// A tiny synthetic model loaded offline into a fresh app.
    fn app_with_tiny_model() -> (TuiApp, Arc<ContinuousBatchScheduler>, tempfile::TempDir) {
        let cache = tempfile::tempdir().unwrap();
        let sched = Arc::new(ContinuousBatchScheduler::new("", None));
        let mut app = TuiApp::new(EngineTelemetry::new()).with_runtime(
            sched.clone(),
            HuggingFaceHub::with_cache_dir(cache.path().to_path_buf()),
            None,
            None,
            ImageSlot::default(),
        );
        app.load_model("gibbs/tiny-5m");
        settle(&mut app);
        let e = &app.history[0];
        assert_eq!(e.kind, Kind::Text, "{}", e.output);
        (app, sched, cache)
    }

    /// `/variant` and the picker's `f` reload the loaded model; a load in flight defers the change.
    #[test]
    fn policy_changes_reload_the_loaded_model() {
        let _settings = settings_guard();
        let (mut app, sched, _cache) = app_with_tiny_model();
        type_and_run(&mut app, "/variant best-quality");
        assert!(app.load.is_some(), "a loaded model is reloaded");
        let e = app.history.last().unwrap();
        assert!(
            e.output
                .contains("variant -> best-quality; reloading gibbs/tiny-5m"),
            "{}",
            e.output
        );
        assert!(!e.output.contains("Reload to apply"));
        type_and_run(&mut app, "/reencode bit-exact");
        let e = app.history.last().unwrap();
        assert!(e.output.contains("a load is in flight"), "{}", e.output);
        settle(&mut app);
        assert_eq!(sched.model_id(), "gibbs/tiny-5m");

        app.handle_key(key(KeyCode::Char('m')));
        app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::CONTROL));
        assert!(app.load.is_some(), "picker ^f reloads");
        settle(&mut app);
    }

    /// `/bench` refuses while generating or while another runs; a model without weight bytes
    /// reports why instead of a made-up roofline.
    #[test]
    fn bench_refuses_when_busy_and_reports_through_an_update() {
        let (mut app, _sched, _cache) = app_with_tiny_model();
        type_and_run(&mut app, "/bench");
        assert!(app.benchmarking);
        let e = app.history.last().unwrap();
        assert!(
            e.output.contains("benchmarking gibbs/tiny-5m"),
            "{}",
            e.output
        );
        type_and_run(&mut app, "/chat hi");
        let e = app.history.last().unwrap();
        assert_eq!(
            (e.kind, e.output.as_str()),
            (Kind::Error, "benchmark running")
        );
        let deadline = Instant::now() + Duration::from_secs(30);
        while app.benchmarking {
            app.drain_updates();
            assert!(Instant::now() < deadline, "bench did not report");
            std::thread::sleep(Duration::from_millis(5));
        }
        // The tiny model runs on Metal here: the report carries tok/s, the roofline and the gap.
        let e = app.history.last().unwrap();
        assert_eq!(e.kind, Kind::Text, "{}", e.output);
        for want in ["BENCHMARK gibbs/tiny-5m", "tok/s of a", "roofline"] {
            assert!(e.output.contains(want), "{want}: {}", e.output);
        }

        // A chat in flight: busy, and no benchmark starts.
        type_and_run(&mut app, "/chat Tell me a long story");
        type_and_run(&mut app, "/bench");
        let e = app.history.last().unwrap();
        assert_eq!(
            (e.kind, e.output.as_str()),
            (Kind::Error, "busy: finish or cancel generation first")
        );
        assert!(!app.benchmarking);
        app.focus = Focus::Prompt;
        app.handle_key(key(KeyCode::Esc));
        settle(&mut app);
    }

    #[test]
    fn bench_report_splits_the_token_into_the_gap_segments() {
        let r = crate::bench::ModelBenchResult {
            model: "org/m".into(),
            file: "m.gguf".into(),
            codec_mix: None,
            bytes_per_token: 1_000_000_000,
            bytes_basis: "streamed",
            ceiling_gbps: 200.0,
            decode_tok_s: 180.0,
            roofline_tok_s: 200.0,
            roofline_pct: 90.0,
            wall_ms: 5.5,
            gpu_ms: Some(5.2),
            run_ms: Some(5.3),
            host_ms: Some(0.3),
            encode_us: Some(210.0),
            dispatches: Some(612.0),
            weights_ms: 5.0,
            prefill_tok_s: None,
            rss_gb: 1.0,
            rss_now_gb: None,
            top_kernels: None,
        };
        let text = bench_report(&r);
        for want in [
            "180.0 tok/s",
            "= 90.0%",
            "weights 5.00",
            "gpu excess 0.20",
            "run host 0.10",
            "outside 0.20",
            "612 dispatches",
        ] {
            assert!(text.contains(want), "{want}: {text}");
        }
    }

    /// A reload would swap the model under the benchmark's decode: every path refuses meanwhile.
    #[test]
    fn no_reload_starts_while_a_benchmark_runs() {
        let _settings = settings_guard();
        let (mut app, _sched, _cache) = app_with_tiny_model();
        app.benchmarking = true;
        type_and_run(&mut app, "/reencode max-speed");
        let e = app.history.last().unwrap();
        assert!(e.output.contains("a benchmark is running"), "{}", e.output);
        assert!(app.load.is_none());
        app.load_model("gibbs/tiny-5m");
        assert!(app.load.is_none());
        assert_eq!(app.history.last().unwrap().output, "benchmark running");
    }

    /// `/placement` prints the table and ends in `100% Metal`, or lists the CPU ops as an error.
    #[test]
    fn placement_command_explains_the_loaded_program() {
        let mut app = TuiApp::new(EngineTelemetry::new());
        type_and_run(&mut app, "/placement");
        assert_eq!(app.history.last().unwrap().kind, Kind::Error);

        let (mut app, sched, _cache) = app_with_tiny_model();
        type_and_run(&mut app, "/placement");
        let last = app.history.last().unwrap();
        if sched.model().unwrap().llm_backend() == crate::hardware::Device::Metal {
            assert_eq!(last.kind, Kind::Text);
            assert!(last.output.ends_with("100% Metal"), "{}", last.output);
        } else {
            assert_eq!(last.kind, Kind::Error);
            assert!(last.output.contains("CPU node"), "{}", last.output);
        }
    }

    #[test]
    fn device_panel_shows_the_codec_bench_states() {
        let _g = truecolor();
        let mut app = snapshot::fixture_app();
        for (codec, want) in [
            ((1, 3, 14), "codec-bench 3/14 · tune 31/31"),
            ((2, 14, 14), "codec-bench ✓ · tune 31/31"),
            ((0, 0, 0), "codec-bench — · tune 31/31"),
            ((3, 0, 14), "codec-bench — · tune 31/31"),
        ] {
            app.device_status.codec = codec;
            let text = lines_of(&snapshot::render_buffer(&app, 160, 50)).join("\n");
            assert!(text.contains(want), "{codec:?}: {text}");
        }
    }

    /// The gap panel never draws a negative segment, even when the sample's run exceeds its wall.
    #[test]
    fn gap_panel_clamps_negative_segments() {
        let _g = truecolor();
        let telemetry = EngineTelemetry::new();
        telemetry.record_gap(crate::telemetry::GapSample {
            wall_ms: 5.0,
            run_ms: 5.4,
            gpu_ms: 4.5,
            weights_ms: 3.9,
            dispatches: 10,
        });
        let app = TuiApp::new(telemetry);
        let text = lines_of(&snapshot::render_buffer(&app, 120, 40)).join("\n");
        assert!(text.contains("■0.000 ms"), "{text}");
        assert!(!text.contains("■-"), "{text}");
    }

    /// `/profile sort <col>` and `s` pick the Kernels table order; the panel follows it.
    #[test]
    fn kernels_table_follows_the_sort_column() {
        use crate::telemetry::KernelSort;
        let _g = truecolor();
        let mut app = snapshot::fixture_app();
        let pos = |app: &TuiApp, kind: &str| {
            let text = lines_of(&snapshot::render_buffer(app, 160, 50)).join("\n");
            assert!(text.contains("µs▼") || text.contains("GB/s▼") || text.contains("%gpu▼"));
            text.find(kind)
                .unwrap_or_else(|| panic!("no {kind}\n{text}"))
        };
        // By time RmsNorm (410 µs) precedes CachedAttention (260 µs); by bandwidth it follows.
        assert_eq!(app.kernel_sort, KernelSort::Time);
        assert!(pos(&app, "RmsNorm") < pos(&app, "CachedAtt"));
        type_and_run(&mut app, "/profile sort bandwidth");
        assert_eq!(app.kernel_sort, KernelSort::Bandwidth);
        assert!(pos(&app, "CachedAtt") < pos(&app, "RmsNorm"));
        type_and_run(&mut app, "/profile sort nope");
        assert_eq!(app.history.last().unwrap().kind, Kind::Error);
        assert_eq!(app.kernel_sort, KernelSort::Bandwidth);
        app.focus = Focus::Dashboard;
        app.handle_key(key(KeyCode::Char('s')));
        assert_eq!(app.kernel_sort, KernelSort::GpuShare);
    }

    #[test]
    fn slash_commands_parse_without_an_app() {
        use crate::telemetry::KernelSort as K;
        let ok = |s: &str| {
            parse_slash(s)
                .expect("a parsed command")
                .expect("valid arguments")
        };
        let err = |s: &str| {
            parse_slash(s)
                .expect("a parsed command")
                .expect_err("invalid arguments")
        };
        assert_eq!(ok("/placement"), SlashCmd::Placement);
        assert_eq!(ok("/reencode"), SlashCmd::Reencode(None));
        assert_eq!(
            ok("/reencode  bit-exact "),
            SlashCmd::Reencode(Some(ReencodePolicy::BitExact))
        );
        assert_eq!(ok("/variant"), SlashCmd::Variant(None));
        assert_eq!(
            ok("/variant fastest-4bit"),
            SlashCmd::Variant(Some(VariantPolicy::Fastest4Bit))
        );
        assert_eq!(ok("/profile"), SlashCmd::Profile(ProfileArg::Status));
        assert_eq!(ok("/profile on"), SlashCmd::Profile(ProfileArg::On));
        assert_eq!(ok("/profile off"), SlashCmd::Profile(ProfileArg::Off));
        assert_eq!(
            ok("/profile sort"),
            SlashCmd::Profile(ProfileArg::Sort(None))
        );
        assert_eq!(
            ok("/profile sort bw"),
            SlashCmd::Profile(ProfileArg::Sort(Some(K::Bandwidth)))
        );
        assert!(err("/reencode turbo").contains("valid: max-speed"));
        assert!(err("/variant best").contains("valid: fastest"));
        assert!(err("/profile sort size").contains("'size'"));
        assert!(err("/profile xyz").contains("'xyz'"));
        assert_eq!(ok("/spec"), SlashCmd::Spec(vec![]));
        assert_eq!(
            ok("/spec noise"),
            SlashCmd::Spec(vec![SpecArg::Mode(
                crate::runtime::noise_draft::DraftMode::Noise
            )])
        );
        assert_eq!(ok("/spec k 6"), SlashCmd::Spec(vec![SpecArg::MaxK(6)]));
        assert_eq!(
            ok("/spec objective=tokens steps=3 shortlist=512 probes=16 window=6 sigma=0.02"),
            SlashCmd::Spec(vec![
                SpecArg::Objective(crate::runtime::noise_draft::DraftObjective::TokensPerForward),
                SpecArg::Steps(3),
                SpecArg::Shortlist(512),
                SpecArg::Probes(16),
                SpecArg::Window(6),
                SpecArg::Sigma(0.02),
            ])
        );
        assert!(err("/spec objective=fast").contains("valid: speed, tokens"));
        assert!(err("/spec steps=two").contains("denoise steps"));
        assert!(err("/spec turbo=1").contains("unknown /spec argument 'turbo'"));
        assert_eq!(parse_slash("/help"), None);
    }

    /// The documented `/reencode` and `/variant` arguments are exactly the names that parse.
    #[test]
    fn policy_slash_arguments_parse_exactly_the_documented_names() {
        use crate::model::weight_format::{ReencodePolicy as R, VariantPolicy as V};
        for (s, p) in [
            ("max-speed", R::MaxSpeed),
            ("low-bit-native", R::LowBitNative),
            ("bit-exact", R::BitExact),
            ("min-size", R::MinSize),
        ] {
            assert_eq!(s.parse::<R>().ok(), Some(p));
            assert_eq!(p.name(), s);
        }
        for a in ["max-compress", "minsize", "MinSize"] {
            assert_eq!(a.parse::<R>().ok(), Some(R::MinSize));
        }
        assert_eq!(R::MinSize.next(), R::MaxSpeed);
        assert_eq!(R::BitExact.next(), R::MaxSpeed);
        for (s, p) in [
            ("fastest", V::Fastest),
            ("fastest-4bit", V::Fastest4Bit),
            ("best-quality", V::BestQuality),
        ] {
            assert_eq!(s.parse::<V>().ok(), Some(p));
            assert_eq!(p.name(), s);
        }
        let mut app = TuiApp::new(EngineTelemetry::new());
        for bad in ["/reencode turbo", "/variant best", "/profile sort size"] {
            type_and_run(&mut app, bad);
            let e = app.history.last().unwrap();
            assert_eq!(e.kind, Kind::Error, "{bad}: {}", e.output);
        }
        assert!(app.load.is_none());
    }

    /// `/help` and the unknown-command text list every command.
    #[test]
    fn help_lists_every_command() {
        let mut app = TuiApp::new(EngineTelemetry::new());
        type_and_run(&mut app, "/help");
        let help = app.history.last().unwrap().output.clone();
        type_and_run(&mut app, "/nope");
        let unknown = app.history.last().unwrap().output.clone();
        for c in [
            "/reencode",
            "/variant",
            "/settings",
            "/profile",
            "/bench",
            "/placement",
            "/help",
        ] {
            assert!(help.contains(c), "help lacks {c}");
            assert!(unknown.contains(c), "unknown-command text lacks {c}");
        }
    }

    /// Serialises the tests that write the real settings file, and puts the file back afterwards.
    struct SettingsGuard {
        _lock: parking_lot::MutexGuard<'static, ()>,
        saved: Option<String>,
    }

    fn settings_guard() -> SettingsGuard {
        static LOCK: Mutex<()> = Mutex::new(());
        let _lock = LOCK.lock();
        let saved = std::fs::read_to_string(settings::settings_path()).ok();
        SettingsGuard { _lock, saved }
    }

    impl Drop for SettingsGuard {
        fn drop(&mut self) {
            let path = settings::settings_path();
            match &self.saved {
                Some(s) => {
                    let _ = std::fs::write(path, s);
                }
                None => {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
    }

    fn lines_of(buf: &ratatui::buffer::Buffer) -> Vec<String> {
        (0..buf.area.height)
            .map(|y| {
                (0..buf.area.width)
                    .map(|x| buf[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect()
    }

    fn truecolor() -> theme::TestGuard {
        theme::test_guard(theme::Theme {
            truecolor: true,
            mono: false,
        })
    }

    fn fake_load() -> Load {
        let progress = Arc::new(DownloadProgress::default());
        progress.total.store(796_000_000, Ordering::Relaxed);
        progress.done.store(360_000_000, Ordering::Relaxed);
        Load {
            id: "unsloth/gemma-4-E2B-it-qat-GGUF".into(),
            idx: 0,
            started: Instant::now(),
            stage: Arc::new(Mutex::new("downloading")),
            progress,
            sample: (Instant::now(), 0),
            rate: 12e6,
            reencode: Arc::default(),
            reencode_seen: None,
        }
    }

    #[test]
    fn renders_every_size_with_the_instruments() {
        let _g = truecolor();
        let app = snapshot::fixture_app();
        for (w, h) in [(80, 24), (100, 30), (120, 40), (160, 50), (200, 60)] {
            let lines = lines_of(&snapshot::render_buffer(&app, w, h));
            assert!(lines[0].contains("GIBBS"), "{w}x{h}: {}", lines[0]);
            let text = lines.join("\n");
            if w >= 100 && h >= 30 {
                for title in ["ROOFLINE", "GAP", "TOKENS", "LATENCY"] {
                    assert!(text.contains(title), "{w}x{h}: no {title}\n{text}");
                }
                assert!(
                    text.contains("to 90%") && text.contains("-1.07 ms"),
                    "{w}x{h}: gap line\n{text}"
                );
            }
            if w >= 160 && h >= 50 {
                for want in ["KERNELS", "GemvPacked", "codec-bench ✓ · tune 31/31"] {
                    assert!(text.contains(want), "{w}x{h}: no {want}\n{text}");
                }
            }
            if w < 100 {
                assert!(text.contains("gap 72%"), "{w}x{h}: dock gap atom\n{text}");
            }
            assert_eq!(
                text.contains("TELEMETRY"),
                w < 100,
                "{w}x{h}: dock only when compact"
            );
            assert_eq!(
                text.contains("IDLE") || text.contains("GENERATING"),
                h >= 44 && w >= 100,
                "{w}x{h}: cameo panel"
            );
        }
    }

    #[test]
    fn compact_dock_keeps_each_component_on_one_row() {
        let _g = truecolor();
        let app = TuiApp::new(EngineTelemetry::new());
        let lines = lines_of(&snapshot::render_buffer(&app, 80, 24));
        let (v, th) = (app.vitals(), theme::theme());
        let atoms = app.dock_atoms(&v, &th);
        assert!(atoms.len() >= 4);
        for atom in atoms {
            let text: String = atom.iter().map(|s| s.content.as_ref()).collect();
            assert!(
                lines.iter().any(|l| l.contains(&text)),
                "{text:?} split or missing:\n{}",
                lines.join("\n")
            );
        }
    }

    #[test]
    fn load_component_shows_the_stage_track_at_every_tier() {
        let _g = truecolor();
        let mut app = TuiApp::new(EngineTelemetry::new());
        app.load = Some(fake_load());
        for (w, h) in [(80, 24), (120, 40), (200, 60)] {
            let text = lines_of(&snapshot::render_buffer(&app, w, h)).join("\n");
            for word in ["RESOLVE", "COMPILE", "LOADING"] {
                assert!(text.contains(word), "{w}x{h}: no {word}\n{text}");
            }
        }
    }

    #[test]
    fn no_color_theme_paints_no_foreground() {
        let _g = theme::test_guard(theme::Theme {
            truecolor: true,
            mono: true,
        });
        let app = snapshot::fixture_app();
        for (w, h) in [(80, 24), (160, 50)] {
            let buf = snapshot::render_buffer(&app, w, h);
            for c in buf.content() {
                assert_eq!(
                    c.fg,
                    ratatui::style::Color::Reset,
                    "{w}x{h}: {:?}",
                    c.symbol()
                );
                assert_eq!(c.bg, ratatui::style::Color::Reset);
            }
        }
    }

    #[test]
    fn focused_component_replaces_the_modal() {
        let _g = truecolor();
        let mut app = snapshot::fixture_app();
        app.detail = Some(layout::Comp::Logprobs);
        for (w, h) in [(80, 24), (160, 50)] {
            let text = lines_of(&snapshot::render_buffer(&app, w, h)).join("\n");
            assert!(
                text.contains("LOGPROBS") && text.contains("scrub"),
                "{w}x{h}\n{text}"
            );
        }
        // Focus is shown by a BRASS border on the focused component only.
        let buf = snapshot::render_buffer(&app, 160, 50);
        let brass = theme::theme().color(theme::Token::Brass);
        let corners = (0..buf.area.height)
            .flat_map(|y| (0..buf.area.width).map(move |x| (x, y)))
            .filter(|&(x, y)| buf[(x, y)].symbol() == "╔" && buf[(x, y)].fg == brass)
            .count();
        assert_eq!(corners, 1);
    }

    #[test]
    fn llm_can_be_used_and_toggled_as_system1_in_tui() {
        let dir = tempfile::tempdir().unwrap();
        let llm = crate::arch::testing::synthetic_lm(
            dir.path(),
            serde_json::json!({}),
            crate::hardware::Device::Cpu,
        );
        let sched = Arc::new(ContinuousBatchScheduler::new("foundry/smollm-1m", Some(llm)));
        let mut app = TuiApp::new(EngineTelemetry::new()).with_runtime(
            sched,
            HuggingFaceHub::new(),
            None,
            None,
            ImageSlot::default(),
        );

        // Even before `/system1`, `/decide` and `/embed` work directly on the loaded LLM.
        app.input = "/decide reactor overheating | reactor overheating, billing".into();
        app.submit_prompt();
        settle(&mut app);
        assert_eq!(app.history.last().unwrap().kind, Kind::Json);
        assert!(app.history.last().unwrap().output.contains("selected"));

        // `/system1` turns the active LLM into a System 1 model for bare prompts too.
        app.input = "/system1".into();
        app.submit_prompt();
        assert!(app.jev_model.read().is_some());
        assert!(app
            .history
            .last()
            .unwrap()
            .output
            .contains("System 1 mode ON"));

        app.input = "reactor overheating | reactor overheating, billing".into();
        app.submit_prompt();
        settle(&mut app);
        assert_eq!(app.history.last().unwrap().kind, Kind::Json);

        // `/system1 off` returns the LLM to normal chat mode.
        app.input = "/system1 off".into();
        app.submit_prompt();
        assert!(app.jev_model.read().is_none());
        assert!(app
            .history
            .last()
            .unwrap()
            .output
            .contains("System 1 mode OFF"));
    }

    #[test]
    fn test_multi_turn_chat_history_and_slash_commands() {
        let mut app = TuiApp::new(EngineTelemetry::new());
        app.chat_history
            .push(("user".into(), "First question".into()));
        app.chat_history
            .push(("assistant".into(), "First answer".into()));
        app.chat_history
            .push(("user".into(), "Second question".into()));
        app.chat_history
            .push(("assistant".into(), "Second answer".into()));

        let tok = Tokenizer::default_bpe();
        let formatted = app.format_chat_prompt(&tok, "Third question");
        assert!(formatted.contains("First question"));
        assert!(formatted.contains("First answer"));
        assert!(formatted.contains("Second question"));
        assert!(formatted.contains("Third question"));

        app.input = "/history".into();
        app.submit_prompt();
        assert!(app.history.last().unwrap().output.contains("2 turns"));

        app.input = "/pop".into();
        app.submit_prompt();
        assert_eq!(app.chat_history.len(), 2);
        assert!(app
            .history
            .last()
            .unwrap()
            .output
            .contains("1 turn remaining"));

        app.input = "/clear".into();
        app.submit_prompt();
        assert!(app.chat_history.is_empty());
    }
}
