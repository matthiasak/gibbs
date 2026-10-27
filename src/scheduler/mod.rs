pub mod spec;

use crate::hardware::Device;
use crate::kv::{BlockId, PagedKVPool, TOKENS_PER_BLOCK};
use crate::model::native::{MediaInput, NativeLm, PrepareCtx, SeqState};
use crate::model::tokenizer::Tokenizer;
use crate::runtime::lm::sample_logits;
use crate::runtime::noise_draft::{DraftKnobs, DraftMode, LiveDraftKnobs};
use crate::telemetry::{EngineTelemetry, GapSample};
use parking_lot::{Condvar, Mutex, RwLock};
use spec::{Drafter, KTuner, SpecSeq};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

/// Prefix of the error that ends a sequence aborted by [`Scheduler::swap_model`]; the server
/// answers such a request with 503 (retry) rather than 500.
pub const MODEL_SWAPPED: &str = "model swapped to ";

/// Top-k every stochastic sampler draws from: the scheduler's, speculative verification's and the
/// Metal stream's (`GraphProgram::stream_next_sampled`), so all draw from one distribution.
pub(crate) const SAMPLE_TOP_K: usize = 50;

/// Label for CPU execution, shared by the banner, logs, and TUI.
pub fn cpu_backend_label() -> String {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    format!("CPU (NEON SIMD), {threads} threads")
}

/// A loaded causal LM: any [`NativeLm`] (family specs through [`crate::runtime::lm`], the
/// encoder-decoder pipeline). Each sequence carries the [`SeqState`] its `prepare` built.
#[derive(Clone)]
pub struct ExecutableModel(pub Arc<dyn NativeLm>);

/// Last 64 tokens: the window the sampler sees.
fn recent(tokens: &[u32]) -> &[u32] {
    &tokens[tokens.len().saturating_sub(64)..]
}

const NO_STATE: &str = "sequence has no model state (it was not prepared by this model)";

impl ExecutableModel {
    pub fn new(lm: Arc<dyn NativeLm>) -> Self {
        Self(lm)
    }

    /// The model behind this handle.
    pub fn lm(&self) -> &Arc<dyn NativeLm> {
        &self.0
    }

    pub fn llm_backend(&self) -> Device {
        self.0.device()
    }

    /// e.g. `CPU (NEON SIMD), 14 threads` or `Metal (Apple M5 Pro)` (device name from the Metal API).
    pub fn backend_label(&self) -> String {
        self.0.backend_label()
    }
    /// Graph nodes by op kind, device and avenue; `None` for models that do not run a graph
    /// program.
    pub fn placement_summary(&self) -> Option<Vec<crate::graph::placement::PlacementRow>> {
        self.0.placement_summary()
    }
    /// Row `id` of the model's token-embedding table ([`NativeLm::embed_row`]).
    pub fn embed_row(&self, id: u32, out: &mut [f32]) -> bool {
        self.0.embed_row(id, out)
    }
    /// Decodes `last_token` against the sequence's state and samples the next token.
    /// `logprobs: Some(k)` also reports the chosen token's probability, the top `k` candidates
    /// and the entropy.
    #[allow(clippy::too_many_arguments)]
    pub fn step_token_with_kv(
        &self,
        all_tokens: &[u32],
        last_token: u32,
        pos: usize,
        temp: f32,
        top_p: f32,
        logprobs: Option<usize>,
        state: &mut Option<SeqState>,
    ) -> Result<(u32, Option<TokenLogprobInfo>), String> {
        let state = state.as_mut().ok_or(NO_STATE)?;
        let logits = self.0.forward(&[last_token], pos - 1, state)?;
        let chosen = self.sample(&logits, recent(all_tokens), temp, top_p);
        Ok((chosen, logprobs.map(|k| logprob_info(&logits, chosen, k))))
    }

    /// The next greedy token of a device decode stream ([`NativeLm::stream_greedy_token`]): `last`
    /// is the newest token, `pos` counts it (as in [`Self::step_token_with_kv`]), and at most
    /// `max_new` more tokens are queued. `None` when this model or sequence cannot stream (use
    /// `step_token_with_kv`). A stream keeps one speculative token in flight: end it with
    /// [`Self::stream_end`] before the sequence state is reused.
    pub fn stream_greedy_token(
        &self,
        last: u32,
        pos: usize,
        max_new: usize,
        state: &mut Option<SeqState>,
    ) -> Result<Option<u32>, String> {
        match state {
            Some(st) => self.0.stream_greedy_token(last, pos - 1, max_new, st),
            None => Ok(None),
        }
    }

    /// Device decode stream supporting both greedy (`temp <= 1e-4`) and stochastic (`temp > 1e-4`) sampling.
    #[allow(clippy::too_many_arguments)]
    pub fn stream_token(
        &self,
        last: u32,
        pos: usize,
        max_new: usize,
        temp: f32,
        top_p: f32,
        all_tokens: &[u32],
        state: &mut Option<SeqState>,
    ) -> Result<Option<u32>, String> {
        match state {
            Some(st) => {
                self.0
                    .stream_token(last, pos - 1, max_new, temp, top_p, recent(all_tokens), st)
            }
            None => Ok(None),
        }
    }

    /// Ends the sequence's device decode stream, if any (waits for its in-flight tokens).
    pub fn stream_end(&self, state: &mut Option<SeqState>) -> Result<(), String> {
        match state {
            Some(st) => self.0.stream_end(st),
            None => Ok(()),
        }
    }

    /// Evaluates a chunk of prompt tokens. If this chunk completes the prompt,
    /// returns `Some(next_token)`. If more prompt tokens remain, returns `None`.
    #[allow(clippy::too_many_arguments)]
    pub fn step_prefill_chunk(
        &self,
        chunk_tokens: &[u32],
        start_pos: usize,
        is_last_chunk: bool,
        all_prompt_tokens: &[u32],
        temp: f32,
        top_p: f32,
        state: &mut Option<SeqState>,
    ) -> Result<Option<u32>, String> {
        if chunk_tokens.is_empty() {
            return Ok(None);
        }
        let logits = self.prefill_logits(chunk_tokens, start_pos, state)?;
        Ok(is_last_chunk.then(|| self.sample(&logits, recent(all_prompt_tokens), temp, top_p)))
    }

    /// Logits at the last position of `chunk_tokens` (non-empty), advancing `state`.
    pub(crate) fn prefill_logits(
        &self,
        chunk_tokens: &[u32],
        start_pos: usize,
        state: &mut Option<SeqState>,
    ) -> Result<Vec<f32>, String> {
        let st = state.as_mut().ok_or(NO_STATE)?;
        self.0.forward(chunk_tokens, start_pos, st)
    }

    /// Last-position logits for a text prompt (already tokenized, template applied by the caller).
    pub fn prompt_logits(&self, ids: Vec<u32>) -> Result<Vec<f32>, String> {
        let (ids, mut st) = self.prepare(ids, &[], &PrepareCtx::default())?;
        if ids.is_empty() {
            return Err("the prompt encodes to no tokens".into());
        }
        self.prefill_logits(&ids, 0, &mut st)
    }

    /// Top-[`SAMPLE_TOP_K`] / top-p sampling without repetition penalty; `temp <= 0` is greedy argmax.
    fn sample(&self, logits: &[f32], recent: &[u32], temp: f32, top_p: f32) -> u32 {
        let vocab = self.0.tokenizer().vocab_size();
        let valid = if vocab > 0 {
            logits.len().min(vocab)
        } else {
            logits.len()
        };
        sample_logits(&logits[..valid], recent, temp, SAMPLE_TOP_K, top_p, 1.0)
    }

    /// One-shot generation through a private [`ContinuousBatchScheduler`]: (text, decode-step
    /// tokens, decode time). `drafter` speculates; models otherwise use the drafter they ship.
    pub fn generate_text_measured(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        drafter: Option<Arc<dyn Drafter>>,
    ) -> Result<(String, usize, std::time::Duration), String> {
        let s = self.one_shot_scheduler();
        if drafter.is_some() {
            s.set_drafter(drafter);
        }
        let g = s.generate(prompt, max_new_tokens, temperature)?;
        Ok((g.text, g.decode.1, g.decode.0))
    }

    /// [`Self::generate_text_measured`] with drafters off: one forward pass per generated token,
    /// the quantity the bench's roofline is defined over.
    pub fn generate_text_measured_plain(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<(String, usize, std::time::Duration), String> {
        let s = self.one_shot_scheduler();
        s.set_drafter(None);
        let g = s.generate(prompt, max_new_tokens, temperature)?;
        Ok((g.text, g.decode.1, g.decode.0))
    }

    /// A private scheduler for one-shot generation: its KV pool holds the model's whole context
    /// (at least [`ONE_SHOT_POOL_BLOCKS`]).
    pub(crate) fn one_shot_scheduler(&self) -> ContinuousBatchScheduler {
        // ponytail: block ids are bookkeeping only; the 4M-token cap bounds that list.
        let blocks = self
            .context_len()
            .unwrap_or(0)
            .min(1 << 22)
            .div_ceil(TOKENS_PER_BLOCK)
            .max(ONE_SHOT_POOL_BLOCKS);
        ContinuousBatchScheduler::new("", Arc::new(PagedKVPool::new(blocks)), Some(self.clone()))
    }

    pub fn tokenizer(&self) -> Tokenizer {
        self.0.tokenizer().clone()
    }

    /// Maximum positions the model's state holds.
    pub fn context_len(&self) -> Option<usize> {
        Some(self.0.context_len().max(1))
    }

    /// Expands `ids` for the model ([`NativeLm::prepare_ctx`], media allowed) and builds the
    /// sequence's initial state.
    fn prepare(
        &self,
        ids: Vec<u32>,
        media: &[MediaInput],
        ctx: &PrepareCtx,
    ) -> Result<(Vec<u32>, Option<SeqState>), String> {
        let (ids, state) = self.0.prepare_ctx(ids, media, ctx)?;
        Ok((ids, Some(state)))
    }

    /// [`Self::prepare`]'s id count without building state ([`NativeLm::prompt_len`]).
    fn prompt_len(&self, ids: Vec<u32>, media: &[MediaInput]) -> Result<usize, String> {
        self.0.prompt_len(ids, media)
    }

    pub fn export_kv_snapshot(
        &self,
        state: &Option<SeqState>,
        token_len: usize,
    ) -> Option<crate::kv::CachedKvSnapshot> {
        self.0.export_kv_snapshot(state.as_ref()?, token_len)
    }

    pub fn restore_kv_prefix(
        &self,
        state: &mut Option<SeqState>,
        snap: &crate::kv::CachedKvSnapshot,
        prefix_len: usize,
    ) -> bool {
        let Some(st) = state.as_mut() else {
            return false;
        };
        self.0.restore_kv_prefix(st, snap, prefix_len)
    }

    /// [`NativeLm::thinking_delimiters`].
    pub fn thinking_delimiters(&self) -> Vec<(String, String)> {
        self.0.thinking_delimiters()
    }

    /// [`NativeLm::has_vision`].
    pub fn has_vision(&self) -> bool {
        self.0.has_vision()
    }
}

/// `chosen`'s probability, the top `top_k` candidates and the exact entropy of `logits`.
fn logprob_info(logits: &[f32], chosen: u32, top_k: usize) -> TokenLogprobInfo {
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
    let lse = max_l + sum_exp.ln();
    let entropy = -logits
        .iter()
        .map(|&l| {
            let p = (l - lse).exp();
            if p > 1e-12 {
                p * (l - lse)
            } else {
                0.0
            }
        })
        .sum::<f32>();
    // Top k in O(V + k log k).
    let mut indexed: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, l))
        .collect();
    let k_take = top_k.min(indexed.len()).max(1);
    if k_take < indexed.len() {
        indexed.select_nth_unstable_by(k_take - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let top_slice = &mut indexed[..k_take];
    top_slice.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_candidates = top_slice
        .iter()
        .map(|&(tok, l)| (tok, (l - lse).exp()))
        .collect();
    let chosen_prob = (logits.get(chosen as usize).copied().unwrap_or(max_l) - lse)
        .exp()
        .min(1.0);
    TokenLogprobInfo {
        chosen_id: chosen,
        chosen_prob,
        top_candidates,
        entropy,
    }
}

/// What [`ContinuousBatchScheduler::generate`] produced and what it cost.
pub struct Generation {
    pub text: String,
    /// (time, prompt tokens) of prefill.
    pub prefill: (Duration, usize),
    /// (time, tokens emitted by decode steps); the first token comes from prefill.
    pub decode: (Duration, usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceState {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
}

/// Default output cap for requests that do not set `max_tokens`.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 64;
/// Sampling temperature for requests that do not set `temperature` (the OpenAI default).
pub const DEFAULT_TEMPERATURE: f32 = 1.0;
/// Nucleus cutoff for requests that do not set `top_p`.
pub const DEFAULT_TOP_P: f32 = 0.9;
/// Most prompt tokens one scheduler step prefills, across all sequences.
pub const MAX_PREFILL_CHUNK: usize = 2048;
/// KV blocks of the server's pool, and the least a one-shot generation's private pool gets.
pub const ONE_SHOT_POOL_BLOCKS: usize = 16384;
/// Prefill budget before the first prefill has been timed.
const INITIAL_PREFILL_CHUNK: usize = 64;
/// With nothing decoding, a step's prefill is sized to take about this long, so a request that
/// arrives mid-prefill waits at most about this long before it is admitted.
const IDLE_STEP: Duration = Duration::from_millis(100);

/// Measured costs that size the prefill budget. While sequences decode, a step's prefill is
/// sized to cost about one decode round (at least 1 token), so prompt ingestion takes roughly
/// half of each step; on CPU one prompt token can cost more than a decode round, so the 1-token
/// floor can make a step longer than that. Otherwise prefill is sized to cost [`IDLE_STEP`].
/// Backends that batch prefill (Metal) measure a lower cost per token than per-token CPU
/// prefill, so they get larger budgets.
// ponytail: one EWMA of cost per token, though batched prefill gets cheaper per token as chunks
// grow; fit cost(chunk) if budgets oscillate.
#[derive(Default)]
struct PrefillTuner {
    /// EWMA of prefill time per prompt token (ns); 0 until measured.
    prefill_ns_per_token: f64,
    /// EWMA of one step's decode work over all decoding sequences (ns); 0 until measured.
    decode_round_ns: f64,
    /// Prefill each prompt in one call (one-shot generation: matches a lone prefill's logits
    /// and timing).
    whole_prompt: bool,
}

impl PrefillTuner {
    fn budget(&self, decoding: bool) -> usize {
        if self.whole_prompt {
            return usize::MAX;
        }
        if self.prefill_ns_per_token == 0.0 {
            return INITIAL_PREFILL_CHUNK;
        }
        let target_ns = match (decoding, self.decode_round_ns) {
            // Decode cost not timed yet: give the decoding sequences the step.
            (true, d) if d == 0.0 => return 1,
            (true, d) => d,
            (false, _) => IDLE_STEP.as_nanos() as f64,
        };
        ((target_ns / self.prefill_ns_per_token) as usize).clamp(1, MAX_PREFILL_CHUNK)
    }

    /// `prefill` = (time, prompt tokens) and `decode` = (time, decoded tokens) spent in one step.
    fn record(&mut self, prefill: (Duration, usize), decode: (Duration, usize)) {
        let ewma = |old: f64, new: f64| {
            if old == 0.0 {
                new
            } else {
                0.75 * old + 0.25 * new
            }
        };
        if prefill.1 > 0 {
            let per_token = prefill.0.as_nanos() as f64 / prefill.1 as f64;
            self.prefill_ns_per_token = ewma(self.prefill_ns_per_token, per_token);
        }
        if decode.1 > 0 {
            self.decode_round_ns = ewma(self.decode_round_ns, decode.0.as_nanos() as f64);
        }
    }
}

/// Per-sequence sampling controls (OpenAI `temperature` / `top_p` / `max_tokens`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    /// `None` uses the model's generation_config default, else [`DEFAULT_TEMPERATURE`]; `<= 0` is
    /// exact greedy argmax.
    pub temperature: Option<f32>,
    /// `None` uses the model's generation_config default, else [`DEFAULT_TOP_P`].
    pub top_p: Option<f32>,
    pub max_new_tokens: usize,
    pub top_logprobs: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprobInfo {
    pub chosen_id: u32,
    pub chosen_prob: f32,
    pub top_candidates: Vec<(u32, f32)>,
    pub entropy: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            top_logprobs: None,
        }
    }
}

impl SamplingParams {
    /// An unset `max_tokens` runs to EOS or the context end (`submit_with_media` clamps it).
    pub fn from_request(temperature: Option<f32>, max_tokens: Option<usize>) -> Self {
        Self {
            temperature,
            top_p: None,
            max_new_tokens: max_tokens.unwrap_or(usize::MAX),
            top_logprobs: None,
        }
    }
}

/// Why a sequence stopped, in OpenAI `finish_reason` terms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// EOS/special token, or no model.
    Stop,
    /// `max_new_tokens` or the model's context length was reached.
    Length,
    /// The engine failed mid-request (prefill/decode error); the message is the cause.
    Error(String),
}

/// Set once by the scheduler before the token stream closes; unset means `Stop`.
pub type FinishSignal = Arc<std::sync::OnceLock<FinishReason>>;

/// OpenAI `finish_reason` for a closed stream, or the engine error that aborted it.
pub fn finish_outcome(signal: &FinishSignal) -> Result<&'static str, String> {
    match signal.get() {
        None | Some(FinishReason::Stop) => Ok("stop"),
        Some(FinishReason::Length) => Ok("length"),
        Some(FinishReason::Error(e)) => Err(e.clone()),
    }
}

pub struct Sequence {
    pub id: u64,
    pub prompt_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub allocated_blocks: Vec<BlockId>,
    pub params: SamplingParams,
    pub state: SequenceState,
    pub submitted: Instant,
    /// When the last output token was sent; `None` before the first.
    pub last_emit: Option<Instant>,
    pub token_sender: Option<UnboundedSender<u32>>,
    pub kv_cache: Option<SeqState>,
    pub prefilled_tokens: usize,
    pub finish: FinishSignal,
    /// Speculative decoding state; set when the sequence starts prefill with a drafter.
    pub spec: Option<SpecSeq>,
}

impl Sequence {
    pub fn new(id: u64, prompt_tokens: Vec<u32>) -> Self {
        Self {
            id,
            prompt_tokens,
            output_tokens: Vec::new(),
            allocated_blocks: Vec::new(),
            params: SamplingParams::default(),
            state: SequenceState::Waiting,
            submitted: Instant::now(),
            last_emit: None,
            token_sender: None,
            kv_cache: None,
            prefilled_tokens: 0,
            finish: FinishSignal::default(),
            spec: None,
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.output_tokens.len()
    }

    /// Appends `toks` (one decode step's output), streams them, and records their latency: time
    /// to first token, else the gap since the previous step shared evenly by the step's tokens
    /// (returned).
    fn emit(&mut self, toks: &[u32], telemetry: &EngineTelemetry) -> Option<Duration> {
        self.output_tokens.extend_from_slice(toks);
        if let Some(sender) = &self.token_sender {
            for &t in toks {
                let _ = sender.send(t);
            }
        }
        let now = Instant::now();
        match self.last_emit.replace(now) {
            None => {
                telemetry.record_ttft(now - self.submitted);
                None
            }
            Some(prev) => {
                let itl = (now - prev) / toks.len().max(1) as u32;
                for _ in toks {
                    telemetry.record_itl(itl);
                }
                (!toks.is_empty()).then_some(itl)
            }
        }
    }
}

pub struct BatchStepOutput {
    pub tokens_generated: usize,
    pub active_sequences_count: usize,
    /// (time, prompt tokens) this step prefilled.
    pub prefill: (Duration, usize),
    /// (time, tokens emitted) of this step's decode work.
    pub decode: (Duration, usize),
}

/// One admitted sequence, as of the end of the last step.
#[derive(Debug, Clone)]
pub struct SeqView {
    pub id: u64,
    pub state: SequenceState,
    pub prompt: usize,
    pub prefilled: usize,
    pub generated: usize,
    pub max_new: usize,
    pub submitted: Instant,
}

/// What the scheduler did in its last step, published for dashboards without touching the
/// locks a step holds.
#[derive(Debug, Clone, Default)]
pub struct SchedulerView {
    pub seqs: Vec<SeqView>,
    pub waiting: usize,
    /// Prefill budget (prompt tokens) the last step used; see [`PrefillTuner`].
    pub prefill_budget: usize,
    /// Duration of the last step that ran a forward pass.
    pub busy_step: Duration,
    /// The speculative drafter, when one is active.
    pub spec: Option<SpecView>,
}

/// Speculative decoding as of the last step; rates are in [`EngineTelemetry`]'s snapshot.
#[derive(Debug, Clone)]
pub struct SpecView {
    pub label: String,
    /// Draft tokens per step the tuner currently rates best (0 = plain decoding pays off);
    /// `None` until it has measured a step.
    pub k: Option<usize>,
    /// The drafter's online learning updates so far (online drafters).
    pub updates: Option<u64>,
}

pub struct ContinuousBatchScheduler {
    model_id: RwLock<String>,
    pub kv_pool: Arc<PagedKVPool>,
    model: RwLock<Option<ExecutableModel>>,
    tokenizer: RwLock<Arc<Tokenizer>>,
    pub telemetry: Arc<EngineTelemetry>,
    pub waiting_queue: RwLock<VecDeque<Sequence>>,
    pub active_sequences: RwLock<Vec<Sequence>>,
    pub step_counter: AtomicU32,
    next_seq_id: AtomicU64,
    prefill_tuner: Mutex<PrefillTuner>,
    view: Mutex<SchedulerView>,
    /// Drafter for speculative decoding: the model's own (MTP heads) unless replaced.
    drafter: RwLock<Option<Arc<dyn Drafter>>>,
    k_tuner: Mutex<KTuner>,
    draft_knobs: Arc<LiveDraftKnobs>,
    pub logprob_tx: Arc<RwLock<Option<UnboundedSender<TokenLogprobInfo>>>>,
    work_signal: (Mutex<bool>, Condvar),
}

/// The drafter `model` ships (or an [`OnlineNoiseDrafter`] when `knobs.mode` is `Noise`/`Suffix`
/// or `GIBBS_SPEC=noise|suffix`), unless `GIBBS_SPEC=off` or `knobs.mode == DraftMode::Off`.
fn resolve_drafter(
    model: Option<&ExecutableModel>,
    knobs: &Arc<LiveDraftKnobs>,
) -> Option<Arc<dyn Drafter>> {
    let m = model?;
    if !spec_enabled() {
        return None;
    }
    match knobs.snapshot().mode {
        DraftMode::Off => None,
        DraftMode::Noise | DraftMode::Suffix => {
            m.0.online_drafter(knobs.clone()).or_else(|| m.0.drafter())
        }
        DraftMode::Model => m.0.drafter(),
        DraftMode::Auto => {
            if matches!(
                std::env::var("GIBBS_SPEC").as_deref(),
                Ok("noise" | "online" | "suffix")
            ) {
                m.0.online_drafter(knobs.clone()).or_else(|| m.0.drafter())
            } else {
                m.0.drafter().or_else(|| m.0.online_drafter(knobs.clone()))
            }
        }
    }
}

fn spec_enabled() -> bool {
    std::env::var("GIBBS_SPEC").map_or(true, |v| v != "off")
}

impl ContinuousBatchScheduler {
    pub fn new(
        model_id: impl Into<String>,
        kv_pool: Arc<PagedKVPool>,
        model: Option<ExecutableModel>,
    ) -> Self {
        Self::with_telemetry(model_id, kv_pool, model, EngineTelemetry::new())
    }

    pub fn with_telemetry(
        model_id: impl Into<String>,
        kv_pool: Arc<PagedKVPool>,
        model: Option<ExecutableModel>,
        telemetry: Arc<EngineTelemetry>,
    ) -> Self {
        let tokenizer = Arc::new(
            model
                .as_ref()
                .map(ExecutableModel::tokenizer)
                .unwrap_or_else(Tokenizer::default_bpe),
        );
        let has_model = model.is_some();
        let mut init_knobs = crate::tui::settings::load_settings().draft;
        if let Ok(env_spec) = std::env::var("GIBBS_SPEC") {
            if let Ok(mode) = env_spec.parse::<DraftMode>() {
                init_knobs.mode = mode;
            }
        }
        let draft_knobs = Arc::new(LiveDraftKnobs::new(init_knobs));
        let s = Self {
            model_id: RwLock::new(model_id.into()),
            kv_pool,
            drafter: RwLock::new(resolve_drafter(model.as_ref(), &draft_knobs)),
            model: RwLock::new(model),
            tokenizer: RwLock::new(tokenizer),
            telemetry,
            waiting_queue: RwLock::new(VecDeque::new()),
            active_sequences: RwLock::new(Vec::new()),
            step_counter: AtomicU32::new(0),
            next_seq_id: AtomicU64::new(0),
            prefill_tuner: Mutex::new(PrefillTuner::default()),
            view: Mutex::new(SchedulerView::default()),
            k_tuner: Mutex::new(KTuner::default()),
            draft_knobs,
            logprob_tx: Arc::new(RwLock::new(None)),
            work_signal: (Mutex::new(false), Condvar::new()),
        };
        s.sync_spec_view();
        if has_model {
            s.log_drafter();
        }
        s
    }

    fn spec_view(&self) -> Option<SpecView> {
        self.drafter.read().as_ref().map(|d| SpecView {
            label: d.label(),
            k: self.k_tuner.lock().best(),
            updates: d.online_updates(),
        })
    }

    /// Updates `self.view.lock().spec` immediately when the active drafter or model changes.
    fn sync_spec_view(&self) {
        let spec = self.spec_view();
        self.view.lock().spec = spec;
    }

    /// Says which drafter new sequences speculate with, if any.
    fn log_drafter(&self) {
        match self.drafter.read().as_ref() {
            Some(d) => self.telemetry.log_info(&format!(
                "speculative decoding: {} drafter, k picked from measured speed (0..={}); GIBBS_SPEC=off disables it",
                d.label(),
                d.max_k()
            )),
            None if !spec_enabled() => self
                .telemetry
                .log_info("speculative decoding: off (GIBBS_SPEC=off)"),
            None => {}
        }
    }

    /// Tokenizer belonging to the currently executing model.
    pub fn tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.read().clone()
    }

    pub fn has_model(&self) -> bool {
        self.model.read().is_some()
    }

    /// The executing model (a cheap clone: every variant shares its weights); `None` without an LLM.
    pub fn model(&self) -> Option<ExecutableModel> {
        self.model.read().clone()
    }

    /// Folds one plain decode token into the telemetry gap: its inter-token latency `itl` against
    /// the newest GPU run sample. Nothing is recorded without one (CPU, or a profiled token).
    #[cfg(target_os = "macos")]
    fn record_gap(&self, itl: Duration) {
        // ponytail: one process-wide cursor; per-scheduler if two schedulers decode at once.
        static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let Some((seq, run)) = crate::graph::metal::last_run_sample() else {
            return;
        };
        if SEEN.swap(seq, std::sync::atomic::Ordering::Relaxed) == seq {
            return;
        }
        let ceiling = crate::model::variant::host_ceiling_gbps();
        let weights_ms = if ceiling > 0.0 {
            self.telemetry.bytes_per_token() as f64 / 1e9 / ceiling * 1e3
        } else {
            0.0
        };
        self.telemetry.record_gap(GapSample {
            wall_ms: itl.as_secs_f32() * 1e3,
            run_ms: run.run_ns as f32 / 1e6,
            gpu_ms: run.gpu_ns as f32 / 1e6,
            weights_ms: weights_ms as f32,
            dispatches: run.dispatches,
        });
    }

    #[cfg(not(target_os = "macos"))]
    fn record_gap(&self, _itl: Duration) {}

    /// Backend the current model executes on (CPU when no LLM is loaded).
    pub fn llm_backend(&self) -> Device {
        self.model
            .read()
            .as_ref()
            .map_or(Device::Cpu, ExecutableModel::llm_backend)
    }

    /// Truthful compute label for the current model (CPU when no LLM is loaded).
    pub fn backend_label(&self) -> String {
        self.model
            .read()
            .as_ref()
            .map_or_else(cpu_backend_label, ExecutableModel::backend_label)
    }

    /// Id of the model this engine serves (the `--model` id, or the last hot-swapped one).
    pub fn model_id(&self) -> String {
        self.model_id.read().clone()
    }

    /// Prefill budget (prompt tokens per step) the engine chose for its last step.
    pub fn prefill_chunk(&self) -> usize {
        self.view.lock().prefill_budget
    }

    /// Sequences, queue, and step timing as of the end of the last step.
    pub fn view(&self) -> SchedulerView {
        self.view.lock().clone()
    }

    /// Replaces the executing model and its tokenizer. In-flight sequences are aborted
    /// (their token streams end with an error) because their KV caches belong to the old weights.
    pub fn swap_model(&self, model_id: &str, model: ExecutableModel) {
        let mut active = self.active_sequences.write();
        let mut waiting = self.waiting_queue.write();
        self.fail_all(
            &mut active,
            &mut waiting,
            &format!("{MODEL_SWAPPED}{model_id}"),
        );
        *self.tokenizer.write() = Arc::new(model.tokenizer());
        *self.drafter.write() = resolve_drafter(Some(&model), &self.draft_knobs);
        *self.model.write() = Some(model);
        *self.model_id.write() = model_id.to_string();
        // Prefill and decode costs were measured on the old model.
        *self.prefill_tuner.lock() = PrefillTuner::default();
        *self.k_tuner.lock() = KTuner::default();
        self.kv_pool.radix_cache.clear();
        self.sync_spec_view();
        self.log_drafter();
    }

    /// Replaces the speculative drafter (`None` turns speculation off); sequences that started
    /// before keep decoding as they were. `GIBBS_SPEC=off` keeps it off.
    pub fn set_drafter(&self, drafter: Option<Arc<dyn Drafter>>) {
        *self.drafter.write() = drafter.filter(|_| spec_enabled());
        *self.k_tuner.lock() = KTuner::default();
        self.sync_spec_view();
        self.log_drafter();
    }

    /// Current speculative decoding & online noise drafter tuning knobs.
    pub fn draft_knobs(&self) -> DraftKnobs {
        self.draft_knobs.snapshot()
    }

    /// Shared atomic handle to the live draft knobs.
    pub fn live_draft_knobs(&self) -> Arc<LiveDraftKnobs> {
        self.draft_knobs.clone()
    }

    /// Updates [`DraftKnobs`] live and reconfigures the active drafter when the mode changes.
    pub fn set_draft_knobs(&self, knobs: DraftKnobs) {
        let prev = self.draft_knobs.snapshot();
        let next = knobs.sanitized();
        self.draft_knobs.store(next);
        if prev.mode != next.mode || self.drafter.read().is_none() != (next.mode == DraftMode::Off)
        {
            let model = self.model.read().clone();
            *self.drafter.write() = resolve_drafter(model.as_ref(), &self.draft_knobs);
            *self.k_tuner.lock() = KTuner::default();
            self.log_drafter();
        } else if prev.max_k != next.max_k {
            *self.k_tuner.lock() = KTuner::default();
        }
        self.sync_spec_view();
    }

    /// What [`KTuner`] measured for each `k`.
    pub fn k_tuner_rates(&self) -> Vec<spec::KRate> {
        self.k_tuner.lock().rates_summary()
    }

    /// Ends every in-flight sequence with `reason`; their streams close with an error.
    pub fn abort_all(&self, reason: &str) {
        let mut active = self.active_sequences.write();
        let mut waiting = self.waiting_queue.write();
        self.fail_all(&mut active, &mut waiting, reason);
    }

    fn fail_all(&self, active: &mut Vec<Sequence>, waiting: &mut VecDeque<Sequence>, reason: &str) {
        let n = active.len() + waiting.len();
        for seq in active.drain(..).chain(waiting.drain(..)) {
            let _ = seq.finish.set(FinishReason::Error(reason.to_string()));
            for &b in &seq.allocated_blocks {
                self.kv_pool.release_block(b);
            }
        }
        if n > 0 {
            self.telemetry
                .log_warn(&format!("{n} in-flight request(s) aborted: {reason}"));
        }
    }

    /// Builds prompt ids with the current model's tokenizer (via `encode`, e.g. template + encode) and enqueues
    /// them atomically with respect to `swap_model` (both hold `waiting_queue`), so a sequence never carries
    /// another vocabulary's ids. Returns the tokenizer to decode the stream with and the prompt token count.
    pub fn submit_with(
        &self,
        encode: impl FnOnce(&Tokenizer) -> Vec<u32>,
    ) -> (u64, UnboundedReceiver<u32>, Arc<Tokenizer>, usize) {
        let (id, rx, tok, n, _) = self.submit_with_params(SamplingParams::default(), encode);
        (id, rx, tok, n)
    }

    /// `submit_with` honoring per-request sampling params; `max_new_tokens` is clamped so prompt plus
    /// output fits the model's context window. The returned `FinishSignal` is final once the stream
    /// closes; a prompt the model can't prepare closes at once with that error.
    pub fn submit_with_params(
        &self,
        params: SamplingParams,
        encode: impl FnOnce(&Tokenizer) -> Vec<u32>,
    ) -> (
        u64,
        UnboundedReceiver<u32>,
        Arc<Tokenizer>,
        usize,
        FinishSignal,
    ) {
        self.submit_with_media(params, &[], encode)
            .unwrap_or_else(|e| {
                let id = self.next_seq_id.fetch_add(1, Ordering::Relaxed);
                let finish = FinishSignal::default();
                let _ = finish.set(FinishReason::Error(e));
                (id, unbounded_channel().1, self.tokenizer(), 0, finish)
            })
    }

    /// [`Self::submit_with_params`] with images/audio for the prompt (one template placeholder per
    /// item, in order). Errs, before anything is queued, when the model can't take the media or
    /// was swapped while the prompt was prepared. Preparing (e.g. encoding images) runs outside
    /// the queue lock so it never stalls a step.
    pub fn submit_with_media(
        &self,
        params: SamplingParams,
        media: &[MediaInput],
        encode: impl FnOnce(&Tokenizer) -> Vec<u32>,
    ) -> Result<
        (
            u64,
            UnboundedReceiver<u32>,
            Arc<Tokenizer>,
            usize,
            FinishSignal,
        ),
        String,
    > {
        self.submit_cancellable(params, media, None, encode)
    }

    /// [`Self::submit_with_media`]; `cancel` (set when the client goes away) stops a model that
    /// decodes inside `prepare` (seq2seq beam search).
    pub fn submit_cancellable(
        &self,
        mut params: SamplingParams,
        media: &[MediaInput],
        cancel: Option<Arc<std::sync::atomic::AtomicBool>>,
        encode: impl FnOnce(&Tokenizer) -> Vec<u32>,
    ) -> Result<
        (
            u64,
            UnboundedReceiver<u32>,
            Arc<Tokenizer>,
            usize,
            FinishSignal,
        ),
        String,
    > {
        let ctx = PrepareCtx {
            max_new_tokens: params.max_new_tokens,
            cancel,
        };
        let (tokenizer, model, ids, kv_cache) = self.prepare_prompt(media, &ctx, encode)?;
        let n = ids.len();
        if let Some(ctx) = model.as_ref().and_then(ExecutableModel::context_len) {
            params.max_new_tokens = params.max_new_tokens.min(ctx.saturating_sub(n));
        }
        if let Some((t, p)) = tokenizer.sampling_defaults {
            params.temperature.get_or_insert(t);
            params.top_p.get_or_insert(p);
        }
        let mut waiting = self.waiting_queue.write();
        if !Arc::ptr_eq(&tokenizer, &self.tokenizer()) {
            return Err("model swapped while the prompt was prepared".into());
        }
        let id = self.next_seq_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = unbounded_channel();
        let mut seq = Sequence::new(id, ids);
        seq.params = params;
        seq.token_sender = Some(tx);
        seq.state = SequenceState::Waiting;
        seq.kv_cache = kv_cache;
        let finish = seq.finish.clone();
        waiting.push_back(seq);
        *self.work_signal.0.lock() = true;
        drop(waiting);
        self.work_signal.1.notify_one();
        Ok((id, rx, tokenizer, n, finish))
    }

    /// Prompt ids the model would prefill for `encode` + `media` (images/audio expanded), without
    /// queueing or encoding media: the Anthropic `count_tokens` path.
    pub fn prompt_len(
        &self,
        media: &[MediaInput],
        encode: impl FnOnce(&Tokenizer) -> Vec<u32>,
    ) -> Result<usize, String> {
        let (tokenizer, model) = self.model_pair();
        let ids = Self::encode_nonempty(&tokenizer, encode);
        match &model {
            Some(m) => m.prompt_len(ids, media),
            None if media.is_empty() => Ok(ids.len()),
            None => Err("no model loaded to take image or audio input".into()),
        }
    }

    /// `swap_model` replaces both under the queue's write lock: read them as one pair.
    fn model_pair(&self) -> (Arc<Tokenizer>, Option<ExecutableModel>) {
        let _queue = self.waiting_queue.read();
        (self.tokenizer(), self.model.read().clone())
    }

    fn encode_nonempty(
        tokenizer: &Tokenizer,
        encode: impl FnOnce(&Tokenizer) -> Vec<u32>,
    ) -> Vec<u32> {
        let mut ids = encode(tokenizer);
        if ids.is_empty() {
            ids.push(1);
        }
        ids
    }

    /// Encodes the prompt and runs the model's `prepare` against one consistent
    /// (tokenizer, model) pair.
    fn prepare_prompt(
        &self,
        media: &[MediaInput],
        ctx: &PrepareCtx,
        encode: impl FnOnce(&Tokenizer) -> Vec<u32>,
    ) -> Result<
        (
            Arc<Tokenizer>,
            Option<ExecutableModel>,
            Vec<u32>,
            Option<SeqState>,
        ),
        String,
    > {
        let (tokenizer, model) = self.model_pair();
        let ids = Self::encode_nonempty(&tokenizer, encode);
        let (ids, kv_cache) = match &model {
            Some(m) => m.prepare(ids, media, ctx)?,
            None if media.is_empty() => (ids, None),
            None => return Err("no model loaded to take image or audio input".into()),
        };
        Ok((tokenizer, model, ids, kv_cache))
    }

    /// Submit request and return sequence ID plus live token streaming receiver
    pub fn submit_request_stream(&self, prompt_tokens: Vec<u32>) -> (u64, UnboundedReceiver<u32>) {
        let id = self.next_seq_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = unbounded_channel();
        // The loaded model builds the sequence state; a refusal leaves none, so the sequence
        // aborts in prefill with the reason logged here.
        let (ids, kv_cache) = match self.model() {
            Some(m) => match m.prepare(prompt_tokens.clone(), &[], &PrepareCtx::default()) {
                Ok(r) => r,
                Err(e) => {
                    self.telemetry
                        .log_warn(&format!("sequence {id}: cannot prepare the prompt: {e}"));
                    (prompt_tokens, None)
                }
            },
            None => (prompt_tokens, None),
        };
        let mut seq = Sequence::new(id, ids);
        seq.kv_cache = kv_cache;
        seq.token_sender = Some(tx);
        seq.state = SequenceState::Waiting;
        let mut waiting = self.waiting_queue.write();
        waiting.push_back(seq);
        *self.work_signal.0.lock() = true;
        drop(waiting);
        self.work_signal.1.notify_one();
        (id, rx)
    }

    /// Waits up to `timeout` for a request to be admitted into `waiting_queue`.
    pub fn wait_for_work(&self, timeout: Duration) {
        let mut ready = self.work_signal.0.lock();
        if !*ready {
            self.work_signal.1.wait_for(&mut ready, timeout);
        }
        *ready = false;
    }

    /// Runs one request to completion on this thread (CLI `--prompt`, bench): the prompt goes
    /// through the tokenizer's chat template when it declares one, sampling uses `temperature`
    /// and [`DEFAULT_TOP_P`]. The prompt is prefilled in one call (the scheduler stays in that
    /// mode). Prints the prefill line unless the TUI owns the terminal.
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<Generation, String> {
        self.prefill_tuner.lock().whole_prompt = true;
        let params = SamplingParams {
            temperature: Some(temperature),
            top_p: Some(DEFAULT_TOP_P),
            max_new_tokens,
            top_logprobs: None,
        };
        let (_, mut rx, tok, _, finish) =
            self.submit_with_media(params, &[], |t| match &t.chat_template {
                Some(_) => t.encode(&t.apply_chat_template(&[("user", prompt)])),
                None => t.encode(prompt),
            })?;
        let (mut prefill, mut decode) = ((Duration::ZERO, 0), (Duration::ZERO, 0));
        let mut out = Vec::new();
        'run: loop {
            let s = self.step();
            prefill = (prefill.0 + s.prefill.0, prefill.1 + s.prefill.1);
            decode = (decode.0 + s.decode.0, decode.1 + s.decode.1);
            loop {
                match rx.try_recv() {
                    Ok(t) => out.push(t),
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break 'run,
                }
            }
        }
        finish_outcome(&finish)?;
        if !crate::telemetry::TUI_OWNS_TERMINAL.load(Ordering::Relaxed) {
            println!(
                "⚡ Prefill: {} tokens in {:.2}ms ({:.1} tok/s)",
                prefill.1,
                prefill.0.as_secs_f64() * 1000.0,
                prefill.1 as f64 / prefill.0.as_secs_f64().max(1e-9)
            );
        }
        Ok(Generation {
            text: tok.decode(&out),
            prefill,
            decode,
        })
    }

    pub fn step(&self) -> BatchStepOutput {
        let step_started = Instant::now();
        // Lock order (active, waiting, model) matches `swap_model`, so a step never mixes models.
        let mut active = self.active_sequences.write();
        let mut waiting = self.waiting_queue.write();
        let model = self.model.read().clone();
        let tokenizer = self.tokenizer();
        let drafter = self.drafter.read().clone();
        // Sequences end at the model's context length: its KV cache holds no more positions.
        let context_len = model
            .as_ref()
            .and_then(ExecutableModel::context_len)
            .unwrap_or(usize::MAX);
        let current_step = self.step_counter.fetch_add(1, Ordering::Relaxed) as usize;
        let decoding = active.iter().any(|s| s.state == SequenceState::Decoding);
        let prefill_budget = self.prefill_tuner.lock().budget(decoding);

        // 1. Cache-aware promotion of waiting sequences if KV blocks are available:
        // When multiple sequences wait and some share cached prefix tokens, promote cache hits first.
        if waiting.len() > 1 {
            let mut vec: Vec<Sequence> = waiting.drain(..).collect();
            vec.sort_by_key(|s| {
                let hit = self.kv_pool.radix_cache.peek_match_len(&s.prompt_tokens);
                (usize::MAX - hit, s.submitted)
            });
            waiting.extend(vec);
        }
        while let Some(mut seq) = waiting.pop_front() {
            let (mut matched_blocks, mut matched_tokens, snapshot) = self
                .kv_pool
                .radix_cache
                .match_prefix_with_snapshot(&seq.prompt_tokens);
            let max_reusable = seq.prompt_tokens.len().saturating_sub(1);
            let mut restored_tokens = 0usize;
            if drafter.as_ref().is_none_or(|d| d.tap_layers().is_empty()) && max_reusable > 0 {
                if let (Some(m), Some(snap)) = (model.as_ref(), snapshot.as_ref()) {
                    let usable = if snap.has_recurrent {
                        if snap.token_len <= max_reusable {
                            snap.token_len
                        } else {
                            0
                        }
                    } else {
                        matched_tokens.min(snap.token_len).min(max_reusable)
                    };
                    if usable > 0 && m.restore_kv_prefix(&mut seq.kv_cache, snap, usable) {
                        restored_tokens = usable;
                    }
                }
            }
            if restored_tokens == 0 {
                matched_blocks.clear();
                matched_tokens = 0;
            } else {
                matched_tokens = restored_tokens;
                let keep_blocks = (matched_tokens + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
                matched_blocks.truncate(keep_blocks);
                for &b in &matched_blocks {
                    let _ = self.kv_pool.retain_block(b, current_step);
                }
            }

            seq.allocated_blocks = matched_blocks;
            let needed_tokens = seq
                .prompt_tokens
                .len()
                .saturating_sub(matched_tokens)
                .min(prefill_budget);
            let total_initial = matched_tokens + needed_tokens;
            let total_blocks = (total_initial + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
            let needed_blocks = total_blocks.saturating_sub(seq.allocated_blocks.len());

            let mut all_allocated = true;
            for _ in 0..needed_blocks {
                if let Some(block_id) = self.kv_pool.allocate_block(current_step) {
                    seq.allocated_blocks.push(block_id);
                } else {
                    all_allocated = false;
                    break;
                }
            }

            if all_allocated {
                seq.prefilled_tokens = matched_tokens;
                seq.state = SequenceState::Prefilling;
                active.push(seq);
            } else {
                for &b in &seq.allocated_blocks {
                    self.kv_pool.release_block(b);
                }
                waiting.push_front(seq);
                break;
            }
        }

        // Release waiting queue lock cleanly so concurrent ingestion is not blocked during forward compute
        let waiting_len = waiting.len();
        *self.work_signal.0.lock() = false;
        drop(waiting);
        // Show admitted sequences now: on CPU one prefill step can take seconds.
        self.publish(&active, waiting_len, prefill_budget, None);

        // 2. Execute chunked prefill and forward decode across active sequences.
        // Prefill is handed out shortest remaining prompt first, so a short prompt queued behind a
        // long one gets its few tokens (and its first output token) this step. The oldest
        // prefilling sequence keeps up to half the budget, so a stream of short prompts cannot
        // hold a long one back forever.
        active.sort_by_key(|s| match s.state {
            SequenceState::Prefilling => s.prompt_tokens.len().saturating_sub(s.prefilled_tokens),
            _ => 0,
        });
        let oldest = active
            .iter()
            .filter(|s| s.state == SequenceState::Prefilling)
            .min_by_key(|s| s.submitted);
        let mut reserved = oldest.map_or(0, |s| {
            (s.prompt_tokens.len() - s.prefilled_tokens).min(prefill_budget.div_ceil(2))
        });
        let oldest = oldest.map(|s| s.id);
        // ponytail: chunked prefill caps prefill tokens per step with global active lock; fine-grained sequence locks if concurrency scales.
        let mut tokens_gen = 0;

        let mut prefill_budget_left = prefill_budget;
        let (mut prefill_time, mut prefill_tokens) = (Duration::ZERO, 0);
        let (mut decode_time, mut decoded, mut decode_emitted) = (Duration::ZERO, 0, 0);

        for seq in active.iter_mut() {
            if seq.state == SequenceState::Finished {
                continue;
            }
            // Nobody reads the stream any more (client gone, TUI cancel): stop computing it.
            if seq.token_sender.as_ref().is_some_and(|tx| tx.is_closed()) {
                seq.state = SequenceState::Finished;
                continue;
            }

            let Some(m) = &model else {
                // No autoregressive model loaded: end the stream instead of inventing tokens.
                seq.state = SequenceState::Finished;
                continue;
            };

            if seq.state == SequenceState::Prefilling {
                let available = if Some(seq.id) == oldest {
                    reserved = 0;
                    prefill_budget_left
                } else {
                    prefill_budget_left.saturating_sub(reserved)
                };
                if available == 0 {
                    // Step's prefill budget exhausted; keep sequence in Prefilling for next step.
                    continue;
                }

                let remaining = seq.prompt_tokens.len().saturating_sub(seq.prefilled_tokens);
                let chunk_len = remaining.min(available);
                if chunk_len == 0 {
                    seq.state = SequenceState::Decoding;
                } else {
                    // Ensure sufficient KV blocks are allocated for this chunk
                    let needed_capacity = seq.prefilled_tokens + chunk_len;
                    let current_capacity = seq.allocated_blocks.len() * TOKENS_PER_BLOCK;
                    if needed_capacity > current_capacity {
                        let blocks_needed = (needed_capacity - current_capacity + TOKENS_PER_BLOCK
                            - 1)
                            / TOKENS_PER_BLOCK;
                        for _ in 0..blocks_needed {
                            if let Some(b) = self.kv_pool.allocate_block(current_step) {
                                seq.allocated_blocks.push(b);
                            } else {
                                break;
                            }
                        }
                    }

                    if seq.spec.is_none() {
                        if let Some(d) = &drafter {
                            let nanos = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map_or(0, |t| t.as_nanos() as u64);
                            let seed = nanos ^ seq.id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                            match SpecSeq::start(m, d.clone(), &mut seq.kv_cache, seed) {
                                Ok(s) => seq.spec = Some(s),
                                Err(e) => self.telemetry.log_warn(&format!(
                                    "sequence {}: no speculative decoding: {e}",
                                    seq.id
                                )),
                            }
                        }
                    }
                    let is_last = (seq.prefilled_tokens + chunk_len) >= seq.prompt_tokens.len();
                    let chunk =
                        &seq.prompt_tokens[seq.prefilled_tokens..seq.prefilled_tokens + chunk_len];
                    let started = Instant::now();
                    let next_tok_opt = match m.step_prefill_chunk(
                        chunk,
                        seq.prefilled_tokens,
                        is_last,
                        &seq.prompt_tokens,
                        seq.params.temperature.unwrap_or(DEFAULT_TEMPERATURE),
                        seq.params.top_p.unwrap_or(DEFAULT_TOP_P),
                        &mut seq.kv_cache,
                    ) {
                        Ok(t) => t,
                        Err(e) => {
                            self.telemetry
                                .log_warn(&format!("sequence {} aborted in prefill: {e}", seq.id));
                            let _ = seq
                                .finish
                                .set(FinishReason::Error(format!("prefill failed: {e}")));
                            seq.state = SequenceState::Finished;
                            continue;
                        }
                    };
                    prefill_time += started.elapsed();
                    prefill_tokens += chunk_len;
                    seq.prefilled_tokens += chunk_len;
                    prefill_budget_left = prefill_budget_left.saturating_sub(chunk_len);

                    if is_last {
                        seq.state = SequenceState::Decoding;
                        let snap = m
                            .export_kv_snapshot(&seq.kv_cache, seq.prompt_tokens.len())
                            .map(Arc::new);
                        self.kv_pool.radix_cache.insert_prefix_with_snapshot(
                            &seq.prompt_tokens,
                            &seq.allocated_blocks,
                            snap,
                        );
                        if seq.params.max_new_tokens == 0 {
                            let _ = seq.finish.set(FinishReason::Length);
                            seq.state = SequenceState::Finished;
                        } else if let Some(next_tok) = next_tok_opt {
                            if tokenizer.ends_generation(next_tok) {
                                seq.state = SequenceState::Finished;
                            } else {
                                seq.emit(&[next_tok], &self.telemetry);
                                tokens_gen += 1;
                                if seq.total_tokens() >= context_len
                                    || seq.output_tokens.len() >= seq.params.max_new_tokens
                                {
                                    let _ = seq.finish.set(FinishReason::Length);
                                    seq.state = SequenceState::Finished;
                                }
                            }
                        }
                    }
                }
            } else if seq.state == SequenceState::Decoding {
                let temp = seq.params.temperature.unwrap_or(DEFAULT_TEMPERATURE);
                let top_p = seq.params.top_p.unwrap_or(DEFAULT_TOP_P);
                let mut all = seq.prompt_tokens.clone();
                all.extend_from_slice(&seq.output_tokens);
                // Draft no more than the sequence may still emit (the target adds one) or hold.
                let knobs_snap = self.draft_knobs.snapshot();
                let spec_k = seq.spec.as_ref().map(|sp| {
                    self.k_tuner
                        .lock()
                        .choose_with_knobs(sp.max_k(), &knobs_snap)
                        .min(
                            seq.params
                                .max_new_tokens
                                .saturating_sub(seq.output_tokens.len() + 1),
                        )
                        .min(context_len.saturating_sub(all.len()))
                });
                // Ensure sufficient KV blocks are allocated for the positions this step may add.
                let needed_capacity = seq.total_tokens() + 1 + spec_k.unwrap_or(0);
                while needed_capacity > seq.allocated_blocks.len() * TOKENS_PER_BLOCK {
                    match self.kv_pool.allocate_block(current_step) {
                        Some(b) => seq.allocated_blocks.push(b),
                        None => break,
                    }
                }

                let started = Instant::now();
                let logprobs = seq.params.top_logprobs;
                let result = match (spec_k, seq.spec.as_mut()) {
                    (Some(k), Some(sp)) => {
                        let vocab = tokenizer.vocab_size();
                        spec::decode(
                            m,
                            sp,
                            &all,
                            &mut seq.kv_cache,
                            temp,
                            top_p,
                            vocab,
                            k,
                            logprobs,
                        )
                        .map(|s| {
                            if s.synced {
                                let msg = self.k_tuner.lock().record_with_knobs(
                                    k,
                                    s.tokens.len(),
                                    s.drafted,
                                    s.accepted,
                                    started.elapsed(),
                                    &knobs_snap,
                                );
                                if let Some(msg) = msg {
                                    self.telemetry.log_info(&msg);
                                }
                            }
                            if s.drafted > 0 {
                                self.telemetry
                                    .record_spec(s.drafted, s.accepted, s.tokens.len());
                            }
                            if !s.logprobs.is_empty() {
                                if let Some(tx) = self.logprob_tx.read().as_ref() {
                                    for info in s.logprobs {
                                        let _ = tx.send(info);
                                    }
                                }
                            }
                            s.tokens
                        })
                    }
                    _ => {
                        let last = all.last().copied().unwrap_or(1);
                        let streamed = if logprobs.is_none() {
                            m.stream_token(
                                last,
                                all.len(),
                                seq.params
                                    .max_new_tokens
                                    .saturating_sub(seq.output_tokens.len())
                                    .min(context_len.saturating_sub(all.len())),
                                temp,
                                top_p,
                                &all,
                                &mut seq.kv_cache,
                            )
                        } else {
                            Ok(None)
                        };
                        match streamed {
                            Ok(Some(t)) => Ok(vec![t]),
                            Ok(None) => m
                                .step_token_with_kv(
                                    &all,
                                    last,
                                    all.len(),
                                    temp,
                                    top_p,
                                    logprobs,
                                    &mut seq.kv_cache,
                                )
                                .map(|(tok, info)| {
                                    if let (Some(info), Some(tx)) =
                                        (info, self.logprob_tx.read().as_ref())
                                    {
                                        let _ = tx.send(info);
                                    }
                                    vec![tok]
                                }),
                            Err(e) => Err(e),
                        }
                    }
                };
                let tokens = match result {
                    Ok(t) => t,
                    Err(e) => {
                        self.telemetry
                            .log_warn(&format!("sequence {} aborted in decode: {e}", seq.id));
                        let _ = seq
                            .finish
                            .set(FinishReason::Error(format!("decode failed: {e}")));
                        seq.state = SequenceState::Finished;
                        continue;
                    }
                };
                decode_time += started.elapsed();
                decoded += 1;

                // Emit up to EOS, max_new_tokens or the context end, whichever comes first.
                let room = seq
                    .params
                    .max_new_tokens
                    .saturating_sub(seq.output_tokens.len())
                    .min(context_len.saturating_sub(seq.total_tokens()));
                let eos = tokens.iter().position(|&t| tokenizer.ends_generation(t));
                let n = eos.unwrap_or(tokens.len()).min(room);
                if n > 0 {
                    let itl = seq.emit(&tokens[..n], &self.telemetry);
                    tokens_gen += n;
                    decode_emitted += n;
                    // A plain decode token (no drafter, one forward) against the run behind it.
                    if let (Some(itl), None, 1) = (itl, spec_k, n) {
                        self.record_gap(itl);
                    }
                }
                if seq.total_tokens() >= context_len
                    || seq.output_tokens.len() >= seq.params.max_new_tokens
                {
                    let _ = seq.finish.set(FinishReason::Length);
                    seq.state = SequenceState::Finished;
                } else if eos == Some(n) {
                    seq.state = SequenceState::Finished;
                }
            }
        }
        self.prefill_tuner
            .lock()
            .record((prefill_time, prefill_tokens), (decode_time, decoded));
        if tokens_gen > 0 {
            self.telemetry.record_tokens(tokens_gen);
        }

        // 4. Release blocks for finished sequences
        active.retain_mut(|seq| {
            if seq.state == SequenceState::Finished {
                if let Some(m) = &model {
                    let _ = m.stream_end(&mut seq.kv_cache);
                    if seq.output_tokens.len() > 1 {
                        let committed_out = seq.output_tokens.len() - 1;
                        let total_committed = seq.prompt_tokens.len() + committed_out;
                        if let Some(snap) = m.export_kv_snapshot(&seq.kv_cache, total_committed) {
                            let mut full = Vec::with_capacity(total_committed);
                            full.extend_from_slice(&seq.prompt_tokens);
                            full.extend_from_slice(&seq.output_tokens[..committed_out]);
                            let n_blocks = ((total_committed + TOKENS_PER_BLOCK - 1)
                                / TOKENS_PER_BLOCK)
                                .min(seq.allocated_blocks.len());
                            self.kv_pool.radix_cache.insert_prefix_with_snapshot(
                                &full,
                                &seq.allocated_blocks[..n_blocks],
                                Some(Arc::new(snap)),
                            );
                        }
                    }
                }
                for &b in &seq.allocated_blocks {
                    self.kv_pool.release_block(b);
                }
                self.telemetry.record_request_completed();
                false
            } else {
                true
            }
        });

        let busy = (prefill_tokens + decoded > 0).then(|| step_started.elapsed());
        self.publish(&active, waiting_len, prefill_budget, busy);

        BatchStepOutput {
            tokens_generated: tokens_gen,
            active_sequences_count: active.len(),
            prefill: (prefill_time, prefill_tokens),
            decode: (decode_time, decode_emitted),
        }
    }

    /// Publishes the dashboard view; `busy` (the step's duration) only when it ran a forward pass.
    fn publish(&self, active: &[Sequence], waiting: usize, budget: usize, busy: Option<Duration>) {
        let mut view = self.view.lock();
        view.seqs.clear();
        view.seqs.extend(active.iter().map(|s| SeqView {
            id: s.id,
            state: s.state,
            prompt: s.prompt_tokens.len(),
            prefilled: s.prefilled_tokens,
            generated: s.output_tokens.len(),
            max_new: s.params.max_new_tokens,
            submitted: s.submitted,
        }));
        view.waiting = waiting;
        view.prefill_budget = budget;
        view.spec = self.spec_view();
        if let Some(d) = busy {
            view.busy_step = d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The synthetic llama of [`crate::arch::testing::synthetic_llama`] (vocab 256, a tokenizer
    /// without special ids: generation stops only on `max_new_tokens` or the context) with
    /// context `ctx`, on `dev`.
    fn spec_model(ctx: usize, dev: Device) -> ExecutableModel {
        let dir = tempfile::tempdir().unwrap();
        let cfg = serde_json::json!({"max_position_embeddings": ctx});
        crate::arch::testing::synthetic_lm(dir.path(), cfg, dev)
    }

    fn cpu_exec(ctx: usize) -> ExecutableModel {
        spec_model(ctx, Device::Cpu)
    }

    /// Room for prompts spanning several prefill budgets.
    const LONG: usize = 4096;
    const SHORT: usize = 512;

    fn argmax(v: &[f32]) -> u32 {
        (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b }) as u32
    }

    /// While a sequence decodes, prompt prefill is cut to a measured budget (and the decoding
    /// sequence still gets a token every step); once it finishes, the budget grows back.
    #[test]
    fn test_prefill_budget_tracks_decoding_load() {
        let s = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(LONG)),
        );
        let greedy = |n| SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: n,
            top_logprobs: None,
        };
        let (_, mut rx1, _, _, _) = s.submit_with_params(greedy(40), |_| vec![1, 2, 3, 4]);
        s.step();
        assert!(rx1.try_recv().is_ok());

        let (_, _long, _, _, _) = s.submit_with_params(greedy(1), |_| vec![42; 3000]);
        let (mut prefilled, mut busiest) = (0, 0);
        while s
            .active_sequences
            .read()
            .iter()
            .any(|q| q.state == SequenceState::Decoding)
        {
            s.step();
            let chunk = s.prefill_chunk();
            busiest = busiest.max(chunk);
            let now = s.active_sequences.read().last().unwrap().prefilled_tokens;
            assert_eq!(now - prefilled, chunk, "prefill must use the whole budget");
            prefilled = now;
            assert!(
                rx1.try_recv().is_ok(),
                "decoding sequence starved by prefill"
            );
        }
        s.step();
        let idle = s.prefill_chunk();
        assert!(
            idle > busiest,
            "budget {idle} with nothing decoding should exceed {busiest} while decoding"
        );
    }

    /// A short prompt queued behind a long one is prefilled first and answers in the same step.
    #[test]
    fn test_short_prompt_not_stuck_behind_long_prefill() {
        let s = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(LONG)),
        );
        let greedy = SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: 4,
            top_logprobs: None,
        };
        let (_, _long, _, _, _) = s.submit_with_params(greedy, |_| vec![42; 3000]);
        let (_, mut short, _, _, _) = s.submit_with_params(greedy, |_| vec![1, 2, 3, 4]);
        s.step();
        assert!(
            short.try_recv().is_ok(),
            "short prompt waited behind the long prefill"
        );
    }

    /// The oldest prompt keeps prefilling while newer, shorter prompts keep arriving.
    #[test]
    fn test_long_prompt_not_starved_by_short_prompts() {
        let s = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(LONG)),
        );
        let greedy = SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: 1,
            top_logprobs: None,
        };
        let (long, _long, _, _, _) = s.submit_with_params(greedy, |_| vec![42; 3000]);
        let (mut streams, mut prefilled) = (Vec::new(), 0);
        for step in 0..6 {
            // Each newcomer has less left to prefill than the long prompt, so it sorts first.
            streams.push(s.submit_with_params(greedy, |_| vec![7; 2000]).1);
            s.step();
            // A fast host may finish the long prefill (and retire it) within these steps.
            let now = s
                .view()
                .seqs
                .iter()
                .find(|q| q.id == long)
                .map_or(3000, |q| q.prefilled);
            assert!(
                now > prefilled,
                "step {step}: the long prompt got no prefill"
            );
            if now >= 3000 {
                break;
            }
            prefilled = now;
        }
    }

    /// Dropping the token receiver (client gone, TUI cancel) ends the sequence on the next step.
    #[test]
    fn test_dropped_receiver_frees_slot() {
        let s = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(SHORT)),
        );
        let params = SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: 1000,
            top_logprobs: None,
        };
        let (_, rx, _, _, _) = s.submit_with_params(params, |_| vec![1, 2, 3]);
        s.step();
        assert_eq!(s.view().seqs.len(), 1);
        drop(rx);
        s.step();
        assert!(s.active_sequences.read().is_empty() && s.waiting_queue.read().is_empty());
    }

    /// Counting a prompt (Anthropic count_tokens) neither queues a sequence nor takes an id.
    #[test]
    fn test_prompt_len_queues_nothing() {
        let s = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(SHORT)),
        );
        assert_eq!(s.prompt_len(&[], |_| vec![1, 2, 3]).unwrap(), 3);
        assert!(s.waiting_queue.read().is_empty());
        let (id, ..) = s.submit_with_params(SamplingParams::default(), |_| vec![1, 2, 3]);
        assert_eq!(id, 0, "prompt_len consumed a sequence id");
        assert_eq!(s.waiting_queue.read().len(), 1);
    }

    /// A model swap fails in-flight requests instead of closing them like finished ones.
    #[test]
    fn test_swap_model_fails_in_flight_requests() {
        let s = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(SHORT)),
        );
        let params = SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: 1000,
            top_logprobs: None,
        };
        let (_, _rx, _, _, running) = s.submit_with_params(params, |_| vec![1, 2, 3]);
        s.step();
        let (_, _rx2, _, _, queued) = s.submit_with_params(params, |_| vec![4, 5]);
        s.swap_model("other", cpu_exec(SHORT));
        assert!(finish_outcome(&running).is_err() && finish_outcome(&queued).is_err());
    }

    #[test]
    fn test_prompt_ingestion_lock_cleanliness_and_state() {
        let kv_pool = Arc::new(PagedKVPool::new(64));
        let scheduler = ContinuousBatchScheduler::new("", kv_pool, None);

        // submit_request_stream must drop waiting lock cleanly
        let (_, _rx) = scheduler.submit_request_stream(vec![10, 20, 30]);

        // Verify waiting lock can immediately be acquired (no leaked lock)
        let waiting = scheduler.waiting_queue.try_write();
        assert!(
            waiting.is_some(),
            "waiting_queue lock must be released cleanly"
        );
        let w = waiting.unwrap();
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].state, SequenceState::Waiting);
        assert_eq!(w[0].prefilled_tokens, 0);
    }

    /// Submitting a request while the server loop is idle wakes `wait_for_work` immediately and
    /// the next `step` admits it active.
    #[test]
    fn test_idle_wait_wakes_immediately_on_submission() {
        let s = Arc::new(ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(SHORT)),
        ));
        assert_eq!(s.step().active_sequences_count, 0);
        let s2 = s.clone();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let b2 = barrier.clone();
        let waiter = std::thread::spawn(move || {
            b2.wait();
            let t0 = Instant::now();
            s2.wait_for_work(Duration::from_secs(5));
            let waited = t0.elapsed();
            let out = s2.step();
            (waited, out.active_sequences_count)
        });
        barrier.wait();
        std::thread::sleep(Duration::from_millis(5));
        let params = SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: 4,
            top_logprobs: None,
        };
        let (_, mut rx, ..) = s.submit_with_params(params, |_| vec![1, 2, 3]);
        let (waited, active) = waiter.join().unwrap();
        assert!(waited < Duration::from_millis(200), "waited {waited:?}");
        assert_eq!(active, 1);
        assert!(rx.try_recv().is_ok());
    }

    fn run_to_completion(
        scheduler: &ContinuousBatchScheduler,
        mut rx: UnboundedReceiver<u32>,
    ) -> Vec<u32> {
        for _ in 0..1000 {
            scheduler.step();
            if scheduler.active_sequences.read().is_empty()
                && scheduler.waiting_queue.read().is_empty()
            {
                break;
            }
        }
        let mut out = Vec::new();
        while let Ok(t) = rx.try_recv() {
            out.push(t);
        }
        out
    }

    #[test]
    fn test_max_new_tokens_stops_exactly() {
        for n in [1usize, 7, 100] {
            let scheduler = ContinuousBatchScheduler::new(
                "",
                Arc::new(PagedKVPool::new(512)),
                Some(cpu_exec(SHORT)),
            );
            let params = SamplingParams {
                temperature: Some(0.0),
                top_p: None,
                max_new_tokens: n,
                top_logprobs: None,
            };
            let (_, rx, _, _, finish) = scheduler.submit_with_params(params, |_| vec![1, 2, 3]);
            assert_eq!(
                run_to_completion(&scheduler, rx).len(),
                n,
                "max_new_tokens={n}"
            );
            assert_eq!(
                finish.get(),
                Some(&FinishReason::Length),
                "max_new_tokens={n}"
            );
        }
    }

    /// Generation ends at the model's context length (its KV cache capacity), not past it.
    #[test]
    fn test_generation_stops_at_model_context_length() {
        let model = cpu_exec(SHORT);
        let ctx = model.context_len().unwrap();
        assert_eq!(ctx, SHORT);
        let scheduler =
            ContinuousBatchScheduler::new("", Arc::new(PagedKVPool::new(512)), Some(model));
        // Unclamped submit path: only the step's context check can stop this sequence.
        let (_, rx) = scheduler.submit_request_stream(vec![7u32; ctx - 12]);
        assert_eq!(run_to_completion(&scheduler, rx).len(), 12);
    }

    #[test]
    fn test_temperature_zero_picks_argmax() {
        let prompt = [5u32, 9, 13];
        // Every load of the synthetic id has the same weights: the reference is a second one.
        let want = argmax(&cpu_exec(SHORT).prompt_logits(prompt.to_vec()).unwrap()[..256]);
        let scheduler = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(SHORT)),
        );
        let params = SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: 1,
            top_logprobs: None,
        };
        let (_, rx, _, _, _) = scheduler.submit_with_params(params, |_| prompt.to_vec());
        assert_eq!(run_to_completion(&scheduler, rx), vec![want]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_metal_scheduler_serves_max_new_tokens() {
        if !crate::hardware::available_devices().contains(&Device::Metal) {
            return;
        }
        let model = spec_model(SHORT, Device::Metal);
        assert_eq!(model.llm_backend(), Device::Metal);
        let scheduler =
            ContinuousBatchScheduler::new("", Arc::new(PagedKVPool::new(512)), Some(model));
        let params = SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: 12,
            top_logprobs: None,
        };
        let (_, rx, _, _, _) = scheduler.submit_with_params(params, |_| vec![5, 9, 13, 2, 77]);
        let out = run_to_completion(&scheduler, rx);
        assert_eq!(out.len(), 12);
        assert!(out.iter().all(|&t| t < 256));
    }

    /// Chunked prefill, the one-call prefill + greedy step and a decode step agree with whole-prompt
    /// logits on `dev`; on Metal the logits also match the CPU model's (a 20-row prefill takes the
    /// batched path).
    fn prefill_and_decode_parity(dev: Device) {
        let m = spec_model(SHORT, dev);
        let cpu = cpu_exec(SHORT);
        let prompt = vec![5u32, 9, 13, 2];
        let close = |what: &str, got: &[f32], want: &[f32]| {
            let d = got
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(d < 1e-3, "{what}: max logit difference {d}");
        };
        let want = cpu.prompt_logits(prompt.clone()).unwrap();
        close(
            "prompt logits vs CPU",
            &m.prompt_logits(prompt.clone()).unwrap(),
            &want,
        );
        // A picked token is right when its reference logit ties the argmax's (synthetic weights
        // can tie the top logits within an ulp).
        let tied = |what: &str, tok: u32, logits: &[f32]| {
            let best = logits[argmax(&logits[..256]) as usize];
            assert!(
                (logits[tok as usize] - best).abs() < 1e-4,
                "{what}: token {tok}"
            );
        };
        let (mut st, mut st2) = (
            m.prepare(prompt.clone(), &[], &PrepareCtx::default())
                .unwrap()
                .1,
            m.prepare(prompt.clone(), &[], &PrepareCtx::default())
                .unwrap()
                .1,
        );
        assert_eq!(
            m.step_prefill_chunk(&prompt[..2], 0, false, &prompt, 0.0, 1.0, &mut st)
                .unwrap(),
            None
        );
        let t = m
            .step_prefill_chunk(&prompt[2..], 2, true, &prompt, 0.0, 1.0, &mut st)
            .unwrap()
            .unwrap();
        tied("chunked prefill", t, &want);
        // Prefill all but the last token, then the greedy step on it.
        m.prefill_logits(&prompt[..3], 0, &mut st2).unwrap();
        let (next, _) = m
            .step_token_with_kv(&prompt, prompt[3], prompt.len(), 0.0, 1.0, None, &mut st2)
            .unwrap();
        tied("greedy step", next, &want);
        let full = [&prompt[..], &[next]].concat();
        let (dec, _) = m
            .step_token_with_kv(&full, next, full.len(), 0.0, 1.0, None, &mut st2)
            .unwrap();
        tied(
            "decode step",
            dec,
            &cpu.prompt_logits(full.clone()).unwrap(),
        );
        let long: Vec<u32> = (0..20).map(|i| (i * 7 + 3) % 256).collect();
        close(
            "20-row prefill vs CPU",
            &m.prompt_logits(long.clone()).unwrap(),
            &cpu.prompt_logits(long).unwrap(),
        );
    }

    #[test]
    fn test_program_scheduler_prefill_and_decode_parity() {
        prefill_and_decode_parity(Device::Cpu);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_program_scheduler_metal_parity() {
        if crate::hardware::available_devices().contains(&Device::Metal) {
            prefill_and_decode_parity(Device::Metal);
        }
    }

    /// OnlineNoiseDrafter (`DraftMode::Noise` and `DraftMode::Suffix`) produces bit-exact greedy
    /// tokens identical to plain decoding (`DraftMode::Off`).
    #[test]
    fn test_online_noise_drafter_matches_plain_greedy_decode() {
        let prompt = vec![5u32, 9, 13, 5, 9, 13, 2];
        let params = SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: 16,
            top_logprobs: None,
        };
        let plain_sched = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(SHORT)),
        );
        plain_sched.set_draft_knobs(DraftKnobs {
            mode: DraftMode::Off,
            ..DraftKnobs::default()
        });
        let (_, rx_plain, ..) = plain_sched.submit_with_params(params, |_| prompt.clone());
        let want = run_to_completion(&plain_sched, rx_plain);
        assert_eq!(want.len(), 16);

        for mode in [DraftMode::Auto, DraftMode::Noise, DraftMode::Suffix] {
            let spec_sched = ContinuousBatchScheduler::new(
                "",
                Arc::new(PagedKVPool::new(512)),
                Some(cpu_exec(SHORT)),
            );
            assert!(
                spec_sched.view().spec.is_some(),
                "DraftMode::Auto should automatically prepare OnlineNoiseDrafter on load"
            );
            spec_sched.set_draft_knobs(DraftKnobs {
                mode,
                max_k: 3,
                suffix_min_match: 2,
                ..DraftKnobs::default()
            });
            assert!(spec_sched.view().spec.is_some());
            let (_, rx_spec, ..) = spec_sched.submit_with_params(params, |_| prompt.clone());
            let got = run_to_completion(&spec_sched, rx_spec);
            assert_eq!(got, want, "mode {mode:?} diverged from plain greedy decode");
            assert!(
                spec_sched.telemetry.get_summary().spec_forwards > 0,
                "mode {mode:?} should run speculative verify steps"
            );
        }
    }

    /// Physical PagedRadixKv prefix reuse skips prefilling the cached prefix and produces
    /// bit-identical tokens to a fresh full-prompt prefill.
    #[test]
    fn test_physical_prefix_cache_skips_prefill_and_matches_fresh_decode() {
        let prefix: Vec<u32> = (0..24).map(|i| 5 + (i % 40) as u32).collect();
        let mut ext = prefix.clone();
        ext.extend_from_slice(&[91, 92, 93, 94, 95, 96]);
        let params = SamplingParams {
            temperature: Some(0.0),
            top_p: None,
            max_new_tokens: 8,
            top_logprobs: None,
        };

        // Reference: fresh scheduler prefills all 30 tokens of `ext` from scratch.
        let fresh = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(SHORT)),
        );
        let (_, rx_fresh, ..) = fresh.submit_with_params(params, |_| ext.clone());
        let want = run_to_completion(&fresh, rx_fresh);
        assert_eq!(want.len(), 8);

        // Cached scheduler: first runs `prefix` (24 tokens), then runs `ext` (30 tokens).
        let cached = ContinuousBatchScheduler::new(
            "",
            Arc::new(PagedKVPool::new(512)),
            Some(cpu_exec(SHORT)),
        );
        let (_, rx1, ..) = cached.submit_with_params(params, |_| prefix.clone());
        let _ = run_to_completion(&cached, rx1);

        let (_, mut rx2, ..) = cached.submit_with_params(params, |_| ext.clone());
        let step1 = cached.step();
        // Only the 6 uncached suffix tokens should be prefilled in step1!
        assert_eq!(
            step1.prefill.1, 6,
            "expected physical prefix cache to skip 24 cached tokens and prefill only 6, got {}",
            step1.prefill.1
        );
        let mut got: Vec<u32> = std::iter::from_fn(|| rx2.try_recv().ok()).collect();
        got.extend(run_to_completion(&cached, rx2));
        assert_eq!(
            got, want,
            "physical prefix reuse output diverged from fresh prefill"
        );
    }
}
