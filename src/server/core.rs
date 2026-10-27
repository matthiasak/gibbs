//! The one generation path every API schema adapts to: (messages, tools, media, sampling,
//! stop) in; text deltas, complete tool calls, and a finish reason out.

use super::tools::{partial_suffix, Piece, ToolCall, ToolFormat, ToolParser};
use super::AppState;
use crate::model::hf_loader::ChatMessage;
use crate::model::native::MediaInput;
use crate::model::tokenizer::Tokenizer;
use crate::scheduler::{finish_outcome, SamplingParams};
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::stream::{self, StreamExt};
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Instant;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};

/// A request error in schema-neutral form; each schema renders its own error body.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub fn bad(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    /// The model does not see the tools.
    None,
    /// The reply must be a tool call (forced by the prompt, checked after generation).
    Required,
    Named(String),
}

pub enum Prompt {
    /// Rendered with the model's chat template.
    Chat {
        messages: Vec<ChatMessage>,
        /// OpenAI shape: `{"type": "function", "function": {"name", "description", "parameters"}}`.
        tools: Vec<Value>,
        tool_choice: ToolChoice,
        /// Extra chat-template variables (OpenAI `chat_template_kwargs`, e.g. `enable_thinking`).
        template_kwargs: Value,
    },
    /// Tokenized as is (`/v1/completions`, `/api/generate` `raw`).
    Raw(String),
}

pub struct GenRequest {
    pub prompt: Prompt,
    pub media: Vec<MediaInput>,
    pub params: SamplingParams,
    pub stop: Vec<String>,
    /// Requests panel label, e.g. `anthropic · claude-cli/2.0`.
    pub label: String,
    /// A chat request's `compact` object, over the saved compaction settings.
    pub compact: Option<crate::compact::CompactionOverride>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Finish {
    Stop,
    Length,
    ToolCalls,
    StopSequence(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Text(String),
    Call(ToolCall),
    Done {
        finish: Finish,
        completion_tokens: usize,
    },
    /// The engine failed or the forced tool call did not come; the stream ends.
    Error(String),
}

pub struct Generation {
    pub id: u64,
    pub prompt_tokens: usize,
    pub events: UnboundedReceiver<Event>,
    /// Set when the chat history was compacted before it was queued.
    pub compaction: Option<Compaction>,
}

/// A compaction applied to a request: its report and the history the model saw.
pub struct Compaction {
    pub report: crate::compact::CompactionReport,
    pub messages: Vec<ChatMessage>,
    pub media: Vec<MediaInput>,
}

impl Compaction {
    /// `x-gibbs-compacted: {before}->{after}` (prompt tokens).
    pub fn header(&self) -> (axum::http::HeaderName, axum::http::HeaderValue) {
        let r = &self.report;
        let v = format!("{}->{}", r.before_tokens, r.after_tokens);
        (
            axum::http::HeaderName::from_static("x-gibbs-compacted"),
            axum::http::HeaderValue::from_str(&v).expect("digits and an arrow"),
        )
    }

    /// The compacted messages as a client resends them: media parts become `image_url` /
    /// `input_audio` data parts holding their bytes.
    pub fn wire_messages(&self) -> Value {
        let mut media = self.media.iter();
        let mut out = serde_json::to_value(&self.messages).unwrap_or_default();
        for m in out.as_array_mut().into_iter().flatten() {
            for p in m["content"].as_array_mut().into_iter().flatten() {
                if p["type"] == "text" {
                    continue;
                }
                *p = match media.next() {
                    Some(MediaInput::Image(b)) => {
                        let mime = if b.starts_with(b"\x89PNG") { "png" } else { "jpeg" };
                        let url = format!(
                            "data:image/{mime};base64,{}",
                            crate::model::image::base64_encode(b)
                        );
                        serde_json::json!({"type": "image_url", "image_url": {"url": url}})
                    }
                    Some(MediaInput::Audio(b)) => serde_json::json!({"type": "input_audio",
                        "input_audio": {"data": crate::model::image::base64_encode(b), "format": "wav"}}),
                    None => continue,
                };
            }
        }
        out
    }
}

/// A finished generation, for non-streaming responses.
pub struct Output {
    pub text: String,
    pub calls: Vec<ToolCall>,
    pub finish: Finish,
    pub completion_tokens: usize,
}

/// Requests panel labels by sequence id; kept after completion so finished rows stay labeled.
static LABELS: LazyLock<Mutex<BTreeMap<u64, String>>> = LazyLock::new(Default::default);
/// Labels kept; sequence ids only grow, so the oldest go first.
const MAX_LABELS: usize = 256;

/// The API and client that submitted sequence `id`; `None` for TUI prompts.
pub fn request_label(id: u64) -> Option<String> {
    LABELS.lock().get(&id).cloned()
}

/// `<api> · <user-agent>` for the Requests panel.
pub fn label(api: &str, headers: &axum::http::HeaderMap) -> String {
    let ua = headers
        .get(axum::http::header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-");
    format!("{api} · {ua}")
}

/// Short unique id (tool calls, responses, messages).
pub fn new_id() -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;
    format!("{t:x}{:04x}", N.fetch_add(1, Ordering::Relaxed) & 0xffff)
}

pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// An LLM is loaded and no classifier or System 1 encoder has taken its place.
pub fn text_model_active(state: &AppState) -> bool {
    state.scheduler.has_model()
        && state.classifier.read().is_none()
        && state.jev_model.read().is_none()
}

pub fn no_text_model(state: &AppState) -> ApiError {
    ApiError::bad(if let Some(c) = state.classifier.read().as_ref() {
        format!(
            "The active model {} classifies text ({}); it does not generate. Use /v1/classify (or /v1/rerank for 1-label pair models)",
            c.repo_id,
            c.summary()
        )
    } else if let Some(jev) = state.jev_model.read().as_ref() {
        format!(
            "The active model {} is running in System 1 mode; use /v1/decisions, /v1/system1/decide, or /v1/embeddings",
            jev.repo_id
        )
    } else if state.image_model.read().is_some() {
        "No text generation model is loaded (this server is serving an image model); use /v1/images/generations".to_string()
    } else {
        "No text generation model is loaded on this server".to_string()
    })
}

/// Agents send their built-in/server tools (`web_search`, `local_shell`, `bash_*`, ...) on every
/// turn; the model can only call function tools, so those are left out of the prompt and named
/// once in the log. A `tool_choice` naming one is a 400: this server cannot run it.
pub fn skip_builtin_tools(
    state: &AppState,
    api: &str,
    skipped: &[String],
    choice: &ToolChoice,
) -> Result<(), ApiError> {
    if let ToolChoice::Named(n) = choice {
        if skipped.contains(n) {
            return Err(ApiError::bad(format!(
                "tool_choice names '{n}', a built-in tool this server cannot run; only function tools are called"
            )));
        }
    }
    if !skipped.is_empty() {
        state.telemetry.log_warn(&format!(
            "{api}: left out built-in tools the model cannot call: {}",
            skipped.join(", ")
        ));
    }
    Ok(())
}

/// OpenAI-shaped messages → what chat templates read, plus the media in part order:
/// text-only part lists become strings, `image_url` / `input_image` / `input_audio` parts become
/// `{"type":"image"|"audio"}` with their bytes in the returned list, tool-call `arguments` strings
/// become objects, null content becomes "", and leading system messages merge into one.
pub async fn prepare_messages(
    messages: Vec<ChatMessage>,
) -> Result<(Vec<ChatMessage>, Vec<MediaInput>), ApiError> {
    let mut media = Vec::new();
    let mut out: Vec<ChatMessage> = Vec::with_capacity(messages.len());
    for mut m in messages {
        if m.role == "developer" {
            m.role = "system".into();
        }
        m.content = match std::mem::take(&mut m.content) {
            Value::Null => Value::String(String::new()),
            Value::Array(parts) => {
                let mut hf = Vec::with_capacity(parts.len());
                for p in parts {
                    hf.push(content_part(p, &mut media).await?);
                }
                if hf.iter().all(|p| p["type"] == "text") {
                    Value::String(hf.iter().filter_map(|p| p["text"].as_str()).collect())
                } else {
                    Value::Array(hf)
                }
            }
            c => c,
        };
        if let Some(Value::Array(calls)) = &mut m.tool_calls {
            for c in calls {
                if let Some(args) = c.pointer_mut("/function/arguments") {
                    if let Some(Ok(v @ Value::Object(_))) =
                        args.as_str().map(serde_json::from_str::<Value>)
                    {
                        *args = v;
                    }
                }
            }
        }
        let leading = out.len() == 1;
        match out.last_mut() {
            Some(prev)
                if leading
                    && m.role == "system"
                    && prev.role == "system"
                    && m.content.is_string()
                    && prev.content.is_string() =>
            {
                prev.content = Value::String(format!("{}\n\n{}", prev.text(), m.text()));
            }
            _ => out.push(m),
        }
    }
    Ok((out, media))
}

async fn content_part(p: Value, media: &mut Vec<MediaInput>) -> Result<Value, ApiError> {
    let kind = p["type"].as_str().unwrap_or_default();
    match kind {
        "text" | "input_text" | "output_text" => {
            Ok(serde_json::json!({"type": "text", "text": p["text"].as_str().unwrap_or_default()}))
        }
        "image_url" | "input_image" => {
            // Chat Completions nests `{"url": ..}`; Responses gives the URL string directly.
            let url = p["image_url"]["url"]
                .as_str()
                .or_else(|| p["image_url"].as_str())
                .ok_or_else(|| ApiError::bad(format!("{kind} part has no URL")))?;
            media.push(MediaInput::Image(fetch_url(url).await?));
            Ok(serde_json::json!({"type": "image"}))
        }
        "input_audio" => {
            let data = p["input_audio"]["data"]
                .as_str()
                .ok_or_else(|| ApiError::bad("input_audio part has no data"))?;
            media.push(MediaInput::Audio(
                crate::model::image::base64_decode(data).map_err(ApiError::bad)?,
            ));
            Ok(serde_json::json!({"type": "audio"}))
        }
        other => Err(ApiError::bad(format!(
            "unsupported content part type '{other}' (text, image_url, input_audio)"
        ))),
    }
}

/// Bytes of a `data:` URL or an http(s) URL (fetched by curl, http(s) only even on redirect).
pub async fn fetch_url(url: &str) -> Result<Vec<u8>, ApiError> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (_, data) = rest
            .split_once(";base64,")
            .ok_or_else(|| ApiError::bad("data URL must be base64 (data:<mime>;base64,...)"))?;
        return crate::model::image::base64_decode(data).map_err(ApiError::bad);
    }
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(ApiError::bad(format!(
            "media URL must be data: or http(s): {url:.64}"
        )));
    }
    let out = tokio::process::Command::new("curl")
        .args(["-sSfLg", "--max-time", "30", "--max-filesize", "67108864"])
        .args([
            "--proto",
            "=http,https",
            "--proto-redir",
            "=http,https",
            "--",
            url,
        ])
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|e| ApiError::bad(format!("curl: {e}")))?;
    if !out.status.success() {
        return Err(ApiError::bad(format!(
            "cannot fetch {url:.64}: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(out.stdout)
}

/// The prompt text and, when tools are in play, the tool parser plus the enforced choice.
fn render(
    tok: &Tokenizer,
    prompt: Prompt,
) -> Result<(String, Option<(ToolParser, ToolChoice)>), ApiError> {
    let (messages, tools, choice, kwargs) = match prompt {
        Prompt::Raw(text) => return Ok((text, None)),
        Prompt::Chat {
            messages,
            tools,
            tool_choice,
            template_kwargs,
        } => (messages, tools, tool_choice, template_kwargs),
    };
    let tools = if choice == ToolChoice::None {
        Vec::new()
    } else {
        tools
    };
    if tools.is_empty() && choice != ToolChoice::Auto && choice != ToolChoice::None {
        return Err(ApiError::bad("tool_choice requires tools"));
    }
    if let ToolChoice::Named(n) = &choice {
        if !tools.iter().any(|t| t["function"]["name"] == n.as_str()) {
            return Err(ApiError::bad(format!(
                "tool_choice names unknown tool '{n}'"
            )));
        }
    }
    let format = tok
        .chat_template
        .as_ref()
        .and_then(|t| ToolFormat::detect(&t.template));
    let mut text = match &tok.chat_template {
        Some(spec) => spec
            .render(
                &messages,
                (!tools.is_empty()).then_some(&tools[..]),
                true,
                &kwargs,
            )
            .map_err(|e| ApiError::bad(format!("chat template: {e}")))?,
        None => tok.apply_chat_template_messages(&messages, None),
    };
    if tools.is_empty() {
        return Ok((text, None));
    }
    let format = format.ok_or_else(|| {
        ApiError::bad("the loaded model's chat template has no tool-calling format; send the request without tools")
    })?;
    let mut parser = ToolParser::new(format, &tools);
    let forced = match &choice {
        ToolChoice::Required => Some(format.forced_prefix(None)),
        ToolChoice::Named(n) => Some(format.forced_prefix(Some(n))),
        _ => None,
    };
    if let Some(prefix) = forced {
        text.push_str(&prefix);
        // The prefix is prompt, not output: the parser sees it, the client never does.
        parser.push(&prefix);
    }
    Ok((text, Some((parser, choice))))
}

/// Template + encode + model prepare (image/audio encoders) on a blocking thread, then queued;
/// returns what `submit_with_media` returns.
async fn submit(
    state: &AppState,
    text: String,
    params: SamplingParams,
    media: Vec<MediaInput>,
) -> Result<
    (
        u64,
        UnboundedReceiver<u32>,
        Arc<Tokenizer>,
        usize,
        crate::scheduler::FinishSignal,
    ),
    ApiError,
> {
    let scheduler = state.scheduler.clone();
    // Dropping this future (client disconnect) sets the flag: a model decoding inside `prepare`
    // (seq2seq beam search) stops at its next step. Setting it after `prepare` is harmless.
    struct CancelOnDrop(Arc<std::sync::atomic::AtomicBool>);
    impl Drop for CancelOnDrop {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
    let guard = CancelOnDrop(Default::default());
    let cancel = Some(guard.0.clone());
    tokio::task::spawn_blocking(move || {
        scheduler.submit_cancellable(params, &media, cancel, |t| t.encode(&text))
    })
    .await
    .map_err(|e| ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: format!("prompt preparation failed: {e}"),
    })?
    .map_err(ApiError::bad)
}

/// Renders the prompt; with media, a template error most likely means a text-only model.
fn render_for(
    state: &AppState,
    prompt: Prompt,
    media: &[MediaInput],
) -> Result<(String, Option<(ToolParser, ToolChoice)>), ApiError> {
    render(&state.scheduler.tokenizer(), prompt).map_err(|mut e| {
        if !media.is_empty() {
            e.message = format!(
                "{}; the loaded model's template cannot place image/audio input (text-only model?)",
                e.message
            );
        }
        e
    })
}

/// Prompt token count of a request (Anthropic `count_tokens`): the ids the model would
/// prefill, media expanded, prepared exactly as [`start`] does, but never queued.
pub async fn count_tokens(
    state: &AppState,
    prompt: Prompt,
    media: Vec<MediaInput>,
) -> Result<usize, ApiError> {
    if !text_model_active(state) {
        return Err(no_text_model(state));
    }
    let (text, _) = render_for(state, prompt, &media)?;
    let scheduler = state.scheduler.clone();
    tokio::task::spawn_blocking(move || scheduler.prompt_len(&media, |t| t.encode(&text)))
        .await
        .map_err(|e| ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("prompt preparation failed: {e}"),
        })?
        .map_err(ApiError::bad)
}

/// Renders and queues the request; generation runs on the scheduler and its decoded,
/// stop-trimmed, tool-parsed events arrive on `events`. Dropping `events` cancels it.
pub async fn start(state: &AppState, mut req: GenRequest) -> Result<Generation, ApiError> {
    if !text_model_active(state) {
        return Err(no_text_model(state));
    }
    let mut compaction = None;
    if let Prompt::Chat { messages, .. } = &mut req.prompt {
        let mut cfg = crate::tui::settings::load_settings().compact;
        if let Some(o) = &req.compact {
            o.apply(&mut cfg);
        }
        if cfg.mode != crate::compact::CompactionMode::Off {
            let sched = state.scheduler.clone();
            let (mut msgs, mut media) = (std::mem::take(messages), std::mem::take(&mut req.media));
            // The summarizer generates on the scheduler: never on an async worker.
            let (msgs, media, result) = tokio::task::spawn_blocking(move || {
                let r = crate::compact::compact_for_scheduler(
                    &sched, &mut msgs, &mut media, None, &cfg, false,
                );
                (msgs, media, r)
            })
            .await
            .map_err(|e| ApiError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: format!("compaction failed: {e}"),
            })?;
            (*messages, req.media) = (msgs, media);
            match result {
                Ok(report) if report.saved_tokens() > 0 => {
                    state.telemetry.log_info(&format!(
                        "compact: {} -> {} tokens (-{} tok; tier1={}B, tier2={}B, tier3={} turns)",
                        report.before_tokens,
                        report.after_tokens,
                        report.saved_tokens(),
                        report.tier1_chars_elided,
                        report.tier2_chars_shaken,
                        report.tier3_turns_summarized,
                    ));
                    compaction = Some(Compaction {
                        report,
                        messages: messages.clone(),
                        media: req.media.clone(),
                    });
                }
                Ok(_) => {}
                Err(e) => state
                    .telemetry
                    .log_warn(&format!("compact: {e}; the request proceeds uncompacted")),
            }
        }
    }
    let (text, mut tools) = render_for(state, req.prompt, &req.media)?;
    let (id, mut rx, tokenizer, prompt_tokens, finish) =
        submit(state, text, req.params, req.media).await?;
    {
        let mut labels = LABELS.lock();
        labels.insert(id, req.label);
        if labels.len() > MAX_LABELS {
            labels.pop_first();
        }
    }
    let (tx, events) = unbounded_channel();
    let telemetry = state.telemetry.clone();
    let mut sink = Sink {
        tx,
        stops: Stops::new(req.stop),
        stopped: None,
        calls: Vec::new(),
    };
    tokio::spawn(async move {
        let started = Instant::now();
        let (mut detok, mut n) = (Vec::new(), 0usize);
        let parse = |tools: &mut Option<(ToolParser, ToolChoice)>, text: String| match tools {
            Some((p, _)) => p.push(&text),
            None => vec![Piece::Text(text)],
        };
        let mut open = true;
        while let Some(tok_id) = rx.recv().await {
            n += 1;
            let text = decode_step(&tokenizer, tok_id, &mut detok);
            if !sink.send(parse(&mut tools, text)) {
                open = false;
                break;
            }
        }
        // A stop sequence or a gone client ends the sequence: dropping `rx` cancels it.
        drop(rx);
        let outcome = finish_outcome(&finish);
        if open {
            let mut tail = parse(&mut tools, String::from_utf8_lossy(&detok).into_owned());
            if let Some((p, _)) = &mut tools {
                tail.extend(p.finish());
            }
            if sink.send(tail) {
                let held = sink.stops.flush();
                sink.send_text(held);
            }
        }
        let forced = tools.map(|(_, c)| c);
        let event = final_event(outcome, sink.stopped, forced, &sink.calls, n);
        let _ = sink.tx.send(event);
        let secs = started.elapsed().as_secs_f32();
        telemetry.log_info(&format!(
            "[Request #{id} Done] {n} tok in {:.1}ms ({:.1} tok/s)",
            secs * 1000.0,
            if secs > 0.0 { n as f32 / secs } else { 0.0 }
        ));
    });
    Ok(Generation {
        id,
        prompt_tokens,
        events,
        compaction,
    })
}

/// Where parsed pieces go: stop sequences applied to text, calls recorded, events sent.
struct Sink {
    tx: tokio::sync::mpsc::UnboundedSender<Event>,
    stops: Stops,
    /// The stop sequence that ended the reply.
    stopped: Option<String>,
    /// Names of the calls sent, for `tool_choice` validation.
    calls: Vec<String>,
}

impl Sink {
    /// False once the reply is over: a stop sequence completed or the client is gone.
    fn send(&mut self, pieces: Vec<Piece>) -> bool {
        for p in pieces {
            let open = match p {
                Piece::Text(t) => {
                    let (emit, hit) = self.stops.push(&t);
                    self.stopped = hit;
                    self.send_text(emit)
                }
                Piece::Call(c) => {
                    let held = self.stops.flush();
                    self.calls.push(c.name.clone());
                    self.send_text(held) && self.tx.send(Event::Call(c)).is_ok()
                }
            };
            // Nothing after a completed stop sequence is sent.
            if !open || self.stopped.is_some() {
                return false;
            }
        }
        true
    }

    fn send_text(&self, text: String) -> bool {
        text.is_empty() || self.tx.send(Event::Text(text)).is_ok()
    }
}

fn done(finish: Finish, completion_tokens: usize) -> Event {
    Event::Done {
        finish,
        completion_tokens,
    }
}

/// The last event of a generation. An engine error wins; then a forced `tool_choice` must have
/// produced its call whatever ended the reply (stop sequence, length, EOS); then the finish.
fn final_event(
    outcome: Result<&'static str, String>,
    stopped: Option<String>,
    forced: Option<ToolChoice>,
    calls: &[String],
    n: usize,
) -> Event {
    let reason = match outcome {
        Ok(r) => r,
        Err(e) => return Event::Error(e),
    };
    match forced {
        Some(ToolChoice::Required) if calls.is_empty() => {
            return Event::Error(
                "the model did not produce a valid tool call (tool_choice: required)".into(),
            )
        }
        Some(ToolChoice::Named(name)) if calls.is_empty() || calls.iter().any(|c| *c != name) => {
            return Event::Error(format!(
                "the model did not produce a valid call to '{name}'"
            ))
        }
        _ => {}
    }
    match stopped {
        Some(s) => done(Finish::StopSequence(s), n),
        None if !calls.is_empty() => done(Finish::ToolCalls, n),
        None if reason == "length" => done(Finish::Length, n),
        None => done(Finish::Stop, n),
    }
}

/// Appends one token's bytes to `pending` and returns the complete UTF-8 prefix; tool-call
/// marker tokens decode to their text.
fn decode_step(tok: &Tokenizer, id: u32, pending: &mut Vec<u8>) -> String {
    if let Some(m) = tok.tool_marker(id) {
        pending.extend_from_slice(m.as_bytes());
    } else if !tok.append_token_bytes(id, pending) {
        pending.extend_from_slice(tok.decode_token(id).as_bytes());
    }
    let valid = match std::str::from_utf8(pending) {
        Ok(s) => s.len(),
        Err(e) if e.error_len().is_none() => e.valid_up_to(),
        Err(_) => return String::from_utf8_lossy(&std::mem::take(pending)).into_owned(),
    };
    let rest = pending.split_off(valid);
    String::from_utf8(std::mem::replace(pending, rest)).expect("validated UTF-8")
}

/// Stop sequences over streamed text: text that may begin a stop sequence is held back.
struct Stops {
    stops: Vec<String>,
    held: String,
}

impl Stops {
    fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            held: String::new(),
        }
    }

    /// Text safe to emit, and the stop sequence if one completed (it and what follows are dropped).
    fn push(&mut self, text: &str) -> (String, Option<String>) {
        self.held.push_str(text);
        let hit = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()).map(|i| (i, s)))
            .min_by_key(|(i, _)| *i);
        if let Some((i, s)) = hit {
            let s = s.clone();
            self.held.truncate(i);
            return (std::mem::take(&mut self.held), Some(s));
        }
        let keep = self
            .stops
            .iter()
            .map(|s| partial_suffix(&self.held, s))
            .max()
            .unwrap_or(0);
        (self.held.drain(..self.held.len() - keep).collect(), None)
    }

    fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

/// 503 for a generation aborted because the model was swapped (the client should retry), else 500.
pub fn error_status(message: &str) -> StatusCode {
    if message.starts_with(crate::scheduler::MODEL_SWAPPED) {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

/// Waits for the whole generation.
pub async fn collect(mut gen: Generation) -> Result<Output, ApiError> {
    let (mut text, mut calls) = (String::new(), Vec::new());
    while let Some(ev) = gen.events.recv().await {
        match ev {
            Event::Text(t) => text.push_str(&t),
            Event::Call(c) => calls.push(c),
            Event::Done {
                finish,
                completion_tokens,
            } => {
                return Ok(Output {
                    text,
                    calls,
                    finish,
                    completion_tokens,
                })
            }
            Event::Error(e) => {
                return Err(ApiError {
                    status: error_status(&e),
                    message: e,
                })
            }
        }
    }
    Err(ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: "generation ended without a result".into(),
    })
}

/// A schema's streaming wire format: core events → that schema's JSON events, in order.
pub trait Wire: Send + 'static {
    /// Events before the first token (e.g. `message_start`).
    fn open(&mut self) -> Vec<Value> {
        Vec::new()
    }
    /// Events for one core event; `Done`/`Error` are the last.
    fn on(&mut self, ev: Event) -> Vec<Value>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    /// `data: {json}` lines closed by `data: [DONE]` (OpenAI Chat/Completions).
    SseDone,
    /// `event: <type>` + `data: {json}` (Anthropic, OpenAI Responses).
    SseNamed,
    /// One JSON object per line (`/api/*`).
    Ndjson,
}

/// All wire events of a generation, in order (streamed by [`stream_response`]).
pub fn wire_stream<W: Wire>(
    gen: Generation,
    mut wire: W,
) -> impl futures::Stream<Item = Value> + Send {
    let head = wire.open();
    let body = stream::unfold(
        (gen, wire, false),
        |(mut gen, mut wire, ended)| async move {
            if ended {
                return None;
            }
            let ev = gen
                .events
                .recv()
                .await
                .unwrap_or_else(|| Event::Error("generation ended without a result".into()));
            let last = matches!(ev, Event::Done { .. } | Event::Error(_));
            Some((stream::iter(wire.on(ev)), (gen, wire, last)))
        },
    )
    .flatten();
    stream::iter(head).chain(body)
}

pub fn stream_response<W: Wire>(gen: Generation, wire: W, framing: Framing) -> Response {
    let events = wire_stream(gen, wire);
    match framing {
        Framing::Ndjson => {
            let lines = events.map(|v| Ok::<_, Infallible>(format!("{v}\n")));
            (
                [(axum::http::header::CONTENT_TYPE, "application/x-ndjson")],
                axum::body::Body::from_stream(lines),
            )
                .into_response()
        }
        Framing::SseNamed => Sse::new(events.map(|v| {
            let name = v["type"].as_str().unwrap_or("message").to_string();
            Ok::<_, Infallible>(SseEvent::default().event(name).data(v.to_string()))
        }))
        .keep_alive(KeepAlive::default())
        .into_response(),
        Framing::SseDone => Sse::new(
            events
                .map(|v| SseEvent::default().data(v.to_string()))
                .chain(stream::once(async { SseEvent::default().data("[DONE]") }))
                .map(Ok::<_, Infallible>),
        )
        .keep_alive(KeepAlive::default())
        .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_swap_abort_is_a_503_and_other_errors_stay_500() {
        let swapped = format!("{}org/model", crate::scheduler::MODEL_SWAPPED);
        assert_eq!(error_status(&swapped), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            error_status("decode failed: boom"),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        let (tx, rx) = unbounded_channel();
        tx.send(Event::Error(swapped)).unwrap();
        let gen = Generation {
            id: 0,
            prompt_tokens: 0,
            events: rx,
            compaction: None,
        };
        let Err(err) = futures::executor::block_on(collect(gen)) else {
            panic!("a swap abort is an error");
        };
        assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn nothing_follows_a_stop_sequence_even_when_it_emits_no_text() {
        let (tx, mut rx) = unbounded_channel();
        let mut sink = Sink {
            tx,
            stops: Stops::new(vec!["5".into()]),
            stopped: None,
            calls: Vec::new(),
        };
        let text = |s: &str| Piece::Text(s.to_string());
        assert!(sink.send(vec![text("4"), text(",")]));
        assert!(!sink.send(vec![text(" "), text("5"), text(",")]));
        assert_eq!(sink.stopped.as_deref(), Some("5"));
        drop(sink);
        let mut got = String::new();
        while let Ok(Event::Text(t)) = rx.try_recv() {
            got.push_str(&t);
        }
        assert_eq!(got, "4, ");
    }

    #[test]
    fn forced_tool_choice_wins_over_stop_and_length() {
        let required = Some(ToolChoice::Required);
        let named = Some(ToolChoice::Named("f".into()));
        let calls = |names: &[&str]| names.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        // A stop sequence (or the length cap) before any call is not a successful forced call.
        assert!(matches!(
            final_event(Ok("stop"), Some("5".into()), required.clone(), &[], 3),
            Event::Error(_)
        ));
        assert!(matches!(
            final_event(Ok("length"), None, named.clone(), &calls(&["g"]), 3),
            Event::Error(_)
        ));
        assert_eq!(
            final_event(Ok("length"), None, named, &calls(&["f"]), 3),
            done(Finish::ToolCalls, 3)
        );
        assert_eq!(
            final_event(Ok("stop"), Some("5".into()), None, &[], 3),
            done(Finish::StopSequence("5".into()), 3)
        );
        assert_eq!(
            final_event(Ok("length"), None, Some(ToolChoice::Auto), &[], 3),
            done(Finish::Length, 3)
        );
        assert!(matches!(
            final_event(Err("boom".into()), None, required, &calls(&["f"]), 3),
            Event::Error(e) if e == "boom"
        ));
    }

    #[test]
    fn stops_hold_partial_matches_and_cut_at_the_earliest() {
        let mut s = Stops::new(vec!["END".into(), "\n\n".into(), String::new()]);
        assert_eq!(s.push("abc E"), ("abc ".to_string(), None));
        assert_eq!(s.push("Nx"), ("ENx".to_string(), None));
        assert_eq!(
            s.push("y\n\nzEND"),
            ("y".to_string(), Some("\n\n".to_string()))
        );
        let mut s = Stops::new(vec!["</s>".into()]);
        assert_eq!(s.push("é</"), ("é".to_string(), None));
        assert_eq!(s.flush(), "</");
    }

    #[tokio::test]
    async fn prepare_messages_normalizes_for_templates() {
        let msgs: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
            {"role": "system", "content": "A"},
            {"role": "developer", "content": [{"type": "text", "text": "B"}]},
            {"role": "user", "content": [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}]},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "1", "type": "function",
                "function": {"name": "f", "arguments": "{\"x\": 1}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "42"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAEC"}},
                                         {"type": "text", "text": "what?"}]}
        ]))
        .unwrap();
        let (m, media) = prepare_messages(msgs).await.unwrap();
        assert_eq!(m.len(), 5, "leading system + developer merge");
        assert_eq!(m[0].content, "A\n\nB");
        assert_eq!(m[1].content, "hi there");
        assert_eq!(m[2].content, "");
        assert_eq!(
            m[2].tool_calls.as_ref().unwrap()[0]["function"]["arguments"]["x"],
            1
        );
        assert_eq!(m[4].content[0]["type"], "image");
        assert_eq!(media, vec![MediaInput::Image(vec![0, 1, 2])]);
        let bad: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": [{"type": "file", "file": {}}]}
        ]))
        .unwrap();
        assert!(prepare_messages(bad)
            .await
            .unwrap_err()
            .message
            .contains("'file'"));
        let file: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file:///etc/passwd"}}]}
        ]))
        .unwrap();
        assert!(prepare_messages(file).await.is_err());
    }

    #[test]
    fn decode_step_holds_split_utf8() {
        let mut tok = Tokenizer::default_bpe();
        // Two byte tokens that together spell "é" (0xC3 0xA9).
        let (a, b) = (0xC3u32, 0xA9u32);
        tok.inv_vocab.retain(|&id, _| id != a && id != b);
        let mut pending = Vec::new();
        assert_eq!(decode_step(&tok, a, &mut pending), "");
        assert_eq!(decode_step(&tok, b, &mut pending), "é");
        assert!(pending.is_empty());
    }

    /// The real Qwen3.5 template (XML tool calls, `arguments|items`) renders an OpenAI-shaped
    /// tool round trip once `prepare_messages` has normalized it, and the forced prefix follows.
    #[tokio::test]
    async fn qwen35_template_renders_normalized_tool_turns() {
        let path = "disk/models--Qwen--Qwen3.5-0.8B/snapshots/main/chat_template.jinja";
        let Ok(template) = std::fs::read_to_string(path) else {
            println!("SKIP: {path} missing");
            return;
        };
        let mut tok = Tokenizer::default_bpe();
        tok.chat_template = Some(crate::model::tokenizer::ChatTemplateSpec {
            template,
            bos_token: String::new(),
            eos_token: "<|im_end|>".into(),
        });
        let tools = vec![
            serde_json::json!({"type": "function", "function": {"name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}),
        ];
        let msgs: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
            {"role": "developer", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "18C"}
        ]))
        .unwrap();
        let (messages, _) = prepare_messages(msgs).await.unwrap();
        let prompt = Prompt::Chat {
            messages,
            tools,
            tool_choice: ToolChoice::Named("get_weather".into()),
            template_kwargs: Value::Null,
        };
        let (text, parser) = render(&tok, prompt).unwrap();
        assert!(
            text.contains("<function=get_weather>\n<parameter=city>\nParis\n</parameter>"),
            "{text}"
        );
        assert!(text.contains("<tool_response>\n18C"), "{text}");
        assert!(
            text.ends_with("<tool_call>\n<function=get_weather>\n"),
            "{text}"
        );
        assert!(parser.is_some());
    }
}
