mod classify;

mod anthropic;
pub mod connect;
pub mod core;
mod ndjson_api;
mod openai;
mod responses;
pub mod tools;

use crate::runtime::encoder::system1::{ChoiceOption, JevDecision, JevQuestion, JevSystem1Model};
use crate::scheduler::ContinuousBatchScheduler;
use crate::telemetry::EngineTelemetry;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
pub struct AppState {
    pub scheduler: Arc<ContinuousBatchScheduler>,
    pub telemetry: Arc<EngineTelemetry>,
    pub jev_model: crate::runtime::encoder::system1::System1Slot,
    /// Active image model (shared with the TUI, hot-swapped by its picker).
    pub image_model: crate::runtime::diffusion::ImageSlot,
    /// Active sequence classifier (shared with the TUI); while set, text generation answers 400.
    pub classifier: crate::model::classifier::ClassifierSlot,
}

pub use crate::model::hf_loader::ChatMessage;

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub input: EmbeddingInput,
    pub model: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Serialize)]
pub struct EmbeddingItem {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingItem>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Deserialize)]
pub struct DecisionRequest {
    #[serde(alias = "input")]
    pub state: Option<serde_json::Value>,
    #[serde(default)]
    pub questions: Vec<System1QuestionItem>,
    pub model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DecisionResponse {
    pub model: String,
    pub decisions: HashMap<String, serde_json::Value>,
    pub latency_us: u128,
}

// System 1 Request / Response Schemas
#[derive(Debug, Deserialize)]
pub struct System1JudgeRequest {
    pub state: serde_json::Value,
    #[serde(default)]
    pub questions: Vec<System1QuestionItem>,
    pub model: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum System1QuestionItem {
    #[serde(rename = "bool")]
    Bool {
        id: String,
        instructions: String,
        criteria_true: Option<String>,
        criteria_false: Option<String>,
    },
    #[serde(rename = "choice")]
    Choice {
        id: String,
        instructions: String,
        options: Vec<System1ChoiceOption>,
    },
    #[serde(rename = "score")]
    Score {
        id: String,
        instructions: String,
        criteria: Vec<String>,
    },
}

#[derive(Debug, Deserialize)]
pub struct System1ChoiceOption {
    pub label: String,
    pub description: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct System1JudgeResponse {
    pub model: String,
    pub decisions: HashMap<String, serde_json::Value>,
    pub latency_us: u128,
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/responses", post(responses::responses))
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/api/chat", post(ndjson_api::chat))
        .route("/api/generate", post(ndjson_api::generate))
        .route("/api/tags", get(ndjson_api::tags))
        .route("/api/show", post(ndjson_api::show))
        .route("/api/ps", get(ndjson_api::ps))
        .route("/api/version", get(ndjson_api::version))
        .route("/", get(ndjson_api::root))
        .route("/v1/models", get(handle_list_models))
        .route("/v1/system1/judge", post(handle_system1_judge))
        .route("/v1/decisions", post(handle_decisions))
        .route("/v1/images/generations", post(handle_image_generations))
        .route("/v1/images/edits", post(handle_image_edits))
        .route("/v1/embeddings", post(handle_embeddings))
        .route("/v1/classify", post(classify::handle_classify))
        .route("/v1/rerank", post(classify::handle_rerank))
        .route("/health", get(handle_health))
        .route("/metrics", get(handle_metrics))
        .route(
            "/v1/draft/config",
            get(handle_get_draft_config).post(handle_set_draft_config),
        )
        .route("/v1/compact", post(handle_compact))
        .route("/v1/artifacts/:hex", get(handle_artifact))
        // Agent requests carry whole conversations and base64 images; axum's default is 2 MB.
        .layer(axum::extract::DefaultBodyLimit::max(64 << 20))
        .with_state(state)
}

/// The loaded System 1 encoder (or active LLM turned into a System 1 encoder), or a 400.
fn system1_model(state: &AppState) -> Result<Arc<JevSystem1Model>, (StatusCode, String)> {
    if let Some(jev) = state.jev_model.read().clone() {
        return Ok(jev);
    }
    if let Some(model) = state.scheduler.model() {
        return Ok(Arc::new(JevSystem1Model::from_llm(
            state.scheduler.model_id(),
            model,
        )));
    }
    Err((
        StatusCode::BAD_REQUEST,
        "no System 1 or LLM model loaded".to_string(),
    ))
}

fn to_jev_question(q: System1QuestionItem) -> JevQuestion {
    match q {
        System1QuestionItem::Bool {
            id,
            instructions,
            criteria_true,
            criteria_false,
        } => JevQuestion::Bool {
            id,
            instructions,
            criteria_true,
            criteria_false,
        },
        System1QuestionItem::Choice {
            id,
            instructions,
            options,
        } => JevQuestion::Choice {
            id,
            instructions,
            options: options
                .into_iter()
                .map(|o| ChoiceOption {
                    label: o.label,
                    description: o.description,
                })
                .collect(),
        },
        System1QuestionItem::Score {
            id,
            instructions,
            criteria,
        } => JevQuestion::Score {
            id,
            instructions,
            criteria,
        },
    }
}

/// Answers every question with one shared state embedding; unsupported questions become
/// per-question `{"error": ...}` entries while the rest still answer.
fn evaluate_questions(
    jev: &JevSystem1Model,
    state_str: &str,
    questions: Vec<System1QuestionItem>,
) -> Result<HashMap<String, serde_json::Value>, (StatusCode, String)> {
    let questions: Vec<JevQuestion> = questions.into_iter().map(to_jev_question).collect();
    let results = jev
        .batch_evaluate(state_str, &questions)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(results
        .into_iter()
        .map(|(qid, r)| {
            let val = match r {
                Ok(JevDecision::Bool {
                    bool_prob,
                    decision,
                }) => serde_json::json!({
                    "type": "bool",
                    "decision": decision,
                    "bool_prob": bool_prob,
                }),
                Ok(JevDecision::Choice {
                    selected,
                    probabilities,
                    confidence,
                }) => serde_json::json!({
                    "type": "choice",
                    "selected": selected,
                    "probabilities": probabilities,
                    "confidence": confidence,
                }),
                Err(e) => serde_json::json!({ "error": e }),
            };
            (qid, val)
        })
        .collect())
}

async fn handle_embeddings(
    State(state): State<AppState>,
    Json(payload): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, String)> {
    let jev = system1_model(&state)?;
    let inputs = match payload.input {
        EmbeddingInput::Single(s) => vec![s],
        EmbeddingInput::Multiple(v) => v,
    };

    let mut data = Vec::with_capacity(inputs.len());
    let mut total_tokens = 0usize;
    for (index, text) in inputs.iter().enumerate() {
        total_tokens += jev.token_count(text).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("input {index}: {e}"),
            )
        })?;
        let embedding = jev.embed(text).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("input {index}: {e}"),
            )
        })?;
        data.push(EmbeddingItem {
            object: "embedding".to_string(),
            embedding,
            index,
        });
    }

    Ok(Json(EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: jev.repo_id.clone(),
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    }))
}

async fn handle_decisions(
    State(state): State<AppState>,
    Json(payload): Json<DecisionRequest>,
) -> Result<Json<DecisionResponse>, (StatusCode, String)> {
    let start = Instant::now();
    let jev = system1_model(&state)?;
    if payload.questions.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "`questions` is required: send bool (criteria_true/criteria_false) or choice (options) questions".to_string(),
        ));
    }
    let state_str = match &payload.state {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(v) => serde_json::to_string(v).unwrap_or_default(),
        None => String::new(),
    };
    let decisions = evaluate_questions(&jev, &state_str, payload.questions)?;
    Ok(Json(DecisionResponse {
        model: jev.repo_id.clone(),
        decisions,
        latency_us: start.elapsed().as_micros(),
    }))
}

async fn handle_system1_judge(
    State(state): State<AppState>,
    Json(payload): Json<System1JudgeRequest>,
) -> Result<Json<System1JudgeResponse>, (StatusCode, String)> {
    let start = Instant::now();
    let jev = system1_model(&state)?;
    let state_str = if let serde_json::Value::String(s) = &payload.state {
        s.clone()
    } else {
        serde_json::to_string(&payload.state).unwrap_or_default()
    };
    let decisions = evaluate_questions(&jev, &state_str, payload.questions)?;
    Ok(Json(System1JudgeResponse {
        model: jev.repo_id.clone(),
        decisions,
        latency_us: start.elapsed().as_micros(),
    }))
}

async fn handle_list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let model_id = state.scheduler.model_id();
    let entry = |id: &str, mode: &str| serde_json::json!({ "id": id, "object": "model", "owned_by": "gibbs", "mode": mode });
    // Only models that are actually loaded in this process.
    let mut data = Vec::new();
    if core::text_model_active(&state) {
        data.push(entry(&model_id, "text_generation"));
    }
    if let Some(c) = state.classifier.read().as_ref() {
        data.push(entry(&c.repo_id, "text_classification"));
    }
    if let Some(jev) = state.jev_model.read().as_ref() {
        data.push(entry(&jev.repo_id, "sentence_encoder"));
    }
    if let Some(m) = state.image_model.read().as_ref() {
        data.push(entry(&m.model_id().unwrap_or(model_id), "text_to_image"));
    }
    Json(serde_json::json!({ "object": "list", "data": data }))
}

async fn handle_health() -> &'static str {
    "OK"
}

async fn handle_metrics(
    State(state): State<AppState>,
) -> Json<crate::telemetry::TelemetrySnapshot> {
    Json(state.telemetry.get_summary())
}

#[derive(Debug, Deserialize)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    pub model: Option<String>,
    pub n: Option<usize>,
    pub size: Option<String>,
    pub response_format: Option<String>,
    #[serde(alias = "num_inference_steps")]
    pub steps: Option<usize>,
    pub seed: Option<u64>,
    pub negative_prompt: Option<String>,
    #[serde(alias = "guidance_scale", alias = "cfg_scale")]
    pub guidance: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct ImageEditRequest {
    /// Base64 PNG/JPEG, a `data:` URL or an http(s) URL (never a server-side path).
    pub image: String,
    pub prompt: String,
    pub model: Option<String>,
    pub n: Option<usize>,
    pub size: Option<String>,
    pub response_format: Option<String>,
    pub strength: Option<f32>,
    #[serde(alias = "num_inference_steps")]
    pub steps: Option<usize>,
    pub seed: Option<u64>,
    pub negative_prompt: Option<String>,
    #[serde(alias = "guidance_scale", alias = "cfg_scale")]
    pub guidance: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct ImageResponseItem {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ImageGenerationResponse {
    pub created: u64,
    pub data: Vec<ImageResponseItem>,
}

/// OpenAI-style error body; 5xx are `server_error`, everything else `invalid_request_error`.
fn api_error_body(status: StatusCode, message: impl Into<String>) -> serde_json::Value {
    let kind = if status.is_server_error() {
        "server_error"
    } else {
        "invalid_request_error"
    };
    serde_json::json!({
        "error": { "message": message.into(), "type": kind, "code": status.as_u16() }
    })
}

/// OpenAI-style error response.
fn api_error(status: StatusCode, message: impl Into<String>) -> axum::response::Response {
    (status, Json(api_error_body(status, message))).into_response()
}

fn check_steps(steps: Option<usize>) -> Result<Option<usize>, axum::response::Response> {
    match steps {
        Some(s) if !(1..=100).contains(&s) => Err(api_error(
            StatusCode::BAD_REQUEST,
            format!("steps must be in 1..=100, got {s}"),
        )),
        s => Ok(s),
    }
}

/// `WxH` (sides 64..=2048; the pipeline checks its own multiple), or `None`/`auto` for its default.
fn parse_size(size_str: Option<&str>) -> Result<Option<(usize, usize)>, String> {
    let Some(s) = size_str.filter(|s| *s != "auto") else {
        return Ok(None);
    };
    let parsed = s.split_once(['x', 'X']).and_then(|(w, h)| {
        Some((
            w.trim().parse::<usize>().ok()?,
            h.trim().parse::<usize>().ok()?,
        ))
    });
    match parsed {
        Some((w, h)) if (64..=2048).contains(&w) && (64..=2048).contains(&h) => Ok(Some((w, h))),
        _ => Err(format!(
            "Invalid size '{s}': expected WxH, each side in 64..=2048"
        )),
    }
}

type ImageModel = Arc<dyn crate::runtime::diffusion::ImageGenerator>;

/// Validation shared by both image endpoints: returns (pipeline, n, base seed).
fn image_request_common(
    state: &AppState,
    n: Option<usize>,
    response_format: Option<&str>,
    seed: Option<u64>,
) -> Result<(ImageModel, usize, u64), axum::response::Response> {
    let Some(sd) = state.image_model.read().clone() else {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "No image model is loaded. Load one from the TUI picker or start the server with a text-to-image repo, e.g. --model stabilityai/sd-turbo",
        ));
    };
    match response_format.unwrap_or("b64_json") {
        "b64_json" => {}
        "url" => {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "response_format 'url' is not supported (images are not hosted); use 'b64_json'",
            ))
        }
        other => {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                format!("Unknown response_format '{other}'"),
            ))
        }
    }
    let n = n.unwrap_or(1);
    // ponytail: n capped at 4 because each image is a full CPU diffusion run inside one request.
    if !(1..=4).contains(&n) {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            format!("n must be in 1..=4, got {n}"),
        ));
    }
    let seed = seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    });
    Ok((sd, n, seed))
}

fn image_request(
    prompt: String,
    negative_prompt: Option<String>,
    size: Option<&str>,
    steps: Option<usize>,
    guidance: Option<f32>,
) -> Result<crate::runtime::diffusion::ImageRequest, axum::response::Response> {
    let size = parse_size(size).map_err(|e| api_error(StatusCode::BAD_REQUEST, e))?;
    Ok(crate::runtime::diffusion::ImageRequest {
        prompt,
        negative_prompt,
        width: size.map(|s| s.0),
        height: size.map(|s| s.1),
        steps: check_steps(steps)?,
        guidance,
        seed: 0,
    })
}

/// Runs `n` generations (seeds seed, seed+1, ...) on a blocking thread; returns b64_json items.
async fn run_image_jobs<F>(
    n: usize,
    seed: u64,
    job: F,
) -> Result<Json<ImageGenerationResponse>, axum::response::Response>
where
    F: Fn(u64) -> Result<crate::model::image::ImageBuffer, String> + Send + 'static,
{
    let images = tokio::task::spawn_blocking(move || {
        (0..n)
            .map(|i| job(seed.wrapping_add(i as u64)))
            .collect::<Result<Vec<_>, _>>()
    })
    .await
    .map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("image worker failed: {e}"),
        )
    })?
    .map_err(|e| {
        api_error(
            StatusCode::BAD_REQUEST,
            format!("image generation failed: {e}"),
        )
    })?;
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let data = images
        .iter()
        .map(|img| ImageResponseItem {
            b64_json: Some(img.to_base64_png()),
            url: None,
            revised_prompt: None,
        })
        .collect();
    Ok(Json(ImageGenerationResponse { created, data }))
}

async fn handle_image_generations(
    State(state): State<AppState>,
    Json(payload): Json<ImageGenerationRequest>,
) -> Result<Json<ImageGenerationResponse>, axum::response::Response> {
    let (sd, n, seed) = image_request_common(
        &state,
        payload.n,
        payload.response_format.as_deref(),
        payload.seed,
    )?;
    let req = image_request(
        payload.prompt,
        payload.negative_prompt,
        payload.size.as_deref(),
        payload.steps,
        payload.guidance,
    )?;
    run_image_jobs(n, seed, move |s| {
        let req = crate::runtime::diffusion::ImageRequest {
            seed: s,
            ..req.clone()
        };
        sd.text_to_image(&req)
    })
    .await
}

async fn handle_image_edits(
    State(state): State<AppState>,
    Json(payload): Json<ImageEditRequest>,
) -> Result<Json<ImageGenerationResponse>, axum::response::Response> {
    // Base64, data: or http(s) only: the server binds 0.0.0.0, so a path would let any client read
    // server-local files.
    let src = payload.image.trim();
    let bytes = if ["data:", "http://", "https://"]
        .iter()
        .any(|p| src.starts_with(p))
    {
        core::fetch_url(src)
            .await
            .map_err(|e| api_error(e.status, format!("Invalid input image: {}", e.message)))?
    } else {
        crate::model::image::base64_decode(src).map_err(|e| {
            api_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "Invalid input image (expected base64, a data: URL or an http(s) URL): {e}"
                ),
            )
        })?
    };
    let init = crate::model::image::ImageBuffer::decode(&bytes)
        .map_err(|e| api_error(StatusCode::BAD_REQUEST, format!("Invalid input image: {e}")))?;
    let (sd, n, seed) = image_request_common(
        &state,
        payload.n,
        payload.response_format.as_deref(),
        payload.seed,
    )?;
    let strength = payload.strength.unwrap_or(0.7);
    if !(strength > 0.0 && strength <= 1.0) {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            format!("strength must be in (0, 1], got {strength}"),
        ));
    }
    let mut req = image_request(
        payload.prompt,
        payload.negative_prompt,
        payload.size.as_deref(),
        payload.steps,
        payload.guidance,
    )?;
    // The pipeline default, raised so floor(steps * strength) >= 1.
    let steps = req
        .steps
        .unwrap_or_else(|| sd.defaults().steps.max((1.0 / strength).ceil() as usize));
    if (steps as f32 * strength) < 1.0 {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            format!("strength {strength} with {steps} steps yields zero denoising steps"),
        ));
    }
    req.steps = Some(steps);
    run_image_jobs(n, seed, move |s| {
        let req = crate::runtime::diffusion::ImageRequest {
            seed: s,
            ..req.clone()
        };
        sd.image_to_image(&req, &init, strength)
    })
    .await
}

#[derive(Debug, Deserialize, Default)]
pub struct DraftConfigUpdate {
    pub mode: Option<crate::runtime::noise_draft::DraftMode>,
    pub max_k: Option<usize>,
    pub target_acceptance: Option<f32>,
    pub cost_weight: Option<f32>,
    pub suffix_min_match: Option<usize>,
    pub noise_lr: Option<f32>,
    pub noise_sigma: Option<f32>,
}

fn draft_status_json(state: &AppState) -> serde_json::Value {
    let knobs = state.scheduler.draft_knobs();
    let view = state.scheduler.view();
    let sum = state.telemetry.get_summary();
    let rates: Vec<serde_json::Value> = state
        .scheduler
        .k_tuner_rates()
        .into_iter()
        .map(|(k, tok_s, acc)| {
            serde_json::json!({
                "k": k,
                "tok_per_sec": tok_s,
                "acceptance_rate": acc,
            })
        })
        .collect();
    serde_json::json!({
        "knobs": knobs,
        "active_drafter": view.spec.as_ref().map(|s| s.label.clone()),
        "best_k": view.spec.as_ref().and_then(|s| s.k),
        "spec_accept_rate": sum.spec_accept_rate,
        "spec_tokens_per_forward": sum.spec_tokens_per_forward,
        "k_tuner": rates,
    })
}

async fn handle_get_draft_config(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(draft_status_json(&state))
}

async fn handle_set_draft_config(
    State(state): State<AppState>,
    Json(update): Json<DraftConfigUpdate>,
) -> Json<serde_json::Value> {
    let mut knobs = state.scheduler.draft_knobs();
    if let Some(m) = update.mode {
        knobs.mode = m;
    }
    if let Some(k) = update.max_k {
        knobs.max_k = k;
    }
    if let Some(a) = update.target_acceptance {
        knobs.target_acceptance = a;
    }
    if let Some(c) = update.cost_weight {
        knobs.cost_weight = c;
    }
    if let Some(s) = update.suffix_min_match {
        knobs.suffix_min_match = s;
    }
    if let Some(lr) = update.noise_lr {
        knobs.noise_lr = lr;
    }
    if let Some(sig) = update.noise_sigma {
        knobs.noise_sigma = sig;
    }
    let knobs = knobs.sanitized();
    state.scheduler.set_draft_knobs(knobs);
    let mut s = crate::tui::settings::load_settings();
    s.draft = knobs;
    let _ = crate::tui::settings::save_settings(&s);
    Json(draft_status_json(&state))
}

/// `/v1/compact`: chat messages (OpenAI parts allowed) plus the chat request's `compact` fields.
#[derive(Debug, Deserialize)]
pub struct CompactRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(flatten)]
    pub compact: crate::compact::CompactionOverride,
    /// Context to compact for; default the active model's.
    #[serde(default)]
    pub context_limit: Option<usize>,
    #[serde(default)]
    pub enable_snapcompact_vision: Option<bool>,
}

/// Compacts now (mode default `aggressive`); with `trigger`, only once the history is over it.
async fn handle_compact(
    State(state): State<AppState>,
    Json(req): Json<CompactRequest>,
) -> axum::response::Response {
    match compact_now(state, req).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => api_error(e.status, format!("compaction failed: {}", e.message)),
    }
}

async fn compact_now(
    state: AppState,
    req: CompactRequest,
) -> Result<serde_json::Value, core::ApiError> {
    req.compact.check_local().map_err(core::ApiError::bad)?;
    let mut cfg = crate::tui::settings::load_settings().compact;
    cfg.mode = crate::compact::CompactionMode::Aggressive;
    req.compact.apply(&mut cfg);
    if let Some(v) = req.enable_snapcompact_vision {
        cfg.enable_snapcompact_vision = v;
    }
    let force = req.compact.trigger.is_none();
    let (mut messages, mut media) = core::prepare_messages(req.messages).await?;
    let sched = state.scheduler.clone();
    let ctx = req.context_limit;
    let c = tokio::task::spawn_blocking(move || {
        crate::compact::compact_for_scheduler(&sched, &mut messages, &mut media, ctx, &cfg, force)
            .map(|report| core::Compaction {
                report,
                messages,
                media,
            })
    })
    .await
    .map_err(|e| core::ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: e.to_string(),
    })?
    .map_err(core::ApiError::bad)?;
    Ok(serde_json::json!({
        "messages": c.wire_messages(),
        "media_count": c.media.len(),
        "report": c.report,
    }))
}

/// `GET /v1/artifacts/{hex}`: a block compaction spilled (`[… artifact {hex} …]`), as text.
async fn handle_artifact(
    axum::extract::Path(hex): axum::extract::Path<String>,
) -> axum::response::Response {
    match crate::compact::read_artifact(&hex) {
        Some(text) => text.into_response(),
        None => api_error(StatusCode::NOT_FOUND, format!("no artifact '{hex}'")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::PagedKVPool;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn create_test_state() -> AppState {
        AppState {
            scheduler: Arc::new(ContinuousBatchScheduler::new(
                "",
                Arc::new(PagedKVPool::new(64)),
                None,
            )),
            telemetry: EngineTelemetry::new(),
            jev_model: Default::default(),
            image_model: Default::default(),
            classifier: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_models_endpoint_lists_only_loaded_models() {
        let state = create_test_state();
        let app = create_router(state);

        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(val["object"], "list");
        // Nothing is loaded in the test state, so nothing may be advertised.
        assert_eq!(val["data"].as_array().unwrap().len(), 0);
    }

    async fn post_status(uri: &str, body: serde_json::Value) -> StatusCode {
        let app = create_router(create_test_state());
        let req = Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        app.oneshot(req).await.unwrap().status()
    }

    #[tokio::test]
    async fn system1_endpoints_reject_without_encoder() {
        let decisions = serde_json::json!({
            "state": "reactor overheating",
            "questions": [{"type": "bool", "id": "q", "instructions": "", "criteria_true": "danger", "criteria_false": "safe"}]
        });
        assert_eq!(
            post_status("/v1/embeddings", serde_json::json!({"input": "cat"})).await,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            post_status("/v1/decisions", decisions).await,
            StatusCode::BAD_REQUEST
        );
    }

    async fn post_body(uri: &str, body: serde_json::Value) -> String {
        let app = create_router(create_test_state());
        let req = Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn chat_tools_without_encoder_reach_llm_path() {
        let tools = serde_json::json!({
            "messages": [{"role": "user", "content": "Cancel my plan"}],
            "tools": [{"type": "function", "function": {"name": "cancel_subscription"}}]
        });
        let body = post_body("/v1/chat/completions", tools).await;
        assert!(!body.contains("System 1"), "{body}");
        assert!(
            body.contains("No text generation model is loaded"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn chat_model_id_system1_is_not_special() {
        let req = serde_json::json!({
            "model": "system1",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let body = post_body("/v1/chat/completions", req).await;
        assert!(!body.contains("System 1"), "{body}");
        assert!(
            body.contains("No text generation model is loaded"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn image_endpoints_reject_without_sd_pipeline() {
        let gen = serde_json::json!({"prompt": "a red fox"});
        assert_eq!(
            post_status("/v1/images/generations", gen).await,
            StatusCode::BAD_REQUEST
        );
        let edit = serde_json::json!({"prompt": "x", "image": "not a png"});
        assert_eq!(
            post_status("/v1/images/edits", edit).await,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(parse_size(Some("512x512")).unwrap(), Some((512, 512)));
        assert_eq!(parse_size(Some("auto")).unwrap(), None);
        assert!(parse_size(Some("4096x512")).is_err());
        assert!(parse_size(Some("big")).is_err());
    }

    /// The server binds 0.0.0.0: an edit `image` naming a server-side file must never be read.
    #[tokio::test]
    async fn image_edit_never_reads_server_paths() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("secret.png");
        std::fs::write(
            &path,
            crate::model::image::ImageBuffer::new(64, 64).to_png(),
        )
        .unwrap();
        let edit = serde_json::json!({"prompt": "x", "image": path.to_string_lossy()});
        let body = post_body("/v1/images/edits", edit).await;
        assert!(body.contains("Invalid input image"), "{body}");
    }

    #[tokio::test]
    async fn draft_config_endpoint_updates_live_knobs() {
        let payload = serde_json::json!({
            "mode": "noise",
            "max_k": 6,
            "target_acceptance": 0.72,
            "cost_weight": 0.4,
            "suffix_min_match": 3,
            "noise_lr": 0.008
        });
        let body = post_body("/v1/draft/config", payload).await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["knobs"]["mode"], "noise");
        assert_eq!(v["knobs"]["max_k"], 6);
        assert_eq!(v["knobs"]["suffix_min_match"], 3);
    }

    /// A served synthetic model (one token per character, never stops before `max_tokens`) with
    /// a thread stepping its scheduler while the state lives.
    fn model_state(dir: &std::path::Path) -> AppState {
        let cfg = serde_json::json!({"max_position_embeddings": 16384});
        let model = crate::arch::testing::synthetic_lm(dir, cfg, crate::hardware::Device::Cpu);
        let scheduler = Arc::new(ContinuousBatchScheduler::new(
            "gibbs/test-llama",
            Arc::new(PagedKVPool::new(1024)),
            Some(model),
        ));
        let weak = Arc::downgrade(&scheduler);
        std::thread::spawn(move || {
            while let Some(s) = weak.upgrade() {
                if s.step().active_sequences_count == 0 {
                    s.wait_for_work(std::time::Duration::from_millis(5));
                }
            }
        });
        AppState {
            scheduler,
            ..create_test_state()
        }
    }

    #[tokio::test]
    async fn compact_endpoint_elides_thinking_and_folds_turns() {
        let dir = tempfile::tempdir().unwrap();
        let payload = serde_json::json!({
            "messages": [
                {"role": "user", "content": "Turn 1 user question"},
                {"role": "assistant", "content": "<think>Very long reasoning trace...</think>\nTurn 1 answer"},
                {"role": "user", "content": "Turn 2 user question"},
                {"role": "assistant", "content": "<think>Second reasoning trace...</think>\nTurn 2 answer"},
                {"role": "user", "content": "Turn 3 user question"},
                {"role": "assistant", "content": "Turn 3 answer"}
            ],
            "mode": "aggressive",
            "keep_recent_turns": 1
        });
        let app = create_router(model_state(dir.path()));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/compact")
            .header("content-type", "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v["report"]["tier1_chars_elided"].as_u64().unwrap() > 0, "{v}");
        assert_eq!(v["report"]["tier3_turns_summarized"], 2, "{v}");
        let m = v["messages"].as_array().unwrap();
        assert_eq!(m.len(), 4, "{v}");
        assert!(m[0]["content"]
            .as_str()
            .unwrap()
            .starts_with("[Compacted session summary of 4 earlier messages]\n"));
        assert_eq!(m[3]["content"], "Turn 3 answer");
        // Without a model there is nothing to summarize with (a transcript with no cached summary).
        let fresh = payload.to_string().replace("Turn 1 user", "Turn 1 other");
        let body = post_body("/v1/compact", serde_json::from_str(&fresh).unwrap()).await;
        assert!(body.contains("no text model is loaded"), "{body}");
    }

    /// A chat request's `compact` object compacts its long history before generation: the
    /// response carries the `x-gibbs-compacted` header, `usage.compaction` and (asked for) the
    /// compacted history; spilled blocks are served from `/v1/artifacts`.
    #[tokio::test]
    async fn chat_compact_field_sets_usage_and_header() {
        let dir = tempfile::tempdir().unwrap();
        let log: String = (0..30)
            .map(|i| format!("chat_compact build step {i}: ok"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut messages = vec![serde_json::json!({"role": "system", "content": "Be terse."})];
        for i in 0..20 {
            let user = match i {
                0 => format!("Here is the log:\n\n{log}"),
                _ => format!("Turn {i}: {}", "explain the second law in detail. ".repeat(4)),
            };
            let answer = format!(
                "<think>reasoning {i}</think>Answer {i}: {}",
                "entropy of an isolated system never decreases. ".repeat(3)
            );
            messages.push(serde_json::json!({"role": "user", "content": user}));
            messages.push(serde_json::json!({"role": "assistant", "content": answer}));
        }
        messages.push(serde_json::json!({"role": "user", "content": "And the third law?"}));
        let payload = serde_json::json!({
            "messages": messages,
            "max_tokens": 4,
            "compact": {"mode": "aggressive", "trigger": 0.3},
            "compact_return_messages": true
        });
        let app = create_router(model_state(dir.path()));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let header = resp.headers()["x-gibbs-compacted"]
            .to_str()
            .unwrap()
            .to_string();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let report = &v["usage"]["compaction"];
        let (before, after) = (report["before_tokens"].as_u64().unwrap(), report["after_tokens"].as_u64().unwrap());
        assert_eq!(header, format!("{before}->{after}"));
        assert!(after < before, "{report}");
        assert!(report["tier1_chars_elided"].as_u64().unwrap() > 0, "{report}");
        assert!(report["tier3_turns_summarized"].as_u64().unwrap() > 0, "{report}");
        let m = v["gibbs_compacted_messages"].as_array().unwrap();
        assert_eq!(m[0]["role"], "system");
        assert!(m[1]["content"]
            .as_str()
            .unwrap()
            .starts_with("[Compacted session summary of "));
        assert_eq!(m.last().unwrap()["content"], "And the third law?");
        // The 30-line log was spilled whole before its turn was folded.
        let hex = report["artifacts"][0].as_str().unwrap();
        let req = Request::builder()
            .uri(format!("/v1/artifacts/{hex}"))
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), log);
        let req = Request::builder()
            .uri("/v1/artifacts/..%2Fsettings")
            .body(Body::empty())
            .unwrap();
        assert_eq!(app.oneshot(req).await.unwrap().status(), StatusCode::NOT_FOUND);
    }
}
