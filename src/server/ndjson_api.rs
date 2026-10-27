//! NDJSON local-agent API (`/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/ps`,
//! `/api/version`, `/`) on the shared core.

use super::core::{
    self, collect, label, prepare_messages, stream_response, ApiError, Event, Finish, Framing,
    GenRequest, Prompt, ToolChoice, Wire,
};
use super::tools::{ToolCall, ToolFormat};
use super::AppState;
use crate::model::hf_loader::ChatMessage;
use crate::scheduler::SamplingParams;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Json, Response};
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Instant;

/// API level this server mirrors (clients gate features on it).
const COMPAT_VERSION: &str = "0.35.1";

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Vec<ChatTurn>,
    #[serde(default)]
    pub tools: Vec<Value>,
    pub stream: Option<bool>,
    pub format: Option<Value>,
    #[serde(default)]
    pub options: Options,
    /// `false` renders the template with `enable_thinking: false`.
    pub think: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ChatTurn {
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub images: Vec<String>,
    pub tool_calls: Option<Vec<Value>>,
    pub tool_name: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct Options {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    /// Tokens to generate; negative or unset = until the model stops.
    pub num_predict: Option<i64>,
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub model: Option<String>,
    #[serde(default)]
    pub prompt: String,
    pub system: Option<String>,
    #[serde(default)]
    pub images: Vec<String>,
    #[serde(default)]
    pub raw: bool,
    pub stream: Option<bool>,
    pub format: Option<Value>,
    pub suffix: Option<String>,
    pub context: Option<Vec<u32>>,
    #[serde(default)]
    pub options: Options,
}

fn error(e: ApiError) -> Response {
    (e.status, Json(json!({ "error": e.message }))).into_response()
}

/// Accepts a JSON body whatever its `Content-Type` (clients may use bare `curl -d`).
fn parse<T: serde::de::DeserializeOwned>(body: &[u8]) -> Result<T, ApiError> {
    serde_json::from_slice(body).map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))
}

impl Options {
    fn params(&self) -> SamplingParams {
        // Negative `num_predict` ("unlimited") is the same as unset.
        let max = self.num_predict.and_then(|n| usize::try_from(n).ok());
        SamplingParams {
            top_p: self.top_p,
            ..SamplingParams::from_request(self.temperature, max)
        }
    }
}

fn no_format(format: &Option<Value>) -> Result<(), ApiError> {
    match format {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(s)) if s.is_empty() => Ok(()),
        Some(_) => Err(ApiError::bad(
            "format (JSON / schema output) is not supported",
        )),
    }
}

/// A user/assistant turn with base64 `images` as image parts ahead of the text.
fn with_images(role: String, text: String, images: &[String]) -> ChatMessage {
    let mut m = ChatMessage::new(role, text);
    if !images.is_empty() {
        let mut parts: Vec<Value> = images
            .iter()
            .map(|b| json!({"type": "image_url", "image_url": {"url": format!("data:image;base64,{b}")}}))
            .collect();
        parts.push(json!({"type": "text", "text": m.text()}));
        m.content = Value::Array(parts);
    }
    m
}

fn chat_messages(messages: Vec<ChatTurn>) -> Result<Vec<ChatMessage>, ApiError> {
    messages
        .into_iter()
        .map(|m| {
            let mut c = with_images(m.role, m.content, &m.images);
            c.tool_calls = match m.tool_calls {
                None => None,
                Some(calls) => Some(
                    calls
                        .into_iter()
                        .map(|mut t| match t.as_object_mut() {
                            Some(o) => {
                                o.insert("type".into(), json!("function"));
                                Ok(t)
                            }
                            None => Err(ApiError::bad(format!(
                                "tool_calls entries must be objects, got {t}"
                            ))),
                        })
                        .collect::<Result<_, _>>()?,
                ),
            };
            c.name = m.tool_name;
            Ok(c)
        })
        .collect()
}

/// `/api/generate` options this server cannot honor are a 400, never silently dropped.
fn check_generate(req: &GenerateRequest) -> Result<(), ApiError> {
    no_format(&req.format)?;
    if req.suffix.as_deref().is_some_and(|s| !s.is_empty()) {
        return Err(ApiError::bad(
            "suffix (fill-in-the-middle) is not supported",
        ));
    }
    if req.context.as_ref().is_some_and(|c| !c.is_empty()) {
        return Err(ApiError::bad(
            "context is not supported; send the conversation with /api/chat",
        ));
    }
    if req.raw && (!req.images.is_empty() || req.system.is_some()) {
        return Err(ApiError::bad(
            "raw prompts are sent as is: images and system need the chat template (raw: false)",
        ));
    }
    Ok(())
}

fn model_name(state: &AppState, requested: Option<String>) -> String {
    requested
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| state.scheduler.model_id())
}

/// Reply to a request with nothing to generate: clients send these to preload a model.
fn load_reply(model: &str, key: &str) -> Response {
    let mut v = json!({"model": model, "created_at": now(), "done": true, "done_reason": "load"});
    v[key] = if key == "message" {
        json!({"role": "assistant", "content": ""})
    } else {
        json!("")
    };
    Json(v).into_response()
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Nanos, true)
}

pub async fn chat(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    async {
        let req: ChatRequest = parse(&body)?;
        no_format(&req.format)?;
        let model = model_name(&state, req.model.clone());
        if req.messages.is_empty() {
            return Ok(load_reply(&model, "message"));
        }
        let (messages, media) = prepare_messages(chat_messages(req.messages)?).await?;
        let gen = core::start(
            &state,
            GenRequest {
                prompt: Prompt::Chat {
                    messages,
                    tools: req.tools,
                    tool_choice: ToolChoice::Auto,
                    template_kwargs: req
                        .think
                        .map_or(Value::Null, |t| json!({"enable_thinking": t})),
                },
                media,
                params: req.options.params(),
                stop: req.options.stop.clone().unwrap_or_default(),
                label: label(
                    crate::arch::compat()["ndjson_api"]["label"]
                        .as_str()
                        .expect("compat.ndjson_api.label"),
                    &headers,
                ),
                compact: None,
            },
        )
        .await?;
        let mut wire = NdjsonWire::new(model, "message", gen.prompt_tokens);
        if req.stream.unwrap_or(true) {
            return Ok(stream_response(gen, wire, Framing::Ndjson));
        }
        let out = collect(gen).await?;
        let mut v = wire.chunk(&out.text);
        if !out.calls.is_empty() {
            v["message"]["tool_calls"] = out.calls.iter().map(call_json).collect();
        }
        wire.finish(&mut v, &out.finish, out.completion_tokens);
        Ok(Json(v).into_response())
    }
    .await
    .unwrap_or_else(error)
}

pub async fn generate(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    async {
        let req: GenerateRequest = parse(&body)?;
        check_generate(&req)?;
        let model = model_name(&state, req.model.clone());
        if req.prompt.is_empty() && req.images.is_empty() {
            return Ok(load_reply(&model, "response"));
        }
        let (prompt, media) = if req.raw {
            (Prompt::Raw(req.prompt), Vec::new())
        } else {
            let mut messages: Vec<ChatMessage> = req
                .system
                .map(|s| ChatMessage::new("system", s))
                .into_iter()
                .collect();
            messages.push(with_images("user".into(), req.prompt, &req.images));
            let (messages, media) = prepare_messages(messages).await?;
            (
                Prompt::Chat {
                    messages,
                    tools: Vec::new(),
                    tool_choice: ToolChoice::Auto,
                    template_kwargs: Value::Null,
                },
                media,
            )
        };
        let gen = core::start(
            &state,
            GenRequest {
                prompt,
                media,
                params: req.options.params(),
                stop: req.options.stop.clone().unwrap_or_default(),
                label: label(
                    crate::arch::compat()["ndjson_api"]["label"]
                        .as_str()
                        .expect("compat.ndjson_api.label"),
                    &headers,
                ),
                compact: None,
            },
        )
        .await?;
        let mut wire = NdjsonWire::new(model, "response", gen.prompt_tokens);
        if req.stream.unwrap_or(true) {
            return Ok(stream_response(gen, wire, Framing::Ndjson));
        }
        let out = collect(gen).await?;
        let mut v = wire.chunk(&out.text);
        wire.finish(&mut v, &out.finish, out.completion_tokens);
        Ok(Json(v).into_response())
    }
    .await
    .unwrap_or_else(error)
}

fn call_json(c: &ToolCall) -> Value {
    let args: Value = serde_json::from_str(&c.arguments).unwrap_or_else(|_| json!({}));
    json!({"function": {"name": c.name, "arguments": args}})
}

pub struct NdjsonWire {
    model: String,
    /// `message` (`/api/chat`) or `response` (`/api/generate`).
    key: &'static str,
    prompt_tokens: usize,
    started: Instant,
}

impl NdjsonWire {
    pub fn new(model: String, key: &'static str, prompt_tokens: usize) -> Self {
        Self {
            model,
            key,
            prompt_tokens,
            started: Instant::now(),
        }
    }

    fn chunk(&self, text: &str) -> Value {
        let mut v = json!({"model": self.model, "created_at": now(), "done": false});
        v[self.key] = if self.key == "message" {
            json!({"role": "assistant", "content": text})
        } else {
            json!(text)
        };
        v
    }

    fn finish(&mut self, v: &mut Value, finish: &Finish, completion_tokens: usize) {
        let ns = self.started.elapsed().as_nanos() as u64;
        v["done"] = json!(true);
        v["done_reason"] = json!(if *finish == Finish::Length {
            "length"
        } else {
            "stop"
        });
        v["total_duration"] = ns.into();
        v["load_duration"] = 0.into();
        v["prompt_eval_count"] = self.prompt_tokens.into();
        v["eval_count"] = completion_tokens.into();
        v["eval_duration"] = ns.into();
    }
}

impl Wire for NdjsonWire {
    fn on(&mut self, ev: Event) -> Vec<Value> {
        match ev {
            Event::Text(t) => vec![self.chunk(&t)],
            Event::Call(c) => {
                let mut v = self.chunk("");
                v["message"]["tool_calls"] = json!([call_json(&c)]);
                vec![v]
            }
            Event::Done {
                finish,
                completion_tokens,
            } => {
                let mut v = self.chunk("");
                self.finish(&mut v, &finish, completion_tokens);
                vec![v]
            }
            Event::Error(e) => vec![json!({ "error": e })],
        }
    }
}

/// The loaded model as a model entry (`/api/tags`, `/api/ps`).
fn model_entry(state: &AppState) -> Value {
    let id = state.scheduler.model_id();
    let family = id
        .rsplit('/')
        .next()
        .unwrap_or(&id)
        .split('-')
        .next()
        .unwrap_or_default()
        .to_lowercase();
    // Clients show `digest[..12]`; a hash of the id identifies the model (no manifest here).
    let digest = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        id.hash(&mut h);
        format!("{:016x}", h.finish())
    };
    json!({"name": id, "model": id, "modified_at": now(), "size": 0, "digest": digest,
           "details": {"parent_model": "", "format": "gibbs", "family": family, "families": [family],
                       "parameter_size": "", "quantization_level": ""}})
}

fn loaded(state: &AppState) -> Vec<Value> {
    if core::text_model_active(state) {
        vec![model_entry(state)]
    } else {
        Vec::new()
    }
}

pub async fn tags(State(state): State<AppState>) -> Json<Value> {
    Json(json!({ "models": loaded(&state) }))
}

pub async fn ps(State(state): State<AppState>) -> Json<Value> {
    let models: Vec<Value> = loaded(&state)
        .into_iter()
        .map(|mut m| {
            m["expires_at"] = json!("2999-01-01T00:00:00Z");
            m["size_vram"] = 0.into();
            m
        })
        .collect();
    Json(json!({ "models": models }))
}

pub async fn show(State(state): State<AppState>) -> Response {
    if !core::text_model_active(&state) {
        return error(core::no_text_model(&state));
    }
    let tok = state.scheduler.tokenizer();
    let template = tok
        .chat_template
        .as_ref()
        .map(|t| t.template.clone())
        .unwrap_or_default();
    let mut capabilities = vec!["completion"];
    if ToolFormat::detect(&template).is_some() {
        capabilities.push("tools");
    }
    let entry = model_entry(&state);
    let arch = entry["details"]["family"].clone();
    Json(json!({
        "license": "", "modelfile": "", "parameters": "", "template": template,
        "details": entry["details"], "modified_at": entry["modified_at"],
        "model_info": {"general.architecture": arch},
        "capabilities": capabilities,
    }))
    .into_response()
}

pub async fn version() -> Json<Value> {
    Json(json!({ "version": COMPAT_VERSION }))
}

pub async fn root() -> &'static str {
    crate::arch::compat()["ndjson_api"]["root_banner"]
        .as_str()
        .expect("compat.ndjson_api.root_banner")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndjson_chat_order_text_call_done() {
        let mut w = NdjsonWire::new("m".into(), "message", 4);
        let evs: Vec<Value> = [
            Event::Text("ok".into()),
            Event::Call(ToolCall {
                id: "1".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"Paris\"}".into(),
            }),
            Event::Done {
                finish: Finish::ToolCalls,
                completion_tokens: 9,
            },
        ]
        .into_iter()
        .flat_map(|e| w.on(e))
        .collect();
        assert_eq!(evs.len(), 3);
        assert_eq!(
            (
                evs[0]["message"]["content"].as_str(),
                evs[0]["done"].as_bool()
            ),
            (Some("ok"), Some(false))
        );
        assert_eq!(
            evs[1]["message"]["tool_calls"][0]["function"]["arguments"]["city"],
            "Paris"
        );
        assert_eq!(
            (evs[2]["done"].as_bool(), evs[2]["done_reason"].as_str()),
            (Some(true), Some("stop"))
        );
        assert_eq!(
            (
                evs[2]["prompt_eval_count"].as_u64(),
                evs[2]["eval_count"].as_u64()
            ),
            (Some(4), Some(9))
        );
        let mut g = NdjsonWire::new("m".into(), "response", 1);
        let done = g.on(Event::Done {
            finish: Finish::Length,
            completion_tokens: 2,
        });
        assert_eq!(
            (
                done[0]["response"].as_str(),
                done[0]["done_reason"].as_str()
            ),
            (Some(""), Some("length"))
        );
    }

    #[tokio::test]
    async fn images_and_tool_turns_convert() {
        let msgs: Vec<ChatTurn> = serde_json::from_value(json!([
            {"role": "user", "content": "what is this?", "images": ["AAEC"]},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}]},
            {"role": "tool", "content": "42", "tool_name": "f"}
        ]))
        .unwrap();
        let (m, media) = prepare_messages(chat_messages(msgs).unwrap())
            .await
            .unwrap();
        assert_eq!(m[0].content[0]["type"], "image");
        assert_eq!(m[0].content[1]["text"], "what is this?");
        assert_eq!(media.len(), 1);
        assert_eq!(m[1].tool_calls.as_ref().unwrap()[0]["type"], "function");
        assert_eq!(m[2].name.as_deref(), Some("f"));
        // A non-object tool call is a 400, not a panic.
        let bad: Vec<ChatTurn> = serde_json::from_value(json!([
            {"role": "assistant", "content": "", "tool_calls": [1]}
        ]))
        .unwrap();
        assert!(chat_messages(bad).unwrap_err().message.contains("objects"));
    }

    #[test]
    fn generate_rejects_what_it_cannot_honor() {
        let req = |v: Value| serde_json::from_value::<GenerateRequest>(v).unwrap();
        assert!(check_generate(&req(json!({"prompt": "hi", "raw": true}))).is_ok());
        assert!(check_generate(&req(json!({"prompt": "hi", "system": "s"}))).is_ok());
        for bad in [
            json!({"prompt": "hi", "raw": true, "images": ["AAEC"]}),
            json!({"prompt": "hi", "raw": true, "system": "s"}),
            json!({"prompt": "hi", "suffix": "x"}),
            json!({"prompt": "hi", "format": "json"}),
        ] {
            assert!(check_generate(&req(bad.clone())).is_err(), "{bad}");
        }
    }

    #[tokio::test]
    async fn ndjson_compat_wire_strings_pinned() {
        assert_eq!(root().await, "Ollama is running");
        assert_eq!(COMPAT_VERSION, "0.35.1");
        assert_eq!(
            crate::arch::compat()["ndjson_api"]["host_env"].as_str(),
            Some("OLLAMA_HOST")
        );
        assert_eq!(
            crate::arch::compat()["ndjson_api"]["client_name"].as_str(),
            Some("Ollama")
        );
        assert_eq!(
            crate::arch::compat()["ndjson_api"]["label"].as_str(),
            Some("ollama")
        );
    }
}
