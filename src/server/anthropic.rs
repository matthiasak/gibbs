//! Anthropic Messages API `/v1/messages` (+ `count_tokens`) on the shared core. Any model
//! name serves the loaded model and auth headers are ignored, so Claude Code works with
//! `ANTHROPIC_BASE_URL` pointed here.

use super::core::{
    self, collect, label, prepare_messages, stream_response, ApiError, Event, Finish, Framing,
    GenRequest, Prompt, ToolChoice, Wire,
};
use super::tools::ToolCall;
use super::AppState;
use crate::model::hf_loader::ChatMessage;
use crate::scheduler::SamplingParams;
use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: Option<String>,
    pub max_tokens: Option<usize>,
    pub system: Option<Value>,
    #[serde(default)]
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub tools: Vec<Value>,
    pub tool_choice: Option<Value>,
    pub stop_sequences: Option<Vec<String>>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: Value,
}

fn error(e: ApiError) -> Response {
    (e.status, Json(error_body(e.status, &e.message))).into_response()
}

fn error_body(status: StatusCode, message: &str) -> Value {
    let kind = if status.is_server_error() {
        "api_error"
    } else {
        "invalid_request_error"
    };
    json!({"type": "error", "error": {"type": kind, "message": message}})
}

fn parse<T>(body: Result<Json<T>, JsonRejection>) -> Result<T, Response> {
    body.map(|Json(r)| r)
        .map_err(|e| error(ApiError::bad(e.body_text())))
}

/// `text` / `image` blocks as OpenAI content parts (images as data or http URLs).
fn part(b: &Value) -> Result<Value, ApiError> {
    match b["type"].as_str() {
        Some("text") => Ok(json!({"type": "text", "text": b["text"]})),
        Some("image") => {
            let s = &b["source"];
            let url = match s["type"].as_str() {
                Some("base64") => format!(
                    "data:{};base64,{}",
                    s["media_type"].as_str().unwrap_or("image/png"),
                    s["data"].as_str().unwrap_or_default()
                ),
                Some("url") => s["url"].as_str().unwrap_or_default().to_string(),
                other => {
                    return Err(ApiError::bad(format!(
                        "unsupported image source '{}'",
                        other.unwrap_or("?")
                    )))
                }
            };
            Ok(json!({"type": "image_url", "image_url": {"url": url}}))
        }
        other => Err(ApiError::bad(format!(
            "unsupported content block '{}' (text, image, tool_use, tool_result)",
            other.unwrap_or("?")
        ))),
    }
}

/// `system` + Anthropic turns as OpenAI-shaped chat messages: `tool_use` blocks become
/// assistant `tool_calls`, each `tool_result` a `role: "tool"` message before the user's text.
fn chat_messages(
    system: Option<Value>,
    messages: Vec<AnthropicMessage>,
) -> Result<Vec<ChatMessage>, ApiError> {
    let mut out = Vec::new();
    match system {
        Some(Value::String(s)) => out.push(ChatMessage::new("system", s)),
        Some(Value::Array(blocks)) => {
            let text: Vec<&str> = blocks.iter().filter_map(|b| b["text"].as_str()).collect();
            out.push(ChatMessage::new("system", text.join("\n")));
        }
        _ => {}
    }
    for m in messages {
        let blocks = match m.content {
            Value::String(s) => {
                out.push(ChatMessage::new(m.role, s));
                continue;
            }
            Value::Array(b) => b,
            _ => return Err(ApiError::bad("message content must be a string or blocks")),
        };
        let (mut parts, mut calls) = (Vec::new(), Vec::new());
        for b in &blocks {
            match b["type"].as_str() {
                Some("tool_use") => calls.push(json!({"id": b["id"], "type": "function",
                    "function": {"name": b["name"], "arguments": b["input"]}})),
                Some("tool_result") => {
                    let content = match &b["content"] {
                        Value::Array(inner) => {
                            Value::Array(inner.iter().map(part).collect::<Result<_, _>>()?)
                        }
                        c => c.clone(),
                    };
                    out.push(ChatMessage {
                        role: "tool".into(),
                        content,
                        tool_calls: None,
                        tool_call_id: b["tool_use_id"].as_str().map(String::from),
                        name: None,
                    });
                }
                // Earlier turns' reasoning: Anthropic itself drops it from the prompt.
                Some("thinking" | "redacted_thinking") => {}
                _ => parts.push(part(b)?),
            }
        }
        if !parts.is_empty() || !calls.is_empty() {
            out.push(ChatMessage {
                role: m.role,
                content: if parts.is_empty() {
                    Value::Null
                } else {
                    Value::Array(parts)
                },
                tool_calls: (!calls.is_empty()).then(|| Value::Array(calls)),
                tool_call_id: None,
                name: None,
            });
        }
    }
    Ok(out)
}

/// Client tools (`input_schema`) in Chat Completions shape, and the names of server tools
/// (`web_search_*`, `bash_*`, ...) the model cannot call.
fn chat_tools(tools: Vec<Value>) -> (Vec<Value>, Vec<String>) {
    let (mut kept, mut skipped) = (Vec::new(), Vec::new());
    for t in tools {
        match (t["name"].as_str(), t.get("input_schema")) {
            (Some(name), Some(schema)) => kept.push(json!({"type": "function", "function": {
                "name": name, "description": t["description"], "parameters": schema}})),
            (name, _) => skipped.push(name.or(t["type"].as_str()).unwrap_or("?").to_string()),
        }
    }
    (kept, skipped)
}

fn tool_choice(v: Option<&Value>) -> Result<ToolChoice, ApiError> {
    let Some(v) = v else {
        return Ok(ToolChoice::Auto);
    };
    match v["type"].as_str() {
        Some("auto") => Ok(ToolChoice::Auto),
        Some("none") => Ok(ToolChoice::None),
        Some("any") => Ok(ToolChoice::Required),
        Some("tool") => v["name"]
            .as_str()
            .map(|n| ToolChoice::Named(n.into()))
            .ok_or_else(|| ApiError::bad("tool_choice type 'tool' needs a name")),
        _ => Err(ApiError::bad(
            "tool_choice.type must be auto, any, tool, or none",
        )),
    }
}

async fn prompt(
    state: &AppState,
    req: &mut MessagesRequest,
) -> Result<(Prompt, Vec<crate::model::native::MediaInput>), ApiError> {
    let messages = chat_messages(req.system.take(), std::mem::take(&mut req.messages))?;
    let (messages, media) = prepare_messages(messages).await?;
    let (tools, skipped) = chat_tools(std::mem::take(&mut req.tools));
    let choice = tool_choice(req.tool_choice.as_ref())?;
    core::skip_builtin_tools(state, "anthropic", &skipped, &choice)?;
    Ok((
        Prompt::Chat {
            messages,
            tools,
            tool_choice: choice,
            template_kwargs: Value::Null,
        },
        media,
    ))
}

pub async fn count_tokens(
    State(state): State<AppState>,
    body: Result<Json<MessagesRequest>, JsonRejection>,
) -> Response {
    let mut req = match parse(body) {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    let counted = match prompt(&state, &mut req).await {
        Ok((p, media)) => core::count_tokens(&state, p, media).await,
        Err(e) => Err(e),
    };
    counted
        .map(|n| Json(json!({ "input_tokens": n })).into_response())
        .unwrap_or_else(error)
}

pub async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<MessagesRequest>, JsonRejection>,
) -> Response {
    let req = match parse(body) {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    run(state, headers, req).await.unwrap_or_else(error)
}

async fn run(
    state: AppState,
    headers: HeaderMap,
    mut req: MessagesRequest,
) -> Result<Response, ApiError> {
    let (prompt, media) = prompt(&state, &mut req).await?;
    let gen = core::start(
        &state,
        GenRequest {
            prompt,
            media,
            params: SamplingParams {
                top_p: req.top_p,
                ..SamplingParams::from_request(req.temperature, req.max_tokens)
            },
            stop: req.stop_sequences.unwrap_or_default(),
            label: label("anthropic", &headers),
            compact: None,
        },
    )
    .await?;
    let model = req.model.unwrap_or_else(|| state.scheduler.model_id());
    let wire = MessagesWire::new(gen.id, model, gen.prompt_tokens);
    if req.stream.unwrap_or(false) {
        return Ok(stream_response(gen, wire, Framing::SseNamed));
    }
    let out = collect(gen).await?;
    let mut content: Vec<Value> = Vec::new();
    if !out.text.is_empty() {
        content.push(json!({"type": "text", "text": out.text}));
    }
    content.extend(out.calls.iter().map(tool_use));
    let mut msg = wire.message(content);
    let (reason, seq) = stop_reason(&out.finish);
    msg["stop_reason"] = reason.into();
    msg["stop_sequence"] = seq.into();
    msg["usage"]["output_tokens"] = out.completion_tokens.into();
    Ok(Json(msg).into_response())
}

fn tool_use(c: &ToolCall) -> Value {
    let input: Value = serde_json::from_str(&c.arguments).unwrap_or_else(|_| json!({}));
    json!({"type": "tool_use", "id": format!("toolu_{}", c.id), "name": c.name, "input": input})
}

fn stop_reason(f: &Finish) -> (&'static str, Option<String>) {
    match f {
        Finish::Stop => ("end_turn", None),
        Finish::Length => ("max_tokens", None),
        Finish::ToolCalls => ("tool_use", None),
        Finish::StopSequence(s) => ("stop_sequence", Some(s.clone())),
    }
}

pub struct MessagesWire {
    id: String,
    model: String,
    prompt_tokens: usize,
    /// Content blocks started so far.
    blocks: usize,
    text_open: bool,
}

impl MessagesWire {
    pub fn new(id: u64, model: String, prompt_tokens: usize) -> Self {
        Self {
            id: format!("msg_{id}"),
            model,
            prompt_tokens,
            blocks: 0,
            text_open: false,
        }
    }

    fn message(&self, content: Vec<Value>) -> Value {
        json!({"id": self.id, "type": "message", "role": "assistant", "model": self.model,
               "content": content, "stop_reason": null, "stop_sequence": null,
               "usage": {"input_tokens": self.prompt_tokens, "output_tokens": 0}})
    }

    fn close_text(&mut self, out: &mut Vec<Value>) {
        if std::mem::take(&mut self.text_open) {
            out.push(json!({"type": "content_block_stop", "index": self.blocks - 1}));
        }
    }
}

impl Wire for MessagesWire {
    fn open(&mut self) -> Vec<Value> {
        vec![
            json!({"type": "message_start", "message": self.message(Vec::new())}),
            json!({"type": "ping"}),
        ]
    }

    fn on(&mut self, ev: Event) -> Vec<Value> {
        let mut out = Vec::new();
        match ev {
            Event::Text(t) => {
                if !self.text_open {
                    self.text_open = true;
                    self.blocks += 1;
                    out.push(
                        json!({"type": "content_block_start", "index": self.blocks - 1,
                                    "content_block": {"type": "text", "text": ""}}),
                    );
                }
                out.push(
                    json!({"type": "content_block_delta", "index": self.blocks - 1,
                                "delta": {"type": "text_delta", "text": t}}),
                );
            }
            Event::Call(c) => {
                self.close_text(&mut out);
                let i = self.blocks;
                self.blocks += 1;
                let mut block = tool_use(&c);
                block["input"] = json!({});
                out.push(
                    json!({"type": "content_block_start", "index": i, "content_block": block}),
                );
                out.push(json!({"type": "content_block_delta", "index": i,
                                "delta": {"type": "input_json_delta", "partial_json": c.arguments}}));
                out.push(json!({"type": "content_block_stop", "index": i}));
            }
            Event::Done {
                finish,
                completion_tokens,
            } => {
                self.close_text(&mut out);
                let (reason, seq) = stop_reason(&finish);
                out.push(json!({"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": seq},
                                "usage": {"output_tokens": completion_tokens}}));
                out.push(json!({"type": "message_stop"}));
            }
            Event::Error(e) => out.push(error_body(StatusCode::INTERNAL_SERVER_ERROR, &e)),
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_event_order_text_then_tool_use() {
        let mut w = MessagesWire::new(1, "claude-x".into(), 10);
        let mut evs = w.open();
        for ev in [
            Event::Text("Checking".into()),
            Event::Text(" now.".into()),
            Event::Call(ToolCall {
                id: "q".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"Paris\"}".into(),
            }),
            Event::Done {
                finish: Finish::ToolCalls,
                completion_tokens: 12,
            },
        ] {
            evs.extend(w.on(ev));
        }
        let types: Vec<&str> = evs.iter().map(|e| e["type"].as_str().unwrap()).collect();
        assert_eq!(
            types,
            [
                "message_start",
                "ping",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(evs[2]["index"], 0);
        assert_eq!(evs[6]["index"], 1);
        assert_eq!(evs[6]["content_block"]["id"], "toolu_q");
        assert_eq!(evs[7]["delta"]["partial_json"], "{\"city\":\"Paris\"}");
        assert_eq!(evs[9]["delta"]["stop_reason"], "tool_use");
        assert_eq!(evs[9]["usage"]["output_tokens"], 12);
    }

    #[test]
    fn blocks_become_chat_turns() {
        let msgs: Vec<AnthropicMessage> = serde_json::from_value(json!([
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": [{"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "18C"},
                {"type": "text", "text": "and tomorrow?"}]}
        ]))
        .unwrap();
        let system = json!([{"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}]);
        let m = chat_messages(Some(system), msgs).unwrap();
        let roles: Vec<&str> = m.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "user", "assistant", "tool", "user"]);
        assert_eq!(
            m[2].tool_calls.as_ref().unwrap()[0]["function"]["arguments"]["city"],
            "Paris"
        );
        assert_eq!(m[3].tool_call_id.as_deref(), Some("t1"));
        assert_eq!(
            tool_choice(Some(&json!({"type": "any"}))).unwrap(),
            ToolChoice::Required
        );
        let (kept, skipped) = chat_tools(vec![
            json!({"type": "web_search_20250305", "name": "web_search"}),
            json!({"name": "get_weather", "input_schema": {"type": "object"}}),
        ]);
        assert_eq!(kept.len(), 1);
        assert_eq!(skipped, ["web_search"]);
    }
}
