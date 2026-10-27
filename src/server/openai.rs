//! OpenAI `/v1/chat/completions` and `/v1/completions` on the shared core.

use super::core::{
    self, collect, label, prepare_messages, stream_response, unix_now, ApiError, Event, Finish,
    Framing, GenRequest, Prompt, ToolChoice, Wire,
};
use super::{api_error_body, AppState};
use crate::model::hf_loader::ChatMessage;
use crate::runtime::encoder::system1::{ChoiceOption, JevDecision};
use crate::scheduler::SamplingParams;
use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    pub stream: Option<bool>,
    pub stream_options: Option<Value>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub n: Option<usize>,
    pub stop: Option<Value>,
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,
    pub response_format: Option<Value>,
    /// vLLM/SGLang extension: extra chat-template variables (`{"enable_thinking": false}`).
    pub chat_template_kwargs: Option<Value>,
    /// Compaction settings for this request, over the saved ones.
    pub compact: Option<crate::compact::CompactionOverride>,
    /// Return the compacted history as `gibbs_compacted_messages` (when compaction ran).
    #[serde(default)]
    pub compact_return_messages: bool,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: Option<Value>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub n: Option<usize>,
    pub stop: Option<Value>,
    pub stream: Option<bool>,
    pub stream_options: Option<Value>,
}

pub fn error(e: ApiError) -> Response {
    (e.status, Json(api_error_body(e.status, e.message))).into_response()
}

/// `"stop"`-style fields: a string or a list of strings.
pub fn strings(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|s| s.as_str().map(String::from))
            .collect(),
        _ => Vec::new(),
    }
}

/// Chat Completions (`{"type":"function","function":{"name"}}`) and Responses
/// (`{"type":"function","name"}`) `tool_choice`.
pub fn tool_choice(v: Option<&Value>) -> Result<ToolChoice, ApiError> {
    match v {
        None | Some(Value::Null) => Ok(ToolChoice::Auto),
        Some(Value::String(s)) => match s.as_str() {
            "auto" => Ok(ToolChoice::Auto),
            "none" => Ok(ToolChoice::None),
            "required" => Ok(ToolChoice::Required),
            other => Err(ApiError::bad(format!("unknown tool_choice '{other}'"))),
        },
        Some(o) => o["function"]["name"]
            .as_str()
            .or_else(|| o["name"].as_str())
            .map(|n| ToolChoice::Named(n.to_string()))
            .ok_or_else(|| ApiError::bad("tool_choice object must name a function")),
    }
}

fn one_choice(n: Option<usize>) -> Result<(), ApiError> {
    match n {
        None | Some(1) => Ok(()),
        Some(n) => Err(ApiError::bad(format!(
            "n = {n} is not supported; send one request per choice"
        ))),
    }
}

fn include_usage(opts: Option<&Value>) -> bool {
    opts.is_some_and(|o| o["include_usage"] == true)
}

fn finish_reason(f: &Finish) -> &'static str {
    match f {
        Finish::Stop | Finish::StopSequence(_) => "stop",
        Finish::Length => "length",
        Finish::ToolCalls => "tool_calls",
    }
}

fn usage(prompt: usize, completion: usize) -> Value {
    json!({"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion})
}

pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let req = match body {
        Ok(Json(r)) => r,
        Err(e) => return error(ApiError::bad(e.body_text())),
    };
    let tools = req.tools.as_deref().unwrap_or_default();
    // System 1 handles `tools` when an encoder (or an LLM in System 1 mode) is active in the slot.
    if let (Some(jev), false) = (state.jev_model.read().clone(), tools.is_empty()) {
        return system1_tool_route(&jev, &req).unwrap_or_else(error);
    }
    chat(state, headers, req).await.unwrap_or_else(error)
}

async fn chat(
    state: AppState,
    headers: HeaderMap,
    req: ChatCompletionRequest,
) -> Result<Response, ApiError> {
    one_choice(req.n)?;
    if req.messages.is_empty() {
        return Err(ApiError::bad("messages is empty"));
    }
    match &req.response_format {
        None | Some(Value::Null) => {}
        Some(f) if f["type"] == "text" => {}
        Some(_) => return Err(ApiError::bad(
            "response_format (JSON mode / json_schema) is not supported; only {\"type\": \"text\"}",
        )),
    }
    if let Some(o) = &req.compact {
        o.check_local().map_err(ApiError::bad)?;
    }
    let (messages, media) = prepare_messages(req.messages).await?;
    let mut gen = core::start(
        &state,
        GenRequest {
            prompt: Prompt::Chat {
                messages,
                tools: req.tools.unwrap_or_default(),
                tool_choice: tool_choice(req.tool_choice.as_ref())?,
                template_kwargs: req.chat_template_kwargs.unwrap_or_default(),
            },
            media,
            params: SamplingParams {
                top_p: req.top_p,
                ..SamplingParams::from_request(
                    req.temperature,
                    req.max_completion_tokens.or(req.max_tokens),
                )
            },
            stop: strings(req.stop.as_ref()),
            label: label("openai", &headers),
            compact: req.compact,
        },
    )
    .await?;
    let compaction = gen.compaction.take();
    let model = req.model.unwrap_or_else(|| state.scheduler.model_id());
    let id = format!("chatcmpl-{}", gen.id);
    let prompt_tokens = gen.prompt_tokens;
    let report = compaction.as_ref().map(|c| json!(c.report));
    if req.stream.unwrap_or(false) {
        let wire = ChatWire {
            id,
            model,
            created: unix_now(),
            calls: 0,
            include_usage: include_usage(req.stream_options.as_ref()),
            prompt_tokens,
            compaction: report,
        };
        let mut resp = stream_response(gen, wire, Framing::SseDone);
        resp.headers_mut().extend(compaction.as_ref().map(|c| c.header()));
        return Ok(resp);
    }
    let out = collect(gen).await?;
    let mut message = json!({"role": "assistant", "content": out.text});
    if !out.calls.is_empty() {
        if out.text.is_empty() {
            message["content"] = Value::Null;
        }
        message["tool_calls"] = out.calls.iter().map(call_json).collect();
    }
    let mut body = json!({
        "id": id,
        "object": "chat.completion",
        "created": unix_now(),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason(&out.finish)}],
        "usage": usage(prompt_tokens, out.completion_tokens),
    });
    if let Some(c) = &compaction {
        body["usage"]["compaction"] = json!(c.report);
        if req.compact_return_messages {
            body["gibbs_compacted_messages"] = c.wire_messages();
        }
    }
    let mut resp = Json(body).into_response();
    resp.headers_mut().extend(compaction.as_ref().map(|c| c.header()));
    Ok(resp)
}

fn call_json(c: &super::tools::ToolCall) -> Value {
    json!({"id": format!("call_{}", c.id), "type": "function",
           "function": {"name": c.name, "arguments": c.arguments}})
}

pub struct ChatWire {
    pub id: String,
    pub model: String,
    pub created: u64,
    pub calls: usize,
    pub include_usage: bool,
    pub prompt_tokens: usize,
    /// `usage.compaction` of the final usage chunk.
    pub compaction: Option<Value>,
}

impl ChatWire {
    fn chunk(&self, choices: Value) -> Value {
        json!({"id": self.id, "object": "chat.completion.chunk", "created": self.created,
               "model": self.model, "choices": choices})
    }

    fn delta(&self, delta: Value, finish: Option<&str>) -> Value {
        self.chunk(json!([{"index": 0, "delta": delta, "finish_reason": finish}]))
    }
}

impl Wire for ChatWire {
    fn open(&mut self) -> Vec<Value> {
        vec![self.delta(json!({"role": "assistant", "content": ""}), None)]
    }

    fn on(&mut self, ev: Event) -> Vec<Value> {
        match ev {
            Event::Text(t) => vec![self.delta(json!({ "content": t }), None)],
            Event::Call(c) => {
                let mut call = call_json(&c);
                call["index"] = self.calls.into();
                self.calls += 1;
                vec![self.delta(json!({ "tool_calls": [call] }), None)]
            }
            Event::Done {
                finish,
                completion_tokens,
            } => {
                let mut out = vec![self.delta(json!({}), Some(finish_reason(&finish)))];
                if self.include_usage {
                    let mut last = self.chunk(json!([]));
                    last["usage"] = usage(self.prompt_tokens, completion_tokens);
                    if let Some(c) = &self.compaction {
                        last["usage"]["compaction"] = c.clone();
                    }
                    out.push(last);
                }
                out
            }
            Event::Error(e) => vec![api_error_body(StatusCode::INTERNAL_SERVER_ERROR, e)],
        }
    }
}

/// Zero-shot tool routing with the System 1 encoder (no LLM): picks the tool only.
fn system1_tool_route(
    jev: &crate::runtime::encoder::system1::JevSystem1Model,
    req: &ChatCompletionRequest,
) -> Result<Response, ApiError> {
    let options: Vec<ChoiceOption> = req
        .tools
        .iter()
        .flatten()
        .filter_map(|t| {
            let f = t.get("function")?;
            Some(ChoiceOption::new(
                f.get("name")?.as_str()?,
                f.get("description").and_then(|d| d.as_str()),
            ))
        })
        .collect();
    if options.is_empty() {
        return Err(ApiError::bad("no tool has a function.name"));
    }
    let query = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.text())
        .ok_or_else(|| ApiError::bad("tool routing needs a user message"))?;
    let internal = |e: String| ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: e,
    };
    let selected = match jev
        .embed(&query)
        .and_then(|s| jev.evaluate_choice_embedded(&s, &options))
    {
        Ok(JevDecision::Choice { selected, .. }) => selected,
        Ok(other) => unreachable!("choice evaluation returned {other:?}"),
        Err(e) => return Err(internal(e)),
    };
    let prompt_tokens = jev.token_count(&query).map_err(internal)?;
    let id = core::new_id();
    // ponytail: zero-shot routing picks the tool only; arguments stay "{}" (no argument extraction).
    Ok(Json(json!({
        "id": format!("chatcmpl-{id}"),
        "object": "chat.completion",
        "created": unix_now(),
        "model": jev.repo_id,
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": null,
            "tool_calls": [{"id": format!("call_{id}"), "type": "function",
                            "function": {"name": selected, "arguments": "{}"}}]}}],
        "usage": usage(prompt_tokens, 0),
    }))
    .into_response())
}

pub async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let req = match body {
        Ok(Json(r)) => r,
        Err(e) => return error(ApiError::bad(e.body_text())),
    };
    completion(state, headers, req).await.unwrap_or_else(error)
}

async fn completion(
    state: AppState,
    headers: HeaderMap,
    req: CompletionRequest,
) -> Result<Response, ApiError> {
    one_choice(req.n)?;
    let prompt = match &req.prompt {
        Some(Value::Array(_)) => {
            return Err(ApiError::bad(
                "prompt lists are not supported; send one prompt per request",
            ))
        }
        p => strings(p.as_ref()).concat(),
    };
    let gen = core::start(
        &state,
        GenRequest {
            prompt: Prompt::Raw(prompt),
            media: Vec::new(),
            params: SamplingParams {
                top_p: req.top_p,
                ..SamplingParams::from_request(req.temperature, req.max_tokens)
            },
            stop: strings(req.stop.as_ref()),
            label: label("openai", &headers),
            compact: None,
        },
    )
    .await?;
    let model = req.model.unwrap_or_else(|| state.scheduler.model_id());
    let id = format!("cmpl-{}", gen.id);
    let prompt_tokens = gen.prompt_tokens;
    if req.stream.unwrap_or(false) {
        let wire = CompletionWire {
            id,
            model,
            created: unix_now(),
            include_usage: include_usage(req.stream_options.as_ref()),
            prompt_tokens,
        };
        return Ok(stream_response(gen, wire, Framing::SseDone));
    }
    let out = collect(gen).await?;
    Ok(Json(json!({
        "id": id,
        "object": "text_completion",
        "created": unix_now(),
        "model": model,
        "choices": [{"index": 0, "text": out.text, "logprobs": null,
                     "finish_reason": finish_reason(&out.finish)}],
        "usage": usage(prompt_tokens, out.completion_tokens),
    }))
    .into_response())
}

pub struct CompletionWire {
    pub id: String,
    pub model: String,
    pub created: u64,
    pub include_usage: bool,
    pub prompt_tokens: usize,
}

impl CompletionWire {
    fn chunk(&self, text: &str, finish: Option<&str>) -> Value {
        json!({"id": self.id, "object": "text_completion", "created": self.created, "model": self.model,
               "choices": [{"text": text, "index": 0, "logprobs": null, "finish_reason": finish}]})
    }
}

impl Wire for CompletionWire {
    fn on(&mut self, ev: Event) -> Vec<Value> {
        match ev {
            Event::Text(t) => vec![self.chunk(&t, None)],
            // Raw prompts have no tools.
            Event::Call(_) => Vec::new(),
            Event::Done {
                finish,
                completion_tokens,
            } => {
                let mut out = vec![self.chunk("", Some(finish_reason(&finish)))];
                if self.include_usage {
                    let mut last = self.chunk("", None);
                    last["choices"] = json!([]);
                    last["usage"] = usage(self.prompt_tokens, completion_tokens);
                    out.push(last);
                }
                out
            }
            Event::Error(e) => vec![api_error_body(StatusCode::INTERNAL_SERVER_ERROR, e)],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::tools::ToolCall;

    #[test]
    fn chat_stream_order_tool_call_then_finish_then_usage() {
        let mut w = ChatWire {
            id: "chatcmpl-1".into(),
            model: "m".into(),
            created: 0,
            calls: 0,
            include_usage: true,
            prompt_tokens: 7,
            compaction: None,
        };
        let call = |name: &str| {
            Event::Call(ToolCall {
                id: "x".into(),
                name: name.into(),
                arguments: "{\"city\":\"Paris\"}".into(),
            })
        };
        let mut evs = w.open();
        for ev in [
            Event::Text("Let me check.".into()),
            call("get_weather"),
            call("get_time"),
            Event::Done {
                finish: Finish::ToolCalls,
                completion_tokens: 20,
            },
        ] {
            evs.extend(w.on(ev));
        }
        assert_eq!(evs[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(evs[1]["choices"][0]["delta"]["content"], "Let me check.");
        let tc = &evs[2]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(
            (tc["index"].as_u64(), tc["function"]["name"].as_str()),
            (Some(0), Some("get_weather"))
        );
        assert_eq!(tc["function"]["arguments"], "{\"city\":\"Paris\"}");
        assert_eq!(evs[3]["choices"][0]["delta"]["tool_calls"][0]["index"], 1);
        assert_eq!(evs[4]["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(evs[5]["choices"], json!([]));
        assert_eq!(evs[5]["usage"]["total_tokens"], 27);
        assert_eq!(evs.len(), 6);
    }

    #[test]
    fn tool_choice_forms() {
        assert_eq!(tool_choice(None).unwrap(), ToolChoice::Auto);
        assert_eq!(
            tool_choice(Some(&json!("required"))).unwrap(),
            ToolChoice::Required
        );
        let named = ToolChoice::Named("f".into());
        assert_eq!(
            tool_choice(Some(
                &json!({"type": "function", "function": {"name": "f"}})
            ))
            .unwrap(),
            named
        );
        assert_eq!(
            tool_choice(Some(&json!({"type": "function", "name": "f"}))).unwrap(),
            named
        );
        assert!(tool_choice(Some(&json!("sometimes"))).is_err());
    }
}
