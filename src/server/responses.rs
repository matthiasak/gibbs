//! OpenAI Responses API `/v1/responses` (Codex CLI's wire API), stateless, on the shared core.

use super::core::{
    self, collect, label, prepare_messages, stream_response, unix_now, ApiError, Event, Finish,
    Framing, GenRequest, Prompt, Wire,
};
use super::openai::{error, tool_choice};
use super::tools::ToolCall;
use super::AppState;
use crate::model::hf_loader::ChatMessage;
use crate::scheduler::SamplingParams;
use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Json, Response};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    pub model: Option<String>,
    #[serde(default)]
    pub input: Value,
    pub instructions: Option<String>,
    #[serde(default)]
    pub tools: Vec<Value>,
    pub tool_choice: Option<Value>,
    pub max_output_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
    pub previous_response_id: Option<String>,
}

/// `instructions` + `input` (a string, or message / function_call / function_call_output items)
/// as chat messages.
fn input_messages(
    instructions: Option<String>,
    input: Value,
) -> Result<Vec<ChatMessage>, ApiError> {
    let mut out: Vec<ChatMessage> = instructions
        .map(|i| ChatMessage::new("system", i))
        .into_iter()
        .collect();
    let items = match input {
        Value::String(s) => return Ok([out, vec![ChatMessage::new("user", s)]].concat()),
        Value::Array(items) => items,
        _ => return Err(ApiError::bad("input must be a string or a list of items")),
    };
    for item in items {
        let str_of = |k: &str| item[k].as_str().map(String::from);
        match item["type"].as_str().unwrap_or("message") {
            "message" => out.push(ChatMessage {
                role: str_of("role").ok_or_else(|| ApiError::bad("message item has no role"))?,
                content: item["content"].clone(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }),
            "function_call" => {
                let call = json!({"id": item["call_id"], "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]}});
                // Consecutive calls (and a preceding assistant text) form one assistant turn.
                match out.last_mut() {
                    Some(m) if m.role == "assistant" => match &mut m.tool_calls {
                        Some(Value::Array(calls)) => calls.push(call),
                        _ => m.tool_calls = Some(json!([call])),
                    },
                    _ => out.push(ChatMessage {
                        role: "assistant".into(),
                        content: Value::Null,
                        tool_calls: Some(json!([call])),
                        tool_call_id: None,
                        name: None,
                    }),
                }
            }
            "function_call_output" => out.push(ChatMessage {
                role: "tool".into(),
                content: item["output"].clone(),
                tool_calls: None,
                tool_call_id: str_of("call_id"),
                name: None,
            }),
            other => {
                return Err(ApiError::bad(format!(
                    "unsupported input item type '{other}' (message, function_call, function_call_output)"
                )))
            }
        }
    }
    Ok(out)
}

/// Function tools (`{"type":"function","name",..}`) in Chat Completions shape, and the names
/// (or types) of built-in tools (`web_search`, `local_shell`, `custom`, ...) the model cannot call.
fn chat_tools(tools: Vec<Value>) -> (Vec<Value>, Vec<String>) {
    let (mut kept, mut skipped) = (Vec::new(), Vec::new());
    for t in tools {
        if t["type"] == "function" {
            kept.push(json!({"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["parameters"]}}));
        } else {
            let name = t["name"].as_str().or(t["type"].as_str()).unwrap_or("?");
            skipped.push(name.to_string());
        }
    }
    (kept, skipped)
}

pub async fn responses(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<ResponsesRequest>, JsonRejection>,
) -> Response {
    let req = match body {
        Ok(Json(r)) => r,
        Err(e) => return error(ApiError::bad(e.body_text())),
    };
    respond(state, headers, req).await.unwrap_or_else(error)
}

async fn respond(
    state: AppState,
    headers: HeaderMap,
    req: ResponsesRequest,
) -> Result<Response, ApiError> {
    if req.previous_response_id.is_some() {
        return Err(ApiError::bad(
            "previous_response_id is not supported: this server is stateless; send the full input",
        ));
    }
    let (messages, media) = prepare_messages(input_messages(req.instructions, req.input)?).await?;
    let (tools, skipped) = chat_tools(req.tools);
    let choice = tool_choice(req.tool_choice.as_ref())?;
    core::skip_builtin_tools(&state, "responses", &skipped, &choice)?;
    let gen = core::start(
        &state,
        GenRequest {
            prompt: Prompt::Chat {
                messages,
                tools,
                tool_choice: choice,
                template_kwargs: Value::Null,
            },
            media,
            params: SamplingParams {
                top_p: req.top_p,
                ..SamplingParams::from_request(req.temperature, req.max_output_tokens)
            },
            stop: Vec::new(),
            label: label("responses", &headers),
            compact: None,
        },
    )
    .await?;
    let mut wire = ResponsesWire::new(
        gen.id,
        req.model.unwrap_or_else(|| state.scheduler.model_id()),
        gen.prompt_tokens,
    );
    if req.stream.unwrap_or(false) {
        return Ok(stream_response(gen, wire, Framing::SseNamed));
    }
    let out = collect(gen).await?;
    let status = if out.finish == Finish::Length {
        "incomplete"
    } else {
        "completed"
    };
    if !out.text.is_empty() {
        wire.output
            .push(message_item(&wire.msg_id(), &out.text, status));
    }
    wire.output
        .extend(out.calls.iter().map(|c| call_item(c, "completed")));
    Ok(Json(wire.response(Some((&out.finish, out.completion_tokens)))).into_response())
}

/// A message output item; only an item just opened for streaming has no content yet.
fn message_item(id: &str, text: &str, status: &str) -> Value {
    let content = if status == "in_progress" {
        json!([])
    } else {
        json!([{"type": "output_text", "text": text, "annotations": []}])
    };
    json!({"type": "message", "id": id, "status": status, "role": "assistant", "content": content})
}

fn call_item(c: &ToolCall, status: &str) -> Value {
    let args = if status == "completed" {
        c.arguments.as_str()
    } else {
        ""
    };
    json!({"type": "function_call", "id": format!("fc_{}", c.id), "call_id": format!("call_{}", c.id),
           "name": c.name, "arguments": args, "status": status})
}

pub struct ResponsesWire {
    id: String,
    model: String,
    created: u64,
    prompt_tokens: usize,
    seq: u64,
    /// Finished output items, in order.
    output: Vec<Value>,
    /// Text of the message item being streamed, if one is open.
    text: Option<String>,
    msgs: usize,
}

impl ResponsesWire {
    pub fn new(id: u64, model: String, prompt_tokens: usize) -> Self {
        Self {
            id: format!("resp_{id}"),
            model,
            created: unix_now(),
            prompt_tokens,
            seq: 0,
            output: Vec::new(),
            text: None,
            msgs: 0,
        }
    }

    fn msg_id(&self) -> String {
        format!("msg_{}_{}", &self.id[5..], self.msgs)
    }

    /// The response object; `None` while in progress.
    fn response(&self, done: Option<(&Finish, usize)>) -> Value {
        let mut r = json!({
            "id": self.id, "object": "response", "created_at": self.created, "model": self.model,
            "status": "in_progress", "output": self.output, "error": null,
            "incomplete_details": null, "usage": null,
        });
        if let Some((finish, completion)) = done {
            r["status"] = json!(if *finish == Finish::Length {
                "incomplete"
            } else {
                "completed"
            });
            if *finish == Finish::Length {
                r["incomplete_details"] = json!({"reason": "max_output_tokens"});
            }
            r["usage"] = json!({"input_tokens": self.prompt_tokens, "output_tokens": completion,
                                "total_tokens": self.prompt_tokens + completion});
        }
        r
    }

    fn ev(&mut self, mut v: Value) -> Value {
        v["sequence_number"] = self.seq.into();
        self.seq += 1;
        v
    }

    /// Ends the streamed message item; `status` is "incomplete" when the length cap cut it.
    fn close_text(&mut self, out: &mut Vec<Value>, status: &str) {
        let Some(text) = self.text.take() else { return };
        let (id, i) = (self.msg_id(), self.output.len());
        let part = json!({"type": "output_text", "text": text, "annotations": []});
        let item = message_item(&id, &text, status);
        for v in [
            json!({"type": "response.output_text.done", "item_id": id, "output_index": i, "content_index": 0, "text": text}),
            json!({"type": "response.content_part.done", "item_id": id, "output_index": i, "content_index": 0, "part": part}),
            json!({"type": "response.output_item.done", "output_index": i, "item": item}),
        ] {
            out.push(self.ev(v));
        }
        self.output.push(item);
        self.msgs += 1;
    }
}

impl Wire for ResponsesWire {
    fn open(&mut self) -> Vec<Value> {
        let r = self.response(None);
        vec![
            self.ev(json!({"type": "response.created", "response": r})),
            self.ev(json!({"type": "response.in_progress", "response": r})),
        ]
    }

    fn on(&mut self, ev: Event) -> Vec<Value> {
        let mut out = Vec::new();
        match ev {
            Event::Text(t) => {
                let (id, i) = (self.msg_id(), self.output.len());
                if self.text.is_none() {
                    self.text = Some(String::new());
                    let item = message_item(&id, "", "in_progress");
                    let added = json!({"type": "response.output_item.added", "output_index": i, "item": item});
                    out.push(self.ev(added));
                    let part = json!({"type": "response.content_part.added", "item_id": id, "output_index": i,
                        "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}});
                    out.push(self.ev(part));
                }
                self.text.as_mut().expect("opened above").push_str(&t);
                let delta = json!({"type": "response.output_text.delta", "item_id": id, "output_index": i,
                    "content_index": 0, "delta": t});
                out.push(self.ev(delta));
            }
            Event::Call(c) => {
                self.close_text(&mut out, "completed");
                let (i, item_id) = (self.output.len(), format!("fc_{}", c.id));
                for v in [
                    json!({"type": "response.output_item.added", "output_index": i, "item": call_item(&c, "in_progress")}),
                    json!({"type": "response.function_call_arguments.delta", "item_id": item_id, "output_index": i, "delta": c.arguments}),
                    json!({"type": "response.function_call_arguments.done", "item_id": item_id, "output_index": i, "arguments": c.arguments}),
                    json!({"type": "response.output_item.done", "output_index": i, "item": call_item(&c, "completed")}),
                ] {
                    out.push(self.ev(v));
                }
                self.output.push(call_item(&c, "completed"));
            }
            Event::Done {
                finish,
                completion_tokens,
            } => {
                let status = if finish == Finish::Length {
                    "incomplete"
                } else {
                    "completed"
                };
                self.close_text(&mut out, status);
                let r = self.response(Some((&finish, completion_tokens)));
                let kind = if finish == Finish::Length {
                    "response.incomplete"
                } else {
                    "response.completed"
                };
                out.push(self.ev(json!({"type": kind, "response": r})));
            }
            Event::Error(e) => {
                let mut r = self.response(None);
                r["status"] = json!("failed");
                r["error"] = json!({"code": "server_error", "message": e});
                out.push(self.ev(json!({"type": "response.failed", "response": r})));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_event_order_text_then_call() {
        let mut w = ResponsesWire::new(9, "m".into(), 5);
        let mut evs = w.open();
        for ev in [
            Event::Text("Hi".into()),
            Event::Text(" there".into()),
            Event::Call(ToolCall {
                id: "a".into(),
                name: "get_weather".into(),
                arguments: "{}".into(),
            }),
            Event::Done {
                finish: Finish::ToolCalls,
                completion_tokens: 3,
            },
        ] {
            evs.extend(w.on(ev));
        }
        let types: Vec<&str> = evs.iter().map(|e| e["type"].as_str().unwrap()).collect();
        assert_eq!(
            types,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert!(evs
            .iter()
            .enumerate()
            .all(|(i, e)| e["sequence_number"] == i as u64));
        let done = &evs.last().unwrap()["response"];
        assert_eq!(done["output"][0]["content"][0]["text"], "Hi there");
        assert_eq!(done["output"][1]["call_id"], "call_a");
        assert_eq!(evs[9]["output_index"], 1);
        assert_eq!(done["usage"]["total_tokens"], 8);
    }

    #[test]
    fn input_items_become_chat_turns() {
        let input = json!([
            {"role": "user", "content": [{"type": "input_text", "text": "weather in Paris?"}]},
            {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
            {"type": "function_call", "call_id": "c2", "name": "get_weather", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "18C"}
        ]);
        let m = input_messages(Some("be brief".into()), input).unwrap();
        let roles: Vec<&str> = m.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "user", "assistant", "tool"]);
        assert_eq!(
            m[2].tool_calls.as_ref().unwrap().as_array().unwrap().len(),
            2
        );
        assert_eq!(m[3].tool_call_id.as_deref(), Some("c1"));
        assert!(input_messages(None, json!([{"type": "reasoning"}])).is_err());
        let (kept, skipped) = chat_tools(vec![
            json!({"type": "web_search"}),
            json!({"type": "custom", "name": "apply_patch"}),
            json!({"type": "function", "name": "f", "parameters": {}}),
        ]);
        assert_eq!(kept.len(), 1);
        assert_eq!(skipped, ["web_search", "apply_patch"]);
    }

    #[test]
    fn length_cut_message_is_incomplete_and_keeps_its_text() {
        let mut w = ResponsesWire::new(1, "m".into(), 2);
        w.open();
        w.on(Event::Text("partial".into()));
        let evs = w.on(Event::Done {
            finish: Finish::Length,
            completion_tokens: 1,
        });
        let last = evs.last().unwrap();
        assert_eq!(last["type"], "response.incomplete");
        let item = &last["response"]["output"][0];
        assert_eq!(
            (item["status"].as_str(), item["content"][0]["text"].as_str()),
            (Some("incomplete"), Some("partial"))
        );
    }
}
