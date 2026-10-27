//! Agent-facing schemas against a real tool-capable model (Qwen2.5-0.5B-Instruct, Hermes
//! `<tool_call>` format): the template renders the tools, the model's call comes back parsed in
//! each schema's shape, and the tool result yields a final answer.

use axum::body::Body;
use axum::http::Request;
use gibbs_engine::model::hub::ModelArtifacts;
use gibbs_engine::scheduler::{ContinuousBatchScheduler, ExecutableModel};
use gibbs_engine::server::{create_router, AppState};
use gibbs_engine::telemetry::EngineTelemetry;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tower::ServiceExt;

const QWEN: &str = "disk/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/main";

async fn post(app: &axum::Router, uri: &str, body: Value) -> String {
    let req = Request::post(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

/// Image parts reach `submit_with_media`, and a text-only model's refusal comes back as a 400
/// in each schema's error shape (the scheduler's message, not a template error).
#[tokio::test]
async fn media_for_a_text_only_model_is_a_400() {
    // A synthetic llama (no chat template: the built-in fallback renders the text and leaves
    // media to the model).
    let app = create_router(AppState {
        scheduler: Arc::new(ContinuousBatchScheduler::new("tiny", Some({
        // A fresh hub dir: a synthetic repo left in the user's cache would be reused as is.
        let hub_dir = tempfile::tempdir().unwrap();
        let a = gibbs_engine::model::hub::HuggingFaceHub::with_cache_dir(hub_dir.path().to_path_buf())
            .resolve_or_synthesize_tiny("gibbs/tiny-5m")
            .unwrap();
        let reg = gibbs_engine::arch::registry().unwrap();
        let cpu = gibbs_engine::hardware::Device::Cpu.into();
        ExecutableModel::new(gibbs_engine::runtime::lm::load(&a, reg, None, cpu).unwrap())
                }))),
        telemetry: EngineTelemetry::new(),
        jev_model: Default::default(),
        image_model: Default::default(),
        classifier: Default::default(),
    });
    let png = "iVBORw0KGgo=";
    for (uri, body) in [
        (
            "/v1/chat/completions",
            json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{png}")}}]}]}),
        ),
        (
            "/v1/messages",
            json!({"max_tokens": 8, "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}},
                {"type": "text", "text": "what is this?"}]}]}),
        ),
        (
            "/api/chat",
            json!({"stream": false, "messages": [{"role": "user", "content": "what is this?", "images": [png]}]}),
        ),
    ] {
        let req = Request::post(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert_eq!(status, 400, "{uri}: {text}");
        assert!(text.contains("takes text only"), "{uri}: {text}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn qwen_tool_calls_in_every_schema() {
    let dir = std::path::Path::new(QWEN);
    if cfg!(debug_assertions) || !dir.join("model.safetensors").exists() {
        println!("SKIP: needs --release and {QWEN}");
        return;
    }
    let artifacts = ModelArtifacts {
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".into(),
        config_path: Some(dir.join("config.json")),
        tokenizer_path: Some(dir.join("tokenizer.json")),
        weights_path: Some(dir.join("model.safetensors")),
        ..Default::default()
    };
    let reg = gibbs_engine::arch::registry().unwrap();
    let cpu = gibbs_engine::hardware::Device::Cpu.into();
    let model = ExecutableModel::new(gibbs_engine::runtime::lm::load(&artifacts, reg, None, cpu).unwrap());
    let sched = Arc::new(ContinuousBatchScheduler::new("Qwen/Qwen2.5-0.5B-Instruct", Some(model)));
    let stop = Arc::new(AtomicBool::new(false));
    let engine = {
        let (sched, stop) = (sched.clone(), stop.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if sched.step().active_sequences_count == 0 {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
            }
        })
    };
    let app = create_router(AppState {
        scheduler: sched,
        telemetry: EngineTelemetry::new(),
        jev_model: Default::default(),
        image_model: Default::default(),
        classifier: Default::default(),
    });
    let params =
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]});
    let ask = "What's the weather in Paris right now?";
    let result = r#"{"city": "Paris", "temperature_c": 18, "condition": "sunny"}"#;

    // OpenAI Chat Completions: call, then the answer from the tool result.
    let tools = json!([{"type": "function", "function": {"name": "get_weather",
        "description": "Get the current weather for a city", "parameters": params}}]);
    let r: Value = serde_json::from_str(
        &post(
            &app,
            "/v1/chat/completions",
            json!({"messages": [{"role": "user", "content": ask}],
            "tools": tools, "temperature": 0}),
        )
        .await,
    )
    .unwrap();
    let choice = &r["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls", "{r}");
    let call = &choice["message"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "get_weather", "{r}");
    let args: Value =
        serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["city"], "Paris", "{r}");
    let r: Value = serde_json::from_str(
        &post(
            &app,
            "/v1/chat/completions",
            json!({"temperature": 0, "tools": tools, "max_tokens": 60,
            "messages": [{"role": "user", "content": ask}, choice["message"],
                         {"role": "tool", "tool_call_id": call["id"], "content": result}]}),
        )
        .await,
    )
    .unwrap();
    let answer = r["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(answer.contains("18"), "{r}");

    // Anthropic streaming: the call arrives as a tool_use block, stop_reason tool_use.
    let sse = post(
        &app,
        "/v1/messages",
        json!({"model": "claude-anything", "max_tokens": 128,
        "temperature": 0, "stream": true, "messages": [{"role": "user", "content": ask}],
        "tools": [{"name": "get_weather", "description": "Get the current weather for a city",
                   "input_schema": params}]}),
    )
    .await;
    let events: Vec<Value> = sse
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    let block = events
        .iter()
        .find(|e| e["type"] == "content_block_start" && e["content_block"]["type"] == "tool_use")
        .unwrap_or_else(|| panic!("{sse}"));
    assert_eq!(block["content_block"]["name"], "get_weather");
    assert!(
        events
            .iter()
            .any(|e| e["delta"]["stop_reason"] == "tool_use"),
        "{sse}"
    );
    assert_eq!(events.last().unwrap()["type"], "message_stop");

    // Ollama: tool_calls with object arguments.
    let r: Value = serde_json::from_str(
        &post(
            &app,
            "/api/chat",
            json!({"model": "x", "stream": false, "options": {"temperature": 0},
            "messages": [{"role": "user", "content": ask}], "tools": tools}),
        )
        .await,
    )
    .unwrap();
    assert_eq!(
        r["message"]["tool_calls"][0]["function"]["arguments"]["city"], "Paris",
        "{r}"
    );

    // Anthropic count_tokens counts exactly the prompt /v1/messages runs.
    let msg = json!({"model": "x", "messages": [{"role": "user", "content": ask}],
        "tools": [{"name": "get_weather", "description": "Get the current weather for a city",
                   "input_schema": params}]});
    let counted: Value =
        serde_json::from_str(&post(&app, "/v1/messages/count_tokens", msg.clone()).await).unwrap();
    let mut run = msg;
    run["max_tokens"] = json!(1);
    let r: Value = serde_json::from_str(&post(&app, "/v1/messages", run).await).unwrap();
    assert_eq!(
        counted["input_tokens"], r["usage"]["input_tokens"],
        "{counted} vs {r}"
    );

    stop.store(true, Ordering::Relaxed);
    engine.join().unwrap();
}
