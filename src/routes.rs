use axum::extract::State;
use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use std::convert::Infallible;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info};

use crate::adapter::anthropic_to_cli;
use crate::adapter::cli_to_anthropic;
use crate::adapter::cli_to_openai;
use crate::adapter::openai_to_cli;
use crate::error::AppError;
use crate::server::AppState;
use crate::subprocess::{self, SubprocessEvent, SubprocessOptions};
use crate::types::anthropic::{AnthropicErrorDetail, AnthropicErrorResponse, MessagesRequest};
use crate::types::openai::{ChatCompletionRequest, ModelInfo, ModelsResponse};

fn generate_request_id() -> String {
    uuid::Uuid::new_v4()
        .to_string()
        .replace('-', "")
        .chars()
        .take(8)
        .collect()
}

pub async fn health() -> impl IntoResponse {
    let uptime = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Json(json!({
        "status": "ok",
        "uptime": uptime,
    }))
}

pub async fn models() -> impl IntoResponse {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![
            ModelInfo {
                id: "claude-opus-4".to_string(),
                object: "model".to_string(),
                owned_by: "anthropic".to_string(),
                created,
                context_window: 1_000_000,
                max_tokens: 128_000,
            },
            ModelInfo {
                id: "claude-sonnet-4".to_string(),
                object: "model".to_string(),
                owned_by: "anthropic".to_string(),
                created,
                context_window: 200_000,
                max_tokens: 64_000,
            },
            ModelInfo {
                id: "claude-haiku-4".to_string(),
                object: "model".to_string(),
                owned_by: "anthropic".to_string(),
                created,
                context_window: 200_000,
                max_tokens: 64_000,
            },
        ],
    })
}

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, AppError> {
    // Validate messages
    let messages = request.messages.as_ref().ok_or_else(|| {
        AppError::BadRequest("messages is required and must be a non-empty array".to_string())
    })?;
    if messages.is_empty() {
        return Err(AppError::BadRequest(
            "messages is required and must be a non-empty array".to_string(),
        ));
    }

    let request_id = generate_request_id();
    let is_streaming = request.stream;

    let (model, prompt, session_id) = openai_to_cli::openai_to_cli(&request);

    info!("[req={request_id}] OpenAI chat completions model={model} streaming={is_streaming}");

    let options = SubprocessOptions {
        request_id: request_id.clone(),
        model: model.to_string(),
        session_id,
        cwd: state.cwd.clone(),
        api: "openai",
    };

    if is_streaming {
        handle_streaming(request_id, prompt, options).await
    } else {
        let start = Instant::now();
        let result = handle_non_streaming(request_id.clone(), prompt, options).await;
        let elapsed = start.elapsed().as_secs_f64();
        match &result {
            Ok(_) => info!("[req={request_id}] Request complete after {elapsed:.2}s"),
            Err(e) => error!("[req={request_id}] Request failed after {elapsed:.2}s: {e}"),
        }
        result
    }
}

async fn handle_non_streaming(
    request_id: String,
    prompt: String,
    options: SubprocessOptions,
) -> Result<Response, AppError> {
    let (tx, mut rx) = mpsc::channel::<SubprocessEvent>(64);

    tokio::spawn(async move {
        subprocess::spawn_subprocess(prompt, options, tx).await;
    });

    let mut result_msg = None;
    let mut error_msg = None;
    let mut exit_code = None;

    while let Some(event) = rx.recv().await {
        match event {
            SubprocessEvent::Result(result) => {
                result_msg = Some(result);
            }
            SubprocessEvent::Error(msg) => {
                error_msg = Some(msg);
            }
            SubprocessEvent::Close(code) => {
                exit_code = Some(code);
            }
            _ => {}
        }
    }

    if let Some(err) = error_msg {
        return Err(AppError::Subprocess(err));
    }

    if let Some(result) = result_msg {
        let response = cli_to_openai::cli_result_to_openai(&result, &request_id);
        Ok((
            [(header::HeaderName::from_static("x-request-id"), request_id)],
            Json(response),
        )
            .into_response())
    } else {
        let code = exit_code.unwrap_or(-1);
        Err(AppError::Subprocess(format!(
            "Process exited with code {} without producing a response",
            code
        )))
    }
}

async fn handle_streaming(
    request_id: String,
    prompt: String,
    options: SubprocessOptions,
) -> Result<Response, AppError> {
    let (tx, mut rx) = mpsc::channel::<SubprocessEvent>(64);

    tokio::spawn(async move {
        subprocess::spawn_subprocess(prompt, options, tx).await;
    });

    let req_id = request_id.clone();
    let (sse_tx, sse_rx) = mpsc::channel::<Result<Event, Infallible>>(64);

    // Spawn a task to convert subprocess events to SSE events
    tokio::spawn(async move {
        let mut is_first = true;
        let mut last_model = "claude-sonnet-4".to_string();
        let mut got_result = false;

        // Send initial :ok comment
        let ok_event = Event::default().comment("ok");
        if sse_tx.send(Ok(ok_event)).await.is_err() {
            return;
        }

        while let Some(event) = rx.recv().await {
            match event {
                SubprocessEvent::Model(model) => {
                    last_model = model;
                }
                SubprocessEvent::ContentDelta(text) => {
                    let chunk =
                        cli_to_openai::create_stream_chunk(&req_id, &last_model, &text, is_first);
                    is_first = false;

                    match serde_json::to_string(&chunk) {
                        Ok(json) => {
                            let event = Event::default().data(json);
                            if sse_tx.send(Ok(event)).await.is_err() {
                                return; // Client disconnected
                            }
                        }
                        Err(e) => {
                            error!("[req={req_id}] Failed to serialize chunk: {e}");
                        }
                    }
                }
                SubprocessEvent::Result(_result) => {
                    got_result = true;

                    // Send done chunk with finish_reason: "stop"
                    let done_chunk = cli_to_openai::create_done_chunk(&req_id, &last_model);
                    if let Ok(json) = serde_json::to_string(&done_chunk) {
                        let event = Event::default().data(json);
                        let _ = sse_tx.send(Ok(event)).await;
                    }

                    // Send [DONE] sentinel
                    let done_event = Event::default().data("[DONE]");
                    let _ = sse_tx.send(Ok(done_event)).await;
                }
                SubprocessEvent::Error(msg) => {
                    let error_data = json!({
                        "error": {
                            "message": msg,
                            "type": "server_error",
                            "code": null,
                        }
                    });
                    if let Ok(json) = serde_json::to_string(&error_data) {
                        let event = Event::default().data(json);
                        let _ = sse_tx.send(Ok(event)).await;
                    }
                }
                SubprocessEvent::Close(code) => {
                    if !got_result && code != 0 {
                        let error_data = json!({
                            "error": {
                                "message": format!("Process exited with code {}", code),
                                "type": "server_error",
                                "code": null,
                            }
                        });
                        if let Ok(json) = serde_json::to_string(&error_data) {
                            let event = Event::default().data(json);
                            let _ = sse_tx.send(Ok(event)).await;
                        }
                        let done_event = Event::default().data("[DONE]");
                        let _ = sse_tx.send(Ok(done_event)).await;
                    }
                }
            }
        }
    });

    let stream = ReceiverStream::new(sse_rx);

    let sse = Sse::new(stream).keep_alive(KeepAlive::default());

    Ok((
        [
            (
                header::HeaderName::from_static("x-request-id"),
                request_id,
            ),
            (
                header::CACHE_CONTROL,
                "no-cache".to_string(),
            ),
        ],
        sse,
    )
        .into_response())
}

// ── Anthropic Messages API ──────────────────────────────────────

pub async fn messages(
    State(state): State<AppState>,
    Json(request): Json<MessagesRequest>,
) -> Result<Response, AppError> {
    if request.messages.is_empty() {
        return Err(AppError::BadRequest(
            "messages is required and must be a non-empty array".to_string(),
        ));
    }

    let request_id = generate_request_id();
    let is_streaming = request.stream;

    let (model, prompt, session_id) = anthropic_to_cli::anthropic_to_cli(&request);

    info!("[req={request_id}] Anthropic messages model={model} streaming={is_streaming}");

    let options = SubprocessOptions {
        request_id: request_id.clone(),
        model: model.to_string(),
        session_id,
        cwd: state.cwd.clone(),
        api: "anthropic",
    };

    if is_streaming {
        handle_messages_streaming(request_id, prompt, options).await
    } else {
        let start = Instant::now();
        let result = handle_messages_non_streaming(request_id.clone(), prompt, options).await;
        let elapsed = start.elapsed().as_secs_f64();
        match &result {
            Ok(_) => info!("[req={request_id}] Request complete after {elapsed:.2}s"),
            Err(e) => error!("[req={request_id}] Request failed after {elapsed:.2}s: {e}"),
        }
        result
    }
}

async fn handle_messages_non_streaming(
    request_id: String,
    prompt: String,
    options: SubprocessOptions,
) -> Result<Response, AppError> {
    let (tx, mut rx) = mpsc::channel::<SubprocessEvent>(64);

    tokio::spawn(async move {
        subprocess::spawn_subprocess(prompt, options, tx).await;
    });

    let mut result_msg = None;
    let mut error_msg = None;
    let mut exit_code = None;

    while let Some(event) = rx.recv().await {
        match event {
            SubprocessEvent::Result(result) => {
                result_msg = Some(result);
            }
            SubprocessEvent::Error(msg) => {
                error_msg = Some(msg);
            }
            SubprocessEvent::Close(code) => {
                exit_code = Some(code);
            }
            _ => {}
        }
    }

    if let Some(err) = error_msg {
        return Err(AppError::Subprocess(err));
    }

    if let Some(result) = result_msg {
        let response = cli_to_anthropic::cli_result_to_anthropic(&result, &request_id);
        Ok((
            [(header::HeaderName::from_static("x-request-id"), request_id)],
            Json(response),
        )
            .into_response())
    } else {
        let code = exit_code.unwrap_or(-1);
        Err(AppError::Subprocess(format!(
            "Process exited with code {} without producing a response",
            code
        )))
    }
}

async fn handle_messages_streaming(
    request_id: String,
    prompt: String,
    options: SubprocessOptions,
) -> Result<Response, AppError> {
    let (tx, mut rx) = mpsc::channel::<SubprocessEvent>(64);

    tokio::spawn(async move {
        subprocess::spawn_subprocess(prompt, options, tx).await;
    });

    let req_id = request_id.clone();
    let (sse_tx, sse_rx) = mpsc::channel::<Result<Event, Infallible>>(64);

    tokio::spawn(async move {
        let mut last_model = "claude-sonnet-4".to_string();
        let mut sent_start = false;
        let mut output_tokens: u64 = 0;

        while let Some(event) = rx.recv().await {
            match event {
                SubprocessEvent::Model(model) => {
                    last_model = model;
                }
                SubprocessEvent::ContentDelta(text) => {
                    // Lazily emit message_start + ping + content_block_start on first delta
                    if !sent_start {
                        let start = cli_to_anthropic::create_message_start(&req_id, &last_model);
                        if send_named_event(&sse_tx, "message_start", &start).await.is_err() {
                            return;
                        }
                        let ping = cli_to_anthropic::create_ping();
                        if send_named_event(&sse_tx, "ping", &ping).await.is_err() {
                            return;
                        }
                        let block_start = cli_to_anthropic::create_content_block_start();
                        if send_named_event(&sse_tx, "content_block_start", &block_start)
                            .await
                            .is_err()
                        {
                            return;
                        }
                        sent_start = true;
                    }

                    let delta = cli_to_anthropic::create_content_block_delta(&text);
                    if send_named_event(&sse_tx, "content_block_delta", &delta)
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                SubprocessEvent::Result(result) => {
                    // Extract output token count from result
                    if let Some(mu) = &result.model_usage {
                        for u in mu.values() {
                            output_tokens += u.output_tokens.unwrap_or(0);
                        }
                    }

                    // If we never sent start (empty response), emit it now
                    if !sent_start {
                        let start = cli_to_anthropic::create_message_start(&req_id, &last_model);
                        let _ = send_named_event(&sse_tx, "message_start", &start).await;
                        let ping = cli_to_anthropic::create_ping();
                        let _ = send_named_event(&sse_tx, "ping", &ping).await;
                        let block_start = cli_to_anthropic::create_content_block_start();
                        let _ =
                            send_named_event(&sse_tx, "content_block_start", &block_start).await;
                    }

                    let block_stop = cli_to_anthropic::create_content_block_stop();
                    let _ = send_named_event(&sse_tx, "content_block_stop", &block_stop).await;

                    let msg_delta = cli_to_anthropic::create_message_delta(output_tokens);
                    let _ = send_named_event(&sse_tx, "message_delta", &msg_delta).await;

                    let msg_stop = cli_to_anthropic::create_message_stop();
                    let _ = send_named_event(&sse_tx, "message_stop", &msg_stop).await;
                }
                SubprocessEvent::Error(msg) => {
                    let err = to_anthropic_error("server_error", &msg);
                    if let Ok(json) = serde_json::to_string(&err) {
                        let event = Event::default().event("error").data(json);
                        let _ = sse_tx.send(Ok(event)).await;
                    }
                }
                SubprocessEvent::Close(code) => {
                    if !sent_start && code != 0 {
                        let err = to_anthropic_error(
                            "server_error",
                            &format!("Process exited with code {}", code),
                        );
                        if let Ok(json) = serde_json::to_string(&err) {
                            let event = Event::default().event("error").data(json);
                            let _ = sse_tx.send(Ok(event)).await;
                        }
                    }
                }
            }
        }
    });

    let stream = ReceiverStream::new(sse_rx);
    let sse = Sse::new(stream).keep_alive(KeepAlive::default());

    Ok((
        [
            (
                header::HeaderName::from_static("x-request-id"),
                request_id,
            ),
            (header::CACHE_CONTROL, "no-cache".to_string()),
        ],
        sse,
    )
        .into_response())
}

/// Serialize and send a named SSE event.
async fn send_named_event<T: serde::Serialize>(
    tx: &mpsc::Sender<Result<Event, Infallible>>,
    event_name: &str,
    data: &T,
) -> Result<(), ()> {
    match serde_json::to_string(data) {
        Ok(json) => {
            let event = Event::default().event(event_name).data(json);
            tx.send(Ok(event)).await.map_err(|_| ())
        }
        Err(e) => {
            error!("Failed to serialize {} event: {}", event_name, e);
            Err(())
        }
    }
}

/// Convert an error to an Anthropic-format error response.
fn to_anthropic_error(error_type: &str, message: &str) -> AnthropicErrorResponse {
    AnthropicErrorResponse {
        response_type: "error".to_string(),
        error: AnthropicErrorDetail {
            error_type: error_type.to_string(),
            message: message.to_string(),
        },
    }
}

pub async fn fallback() -> impl IntoResponse {
    AppError::NotFound("The requested endpoint does not exist".to_string())
}
