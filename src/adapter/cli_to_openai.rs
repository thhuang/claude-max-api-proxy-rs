use crate::types::claude_cli::ResultMessage;
use crate::types::openai::{
    ChatCompletionChunk, ChatCompletionResponse, Choice, ChunkChoice, ChunkDelta, ResponseMessage,
    Usage,
};
use std::time::{SystemTime, UNIX_EPOCH};

/// Normalize a full Claude model string to the short OpenAI-style name.
/// e.g. "claude-sonnet-4-5-20250929" → "claude-sonnet-4"
pub fn normalize_model_name(model: &str) -> &'static str {
    if model.contains("opus") {
        "claude-opus-4"
    } else if model.contains("sonnet") {
        "claude-sonnet-4"
    } else if model.contains("haiku") {
        "claude-haiku-4"
    } else {
        "claude-sonnet-4"
    }
}

fn unix_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Convert a CLI result message to an OpenAI chat completion response.
pub fn cli_result_to_openai(result: &ResultMessage, request_id: &str) -> ChatCompletionResponse {
    let content = result.result.clone().unwrap_or_default();

    // Get model from modelUsage (first key), default to "claude-sonnet-4"
    let model = result
        .model_usage
        .as_ref()
        .and_then(|mu| mu.keys().next())
        .map(|m| normalize_model_name(m))
        .unwrap_or("claude-sonnet-4");

    // Calculate usage from modelUsage
    let usage = result.model_usage.as_ref().map(|mu| {
        let mut input_tokens = 0u64;
        let mut output_tokens = 0u64;
        for u in mu.values() {
            input_tokens += u.input_tokens.unwrap_or(0);
            output_tokens += u.output_tokens.unwrap_or(0);
        }
        Usage {
            prompt_tokens: input_tokens,
            completion_tokens: output_tokens,
            total_tokens: input_tokens + output_tokens,
        }
    });

    ChatCompletionResponse {
        id: format!("chatcmpl-{}", request_id),
        object: "chat.completion".to_string(),
        created: unix_epoch_secs(),
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".to_string(),
                content,
            },
            finish_reason: "stop".to_string(),
        }],
        usage,
    }
}

/// Create a streaming content chunk.
pub fn create_stream_chunk(
    request_id: &str,
    model: &str,
    text: &str,
    is_first: bool,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: format!("chatcmpl-{}", request_id),
        object: "chat.completion.chunk".to_string(),
        created: unix_epoch_secs(),
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: if is_first {
                    Some("assistant".to_string())
                } else {
                    None
                },
                content: Some(text.to_string()),
            },
            finish_reason: None,
        }],
    }
}

/// Create the final "done" chunk with finish_reason: "stop".
pub fn create_done_chunk(request_id: &str, model: &str) -> ChatCompletionChunk {
    let normalized = normalize_model_name(model);
    ChatCompletionChunk {
        id: format!("chatcmpl-{}", request_id),
        object: "chat.completion.chunk".to_string(),
        created: unix_epoch_secs(),
        model: normalized.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: None,
                content: None,
            },
            finish_reason: Some("stop".to_string()),
        }],
    }
}
