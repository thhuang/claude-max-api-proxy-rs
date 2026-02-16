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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::claude_cli::{ModelUsage, ResultMessage};
    use std::collections::HashMap;

    // ── normalize_model_name ──────────────────────────────────

    #[test]
    fn normalize_opus() {
        assert_eq!(normalize_model_name("claude-opus-4-20250514"), "claude-opus-4");
        assert_eq!(normalize_model_name("opus"), "claude-opus-4");
    }

    #[test]
    fn normalize_sonnet() {
        assert_eq!(normalize_model_name("claude-sonnet-4-5-20250929"), "claude-sonnet-4");
        assert_eq!(normalize_model_name("sonnet"), "claude-sonnet-4");
    }

    #[test]
    fn normalize_haiku() {
        assert_eq!(normalize_model_name("claude-haiku-4-5-20251001"), "claude-haiku-4");
        assert_eq!(normalize_model_name("haiku"), "claude-haiku-4");
    }

    #[test]
    fn normalize_unknown_defaults_sonnet() {
        assert_eq!(normalize_model_name("gpt-4"), "claude-sonnet-4");
        assert_eq!(normalize_model_name(""), "claude-sonnet-4");
    }

    // ── cli_result_to_openai ─────────────────────────────────

    #[test]
    fn result_to_openai_basic() {
        let result = ResultMessage {
            result: Some("Hello world".to_string()),
            exit_code: Some(0),
            duration_ms: Some(1000),
            duration_api_ms: Some(800),
            num_turns: Some(1),
            model_usage: None,
        };
        let resp = cli_result_to_openai(&result, "abc123");
        assert_eq!(resp.id, "chatcmpl-abc123");
        assert_eq!(resp.object, "chat.completion");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.role, "assistant");
        assert_eq!(resp.choices[0].message.content, "Hello world");
        assert_eq!(resp.choices[0].finish_reason, "stop");
        assert!(resp.usage.is_none());
    }

    #[test]
    fn result_to_openai_with_usage() {
        let mut usage = HashMap::new();
        usage.insert(
            "claude-opus-4-20250514".to_string(),
            ModelUsage {
                input_tokens: Some(100),
                output_tokens: Some(50),
                cache_read_tokens: Some(10),
                cache_write_tokens: Some(5),
            },
        );
        let result = ResultMessage {
            result: Some("test".to_string()),
            exit_code: Some(0),
            duration_ms: None,
            duration_api_ms: None,
            num_turns: None,
            model_usage: Some(usage),
        };
        let resp = cli_result_to_openai(&result, "xyz");
        assert_eq!(resp.model, "claude-opus-4");
        let u = resp.usage.unwrap();
        assert_eq!(u.prompt_tokens, 100);
        assert_eq!(u.completion_tokens, 50);
        assert_eq!(u.total_tokens, 150);
    }

    #[test]
    fn result_to_openai_empty_result() {
        let result = ResultMessage {
            result: None,
            exit_code: Some(0),
            duration_ms: None,
            duration_api_ms: None,
            num_turns: None,
            model_usage: None,
        };
        let resp = cli_result_to_openai(&result, "id");
        assert_eq!(resp.choices[0].message.content, "");
    }

    // ── create_stream_chunk ──────────────────────────────────

    #[test]
    fn stream_chunk_first() {
        let chunk = create_stream_chunk("req1", "claude-sonnet-4", "Hello", true);
        assert_eq!(chunk.id, "chatcmpl-req1");
        assert_eq!(chunk.object, "chat.completion.chunk");
        assert_eq!(chunk.choices[0].delta.role, Some("assistant".to_string()));
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
        assert_eq!(chunk.choices[0].finish_reason, None);
    }

    #[test]
    fn stream_chunk_subsequent() {
        let chunk = create_stream_chunk("req1", "claude-sonnet-4", "world", false);
        assert_eq!(chunk.choices[0].delta.role, None);
        assert_eq!(chunk.choices[0].delta.content, Some("world".to_string()));
    }

    // ── create_done_chunk ────────────────────────────────────

    #[test]
    fn done_chunk() {
        let chunk = create_done_chunk("req1", "claude-opus-4-20250514");
        assert_eq!(chunk.model, "claude-opus-4");
        assert_eq!(chunk.choices[0].finish_reason, Some("stop".to_string()));
        assert_eq!(chunk.choices[0].delta.content, None);
        assert_eq!(chunk.choices[0].delta.role, None);
    }
}
