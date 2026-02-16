use crate::adapter::cli_to_openai::normalize_model_name;
use crate::types::anthropic::*;
use crate::types::claude_cli::ResultMessage;

/// Convert a CLI ResultMessage to an Anthropic MessagesResponse.
pub fn cli_result_to_anthropic(result: &ResultMessage, message_id: &str) -> MessagesResponse {
    let content_text = result.result.clone().unwrap_or_default();

    let model = result
        .model_usage
        .as_ref()
        .and_then(|mu| mu.keys().next())
        .map(|m| normalize_model_name(m))
        .unwrap_or("claude-sonnet-4");

    let (input_tokens, output_tokens, cache_write, cache_read) =
        result
            .model_usage
            .as_ref()
            .map(|mu| {
                let mut inp = 0u64;
                let mut out = 0u64;
                let mut cw = 0u64;
                let mut cr = 0u64;
                for u in mu.values() {
                    inp += u.input_tokens.unwrap_or(0);
                    out += u.output_tokens.unwrap_or(0);
                    cw += u.cache_write_tokens.unwrap_or(0);
                    cr += u.cache_read_tokens.unwrap_or(0);
                }
                (inp, out, cw, cr)
            })
            .unwrap_or((0, 0, 0, 0));

    MessagesResponse {
        id: format!("msg_{}", message_id),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content: vec![ContentBlock {
            block_type: "text".to_string(),
            text: content_text,
        }],
        model: model.to_string(),
        stop_reason: "end_turn".to_string(),
        stop_sequence: None,
        usage: ResponseUsage {
            input_tokens,
            output_tokens,
            cache_creation_input_tokens: cache_write,
            cache_read_input_tokens: cache_read,
        },
    }
}

// ── Streaming event builders ───────────────────────────────────

pub fn create_message_start(id: &str, model: &str) -> MessageStartEvent {
    MessageStartEvent {
        event_type: "message_start".to_string(),
        message: MessageStartPayload {
            id: format!("msg_{}", id),
            payload_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![],
            model: model.to_string(),
            stop_reason: None,
            stop_sequence: None,
            usage: ResponseUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        },
    }
}

pub fn create_content_block_start() -> ContentBlockStartEvent {
    ContentBlockStartEvent {
        event_type: "content_block_start".to_string(),
        index: 0,
        content_block: ContentBlock {
            block_type: "text".to_string(),
            text: String::new(),
        },
    }
}

pub fn create_ping() -> PingEvent {
    PingEvent {
        event_type: "ping".to_string(),
    }
}

pub fn create_content_block_delta(text: &str) -> ContentBlockDeltaEvent {
    ContentBlockDeltaEvent {
        event_type: "content_block_delta".to_string(),
        index: 0,
        delta: TextDelta {
            delta_type: "text_delta".to_string(),
            text: text.to_string(),
        },
    }
}

pub fn create_content_block_stop() -> ContentBlockStopEvent {
    ContentBlockStopEvent {
        event_type: "content_block_stop".to_string(),
        index: 0,
    }
}

pub fn create_message_delta(output_tokens: u64) -> MessageDeltaEvent {
    MessageDeltaEvent {
        event_type: "message_delta".to_string(),
        delta: MessageDeltaPayload {
            stop_reason: "end_turn".to_string(),
            stop_sequence: None,
        },
        usage: OutputUsage { output_tokens },
    }
}

pub fn create_message_stop() -> MessageStopEvent {
    MessageStopEvent {
        event_type: "message_stop".to_string(),
    }
}
