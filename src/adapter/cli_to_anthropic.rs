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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::claude_cli::{ModelUsage, ResultMessage};
    use std::collections::HashMap;

    // ── cli_result_to_anthropic ───────────────────────────────

    #[test]
    fn result_to_anthropic_basic() {
        let result = ResultMessage {
            result: Some("Hello".to_string()),
            exit_code: Some(0),
            duration_ms: None,
            duration_api_ms: None,
            num_turns: None,
            model_usage: None,
        };
        let resp = cli_result_to_anthropic(&result, "msg1");
        assert_eq!(resp.id, "msg_msg1");
        assert_eq!(resp.response_type, "message");
        assert_eq!(resp.role, "assistant");
        assert_eq!(resp.content.len(), 1);
        assert_eq!(resp.content[0].block_type, "text");
        assert_eq!(resp.content[0].text, "Hello");
        assert_eq!(resp.stop_reason, "end_turn");
        assert_eq!(resp.stop_sequence, None);
    }

    #[test]
    fn result_to_anthropic_with_usage() {
        let mut usage = HashMap::new();
        usage.insert(
            "claude-sonnet-4-5-20250929".to_string(),
            ModelUsage {
                input_tokens: Some(200),
                output_tokens: Some(100),
                cache_read_tokens: Some(30),
                cache_write_tokens: Some(20),
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
        let resp = cli_result_to_anthropic(&result, "id");
        assert_eq!(resp.model, "claude-sonnet-4");
        assert_eq!(resp.usage.input_tokens, 200);
        assert_eq!(resp.usage.output_tokens, 100);
        assert_eq!(resp.usage.cache_creation_input_tokens, 20);
        assert_eq!(resp.usage.cache_read_input_tokens, 30);
    }

    #[test]
    fn result_to_anthropic_empty() {
        let result = ResultMessage {
            result: None,
            exit_code: Some(1),
            duration_ms: None,
            duration_api_ms: None,
            num_turns: None,
            model_usage: None,
        };
        let resp = cli_result_to_anthropic(&result, "x");
        assert_eq!(resp.content[0].text, "");
        assert_eq!(resp.usage.input_tokens, 0);
        assert_eq!(resp.usage.output_tokens, 0);
    }

    // ── streaming event builders ─────────────────────────────

    #[test]
    fn message_start_event() {
        let event = create_message_start("req1", "claude-opus-4");
        assert_eq!(event.event_type, "message_start");
        assert_eq!(event.message.id, "msg_req1");
        assert_eq!(event.message.role, "assistant");
        assert_eq!(event.message.model, "claude-opus-4");
        assert!(event.message.content.is_empty());
    }

    #[test]
    fn content_block_start_event() {
        let event = create_content_block_start();
        assert_eq!(event.event_type, "content_block_start");
        assert_eq!(event.index, 0);
        assert_eq!(event.content_block.block_type, "text");
        assert_eq!(event.content_block.text, "");
    }

    #[test]
    fn ping_event() {
        let event = create_ping();
        assert_eq!(event.event_type, "ping");
    }

    #[test]
    fn content_block_delta_event() {
        let event = create_content_block_delta("hello");
        assert_eq!(event.event_type, "content_block_delta");
        assert_eq!(event.delta.delta_type, "text_delta");
        assert_eq!(event.delta.text, "hello");
    }

    #[test]
    fn content_block_stop_event() {
        let event = create_content_block_stop();
        assert_eq!(event.event_type, "content_block_stop");
        assert_eq!(event.index, 0);
    }

    #[test]
    fn message_delta_event() {
        let event = create_message_delta(42);
        assert_eq!(event.event_type, "message_delta");
        assert_eq!(event.delta.stop_reason, "end_turn");
        assert_eq!(event.usage.output_tokens, 42);
    }

    #[test]
    fn message_stop_event() {
        let event = create_message_stop();
        assert_eq!(event.event_type, "message_stop");
    }

    // ── JSON serialization spot checks ───────────────────────

    #[test]
    fn message_start_serializes_correctly() {
        let event = create_message_start("abc", "claude-sonnet-4");
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "message_start");
        assert_eq!(json["message"]["type"], "message");
        assert_eq!(json["message"]["id"], "msg_abc");
    }

    #[test]
    fn content_block_delta_serializes_correctly() {
        let event = create_content_block_delta("chunk");
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "content_block_delta");
        assert_eq!(json["delta"]["type"], "text_delta");
        assert_eq!(json["delta"]["text"], "chunk");
    }

    #[test]
    fn result_response_serializes_correctly() {
        let result = ResultMessage {
            result: Some("response text".to_string()),
            exit_code: Some(0),
            duration_ms: None,
            duration_api_ms: None,
            num_turns: None,
            model_usage: None,
        };
        let resp = cli_result_to_anthropic(&result, "test-id");
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["type"], "message");
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][0]["text"], "response text");
        assert_eq!(json["stop_reason"], "end_turn");
    }
}
