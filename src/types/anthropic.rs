#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ── Request types ──────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u64,
    pub messages: Vec<MessageInput>,
    #[serde(default)]
    pub stream: bool,
    pub system: Option<ContentInput>,
    pub metadata: Option<RequestMetadata>,
}

#[derive(Debug, Deserialize)]
pub struct MessageInput {
    pub role: String,
    pub content: ContentInput,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ContentInput {
    Text(String),
    Blocks(Vec<ContentBlockInput>),
}

#[derive(Debug, Deserialize)]
pub struct ContentBlockInput {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RequestMetadata {
    pub user_id: Option<String>,
}

// ── Non-streaming response ─────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
    pub usage: ResponseUsage,
}

#[derive(Debug, Serialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct ResponseUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
}

// ── Streaming event types ──────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct MessageStartEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub message: MessageStartPayload,
}

#[derive(Debug, Serialize)]
pub struct MessageStartPayload {
    pub id: String,
    #[serde(rename = "type")]
    pub payload_type: String,
    pub role: String,
    pub content: Vec<()>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: ResponseUsage,
}

#[derive(Debug, Serialize)]
pub struct ContentBlockStartEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub index: u32,
    pub content_block: ContentBlock,
}

#[derive(Debug, Serialize)]
pub struct PingEvent {
    #[serde(rename = "type")]
    pub event_type: String,
}

#[derive(Debug, Serialize)]
pub struct ContentBlockDeltaEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub index: u32,
    pub delta: TextDelta,
}

#[derive(Debug, Serialize)]
pub struct TextDelta {
    #[serde(rename = "type")]
    pub delta_type: String,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct ContentBlockStopEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub index: u32,
}

#[derive(Debug, Serialize)]
pub struct MessageDeltaEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub delta: MessageDeltaPayload,
    pub usage: OutputUsage,
}

#[derive(Debug, Serialize)]
pub struct MessageDeltaPayload {
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OutputUsage {
    pub output_tokens: u64,
}

#[derive(Debug, Serialize)]
pub struct MessageStopEvent {
    #[serde(rename = "type")]
    pub event_type: String,
}

// ── Error response ─────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct AnthropicErrorResponse {
    #[serde(rename = "type")]
    pub response_type: String,
    pub error: AnthropicErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_simple_text_message() {
        let json = r#"{"model":"claude-sonnet-4-5-20250929","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "claude-sonnet-4-5-20250929");
        assert_eq!(req.max_tokens, 100);
        assert!(!req.stream);
        assert_eq!(req.messages.len(), 1);
        match &req.messages[0].content {
            ContentInput::Text(t) => assert_eq!(t, "Hello"),
            other => panic!("Expected Text, got {:?}", other),
        }
    }

    #[test]
    fn deserialize_block_content() {
        let json = r#"{"model":"opus","max_tokens":50,"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image","source":{"type":"base64"}}]}]}"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            ContentInput::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert_eq!(blocks[0].block_type, "text");
                assert_eq!(blocks[0].text.as_deref(), Some("hi"));
                assert_eq!(blocks[1].block_type, "image");
                assert_eq!(blocks[1].text, None);
            }
            other => panic!("Expected Blocks, got {:?}", other),
        }
    }

    #[test]
    fn deserialize_with_system() {
        let json = r#"{"model":"opus","max_tokens":50,"messages":[{"role":"user","content":"hi"}],"system":"Be helpful"}"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        match req.system.as_ref().unwrap() {
            ContentInput::Text(t) => assert_eq!(t, "Be helpful"),
            other => panic!("Expected Text system, got {:?}", other),
        }
    }

    #[test]
    fn deserialize_with_system_blocks() {
        let json = r#"{"model":"opus","max_tokens":50,"messages":[{"role":"user","content":"hi"}],"system":[{"type":"text","text":"System text"}]}"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        match req.system.as_ref().unwrap() {
            ContentInput::Blocks(blocks) => {
                assert_eq!(blocks[0].text.as_deref(), Some("System text"));
            }
            other => panic!("Expected Blocks system, got {:?}", other),
        }
    }

    #[test]
    fn deserialize_with_metadata() {
        let json = r#"{"model":"opus","max_tokens":50,"messages":[{"role":"user","content":"hi"}],"metadata":{"user_id":"user-123"}}"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(
            req.metadata.unwrap().user_id,
            Some("user-123".to_string())
        );
    }

    #[test]
    fn deserialize_streaming() {
        let json = r#"{"model":"opus","max_tokens":50,"messages":[{"role":"user","content":"hi"}],"stream":true}"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert!(req.stream);
    }

    #[test]
    fn deserialize_multi_turn() {
        let json = r#"{"model":"opus","max_tokens":50,"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"},{"role":"user","content":"Bye"}]}"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[1].role, "assistant");
        assert_eq!(req.messages[2].role, "user");
    }

    // ── Serialization tests ──────────────────────────────────

    #[test]
    fn serialize_messages_response() {
        let resp = MessagesResponse {
            id: "msg_abc".to_string(),
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![ContentBlock {
                block_type: "text".to_string(),
                text: "Hello".to_string(),
            }],
            model: "claude-sonnet-4".to_string(),
            stop_reason: "end_turn".to_string(),
            stop_sequence: None,
            usage: ResponseUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["type"], "message");
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][0]["text"], "Hello");
        assert_eq!(json["stop_reason"], "end_turn");
        assert_eq!(json["usage"]["input_tokens"], 10);
    }

    #[test]
    fn serialize_error_response() {
        let err = AnthropicErrorResponse {
            response_type: "error".to_string(),
            error: AnthropicErrorDetail {
                error_type: "server_error".to_string(),
                message: "something failed".to_string(),
            },
        };
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "server_error");
        assert_eq!(json["error"]["message"], "something failed");
    }
}
