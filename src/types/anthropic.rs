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
