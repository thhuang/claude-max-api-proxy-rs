#![allow(dead_code)]

use serde::Deserialize;
use std::collections::HashMap;

/// Represents the different message types from the Claude CLI's stream-json output.
/// Each line of stdout is one of these JSON objects.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ClaudeCliMessage {
    #[serde(rename = "system")]
    System(SystemMessage),

    #[serde(rename = "assistant")]
    Assistant(AssistantMessage),

    #[serde(rename = "result")]
    Result(ResultMessage),
}

/// A streaming event within the assistant message flow.
/// These appear as nested JSON within assistant messages when using --include-partial-messages.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: Option<u64>,
        content_block: Option<ContentBlock>,
    },

    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: Option<u64>, delta: Delta },

    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: Option<u64> },

    #[serde(rename = "message_start")]
    MessageStart {},

    #[serde(rename = "message_delta")]
    MessageDelta {},

    #[serde(rename = "message_stop")]
    MessageStop {},
}

#[derive(Debug, Deserialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub block_type: Option<String>,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Delta {
    #[serde(rename = "type")]
    pub delta_type: Option<String>,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SystemMessage {
    pub subtype: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AssistantMessage {
    pub message: Option<AssistantInner>,
}

#[derive(Debug, Deserialize)]
pub struct AssistantInner {
    pub model: Option<String>,
    pub content: Option<Vec<ContentBlock>>,
}

#[derive(Debug, Deserialize)]
pub struct ResultMessage {
    pub result: Option<String>,
    #[serde(rename = "exitCode")]
    pub exit_code: Option<i32>,
    pub duration_ms: Option<u64>,
    pub duration_api_ms: Option<u64>,
    pub num_turns: Option<u64>,
    #[serde(rename = "modelUsage")]
    pub model_usage: Option<HashMap<String, ModelUsage>>,
}

#[derive(Debug, Deserialize)]
pub struct ModelUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
}
