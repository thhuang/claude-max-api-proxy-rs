use crate::types::openai::{ChatCompletionRequest, Message, MessageContent};
use std::collections::HashMap;

/// Maps OpenAI model names to Claude CLI model aliases
fn model_map() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("claude-opus-4", "opus"),
        ("claude-sonnet-4", "sonnet"),
        ("claude-haiku-4", "haiku"),
        ("claude-code-cli/claude-opus-4", "opus"),
        ("claude-code-cli/claude-sonnet-4", "sonnet"),
        ("claude-code-cli/claude-haiku-4", "haiku"),
        ("opus", "opus"),
        ("sonnet", "sonnet"),
        ("haiku", "haiku"),
    ])
}

/// Extract the CLI model alias from an OpenAI model name.
/// Defaults to "opus" for unrecognized models.
pub fn extract_model(model: &str) -> &'static str {
    let map = model_map();

    if let Some(&alias) = map.get(model) {
        return alias;
    }

    // Try stripping "claude-code-cli/" prefix
    if let Some(stripped) = model.strip_prefix("claude-code-cli/") {
        if let Some(&alias) = map.get(stripped) {
            return alias;
        }
    }

    // Substring fallback for date-suffixed model IDs (e.g. "claude-opus-4-20250514")
    if model.contains("opus") {
        return "opus";
    }
    if model.contains("sonnet") {
        return "sonnet";
    }
    if model.contains("haiku") {
        return "haiku";
    }

    "opus"
}

/// Extract text from MessageContent
fn extract_text(content: &Option<MessageContent>) -> String {
    match content {
        Some(MessageContent::Text(s)) => s.clone(),
        Some(MessageContent::Parts(parts)) => parts
            .iter()
            .filter(|p| p.part_type == "text")
            .filter_map(|p| p.text.as_deref())
            .collect::<Vec<_>>()
            .join(""),
        None => String::new(),
    }
}

/// Convert OpenAI messages to a CLI prompt string.
///
/// - System messages are wrapped in `<system>` tags
/// - User messages are included as bare text
/// - Assistant messages are wrapped in `<previous_response>` tags
pub fn messages_to_prompt(messages: &[Message]) -> String {
    let mut parts: Vec<String> = Vec::new();

    for msg in messages {
        let text = extract_text(&msg.content);
        match msg.role.as_str() {
            "system" => {
                parts.push(format!("<system>\n{}\n</system>\n", text));
            }
            "user" => {
                parts.push(text);
            }
            "assistant" => {
                parts.push(format!("<previous_response>\n{}\n</previous_response>\n", text));
            }
            _ => {
                // Treat unknown roles as user messages
                parts.push(text);
            }
        }
    }

    parts.join("\n").trim().to_string()
}

/// Convert an OpenAI request to CLI arguments and prompt.
/// Returns (model_alias, prompt, optional_session_id).
pub fn openai_to_cli(request: &ChatCompletionRequest) -> (&'static str, String, Option<String>) {
    let model = request
        .model
        .as_deref()
        .map(extract_model)
        .unwrap_or("opus");

    let prompt = request
        .messages
        .as_ref()
        .map(|msgs| messages_to_prompt(msgs))
        .unwrap_or_default();

    let session_id = request.user.clone();

    (model, prompt, session_id)
}
