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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::openai::ContentPart;

    // ── extract_model ─────────────────────────────────────────

    #[test]
    fn exact_model_names() {
        assert_eq!(extract_model("claude-opus-4"), "opus");
        assert_eq!(extract_model("claude-sonnet-4"), "sonnet");
        assert_eq!(extract_model("claude-haiku-4"), "haiku");
    }

    #[test]
    fn short_aliases() {
        assert_eq!(extract_model("opus"), "opus");
        assert_eq!(extract_model("sonnet"), "sonnet");
        assert_eq!(extract_model("haiku"), "haiku");
    }

    #[test]
    fn prefixed_model_names() {
        assert_eq!(extract_model("claude-code-cli/claude-opus-4"), "opus");
        assert_eq!(extract_model("claude-code-cli/claude-sonnet-4"), "sonnet");
        assert_eq!(extract_model("claude-code-cli/claude-haiku-4"), "haiku");
    }

    #[test]
    fn date_suffixed_model_names() {
        assert_eq!(extract_model("claude-opus-4-20250514"), "opus");
        assert_eq!(extract_model("claude-sonnet-4-5-20250929"), "sonnet");
        assert_eq!(extract_model("claude-haiku-4-5-20251001"), "haiku");
    }

    #[test]
    fn unknown_model_defaults_to_opus() {
        assert_eq!(extract_model("gpt-4"), "opus");
        assert_eq!(extract_model("unknown-model"), "opus");
        assert_eq!(extract_model(""), "opus");
    }

    // ── messages_to_prompt ────────────────────────────────────

    #[test]
    fn single_user_message() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Some(MessageContent::Text("Hello".to_string())),
        }];
        assert_eq!(messages_to_prompt(&messages), "Hello");
    }

    #[test]
    fn system_message_wrapped_in_tags() {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: Some(MessageContent::Text("You are helpful.".to_string())),
            },
            Message {
                role: "user".to_string(),
                content: Some(MessageContent::Text("Hi".to_string())),
            },
        ];
        let prompt = messages_to_prompt(&messages);
        assert!(prompt.starts_with("<system>\nYou are helpful.\n</system>"));
        assert!(prompt.contains("Hi"));
    }

    #[test]
    fn assistant_message_wrapped_in_previous_response() {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: Some(MessageContent::Text("Hi".to_string())),
            },
            Message {
                role: "assistant".to_string(),
                content: Some(MessageContent::Text("Hello!".to_string())),
            },
            Message {
                role: "user".to_string(),
                content: Some(MessageContent::Text("How are you?".to_string())),
            },
        ];
        let prompt = messages_to_prompt(&messages);
        assert!(prompt.contains("<previous_response>\nHello!\n</previous_response>"));
        assert!(prompt.contains("How are you?"));
    }

    #[test]
    fn multipart_content() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Some(MessageContent::Parts(vec![
                ContentPart {
                    part_type: "text".to_string(),
                    text: Some("Hello ".to_string()),
                },
                ContentPart {
                    part_type: "text".to_string(),
                    text: Some("world".to_string()),
                },
                ContentPart {
                    part_type: "image_url".to_string(),
                    text: None,
                },
            ])),
        }];
        assert_eq!(messages_to_prompt(&messages), "Hello world");
    }

    #[test]
    fn none_content_produces_empty_string() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: None,
        }];
        assert_eq!(messages_to_prompt(&messages), "");
    }

    #[test]
    fn unknown_role_treated_as_user() {
        let messages = vec![Message {
            role: "tool".to_string(),
            content: Some(MessageContent::Text("tool output".to_string())),
        }];
        assert_eq!(messages_to_prompt(&messages), "tool output");
    }

    // ── openai_to_cli ────────────────────────────────────────

    #[test]
    fn openai_to_cli_extracts_all_fields() {
        let request = ChatCompletionRequest {
            model: Some("claude-sonnet-4".to_string()),
            messages: Some(vec![Message {
                role: "user".to_string(),
                content: Some(MessageContent::Text("test".to_string())),
            }]),
            stream: false,
            user: Some("session-123".to_string()),
        };
        let (model, prompt, session_id) = openai_to_cli(&request);
        assert_eq!(model, "sonnet");
        assert_eq!(prompt, "test");
        assert_eq!(session_id, Some("session-123".to_string()));
    }

    #[test]
    fn openai_to_cli_defaults_no_model() {
        let request = ChatCompletionRequest {
            model: None,
            messages: Some(vec![Message {
                role: "user".to_string(),
                content: Some(MessageContent::Text("test".to_string())),
            }]),
            stream: false,
            user: None,
        };
        let (model, _, session_id) = openai_to_cli(&request);
        assert_eq!(model, "opus");
        assert_eq!(session_id, None);
    }

    #[test]
    fn openai_to_cli_no_messages() {
        let request = ChatCompletionRequest {
            model: Some("opus".to_string()),
            messages: None,
            stream: false,
            user: None,
        };
        let (_, prompt, _) = openai_to_cli(&request);
        assert_eq!(prompt, "");
    }
}
