use crate::adapter::openai_to_cli::extract_model;
use crate::types::anthropic::{ContentInput, MessagesRequest};

/// Extract text from an Anthropic ContentInput (string or array of blocks).
fn extract_text(content: &ContentInput) -> String {
    match content {
        ContentInput::Text(s) => s.clone(),
        ContentInput::Blocks(blocks) => blocks
            .iter()
            .filter(|b| b.block_type == "text")
            .filter_map(|b| b.text.as_deref())
            .collect::<Vec<_>>()
            .join(""),
    }
}

/// Convert Anthropic messages (with optional top-level system) to a CLI prompt string.
///
/// - System text is wrapped in `<system>` tags at the top
/// - User messages are included as bare text
/// - Assistant messages are wrapped in `<previous_response>` tags
fn messages_to_prompt(system: Option<&ContentInput>, messages: &[crate::types::anthropic::MessageInput]) -> String {
    let mut parts: Vec<String> = Vec::new();

    if let Some(sys) = system {
        let sys_text = extract_text(sys);
        if !sys_text.is_empty() {
            parts.push(format!("<system>\n{}\n</system>\n", sys_text));
        }
    }

    for msg in messages {
        let text = extract_text(&msg.content);
        match msg.role.as_str() {
            "user" => parts.push(text),
            "assistant" => {
                parts.push(format!("<previous_response>\n{}\n</previous_response>\n", text));
            }
            _ => parts.push(text),
        }
    }

    parts.join("\n").trim().to_string()
}

/// Convert an Anthropic MessagesRequest to CLI arguments.
/// Returns (model_alias, prompt, optional_session_id).
pub fn anthropic_to_cli(request: &MessagesRequest) -> (&'static str, String, Option<String>) {
    let model = extract_model(&request.model);
    let prompt = messages_to_prompt(request.system.as_ref(), &request.messages);
    let session_id = request
        .metadata
        .as_ref()
        .and_then(|m| m.user_id.clone());

    (model, prompt, session_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::anthropic::{ContentBlockInput, MessageInput, RequestMetadata};

    // ── extract_text ──────────────────────────────────────────

    #[test]
    fn extract_text_from_string() {
        let content = ContentInput::Text("hello".to_string());
        assert_eq!(extract_text(&content), "hello");
    }

    #[test]
    fn extract_text_from_blocks() {
        let content = ContentInput::Blocks(vec![
            ContentBlockInput {
                block_type: "text".to_string(),
                text: Some("hello ".to_string()),
            },
            ContentBlockInput {
                block_type: "image".to_string(),
                text: None,
            },
            ContentBlockInput {
                block_type: "text".to_string(),
                text: Some("world".to_string()),
            },
        ]);
        assert_eq!(extract_text(&content), "hello world");
    }

    #[test]
    fn extract_text_empty_blocks() {
        let content = ContentInput::Blocks(vec![]);
        assert_eq!(extract_text(&content), "");
    }

    // ── messages_to_prompt ────────────────────────────────────

    #[test]
    fn system_at_top() {
        let system = ContentInput::Text("Be helpful.".to_string());
        let messages = vec![MessageInput {
            role: "user".to_string(),
            content: ContentInput::Text("Hi".to_string()),
        }];
        let prompt = messages_to_prompt(Some(&system), &messages);
        assert!(prompt.starts_with("<system>\nBe helpful.\n</system>"));
        assert!(prompt.contains("Hi"));
    }

    #[test]
    fn no_system() {
        let messages = vec![MessageInput {
            role: "user".to_string(),
            content: ContentInput::Text("Hi".to_string()),
        }];
        let prompt = messages_to_prompt(None, &messages);
        assert_eq!(prompt, "Hi");
    }

    #[test]
    fn empty_system_omitted() {
        let system = ContentInput::Text("".to_string());
        let messages = vec![MessageInput {
            role: "user".to_string(),
            content: ContentInput::Text("Hi".to_string()),
        }];
        let prompt = messages_to_prompt(Some(&system), &messages);
        assert!(!prompt.contains("<system>"));
        assert_eq!(prompt, "Hi");
    }

    #[test]
    fn assistant_wrapped() {
        let messages = vec![
            MessageInput {
                role: "user".to_string(),
                content: ContentInput::Text("Hi".to_string()),
            },
            MessageInput {
                role: "assistant".to_string(),
                content: ContentInput::Text("Hello!".to_string()),
            },
            MessageInput {
                role: "user".to_string(),
                content: ContentInput::Text("How?".to_string()),
            },
        ];
        let prompt = messages_to_prompt(None, &messages);
        assert!(prompt.contains("<previous_response>\nHello!\n</previous_response>"));
    }

    #[test]
    fn unknown_role_as_user() {
        let messages = vec![MessageInput {
            role: "tool".to_string(),
            content: ContentInput::Text("result".to_string()),
        }];
        let prompt = messages_to_prompt(None, &messages);
        assert_eq!(prompt, "result");
    }

    // ── anthropic_to_cli ─────────────────────────────────────

    #[test]
    fn anthropic_to_cli_full() {
        let request = MessagesRequest {
            model: "claude-sonnet-4-5-20250929".to_string(),
            max_tokens: 100,
            messages: vec![MessageInput {
                role: "user".to_string(),
                content: ContentInput::Text("test".to_string()),
            }],
            stream: false,
            system: Some(ContentInput::Text("system prompt".to_string())),
            metadata: Some(RequestMetadata {
                user_id: Some("user-42".to_string()),
            }),
        };
        let (model, prompt, session_id) = anthropic_to_cli(&request);
        assert_eq!(model, "sonnet");
        assert!(prompt.contains("<system>"));
        assert!(prompt.contains("test"));
        assert_eq!(session_id, Some("user-42".to_string()));
    }

    #[test]
    fn anthropic_to_cli_minimal() {
        let request = MessagesRequest {
            model: "opus".to_string(),
            max_tokens: 50,
            messages: vec![MessageInput {
                role: "user".to_string(),
                content: ContentInput::Text("hi".to_string()),
            }],
            stream: true,
            system: None,
            metadata: None,
        };
        let (model, prompt, session_id) = anthropic_to_cli(&request);
        assert_eq!(model, "opus");
        assert_eq!(prompt, "hi");
        assert_eq!(session_id, None);
    }
}
