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
