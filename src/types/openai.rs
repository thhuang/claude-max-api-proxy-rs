use serde::{Deserialize, Serialize};

/// OpenAI chat completion request
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Option<Vec<Message>>,
    #[serde(default)]
    pub stream: bool,
    pub user: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Option<MessageContent>,
}

/// Message content can be a simple string or an array of content parts
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub part_type: String,
    pub text: Option<String>,
}

/// OpenAI chat completion response (non-streaming)
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// OpenAI streaming chunk
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// OpenAI error response format
#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

/// Models list response
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub created: u64,
    pub context_window: u64,
    pub max_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_simple_message() {
        let json = r#"{"model":"claude-sonnet-4","messages":[{"role":"user","content":"Hello"}],"stream":false}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model.as_deref(), Some("claude-sonnet-4"));
        assert!(!req.stream);
        let msgs = req.messages.unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
        match &msgs[0].content {
            Some(MessageContent::Text(t)) => assert_eq!(t, "Hello"),
            other => panic!("Expected Text, got {:?}", other),
        }
    }

    #[test]
    fn deserialize_multipart_content() {
        let json = r#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"data:..."}}]}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let msgs = req.messages.unwrap();
        match &msgs[0].content {
            Some(MessageContent::Parts(parts)) => {
                assert_eq!(parts.len(), 2);
                assert_eq!(parts[0].part_type, "text");
                assert_eq!(parts[0].text.as_deref(), Some("hi"));
                assert_eq!(parts[1].part_type, "image_url");
                assert_eq!(parts[1].text, None);
            }
            other => panic!("Expected Parts, got {:?}", other),
        }
    }

    #[test]
    fn deserialize_minimal_request() {
        let json = r#"{"messages":[{"role":"user","content":"test"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, None);
        assert!(!req.stream); // default
        assert_eq!(req.user, None);
    }

    #[test]
    fn deserialize_with_user_and_stream() {
        let json = r#"{"model":"opus","messages":[{"role":"user","content":"hi"}],"stream":true,"user":"session-42"}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.stream);
        assert_eq!(req.user, Some("session-42".to_string()));
    }

    #[test]
    fn deserialize_multi_turn() {
        let json = r#"{"messages":[{"role":"system","content":"Be brief"},{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"},{"role":"user","content":"Bye"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let msgs = req.messages.unwrap();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[2].role, "assistant");
        assert_eq!(msgs[3].role, "user");
    }

    #[test]
    fn serialize_response() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-abc".to_string(),
            object: "chat.completion".to_string(),
            created: 1000,
            model: "claude-sonnet-4".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".to_string(),
                    content: "Hello".to_string(),
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["id"], "chatcmpl-abc");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["total_tokens"], 15);
    }

    #[test]
    fn serialize_response_no_usage() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-abc".to_string(),
            object: "chat.completion".to_string(),
            created: 1000,
            model: "claude-sonnet-4".to_string(),
            choices: vec![],
            usage: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert!(json.get("usage").is_none()); // skip_serializing_if
    }

    #[test]
    fn serialize_chunk_skips_none() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-x".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1000,
            model: "claude-sonnet-4".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert!(json["choices"][0]["delta"].get("role").is_none());
        assert!(json["choices"][0]["delta"].get("content").is_none());
    }
}
