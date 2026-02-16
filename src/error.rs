use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum AppError {
    #[error("Invalid request: {0}")]
    BadRequest(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Subprocess error: {0}")]
    Subprocess(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_type, code, message) = match &self {
            AppError::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                Some("invalid_messages"),
                msg.clone(),
            ),
            AppError::NotFound(msg) => (
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                Some("not_found"),
                msg.clone(),
            ),
            AppError::Internal(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                None,
                msg.clone(),
            ),
            AppError::Subprocess(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                None,
                msg.clone(),
            ),
        };

        let body = json!({
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            }
        });

        (status, axum::Json(body)).into_response()
    }
}
