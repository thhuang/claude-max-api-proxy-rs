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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http::Response;

    async fn body_to_json(response: Response<Body>) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn bad_request_returns_400() {
        let err = AppError::BadRequest("missing field".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let json = body_to_json(response).await;
        assert_eq!(json["error"]["message"], "missing field");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["code"], "invalid_messages");
    }

    #[tokio::test]
    async fn not_found_returns_404() {
        let err = AppError::NotFound("endpoint not found".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        let json = body_to_json(response).await;
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["code"], "not_found");
    }

    #[tokio::test]
    async fn internal_returns_500() {
        let err = AppError::Internal("something broke".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let json = body_to_json(response).await;
        assert_eq!(json["error"]["type"], "server_error");
        assert!(json["error"]["code"].is_null());
    }

    #[tokio::test]
    async fn subprocess_returns_500() {
        let err = AppError::Subprocess("process crashed".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let json = body_to_json(response).await;
        assert_eq!(json["error"]["message"], "process crashed");
        assert_eq!(json["error"]["type"], "server_error");
    }

    #[test]
    fn display_trait() {
        assert_eq!(
            AppError::BadRequest("x".to_string()).to_string(),
            "Invalid request: x"
        );
        assert_eq!(
            AppError::NotFound("y".to_string()).to_string(),
            "Not found: y"
        );
        assert_eq!(
            AppError::Internal("z".to_string()).to_string(),
            "Internal error: z"
        );
        assert_eq!(
            AppError::Subprocess("w".to_string()).to_string(),
            "Subprocess error: w"
        );
    }
}
