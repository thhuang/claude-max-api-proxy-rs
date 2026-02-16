use axum::Router;
use axum::routing::{get, post};
use tower_http::cors::CorsLayer;

use crate::routes;
use crate::session::SessionManager;

#[derive(Clone)]
pub struct AppState {
    pub cwd: String,
    #[allow(dead_code)]
    pub session_manager: SessionManager,
}

pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::permissive();

    Router::new()
        .route("/health", get(routes::health))
        .route("/v1/models", get(routes::models))
        .route("/v1/chat/completions", post(routes::chat_completions))
        .route("/v1/messages", post(routes::messages))
        .fallback(routes::fallback)
        .layer(cors)
        .layer(axum::extract::DefaultBodyLimit::max(10 * 1024 * 1024)) // 10MB
        .with_state(state)
}
