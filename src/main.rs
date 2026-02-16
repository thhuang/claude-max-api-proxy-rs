mod adapter;
mod error;
mod routes;
mod server;
mod session;
mod subprocess;
mod types;

use clap::Parser;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tracing::{error, info};

#[derive(Parser)]
#[command(name = "claude-max-api")]
#[command(about = "OpenAI & Anthropic-compatible API proxy for Claude Code CLI")]
struct Args {
    /// Port to listen on
    #[arg(default_value = "8080")]
    port: u16,

    /// Working directory for the Claude CLI subprocess
    #[arg(long = "cwd", default_value = ".")]
    cwd: String,
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "claude_max_api=info".parse().unwrap()),
        )
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();

    // Resolve cwd to absolute path
    let cwd = std::fs::canonicalize(&args.cwd)
        .unwrap_or_else(|_| std::path::PathBuf::from(&args.cwd))
        .to_string_lossy()
        .to_string();

    // Verify claude CLI is available
    match tokio::process::Command::new("claude")
        .arg("--version")
        .output()
        .await
    {
        Ok(output) => {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            info!("Found claude CLI: {}", version);
        }
        Err(e) => {
            error!("claude CLI not found: {}. Install it with: npm install -g @anthropic-ai/claude-code", e);
            std::process::exit(1);
        }
    }

    // Set up session manager with cleanup task
    let session_manager = session::SessionManager::new();
    session_manager.spawn_cleanup_task();

    let state = server::AppState {
        cwd: cwd.clone(),
        session_manager,
    };

    let app = server::create_router(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));

    let listener = match TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            error!("Failed to bind to {}: {}", addr, e);
            if e.kind() == std::io::ErrorKind::AddrInUse {
                error!("Port {} is already in use", args.port);
            }
            std::process::exit(1);
        }
    };

    info!("╔══════════════════════════════════════════╗");
    info!("║     Claude Max API Proxy (Rust)          ║");
    info!("╠══════════════════════════════════════════╣");
    info!("║  Listening: http://127.0.0.1:{:<5}       ║", args.port);
    info!("║  CWD:       {:<29}║", truncate_str(&cwd, 29));
    info!("╚══════════════════════════════════════════╝");
    info!("");
    info!("Endpoints:");
    info!("  GET  /health              - Health check");
    info!("  GET  /v1/models           - List models");
    info!("  POST /v1/chat/completions - Chat completions (OpenAI)");
    info!("  POST /v1/messages         - Messages (Anthropic)");
    info!("");

    // Graceful shutdown on SIGINT/SIGTERM
    let shutdown = async {
        let ctrl_c = tokio::signal::ctrl_c();
        #[cfg(unix)]
        {
            let mut sigterm =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("failed to install SIGTERM handler");
            tokio::select! {
                _ = ctrl_c => { info!("Received SIGINT, shutting down..."); }
                _ = sigterm.recv() => { info!("Received SIGTERM, shutting down..."); }
            }
        }
        #[cfg(not(unix))]
        {
            ctrl_c.await.ok();
            info!("Received SIGINT, shutting down...");
        }
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .unwrap_or_else(|e| {
            error!("Server error: {}", e);
            std::process::exit(1);
        });

    info!("Server stopped.");
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("...{}", &s[s.len() - (max_len - 3)..])
    }
}
