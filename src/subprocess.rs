use crate::types::claude_cli::{AssistantInner, ClaudeCliMessage, Delta, StreamEvent};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

const INACTIVITY_TIMEOUT: Duration = Duration::from_secs(30 * 60); // 30 minutes

/// Events emitted by the subprocess to the route handler.
#[derive(Debug)]
pub enum SubprocessEvent {
    /// Model name from the assistant message
    Model(String),
    /// A content delta (streaming text)
    ContentDelta(String),
    /// The final result message
    Result(crate::types::claude_cli::ResultMessage),
    /// An error occurred
    Error(String),
    /// Process exited (exit_code)
    Close(i32),
}

pub struct SubprocessOptions {
    pub model: String,
    pub session_id: Option<String>,
    pub cwd: String,
}

fn build_args(prompt: &str, options: &SubprocessOptions) -> Vec<String> {
    let mut args = vec![
        "--print".to_string(),
        "--output-format".to_string(),
        "stream-json".to_string(),
        "--verbose".to_string(),
        "--include-partial-messages".to_string(),
        "--model".to_string(),
        options.model.clone(),
        "--no-session-persistence".to_string(),
        "--permission-mode".to_string(),
        "bypassPermissions".to_string(),
        prompt.to_string(),
    ];

    if let Some(ref session_id) = options.session_id {
        args.push("--session-id".to_string());
        args.push(session_id.clone());
    }

    args
}

/// Spawn the claude CLI subprocess and send events through the channel.
/// Returns immediately; events are sent asynchronously.
/// When the receiver is dropped (client disconnect), the sender will error and the subprocess
/// will be killed.
pub async fn spawn_subprocess(
    prompt: String,
    options: SubprocessOptions,
    tx: mpsc::Sender<SubprocessEvent>,
) {
    let args = build_args(&prompt, &options);
    let start = Instant::now();

    info!(
        "Spawning claude subprocess with model={}, cwd={}",
        options.model, options.cwd
    );

    let mut child = match Command::new("claude")
        .args(&args)
        .current_dir(&options.cwd)
        .env("CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS", "1")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            let msg = if e.kind() == std::io::ErrorKind::NotFound {
                "claude CLI not found. Install it with: npm install -g @anthropic-ai/claude-code"
                    .to_string()
            } else {
                format!("Failed to spawn claude: {}", e)
            };
            let _ = tx.send(SubprocessEvent::Error(msg)).await;
            return;
        }
    };

    let pid = child.id().unwrap_or(0);
    info!("Claude subprocess started with PID {}", pid);

    let stdout = child.stdout.take().expect("stdout not captured");
    let stderr = child.stderr.take().expect("stderr not captured");

    let mut stdout_reader = BufReader::new(stdout).lines();
    let mut stderr_reader = BufReader::new(stderr).lines();
    let mut first_token = true;
    let inactivity_timeout = tokio::time::sleep(INACTIVITY_TIMEOUT);
    tokio::pin!(inactivity_timeout);

    loop {
        tokio::select! {
            line = stdout_reader.next_line() => {
                match line {
                    Ok(Some(line)) => {
                        // Reset inactivity timer
                        inactivity_timeout.as_mut().reset(tokio::time::Instant::now() + INACTIVITY_TIMEOUT);

                        if line.trim().is_empty() {
                            continue;
                        }

                        match process_line(&line) {
                            Some(events) => {
                                for event in events {
                                    if first_token {
                                        if matches!(&event, SubprocessEvent::ContentDelta(_)) {
                                            let elapsed = start.elapsed();
                                            info!("First token after {:.2}s", elapsed.as_secs_f64());
                                            first_token = false;
                                        }
                                    }
                                    if tx.send(event).await.is_err() {
                                        // Receiver dropped (client disconnected)
                                        info!("Client disconnected, killing subprocess PID {}", pid);
                                        let _ = child.kill().await;
                                        return;
                                    }
                                }
                            }
                            None => {
                                debug!("Ignoring non-JSON line: {}", line);
                            }
                        }
                    }
                    Ok(None) => {
                        // stdout closed
                        break;
                    }
                    Err(e) => {
                        error!("Error reading stdout: {}", e);
                        break;
                    }
                }
            }
            line = stderr_reader.next_line() => {
                match line {
                    Ok(Some(line)) => {
                        // Reset inactivity timer on stderr too
                        inactivity_timeout.as_mut().reset(tokio::time::Instant::now() + INACTIVITY_TIMEOUT);
                        debug!("claude stderr: {}", line);
                    }
                    Ok(None) => {
                        // stderr closed
                    }
                    Err(e) => {
                        debug!("Error reading stderr: {}", e);
                    }
                }
            }
            () = &mut inactivity_timeout => {
                warn!("Inactivity timeout after 30 minutes, killing subprocess PID {}", pid);
                let _ = tx.send(SubprocessEvent::Error("Inactivity timeout after 30 minutes".to_string())).await;
                let _ = child.kill().await;
                return;
            }
        }
    }

    // Wait for process to exit
    let exit_code = match child.wait().await {
        Ok(status) => status.code().unwrap_or(-1),
        Err(e) => {
            error!("Error waiting for subprocess: {}", e);
            -1
        }
    };

    let elapsed = start.elapsed();
    info!(
        "Claude subprocess PID {} exited with code {} after {:.2}s",
        pid,
        exit_code,
        elapsed.as_secs_f64()
    );

    let _ = tx.send(SubprocessEvent::Close(exit_code)).await;
}

/// Parse a single line of NDJSON output and return subprocess events.
fn process_line(line: &str) -> Option<Vec<SubprocessEvent>> {
    // First, try to parse as a top-level message
    if let Ok(msg) = serde_json::from_str::<ClaudeCliMessage>(line) {
        return Some(process_cli_message(msg));
    }

    // Try to parse as a stream event (partial message content)
    if let Ok(event) = serde_json::from_str::<StreamEvent>(line) {
        return Some(process_stream_event(event));
    }

    // Not JSON we recognize
    None
}

fn process_cli_message(msg: ClaudeCliMessage) -> Vec<SubprocessEvent> {
    match msg {
        ClaudeCliMessage::System(_) => {
            // System messages are informational
            vec![]
        }
        ClaudeCliMessage::Assistant(assistant_msg) => {
            let mut events = Vec::new();

            // Extract model name
            if let Some(AssistantInner {
                model: Some(model), ..
            }) = &assistant_msg.message
            {
                events.push(SubprocessEvent::Model(model.clone()));
            }

            // Check for inline content (non-streaming assistant messages)
            if let Some(AssistantInner {
                content: Some(blocks),
                ..
            }) = &assistant_msg.message
            {
                for block in blocks {
                    if let Some(text) = &block.text {
                        if !text.is_empty() {
                            events.push(SubprocessEvent::ContentDelta(text.clone()));
                        }
                    }
                }
            }

            events
        }
        ClaudeCliMessage::Result(result) => {
            vec![SubprocessEvent::Result(result)]
        }
    }
}

fn process_stream_event(event: StreamEvent) -> Vec<SubprocessEvent> {
    match event {
        StreamEvent::ContentBlockDelta {
            delta: Delta {
                text: Some(text), ..
            },
            ..
        } if !text.is_empty() => {
            vec![SubprocessEvent::ContentDelta(text)]
        }
        _ => vec![],
    }
}
