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
    pub request_id: String,
    pub model: String,
    pub session_id: Option<String>,
    pub cwd: String,
    pub api: &'static str, // "openai" or "anthropic"
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
    let rid = &options.request_id;
    let api = options.api;
    let mut ttft_secs: Option<f64> = None;

    info!("[req={rid}] Spawning subprocess model={} api={api}", options.model);

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
            error!("[req={rid}] Spawn failed: {msg}");
            let _ = tx.send(SubprocessEvent::Error(msg)).await;
            return;
        }
    };

    let pid = child.id().unwrap_or(0);
    info!("[req={rid}][pid={pid}] Subprocess started");

    let stdout = child.stdout.take().expect("stdout not captured");
    let stderr = child.stderr.take().expect("stderr not captured");

    let mut stdout_reader = BufReader::new(stdout).lines();
    let mut stderr_reader = BufReader::new(stderr).lines();
    let mut first_token = true;
    let mut chunk_count: u64 = 0;
    let inactivity_timeout = tokio::time::sleep(INACTIVITY_TIMEOUT);
    tokio::pin!(inactivity_timeout);
    let progress_interval = tokio::time::sleep(Duration::from_secs(30));
    tokio::pin!(progress_interval);

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
                                            let ttft = start.elapsed().as_secs_f64();
                                            ttft_secs = Some(ttft);
                                            info!("[req={rid}][pid={pid}] First token after {ttft:.2}s");
                                            first_token = false;
                                        }
                                    }
                                    if matches!(&event, SubprocessEvent::ContentDelta(_)) {
                                        chunk_count += 1;
                                    }
                                    if tx.send(event).await.is_err() {
                                        let elapsed = start.elapsed().as_secs_f64();
                                        let ttft_str = match ttft_secs {
                                            Some(t) => format!("{t:.2}s"),
                                            None => "-".to_string(),
                                        };
                                        warn!("[req={rid}][pid={pid}] Disconnected api={api} model={} ttft={ttft_str} total={elapsed:.2}s", options.model);
                                        let _ = child.kill().await;
                                        return;
                                    }
                                }
                            }
                            None => {
                                debug!("[req={rid}][pid={pid}] Ignoring non-JSON line: {line}");
                            }
                        }
                    }
                    Ok(None) => {
                        // stdout closed
                        break;
                    }
                    Err(e) => {
                        error!("[req={rid}][pid={pid}] Error reading stdout: {e}");
                        break;
                    }
                }
            }
            line = stderr_reader.next_line() => {
                match line {
                    Ok(Some(line)) => {
                        // Reset inactivity timer on stderr too
                        inactivity_timeout.as_mut().reset(tokio::time::Instant::now() + INACTIVITY_TIMEOUT);
                        debug!("[req={rid}][pid={pid}] stderr: {line}");
                    }
                    Ok(None) => {
                        // stderr closed
                    }
                    Err(e) => {
                        debug!("[req={rid}][pid={pid}] stderr read error: {e}");
                    }
                }
            }
            () = &mut progress_interval => {
                let elapsed = start.elapsed().as_secs_f64();
                info!("[req={rid}][pid={pid}] Still running {elapsed:.0}s chunks={chunk_count}");
                progress_interval.as_mut().reset(tokio::time::Instant::now() + Duration::from_secs(30));
            }
            () = &mut inactivity_timeout => {
                let elapsed = start.elapsed().as_secs_f64();
                let ttft_str = match ttft_secs {
                    Some(t) => format!("{t:.2}s"),
                    None => "-".to_string(),
                };
                warn!("[req={rid}][pid={pid}] Timeout api={api} model={} ttft={ttft_str} total={elapsed:.2}s (30m inactivity)", options.model);
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
            error!("[req={rid}][pid={pid}] Error waiting for subprocess: {e}");
            -1
        }
    };

    let elapsed = start.elapsed().as_secs_f64();
    let ttft_str = match ttft_secs {
        Some(t) => format!("{t:.2}s"),
        None => "-".to_string(),
    };
    info!(
        "[req={rid}][pid={pid}] Done api={api} model={} ttft={ttft_str} total={elapsed:.2}s exit={exit_code}",
        options.model
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── build_args ────────────────────────────────────────────

    #[test]
    fn build_args_basic() {
        let options = SubprocessOptions {
            request_id: "abc".to_string(),
            model: "opus".to_string(),
            session_id: None,
            cwd: "/tmp".to_string(),
            api: "anthropic",
        };
        let args = build_args("Hello world", &options);
        assert!(args.contains(&"--print".to_string()));
        assert!(args.contains(&"--output-format".to_string()));
        assert!(args.contains(&"stream-json".to_string()));
        assert!(args.contains(&"--model".to_string()));
        assert!(args.contains(&"opus".to_string()));
        assert!(args.contains(&"--permission-mode".to_string()));
        assert!(args.contains(&"bypassPermissions".to_string()));
        assert!(args.contains(&"Hello world".to_string()));
        assert!(!args.contains(&"--session-id".to_string()));
    }

    #[test]
    fn build_args_with_session_id() {
        let options = SubprocessOptions {
            request_id: "abc".to_string(),
            model: "sonnet".to_string(),
            session_id: Some("sess-123".to_string()),
            cwd: "/tmp".to_string(),
            api: "openai",
        };
        let args = build_args("test", &options);
        assert!(args.contains(&"--session-id".to_string()));
        assert!(args.contains(&"sess-123".to_string()));
    }

    // ── process_line ──────────────────────────────────────────

    #[test]
    fn process_line_system_message() {
        let line = r#"{"type":"system","subtype":"init"}"#;
        let events = process_line(line).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn process_line_assistant_with_model() {
        let line = r#"{"type":"assistant","message":{"model":"claude-opus-4-20250514","content":[]}}"#;
        let events = process_line(line).unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            SubprocessEvent::Model(m) => assert_eq!(m, "claude-opus-4-20250514"),
            other => panic!("Expected Model event, got {:?}", other),
        }
    }

    #[test]
    fn process_line_assistant_with_content() {
        let line = r#"{"type":"assistant","message":{"model":"claude-sonnet-4","content":[{"type":"text","text":"Hello"}]}}"#;
        let events = process_line(line).unwrap();
        assert_eq!(events.len(), 2);
        match &events[0] {
            SubprocessEvent::Model(m) => assert_eq!(m, "claude-sonnet-4"),
            other => panic!("Expected Model, got {:?}", other),
        }
        match &events[1] {
            SubprocessEvent::ContentDelta(t) => assert_eq!(t, "Hello"),
            other => panic!("Expected ContentDelta, got {:?}", other),
        }
    }

    #[test]
    fn process_line_assistant_empty_content_skipped() {
        let line = r#"{"type":"assistant","message":{"model":"opus","content":[{"type":"text","text":""}]}}"#;
        let events = process_line(line).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], SubprocessEvent::Model(_)));
    }

    #[test]
    fn process_line_result() {
        let line = r#"{"type":"result","result":"Done","exitCode":0,"duration_ms":1234,"duration_api_ms":1000,"num_turns":1,"modelUsage":{"claude-opus-4":{"input_tokens":50,"output_tokens":25}}}"#;
        let events = process_line(line).unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            SubprocessEvent::Result(r) => {
                assert_eq!(r.result, Some("Done".to_string()));
                assert_eq!(r.exit_code, Some(0));
                assert_eq!(r.duration_ms, Some(1234));
                let usage = r.model_usage.as_ref().unwrap();
                assert_eq!(usage["claude-opus-4"].input_tokens, Some(50));
                assert_eq!(usage["claude-opus-4"].output_tokens, Some(25));
            }
            other => panic!("Expected Result, got {:?}", other),
        }
    }

    #[test]
    fn process_line_content_block_delta() {
        let line = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"streaming text"}}"#;
        let events = process_line(line).unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            SubprocessEvent::ContentDelta(t) => assert_eq!(t, "streaming text"),
            other => panic!("Expected ContentDelta, got {:?}", other),
        }
    }

    #[test]
    fn process_line_content_block_delta_empty_text() {
        let line = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":""}}"#;
        let events = process_line(line).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn process_line_content_block_start() {
        let line = r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let events = process_line(line).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn process_line_content_block_stop() {
        let line = r#"{"type":"content_block_stop","index":0}"#;
        let events = process_line(line).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn process_line_message_start() {
        let line = r#"{"type":"message_start"}"#;
        let events = process_line(line).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn process_line_message_delta() {
        let line = r#"{"type":"message_delta"}"#;
        let events = process_line(line).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn process_line_message_stop() {
        let line = r#"{"type":"message_stop"}"#;
        let events = process_line(line).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn process_line_not_json() {
        assert!(process_line("not json at all").is_none());
        assert!(process_line("").is_none());
    }

    #[test]
    fn process_line_unknown_json() {
        assert!(process_line(r#"{"type":"unknown","data":123}"#).is_none());
    }
}
