use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{error, info};

const SESSION_TTL_MS: u64 = 24 * 60 * 60 * 1000; // 24 hours

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMapping {
    pub clawdbot_id: String,
    pub claude_session_id: String,
    pub created_at: u64,
    pub last_used_at: u64,
    pub model: String,
}

#[derive(Clone)]
pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<String, SessionMapping>>>,
    file_path: PathBuf,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

impl SessionManager {
    pub fn new() -> Self {
        let file_path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join(".claude-code-cli-sessions.json");

        let manager = Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            file_path,
        };

        // Fire-and-forget load
        let m = manager.clone();
        tokio::spawn(async move {
            m.load().await;
        });

        manager
    }

    async fn load(&self) {
        match tokio::fs::read_to_string(&self.file_path).await {
            Ok(data) => match serde_json::from_str::<HashMap<String, SessionMapping>>(&data) {
                Ok(sessions) => {
                    let mut lock = self.sessions.write().await;
                    *lock = sessions;
                    info!(
                        "Loaded {} sessions from {}",
                        lock.len(),
                        self.file_path.display()
                    );
                }
                Err(e) => {
                    error!("Failed to parse sessions file: {}", e);
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // No sessions file yet, that's fine
            }
            Err(e) => {
                error!("Failed to read sessions file: {}", e);
            }
        }
    }

    async fn save(&self) {
        let sessions = self.sessions.read().await;
        match serde_json::to_string_pretty(&*sessions) {
            Ok(data) => {
                if let Err(e) = tokio::fs::write(&self.file_path, data).await {
                    error!("Failed to write sessions file: {}", e);
                }
            }
            Err(e) => {
                error!("Failed to serialize sessions: {}", e);
            }
        }
    }

    #[allow(dead_code)]
    pub async fn get_or_create(&self, clawdbot_id: &str, model: &str) -> String {
        {
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(clawdbot_id) {
                session.last_used_at = now_ms();
                session.model = model.to_string();
                return session.claude_session_id.clone();
            }
        }

        let session_id = uuid::Uuid::new_v4().to_string();
        let mapping = SessionMapping {
            clawdbot_id: clawdbot_id.to_string(),
            claude_session_id: session_id.clone(),
            created_at: now_ms(),
            last_used_at: now_ms(),
            model: model.to_string(),
        };

        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(clawdbot_id.to_string(), mapping);
        }

        // Fire-and-forget save
        let m = self.clone();
        tokio::spawn(async move {
            m.save().await;
        });

        session_id
    }

    pub async fn cleanup_expired(&self) {
        let now = now_ms();
        let mut removed = 0;

        {
            let mut sessions = self.sessions.write().await;
            sessions.retain(|_, v| {
                let keep = (now - v.last_used_at) < SESSION_TTL_MS;
                if !keep {
                    removed += 1;
                }
                keep
            });
        }

        if removed > 0 {
            info!("Cleaned up {} expired sessions", removed);
            self.save().await;
        }
    }

    /// Spawn the hourly cleanup task
    pub fn spawn_cleanup_task(&self) {
        let manager = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600));
            loop {
                interval.tick().await;
                manager.cleanup_expired().await;
            }
        });
    }
}
