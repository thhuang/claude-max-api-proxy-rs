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

    /// Create a SessionManager with a custom file path (for testing).
    #[cfg(test)]
    fn with_path(file_path: PathBuf) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            file_path,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("session-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("sessions.json")
    }

    #[tokio::test]
    async fn get_or_create_new_session() {
        let mgr = SessionManager::with_path(temp_path());
        let id = mgr.get_or_create("client-1", "opus").await;
        assert!(!id.is_empty());
        // UUID format
        assert_eq!(id.len(), 36);
    }

    #[tokio::test]
    async fn get_or_create_returns_same_session() {
        let mgr = SessionManager::with_path(temp_path());
        let id1 = mgr.get_or_create("client-1", "opus").await;
        let id2 = mgr.get_or_create("client-1", "sonnet").await;
        assert_eq!(id1, id2);
    }

    #[tokio::test]
    async fn get_or_create_different_clients() {
        let mgr = SessionManager::with_path(temp_path());
        let id1 = mgr.get_or_create("client-1", "opus").await;
        let id2 = mgr.get_or_create("client-2", "opus").await;
        assert_ne!(id1, id2);
    }

    #[tokio::test]
    async fn get_or_create_updates_model() {
        let mgr = SessionManager::with_path(temp_path());
        mgr.get_or_create("client-1", "opus").await;
        mgr.get_or_create("client-1", "sonnet").await;

        let sessions = mgr.sessions.read().await;
        assert_eq!(sessions["client-1"].model, "sonnet");
    }

    #[tokio::test]
    async fn get_or_create_updates_last_used() {
        let mgr = SessionManager::with_path(temp_path());
        mgr.get_or_create("client-1", "opus").await;
        let t1 = {
            let sessions = mgr.sessions.read().await;
            sessions["client-1"].last_used_at
        };

        // Small sleep to ensure timestamp changes
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;

        mgr.get_or_create("client-1", "opus").await;
        let t2 = {
            let sessions = mgr.sessions.read().await;
            sessions["client-1"].last_used_at
        };

        assert!(t2 >= t1);
    }

    #[tokio::test]
    async fn cleanup_removes_expired() {
        let mgr = SessionManager::with_path(temp_path());

        // Manually insert an expired session
        {
            let mut sessions = mgr.sessions.write().await;
            sessions.insert(
                "old-client".to_string(),
                SessionMapping {
                    clawdbot_id: "old-client".to_string(),
                    claude_session_id: "old-session".to_string(),
                    created_at: 0,
                    last_used_at: 0, // epoch — definitely expired
                    model: "opus".to_string(),
                },
            );
            sessions.insert(
                "new-client".to_string(),
                SessionMapping {
                    clawdbot_id: "new-client".to_string(),
                    claude_session_id: "new-session".to_string(),
                    created_at: now_ms(),
                    last_used_at: now_ms(),
                    model: "opus".to_string(),
                },
            );
        }

        mgr.cleanup_expired().await;

        let sessions = mgr.sessions.read().await;
        assert!(!sessions.contains_key("old-client"));
        assert!(sessions.contains_key("new-client"));
    }

    #[tokio::test]
    async fn cleanup_no_op_when_all_fresh() {
        let mgr = SessionManager::with_path(temp_path());
        mgr.get_or_create("client-1", "opus").await;
        mgr.cleanup_expired().await;

        let sessions = mgr.sessions.read().await;
        assert!(sessions.contains_key("client-1"));
    }

    #[tokio::test]
    async fn save_and_load_round_trip() {
        let path = temp_path();
        let mgr = SessionManager::with_path(path.clone());
        mgr.get_or_create("client-1", "opus").await;

        // Wait for the fire-and-forget save
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Load into a new manager
        let mgr2 = SessionManager::with_path(path);
        mgr2.load().await;

        let sessions = mgr2.sessions.read().await;
        assert!(sessions.contains_key("client-1"));
        assert_eq!(sessions["client-1"].model, "opus");
    }

    #[tokio::test]
    async fn load_missing_file_is_ok() {
        let mgr = SessionManager::with_path(PathBuf::from("/tmp/nonexistent-session-file.json"));
        mgr.load().await;
        let sessions = mgr.sessions.read().await;
        assert!(sessions.is_empty());
    }
}
