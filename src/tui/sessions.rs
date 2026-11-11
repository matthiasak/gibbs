//! Persistent multi-turn TUI chat sessions, one JSON file per session under
//! `~/.cache/gibbs/sessions/<id>.json` (next to `settings.json`, so tests use a temp dir).
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedTurn {
    pub role: String,
    pub text: String,
    #[serde(default)]
    pub media_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedSession {
    pub id: String,
    pub title: String,
    pub updated_unix: u64,
    pub model_id: String,
    pub system_prompt: String,
    pub turns: Vec<SavedTurn>,
}

fn path(id: &str) -> Result<PathBuf, String> {
    // Ids come from `/session load <id>` too: never let one escape the directory.
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err(format!("invalid session id '{id}'"));
    }
    Ok(super::settings::settings_path()
        .with_file_name("sessions")
        .join(format!("{id}.json")))
}

pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// A fresh id: the current time in nanoseconds, hex.
pub fn new_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("{nanos:x}")
}

/// The first user message's first line, cut to 48 characters.
pub fn title_of(turns: &[SavedTurn]) -> String {
    let first = turns
        .iter()
        .find(|t| t.role == "user")
        .and_then(|t| t.text.lines().find(|l| !l.trim().is_empty()))
        .unwrap_or("(untitled)")
        .trim();
    match first.char_indices().nth(48) {
        Some((i, _)) => format!("{}…", &first[..i]),
        None => first.to_string(),
    }
}

pub fn save(s: &SavedSession) -> Result<(), String> {
    let p = path(&s.id)?;
    std::fs::create_dir_all(p.parent().expect("sessions dir")).map_err(|e| e.to_string())?;
    let json = serde_json::to_string_pretty(s).map_err(|e| e.to_string())?;
    std::fs::write(&p, json).map_err(|e| format!("{}: {e}", p.display()))
}

pub fn load(id: &str) -> Result<SavedSession, String> {
    let p = path(id)?;
    let text = std::fs::read_to_string(&p).map_err(|_| format!("no saved session '{id}'"))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))
}

/// Every readable session, newest first.
pub fn list() -> Vec<SavedSession> {
    let dir = super::settings::settings_path().with_file_name("sessions");
    let mut all: Vec<SavedSession> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| serde_json::from_str(&std::fs::read_to_string(e.path()).ok()?).ok())
        .collect();
    all.sort_by(|a, b| b.updated_unix.cmp(&a.updated_unix).then(b.id.cmp(&a.id)));
    all
}

pub fn delete(id: &str) -> Result<(), String> {
    std::fs::remove_file(path(id)?).map_err(|_| format!("no saved session '{id}'"))
}
