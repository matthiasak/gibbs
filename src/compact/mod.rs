//! Four-tier automatic session compaction engine (`src/compact/`).
//!
//! Keeps long-horizon agent and chat sessions inside the active model's context window while
//! preserving [`RadixPrefixCache`] prefix stability (every tier is deterministic):
//! - **Tier 1 (`elide_thinking_traces`)**: Strips thinking blocks (the defaults plus the model's
//!   `runtime.thinking_delimiters`) from older assistant turns.
//! - **Tier 2 (`shake_messages`)**: Spills oversized blocks of older turns to content-addressed
//!   artifacts (`~/.cache/gibbs/artifacts/<hex>.txt`, served by `GET /v1/artifacts/{hex}`) behind a
//!   head+tail digest, and replaces repeated blocks with a reference to their artifact.
//! - **Tier 3 (`Summarizer`)**: Folds the oldest turns into a greedy, cached model summary; on
//!   vision models in Aggressive mode the folded transcript also rides along as `SnapCompact`
//!   glyph pages (HD44780 5x7 font, `render_snapcompact_pages`).
//! - **Tier 4 ([`snapkv::evict_kv_state`])**: Physical `SnapKV` / `H2O` heavy-hitter + attention-sink
//!   KV row compaction inside [`StateStore`].

pub mod snapkv;

use crate::model::hf_loader::ChatMessage;
use crate::model::image::ImageBuffer;
use crate::model::native::MediaInput;
use crate::model::tokenizer::Tokenizer;
use crate::scheduler::{finish_outcome, ContinuousBatchScheduler, ExecutableModel, SamplingParams};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

/// Session compaction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum CompactionMode {
    /// Automatically trigger compaction when prompt tokens exceed `trigger_ratio * context_len`.
    #[default]
    Auto,
    /// Disable automatic session compaction.
    Off,
    /// Run Tier 1 + Tier 2 only (lossless final-answer preservation).
    Light,
    /// Force all tiers (Tier 1 + Tier 2 + Tier 3 summary/SnapCompact).
    Aggressive,
}

impl CompactionMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::Light => "light",
            Self::Aggressive => "aggressive",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Auto => Self::Light,
            Self::Light => Self::Aggressive,
            Self::Aggressive => Self::Off,
            Self::Off => Self::Auto,
        }
    }
}

impl FromStr for CompactionMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "on" | "true" | "1" => Ok(Self::Auto),
            "off" | "none" | "false" | "0" => Ok(Self::Off),
            "light" | "think" | "tier1" => Ok(Self::Light),
            "aggressive" | "full" | "snap" | "snapcompact" => Ok(Self::Aggressive),
            other => Err(format!(
                "unknown compaction mode '{other}'; valid: auto, light, aggressive, off"
            )),
        }
    }
}

/// Configuration for automatic and manual session compaction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CompactionConfig {
    pub mode: CompactionMode,
    /// Context fraction (`0.30..=0.95`, default `0.75`) above which automatic compaction triggers.
    pub trigger_ratio: f32,
    /// Target context fraction (`0.20..=0.85`, default `0.50`) after compaction.
    pub target_ratio: f32,
    /// Recent user/assistant turns (`1..=16`, default `2`) kept untouched.
    pub keep_recent_turns: usize,
    /// Render the folded transcript as `SnapCompact` glyph pages for vision models (Aggressive).
    pub enable_snapcompact_vision: bool,
    /// Tier 3 summarizer: `None` is the active model, else a model id loaded beside it.
    pub summarizer: Option<String>,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            mode: CompactionMode::Auto,
            trigger_ratio: 0.75,
            target_ratio: 0.50,
            keep_recent_turns: 2,
            enable_snapcompact_vision: true,
            summarizer: None,
        }
    }
}

impl CompactionConfig {
    pub fn sanitized(mut self) -> Self {
        self.trigger_ratio = self.trigger_ratio.clamp(0.25, 0.98);
        self.target_ratio = self
            .target_ratio
            .clamp(0.15, (self.trigger_ratio - 0.05).max(0.20));
        self.keep_recent_turns = self.keep_recent_turns.clamp(1, 16);
        self
    }
}

/// [`CompactionConfig::summarizer`] for a `--compact-summarizer` / `summarizer=` value: `active`
/// (or empty) is the active model.
pub fn summarizer_id(s: &str) -> Option<String> {
    Some(s.trim())
        .filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("active"))
        .map(String::from)
}

/// Settings a request gives over the saved ones: a chat request's `compact` object, the
/// `/v1/compact` body and the TUI's `/compact key=value` arguments.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CompactionOverride {
    pub mode: Option<CompactionMode>,
    /// [`CompactionConfig::trigger_ratio`].
    pub trigger: Option<f32>,
    /// [`CompactionConfig::target_ratio`].
    pub target: Option<f32>,
    /// [`CompactionConfig::keep_recent_turns`].
    #[serde(alias = "keep_recent_turns")]
    pub keep_turns: Option<usize>,
    /// `active` or a model id ([`CompactionConfig::summarizer`]).
    pub summarizer: Option<String>,
}

impl CompactionOverride {
    pub fn apply(&self, cfg: &mut CompactionConfig) {
        if let Some(m) = self.mode {
            cfg.mode = m;
        }
        if let Some(t) = self.trigger {
            cfg.trigger_ratio = t;
        }
        if let Some(t) = self.target {
            cfg.target_ratio = t;
        }
        if let Some(k) = self.keep_turns {
            cfg.keep_recent_turns = k;
        }
        if let Some(s) = &self.summarizer {
            cfg.summarizer = summarizer_id(s);
        }
    }

    /// A summarizer named over HTTP must already be in the local model cache: a request never
    /// starts a download.
    pub fn check_local(&self) -> Result<(), String> {
        match self.summarizer.as_deref().and_then(summarizer_id) {
            Some(id)
                if crate::model::hub::HuggingFaceHub::new()
                    .locate_model(&id)
                    .weights_path
                    .is_none() =>
            {
                Err(format!(
                    "compact.summarizer '{id}' is not in the local model cache"
                ))
            }
            _ => Ok(()),
        }
    }
}

/// Telemetry report from a compaction pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompactionReport {
    pub before_tokens: usize,
    pub after_tokens: usize,
    pub tier1_chars_elided: usize,
    pub tier2_chars_shaken: usize,
    pub tier3_turns_summarized: usize,
    /// Tier 2 artifacts (`GET /v1/artifacts/{hex}`) the compacted messages reference.
    pub artifacts: Vec<String>,
    pub snapcompact_pages: usize,
    pub snapcompact_png_bytes: usize,
}

impl CompactionReport {
    pub fn unchanged(tokens: usize) -> Self {
        Self {
            before_tokens: tokens,
            after_tokens: tokens,
            ..Self::default()
        }
    }

    pub fn saved_tokens(&self) -> usize {
        self.before_tokens.saturating_sub(self.after_tokens)
    }
}

/// Default thinking block delimiters (`<think>...</think>`, `<|thought|>...<|/thought|>`).
pub fn default_thinking_delimiters() -> Vec<(String, String)> {
    vec![
        ("<think>".to_string(), "</think>".to_string()),
        ("<|thought|>".to_string(), "<|/thought|>".to_string()),
        ("<thought>".to_string(), "</thought>".to_string()),
    ]
}

/// **Tier 1**: Elides reasoning/thinking traces (`<think>...</think>`) from assistant turns older
/// than the last `keep_recent_turns` turns. Returns the total number of characters removed.
pub fn elide_thinking_traces(
    messages: &mut [ChatMessage],
    keep_recent_turns: usize,
    extra_delimiters: &[(String, String)],
) -> usize {
    let mut delims = default_thinking_delimiters();
    for pair in extra_delimiters {
        if !delims.contains(pair) {
            delims.push(pair.clone());
        }
    }
    let cutoff = messages.len().saturating_sub(keep_recent_turns * 2);
    let mut elided_chars = 0usize;

    for msg in &mut messages[..cutoff] {
        if msg.role != "assistant" {
            continue;
        }
        let Some(text) = msg.content.as_str() else {
            continue;
        };
        let mut s = text.to_string();
        let orig_len = s.len();
        for (open, close) in &delims {
            while let Some(start) = s.find(open.as_str()) {
                if let Some(end_rel) = s[start + open.len()..].find(close.as_str()) {
                    let mut end = start + open.len() + end_rel + close.len();
                    while end < s.len() && matches!(s.as_bytes()[end], b'\n' | b'\r' | b' ') {
                        end += 1;
                    }
                    s.replace_range(start..end, "");
                } else {
                    break;
                }
            }
        }
        if s.len() < orig_len {
            elided_chars += orig_len - s.len();
            msg.content = Value::String(s.trim().to_string());
        }
    }
    elided_chars
}

/// FNV-1a 64: artifact and summary names (stable across runs and Rust releases).
fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

/// `~/.cache/gibbs/artifacts`: Tier 2 spills and Tier 3 summaries, named by content hash.
#[cfg(not(test))]
fn artifacts_dir() -> Result<PathBuf, String> {
    Ok(crate::hardware::default_cache_dir()?.join("artifacts"))
}

/// Tests share one process-wide directory: names are content hashes, so tests never clash.
#[cfg(test)]
fn artifacts_dir() -> Result<PathBuf, String> {
    static DIR: std::sync::LazyLock<tempfile::TempDir> =
        std::sync::LazyLock::new(|| tempfile::tempdir().unwrap());
    Ok(DIR.path().to_path_buf())
}

/// Writes `text` to `path` unless it exists; through a temp file, so readers never see a part.
fn write_once(path: &std::path::Path, text: &str) -> Result<(), String> {
    if path.exists() {
        return Ok(());
    }
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = path.parent().ok_or("artifact path has no directory")?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = path.with_extension(format!("{}-{n}.tmp", std::process::id()));
    std::fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

/// Stores `text` as artifact `<fnv64-hex>.txt` (written once) and returns the hex name.
pub fn store_artifact(text: &str) -> Result<String, String> {
    let hex = format!("{:016x}", fnv64(text.as_bytes()));
    write_once(&artifacts_dir()?.join(format!("{hex}.txt")), text)?;
    Ok(hex)
}

/// The artifact named `hex` (16 hex digits); `None` for any other name or a missing file.
pub fn read_artifact(hex: &str) -> Option<String> {
    if hex.len() != 16 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let name = format!("{}.txt", hex.to_ascii_lowercase());
    std::fs::read_to_string(artifacts_dir().ok()?.join(name)).ok()
}

/// Byte spans of `text`'s blocks: runs of lines separated by blank lines, where a fenced code
/// block (```) stays one block. Spans exclude the final line break.
fn block_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let (mut start, mut end, mut fence, mut pos) = (None, 0, false, 0);
    for line in text.split_inclusive('\n') {
        let t = line.trim();
        if t.starts_with("```") {
            fence = !fence;
        }
        if t.is_empty() && !fence {
            if let Some(s) = start.take() {
                spans.push((s, end));
            }
        } else {
            start.get_or_insert(pos);
            end = pos + line.trim_end_matches(['\n', '\r']).len();
        }
        pos += line.len();
    }
    if let Some(s) = start {
        spans.push((s, end));
    }
    spans
}

/// The first `n` and last `m` characters of `s`.
fn head_tail(s: &str, n: usize, m: usize) -> (&str, &str) {
    let head = s.char_indices().nth(n).map_or(s, |(i, _)| &s[..i]);
    let count = s.chars().count();
    let tail = s
        .char_indices()
        .nth(count.saturating_sub(m))
        .map_or("", |(i, _)| &s[i..]);
    (head, tail)
}

/// **Tier 2**: In string messages before the last `keep_recent_turns` turns (system messages
/// excepted), a block over 16 lines or 800 characters is stored whole as an artifact and replaced
/// by its head and tail around `[… {n} lines compacted; artifact {hex} …]`; a block of at least
/// 200 characters already seen in an earlier message becomes `[duplicate of artifact {hex}]`.
/// Returns the characters removed and the artifacts referenced. A block whose artifact cannot be
/// stored stays verbatim.
pub fn shake_messages(
    messages: &mut [ChatMessage],
    keep_recent_turns: usize,
) -> (usize, Vec<String>) {
    let cutoff = messages.len().saturating_sub(keep_recent_turns * 2);
    let (mut shaken, mut artifacts) = (0usize, Vec::new());
    let mut seen: HashSet<u64> = HashSet::new();
    for (i, msg) in messages.iter_mut().enumerate() {
        let Some(text) = msg.content.as_str() else {
            continue;
        };
        let spans = block_spans(text);
        let hashes: Vec<u64> = spans
            .iter()
            .map(|&(s, e)| fnv64(text[s..e].as_bytes()))
            .collect();
        if i < cutoff && msg.role != "system" {
            let (mut out, mut last) = (String::with_capacity(text.len()), 0);
            for (&(s, e), h) in spans.iter().zip(&hashes) {
                let block = &text[s..e];
                let lines = block.lines().count();
                let replacement = if block.len() >= 200 && seen.contains(h) {
                    store_artifact(block)
                        .ok()
                        .map(|hex| (format!("[duplicate of artifact {hex}]"), hex))
                } else if lines > 16 || block.len() > 800 {
                    store_artifact(block).ok().map(|hex| {
                        let digest = if lines > 16 {
                            let l: Vec<&str> = block.lines().collect();
                            format!(
                                "{}\n[… {} lines compacted; artifact {hex} …]\n{}",
                                l[..5].join("\n"),
                                lines - 9,
                                l[lines - 4..].join("\n")
                            )
                        } else {
                            let (head, tail) = head_tail(block, 240, 160);
                            let n = block.chars().count().saturating_sub(400);
                            format!("{head}[… {n} chars compacted; artifact {hex} …]{tail}")
                        };
                        (digest, hex)
                    })
                } else {
                    None
                };
                out.push_str(&text[last..s]);
                match replacement {
                    Some((r, hex)) if r.len() < block.len() => {
                        out.push_str(&r);
                        artifacts.push(hex);
                    }
                    _ => out.push_str(block),
                }
                last = e;
            }
            out.push_str(&text[last..]);
            if out.len() < text.len() {
                shaken += text.len() - out.len();
                msg.content = Value::String(out);
            }
        }
        seen.extend(hashes);
    }
    artifacts.sort_unstable();
    artifacts.dedup();
    (shaken, artifacts)
}

/// SnapCompact page geometry: 64 columns x 48 rows of 12x16 cells (768x768 px); glyphs are
/// drawn at 2x, one pixel in from the cell's top-left corner.
pub const PAGE_COLS: usize = 64;
pub const PAGE_ROWS: usize = 48;
const CELL_W: usize = 12;
const CELL_H: usize = 16;
const PAPER: [u8; 3] = [248, 245, 236];
const INK: [u8; 3] = [20, 20, 24];

/// The HD44780 (A00) character ROM's 5x7 glyphs for ASCII `0x20..=0x7E`, as rasterized from the
/// Hitachi datasheet by wokwi-elements (`lcd1602-font-a00.ts`, MIT). One byte per row, top to
/// bottom; bit 0 is the leftmost pixel. A00 has `¥` and `→` at `\` and `~`; those two use the
/// ASCII shapes instead.
const GLYPHS_5X7: [[u8; 7]; 95] = [
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // ' '
    [0x04, 0x04, 0x04, 0x04, 0x00, 0x00, 0x04], // '!'
    [0x0a, 0x0a, 0x0a, 0x00, 0x00, 0x00, 0x00], // '"'
    [0x0a, 0x0a, 0x1f, 0x0a, 0x1f, 0x0a, 0x0a], // '#'
    [0x04, 0x1e, 0x05, 0x0e, 0x14, 0x0f, 0x04], // '$'
    [0x03, 0x13, 0x08, 0x04, 0x02, 0x19, 0x18], // '%'
    [0x06, 0x09, 0x05, 0x02, 0x15, 0x09, 0x16], // '&'
    [0x06, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00], // "'"
    [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08], // '('
    [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02], // ')'
    [0x00, 0x04, 0x15, 0x0e, 0x15, 0x04, 0x00], // '*'
    [0x00, 0x04, 0x04, 0x1f, 0x04, 0x04, 0x00], // '+'
    [0x00, 0x00, 0x00, 0x00, 0x06, 0x04, 0x02], // ','
    [0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00], // '-'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x06], // '.'
    [0x00, 0x10, 0x08, 0x04, 0x02, 0x01, 0x00], // '/'
    [0x0e, 0x11, 0x19, 0x15, 0x13, 0x11, 0x0e], // '0'
    [0x04, 0x06, 0x04, 0x04, 0x04, 0x04, 0x0e], // '1'
    [0x0e, 0x11, 0x10, 0x08, 0x04, 0x02, 0x1f], // '2'
    [0x1f, 0x08, 0x04, 0x08, 0x10, 0x11, 0x0e], // '3'
    [0x08, 0x0c, 0x0a, 0x09, 0x1f, 0x08, 0x08], // '4'
    [0x1f, 0x01, 0x0f, 0x10, 0x10, 0x11, 0x0e], // '5'
    [0x0c, 0x02, 0x01, 0x0f, 0x11, 0x11, 0x0e], // '6'
    [0x1f, 0x11, 0x10, 0x08, 0x04, 0x04, 0x04], // '7'
    [0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e], // '8'
    [0x0e, 0x11, 0x11, 0x1e, 0x10, 0x08, 0x06], // '9'
    [0x00, 0x06, 0x06, 0x00, 0x06, 0x06, 0x00], // ':'
    [0x00, 0x06, 0x06, 0x00, 0x06, 0x04, 0x02], // ';'
    [0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08], // '<'
    [0x00, 0x00, 0x1f, 0x00, 0x1f, 0x00, 0x00], // '='
    [0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02], // '>'
    [0x0e, 0x11, 0x10, 0x08, 0x04, 0x00, 0x04], // '?'
    [0x0e, 0x11, 0x10, 0x16, 0x15, 0x15, 0x0e], // '@'
    [0x0e, 0x11, 0x11, 0x11, 0x1f, 0x11, 0x11], // 'A'
    [0x0f, 0x11, 0x11, 0x0f, 0x11, 0x11, 0x0f], // 'B'
    [0x0e, 0x11, 0x01, 0x01, 0x01, 0x11, 0x0e], // 'C'
    [0x07, 0x09, 0x11, 0x11, 0x11, 0x09, 0x07], // 'D'
    [0x1f, 0x01, 0x01, 0x0f, 0x01, 0x01, 0x1f], // 'E'
    [0x1f, 0x01, 0x01, 0x0f, 0x01, 0x01, 0x01], // 'F'
    [0x0e, 0x11, 0x01, 0x1d, 0x11, 0x11, 0x1e], // 'G'
    [0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11], // 'H'
    [0x0e, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e], // 'I'
    [0x1c, 0x08, 0x08, 0x08, 0x08, 0x09, 0x06], // 'J'
    [0x11, 0x09, 0x05, 0x03, 0x05, 0x09, 0x11], // 'K'
    [0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x1f], // 'L'
    [0x11, 0x1b, 0x15, 0x15, 0x11, 0x11, 0x11], // 'M'
    [0x11, 0x11, 0x13, 0x15, 0x19, 0x11, 0x11], // 'N'
    [0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e], // 'O'
    [0x0f, 0x11, 0x11, 0x0f, 0x01, 0x01, 0x01], // 'P'
    [0x0e, 0x11, 0x11, 0x11, 0x15, 0x09, 0x16], // 'Q'
    [0x0f, 0x11, 0x11, 0x0f, 0x05, 0x09, 0x11], // 'R'
    [0x1e, 0x01, 0x01, 0x0e, 0x10, 0x10, 0x0f], // 'S'
    [0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04], // 'T'
    [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e], // 'U'
    [0x11, 0x11, 0x11, 0x11, 0x11, 0x0a, 0x04], // 'V'
    [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0a], // 'W'
    [0x11, 0x11, 0x0a, 0x04, 0x0a, 0x11, 0x11], // 'X'
    [0x11, 0x11, 0x11, 0x0a, 0x04, 0x04, 0x04], // 'Y'
    [0x1f, 0x10, 0x08, 0x04, 0x02, 0x01, 0x1f], // 'Z'
    [0x07, 0x01, 0x01, 0x01, 0x01, 0x01, 0x07], // '['
    [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x00], // '\\'
    [0x0e, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0e], // ']'
    [0x04, 0x0a, 0x11, 0x00, 0x00, 0x00, 0x00], // '^'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f], // '_'
    [0x02, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00], // '`'
    [0x00, 0x00, 0x0e, 0x10, 0x1e, 0x11, 0x1e], // 'a'
    [0x01, 0x01, 0x0d, 0x13, 0x11, 0x11, 0x0f], // 'b'
    [0x00, 0x00, 0x0e, 0x01, 0x01, 0x11, 0x0e], // 'c'
    [0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x1e], // 'd'
    [0x00, 0x00, 0x0e, 0x11, 0x1f, 0x01, 0x0e], // 'e'
    [0x0c, 0x12, 0x02, 0x07, 0x02, 0x02, 0x02], // 'f'
    [0x00, 0x1e, 0x11, 0x11, 0x1e, 0x10, 0x0e], // 'g'
    [0x01, 0x01, 0x0d, 0x13, 0x11, 0x11, 0x11], // 'h'
    [0x04, 0x00, 0x06, 0x04, 0x04, 0x04, 0x0e], // 'i'
    [0x08, 0x00, 0x0c, 0x08, 0x08, 0x09, 0x06], // 'j'
    [0x01, 0x01, 0x09, 0x05, 0x03, 0x05, 0x09], // 'k'
    [0x06, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e], // 'l'
    [0x00, 0x00, 0x0b, 0x15, 0x15, 0x11, 0x11], // 'm'
    [0x00, 0x00, 0x0d, 0x13, 0x11, 0x11, 0x11], // 'n'
    [0x00, 0x00, 0x0e, 0x11, 0x11, 0x11, 0x0e], // 'o'
    [0x00, 0x00, 0x0f, 0x11, 0x0f, 0x01, 0x01], // 'p'
    [0x00, 0x00, 0x16, 0x19, 0x1e, 0x10, 0x10], // 'q'
    [0x00, 0x00, 0x0d, 0x13, 0x01, 0x01, 0x01], // 'r'
    [0x00, 0x00, 0x0e, 0x01, 0x0e, 0x10, 0x0f], // 's'
    [0x02, 0x02, 0x07, 0x02, 0x02, 0x12, 0x0c], // 't'
    [0x00, 0x00, 0x11, 0x11, 0x11, 0x19, 0x16], // 'u'
    [0x00, 0x00, 0x11, 0x11, 0x11, 0x0a, 0x04], // 'v'
    [0x00, 0x00, 0x11, 0x15, 0x15, 0x15, 0x0a], // 'w'
    [0x00, 0x00, 0x11, 0x0a, 0x04, 0x0a, 0x11], // 'x'
    [0x00, 0x00, 0x11, 0x11, 0x1e, 0x10, 0x0e], // 'y'
    [0x00, 0x00, 0x1f, 0x08, 0x04, 0x02, 0x1f], // 'z'
    [0x08, 0x04, 0x04, 0x02, 0x04, 0x04, 0x08], // '{'
    [0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04], // '|'
    [0x02, 0x04, 0x04, 0x08, 0x04, 0x04, 0x02], // '}'
    [0x00, 0x00, 0x16, 0x09, 0x00, 0x00, 0x00], // '~'
];

/// The glyph for `c`; characters outside `0x20..=0x7E` draw as `?`.
fn glyph_5x7(c: char) -> &'static [u8; 7] {
    let i = match c {
        ' '..='~' => c as usize - 0x20,
        _ => '?' as usize - 0x20,
    };
    &GLYPHS_5X7[i]
}

/// `text` wrapped to [`PAGE_COLS`] columns (tabs become spaces), one entry per page row.
fn page_rows(text: &str) -> Vec<Vec<char>> {
    let mut rows = Vec::new();
    for line in text.lines() {
        let chars: Vec<char> = line
            .chars()
            .map(|c| if c == '\t' { ' ' } else { c })
            .collect();
        if chars.is_empty() {
            rows.push(Vec::new());
        }
        rows.extend(chars.chunks(PAGE_COLS).map(<[char]>::to_vec));
    }
    rows
}

/// **SnapCompact**: `text` drawn in the HD44780 5x7 font at 2x, [`PAGE_COLS`] x [`PAGE_ROWS`]
/// characters per 768x768 page, as many pages as the text needs.
pub fn render_snapcompact_pages(text: &str) -> Vec<ImageBuffer> {
    let (w, h) = (PAGE_COLS * CELL_W, PAGE_ROWS * CELL_H);
    page_rows(text)
        .chunks(PAGE_ROWS)
        .map(|rows| {
            let mut img = ImageBuffer::new(w, h);
            for px in img.data.chunks_exact_mut(3) {
                px.copy_from_slice(&PAPER);
            }
            for (r, row) in rows.iter().enumerate() {
                for (c, &ch) in row.iter().enumerate() {
                    for (gy, bits) in glyph_5x7(ch).iter().enumerate() {
                        for gx in (0..5).filter(|gx| bits >> gx & 1 != 0) {
                            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                                let x = c * CELL_W + 1 + 2 * gx + dx;
                                let y = r * CELL_H + 1 + 2 * gy + dy;
                                let i = (y * w + x) * 3;
                                img.data[i..i + 3].copy_from_slice(&INK);
                            }
                        }
                    }
                }
            }
            img
        })
        .collect()
}

/// Generates Tier 3 summaries. Implementations decode greedily, so the same transcript always
/// gives the same summary.
pub trait Summarizer {
    fn summarize(&self, transcript: &str, max_tokens: usize) -> Result<String, String>;
}

/// The fixed Tier 3 instruction; the transcript follows in a user turn.
const SUMMARY_PROMPT: &str = "Summarize the conversation below so it can be continued without it. Keep user goals, decisions, constraints, facts, file paths, code identifiers, numbers, open questions and the current state. Use at most {max_tokens} tokens.";

/// Tokens reserved beside the transcript and the summary (template, instruction).
const SUMMARY_OVERHEAD: usize = 256;

/// The summary of `transcript`, cached as artifact `summary-<fnv64 of transcript>.txt` so every
/// later compaction of the same turns re-renders a byte-identical prefix.
fn cached_summary(
    s: &dyn Summarizer,
    transcript: &str,
    max_tokens: usize,
    tok: &Tokenizer,
    ctx: usize,
) -> Result<String, String> {
    let path = artifacts_dir()?.join(format!(
        "summary-{:016x}.txt",
        fnv64(transcript.as_bytes())
    ));
    if let Ok(text) = std::fs::read_to_string(&path) {
        return Ok(text);
    }
    let limit = ctx.saturating_sub(max_tokens + SUMMARY_OVERHEAD);
    let summary = summarize_fitting(s, transcript, max_tokens, tok, limit)?;
    write_once(&path, &summary)?;
    Ok(summary)
}

/// Summarizes `text` in one call when it fits in `limit` tokens; otherwise summarizes
/// order-preserving `limit`-token chunks and then their joined summaries (map-reduce).
fn summarize_fitting(
    s: &dyn Summarizer,
    text: &str,
    max_tokens: usize,
    tok: &Tokenizer,
    limit: usize,
) -> Result<String, String> {
    let ids = tok.encode(text);
    if ids.len() <= limit {
        return s.summarize(text, max_tokens);
    }
    if limit < 64 {
        return Err(format!(
            "the context leaves {limit} tokens for the transcript; too few to summarize"
        ));
    }
    let parts = ids
        .chunks(limit)
        .map(|c| s.summarize(&tok.decode(c), max_tokens))
        .collect::<Result<Vec<_>, _>>()?;
    let joined = parts.join("\n");
    if tok.encode(&joined).len() >= ids.len() {
        return Err("chunk summaries are no shorter than the transcript".into());
    }
    summarize_fitting(s, &joined, max_tokens, tok, limit)
}

/// Media parts (`{"type": "image" | "audio"}`) of `m`, each holding one request media item.
fn media_parts(m: &ChatMessage) -> usize {
    m.content
        .as_array()
        .map_or(0, |p| p.iter().filter(|p| p["type"] != "text").count())
}

fn estimate_tokens(messages: &[ChatMessage], tok: &Tokenizer) -> usize {
    let rendered = tok.apply_chat_template_messages(messages, None);
    tok.encode(&rendered).len()
}

/// What compaction needs from the active model.
#[derive(Debug, Clone, Default)]
pub struct ModelTraits {
    /// Thinking delimiters beyond [`default_thinking_delimiters`] (`runtime.thinking_delimiters`).
    pub delimiters: Vec<(String, String)>,
    pub has_vision: bool,
}

/// A compacted session.
#[derive(Debug, Clone)]
pub struct Compacted {
    pub messages: Vec<ChatMessage>,
    /// SnapCompact pages (PNG): the first media items of the compacted request.
    pub media: Vec<MediaInput>,
    /// Leading request media items that belonged to folded turns and are gone.
    pub folded_media: usize,
    pub report: CompactionReport,
}

/// Runs the tiers on `messages` once its rendered prompt exceeds `cfg.trigger_ratio x ctx`
/// (`force` skips that check): Tier 1, then Tier 2, then (Auto above `cfg.target_ratio x ctx`, or
/// Aggressive) Tier 3. Light stops after Tier 2; Off and sessions no longer than the kept turns
/// are returned unchanged. A summarizer error aborts the whole compaction.
pub fn compact_chat_session(
    messages: &[ChatMessage],
    tok: &Tokenizer,
    ctx: usize,
    cfg: &CompactionConfig,
    model: &ModelTraits,
    summarizer: &dyn Summarizer,
    force: bool,
) -> Result<Compacted, String> {
    let cfg = cfg.clone().sanitized();
    let before_tokens = estimate_tokens(messages, tok);
    let trigger_tokens = ((ctx as f32) * cfg.trigger_ratio).max(16.0) as usize;
    let target_tokens = ((ctx as f32) * cfg.target_ratio).max(8.0) as usize;
    let mut done = Compacted {
        messages: messages.to_vec(),
        media: Vec::new(),
        folded_media: 0,
        report: CompactionReport::unchanged(before_tokens),
    };
    if cfg.mode == CompactionMode::Off
        || messages.len() <= cfg.keep_recent_turns * 2
        || (!force && before_tokens <= trigger_tokens)
    {
        return Ok(done);
    }
    let (out, report) = (&mut done.messages, &mut done.report);

    report.tier1_chars_elided =
        elide_thinking_traces(out, cfg.keep_recent_turns, &model.delimiters);
    report.after_tokens = estimate_tokens(out, tok);
    if cfg.mode == CompactionMode::Auto && report.after_tokens <= target_tokens {
        return Ok(done);
    }

    (report.tier2_chars_shaken, report.artifacts) = shake_messages(out, cfg.keep_recent_turns);
    report.after_tokens = estimate_tokens(out, tok);
    if (cfg.mode == CompactionMode::Auto && report.after_tokens <= target_tokens)
        || cfg.mode == CompactionMode::Light
    {
        return Ok(done);
    }

    // Tier 3: the turns between the leading system messages and the kept turns fold into one
    // summary turn plus an assistant acknowledgement; the kept part starts at a user message, so
    // roles keep alternating.
    let first = out
        .iter()
        .position(|m| m.role != "system")
        .unwrap_or(out.len());
    let mut fold_end = out.len().saturating_sub(cfg.keep_recent_turns * 2);
    while fold_end > first && out[fold_end].role != "user" {
        fold_end -= 1;
    }
    if fold_end <= first + 1 {
        return Ok(done);
    }
    let folded = &out[first..fold_end];
    let transcript: String = folded
        .iter()
        .map(|m| format!("{}: {}\n", m.role, m.text()))
        .collect();
    let kept: Vec<ChatMessage> = out[..first].iter().chain(&out[fold_end..]).cloned().collect();
    let max_tokens = target_tokens
        .saturating_sub(estimate_tokens(&kept, tok))
        .clamp(64, 1024);
    let summary = cached_summary(summarizer, &transcript, max_tokens, tok, ctx)?;
    let text = format!(
        "[Compacted session summary of {} earlier messages]\n{summary}",
        folded.len()
    );
    let mut summary_msg = ChatMessage::new("user", text.as_str());
    if cfg.mode == CompactionMode::Aggressive && cfg.enable_snapcompact_vision && model.has_vision {
        let pages: Vec<Vec<u8>> = render_snapcompact_pages(&transcript)
            .iter()
            .map(ImageBuffer::to_png)
            .collect();
        let mut parts = vec![serde_json::json!({"type": "text", "text": text})];
        parts.extend(pages.iter().map(|_| serde_json::json!({"type": "image"})));
        summary_msg.content = Value::Array(parts);
        report.snapcompact_pages = pages.len();
        report.snapcompact_png_bytes = pages.iter().map(Vec::len).sum();
        done.media = pages.into_iter().map(MediaInput::Image).collect();
    }
    done.folded_media = folded.iter().map(media_parts).sum();
    report.tier3_turns_summarized = folded.len().div_ceil(2);
    out.splice(
        first..fold_end,
        [summary_msg, ChatMessage::new("assistant", "Ack.")],
    );
    report.after_tokens = estimate_tokens(out, tok);
    Ok(done)
}

/// Greedy generation of the summary on `s`: through its serving loop, or stepped on this thread
/// when `drive` (a private one-shot scheduler).
fn summarize_on(
    s: &ContinuousBatchScheduler,
    drive: bool,
    transcript: &str,
    max_tokens: usize,
) -> Result<String, String> {
    let msgs = [
        ChatMessage::new(
            "system",
            SUMMARY_PROMPT.replace("{max_tokens}", &max_tokens.to_string()),
        ),
        ChatMessage::new("user", transcript),
    ];
    let params = SamplingParams {
        temperature: Some(0.0),
        top_p: None,
        max_new_tokens: max_tokens,
        top_logprobs: None,
    };
    let (_, mut rx, tok, _, finish) = s.submit_with_media(params, &[], |t| {
        t.encode(&t.apply_chat_template_messages(&msgs, None))
    })?;
    let mut ids = Vec::new();
    if drive {
        use tokio::sync::mpsc::error::TryRecvError;
        'run: loop {
            s.step();
            loop {
                match rx.try_recv() {
                    Ok(t) => ids.push(t),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => break 'run,
                }
            }
        }
    } else {
        while let Some(t) = rx.blocking_recv() {
            ids.push(t);
        }
    }
    finish_outcome(&finish)?;
    Ok(tok.decode(&ids).trim().to_string())
}

/// Summarizes with the active model through its scheduler.
pub struct SchedulerSummarizer(pub Arc<ContinuousBatchScheduler>);

impl Summarizer for SchedulerSummarizer {
    fn summarize(&self, transcript: &str, max_tokens: usize) -> Result<String, String> {
        if !self.0.has_model() {
            return Err("no text model is loaded to summarize with".into());
        }
        summarize_on(&self.0, false, transcript, max_tokens)
    }
}

/// Summarizes with another model (`--compact-summarizer <id>`), loaded once and kept in a
/// one-entry cache keyed by id.
pub struct LocalModelSummarizer(ExecutableModel);

impl LocalModelSummarizer {
    /// Loads `id` as the CLI loads models: default device, saved re-encode policy, `HF_TOKEN`.
    pub fn load(id: &str) -> Result<Self, String> {
        static CACHE: parking_lot::Mutex<Option<(String, ExecutableModel)>> =
            parking_lot::Mutex::new(None);
        let mut cache = CACHE.lock();
        if let Some((_, m)) = cache.as_ref().filter(|(c, _)| c == id) {
            return Ok(Self(m.clone()));
        }
        let hub = crate::model::hub::HuggingFaceHub::new().with_budget(usize::MAX);
        let token = std::env::var("HF_TOKEN").ok();
        let loaded = crate::arch::load_model(
            &hub,
            id,
            &crate::arch::dispatch::LoadOpts {
                placement: crate::hardware::PlacementMode::Pin(crate::hardware::default_device()),
                token: token.as_deref(),
                quant_name: None,
                hw: &crate::hardware::HardwareCapability::probe_system(),
                reencode: crate::model::weight_format::ReencodeConfig::from_settings(),
                family: None,
                ctx: None,
                stage: &|_| {},
                admit: &|_| Ok(()),
                system1_first: false,
            },
        )
        .map_err(|e| format!("summarizer '{id}': {e}"))?;
        let crate::arch::dispatch::Loaded::Lm { model, .. } = loaded else {
            return Err(format!("summarizer '{id}' is not a text generation model"));
        };
        *cache = Some((id.to_string(), model.clone()));
        Ok(Self(model))
    }
}

impl Summarizer for LocalModelSummarizer {
    fn summarize(&self, transcript: &str, max_tokens: usize) -> Result<String, String> {
        summarize_on(&self.0.one_shot_scheduler(), true, transcript, max_tokens)
    }
}

/// Compacts `messages` and their `media` in place for `sched`'s active model: its tokenizer,
/// context (`ctx` overrides), thinking delimiters and vision support, with `cfg.summarizer`
/// (the active model when `None`). Token counts in the report include image tokens. On error
/// nothing changes.
pub fn compact_for_scheduler(
    sched: &Arc<ContinuousBatchScheduler>,
    messages: &mut Vec<ChatMessage>,
    media: &mut Vec<MediaInput>,
    ctx: Option<usize>,
    cfg: &CompactionConfig,
    force: bool,
) -> Result<CompactionReport, String> {
    let model = sched.model();
    let traits = model.as_ref().map_or_else(ModelTraits::default, |m| ModelTraits {
        delimiters: m.thinking_delimiters(),
        has_vision: m.has_vision(),
    });
    let ctx = ctx
        .or_else(|| model.as_ref().and_then(ExecutableModel::context_len))
        .unwrap_or(8192);
    let local;
    let active = SchedulerSummarizer(sched.clone());
    let summarizer: &dyn Summarizer = match &cfg.summarizer {
        Some(id) => {
            local = LocalModelSummarizer::load(id)?;
            &local
        }
        None => &active,
    };
    let tok = sched.tokenizer();
    let mut c = compact_chat_session(messages, &tok, ctx, cfg, &traits, summarizer, force)?;
    fn render(m: &[ChatMessage]) -> impl FnOnce(&Tokenizer) -> Vec<u32> + '_ {
        move |t| t.encode(&t.apply_chat_template_messages(m, None))
    }
    if !media.is_empty() {
        if let Ok(n) = sched.prompt_len(media, render(messages)) {
            c.report.before_tokens = n;
        }
    }
    media.splice(..c.folded_media.min(media.len()), c.media);
    if !media.is_empty() {
        if let Ok(n) = sched.prompt_len(media, render(&c.messages)) {
            c.report.after_tokens = n;
        }
    }
    *messages = c.messages;
    Ok(c.report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// Returns a fixed summary and counts its calls.
    struct Counting(Cell<usize>, &'static str);

    impl Summarizer for Counting {
        fn summarize(&self, transcript: &str, max_tokens: usize) -> Result<String, String> {
            assert!(!transcript.is_empty() && (64..=1024).contains(&max_tokens));
            self.0.set(self.0.get() + 1);
            Ok(self.1.to_string())
        }
    }

    #[test]
    fn tier1_elides_thinking_traces_only_from_older_turns() {
        let mut msgs = vec![
            ChatMessage::new("user", "Solve 2+2"),
            ChatMessage::new(
                "assistant",
                "<think>\nLet me compute 2+2 step by step...\n</think>\nThe answer is 4.",
            ),
            ChatMessage::new("user", "And 3+3?"),
            ChatMessage::new("assistant", "<think>Recent thought</think>\n6."),
        ];
        let elided = elide_thinking_traces(&mut msgs, 1, &[]);
        assert!(elided > 20);
        assert_eq!(msgs[1].text(), "The answer is 4.");
        // The most recent turn (last 2 messages) preserves its `<think>` block.
        assert!(msgs[3].text().contains("<think>Recent thought</think>"));
    }

    #[test]
    fn tier2_spills_long_blocks_and_dedups_repeats_in_older_turns() {
        let long_log: String = (0..30)
            .map(|i| format!("tier2 log line {i}: status ok"))
            .collect::<Vec<_>>()
            .join("\n");
        let boiler = "Boilerplate notice repeated by a tool on every call. ".repeat(5);
        let mut msgs = vec![
            ChatMessage::new("user", format!("Look at this:\n\n{long_log}\n\n{boiler}")),
            ChatMessage::new("assistant", "Ack."),
            ChatMessage::new("tool", format!("{boiler}\n\nshort tail")),
            ChatMessage::new("assistant", "Ok."),
            ChatMessage::new("user", "Recent prompt"),
            ChatMessage::new("assistant", "Recent reply"),
        ];
        let (shaken, artifacts) = shake_messages(&mut msgs, 1);
        assert!(shaken > 100);
        let first = msgs[0].text();
        assert!(first.starts_with("Look at this:\n\ntier2 log line 0"), "{first}");
        assert!(first.contains("[… 21 lines compacted; artifact "), "{first}");
        // The boilerplate block is new in message 0, a repeat in message 2.
        assert!(first.ends_with(&boiler), "{first}");
        let hex = format!("{:016x}", fnv64(boiler.as_bytes()));
        assert_eq!(
            msgs[2].text(),
            format!("[duplicate of artifact {hex}]\n\nshort tail")
        );
        assert_eq!(read_artifact(&hex).as_deref(), Some(boiler.as_str()));
        let log_hex = format!("{:016x}", fnv64(long_log.as_bytes()));
        assert_eq!(read_artifact(&log_hex), Some(long_log));
        assert_eq!(artifacts.len(), 2);
        assert!(artifacts.contains(&hex) && artifacts.contains(&log_hex));
        assert_eq!(read_artifact("../settings.json"), None);
    }

    /// Every printable ASCII character renders to a glyph that template-matches back to exactly
    /// that character, through PNG encoding, across page breaks and line wraps.
    #[test]
    fn glyph_pages_round_trip_ascii() {
        let unique: HashSet<[u8; 7]> = GLYPHS_5X7.iter().copied().collect();
        assert_eq!(unique.len(), 95, "two printable characters share a glyph");
        let printable: String = (0x20u8..=0x7e).map(char::from).collect();
        let text: String = (0..30)
            .map(|i| printable.chars().cycle().skip(i).take(95).collect::<String>() + "\n")
            .collect();
        let pages = render_snapcompact_pages(&text);
        assert_eq!(pages.len(), 2, "60 rows fill one 48-row page and part of another");
        let mut read = Vec::new();
        for page in &pages {
            let img = ImageBuffer::decode(&page.to_png()).unwrap();
            assert_eq!((img.width, img.height), (768, 768));
            let ink = |x: usize, y: usize| img.data[(y * img.width + x) * img.channels] < 128;
            for r in 0..PAGE_ROWS {
                let row: String = (0..PAGE_COLS)
                    .map(|c| {
                        let mut bits = [0u8; 7];
                        for (gy, b) in bits.iter_mut().enumerate() {
                            for gx in 0..5 {
                                let (x, y) = (c * CELL_W + 1 + 2 * gx, r * CELL_H + 1 + 2 * gy);
                                if ink(x, y) {
                                    *b |= 1 << gx;
                                }
                            }
                        }
                        let hits: Vec<usize> =
                            (0..95).filter(|&i| GLYPHS_5X7[i] == bits).collect();
                        assert_eq!(hits.len(), 1, "cell ({r}, {c}) matches {hits:?}");
                        char::from(0x20 + hits[0] as u8)
                    })
                    .collect();
                read.push(row.trim_end().to_string());
            }
        }
        let expected: Vec<String> = page_rows(&text)
            .into_iter()
            .map(|r| r.into_iter().collect::<String>().trim_end().to_string())
            .collect();
        assert_eq!(&read[..expected.len()], &expected[..]);
        assert!(read[expected.len()..].iter().all(String::is_empty));
    }

    fn session(tag: &str) -> Vec<ChatMessage> {
        let mut msgs = vec![ChatMessage::new("system", "You are helpful.")];
        for i in 0..6 {
            msgs.push(ChatMessage::new(
                "user",
                format!("{tag} question {i} about statistical mechanics and thermodynamics"),
            ));
            msgs.push(ChatMessage::new(
                "assistant",
                format!("<think>Long internal chain of thought for turn {i}...</think>\nAnswer {i}: Gibbs free energy is H - TS. {}", "The system minimizes it at constant temperature and pressure. ".repeat(3)),
            ));
        }
        msgs
    }

    /// The summary is generated once per folded transcript; repeating the compaction (the next
    /// request of an agent that resends its history) reuses it and renders identical bytes.
    #[test]
    fn summary_cached_and_prefix_stable() {
        let tok = Tokenizer::default_bpe();
        let msgs = session("summary_cached_and_prefix_stable");
        let cfg = CompactionConfig {
            mode: CompactionMode::Aggressive,
            keep_recent_turns: 1,
            ..CompactionConfig::default()
        };
        let s = Counting(Cell::new(0), "The user asked six thermodynamics questions.");
        let run = || {
            compact_chat_session(&msgs, &tok, 4096, &cfg, &ModelTraits::default(), &s, true)
                .unwrap()
        };
        let (a, b) = (run(), run());
        assert_eq!(s.0.get(), 1);
        assert_eq!(
            serde_json::to_string(&a.messages).unwrap(),
            serde_json::to_string(&b.messages).unwrap()
        );
        assert_eq!(a.messages.len(), 5, "system, summary, ack, last turn");
        assert!(a.messages[1]
            .text()
            .ends_with("The user asked six thermodynamics questions."));
        assert_eq!(a.report.tier3_turns_summarized, 5);
        assert!(a.report.after_tokens < a.report.before_tokens);
    }

    #[test]
    fn full_pipeline_gates_on_trigger_and_attaches_snapcompact_pages() {
        let tok = Tokenizer::default_bpe();
        let msgs = session("full_pipeline");
        let s = Counting(Cell::new(0), "Six questions on Gibbs free energy.");
        let vision = ModelTraits {
            delimiters: Vec::new(),
            has_vision: true,
        };
        let mut cfg = CompactionConfig {
            mode: CompactionMode::Aggressive,
            trigger_ratio: 0.3,
            keep_recent_turns: 1,
            ..CompactionConfig::default()
        };
        // Under the trigger nothing runs.
        let idle = compact_chat_session(&msgs, &tok, 1 << 20, &cfg, &vision, &s, false).unwrap();
        assert_eq!(idle.messages, msgs);
        assert_eq!(idle.report.after_tokens, idle.report.before_tokens);
        // Over it (0.3 x ctx = 0.6 x the prompt) every tier runs.
        let ctx = 2 * idle.report.before_tokens;
        assert!(ctx > 1024, "{ctx}");
        let c = compact_chat_session(&msgs, &tok, ctx, &cfg, &vision, &s, false).unwrap();
        assert!(c.report.tier1_chars_elided > 0 && c.report.tier3_turns_summarized > 0);
        assert!(c.report.after_tokens < c.report.before_tokens);
        assert!(!c.media.is_empty());
        assert_eq!(c.media.len(), c.report.snapcompact_pages);
        let MediaInput::Image(png) = &c.media[0] else {
            panic!("snapcompact page is not an image")
        };
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert_eq!(c.messages[1].content[1]["type"], "image");
        // Without vision (or outside Aggressive) the summary stays text.
        cfg.enable_snapcompact_vision = false;
        let c = compact_chat_session(&msgs, &tok, ctx, &cfg, &vision, &s, false).unwrap();
        assert!(c.media.is_empty() && c.messages[1].content.is_string());
    }
}
