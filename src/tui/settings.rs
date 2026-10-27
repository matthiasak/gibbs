use crate::model::weight_format::{ReencodePolicy, VariantPolicy};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::SystemTime;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct GibbsSettings {
    pub reencode: ReencodePolicy,
    pub variant: VariantPolicy,
    /// Mean-KL limit of the GPU re-encode gate (nats); `None` keeps the policy's own.
    pub reencode_max_kl: Option<f32>,
    /// Speculative decoding and online noise drafter tuning knobs.
    #[serde(default)]
    pub draft: crate::runtime::noise_draft::DraftKnobs,
    /// Automatic session compaction configuration.
    #[serde(default)]
    pub compact: crate::compact::CompactionConfig,
}

#[cfg(not(test))]
pub fn settings_path() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".cache/gibbs/settings.json")
    } else {
        PathBuf::from(".cache/gibbs/settings.json")
    }
}

#[cfg(test)]
pub fn settings_path() -> PathBuf {
    thread_local! {
        static DIR: tempfile::TempDir = tempfile::tempdir().unwrap();
    }
    DIR.with(|d| d.path().join("settings.json"))
}

/// The last settings read or written, by path and mtime (each test thread has its own path).
static CACHE: Mutex<Option<(PathBuf, SystemTime, GibbsSettings)>> = Mutex::new(None);

pub fn load_settings() -> GibbsSettings {
    let path = settings_path();
    let Some(mtime) = std::fs::metadata(&path).and_then(|m| m.modified()).ok() else {
        return GibbsSettings::default();
    };
    if let Some((cp, ct, s)) = CACHE.lock().as_ref() {
        if *cp == path && *ct == mtime {
            return s.clone();
        }
    }
    let Ok(data) = std::fs::read_to_string(&path) else {
        return GibbsSettings::default();
    };
    match serde_json::from_str::<GibbsSettings>(&data) {
        Ok(s) => {
            *CACHE.lock() = Some((path, mtime, s.clone()));
            s
        }
        Err(err) => {
            tracing::warn!("settings: {}: {err}; using defaults", path.display());
            GibbsSettings::default()
        }
    }
}

pub fn save_settings(s: &GibbsSettings) -> Result<(), String> {
    let path = settings_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let data = serde_json::to_string_pretty(s).map_err(|e| e.to_string())?;
    std::fs::write(&path, data).map_err(|e| e.to_string())?;
    *CACHE.lock() = std::fs::metadata(&path)
        .and_then(|m| m.modified())
        .ok()
        .map(|t| (path, t, s.clone()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_settings_load_save() {
        for p in [
            ReencodePolicy::MaxSpeed,
            ReencodePolicy::LowBitNative,
            ReencodePolicy::BitExact,
        ] {
            assert_eq!(
                serde_json::to_string(&p).unwrap(),
                format!("\"{}\"", p.name())
            );
        }
        for v in [
            VariantPolicy::Fastest,
            VariantPolicy::Fastest4Bit,
            VariantPolicy::BestQuality,
        ] {
            assert_eq!(
                serde_json::to_string(&v).unwrap(),
                format!("\"{}\"", v.name())
            );
        }
        let s = GibbsSettings {
            reencode: ReencodePolicy::LowBitNative,
            variant: VariantPolicy::Fastest4Bit,
            reencode_max_kl: Some(0.05),
            draft: crate::runtime::noise_draft::DraftKnobs::default(),
            compact: crate::compact::CompactionConfig::default(),
        };
        save_settings(&s).unwrap();
        assert_eq!(load_settings(), s);
        let _ = std::fs::remove_file(settings_path());
        assert_eq!(load_settings(), GibbsSettings::default());
        // Aliases accepted by FromStr also deserialize.
        let alias: GibbsSettings =
            serde_json::from_str(r#"{"reencode":"lowbit","variant":"quality"}"#).unwrap();
        assert_eq!(alias.reencode, ReencodePolicy::LowBitNative);
        assert_eq!(alias.variant, VariantPolicy::BestQuality);
    }
}
