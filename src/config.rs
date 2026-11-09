use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Placeholder returned instead of secrets by the HTTP API.
pub const REDACTED_SECRET: &str = "__voicebridge_redacted__";

static CONFIG_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Register the configuration file used for every later save (honours `--config`).
pub fn set_config_path(path: PathBuf) {
    let _ = CONFIG_PATH.set(path);
}

pub fn config_path() -> PathBuf {
    CONFIG_PATH.get().cloned().unwrap_or_else(|| PathBuf::from("config.json"))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct AppConfig {
    #[serde(default)]
    pub audio: AudioConfig,
    #[serde(default)]
    pub stt: SttConfig,
    #[serde(default)]
    pub translation: TranslationConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub obs: ObsConfig,
}


impl AppConfig {
    pub fn sanitize(&mut self) {
        use crate::stt::models::{ENGINE_HIBIKI, ENGINE_KYUTAI, ENGINE_REMOTE, ENGINE_WHISPER};
        // 0. Normalise engine identifiers (older versions used "remote_api").
        match self.stt.engine_type.as_str() {
            ENGINE_WHISPER | ENGINE_REMOTE | ENGINE_KYUTAI | ENGINE_HIBIKI => {}
            "remote_api" => self.stt.engine_type = ENGINE_REMOTE.to_string(),
            other => {
                tracing::warn!("Moteur STT inconnu '{}', retour à Whisper local.", other);
                self.stt.engine_type = ENGINE_WHISPER.to_string();
            }
        }
        match self.translation.engine.as_str() {
            crate::translate::engine::ENGINE_MADLAD
            | crate::translate::engine::ENGINE_LOCAL_LLM
            | crate::translate::engine::ENGINE_OFF => {}
            // "auto" used to send text to Google / MyMemory / LibreTranslate mirrors.
            _ => self.translation.engine = crate::translate::engine::ENGINE_MADLAD.to_string(),
        }
        if self.translation.madlad_model_dir.trim().is_empty() {
            self.translation.madlad_model_dir = default_madlad_dir();
        }
        // 100% local: endpoints must stay on this machine or the LAN.
        if !crate::netpolicy::is_local_endpoint(&self.stt.api_url) {
            tracing::warn!(
                "URL ASR '{}' refusée (adresse non locale) : remplacée par le serveur local.",
                self.stt.api_url
            );
            self.stt.api_url = default_api_url();
            self.stt.api_key = None;
            if self.stt.engine_type == ENGINE_REMOTE {
                self.stt.engine_type = ENGINE_WHISPER.to_string();
            }
        }
        if !crate::netpolicy::is_local_endpoint(&self.translation.local_api_url) {
            tracing::warn!(
                "URL LLM de traduction '{}' refusée (adresse non locale).",
                self.translation.local_api_url
            );
            self.translation.local_api_url = TranslationConfig::default().local_api_url;
            self.translation.api_key = None;
        }
        if self.stt.n_threads == 0 {
            self.stt.n_threads = 4;
        }

        // 1. If engine_type is remote_asr pointing to offline local server, fallback to whisper_local
        if self.stt.engine_type != "whisper_local" && self.stt.api_url.contains("127.0.0.1:8000")
            && (self.stt.model_id == "qwen3_asr" || self.stt.model_id.is_empty()) {
                self.stt.engine_type = "whisper_local".to_string();
                self.stt.model_id = "whisper_turbo_q5".to_string();
                self.stt.model_path = "models/ggml-large-v3-turbo-q5_0.bin".to_string();
                self.stt.model_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin".to_string();
                self.stt.model_name = format!("Whisper Large-v3-Turbo Q5_0 ({})", crate::stt::active_whisper_backend_label());
            }

        // 2. Ensure languages are normalized
        if self.stt.language.trim().is_empty() {
            self.stt.language = "auto".to_string();
        }
        if self.translation.source_lang.trim().is_empty() {
            self.translation.source_lang = "auto".to_string();
        }
        if self.translation.target_lang.trim().is_empty() {
            self.translation.target_lang = "en".to_string();
        }

        // 3. Prevent identity translation loop (e.g. source fr -> target fr)
        if !self.translation.source_lang.eq_ignore_ascii_case("auto")
            && self.translation.source_lang.eq_ignore_ascii_case(&self.translation.target_lang)
        {
            if self.translation.source_lang.eq_ignore_ascii_case("en") {
                self.translation.target_lang = "fr".to_string();
            } else {
                self.translation.target_lang = "en".to_string();
            }
        }

        // 4. If distil_whisper is selected with non-English language, switch to turbo
        if self.stt.model_id == "distil_whisper_large_v3"
            && !self.stt.language.eq_ignore_ascii_case("en")
            && !self.translation.source_lang.eq_ignore_ascii_case("en")
        {
            tracing::warn!("Distil-Whisper est réservé à l'anglais. Bascule automatique sur Whisper Large-v3-Turbo pour le support multilingue.");
            self.stt.model_id = "whisper_turbo_q5".to_string();
            self.stt.model_path = "models/ggml-large-v3-turbo-q5_0.bin".to_string();
            self.stt.model_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin".to_string();
            self.stt.model_name = format!("Whisper Large-v3-Turbo Q5_0 ({})", crate::stt::active_whisper_backend_label());
        }
    }

    pub fn load_or_default<P: AsRef<Path>>(path: P) -> Self {
        if let Ok(content) = fs::read_to_string(path.as_ref()) {
            match serde_json::from_str::<Self>(&content) {
                Ok(mut cfg) => {
                    cfg.sanitize();
                    tracing::info!("Loaded configuration from {}", path.as_ref().display());
                    return cfg;
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to parse config file ({}): {}. Falling back to defaults.",
                        path.as_ref().display(),
                        e
                    );
                }
            }
        }
        let mut default_cfg = Self::default();
        default_cfg.sanitize();
        let _ = default_cfg.save_to_file(path.as_ref());
        default_cfg
    }

    /// Atomically write the configuration (temporary file + rename).
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let path = path.as_ref();
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, json)?;
        fs::rename(&tmp, path)
    }

    /// Save to the configuration file selected at startup.
    pub fn save_default(&self) -> Result<(), std::io::Error> {
        let path = config_path();
        let res = self.save_to_file(&path);
        if let Err(e) = &res {
            tracing::error!("Impossible d'enregistrer la configuration {} : {}", path.display(), e);
        }
        res
    }

    /// Copy of the configuration safe to expose over HTTP (secrets masked).
    pub fn redacted(&self) -> Self {
        let mut c = self.clone();
        let mask = |k: &mut Option<String>| {
            if k.as_deref().is_some_and(|v| !v.is_empty()) {
                *k = Some(REDACTED_SECRET.to_string());
            }
        };
        mask(&mut c.stt.api_key);
        mask(&mut c.translation.api_key);
        c
    }

    /// Restore secrets that a client sent back masked.
    pub fn restore_redacted_secrets(&mut self, previous: &AppConfig) {
        if self.stt.api_key.as_deref() == Some(REDACTED_SECRET) {
            self.stt.api_key = previous.stt.api_key.clone();
        }
        if self.translation.api_key.as_deref() == Some(REDACTED_SECRET) {
            self.translation.api_key = previous.translation.api_key.clone();
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub device_name: Option<String>,
    pub sample_rate: u32,
    pub vad_threshold: f32,
    pub vad_silence_duration_ms: u64,
    pub vad_min_speech_ms: u64,
    pub vad_max_speech_ms: u64,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device_name: None,
            sample_rate: 16000,
            vad_threshold: 0.015,
            vad_silence_duration_ms: 700,
            vad_min_speech_ms: 350,
            vad_max_speech_ms: 15000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttConfig {
    #[serde(default = "default_engine_type")]
    pub engine_type: String, // "whisper_local", "remote_asr", "kyutai_streaming", "hibiki_streaming"
    #[serde(default = "default_model_id")]
    pub model_id: String, // catalog id (see stt::models)
    /// Whisper model file, or model directory for Kyutai / Hibiki.
    pub model_path: String,
    pub model_url: String,
    #[serde(default = "default_model_name")]
    pub model_name: String,
    #[serde(default = "default_api_url")]
    pub api_url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    pub language: String,
    pub n_threads: usize,
    pub use_gpu: bool,
    #[serde(default = "default_gpu_backend")]
    pub gpu_backend: String, // "auto", "cuda", "vulkan", "metal", "rocm", "cpu"
    #[serde(default)]
    pub gpu_device: i32,
    #[serde(default)]
    pub flash_attn: bool,
    #[serde(default = "default_true")]
    pub enable_diarization: bool,
}

fn default_engine_type() -> String {
    "whisper_local".to_string()
}
fn default_model_id() -> String {
    "whisper_turbo_q5".to_string()
}
fn default_model_name() -> String {
    format!("Whisper Large-v3-Turbo Q5_0 ({})", crate::stt::active_whisper_backend_label())
}
fn default_api_url() -> String {
    "http://127.0.0.1:8000/v1/audio/transcriptions".to_string()
}
fn default_gpu_backend() -> String {
    "auto".to_string()
}
fn default_true() -> bool {
    true
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            engine_type: default_engine_type(),
            model_id: default_model_id(),
            model_path: "models/ggml-large-v3-turbo-q5_0.bin".to_string(),
            model_url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin".to_string(),
            model_name: default_model_name(),
            api_url: default_api_url(),
            api_key: None,
            language: "auto".to_string(),
            n_threads: 4,
            use_gpu: true,
            gpu_backend: default_gpu_backend(),
            gpu_device: 0,
            flash_attn: false,
            enable_diarization: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationConfig {
    pub enabled: bool,
    pub source_lang: String,
    pub target_lang: String,
    /// "madlad" (in-process MADLAD-400), "local_llm" (local OpenAI-compatible server) or "off".
    pub engine: String,
    pub local_api_url: String,
    pub model_name: String,
    pub api_key: Option<String>,
    /// Directory holding the MADLAD-400 model files.
    #[serde(default = "default_madlad_dir")]
    pub madlad_model_dir: String,
}

fn default_madlad_dir() -> String {
    crate::stt::models::MADLAD_DEFAULT_DIR.to_string()
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            source_lang: "auto".to_string(),
            target_lang: "en".to_string(),
            engine: crate::translate::engine::ENGINE_MADLAD.to_string(),
            local_api_url: "http://127.0.0.1:1234/v1/chat/completions".to_string(),
            model_name: "local-model".to_string(),
            api_key: None,
            madlad_model_dir: default_madlad_dir(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObsConfig {
    pub mode: String,          // "dual", "translated", "source"
    pub font_size: u32,        // e.g. 32
    pub text_color: String,    // e.g. "#ffffff"
    pub highlight_color: String, // e.g. "#38bdf8"
    pub bg_color: String,      // e.g. "rgba(0, 0, 0, 0.65)"
    pub fade_timeout_secs: u32, // e.g. 6 (0 = never fade)
    pub position: String,      // "bottom", "top", "center"
    pub show_speaker: bool,
}

impl Default for ObsConfig {
    fn default() -> Self {
        Self {
            mode: "dual".to_string(),
            font_size: 32,
            text_color: "#ffffff".to_string(),
            highlight_color: "#38bdf8".to_string(),
            bg_color: "rgba(0, 0, 0, 0.65)".to_string(),
            fade_timeout_secs: 6,
            position: "bottom".to_string(),
            show_speaker: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_online_translation_engine_is_migrated_to_local() {
        let mut cfg = AppConfig::default();
        cfg.translation.engine = "auto".into();
        cfg.stt.engine_type = "remote_api".into();
        cfg.sanitize();
        assert_eq!(cfg.translation.engine, "madlad");
        assert_eq!(cfg.stt.engine_type, "remote_asr");
    }

    #[test]
    fn internet_endpoints_are_rejected() {
        let mut cfg = AppConfig::default();
        cfg.stt.engine_type = "remote_asr".into();
        cfg.stt.api_url = "https://evil.example.com/v1/audio/transcriptions".into();
        cfg.stt.api_key = Some("k".into());
        cfg.translation.local_api_url = "https://api.openai.com/v1/chat/completions".into();
        cfg.sanitize();
        assert!(cfg.stt.api_url.starts_with("http://127.0.0.1"));
        assert_eq!(cfg.stt.api_key, None);
        assert_eq!(cfg.stt.engine_type, "whisper_local");
        assert!(cfg.translation.local_api_url.starts_with("http://127.0.0.1"));
    }

    #[test]
    fn secrets_are_redacted_and_restored() {
        let mut cfg = AppConfig::default();
        cfg.stt.api_key = Some("secret-stt".into());
        cfg.translation.api_key = Some("secret-llm".into());
        let public = cfg.redacted();
        let json = serde_json::to_string(&public).unwrap();
        assert!(!json.contains("secret-stt") && !json.contains("secret-llm"));
        let mut incoming = public.clone();
        incoming.restore_redacted_secrets(&cfg);
        assert_eq!(incoming.stt.api_key.as_deref(), Some("secret-stt"));
        assert_eq!(incoming.translation.api_key.as_deref(), Some("secret-llm"));
    }

    #[test]
    fn save_is_atomic_and_round_trips() {
        let dir = std::env::temp_dir().join(format!("vb-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        let mut cfg = AppConfig::default();
        cfg.server.port = 4242;
        cfg.save_to_file(&path).unwrap();
        let loaded = AppConfig::load_or_default(&path);
        assert_eq!(loaded.server.port, 4242);
        assert!(!path.with_extension("json.tmp").exists());
        let _ = std::fs::remove_dir_all(dir);
    }
}
