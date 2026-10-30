use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            audio: AudioConfig::default(),
            stt: SttConfig::default(),
            translation: TranslationConfig::default(),
            server: ServerConfig::default(),
            obs: ObsConfig::default(),
        }
    }
}

impl AppConfig {
    pub fn load_or_default<P: AsRef<Path>>(path: P) -> Self {
        if let Ok(content) = fs::read_to_string(path.as_ref()) {
            match serde_json::from_str(&content) {
                Ok(cfg) => {
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
        let default_cfg = Self::default();
        if let Ok(json) = serde_json::to_string_pretty(&default_cfg) {
            let _ = fs::write(path.as_ref(), json);
        }
        default_cfg
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        fs::write(path.as_ref(), json)
    }

    pub fn save_default(&self) -> Result<(), std::io::Error> {
        self.save_to_file("config.json")
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
    pub engine_type: String, // "whisper_local" or "remote_api"
    #[serde(default = "default_model_id")]
    pub model_id: String, // "whisper_local", "qwen3_asr", "voxtral_mini", "canary_qwen", "vibevoice_asr", "moonshine", "custom_api"
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
    "whisper_tiny".to_string()
}
fn default_model_name() -> String {
    format!("Whisper Tiny ({})", crate::stt::active_whisper_backend_label())
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
            model_path: "models/ggml-tiny.bin".to_string(),
            model_url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin".to_string(),
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
    pub engine: String, // "auto", "whisper", "local_llm", "mock"
    pub local_api_url: String,
    pub model_name: String,
    pub api_key: Option<String>,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            source_lang: "auto".to_string(),
            target_lang: "fr".to_string(),
            engine: "auto".to_string(),
            local_api_url: "http://127.0.0.1:1234/v1/chat/completions".to_string(),
            model_name: "local-model".to_string(),
            api_key: None,
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
