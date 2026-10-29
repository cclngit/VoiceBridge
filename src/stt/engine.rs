use crate::config::SttConfig;
use reqwest::multipart::{Form, Part};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Once, Weak};
use std::time::{Duration, Instant};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub speaker: Option<String>,
    pub language: String,
    pub duration_ms: u128,
    pub model_name: String,
}

struct WhisperInner {
    // Field order matters: `state` must drop before `_ctx` so Metal buffers & residency sets
    // owned by the state and context are released in strict reverse-creation order.
    state: WhisperState,
    #[allow(dead_code)]
    _ctx: WhisperContext,
}

pub struct WhisperEngine {
    inner: Option<WhisperInner>,
    shutting_down: bool,
    language: String,
    n_threads: i32,
    model_name: String,
}

unsafe impl Send for WhisperEngine {}

static ATEXIT_ONCE: Once = Once::new();
static ATEXIT_WHISPER: std::sync::Mutex<Option<Weak<tokio::sync::Mutex<WhisperEngine>>>> =
    std::sync::Mutex::new(None);

unsafe extern "C" {
    fn atexit(cb: extern "C" fn()) -> i32;
}

extern "C" fn whisper_atexit_cleanup() {
    let weak_opt = ATEXIT_WHISPER.lock().ok().and_then(|mut g| g.take());
    if let Some(weak) = weak_opt {
        if let Some(engine_arc) = weak.upgrade() {
            for _ in 0..400 {
                if let Ok(mut engine) = engine_arc.try_lock() {
                    engine.shutdown();
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }
}

/// Registers a LIFO `atexit` safety net *after* `WhisperEngine::new` has initialized
/// `ggml_metal_device`, ensuring all Metal residency sets are freed before C++ static
/// destructors (`~unique_ptr<ggml_metal_device>` -> `ggml_metal_rsets_free`) run on `exit()`.
pub fn register_atexit_whisper_cleanup(engine: &Arc<tokio::sync::Mutex<WhisperEngine>>) {
    if let Ok(mut slot) = ATEXIT_WHISPER.lock() {
        *slot = Some(Arc::downgrade(engine));
    }
    ATEXIT_ONCE.call_once(|| unsafe {
        let _ = atexit(whisper_atexit_cleanup);
    });
}

fn format_whisper_display_name(path_str: &str, gpu_active: bool) -> String {
    if let Some(meta) = crate::stt::find_model_by_path(path_str) {
        if gpu_active {
            meta.name
        } else {
            meta.name.replace("(Metal GPU)", "(CPU)")
        }
    } else {
        let stem = Path::new(path_str)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Whisper");
        let backend = if gpu_active { "Metal GPU" } else { "CPU" };
        format!("{} ({})", stem, backend)
    }
}

impl WhisperEngine {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        language: &str,
        n_threads: usize,
        use_gpu: bool,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(use_gpu);

        let path_str = model_path
            .as_ref()
            .to_str()
            .ok_or_else(|| "Invalid model path")?;

        tracing::info!(
            "Loading Whisper model from '{}' (Requested GPU: {})...",
            path_str,
            use_gpu
        );
        let (ctx, gpu_active) = match WhisperContext::new_with_params(path_str, ctx_params) {
            Ok(c) => (c, use_gpu),
            Err(e) => {
                if use_gpu {
                    tracing::warn!("GPU initialization failed ({:?}), falling back to CPU mode...", e);
                    let mut cpu_params = WhisperContextParameters::default();
                    cpu_params.use_gpu(false);
                    let c = WhisperContext::new_with_params(path_str, cpu_params)
                        .map_err(|e2| format!("Failed to create Whisper context on CPU: {:?}", e2))?;
                    (c, false)
                } else {
                    return Err(format!("Failed to create Whisper context: {:?}", e).into());
                }
            }
        };

        let state = ctx
            .create_state()
            .map_err(|e| format!("Failed to create Whisper state: {:?}", e))?;

        let display_name = format_whisper_display_name(path_str, gpu_active);
        tracing::info!("Whisper model '{}' initialized successfully.", display_name);

        Ok(Self {
            inner: Some(WhisperInner { state, _ctx: ctx }),
            shutting_down: false,
            language: language.to_string(),
            n_threads: n_threads.max(1) as i32,
            model_name: display_name,
        })
    }

    /// Explicitly deallocate Whisper state and Metal GPU residency sets before process exit.
    pub fn shutdown(&mut self) {
        self.shutting_down = true;
        if self.inner.take().is_some() {
            tracing::info!("Whisper Metal GPU resources released cleanly.");
        }
    }

    pub fn set_language(&mut self, language: &str) {
        self.language = language.to_string();
    }

    pub fn reload_model<P: AsRef<Path>>(
        &mut self,
        model_path: P,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.reload_model_with_gpu(model_path, true)
    }

    pub fn reload_model_with_gpu<P: AsRef<Path>>(
        &mut self,
        model_path: P,
        use_gpu: bool,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.shutting_down {
            return Err("WhisperEngine is shutting down".into());
        }

        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(use_gpu);

        let path_str = model_path
            .as_ref()
            .to_str()
            .ok_or_else(|| "Invalid model path")?;

        tracing::info!("Dynamically reloading Whisper model from '{}' (GPU: {})...", path_str, use_gpu);

        // Release previous Metal context & residency sets before allocating the new model
        self.inner = None;

        let (ctx, gpu_active) = match WhisperContext::new_with_params(path_str, ctx_params) {
            Ok(c) => (c, use_gpu),
            Err(e) => {
                if use_gpu {
                    tracing::warn!("GPU reload failed ({:?}), falling back to CPU mode...", e);
                    let mut cpu_params = WhisperContextParameters::default();
                    cpu_params.use_gpu(false);
                    let c = WhisperContext::new_with_params(path_str, cpu_params)
                        .map_err(|e2| format!("Failed to reload Whisper context on CPU: {:?}", e2))?;
                    (c, false)
                } else {
                    return Err(format!("Failed to reload Whisper context: {:?}", e).into());
                }
            }
        };

        let state = ctx
            .create_state()
            .map_err(|e| format!("Failed to create Whisper state: {:?}", e))?;

        self.inner = Some(WhisperInner { state, _ctx: ctx });
        self.model_name = format_whisper_display_name(path_str, gpu_active);
        tracing::info!("Model '{}' reloaded successfully.", self.model_name);
        Ok(())
    }

    /// Transcribe 16kHz mono f32 samples
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        language_override: Option<&str>,
        translate_to_en: bool,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error + Send + Sync>> {
        if self.shutting_down {
            return Err("WhisperEngine is shutting down".into());
        }
        let inner = self
            .inner
            .as_mut()
            .ok_or("WhisperEngine has no loaded model")?;

        let start_time = Instant::now();

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.n_threads);
        params.set_translate(translate_to_en);

        let lang_to_use = language_override.unwrap_or(&self.language).trim();
        let is_auto = lang_to_use.is_empty() || lang_to_use.eq_ignore_ascii_case("auto");

        if !is_auto {
            params.set_language(Some(lang_to_use));
            params.set_detect_language(false);
        } else {
            params.set_language(None);
            params.set_detect_language(true);
        }

        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params.set_temperature(0.0);
        params.set_temperature_inc(0.0);
        params.set_no_speech_thold(0.6);
        params.set_entropy_thold(2.4);
        params.set_logprob_thold(-1.0);

        inner
            .state
            .full(params, samples)
            .map_err(|e| format!("Whisper inference error: {:?}", e))?;

        let mut combined_text = String::new();
        for segment in inner.state.as_iter() {
            let seg_text = segment.to_string();
            combined_text.push_str(&seg_text);
        }

        let raw_text = combined_text.trim();
        let (speaker, cleaned_text) = extract_speaker_if_present(raw_text);
        let duration_ms = start_time.elapsed().as_millis();

        let final_lang = if !is_auto {
            lang_to_use.to_string()
        } else {
            let detected_id = inner.state.full_lang_id_from_state();
            whisper_rs::get_lang_str(detected_id)
                .unwrap_or("auto")
                .to_string()
        };

        Ok(TranscriptionResult {
            text: cleaned_text,
            speaker,
            language: final_lang,
            duration_ms,
            model_name: self.model_name.clone(),
        })
    }
}

/// In-memory WAV encoder: converts 16kHz mono f32 samples to uncompressed 16-bit PCM WAV bytes
pub fn samples_to_wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(44 + samples.len() * 2);
    let num_samples = samples.len() as u32;
    let byte_rate = sample_rate * 2;
    let data_len = num_samples * 2;
    let riff_len = 36 + data_len;

    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_len.to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1Size (16 for PCM)
    bytes.extend_from_slice(&1u16.to_le_bytes());  // AudioFormat (1 for PCM)
    bytes.extend_from_slice(&1u16.to_le_bytes());  // NumChannels (1 mono)
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());  // BlockAlign (2 bytes per frame)
    bytes.extend_from_slice(&16u16.to_le_bytes()); // BitsPerSample (16 bits)
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());

    for &s in samples {
        let i = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        bytes.extend_from_slice(&i.to_le_bytes());
    }
    bytes
}

/// Client for remote/local ASR endpoints (Qwen3-ASR, Voxtral, Canary, VibeVoice, Moonshine)
pub struct RemoteAsrClient {
    client: Client,
}

#[derive(Deserialize)]
struct RemoteTranscriptionResponse {
    text: Option<String>,
    speaker: Option<String>,
}

impl RemoteAsrClient {
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(12))
            .build()
            .unwrap_or_default();
        Self { client }
    }

    pub async fn health_check(&self, url: &str, api_key: Option<&str>) -> bool {
        let mut req = self.client.get(url);
        if let Some(key) = api_key {
            if !key.is_empty() {
                req = req.bearer_auth(key);
            }
        }
        match req.send().await {
            Ok(resp) => resp.status().is_success() || resp.status().as_u16() == 405,
            Err(_) => false,
        }
    }

    pub async fn transcribe(
        &self,
        samples: &[f32],
        config: &SttConfig,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = Instant::now();
        let wav_bytes = samples_to_wav_bytes(samples, 16000);

        let file_part = Part::bytes(wav_bytes)
            .file_name("audio.wav")
            .mime_str("audio/wav")?;

        let mut form = Form::new()
            .part("file", file_part)
            .text("model", config.model_name.clone());

        if config.language != "auto" && !config.language.is_empty() {
            form = form.text("language", config.language.clone());
        }

        let mut req = self.client.post(&config.api_url).multipart(form);

        if let Some(ref key) = config.api_key {
            if !key.is_empty() {
                req = req.bearer_auth(key);
            }
        }

        let resp = req.send().await?;
        if !resp.status().is_success() {
            return Err(format!("STT API returned HTTP status {}", resp.status()).into());
        }

        let data: RemoteTranscriptionResponse = resp.json().await?;
        let raw_text = data.text.unwrap_or_default();
        let (parsed_speaker, cleaned_text) = extract_speaker_if_present(&raw_text);
        let final_speaker = data.speaker.or(parsed_speaker);

        let duration_ms = start_time.elapsed().as_millis();

        Ok(TranscriptionResult {
            text: cleaned_text,
            speaker: final_speaker,
            language: config.language.clone(),
            duration_ms,
            model_name: config.model_name.clone(),
        })
    }
}

/// Helper to parse speaker diarization tags like "[Speaker 1]: hello", "Locuteur A: ...", "Speaker 2 - ..."
pub fn extract_speaker_if_present(text: &str) -> (Option<String>, String) {
    let trimmed = text.trim();

    // Check bracketed format: [Speaker 1] or [Locuteur A]
    if trimmed.starts_with('[') {
        if let Some(close_idx) = trimmed.find(']') {
            let potential_speaker = &trimmed[1..close_idx];
            let lower = potential_speaker.to_lowercase();
            if lower.contains("speaker") || lower.contains("locuteur") || lower.contains("person") {
                let rest = trimmed[close_idx + 1..]
                    .trim_start_matches(':')
                    .trim_start_matches('-')
                    .trim();
                return (Some(potential_speaker.to_string()), rest.to_string());
            }
        }
    }

    // Check prefix format: "Speaker 1:" or "Locuteur 2:"
    for prefix in &["Speaker ", "speaker ", "Locuteur ", "locuteur "] {
        if let Some(after_prefix) = trimmed.strip_prefix(prefix) {
            if let Some(colon_idx) = after_prefix.find(':') {
                let speaker_id = &after_prefix[..colon_idx].trim();
                let full_speaker = format!("{}{}", prefix.trim(), speaker_id);
                let rest = after_prefix[colon_idx + 1..].trim();
                return (Some(full_speaker), rest.to_string());
            }
        }
    }

    (None, trimmed.to_string())
}
