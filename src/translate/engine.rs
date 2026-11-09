//! Local translation service.
//!
//! All engines run on this machine; no text ever leaves it:
//! - `madlad` (default): MADLAD-400 3B running in-process with Candle.
//! - `local_llm`: an OpenAI-compatible chat server on localhost / LAN (LM Studio, Ollama,
//!   llama.cpp server...). Non-local URLs are refused.
//! - `off`: translation disabled (text passed through).

use super::madlad_engine::MadladTranslator;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const ENGINE_MADLAD: &str = "madlad";
pub const ENGINE_LOCAL_LLM: &str = "local_llm";
pub const ENGINE_OFF: &str = "off";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResult {
    pub original_text: String,
    pub translated_text: String,
    pub source_lang: String,
    pub target_lang: String,
    pub duration_ms: u128,
    pub engine_used: String,
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

/// Lazily loaded MADLAD model, shared between translation calls.
#[derive(Default)]
struct MadladSlot {
    loaded: Option<(PathBuf, MadladTranslator)>,
    last_error: Option<(PathBuf, String)>,
}

pub struct TranslationService {
    client: Client,
    pub enabled: bool,
    pub source_lang: String,
    pub target_lang: String,
    pub engine: String,
    pub local_api_url: String,
    pub model_name: String,
    pub api_key: Option<String>,
    /// Directory of the MADLAD-400 model files.
    pub madlad_dir: PathBuf,
    pub use_gpu: bool,
    pub gpu_device: i32,
    madlad: Arc<Mutex<MadladSlot>>,
}

impl TranslationService {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        enabled: bool,
        source_lang: String,
        target_lang: String,
        engine: String,
        local_api_url: String,
        model_name: String,
        api_key: Option<String>,
        madlad_dir: PathBuf,
    ) -> Self {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .timeout(Duration::from_secs(20))
            .build()
            .unwrap_or_default();

        Self {
            client,
            enabled,
            source_lang,
            target_lang,
            engine,
            local_api_url,
            model_name,
            api_key,
            madlad_dir,
            use_gpu: true,
            gpu_device: 0,
            madlad: Arc::new(Mutex::new(MadladSlot::default())),
        }
    }

    pub fn set_languages(&mut self, source: String, target: String) {
        self.source_lang = source;
        self.target_lang = target;
    }

    pub fn apply_config(&mut self, cfg: &crate::config::TranslationConfig, use_gpu: bool, gpu_device: i32) {
        self.enabled = cfg.enabled;
        self.source_lang = cfg.source_lang.clone();
        self.target_lang = cfg.target_lang.clone();
        self.engine = cfg.engine.clone();
        self.local_api_url = cfg.local_api_url.clone();
        self.model_name = cfg.model_name.clone();
        self.api_key = cfg.api_key.clone();
        let dir = PathBuf::from(&cfg.madlad_model_dir);
        if dir != self.madlad_dir {
            self.unload_madlad();
            self.madlad_dir = dir;
        }
        if use_gpu != self.use_gpu || gpu_device != self.gpu_device {
            self.unload_madlad();
            self.use_gpu = use_gpu;
            self.gpu_device = gpu_device;
        }
    }

    pub fn unload_madlad(&self) {
        if let Ok(mut slot) = self.madlad.lock() {
            slot.loaded = None;
            slot.last_error = None;
        }
    }

    /// Handle that can load / run MADLAD from a blocking thread.
    pub fn madlad_loader(&self) -> MadladLoader {
        MadladLoader {
            slot: self.madlad.clone(),
            dir: self.madlad_dir.clone(),
            use_gpu: self.use_gpu,
            gpu_device: self.gpu_device,
        }
    }

    /// Translate a transcription. Never sends text outside of this machine / LAN.
    pub async fn translate(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
    ) -> TranslationResult {
        let start_time = Instant::now();
        let trimmed = text.trim().to_string();
        let src = if source_lang.is_empty() { self.source_lang.as_str() } else { source_lang };
        let src = if src.is_empty() { "auto" } else { src }.to_string();
        let tgt = if target_lang.is_empty() { self.target_lang.as_str() } else { target_lang };
        let tgt = if tgt.is_empty() { "en" } else { tgt }.to_string();

        let passthrough = |engine_used: &str| TranslationResult {
            original_text: trimmed.clone(),
            translated_text: trimmed.clone(),
            source_lang: src.clone(),
            target_lang: tgt.clone(),
            duration_ms: start_time.elapsed().as_millis(),
            engine_used: engine_used.to_string(),
        };

        if !self.enabled
            || self.engine == ENGINE_OFF
            || trimmed.is_empty()
            || (!src.eq_ignore_ascii_case("auto") && primary(&src) == primary(&tgt))
        {
            return passthrough("Pass-Through");
        }

        let outcome = match self.engine.as_str() {
            ENGINE_LOCAL_LLM => self
                .translate_via_local_llm(&trimmed, &tgt)
                .await
                .map(|t| (t, "LLM local".to_string())),
            _ => self
                .translate_via_madlad(&trimmed, &tgt)
                .await
                .map(|(t, hw)| (t, format!("MADLAD-400 local ({hw})"))),
        };

        match outcome {
            Ok((translated, engine_used)) if !translated.trim().is_empty() => TranslationResult {
                original_text: trimmed.clone(),
                translated_text: translated,
                source_lang: src.clone(),
                target_lang: tgt.clone(),
                duration_ms: start_time.elapsed().as_millis(),
                engine_used,
            },
            Ok(_) => passthrough("Non traduit (sortie vide)"),
            Err(e) => {
                tracing::warn!("Traduction locale indisponible : {}", e);
                passthrough("Non traduit (moteur local indisponible)")
            }
        }
    }

    async fn translate_via_madlad(&self, text: &str, tgt: &str) -> Result<(String, String), String> {
        let loader = self.madlad_loader();
        let text = text.to_string();
        let tgt = tgt.to_string();
        tokio::task::spawn_blocking(move || loader.translate(&text, &tgt))
            .await
            .map_err(|e| format!("translation task failed: {e}"))?
    }

    async fn translate_via_local_llm(&self, text: &str, tgt: &str) -> Result<String, String> {
        crate::netpolicy::ensure_local_endpoint(&self.local_api_url)?;
        let target_name = language_name(tgt);
        let system_prompt = format!(
            "You are a fast real-time speech translator. Translate speech into {}. Return ONLY the direct translation, without explanations, quotes, or prefixes.",
            target_name
        );
        let request_payload = ChatCompletionRequest {
            model: self.model_name.clone(),
            messages: vec![
                ChatMessage { role: "system".to_string(), content: system_prompt },
                ChatMessage { role: "user".to_string(), content: text.to_string() },
            ],
            temperature: 0.2,
            max_tokens: 256,
        };
        let mut req = self.client.post(&self.local_api_url).json(&request_payload);
        if let Some(key) = self.api_key.as_deref().filter(|k| !k.is_empty()) {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await.map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("LLM local : HTTP {}", resp.status()));
        }
        let chat_resp: ChatCompletionResponse = resp.json().await.map_err(|e| e.to_string())?;
        chat_resp
            .choices
            .first()
            .map(|c| c.message.content.trim().to_string())
            .ok_or_else(|| "LLM local : aucune réponse".to_string())
    }
}

/// Owned handle on the shared MADLAD model, usable from blocking threads.
#[derive(Clone)]
pub struct MadladLoader {
    slot: Arc<Mutex<MadladSlot>>,
    dir: PathBuf,
    use_gpu: bool,
    gpu_device: i32,
}

impl MadladLoader {
    /// Load the model if needed; returns the device label.
    pub fn preload(&self) -> Result<String, String> {
        let device = crate::ml::resolve_device(crate::ml::DeviceSlot::Translation, self.use_gpu, self.gpu_device);
        let mut slot = self.slot.lock().map_err(|_| "translator lock poisoned".to_string())?;
        slot.last_error = None;
        ensure_madlad_loaded(&mut slot, &self.dir, &device)?;
        Ok(crate::ml::device_label(&device).to_string())
    }

    pub fn translate(&self, text: &str, tgt: &str) -> Result<(String, String), String> {
        let device = crate::ml::resolve_device(crate::ml::DeviceSlot::Translation, self.use_gpu, self.gpu_device);
        let mut slot = self.slot.lock().map_err(|_| "translator lock poisoned".to_string())?;
        ensure_madlad_loaded(&mut slot, &self.dir, &device)?;
        let (_, model) = slot.loaded.as_mut().ok_or("MADLAD not loaded")?;
        let translated = model.translate(text, tgt)?;
        Ok((translated, crate::ml::device_label(&device).to_string()))
    }
}

fn ensure_madlad_loaded(
    slot: &mut MadladSlot,
    dir: &std::path::Path,
    device: &candle_core::Device,
) -> Result<(), String> {
    if slot.loaded.as_ref().is_some_and(|(d, _)| d == dir) {
        return Ok(());
    }
    if let Some((d, err)) = &slot.last_error
        && d == dir
        && !crate::stt::models::is_model_id_installed(crate::stt::models::MADLAD_MODEL_ID)
    {
        return Err(err.clone());
    }
    tracing::info!("Chargement de MADLAD-400 depuis {} sur {}...", dir.display(), crate::ml::device_label(device));
    let started = Instant::now();
    match MadladTranslator::load(dir, device) {
        Ok(model) => {
            tracing::info!("MADLAD-400 prêt en {:.1}s.", started.elapsed().as_secs_f32());
            slot.loaded = Some((dir.to_path_buf(), model));
            slot.last_error = None;
            Ok(())
        }
        Err(e) => {
            let msg = format!(
                "modèle MADLAD-400 non installé ou illisible ({e}). Installez-le depuis l'onglet Modèles."
            );
            slot.last_error = Some((dir.to_path_buf(), msg.clone()));
            Err(msg)
        }
    }
}

fn primary(code: &str) -> String {
    code.trim()
        .to_ascii_lowercase()
        .split(['-', '_'])
        .next()
        .unwrap_or_default()
        .to_string()
}

fn language_name(code: &str) -> String {
    match primary(code).as_str() {
        "fr" => "French",
        "en" => "English",
        "es" => "Spanish",
        "de" => "German",
        "it" => "Italian",
        "pt" => "Portuguese",
        "zh" => "Chinese",
        "ja" => "Japanese",
        "ko" => "Korean",
        "ru" => "Russian",
        "ar" => "Arabic",
        other => return other.to_string(),
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn service(engine: &str) -> TranslationService {
        TranslationService::new(
            true,
            "fr".into(),
            "en".into(),
            engine.into(),
            "https://api.openai.com/v1/chat/completions".into(),
            "m".into(),
            None,
            PathBuf::from("models/__missing_madlad__"),
        )
    }

    #[tokio::test]
    async fn same_language_is_passed_through() {
        let res = service(ENGINE_MADLAD).translate("Bonjour", "fr", "fr-FR").await;
        assert_eq!(res.translated_text, "Bonjour");
        assert_eq!(res.engine_used, "Pass-Through");
    }

    #[tokio::test]
    async fn remote_llm_endpoint_is_refused() {
        let svc = service(ENGINE_LOCAL_LLM);
        let err = svc.translate_via_local_llm("Bonjour", "en").await.unwrap_err();
        assert!(err.contains("locale"), "{err}");
    }

    #[tokio::test]
    async fn missing_madlad_model_falls_back_to_source_text() {
        let res = service(ENGINE_MADLAD).translate("Bonjour", "fr", "en").await;
        assert_eq!(res.translated_text, "Bonjour");
        assert!(res.engine_used.contains("indisponible"));
    }

    /// Real model test — run with `cargo test -- --ignored` after installing MADLAD-400.
    #[tokio::test]
    #[ignore]
    async fn madlad_translates_french_to_english() {
        let dir = PathBuf::from("models/madlad400-3b-mt");
        let mut svc = service(ENGINE_MADLAD);
        svc.madlad_dir = dir;
        let res = svc
            .translate("Bonjour, je suis en direct et tout fonctionne parfaitement.", "fr", "en")
            .await;
        println!("{} -> {} ({}, {} ms)", res.original_text, res.translated_text, res.engine_used, res.duration_ms);
        assert!(res.engine_used.starts_with("MADLAD"), "{}", res.engine_used);
        let lower = res.translated_text.to_lowercase();
        assert!(lower.contains("hello") || lower.contains("good"), "{}", res.translated_text);
        assert!(lower.contains("live"), "{}", res.translated_text);

        // Multi-sentence input must not get an invented continuation.
        let res = svc
            .translate(
                "Bonjour à tous. Aujourd'hui, nous testons la traduction en direct pour le streaming. La qualité du son est très importante.",
                "fr",
                "en",
            )
            .await;
        println!("-> {}", res.translated_text);
        let lower = res.translated_text.to_lowercase();
        assert!(lower.contains("important"), "{}", res.translated_text);
        assert!(!lower.contains("game"), "hallucinated continuation: {}", res.translated_text);
        assert!(res.translated_text.split_whitespace().count() < 30, "{}", res.translated_text);
    }
}
