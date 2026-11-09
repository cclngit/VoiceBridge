//! Engine / model management shared by the desktop GUI and the HTTP API, so both
//! interfaces behave identically (download, verification, activation, persistence).

use crate::config::AppConfig;
use crate::pipeline::{start_streaming_worker, StreamingKind};
use crate::server::AppState;
use crate::stt::downloader::{download_model_files, finish_download, try_begin_download};
use crate::stt::models::{
    find_model, is_model_installed, ModelMetadata, ENGINE_HIBIKI, ENGINE_KYUTAI, ENGINE_MADLAD,
    ENGINE_REMOTE, ENGINE_WHISPER, KIND_TRANSLATION,
};
use serde::Serialize;
use std::time::Duration;

#[derive(Debug, Clone, Serialize)]
pub struct ActivationOutcome {
    pub success: bool,
    /// "ready", "downloading", "offline" or "error".
    pub status: String,
    pub message: String,
    pub model_id: String,
}

impl ActivationOutcome {
    fn ready(model_id: &str, message: impl Into<String>) -> Self {
        Self { success: true, status: "ready".into(), message: message.into(), model_id: model_id.into() }
    }
    fn error(model_id: &str, message: impl Into<String>) -> Self {
        Self { success: false, status: "error".into(), message: message.into(), model_id: model_id.into() }
    }
}

impl AppState {
    /// Persist the current configuration to the file selected at startup.
    pub async fn persist_config(&self) {
        let cfg = self.config.read().await.clone();
        let _ = tokio::task::spawn_blocking(move || cfg.save_default()).await;
    }

    /// Activate a catalog model (STT engine, streaming engine or translation model).
    /// Missing local models are downloaded in the background first.
    pub async fn activate_model(
        &self,
        model_id: &str,
        api_url: Option<String>,
        api_key: Option<String>,
    ) -> ActivationOutcome {
        let Some(meta) = find_model(model_id) else {
            return ActivationOutcome::error(model_id, format!("Modèle inconnu : {model_id}"));
        };

        if !meta.is_local_gpu {
            return self.activate_remote_asr(&meta, api_url, api_key).await;
        }

        if !is_model_installed(&meta) {
            if !try_begin_download(&self.download_status, &meta.id, &meta.name).await {
                return ActivationOutcome {
                    success: false,
                    status: "downloading".into(),
                    message: "Un téléchargement est déjà en cours.".into(),
                    model_id: meta.id.clone(),
                };
            }
            let state = self.clone();
            let meta_bg = meta.clone();
            tokio::spawn(async move {
                let res = download_model_files(&meta_bg, &state.download_status).await;
                finish_download(&state.download_status, &res).await;
                match res {
                    Ok(()) => {
                        let outcome = state.activate_installed(&meta_bg).await;
                        state.set_engine_message(outcome.message);
                    }
                    Err(e) => state.set_engine_message(format!("Échec du téléchargement de {} : {e}", meta_bg.name)),
                }
            });
            return ActivationOutcome {
                success: true,
                status: "downloading".into(),
                message: format!(
                    "Téléchargement de {} ({} Mo) lancé — activation automatique à la fin.",
                    meta.name, meta.file_size_mb
                ),
                model_id: meta.id.clone(),
            };
        }

        let outcome = self.activate_installed(&meta).await;
        self.set_engine_message(outcome.message.clone());
        outcome
    }

    /// Activate a local model whose files are present.
    pub async fn activate_installed(&self, meta: &ModelMetadata) -> ActivationOutcome {
        let _guard = self.activation_lock.lock().await;
        if meta.kind == KIND_TRANSLATION {
            return self.activate_madlad(meta).await;
        }
        match meta.default_engine.as_str() {
            ENGINE_WHISPER => self.activate_whisper(meta).await,
            ENGINE_KYUTAI => self.activate_streaming(meta, StreamingKind::KyutaiStt).await,
            ENGINE_HIBIKI => self.activate_streaming(meta, StreamingKind::Hibiki).await,
            other => ActivationOutcome::error(&meta.id, format!("Moteur non pris en charge : {other}")),
        }
    }

    async fn stop_streaming_worker(&self) {
        let worker = self.streaming_worker.lock().ok().and_then(|mut w| w.take());
        if let Some(worker) = worker {
            // Joining the worker thread may take up to one inference step.
            let _ = tokio::task::spawn_blocking(move || drop(worker)).await;
        }
    }

    async fn activate_whisper(&self, meta: &ModelMetadata) -> ActivationOutcome {
        let (use_gpu, backend, dev, flash, lang) = {
            let cfg = self.config.read().await;
            (cfg.stt.use_gpu, cfg.stt.gpu_backend.clone(), cfg.stt.gpu_device, cfg.stt.flash_attn, cfg.stt.language.clone())
        };
        let path = meta.default_model_name.clone();
        let whisper = self.whisper_engine.clone();
        let reload = tokio::task::spawn_blocking(move || {
            let mut engine = whisper.blocking_lock();
            engine.set_language(&lang);
            engine
                .reload_model_with_gpu_options(&path, use_gpu, &backend, dev, flash)
                .map(|_| engine.active_hardware_label().to_string())
                .map_err(|e| e.to_string())
        })
        .await
        .map_err(|e| e.to_string())
        .and_then(|r| r);

        match reload {
            Ok(hw) => {
                self.stop_streaming_worker().await;
                {
                    let mut cfg = self.config.write().await;
                    cfg.stt.engine_type = ENGINE_WHISPER.into();
                    cfg.stt.model_id = meta.id.clone();
                    cfg.stt.model_path = meta.default_model_name.clone();
                    if let Some(url) = &meta.download_url {
                        cfg.stt.model_url = url.clone();
                    }
                    cfg.stt.model_name = meta.name.clone();
                }
                self.persist_config().await;
                ActivationOutcome::ready(&meta.id, format!("{} activé ({hw}).", meta.name))
            }
            Err(e) => ActivationOutcome::error(&meta.id, format!("Erreur de chargement Whisper : {e}")),
        }
    }

    async fn activate_streaming(&self, meta: &ModelMetadata, kind: StreamingKind) -> ActivationOutcome {
        let (use_gpu, dev) = {
            let cfg = self.config.read().await;
            (cfg.stt.use_gpu, cfg.stt.gpu_device)
        };
        let mut note = String::new();
        if kind == StreamingKind::Hibiki {
            // Hibiki translates French speech into English only.
            let mut cfg = self.config.write().await;
            if cfg.translation.target_lang != "en" || !matches!(cfg.stt.language.as_str(), "fr" | "auto") {
                note = " Langues réglées sur Français → Anglais (seule paire prise en charge par Hibiki).".into();
            }
            cfg.stt.language = "fr".into();
            cfg.translation.source_lang = "fr".into();
            cfg.translation.target_lang = "en".into();
        } else {
            let cfg = self.config.read().await;
            if !matches!(cfg.stt.language.as_str(), "fr" | "en" | "auto") {
                note = " Attention : Kyutai STT ne reconnaît que le français et l'anglais.".into();
            }
        }

        self.stop_streaming_worker().await;
        self.set_engine_message(format!("Chargement de {}...", meta.name));
        match start_streaming_worker(self.clone(), kind, meta.default_model_name.clone(), use_gpu, dev).await {
            Ok(worker) => {
                let device = worker.device_label.clone();
                if let Ok(mut slot) = self.streaming_worker.lock() {
                    *slot = Some(worker);
                }
                {
                    let mut cfg = self.config.write().await;
                    cfg.stt.engine_type = meta.default_engine.clone();
                    cfg.stt.model_id = meta.id.clone();
                    cfg.stt.model_path = meta.default_model_name.clone();
                    cfg.stt.model_name = meta.name.clone();
                    if let Some(url) = &meta.download_url {
                        cfg.stt.model_url = url.clone();
                    }
                }
                self.persist_config().await;
                let mut msg = format!("{} actif en streaming temps réel ({device}).", meta.name);
                if device == "CPU" {
                    msg.push_str(" Sans GPU Metal/CUDA, le temps réel n'est pas garanti.");
                }
                msg.push_str(&note);
                ActivationOutcome::ready(&meta.id, msg)
            }
            Err(e) => ActivationOutcome::error(&meta.id, format!("Impossible de charger {} : {e}", meta.name)),
        }
    }

    async fn activate_madlad(&self, meta: &ModelMetadata) -> ActivationOutcome {
        let loader = {
            let mut svc = self.translation_service.write().await;
            let mut tcfg = self.config.read().await.translation.clone();
            tcfg.engine = ENGINE_MADLAD.into();
            tcfg.enabled = true;
            tcfg.madlad_model_dir = meta.default_model_name.clone();
            let (use_gpu, dev) = {
                let c = self.config.read().await;
                (c.stt.use_gpu, c.stt.gpu_device)
            };
            svc.apply_config(&tcfg, use_gpu, dev);
            svc.madlad_loader()
        };
        let res = tokio::task::spawn_blocking(move || loader.preload())
            .await
            .map_err(|e| e.to_string())
            .and_then(|r| r);
        match res {
            Ok(device) => {
                {
                    let mut cfg = self.config.write().await;
                    cfg.translation.engine = ENGINE_MADLAD.into();
                    cfg.translation.enabled = true;
                    cfg.translation.madlad_model_dir = meta.default_model_name.clone();
                }
                self.persist_config().await;
                ActivationOutcome::ready(&meta.id, format!("Traduction locale {} active ({device}).", meta.name))
            }
            Err(e) => ActivationOutcome::error(&meta.id, e),
        }
    }

    async fn activate_remote_asr(
        &self,
        meta: &ModelMetadata,
        api_url: Option<String>,
        api_key: Option<String>,
    ) -> ActivationOutcome {
        let url = api_url
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| meta.default_api_url.clone());
        if let Err(e) = crate::netpolicy::ensure_local_endpoint(&url) {
            return ActivationOutcome::error(&meta.id, e);
        }
        let api_key = api_key.filter(|k| !k.trim().is_empty() && k != crate::config::REDACTED_SECRET);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap_or_default();
        let mut req = client.get(&url);
        if let Some(key) = &api_key {
            req = req.bearer_auth(key);
        }
        if let Err(e) = req.send().await {
            return ActivationOutcome {
                success: false,
                status: "offline".into(),
                message: format!(
                    "Serveur ASR local injoignable sur {url} ({e}). Démarrez-le (onglet Modèles ou scripts/start_local_asr) ou gardez Whisper local."
                ),
                model_id: meta.id.clone(),
            };
        }
        self.stop_streaming_worker().await;
        {
            let mut cfg = self.config.write().await;
            cfg.stt.engine_type = ENGINE_REMOTE.into();
            cfg.stt.model_id = meta.id.clone();
            cfg.stt.model_name = meta.default_model_name.clone();
            cfg.stt.api_url = url.clone();
            if api_key.is_some() {
                cfg.stt.api_key = api_key;
            }
        }
        self.persist_config().await;
        let msg = format!("Modèle {} actif sur le serveur local {url}.", meta.name);
        self.set_engine_message(msg.clone());
        ActivationOutcome::ready(&meta.id, msg)
    }

    /// Point the ASR engine at a local server URL without catalog metadata.
    pub async fn use_local_asr_server(&self, url: String, api_key: Option<String>) -> ActivationOutcome {
        match find_model("custom_api") {
            Some(meta) => self.activate_remote_asr(&meta, Some(url), api_key).await,
            None => ActivationOutcome::error("custom_api", "modèle custom_api absent du catalogue"),
        }
    }

    /// Change source / target languages everywhere.
    pub async fn set_languages(&self, source: &str, target: &str) {
        {
            let mut cfg = self.config.write().await;
            cfg.translation.source_lang = source.to_string();
            cfg.translation.target_lang = target.to_string();
            cfg.stt.language = source.to_string();
            cfg.sanitize();
        }
        let cfg = self.config.read().await.clone();
        self.translation_service
            .write()
            .await
            .set_languages(cfg.translation.source_lang.clone(), cfg.translation.target_lang.clone());
        self.whisper_engine.lock().await.set_language(&cfg.stt.language);
        self.persist_config().await;
        if cfg.stt.engine_type == ENGINE_HIBIKI && (cfg.translation.target_lang != "en" || cfg.stt.language != "fr") {
            self.set_engine_message("Hibiki ne traduit que du français vers l'anglais : la langue choisie est ignorée tant qu'Hibiki est actif.");
        }
    }

    /// Change the translation engine ("madlad", "local_llm" or "off").
    pub async fn set_translation_engine(&self, engine: &str) -> ActivationOutcome {
        match engine {
            ENGINE_MADLAD => {
                if let Some(meta) = find_model(crate::stt::models::MADLAD_MODEL_ID) {
                    return Box::pin(self.activate_model(&meta.id, None, None)).await;
                }
                ActivationOutcome::error(engine, "MADLAD-400 absent du catalogue")
            }
            crate::translate::engine::ENGINE_LOCAL_LLM | crate::translate::engine::ENGINE_OFF => {
                {
                    let mut cfg = self.config.write().await;
                    cfg.translation.engine = engine.to_string();
                    cfg.translation.enabled = engine != crate::translate::engine::ENGINE_OFF;
                }
                self.sync_translation_service().await;
                self.persist_config().await;
                let msg = if engine == crate::translate::engine::ENGINE_OFF {
                    "Traduction désactivée.".to_string()
                } else {
                    let url = self.config.read().await.translation.local_api_url.clone();
                    format!("Traduction via LLM local ({url}).")
                };
                self.set_engine_message(msg.clone());
                ActivationOutcome::ready(engine, msg)
            }
            other => ActivationOutcome::error(other, format!("Moteur de traduction inconnu : {other}")),
        }
    }

    async fn sync_translation_service(&self) {
        let cfg = self.config.read().await.clone();
        self.translation_service
            .write()
            .await
            .apply_config(&cfg.translation, cfg.stt.use_gpu, cfg.stt.gpu_device);
    }

    /// Apply GPU / CPU settings: reloads Whisper and restarts the streaming engine if any.
    pub async fn apply_hardware(&self, use_gpu: bool, gpu_backend: String, gpu_device: i32, flash_attn: bool, n_threads: usize) {
        {
            let mut cfg = self.config.write().await;
            cfg.stt.use_gpu = use_gpu;
            cfg.stt.gpu_backend = gpu_backend.clone();
            cfg.stt.gpu_device = gpu_device;
            cfg.stt.flash_attn = flash_attn;
            cfg.stt.n_threads = n_threads.max(1);
        }
        self.persist_config().await;
        self.sync_translation_service().await;
        let whisper = self.whisper_engine.clone();
        let res = tokio::task::spawn_blocking(move || {
            whisper
                .blocking_lock()
                .apply_hardware_settings(use_gpu, &gpu_backend, gpu_device, flash_attn, n_threads)
                .map_err(|e| e.to_string())
        })
        .await;
        if let Ok(Err(e)) = res {
            self.set_engine_message(format!("Erreur lors du changement de périphérique : {e}"));
        }
        self.restart_streaming_if_active().await;
    }

    async fn restart_streaming_if_active(&self) {
        let (engine, model_id) = {
            let cfg = self.config.read().await;
            (cfg.stt.engine_type.clone(), cfg.stt.model_id.clone())
        };
        if (engine == ENGINE_KYUTAI || engine == ENGINE_HIBIKI)
            && let Some(meta) = find_model(&model_id)
        {
            let outcome = self.activate_installed(&meta).await;
            self.set_engine_message(outcome.message);
        }
    }

    /// Apply a full configuration received from the web UI.
    pub async fn apply_config(&self, mut new_config: AppConfig) -> Result<(), String> {
        let previous = self.config.read().await.clone();
        new_config.restore_redacted_secrets(&previous);
        // Keep STT and translation source languages in sync.
        let src = if !new_config.stt.language.is_empty() {
            new_config.stt.language.clone()
        } else {
            new_config.translation.source_lang.clone()
        };
        new_config.stt.language = src.clone();
        new_config.translation.source_lang = src;
        for url in [&new_config.stt.api_url, &new_config.translation.local_api_url] {
            crate::netpolicy::ensure_local_endpoint(url)?;
        }
        new_config.sanitize();
        // Engine / model switches go through activation, not raw config edits.
        new_config.stt.engine_type = previous.stt.engine_type.clone();
        new_config.stt.model_id = previous.stt.model_id.clone();
        new_config.stt.model_path = previous.stt.model_path.clone();
        new_config.stt.model_name = previous.stt.model_name.clone();
        new_config.stt.model_url = previous.stt.model_url.clone();

        let hw_changed = previous.stt.use_gpu != new_config.stt.use_gpu
            || previous.stt.gpu_backend != new_config.stt.gpu_backend
            || previous.stt.gpu_device != new_config.stt.gpu_device
            || previous.stt.flash_attn != new_config.stt.flash_attn;
        let translation_engine_changed = previous.translation.engine != new_config.translation.engine;

        *self.config.write().await = new_config.clone();
        self.sync_translation_service().await;
        {
            let mut engine = self.whisper_engine.lock().await;
            engine.set_language(&new_config.stt.language);
            engine.set_n_threads(new_config.stt.n_threads);
        }
        self.persist_config().await;

        if hw_changed {
            self.apply_hardware(
                new_config.stt.use_gpu,
                new_config.stt.gpu_backend.clone(),
                new_config.stt.gpu_device,
                new_config.stt.flash_attn,
                new_config.stt.n_threads,
            )
            .await;
        }
        if translation_engine_changed {
            let outcome = self.set_translation_engine(&new_config.translation.engine).await;
            if !outcome.success && outcome.status != "downloading" {
                return Err(outcome.message);
            }
        }
        Ok(())
    }

    /// At startup, restore a saved streaming engine (Kyutai / Hibiki) or fall back to Whisper.
    pub async fn restore_engine_on_startup(&self) {
        let (engine, model_id, translation_engine) = {
            let cfg = self.config.read().await;
            (cfg.stt.engine_type.clone(), cfg.stt.model_id.clone(), cfg.translation.engine.clone())
        };
        if engine == ENGINE_KYUTAI || engine == ENGINE_HIBIKI {
            match find_model(&model_id).filter(is_model_installed) {
                Some(meta) => {
                    let outcome = self.activate_installed(&meta).await;
                    if !outcome.success {
                        self.fallback_to_whisper(&outcome.message).await;
                    } else {
                        self.set_engine_message(outcome.message);
                    }
                }
                None => self.fallback_to_whisper("modèle de streaming non installé").await,
            }
        }
        if translation_engine == ENGINE_MADLAD {
            if crate::stt::models::is_model_id_installed(crate::stt::models::MADLAD_MODEL_ID) {
                // Load now so the first sentence is not delayed by model loading.
                let loader = self.translation_service.read().await.madlad_loader();
                match tokio::task::spawn_blocking(move || loader.preload()).await {
                    Ok(Ok(device)) => tracing::info!("Traduction MADLAD-400 prête ({device})."),
                    Ok(Err(e)) => self.set_engine_message(format!("Traduction indisponible : {e}")),
                    Err(e) => tracing::error!("Préchargement MADLAD interrompu : {e}"),
                }
            } else {
                self.set_engine_message(
                    "Traduction : installez MADLAD-400 (onglet Modèles) pour traduire 100 % en local. En attendant, le texte source est affiché.",
                );
            }
        }
    }

    async fn fallback_to_whisper(&self, reason: &str) {
        tracing::warn!("Moteur de streaming indisponible ({reason}) : retour à Whisper local.");
        let mut cfg = self.config.write().await;
        cfg.stt.engine_type = ENGINE_WHISPER.into();
        let whisper_path = self.whisper_engine.lock().await.model_path().to_string();
        cfg.stt.model_path = whisper_path.clone();
        if let Some(meta) = crate::stt::models::find_model_by_path(&whisper_path) {
            cfg.stt.model_id = meta.id;
            cfg.stt.model_name = meta.name;
        }
        drop(cfg);
        self.persist_config().await;
        self.set_engine_message(format!("Moteur de streaming indisponible ({reason}) : Whisper local actif."));
    }
}
