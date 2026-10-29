use crate::config::AppConfig;
use crate::stt::{DownloadStatus, WhisperEngine};
use crate::translate::TranslationService;
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, watch, Mutex, RwLock};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineEvent {
    pub id: String,
    pub timestamp: String,
    pub speaker: Option<String>,
    pub original_text: String,
    pub translated_text: String,
    pub source_lang: String,
    pub target_lang: String,
    pub stt_duration_ms: u128,
    pub translation_duration_ms: u128,
    pub total_duration_ms: u128,
    pub model_used: String,
}

impl PipelineEvent {
    pub fn new(
        original: String,
        translated: String,
        speaker: Option<String>,
        source_lang: String,
        target_lang: String,
        stt_ms: u128,
        trans_ms: u128,
        model_used: String,
    ) -> Self {
        let now = Local::now();
        Self {
            id: format!("evt_{}", now.timestamp_millis()),
            timestamp: now.format("%H:%M:%S").to_string(),
            speaker,
            original_text: original,
            translated_text: translated,
            source_lang,
            target_lang,
            stt_duration_ms: stt_ms,
            translation_duration_ms: trans_ms,
            total_duration_ms: stt_ms + trans_ms,
            model_used,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub is_recording: bool,
    pub audio_device: String,
    pub stt_model: String,
    pub stt_engine: String,
    pub source_lang: String,
    pub target_lang: String,
    pub current_volume: f32,
    pub total_transcriptions: usize,
    pub is_downloading: bool,
    pub download_percent: f32,
    pub download_message: String,
    pub download_speed_mb_s: f32,
    pub active_model_id: String,
}

#[derive(Clone)]
pub struct AppState {
    pub event_tx: broadcast::Sender<PipelineEvent>,
    pub history: Arc<RwLock<VecDeque<PipelineEvent>>>,
    pub rms_rx: watch::Receiver<f32>,
    pub is_recording: Arc<AtomicBool>,
    pub is_shutting_down: Arc<AtomicBool>,
    pub audio_device: String,
    pub config: Arc<RwLock<AppConfig>>,
    pub whisper_engine: Arc<Mutex<WhisperEngine>>,
    pub download_status: Arc<RwLock<DownloadStatus>>,
    pub translation_service: Arc<RwLock<TranslationService>>,
}

impl AppState {
    pub fn new(
        event_tx: broadcast::Sender<PipelineEvent>,
        rms_rx: watch::Receiver<f32>,
        is_recording: Arc<AtomicBool>,
        audio_device: String,
        config: AppConfig,
        whisper_engine: Arc<Mutex<WhisperEngine>>,
        download_status: Arc<RwLock<DownloadStatus>>,
        translation_service: Arc<RwLock<TranslationService>>,
    ) -> Self {
        Self {
            event_tx,
            history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            rms_rx,
            is_recording,
            is_shutting_down: Arc::new(AtomicBool::new(false)),
            audio_device,
            config: Arc::new(RwLock::new(config)),
            whisper_engine,
            download_status,
            translation_service,
        }
    }

    /// Synchronously stops audio ingestion, waits for any in-flight Metal inference to finish,
    /// and deallocates Whisper Metal GPU residency sets before AppKit / C `exit()` runs.
    pub fn shutdown_sync(&self) {
        self.is_shutting_down.store(true, Ordering::SeqCst);
        self.is_recording.store(false, Ordering::SeqCst);
        for _ in 0..400 {
            if let Ok(mut engine) = self.whisper_engine.try_lock() {
                engine.shutdown();
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    pub async fn add_event(&self, event: PipelineEvent) {
        let mut history = self.history.write().await;
        if history.len() >= 100 {
            history.pop_front();
        }
        history.push_back(event.clone());
        let _ = self.event_tx.send(event);
    }

    pub fn subscribe_events(&self) -> broadcast::Receiver<PipelineEvent> {
        self.event_tx.subscribe()
    }

    pub async fn get_status(&self) -> SystemStatus {
        let current_volume = *self.rms_rx.borrow();
        let history = self.history.read().await;
        let cfg = self.config.read().await;
        let dl = self.download_status.read().await;

        SystemStatus {
            is_recording: self.is_recording.load(Ordering::Relaxed),
            audio_device: self.audio_device.clone(),
            stt_model: cfg.stt.model_name.clone(),
            stt_engine: cfg.stt.engine_type.clone(),
            source_lang: cfg.translation.source_lang.clone(),
            target_lang: cfg.translation.target_lang.clone(),
            current_volume,
            total_transcriptions: history.len(),
            is_downloading: dl.is_downloading,
            download_percent: dl.percent,
            download_message: dl.status_message.clone(),
            download_speed_mb_s: dl.speed_mb_s,
            active_model_id: cfg.stt.model_id.clone(),
        }
    }
}

