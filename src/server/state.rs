use crate::config::AppConfig;
use crate::pipeline::{StreamingWorker, Utterance};
use crate::stt::{DownloadStatus, GpuDeviceInfo, WhisperEngine};
use crate::translate::TranslationService;
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, watch, Mutex, RwLock};

static EVENT_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_event_id(prefix: &str, millis: i64) -> String {
    format!("{prefix}_{millis}_{}", EVENT_COUNTER.fetch_add(1, Ordering::Relaxed))
}

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
    /// Partial (in-progress) sentence from a streaming engine; replaced by later events.
    #[serde(default)]
    pub is_streaming: bool,
}

impl PipelineEvent {
    #[allow(clippy::too_many_arguments)]
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
            id: next_event_id("evt", now.timestamp_millis()),
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
            is_streaming: false,
        }
    }

    pub fn new_streaming(
        original_partial: String,
        translated_partial: String,
        speaker: Option<String>,
        source_lang: String,
        target_lang: String,
        stt_ms: u128,
        model_used: String,
    ) -> Self {
        let now = Local::now();
        Self {
            id: next_event_id("stream", now.timestamp_millis()),
            timestamp: now.format("%H:%M:%S").to_string(),
            speaker,
            original_text: original_partial,
            translated_text: translated_partial,
            source_lang,
            target_lang,
            stt_duration_ms: stt_ms,
            translation_duration_ms: 0,
            total_duration_ms: stt_ms,
            model_used,
            is_streaming: true,
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
    pub download_error: Option<String>,
    pub active_model_id: String,
    pub hardware_backend: String,
    pub use_gpu: bool,
    pub gpu_backend: String,
    pub gpu_device: i32,
    pub flash_attn: bool,
    pub gpu_devices: Vec<GpuDeviceInfo>,
    pub compiled_gpu_backends: Vec<String>,
    pub translation_engine: String,
    pub streaming_engine: Option<String>,
    pub engine_message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamSend {
    Sent,
    /// The worker's buffer is full: it is slower than real time.
    Lagging,
    /// No streaming model loaded (yet).
    NoWorker,
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
    /// Active Kyutai / Hibiki worker (when a streaming engine is selected).
    pub streaming_worker: Arc<std::sync::Mutex<Option<StreamingWorker>>>,
    /// Serialises model activations (download completion, GUI and HTTP requests).
    pub activation_lock: Arc<Mutex<()>>,
    /// Last human-readable message about engine activation (shown in the UIs).
    pub engine_message: Arc<std::sync::Mutex<String>>,
    pub download_status: Arc<RwLock<DownloadStatus>>,
    pub translation_service: Arc<RwLock<TranslationService>>,
    utterance_tx: mpsc::UnboundedSender<Utterance>,
    utterance_rx: Arc<std::sync::Mutex<Option<mpsc::UnboundedReceiver<Utterance>>>>,
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
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
        let (utterance_tx, utterance_rx) = mpsc::unbounded_channel();
        Self {
            event_tx,
            history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            rms_rx,
            is_recording,
            is_shutting_down: Arc::new(AtomicBool::new(false)),
            audio_device,
            config: Arc::new(RwLock::new(config)),
            whisper_engine,
            streaming_worker: Arc::new(std::sync::Mutex::new(None)),
            activation_lock: Arc::new(Mutex::new(())),
            engine_message: Arc::new(std::sync::Mutex::new(String::new())),
            download_status,
            translation_service,
            utterance_tx,
            utterance_rx: Arc::new(std::sync::Mutex::new(Some(utterance_rx))),
        }
    }

    /// Synchronously stops audio ingestion, waits for any in-flight GPU inference to finish,
    /// and deallocates Whisper / Candle GPU resources before AppKit / C `exit()` runs.
    pub fn shutdown_sync(&self) {
        self.is_shutting_down.store(true, Ordering::SeqCst);
        self.is_recording.store(false, Ordering::SeqCst);
        if let Ok(mut slot) = self.streaming_worker.lock() {
            slot.take();
        }
        for _ in 0..400 {
            if let Ok(mut engine) = self.whisper_engine.try_lock() {
                engine.shutdown();
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        tracing::warn!("Whisper toujours occupé après 2 s : arrêt sans libération explicite.");
    }

    pub(crate) fn take_utterance_receiver(&self) -> Option<mpsc::UnboundedReceiver<Utterance>> {
        self.utterance_rx.lock().ok().and_then(|mut r| r.take())
    }

    pub fn submit_utterance(&self, utterance: Utterance) {
        let _ = self.utterance_tx.send(utterance);
    }

    /// Forward 16 kHz audio to the streaming worker without blocking.
    pub fn send_to_streaming_worker(&self, pcm16: Vec<f32>) -> StreamSend {
        match self.streaming_worker.lock() {
            Ok(slot) => match slot.as_ref() {
                Some(w) if w.try_send(pcm16) => StreamSend::Sent,
                Some(_) => StreamSend::Lagging,
                None => StreamSend::NoWorker,
            },
            Err(_) => StreamSend::NoWorker,
        }
    }

    pub fn set_engine_message(&self, msg: impl Into<String>) {
        let msg = msg.into();
        tracing::info!("{}", msg);
        if let Ok(mut m) = self.engine_message.lock() {
            *m = msg;
        }
    }

    pub fn engine_message(&self) -> String {
        self.engine_message.lock().map(|m| m.clone()).unwrap_or_default()
    }

    pub async fn add_event(&self, event: PipelineEvent) {
        let mut history = self.history.write().await;
        if history.len() >= 100 {
            history.pop_front();
        }
        history.push_back(event.clone());
        drop(history);
        let _ = self.event_tx.send(event);
    }

    pub fn subscribe_events(&self) -> broadcast::Receiver<PipelineEvent> {
        self.event_tx.subscribe()
    }

    pub async fn get_status(&self) -> SystemStatus {
        let current_volume = *self.rms_rx.borrow();
        let total = self.history.read().await.len();
        let cfg = self.config.read().await.clone();
        let dl = self.download_status.read().await.clone();
        let hw_label = crate::stt::active_hardware_summary(
            cfg.stt.use_gpu,
            &cfg.stt.gpu_backend,
            cfg.stt.gpu_device,
        );
        let streaming_engine = self
            .streaming_worker
            .lock()
            .ok()
            .and_then(|w| w.as_ref().map(|w| format!("{:?} ({})", w.kind, w.device_label)));

        SystemStatus {
            is_recording: self.is_recording.load(Ordering::Relaxed),
            audio_device: self.audio_device.clone(),
            stt_model: cfg.stt.model_name.clone(),
            stt_engine: cfg.stt.engine_type.clone(),
            source_lang: cfg.translation.source_lang.clone(),
            target_lang: cfg.translation.target_lang.clone(),
            current_volume,
            total_transcriptions: total,
            is_downloading: dl.is_downloading,
            download_percent: dl.percent,
            download_message: dl.status_message.clone(),
            download_speed_mb_s: dl.speed_mb_s,
            download_error: dl.error.clone(),
            active_model_id: cfg.stt.model_id.clone(),
            hardware_backend: hw_label,
            use_gpu: cfg.stt.use_gpu,
            gpu_backend: cfg.stt.gpu_backend.clone(),
            gpu_device: cfg.stt.gpu_device,
            flash_attn: cfg.stt.flash_attn,
            gpu_devices: crate::stt::list_gpu_devices(),
            compiled_gpu_backends: crate::stt::compiled_gpu_backends()
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            translation_engine: cfg.translation.engine.clone(),
            streaming_engine,
            engine_message: self.engine_message(),
        }
    }
}
