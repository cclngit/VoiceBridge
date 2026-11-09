//! Real-time audio pipeline.
//!
//! ```text
//! CPAL callback ──mono, device rate──▶ router task ──resample 16 kHz──┬─▶ VAD ─▶ segment worker (Whisper / local ASR server)
//!                                                                      └─▶ streaming worker thread (Kyutai STT / Hibiki)
//! segment worker / streaming worker ──Utterance──▶ translation worker ─▶ history + SSE + GUI
//! ```
//!
//! * Utterances are processed strictly in order by single workers, so subtitles never
//!   arrive out of order.
//! * Heavy inference (Whisper, Kyutai, Hibiki, MADLAD) never runs on the async runtime:
//!   it uses `spawn_blocking` or dedicated threads.

use crate::audio::StreamingResampler;
use crate::server::{AppState, PipelineEvent};
use crate::stt::models::{ENGINE_HIBIKI, ENGINE_KYUTAI, ENGINE_WHISPER};
use crate::vad::SpeechSegmenter;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

pub const PIPELINE_RATE: u32 = 16_000;

/// A finalised utterance waiting for translation and publication.
#[derive(Debug, Clone)]
pub struct Utterance {
    pub text: String,
    pub speaker: Option<String>,
    /// Detected or configured spoken language ("auto" when unknown).
    pub language: String,
    pub stt_ms: u128,
    pub model_name: String,
    /// Already translated text (Hibiki produces the translation directly).
    pub pre_translated: Option<String>,
}

/// Start the pipeline tasks. `source_rate` is the native rate of the capture device.
pub fn spawn(state: AppState, mut audio_rx: mpsc::UnboundedReceiver<Vec<f32>>, source_rate: u32) {
    let (segment_tx, segment_rx) = mpsc::channel::<Vec<f32>>(32);
    let utterance_rx = state.take_utterance_receiver();

    tokio::spawn(segment_worker(state.clone(), segment_rx));
    if let Some(rx) = utterance_rx {
        tokio::spawn(translation_worker(state.clone(), rx));
    }

    tokio::spawn(async move {
        let mut resampler = StreamingResampler::new(source_rate, PIPELINE_RATE);
        let (vad_cfg, mut engine) = {
            let cfg = state.config.read().await;
            (cfg.audio.clone(), cfg.stt.engine_type.clone())
        };
        let mut segmenter = SpeechSegmenter::new(
            PIPELINE_RATE,
            vad_cfg.vad_threshold,
            vad_cfg.vad_silence_duration_ms,
            vad_cfg.vad_min_speech_ms,
            vad_cfg.vad_max_speech_ms,
        );
        let mut last_config_check = Instant::now();
        let mut dropped_frames = 0usize;
        tracing::info!(
            "Pipeline audio démarré ({} Hz → {} Hz, 100 % en mémoire).",
            source_rate,
            PIPELINE_RATE
        );

        while let Some(chunk) = audio_rx.recv().await {
            if state.is_shutting_down.load(Ordering::Relaxed) {
                break;
            }
            if last_config_check.elapsed() > Duration::from_millis(300) {
                let cfg = state.config.read().await;
                segmenter.update_params(
                    cfg.audio.vad_threshold,
                    cfg.audio.vad_silence_duration_ms,
                    cfg.audio.vad_min_speech_ms,
                    cfg.audio.vad_max_speech_ms,
                );
                engine = cfg.stt.engine_type.clone();
                last_config_check = Instant::now();
            }
            if !state.is_recording.load(Ordering::Relaxed) {
                resampler.reset();
                continue;
            }
            let pcm16 = resampler.process(&chunk);
            if pcm16.is_empty() {
                continue;
            }

            if engine == ENGINE_KYUTAI || engine == ENGINE_HIBIKI {
                // While a model is (re)loading there is no worker: audio is simply skipped.
                if state.send_to_streaming_worker(pcm16) == crate::server::state::StreamSend::Lagging {
                    dropped_frames += 1;
                    if dropped_frames % 50 == 1 {
                        tracing::warn!(
                            "Le moteur de streaming ne suit pas le temps réel (GPU trop lent ou modèle en cours de chargement) : audio ignoré."
                        );
                    }
                }
                continue;
            }

            if let Some(segment) = segmenter.process_chunk(&pcm16)
                && segment_tx.send(segment).await.is_err() {
                    break;
                }
        }
        tracing::info!("Pipeline audio arrêté.");
    });
}

/// Sequential speech-to-text for VAD segments (Whisper local or local ASR server).
async fn segment_worker(state: AppState, mut rx: mpsc::Receiver<Vec<f32>>) {
    let remote = crate::stt::RemoteAsrClient::new();
    while let Some(segment) = rx.recv().await {
        if state.is_shutting_down.load(Ordering::Relaxed) {
            break;
        }
        let cfg = state.config.read().await.clone();
        let language = effective_source_lang(&cfg);

        let result = if cfg.stt.engine_type == ENGINE_WHISPER {
            transcribe_whisper(&state, segment, &language).await
        } else {
            match remote.transcribe(&segment, &cfg.stt).await {
                Ok(res) => Ok(res),
                Err(err) => {
                    if state.is_shutting_down.load(Ordering::Relaxed) {
                        break;
                    }
                    tracing::warn!(
                        "Serveur ASR local injoignable ({}), bascule sur Whisper local.",
                        err
                    );
                    transcribe_whisper(&state, segment, &language).await
                }
            }
        };

        match result {
            Ok(res) => {
                let trimmed = res.text.trim();
                if is_blank_transcript(trimmed) {
                    continue;
                }
                tracing::info!("🗣️  [STT {} ({} ms)] {}", res.model_name, res.duration_ms, trimmed);
                state.submit_utterance(Utterance {
                    text: trimmed.to_string(),
                    speaker: res.speaker,
                    language: res.language,
                    stt_ms: res.duration_ms,
                    model_name: res.model_name,
                    pre_translated: None,
                });
            }
            Err(err) => tracing::error!("Erreur de reconnaissance vocale : {}", err),
        }
    }
}

async fn transcribe_whisper(
    state: &AppState,
    segment: Vec<f32>,
    language: &str,
) -> Result<crate::stt::engine::TranscriptionResult, String> {
    let whisper = state.whisper_engine.clone();
    let language = language.to_string();
    tokio::task::spawn_blocking(move || {
        let mut engine = whisper.blocking_lock();
        engine
            .transcribe(&segment, Some(&language), false)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("tâche Whisper interrompue : {e}"))?
}

fn is_blank_transcript(text: &str) -> bool {
    text.is_empty()
        || ["[blank_audio]", "[silence]", "(silence)", "[music]", "[musique]"]
            .iter()
            .any(|t| text.eq_ignore_ascii_case(t))
}

pub fn effective_source_lang(cfg: &crate::config::AppConfig) -> String {
    let lang = if !cfg.stt.language.trim().is_empty() {
        cfg.stt.language.clone()
    } else {
        cfg.translation.source_lang.clone()
    };
    if lang.trim().is_empty() { "auto".to_string() } else { lang }
}

/// Sequential translation + publication of finalised utterances.
async fn translation_worker(state: AppState, mut rx: mpsc::UnboundedReceiver<Utterance>) {
    while let Some(utt) = rx.recv().await {
        if state.is_shutting_down.load(Ordering::Relaxed) {
            break;
        }
        let target_lang = state.config.read().await.translation.target_lang.clone();
        let (translated, trans_ms) = match &utt.pre_translated {
            Some(t) => (t.clone(), 0),
            None => {
                let svc = state.translation_service.read().await;
                let res = svc.translate(&utt.text, &utt.language, &target_lang).await;
                tracing::info!(
                    "🌐 [MT {} {} ms → {}] {}",
                    res.engine_used,
                    res.duration_ms,
                    target_lang,
                    res.translated_text
                );
                (res.translated_text, res.duration_ms)
            }
        };
        let event = PipelineEvent::new(
            utt.text,
            translated,
            utt.speaker,
            utt.language,
            target_lang,
            utt.stt_ms,
            trans_ms,
            utt.model_name,
        );
        state.add_event(event).await;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming workers (Kyutai STT / Hibiki) — one dedicated OS thread each.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingKind {
    KyutaiStt,
    Hibiki,
}

pub struct StreamingWorker {
    pub kind: StreamingKind,
    pub device_label: String,
    tx: std::sync::mpsc::SyncSender<Vec<f32>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl StreamingWorker {
    /// Try to hand 16 kHz audio to the worker without blocking the router.
    pub fn try_send(&self, pcm16: Vec<f32>) -> bool {
        self.tx.try_send(pcm16).is_ok()
    }
}

impl Drop for StreamingWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            // The worker polls the stop flag every 200 ms.
            let _ = t.join();
        }
    }
}

/// Load a streaming model on a dedicated thread and start consuming audio.
/// Resolves once the model is loaded (or failed to load).
pub async fn start_streaming_worker(
    state: AppState,
    kind: StreamingKind,
    model_dir: String,
    use_gpu: bool,
    gpu_device: i32,
) -> Result<StreamingWorker, String> {
    // ~2 s of audio buffered at most (chunks are ~10-20 ms).
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(200);
    let stop = Arc::new(AtomicBool::new(false));
    let (ready_tx, ready_rx) = oneshot::channel::<Result<String, String>>();
    let stop_thread = stop.clone();
    let dir = model_dir;
    let thread = std::thread::Builder::new()
        .name(format!("voicebridge-{kind:?}").to_lowercase())
        .spawn(move || {
            let device = crate::ml::resolve_device(crate::ml::DeviceSlot::Streaming, use_gpu, gpu_device);
            let label = crate::ml::device_label(&device).to_string();
            match kind {
                StreamingKind::KyutaiStt => {
                    match crate::stt::kyutai_engine::KyutaiStt::load(std::path::Path::new(&dir), &device) {
                        Ok(model) => {
                            let _ = ready_tx.send(Ok(label.clone()));
                            run_kyutai(state, model, rx, stop_thread, label);
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                        }
                    }
                }
                StreamingKind::Hibiki => {
                    match crate::translate::HibikiTranslator::load(std::path::Path::new(&dir), &device) {
                        Ok(model) => {
                            let _ = ready_tx.send(Ok(label.clone()));
                            run_hibiki(state, model, rx, stop_thread, label);
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                        }
                    }
                }
            }
        })
        .map_err(|e| format!("impossible de démarrer le thread de streaming : {e}"))?;

    match ready_rx.await {
        Ok(Ok(device_label)) => Ok(StreamingWorker {
            kind,
            device_label,
            tx,
            stop,
            thread: Some(thread),
        }),
        Ok(Err(e)) => {
            let _ = thread.join();
            Err(e)
        }
        Err(_) => Err("le thread de chargement du modèle s'est arrêté".into()),
    }
}

/// Tracks the real-time factor (processing time / audio duration) of a worker.
struct RtfMeter {
    audio_s: f64,
    busy_s: f64,
    last_report: Instant,
}

impl RtfMeter {
    fn new() -> Self {
        Self { audio_s: 0.0, busy_s: 0.0, last_report: Instant::now() }
    }

    fn record(&mut self, audio_samples_16k: usize, busy: Duration, name: &str) {
        self.audio_s += audio_samples_16k as f64 / PIPELINE_RATE as f64;
        self.busy_s += busy.as_secs_f64();
        if self.last_report.elapsed() > Duration::from_secs(30) && self.audio_s > 5.0 {
            let rtf = self.busy_s / self.audio_s;
            if rtf > 0.9 {
                tracing::warn!("{name} : facteur temps réel {:.2} (> 1 = plus lent que la parole).", rtf);
            } else {
                tracing::debug!("{name} : facteur temps réel {:.2}", rtf);
            }
            self.audio_s = 0.0;
            self.busy_s = 0.0;
            self.last_report = Instant::now();
        }
    }
}

fn run_kyutai(
    state: AppState,
    mut model: crate::stt::kyutai_engine::KyutaiStt,
    rx: std::sync::mpsc::Receiver<Vec<f32>>,
    stop: Arc<AtomicBool>,
    device_label: String,
) {
    use crate::stt::kyutai_engine::KyutaiEvent;
    let model_name = format!("Kyutai STT 1B ({device_label})");
    let mut up = StreamingResampler::new(PIPELINE_RATE, crate::stt::kyutai_engine::SAMPLE_RATE);
    let mut words: Vec<String> = Vec::new();
    let mut sentence_started = Instant::now();
    let mut last_audio = Instant::now();
    let mut rtf = RtfMeter::new();
    let delay_ms = (model.audio_delay_s() * 1000.0) as u128;
    tracing::info!("Kyutai STT en écoute sur {}.", device_label);

    let finalize = |words: &mut Vec<String>, started: Instant| {
        if words.is_empty() {
            return;
        }
        let text = words.join(" ");
        words.clear();
        let cfg = state.config.blocking_read();
        let lang = effective_source_lang(&cfg);
        drop(cfg);
        tracing::info!("🗣️  [{}] {}", model_name, text);
        let _ = started;
        state.submit_utterance(Utterance {
            text,
            speaker: None,
            language: lang,
            stt_ms: delay_ms,
            model_name: model_name.clone(),
            pre_translated: None,
        });
    };

    while !stop.load(Ordering::Relaxed) {
        let events = match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(pcm16) => {
                last_audio = Instant::now();
                let t0 = Instant::now();
                let pcm24 = up.process(&pcm16);
                let res = model.push_pcm(&pcm24);
                rtf.record(pcm16.len(), t0.elapsed(), "Kyutai STT");
                res
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Microphone paused: flush the words still inside the model delay.
                if !words.is_empty() && last_audio.elapsed() > Duration::from_millis(800) {
                    up.reset();
                    model.flush()
                } else {
                    continue;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        };
        let events = match events {
            Ok(ev) => ev,
            Err(e) => {
                tracing::error!("Kyutai STT : {e}");
                let _ = model.reset();
                words.clear();
                continue;
            }
        };
        for ev in events {
            match ev {
                KyutaiEvent::Word { text, .. } => {
                    if words.is_empty() {
                        sentence_started = Instant::now();
                    }
                    let ends_sentence = text.ends_with(['.', '?', '!']);
                    words.push(text);
                    let partial = words.join(" ");
                    let lang = {
                        let cfg = state.config.blocking_read();
                        effective_source_lang(&cfg)
                    };
                    let target = state.config.blocking_read().translation.target_lang.clone();
                    let _ = state.event_tx.send(PipelineEvent::new_streaming(
                        partial,
                        String::new(),
                        None,
                        lang,
                        target,
                        delay_ms,
                        model_name.clone(),
                    ));
                    if (ends_sentence && words.len() >= 3) || words.len() >= 40 {
                        finalize(&mut words, sentence_started);
                    }
                }
                KyutaiEvent::EndOfTurn => finalize(&mut words, sentence_started),
            }
        }
    }
    finalize(&mut words, sentence_started);
    tracing::info!("Kyutai STT arrêté.");
}

fn run_hibiki(
    state: AppState,
    mut model: crate::translate::HibikiTranslator,
    rx: std::sync::mpsc::Receiver<Vec<f32>>,
    stop: Arc<AtomicBool>,
    device_label: String,
) {
    let model_name = format!("Kyutai Hibiki ({device_label})");
    let mut up = StreamingResampler::new(PIPELINE_RATE, crate::translate::hibiki_engine::SAMPLE_RATE);
    let mut sentence = String::new();
    let mut silent_since: Option<Instant> = None;
    let mut last_audio = Instant::now();
    let mut flushed_after_pause = true;
    let mut rtf = RtfMeter::new();
    tracing::info!("Hibiki (FR → EN) en écoute sur {}.", device_label);

    let publish_final = |sentence: &mut String| {
        let text = sentence.trim().to_string();
        sentence.clear();
        if text.is_empty() {
            return;
        }
        tracing::info!("🌐 [{}] {}", model_name, text);
        state.submit_utterance(Utterance {
            text: String::new(),
            speaker: None,
            language: "fr".to_string(),
            stt_ms: 0,
            model_name: model_name.clone(),
            pre_translated: Some(text),
        });
    };

    while !stop.load(Ordering::Relaxed) {
        let pieces = match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(pcm16) => {
                last_audio = Instant::now();
                flushed_after_pause = false;
                let rms = crate::audio::calculate_rms(&pcm16);
                if rms < 0.01 {
                    silent_since.get_or_insert_with(Instant::now);
                } else {
                    silent_since = None;
                }
                let t0 = Instant::now();
                let pcm24 = up.process(&pcm16);
                let res = model.push_pcm(&pcm24);
                rtf.record(pcm16.len(), t0.elapsed(), "Hibiki");
                res
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if !flushed_after_pause && last_audio.elapsed() > Duration::from_millis(800) {
                    // Microphone paused: let Hibiki finish translating what it heard.
                    flushed_after_pause = true;
                    up.reset();
                    let res = model.flush(3.0);
                    if let Ok(p) = &res {
                        for piece in p {
                            sentence.push_str(piece);
                        }
                    }
                    publish_final(&mut sentence);
                    model.reset();
                    continue;
                } else {
                    continue;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        };
        let pieces = match pieces {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("Hibiki : {e}");
                model.reset();
                sentence.clear();
                continue;
            }
        };
        let mut changed = false;
        for piece in pieces {
            sentence.push_str(&piece);
            changed = true;
            let trimmed = sentence.trim_end();
            if trimmed.ends_with(['.', '?', '!']) && trimmed.split_whitespace().count() >= 3 {
                publish_final(&mut sentence);
                changed = false;
            }
        }
        if changed && !sentence.trim().is_empty() {
            let target = "en".to_string();
            let _ = state.event_tx.send(PipelineEvent::new_streaming(
                String::new(),
                sentence.trim().to_string(),
                None,
                "fr".to_string(),
                target,
                0,
                model_name.clone(),
            ));
        }
        // Long silence: close the sentence and recycle long sessions.
        let silent_for = silent_since.map(|t| t.elapsed()).unwrap_or_default();
        if silent_for > Duration::from_secs(4) && !sentence.trim().is_empty() {
            publish_final(&mut sentence);
        }
        if model.should_recycle(silent_for > Duration::from_secs(2)) {
            tracing::debug!("Hibiki : nouvelle session ({} pas).", model.session_steps());
            publish_final(&mut sentence);
            model.reset();
            up.reset();
        }
    }
    publish_final(&mut sentence);
    tracing::info!("Hibiki arrêté.");
}
