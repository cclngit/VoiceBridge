mod audio;
mod config;
mod gui;
mod server;
mod stt;
mod translate;
mod vad;

use audio::{AudioCapture, AudioDeviceManager};
use clap::Parser;
use config::AppConfig;
use server::{create_router, AppState, PipelineEvent};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use stt::{ensure_model_exists, RemoteAsrClient, WhisperEngine};
use tokio::sync::{broadcast, mpsc, watch};
use translate::TranslationService;
use vad::SpeechSegmenter;

#[derive(Parser, Debug)]
#[command(author, version, about = "VoiceBridge — Speech To Text To Translate in Rust")]
struct Args {
    /// Path to config.json
    #[arg(short, long, default_value = "config.json")]
    config: PathBuf,

    /// Web server port
    #[arg(short, long)]
    port: Option<u16>,

    /// Target language for translation (e.g. fr, en, es, de, it, ja)
    #[arg(short, long)]
    target_lang: Option<String>,

    /// Path to Whisper GGML model
    #[arg(short, long)]
    model: Option<String>,

    /// Run in headless mode without native desktop GUI (web & OBS overlay only)
    #[arg(long, default_value_t = false)]
    headless: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,VoiceBridge=debug".into()),
        )
        .init();

    let args = Args::parse();
    let mut config = AppConfig::load_or_default(&args.config);

    if let Some(port) = args.port {
        config.server.port = port;
    }
    if let Some(target) = args.target_lang {
        config.translation.target_lang = target;
    }
    if let Some(model) = args.model {
        config.stt.model_path = model;
    }

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║       VoiceBridge — Speech To Text To Translate (Rust)    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // 1. Build Multi-threaded Tokio Runtime
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    // 2. Initialize Models, Audio Capture & Server inside Tokio Runtime
    let (app_state, audio_capture) = rt.block_on(async {
        if config.stt.engine_type == "whisper_local" {
            let _ = ensure_model_exists(&config.stt.model_path, &config.stt.model_url).await;
        }

        let whisper_engine = Arc::new(tokio::sync::Mutex::new(WhisperEngine::new(
            &config.stt.model_path,
            &config.stt.language,
            config.stt.n_threads,
            config.stt.use_gpu,
        )?));
        let remote_asr_client = Arc::new(RemoteAsrClient::new());
        let download_status = Arc::new(tokio::sync::RwLock::new(stt::DownloadStatus::default()));

        let translation_service = Arc::new(tokio::sync::RwLock::new(TranslationService::new(
            config.translation.enabled,
            config.translation.source_lang.clone(),
            config.translation.target_lang.clone(),
            config.translation.engine.clone(),
            config.translation.local_api_url.clone(),
            config.translation.model_name.clone(),
            config.translation.api_key.clone(),
        )));

        let (event_tx, _) = broadcast::channel::<PipelineEvent>(128);
        let (rms_tx, rms_rx) = watch::channel(0.0f32);
        let (audio_tx, mut audio_rx) = mpsc::unbounded_channel::<Vec<f32>>();

        let is_recording = Arc::new(AtomicBool::new(true));

        let devices = AudioDeviceManager::list_input_devices();
        let default_device_name = devices
            .iter()
            .find(|d| d.is_default)
            .map(|d| d.name.clone())
            .unwrap_or_else(|| "Default Audio Device".into());

        let audio_capture = AudioCapture::start(
            config.audio.device_name.as_deref(),
            config.audio.sample_rate,
            audio_tx,
            Some(rms_tx),
        )?;

        let app_state = AppState::new(
            event_tx.clone(),
            rms_rx,
            is_recording.clone(),
            default_device_name,
            config.clone(),
            whisper_engine.clone(),
            download_status.clone(),
            translation_service.clone(),
        );

        // Background Audio Processing Pipeline
        let pipeline_state = app_state.clone();
        let whisper_clone = whisper_engine.clone();
        let remote_asr_clone = remote_asr_client.clone();
        let translator_clone = translation_service.clone();
        let audio_cfg = config.audio.clone();
        let is_rec_clone = is_recording.clone();

        tokio::spawn(async move {
            let mut segmenter = SpeechSegmenter::new(
                audio_cfg.sample_rate,
                audio_cfg.vad_threshold,
                audio_cfg.vad_silence_duration_ms,
                audio_cfg.vad_min_speech_ms,
                audio_cfg.vad_max_speech_ms,
            );

            tracing::info!("Audio processing pipeline running (in-memory zero-disk streaming).");

            let mut last_config_check = std::time::Instant::now();
            while let Some(chunk) = audio_rx.recv().await {
                if !is_rec_clone.load(std::sync::atomic::Ordering::Relaxed) {
                    continue;
                }

                if last_config_check.elapsed().as_millis() > 500 {
                    let (th, sil, min_sp, max_sp) = {
                        let cfg = pipeline_state.config.read().await;
                        (
                            cfg.audio.vad_threshold,
                            cfg.audio.vad_silence_duration_ms,
                            cfg.audio.vad_min_speech_ms,
                            cfg.audio.vad_max_speech_ms,
                        )
                    };
                    segmenter.update_params(th, sil, min_sp, max_sp);
                    last_config_check = std::time::Instant::now();
                }

                if let Some(speech_segment) = segmenter.process_chunk(&chunk) {
                    let whisper = whisper_clone.clone();
                    let remote_asr = remote_asr_clone.clone();
                    let translator = translator_clone.clone();
                    let state = pipeline_state.clone();

                    tokio::spawn(async move {
                        let cfg = state.config.read().await.clone();
                        let target_lang = cfg.translation.target_lang.clone();
                        let effective_source_lang = if cfg.stt.language != "auto" && !cfg.stt.language.is_empty() {
                            cfg.stt.language.clone()
                        } else {
                            cfg.translation.source_lang.clone()
                        };

                        let stt_result = if cfg.stt.engine_type == "whisper_local" {
                            let mut engine = whisper.lock().await;
                            engine.transcribe(&speech_segment, Some(&effective_source_lang), false)
                        } else {
                            match remote_asr.transcribe(&speech_segment, &cfg.stt).await {
                                Ok(res) => Ok(res),
                                Err(err) => {
                                    tracing::warn!(
                                        "Serveur distant ASR injoignable ({}), bascule automatique sur Whisper local GPU/CPU.",
                                        err
                                    );
                                    let mut engine = whisper.lock().await;
                                    engine.transcribe(&speech_segment, Some(&effective_source_lang), false)
                                }
                            }
                        };

                        match stt_result {
                            Ok(stt_res) => {
                                let trimmed = stt_res.text.trim();
                                if trimmed.is_empty()
                                    || trimmed.eq_ignore_ascii_case("[blank_audio]")
                                    || trimmed.eq_ignore_ascii_case("[silence]")
                                    || trimmed.eq_ignore_ascii_case("(silence)")
                                {
                                    return;
                                }

                                let speaker_str = stt_res
                                    .speaker
                                    .as_deref()
                                    .map(|s| format!(" [{}]", s))
                                    .unwrap_or_default();

                                tracing::info!(
                                    "🗣️  [STT {} ({}ms){}]: {}",
                                    stt_res.model_name,
                                    stt_res.duration_ms,
                                    speaker_str,
                                    stt_res.text
                                );

                                let trans_res = {
                                    let trans = translator.read().await;
                                    let effective_source = if effective_source_lang == "auto" || effective_source_lang.is_empty() {
                                        &stt_res.language
                                    } else {
                                        &effective_source_lang
                                    };
                                    trans.translate(&stt_res.text, effective_source, &target_lang).await
                                };

                                tracing::info!(
                                    "🌐 [MT {}ms -> {}]: {}",
                                    trans_res.duration_ms,
                                    target_lang,
                                    trans_res.translated_text
                                );

                                let event = PipelineEvent::new(
                                    stt_res.text,
                                    trans_res.translated_text,
                                    stt_res.speaker,
                                    stt_res.language,
                                    target_lang,
                                    stt_res.duration_ms,
                                    trans_res.duration_ms,
                                    stt_res.model_name,
                                );

                                state.add_event(event).await;
                            }
                            Err(err) => {
                                tracing::error!("Speech recognition error: {}", err);
                            }
                        }
                    });
                }
            }
        });

        // 3. Start Background Web Server for OBS Studio Browser Source
        let router = create_router(app_state.clone());
        let addr = SocketAddr::from(([127, 0, 0, 1], config.server.port));

        tracing::info!("VoiceBridge Web Interface available at: http://{}", addr);
        tracing::info!("OBS Studio Subtitle Overlay available at: http://{}/obs", addr);

        tokio::spawn(async move {
            match tokio::net::TcpListener::bind(&addr).await {
                Ok(listener) => {
                    if let Err(e) = axum::serve(listener, router).await {
                        tracing::error!("Web server error: {}", e);
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to bind web server port {}: {}", addr, e);
                }
            }
        });

        Ok::<_, Box<dyn std::error::Error + Send + Sync>>((app_state, audio_capture))
    })?;

    // 4. Run GUI or Headless Event Loop
    if args.headless {
        println!("🚀 Mode headless actif : Interface Web et OBS opérationnelles sur http://127.0.0.1:{}", config.server.port);
        println!("Appuyez sur Ctrl+C pour quitter.");
        rt.block_on(async {
            let _ = tokio::signal::ctrl_c().await;
            println!("\nArrêt du pipeline...");
        });
        audio_capture.set_active(false);
    } else {
        println!("✨ Lancement de l'interface graphique native VoiceBridge (GPUI 120 FPS)...");
        let handle = rt.handle().clone();
        let app_state_gui = app_state.clone();

        let res = gui::run_gui(app_state_gui, handle);

        audio_capture.set_active(false);

        if let Err(e) = res {
            tracing::error!("Erreur interface graphique GPUI: {:?}", e);
        }
    }

    Ok(())
}
