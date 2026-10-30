#![cfg_attr(
    all(target_os = "windows", not(debug_assertions)),
    windows_subsystem = "windows"
)]

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
use stt::{ensure_model_exists, resolve_model_download_url, RemoteAsrClient, WhisperEngine};
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

    /// Hardware acceleration backend (auto, cuda, vulkan, metal, rocm, cpu)
    #[arg(long)]
    gpu_backend: Option<String>,

    /// GPU device index (0, 1, ...) when multiple GPUs are available
    #[arg(long)]
    gpu_device: Option<i32>,

    /// Force CPU execution (disable GPU acceleration)
    #[arg(long, default_value_t = false)]
    no_gpu: bool,

    /// Enable Flash Attention on GPU
    #[arg(long)]
    flash_attn: Option<bool>,

    /// Number of CPU threads for Whisper inference
    #[arg(long)]
    threads: Option<usize>,

    /// Run in headless mode without native desktop GUI (web & OBS overlay only)
    #[arg(long, default_value_t = false)]
    headless: bool,
}

#[cfg(windows)]
fn attach_parent_console_if_needed() {
    unsafe extern "system" {
        fn AttachConsole(dwProcessId: u32) -> i32;
        fn GetStdHandle(nStdHandle: u32) -> *mut std::ffi::c_void;
    }
    const ATTACH_PARENT_PROCESS: u32 = 0xFFFF_FFFF;
    const STD_OUTPUT_HANDLE: u32 = (-11i32) as u32;
    unsafe {
        // Only attach to an interactive parent terminal if stdout is not already redirected/piped.
        if GetStdHandle(STD_OUTPUT_HANDLE).is_null() {
            let _ = AttachConsole(ATTACH_PARENT_PROCESS);
        }
    }
}

/// Ensures the process runs in a writable data directory when launched from an installed
/// system bundle (macOS `.app` from `.dmg` where `cwd == "/"`, Linux `.deb` `/usr/bin/voicebridge`,
/// or Windows `Program Files\VoiceBridge`), while keeping `cargo run` / portable folders untouched.
fn ensure_writable_app_workdir() {
    let is_dev_or_portable = std::env::current_dir()
        .ok()
        .map(|cwd| {
            if cwd.join("Cargo.toml").is_file() {
                return true;
            }
            if cwd.join("config.json").is_file() {
                if let Ok(file) = std::fs::OpenOptions::new()
                    .write(true)
                    .open(cwd.join("config.json"))
                {
                    drop(file);
                    return true;
                }
            }
            false
        })
        .unwrap_or(false);

    if is_dev_or_portable {
        return;
    }

    let user_data_dir: Option<PathBuf> = {
        #[cfg(target_os = "macos")]
        {
            std::env::var_os("HOME").map(|h| {
                PathBuf::from(h)
                    .join("Library")
                    .join("Application Support")
                    .join("VoiceBridge")
            })
        }
        #[cfg(target_os = "windows")]
        {
            std::env::var_os("LOCALAPPDATA")
                .or_else(|| std::env::var_os("APPDATA"))
                .map(|p| PathBuf::from(p).join("VoiceBridge"))
        }
        #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
        {
            std::env::var_os("XDG_DATA_HOME")
                .map(|p| PathBuf::from(p).join("VoiceBridge"))
                .or_else(|| {
                    std::env::var_os("HOME").map(|h| {
                        PathBuf::from(h)
                            .join(".local")
                            .join("share")
                            .join("VoiceBridge")
                    })
                })
        }
    };

    if let Some(app_dir) = user_data_dir {
        if std::fs::create_dir_all(app_dir.join("models")).is_ok() {
            let _ = std::env::set_current_dir(&app_dir);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    #[cfg(windows)]
    attach_parent_console_if_needed();

    let launch_cwd = std::env::current_dir().ok();
    let args = Args::parse();
    ensure_writable_app_workdir();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,VoiceBridge=debug".into()),
        )
        .init();

    let config_path = if args.config != std::path::Path::new("config.json") && args.config.is_relative() {
        launch_cwd
            .as_ref()
            .map(|cwd| cwd.join(&args.config))
            .unwrap_or_else(|| args.config.clone())
    } else {
        args.config.clone()
    };
    let mut config = AppConfig::load_or_default(&config_path);

    if let Some(port) = args.port {
        config.server.port = port;
    }
    if let Some(target) = args.target_lang {
        config.translation.target_lang = target;
    }
    if let Some(model) = args.model {
        config.stt.model_path = model;
    }
    if let Some(backend) = args.gpu_backend {
        let lower = backend.to_ascii_lowercase();
        if lower == "cpu" {
            config.stt.use_gpu = false;
            config.stt.gpu_backend = "cpu".to_string();
        } else {
            config.stt.use_gpu = true;
            config.stt.gpu_backend = lower;
        }
    }
    if let Some(dev_idx) = args.gpu_device {
        config.stt.gpu_device = dev_idx.max(0);
    }
    if args.no_gpu {
        config.stt.use_gpu = false;
        config.stt.gpu_backend = "cpu".to_string();
    }
    if let Some(flash) = args.flash_attn {
        config.stt.flash_attn = flash;
    }
    if let Some(threads) = args.threads {
        config.stt.n_threads = threads.max(1);
    }

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║       VoiceBridge — Speech To Text To Translate (Rust)    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let detected_gpus = stt::list_gpu_devices();
    if detected_gpus.is_empty() {
        tracing::info!(
            "Hardware backend: {} (no dedicated GPU devices detected by ggml)",
            stt::active_whisper_backend_label()
        );
    } else {
        for g in &detected_gpus {
            tracing::info!(
                "Detected GPU #{}: {} [{}] ({} MB free / {} MB total VRAM)",
                g.index,
                g.description,
                g.backend,
                g.vram_free_mb,
                g.vram_total_mb
            );
        }
    }

    // 1. Build Multi-threaded Tokio Runtime
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    // 2. Initialize Models, Audio Capture & Server inside Tokio Runtime
    let (app_state, audio_capture, bound_port) = rt.block_on(async {
        let resolved_url = resolve_model_download_url(
            &config.stt.model_id,
            &config.stt.model_path,
            &config.stt.model_url,
        );
        if config.stt.model_url != resolved_url {
            config.stt.model_url = resolved_url.clone();
        }
        ensure_model_exists(&config.stt.model_path, &resolved_url).await?;

        let whisper_engine = Arc::new(tokio::sync::Mutex::new(
            WhisperEngine::new_with_gpu_options(
                &config.stt.model_path,
                &config.stt.language,
                config.stt.n_threads,
                config.stt.use_gpu,
                &config.stt.gpu_backend,
                config.stt.gpu_device,
                config.stt.flash_attn,
            )?,
        ));
        stt::register_atexit_whisper_cleanup(&whisper_engine);
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

        // Pre-bind web server port (with automatic fallback if desired port is occupied)
        let host_ip: std::net::IpAddr = config
            .server
            .host
            .parse()
            .unwrap_or_else(|_| std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
        let desired_port = config.server.port;
        let mut bound_listener = None;
        for offset in 0..=10u16 {
            let candidate_port = match desired_port.checked_add(offset) {
                Some(p) => p,
                None => break,
            };
            let candidate_addr = SocketAddr::new(host_ip, candidate_port);
            match tokio::net::TcpListener::bind(&candidate_addr).await {
                Ok(listener) => {
                    if offset > 0 {
                        tracing::warn!(
                            "Port {} was already in use; bound web server to fallback port {}.",
                            desired_port,
                            candidate_port
                        );
                        config.server.port = candidate_port;
                    }
                    bound_listener = Some((listener, candidate_addr));
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AddrInUse && offset < 10 => {
                    if offset == 0 {
                        tracing::warn!(
                            "Web server port {} is already in use ({}). Searching for next available port...",
                            candidate_addr,
                            e
                        );
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to bind web server port {}: {}", candidate_addr, e);
                    break;
                }
            }
        }

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
                if pipeline_state
                    .is_shutting_down
                    .load(std::sync::atomic::Ordering::Relaxed)
                {
                    break;
                }
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
                        if state
                            .is_shutting_down
                            .load(std::sync::atomic::Ordering::Relaxed)
                        {
                            return;
                        }
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
                                    if state
                                        .is_shutting_down
                                        .load(std::sync::atomic::Ordering::Relaxed)
                                    {
                                        return;
                                    }
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
        let actual_port = config.server.port;
        if let Some((listener, actual_addr)) = bound_listener {
            let router = create_router(app_state.clone());
            tracing::info!("VoiceBridge Web Interface available at: http://{}", actual_addr);
            tracing::info!("OBS Studio Subtitle Overlay available at: http://{}/obs", actual_addr);

            tokio::spawn(async move {
                if let Err(e) = axum::serve(listener, router).await {
                    tracing::error!("Web server error: {}", e);
                }
            });
        }

        Ok::<_, Box<dyn std::error::Error + Send + Sync>>((app_state, audio_capture, actual_port))
    })?;

    // 4. Run GUI or Headless Event Loop
    if args.headless {
        println!("🚀 Mode headless actif : Interface Web et OBS opérationnelles sur http://127.0.0.1:{}", bound_port);
        println!("Appuyez sur Ctrl+C pour quitter.");
        rt.block_on(async {
            let _ = tokio::signal::ctrl_c().await;
            println!("\nArrêt du pipeline...");
        });
        audio_capture.set_active(false);
        app_state.shutdown_sync();
    } else {
        println!("✨ Lancement de l'interface graphique native VoiceBridge (GPUI 120 FPS)...");
        let handle = rt.handle().clone();
        let app_state_gui = app_state.clone();

        let res = gui::run_gui(app_state_gui, handle);

        audio_capture.set_active(false);
        app_state.shutdown_sync();

        if let Err(e) = res {
            tracing::error!("Erreur interface graphique GPUI: {:?}", e);
        }
    }

    Ok(())
}
