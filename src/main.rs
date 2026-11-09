#![cfg_attr(
    all(target_os = "windows", not(debug_assertions)),
    windows_subsystem = "windows"
)]

mod audio;
mod config;
mod controller;
mod gui;
mod ml;
mod netpolicy;
mod pipeline;
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
use stt::models::{find_model_by_path, get_available_models, is_model_installed, ENGINE_WHISPER};
use stt::WhisperEngine;
use tokio::sync::{broadcast, mpsc, watch};
use translate::TranslationService;

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

    /// Use a WAV file (played at real-time speed, then silence) instead of the microphone.
    /// Useful to test engines or subtitle a recording.
    #[arg(long, value_name = "FILE")]
    input_wav: Option<PathBuf>,
}

/// Feed a WAV file into the pipeline at real-time speed, followed by silence.
fn spawn_wav_input(
    path: &std::path::Path,
    audio_tx: mpsc::UnboundedSender<Vec<f32>>,
    rms_tx: watch::Sender<f32>,
) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
    let (pcm, rate) = audio::wav::read_wav_mono(path)?;
    tracing::info!(
        "Entrée audio : fichier {} ({:.1} s à {} Hz) au lieu du micro.",
        path.display(),
        pcm.len() as f32 / rate as f32,
        rate
    );
    let chunk = (rate as usize / 50).max(1); // 20 ms
    std::thread::Builder::new().name("voicebridge-wav-input".into()).spawn(move || {
        let start = std::time::Instant::now();
        let mut sent = 0usize;
        let silence = vec![0.0f32; chunk];
        let mut it = pcm.chunks(chunk);
        loop {
            let block = it.next().unwrap_or(&silence);
            let _ = rms_tx.send(audio::calculate_rms(block));
            if audio_tx.send(block.to_vec()).is_err() {
                break;
            }
            sent += block.len();
            let due = std::time::Duration::from_secs_f64(sent as f64 / rate as f64);
            if let Some(wait) = due.checked_sub(start.elapsed()) {
                std::thread::sleep(wait);
            }
        }
    })?;
    Ok(rate)
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
            if cwd.join("config.json").is_file()
                && let Ok(file) = std::fs::OpenOptions::new()
                    .write(true)
                    .open(cwd.join("config.json"))
                {
                    drop(file);
                    return true;
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

    if let Some(app_dir) = user_data_dir
        && std::fs::create_dir_all(app_dir.join("models")).is_ok() {
            let _ = std::env::set_current_dir(&app_dir);
        }
}

/// Whisper model loaded at startup. When a streaming engine (Kyutai / Hibiki) is
/// selected, `stt.model_path` is that model's directory, so Whisper (used as fallback)
/// loads an installed Whisper model instead.
fn startup_whisper_path(config: &AppConfig) -> (String, String) {
    if config.stt.engine_type == ENGINE_WHISPER || config.stt.model_path.ends_with(".bin") {
        return (config.stt.model_path.clone(), config.stt.model_url.clone());
    }
    let catalog = get_available_models();
    let whisper_models: Vec<_> = catalog.iter().filter(|m| m.default_engine == ENGINE_WHISPER).collect();
    let chosen = whisper_models
        .iter()
        .find(|m| is_model_installed(m))
        .or_else(|| whisper_models.iter().find(|m| m.id == "whisper_turbo_q5"))
        .or_else(|| whisper_models.first());
    match chosen {
        Some(m) => (m.default_model_name.clone(), m.download_url.clone().unwrap_or_default()),
        None => (config.stt.model_path.clone(), config.stt.model_url.clone()),
    }
}

fn stop_capture(capture: &Option<AudioCapture>) {
    if let Some(c) = capture {
        c.set_active(false);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    #[cfg(windows)]
    attach_parent_console_if_needed();

    let launch_cwd = std::env::current_dir().ok();
    let args = Args::parse();
    ensure_writable_app_workdir();

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
    let config_path = std::path::absolute(&config_path).unwrap_or(config_path);
    config::set_config_path(config_path.clone());
    let mut config = AppConfig::load_or_default(&config_path);

    if let Some(port) = args.port {
        config.server.port = port;
    }
    if let Some(target) = args.target_lang {
        config.translation.target_lang = target;
    }
    if let Some(model) = args.model {
        config.stt.engine_type = ENGINE_WHISPER.to_string();
        config.stt.model_path = model;
        if let Some(meta) = find_model_by_path(&config.stt.model_path) {
            config.stt.model_id = meta.id;
            config.stt.model_name = meta.name;
        }
    }
    if let Some(backend) = args.gpu_backend {
        let lower = backend.to_ascii_lowercase();
        config.stt.use_gpu = lower != "cpu";
        config.stt.gpu_backend = lower;
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
    config.sanitize();

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║   VoiceBridge — Speech To Text To Translate (100 % local)     ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    tracing::info!("Configuration : {}", config_path.display());

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

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;

    let (app_state, audio_capture, bound_addr) = rt.block_on(async {
        let download_status = Arc::new(tokio::sync::RwLock::new(stt::DownloadStatus::default()));
        let (whisper_path, whisper_url) = startup_whisper_path(&config);
        stt::downloader::ensure_model_exists(&whisper_path, &whisper_url, &download_status).await?;
        if config.stt.engine_type == ENGINE_WHISPER {
            config.stt.model_path = whisper_path.clone();
        }

        let whisper_engine = Arc::new(tokio::sync::Mutex::new(WhisperEngine::new_with_gpu_options(
            &whisper_path,
            &config.stt.language,
            config.stt.n_threads,
            config.stt.use_gpu,
            &config.stt.gpu_backend,
            config.stt.gpu_device,
            config.stt.flash_attn,
        )?));
        stt::register_atexit_whisper_cleanup(&whisper_engine);

        let mut translation = TranslationService::new(
            config.translation.enabled,
            config.translation.source_lang.clone(),
            config.translation.target_lang.clone(),
            config.translation.engine.clone(),
            config.translation.local_api_url.clone(),
            config.translation.model_name.clone(),
            config.translation.api_key.clone(),
            PathBuf::from(&config.translation.madlad_model_dir),
        );
        translation.apply_config(&config.translation, config.stt.use_gpu, config.stt.gpu_device);
        let translation_service = Arc::new(tokio::sync::RwLock::new(translation));

        let (event_tx, _) = broadcast::channel::<PipelineEvent>(256);
        let (rms_tx, rms_rx) = watch::channel(0.0f32);
        let (audio_tx, audio_rx) = mpsc::unbounded_channel::<Vec<f32>>();
        let is_recording = Arc::new(AtomicBool::new(true));

        let default_device_name = AudioDeviceManager::list_input_devices()
            .into_iter()
            .find(|d| d.is_default)
            .map(|d| d.name)
            .unwrap_or_else(|| "Default Audio Device".into());

        let (audio_capture, source_rate) = match &args.input_wav {
            Some(path) => (None, spawn_wav_input(path, audio_tx, rms_tx)?),
            None => {
                let capture = AudioCapture::start(config.audio.device_name.as_deref(), audio_tx, Some(rms_tx))?;
                let rate = capture.source_rate();
                (Some(capture), rate)
            }
        };

        // Bind the web server (fallback to the next ports if the desired one is busy).
        let host_ip: std::net::IpAddr = config
            .server
            .host
            .parse()
            .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
        if !host_ip.is_loopback() {
            tracing::warn!(
                "Le serveur web écoute sur {} : l'interface est accessible depuis votre réseau local.",
                host_ip
            );
        }
        let desired_port = config.server.port;
        let mut bound_listener = None;
        for offset in 0..=10u16 {
            let Some(candidate_port) = desired_port.checked_add(offset) else { break };
            let candidate_addr = SocketAddr::new(host_ip, candidate_port);
            match tokio::net::TcpListener::bind(&candidate_addr).await {
                Ok(listener) => {
                    if offset > 0 {
                        tracing::warn!(
                            "Port {} occupé : serveur web démarré sur le port {}.",
                            desired_port,
                            candidate_port
                        );
                        config.server.port = candidate_port;
                    }
                    bound_listener = Some((listener, candidate_addr));
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => continue,
                Err(e) => {
                    tracing::error!("Impossible d'ouvrir le port {} : {}", candidate_addr, e);
                    break;
                }
            }
        }

        let app_state = AppState::new(
            event_tx,
            rms_rx,
            is_recording,
            default_device_name,
            config.clone(),
            whisper_engine,
            download_status,
            translation_service,
        );

        pipeline::spawn(app_state.clone(), audio_rx, source_rate);
        {
            let state = app_state.clone();
            tokio::spawn(async move { state.restore_engine_on_startup().await });
        }

        let bound_addr = bound_listener.map(|(listener, addr)| {
            let router = create_router(app_state.clone());
            tracing::info!("Interface web VoiceBridge : http://{}", addr);
            tracing::info!("Sous-titres OBS Studio : http://{}/obs", addr);
            tokio::spawn(async move {
                if let Err(e) = axum::serve(listener, router).await {
                    tracing::error!("Erreur du serveur web : {}", e);
                }
            });
            addr
        });

        Ok::<_, Box<dyn std::error::Error + Send + Sync>>((app_state, audio_capture, bound_addr))
    })?;

    if args.headless {
        let Some(addr) = bound_addr else {
            stop_capture(&audio_capture);
            app_state.shutdown_sync();
            return Err("aucun port disponible pour le serveur web (mode headless sans interface)".into());
        };
        println!("🚀 Mode headless : interface web et OBS sur http://{addr}");
        println!("Appuyez sur Ctrl+C pour quitter.");
        rt.block_on(async {
            let _ = tokio::signal::ctrl_c().await;
            println!("\nArrêt du pipeline...");
        });
        stop_capture(&audio_capture);
        app_state.shutdown_sync();
    } else {
        if bound_addr.is_none() {
            tracing::error!("Serveur web indisponible (aucun port libre) : l'overlay OBS ne fonctionnera pas.");
        }
        println!("✨ Lancement de l'interface graphique native VoiceBridge...");
        let res = gui::run_gui(app_state.clone(), rt.handle().clone());
        stop_capture(&audio_capture);
        app_state.shutdown_sync();
        if let Err(e) = res {
            tracing::error!("Erreur interface graphique GPUI: {:?}", e);
        }
    }

    Ok(())
}
