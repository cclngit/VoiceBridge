use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use gpui_kit::assets::{AllAssets, IconName};
use gpui_kit::component::{
    h_flex, v_flex,
    notification::Notification,
    scroll::ScrollableElement as _,
    ActiveTheme as _, Icon, Sizable as _, StyledExt as _, TitleBar, WindowExt as _,
};
use gpui_kit::{
    div, hsla, img, linear_color_stop, linear_gradient, prelude::FluentBuilder as _, px,
    relative, size, App, AppContext as _, ClickEvent, ClipboardItem, Context, FocusHandle, Focusable,
    InteractiveElement as _, IntoElement, ParentElement as _, Render, Rgba,
    SharedString, StatefulInteractiveElement as _, Styled as _, Window, WindowBounds,
    WindowOptions,
};

use crate::audio::AudioDeviceManager;
use crate::server::{AppState, PipelineEvent};
use crate::stt::{
    download_model_with_progress, get_available_models, is_model_installed, list_gpu_devices,
    GpuDeviceInfo, ModelMetadata,
};

// ═══════════════════════════════════════════════════════════════
// THEME COLORS (Cyber Neon Dark - matching docs/DESIGN.md)
// ═══════════════════════════════════════════════════════════════
const fn rgba_hex(hex: u32) -> Rgba {
    let r = ((hex >> 16) & 0xff) as f32 / 255.0;
    let g = ((hex >> 8) & 0xff) as f32 / 255.0;
    let b = (hex & 0xff) as f32 / 255.0;
    Rgba { r, g, b, a: 1.0 }
}

const BG_MAIN: Rgba = rgba_hex(0x0e0e11);
const BG_PANEL: Rgba = rgba_hex(0x121215);
const BG_CARD: Rgba = rgba_hex(0x151518);
const BG_CARD_HOVER: Rgba = rgba_hex(0x1c1c20);
const BORDER_DARK: Rgba = rgba_hex(0x26262c);
const BORDER_SUBTLE: Rgba = rgba_hex(0x1f1f24);

const TEXT_MAIN: Rgba = rgba_hex(0xffffff);
const TEXT_BODY: Rgba = rgba_hex(0xd4d4d4);
const TEXT_MUTED: Rgba = rgba_hex(0x888894);
const TEXT_DIM: Rgba = rgba_hex(0x555562);

const ACCENT_CYAN: Rgba = rgba_hex(0x38bdf8);
const ACCENT_GREEN: Rgba = rgba_hex(0x22c55e);
const ACCENT_RED: Rgba = rgba_hex(0xef4444);
const ACCENT_MAGENTA: Rgba = rgba_hex(0xd946ef);
const ACCENT_YELLOW: Rgba = rgba_hex(0xedff00);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GuiTab {
    #[default]
    Direct = 0,
    Modeles = 1,
    Langues = 2,
    AudioVad = 3,
    Incrustation = 4,
}

pub struct VoiceBridgeApp {
    focus_handle: FocusHandle,
    state: AppState,
    tokio_handle: tokio::runtime::Handle,
    active_tab: GuiTab,

    // Live display state (Direct tab)
    latest_source_text: String,
    latest_trans_text: String,
    latest_source_lang: String,
    latest_target_lang: String,
    configured_source_lang: String,
    configured_target_lang: String,
    latest_stt_ms: u128,
    latest_trans_ms: u128,
    latest_confidence: f32,

    // Real-time audio / pipeline metrics
    current_volume: f32,
    is_recording: bool,
    history_entries: Vec<PipelineEvent>,

    // Catalog & Devices
    models_catalog: Vec<ModelMetadata>,
    available_devices: Vec<String>,
    selected_device: String,

    // GPU / CPU Hardware Acceleration configuration
    gpu_devices: Vec<GpuDeviceInfo>,
    use_gpu: bool,
    gpu_backend: String,
    gpu_device: i32,
    flash_attn: bool,
    n_threads: usize,

    // Model configuration
    active_model_id: String,
    active_engine: String,
    api_url_input: String,
    api_key_input: String,
    local_server_status: String,

    // Background ASR process handle
    local_asr_child: Arc<std::sync::Mutex<Option<std::process::Child>>>,

    // VAD customization parameters
    vad_threshold: f32,
    vad_silence_ms: u64,
    vad_min_speech_ms: u64,
    vad_max_speech_ms: u64,

    // Download status
    download_msg: String,
    download_pct: f32,
    is_downloading: bool,
}

impl VoiceBridgeApp {
    pub fn new(
        state: AppState,
        tokio_handle: tokio::runtime::Handle,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let initial_cfg = state.config.blocking_read().clone();

        let api_url_input = if initial_cfg.stt.api_url.is_empty() {
            "http://127.0.0.1:8000/v1/audio/transcriptions".to_string()
        } else {
            initial_cfg.stt.api_url.clone()
        };
        let api_key_input = initial_cfg.stt.api_key.clone().unwrap_or_default();
        let active_model_id = initial_cfg.stt.model_id.clone();
        let active_engine = initial_cfg.stt.engine_type.clone();

        let devices: Vec<String> = AudioDeviceManager::list_input_devices()
            .into_iter()
            .map(|d| d.name)
            .collect();
        let selected_device = initial_cfg
            .audio
            .device_name
            .clone()
            .unwrap_or_else(|| state.audio_device.clone());

        let models_catalog = get_available_models();
        let gpu_devices = list_gpu_devices();

        let init_src = if !initial_cfg.stt.language.is_empty() {
            initial_cfg.stt.language.clone()
        } else {
            initial_cfg.translation.source_lang.clone()
        };
        let init_tgt = initial_cfg.translation.target_lang.clone();

        let mut app = Self {
            focus_handle: cx.focus_handle(),
            state,
            tokio_handle,
            active_tab: GuiTab::Direct,
            latest_source_text: "Allez, c'est parti !".to_string(),
            latest_trans_text: "Let's get started!".to_string(),
            latest_source_lang: if init_src == "auto" { "AUTO".to_string() } else { init_src.to_uppercase() },
            latest_target_lang: init_tgt.to_uppercase(),
            configured_source_lang: init_src,
            configured_target_lang: init_tgt,
            latest_stt_ms: 1152,
            latest_trans_ms: 661,
            latest_confidence: 98.4,
            current_volume: 0.0,
            is_recording: true,
            history_entries: Vec::new(),
            models_catalog,
            available_devices: devices,
            selected_device,
            gpu_devices,
            use_gpu: initial_cfg.stt.use_gpu,
            gpu_backend: initial_cfg.stt.gpu_backend.clone(),
            gpu_device: initial_cfg.stt.gpu_device,
            flash_attn: initial_cfg.stt.flash_attn,
            n_threads: initial_cfg.stt.n_threads.max(1),
            active_model_id,
            active_engine,
            api_url_input,
            api_key_input,
            local_server_status: "Arrêté".to_string(),
            local_asr_child: Arc::new(std::sync::Mutex::new(None)),
            vad_threshold: initial_cfg.audio.vad_threshold,
            vad_silence_ms: initial_cfg.audio.vad_silence_duration_ms,
            vad_min_speech_ms: initial_cfg.audio.vad_min_speech_ms,
            vad_max_speech_ms: initial_cfg.audio.vad_max_speech_ms,
            download_msg: String::new(),
            download_pct: 0.0,
            is_downloading: false,
        };

        // Populate initial history if available
        let hist = app.state.history.blocking_read().clone();
        if !hist.is_empty() {
            app.history_entries = hist.into_iter().rev().collect();
            if let Some(first) = app.history_entries.first() {
                app.latest_source_text = first.original_text.clone();
                app.latest_trans_text = first.translated_text.clone();
                if app.configured_source_lang == "auto" {
                    app.latest_source_lang = first.source_lang.to_uppercase();
                } else {
                    app.latest_source_lang = app.configured_source_lang.to_uppercase();
                }
                app.latest_target_lang = app.configured_target_lang.to_uppercase();
                app.latest_stt_ms = first.stt_duration_ms;
                app.latest_trans_ms = first.translation_duration_ms;
            }
        }

        // Auto-start local Python ASR server if saved configuration uses a local catalog server model
        if initial_cfg.stt.engine_type != "whisper_local"
            && initial_cfg.stt.api_url.contains("127.0.0.1:8000")
            && initial_cfg.stt.model_id != "custom_api"
        {
            if let Some(meta) = crate::stt::find_model(&initial_cfg.stt.model_id) {
                if !meta.is_local_gpu {
                    app.start_local_asr_server(&meta.default_model_name, cx);
                }
            }
        }

        // Start background high-frequency polling loop for volume & events
        let state_clone = app.state.clone();
        cx.spawn(async move |this, cx| {
            let mut event_rx = state_clone.subscribe_events();
            loop {
                // Poll every 50ms for smooth VU meter and instant subtitle updates
                cx.background_executor().timer(Duration::from_millis(50)).await;

                // Check for new broadcast pipeline events
                let mut new_event = None;
                while let Ok(evt) = event_rx.try_recv() {
                    new_event = Some(evt);
                }

                let vol = *state_clone.rms_rx.borrow();
                let rec = state_clone.is_recording.load(Ordering::Relaxed);
                let dl = state_clone.download_status.read().await.clone();

                let res = this.update(cx, |this, cx| {
                    this.current_volume = vol;
                    this.is_recording = rec;
                    this.is_downloading = dl.is_downloading;
                    this.download_pct = dl.percent;
                    this.download_msg = dl.status_message.clone();

                    if let Some(evt) = new_event {
                        this.latest_source_text = evt.original_text.clone();
                        this.latest_trans_text = evt.translated_text.clone();
                        if this.configured_source_lang == "auto" {
                            this.latest_source_lang = evt.source_lang.to_uppercase();
                        } else {
                            this.latest_source_lang = this.configured_source_lang.to_uppercase();
                        }
                        this.latest_target_lang = this.configured_target_lang.to_uppercase();
                        this.latest_stt_ms = evt.stt_duration_ms;
                        this.latest_trans_ms = evt.translation_duration_ms;

                        // Insert at the beginning of history
                        this.history_entries.insert(0, evt);
                        if this.history_entries.len() > 100 {
                            this.history_entries.pop();
                        }
                    }
                    cx.notify();
                });

                if res.is_err() {
                    break;
                }
            }
        })
        .detach();

        app
    }

    // Apply GPU / CPU hardware acceleration changes dynamically
    fn apply_gpu_settings(
        &mut self,
        use_gpu: bool,
        gpu_backend: String,
        gpu_device: i32,
        flash_attn: bool,
        n_threads: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.use_gpu = use_gpu;
        self.gpu_backend = gpu_backend.clone();
        self.gpu_device = gpu_device;
        self.flash_attn = flash_attn;
        self.n_threads = n_threads;
        self.gpu_devices = list_gpu_devices();

        let target_label = crate::stt::active_hardware_summary(use_gpu, &gpu_backend, gpu_device);
        window.push_notification(
            Notification::info(format!(
                "Rechargement du moteur Whisper sur : {} ({} threads CPU)...",
                target_label, n_threads
            )),
            cx,
        );

        let state = self.state.clone();
        self.tokio_handle.spawn(async move {
            {
                let mut cfg = state.config.write().await;
                cfg.stt.use_gpu = use_gpu;
                cfg.stt.gpu_backend = gpu_backend.clone();
                cfg.stt.gpu_device = gpu_device;
                cfg.stt.flash_attn = flash_attn;
                cfg.stt.n_threads = n_threads;
                let _ = cfg.save_default();
            }

            let mut engine = state.whisper_engine.lock().await;
            if let Err(e) = engine.apply_hardware_settings(
                use_gpu,
                &gpu_backend,
                gpu_device,
                flash_attn,
                n_threads,
            ) {
                tracing::error!("Erreur lors du changement de périphérique GPU/CPU : {}", e);
            }
        });

        cx.notify();
    }

    // Toggle speech capture pipeline
    fn toggle_recording(&mut self, _: &ClickEvent, _: &mut Window, cx: &mut Context<Self>) {
        let current = self.state.is_recording.load(Ordering::Relaxed);
        let next = !current;
        self.state.is_recording.store(next, Ordering::Relaxed);
        self.is_recording = next;
        cx.notify();
    }

    // Copy OBS Overlay URL to Clipboard
    fn copy_obs_url(&mut self, _: &ClickEvent, window: &mut Window, cx: &mut Context<Self>) {
        let port = self.state.config.blocking_read().server.port;
        let url = format!("http://127.0.0.1:{}/obs", port);
        cx.write_to_clipboard(ClipboardItem::new_string(url.clone()));
        window.push_notification(
            Notification::info(format!("Lien OBS copié : {}", url)),
            cx,
        );
    }

    // Copy Translation to Clipboard
    fn copy_translation(&mut self, _: &ClickEvent, window: &mut Window, cx: &mut Context<Self>) {
        if !self.latest_trans_text.is_empty() {
            cx.write_to_clipboard(ClipboardItem::new_string(self.latest_trans_text.clone()));
            window.push_notification(
                Notification::info("Traduction copiée dans le presse-papier !"),
                cx,
            );
        }
    }

    // Clear history
    fn clear_history(&mut self, _: &ClickEvent, _: &mut Window, cx: &mut Context<Self>) {
        self.history_entries.clear();
        self.state.history.blocking_write().clear();
        cx.notify();
    }

    // Export history to SRT
    fn export_srt(&mut self, _: &ClickEvent, window: &mut Window, cx: &mut Context<Self>) {
        let mut srt = String::new();
        for (i, entry) in self.history_entries.iter().rev().enumerate() {
            let idx = i + 1;
            let start = &entry.timestamp;
            srt.push_str(&format!("{}\n{} --> {} [durée: {}ms]\n{}\n{}\n\n",
                idx, start, start, entry.total_duration_ms, entry.original_text, entry.translated_text
            ));
        }

        let out_path = PathBuf::from("subtitles.srt");
        if std::fs::write(&out_path, srt).is_ok() {
            window.push_notification(
                Notification::info("Historique exporté avec succès dans subtitles.srt !"),
                cx,
            );
        } else {
            window.push_notification(
                Notification::error("Erreur lors de l'exportation du fichier SRT."),
                cx,
            );
        }
    }

    // Send Test Subtitle to OBS
    fn send_test_subtitle(&mut self, _: &ClickEvent, window: &mut Window, cx: &mut Context<Self>) {
        let state = self.state.clone();
        self.tokio_handle.spawn(async move {
            let evt = PipelineEvent::new(
                "Bonjour ! Ceci est un test de sous-titrage en direct pour OBS Studio.".to_string(),
                "Hello! This is a live subtitle test for OBS Studio.".to_string(),
                None,
                "fr".to_string(),
                "en".to_string(),
                180,
                85,
                "Whisper-Turbo".to_string(),
            );
            state.add_event(evt).await;
        });

        window.push_notification(
            Notification::info("Sous-titre de test envoyé vers OBS Studio !"),
            cx,
        );
    }

    // Start Python Local Server in background (tracked child process)
    fn start_local_asr_server(&mut self, model_name: &str, cx: &mut Context<Self>) {
        {
            let mut guard = self.local_asr_child.lock().unwrap();
            if let Some(mut old_child) = guard.take() {
                terminate_asr_child(&mut old_child);
            }
        }

        let (py_bin, py_prefix_args) = resolve_python_command();
        let script_path = resolve_asr_script_path();
        let mut cmd = std::process::Command::new(&py_bin);
        cmd.args(&py_prefix_args)
            .env(
                "PYTHONWARNINGS",
                "ignore::UserWarning:multiprocessing.resource_tracker",
            )
            .arg(&script_path)
            .arg(model_name);

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        match cmd.spawn() {
            Ok(child) => {
                let mut guard = self.local_asr_child.lock().unwrap();
                *guard = Some(child);
                self.local_server_status = format!("En ligne (Port 8000) — {}", model_name);
                tracing::info!(
                    "Serveur ASR local démarré ({}) avec le modèle: {}",
                    py_bin,
                    model_name
                );
            }
            Err(e) => {
                self.local_server_status = format!("Erreur: {}", e);
                tracing::error!(
                    "Impossible de lancer {} {}: {}",
                    py_bin,
                    script_path.display(),
                    e
                );
            }
        }
        cx.notify();
    }

    // Stop Python Local Server
    fn stop_local_asr_server(&mut self, cx: &mut Context<Self>) {
        let mut guard = self.local_asr_child.lock().unwrap();
        if let Some(mut child) = guard.take() {
            terminate_asr_child(&mut child);
            tracing::info!("Serveur ASR local arrêté proprement.");
        }
        self.local_server_status = "Arrêté (Économie CPU/RAM)".to_string();
        cx.notify();
    }

    // Switch Whisper Local Model
    fn switch_local_whisper(&mut self, meta: ModelMetadata, window: &mut Window, cx: &mut Context<Self>) {
        // Automatically shut down local ASR server to save CPU/GPU and avoid conflict
        self.stop_local_asr_server(cx);

        let model_path = meta.default_model_name.clone();
        let model_id = meta.id.clone();
        let display_name = meta.name.clone();
        let download_url = meta.download_url.clone();

        if !is_model_installed(&meta) {
            window.push_notification(
                Notification::info("Téléchargement du modèle requis avant activation."),
                cx,
            );
            self.download_model(meta, window, cx);
            return;
        }

        let state = self.state.clone();
        let (use_gpu, gpu_backend, gpu_device, flash_attn, lang) = {
            let cfg = state.config.blocking_read();
            (
                cfg.stt.use_gpu,
                cfg.stt.gpu_backend.clone(),
                cfg.stt.gpu_device,
                cfg.stt.flash_attn,
                cfg.stt.language.clone(),
            )
        };

        window.push_notification(
            Notification::info(format!("Whisper Local activé : {} (Serveur ASR éteint)", display_name)),
            cx,
        );

        self.tokio_handle.spawn(async move {
            let reload_res = {
                let mut engine = state.whisper_engine.lock().await;
                engine.set_language(&lang);
                engine.reload_model_with_gpu_options(
                    &model_path,
                    use_gpu,
                    &gpu_backend,
                    gpu_device,
                    flash_attn,
                )
            };
            match reload_res {
                Ok(()) => {
                    let mut cfg = state.config.write().await;
                    cfg.stt.engine_type = "whisper_local".to_string();
                    cfg.stt.model_id = model_id;
                    cfg.stt.model_path = model_path;
                    if let Some(url) = download_url {
                        cfg.stt.model_url = url;
                    }
                    cfg.stt.model_name = display_name;
                    let _ = cfg.save_default();
                    tracing::info!("Modèle Whisper local activé avec succès.");
                }
                Err(e) => {
                    tracing::error!("Erreur lors du chargement du modèle: {}", e);
                }
            }
        });

        self.active_engine = "whisper_local".to_string();
        self.active_model_id = meta.id;
        cx.notify();
    }

    // Switch to Server Model (automatically starts local ASR server)
    fn switch_to_server_model(&mut self, meta: ModelMetadata, window: &mut Window, cx: &mut Context<Self>) {
        let model_name = meta.default_model_name.clone();
        let display_name = meta.name.clone();
        let model_id = meta.id.clone();
        let api_url = if meta.default_api_url.is_empty() {
            "http://127.0.0.1:8000/v1/audio/transcriptions".to_string()
        } else {
            meta.default_api_url.clone()
        };

        // Automatically start local ASR server
        self.start_local_asr_server(&model_name, cx);

        let state = self.state.clone();
        let url_clone = api_url.clone();
        let id_clone = model_id.clone();
        let disp_clone = display_name.clone();

        self.tokio_handle.spawn(async move {
            let mut cfg = state.config.write().await;
            cfg.stt.engine_type = "remote_asr".to_string();
            cfg.stt.api_url = url_clone;
            cfg.stt.api_key = None;
            cfg.stt.model_id = id_clone;
            cfg.stt.model_name = disp_clone;
            let _ = cfg.save_default();
        });

        self.active_engine = "remote_asr".to_string();
        self.active_model_id = model_id;
        self.api_url_input = api_url;

        window.push_notification(
            Notification::info(format!("Modèle {} activé — Serveur ASR auto-démarré sur port 8000.", display_name)),
            cx,
        );
        cx.notify();
    }

    // Switch to Remote / Local Speaches ASR server
    fn switch_remote_asr(&mut self, url: String, key: Option<String>, window: &mut Window, cx: &mut Context<Self>) {
        let state = self.state.clone();
        let url_clone = url.clone();
        let key_clone = key.clone();

        self.tokio_handle.spawn(async move {
            let mut cfg = state.config.write().await;
            cfg.stt.engine_type = "remote_asr".to_string();
            cfg.stt.api_url = url_clone;
            cfg.stt.api_key = key_clone;
            cfg.stt.model_id = "faster-whisper-server".to_string();
            cfg.stt.model_name = "Serveur ASR Local (Speaches)".to_string();
            let _ = cfg.save_default();
        });

        self.active_engine = "remote_asr".to_string();
        self.active_model_id = "faster-whisper-server".to_string();
        window.push_notification(
            Notification::info("Moteur ASR basculé sur Serveur Local (Speaches/Faster-Whisper)."),
            cx,
        );
        cx.notify();
    }

    // Download a model or activate server model
    fn download_model(&mut self, meta: ModelMetadata, window: &mut Window, cx: &mut Context<Self>) {
        if meta.download_url.is_none() || meta.default_engine != "whisper_local" {
            // It's a server model! Switch to it and auto-start ASR server
            self.switch_to_server_model(meta, window, cx);
            return;
        }

        let download_url = meta.download_url.clone().unwrap();
        let state = self.state.clone();
        let status = state.download_status.clone();
        let model_path = meta.default_model_name.clone();
        let model_id = meta.id.clone();
        let display_name = meta.name.clone();

        window.push_notification(
            Notification::info(format!("Démarrage du téléchargement : {}", meta.name)),
            cx,
        );

        let (use_gpu, gpu_backend, gpu_device, flash_attn, lang) = {
            let cfg = state.config.blocking_read();
            (
                cfg.stt.use_gpu,
                cfg.stt.gpu_backend.clone(),
                cfg.stt.gpu_device,
                cfg.stt.flash_attn,
                cfg.stt.language.clone(),
            )
        };

        let state_clone = state.clone();
        let path_clone = model_path.clone();
        let id_clone = model_id.clone();
        let name_clone = display_name.clone();
        let url_clone = download_url.clone();

        self.tokio_handle.spawn(async move {
            match download_model_with_progress(&model_path, &download_url, &model_id, status).await {
                Ok(_) => {
                    tracing::info!("Modèle téléchargé avec succès : {}", path_clone);
                    let reload_res = {
                        let mut engine = state_clone.whisper_engine.lock().await;
                        engine.set_language(&lang);
                        engine.reload_model_with_gpu_options(
                            &path_clone,
                            use_gpu,
                            &gpu_backend,
                            gpu_device,
                            flash_attn,
                        )
                    };
                    if reload_res.is_ok() {
                        let mut cfg = state_clone.config.write().await;
                        cfg.stt.engine_type = "whisper_local".to_string();
                        cfg.stt.model_id = id_clone;
                        cfg.stt.model_path = path_clone;
                        cfg.stt.model_url = url_clone;
                        cfg.stt.model_name = name_clone;
                        let _ = cfg.save_default();
                        tracing::info!("Modèle Whisper chargé et actif en mémoire.");
                    }
                }
                Err(e) => {
                    tracing::error!("Erreur lors du téléchargement du modèle {}: {}", id_clone, e);
                }
            }
        });

        self.is_downloading = true;
        cx.notify();
    }

    // Test Server Connection
    fn test_server_connection(&mut self, url: String, key: Option<String>, window: &mut Window, cx: &mut Context<Self>) {
        let client = crate::stt::RemoteAsrClient::new();
        let url_clone = url.clone();

        window.push_notification(
            Notification::info("Test de connexion au serveur ASR en cours..."),
            cx,
        );

        self.tokio_handle.spawn(async move {
            let res = client.health_check(&url_clone, key.as_deref()).await;
            if res {
                tracing::info!("Connexion au serveur ASR réussie !");
            } else {
                tracing::warn!("Échec de connexion au serveur ASR.");
            }
        });

        cx.notify();
    }

    // Start Python Local Server in background
    fn start_local_server(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.start_local_asr_server("Systran/faster-whisper-base", cx);
        window.push_notification(
            Notification::info("Serveur ASR local démarré sur http://127.0.0.1:8000"),
            cx,
        );
    }

    // Set Translation Languages
    fn set_languages(&mut self, source: String, target: String, cx: &mut Context<Self>) {
        let state = self.state.clone();
        let src = source.clone();
        let tgt = target.clone();

        self.configured_source_lang = source.clone();
        self.configured_target_lang = target.clone();
        self.latest_source_lang = if source == "auto" { "AUTO".to_string() } else { source.to_uppercase() };
        self.latest_target_lang = target.to_uppercase();

        self.tokio_handle.spawn(async move {
            let mut cfg = state.config.write().await;
            cfg.translation.source_lang = src.clone();
            cfg.translation.target_lang = tgt.clone();
            cfg.stt.language = src.clone();
            let _ = cfg.save_default();

            let mut trans = state.translation_service.write().await;
            trans.set_languages(src.clone(), tgt.clone());

            let mut engine = state.whisper_engine.lock().await;
            engine.set_language(&src);
            tracing::info!("Languages updated in memory and saved: source='{}', target='{}'", src, tgt);
        });

        cx.notify();
    }

    // ═══════════════════════════════════════════════════════════════
    // RENDER: TOP TITLEBAR
    // ═══════════════════════════════════════════════════════════════
    fn render_titlebar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let active_engine_label = if self.active_engine == "remote_asr" {
            let model_title = crate::stt::find_model(&self.active_model_id)
                .map(|m| m.name)
                .unwrap_or_else(|| self.active_model_id.clone());
            if model_title.is_empty() {
                "Serveur ASR Distant".to_string()
            } else {
                format!("Serveur ASR ({})", model_title)
            }
        } else {
            crate::stt::find_model(&self.active_model_id)
                .map(|m| m.name)
                .unwrap_or_else(|| "Whisper Local".to_string())
        };
        let hw_label = crate::stt::active_hardware_summary(
            self.use_gpu,
            &self.gpu_backend,
            self.gpu_device,
        );

        TitleBar::new().child(
            h_flex()
                .size_full()
                .items_center()
                .justify_between()
                .px_4()
                .bg(BG_MAIN)
                .child(
                    // App Branding
                    h_flex()
                        .gap_2p5()
                        .items_center()
                        .child(
                            img("docs/logo.png")
                                .size(px(24.))
                                .rounded(px(6.))
                        )
                        .child(
                            div()
                                .text_base()
                                .font_semibold()
                                .text_color(TEXT_MAIN)
                                .child("VoiceBridge Pro"),
                        ),
                )
                .child(
                    // Center Engine Chip
                    h_flex()
                        .gap_2()
                        .items_center()
                        .child(
                            div()
                                .size(px(6.))
                                .rounded_full()
                                .bg(ACCENT_GREEN),
                        )
                        .child(
                            div()
                                .text_xs()
                                .font_medium()
                                .text_color(TEXT_BODY)
                                .child(format!("{}  /  {}", hw_label, active_engine_label)),
                        ),
                )
                .child(
                    // Right OBS Chip & Settings Icon
                    h_flex()
                        .gap_3()
                        .items_center()
                        .child(
                            h_flex()
                                .gap_2()
                                .items_center()
                                .px_3()
                                .py_1()
                                .rounded(px(9999.))
                                .bg(BG_CARD)
                                .border_1()
                                .border_color(BORDER_DARK)
                                .child(
                                    div()
                                        .size(px(8.))
                                        .rounded_full()
                                        .bg(ACCENT_CYAN),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .font_medium()
                                        .text_color(TEXT_BODY)
                                        .child("OBS Live"),
                                ),
                        )
                        .child(
                            div()
                                .id("btn-settings")
                                .p_1p5()
                                .rounded(px(8.))
                                .cursor_pointer()
                                .hover(|s| s.bg(BG_CARD_HOVER))
                                .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                                    this.active_tab = GuiTab::Modeles;
                                    cx.notify();
                                }))
                                .child(
                                    Icon::new(IconName::Settings)
                                        .small()
                                        .text_color(TEXT_MUTED),
                                ),
                        ),
                ),
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // RENDER: SUB-HEADER ACTION ROW
    // ═══════════════════════════════════════════════════════════════
    fn render_action_row(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let is_rec = self.is_recording;
        let vol = (self.current_volume * 100.0).clamp(0.0, 100.0);

        // Compute equalizer heights based on live RMS volume
        let h1 = 4.0 + (vol * 0.12).min(10.0);
        let h2 = 6.0 + (vol * 0.16).min(12.0);
        let h3 = 8.0 + (vol * 0.20).min(14.0);
        let h4 = 5.0 + (vol * 0.14).min(11.0);
        let h5 = 7.0 + (vol * 0.18).min(13.0);

        h_flex()
            .w_full()
            .items_center()
            .justify_between()
            .px_6()
            .py_3()
            .bg(BG_PANEL)
            .border_b_1()
            .border_color(BORDER_SUBTLE)
            .child(
                // Left Mic & Dynamic Equalizer (matches screen.png)
                h_flex()
                    .gap_3()
                    .items_center()
                    .child(
                        div()
                            .size(px(40.))
                            .rounded_full()
                            .bg(rgba_hex(0x231438))
                            .border_1()
                            .border_color(rgba_hex(0x4c1d95))
                            .flex()
                            .items_center()
                            .justify_center()
                            .child(
                                Icon::new(IconName::Mic)
                                    .small()
                                    .text_color(rgba_hex(0xc084fc)),
                            ),
                    )
                    .child(
                        v_flex()
                            .gap_1()
                            .child(
                                h_flex()
                                    .gap_2()
                                    .items_center()
                                    .child(
                                        div()
                                            .text_sm()
                                            .font_semibold()
                                            .text_color(if is_rec { TEXT_MAIN } else { TEXT_MUTED })
                                            .child(if is_rec { "Écoute en cours" } else { "En pause" }),
                                    )
                                    .child(
                                        div()
                                            .size(px(6.))
                                            .rounded_full()
                                            .bg(if is_rec { ACCENT_CYAN } else { rgba_hex(0x6b7280) }),
                                    ),
                            )
                            .child(
                                // Animated 5-bar equalizer
                                h_flex()
                                    .items_end()
                                    .gap_1()
                                    .h(px(14.))
                                    .child(div().w(px(3.)).h(px(h1)).rounded_full().bg(ACCENT_CYAN))
                                    .child(div().w(px(3.)).h(px(h2)).rounded_full().bg(rgba_hex(0xa855f7)))
                                    .child(div().w(px(3.)).h(px(h3)).rounded_full().bg(ACCENT_MAGENTA))
                                    .child(div().w(px(3.)).h(px(h4)).rounded_full().bg(rgba_hex(0x818cf8)))
                                    .child(div().w(px(3.)).h(px(h5)).rounded_full().bg(ACCENT_GREEN)),
                            ),
                    ),
            )
            .child(
                // Center Prominent Pill Toggle Button (Arrêter / Démarrer)
                div()
                    .id("btn-toggle-rec")
                    .cursor_pointer()
                    .on_click(cx.listener(Self::toggle_recording))
                    .child(if is_rec {
                        h_flex()
                            .items_center()
                            .gap_2()
                            .px_6()
                            .py_2()
                            .rounded(px(9999.))
                            .bg(rgba_hex(0x1a0f12))
                            .border_1()
                            .border_color(rgba_hex(0xdc2626))
                            .hover(|s| s.bg(rgba_hex(0x2d1217)))
                            .child(div().size(px(8.)).rounded_full().bg(ACCENT_RED))
                            .child(
                                div()
                                    .text_sm()
                                    .font_medium()
                                    .text_color(rgba_hex(0xfecaca))
                                    .child("Arrêter"),
                            )
                    } else {
                        h_flex()
                            .items_center()
                            .gap_2()
                            .px_6()
                            .py_2()
                            .rounded(px(9999.))
                            .bg(rgba_hex(0x0e1f14))
                            .border_1()
                            .border_color(rgba_hex(0x16a34a))
                            .hover(|s| s.bg(rgba_hex(0x153320)))
                            .child(div().size(px(8.)).rounded_full().bg(ACCENT_GREEN))
                            .child(
                                div()
                                    .text_sm()
                                    .font_medium()
                                    .text_color(rgba_hex(0xbbf7d0))
                                    .child("Démarrer"),
                            )
                    }),
            )
            .child(
                // Right Lien OBS Copy Button
                div()
                    .id("btn-copy-obs")
                    .cursor_pointer()
                    .on_click(cx.listener(Self::copy_obs_url))
                    .child(
                        h_flex()
                            .items_center()
                            .gap_2()
                            .px_4()
                            .py_1p5()
                            .rounded(px(9999.))
                            .bg(BG_CARD)
                            .border_1()
                            .border_color(BORDER_DARK)
                            .hover(|s| s.bg(BG_CARD_HOVER))
                            .child(
                                Icon::new(IconName::Copy)
                                    .small()
                                    .text_color(TEXT_BODY),
                            )
                            .child(
                                div()
                                    .text_sm()
                                    .font_medium()
                                    .text_color(TEXT_BODY)
                                    .child("Lien OBS"),
                            ),
                    ),
            )
    }

    // ═══════════════════════════════════════════════════════════════
    // RENDER: TABS NAVIGATION BAR
    // ═══════════════════════════════════════════════════════════════
    fn render_tabs_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let tabs = [
            (GuiTab::Direct, "Direct"),
            (GuiTab::Modeles, "Modèles"),
            (GuiTab::Langues, "Langues"),
            (GuiTab::AudioVad, "Audio & VAD"),
            (GuiTab::Incrustation, "Incrustation"),
        ];

        h_flex()
            .w_full()
            .items_center()
            .justify_between()
            .px_6()
            .py_3()
            .bg(BG_MAIN)
            .child(
                // Segmented Pill Container
                h_flex()
                    .items_center()
                    .p_1()
                    .rounded(px(9999.))
                    .bg(rgba_hex(0x141418))
                    .border_1()
                    .border_color(BORDER_DARK)
                    .children(tabs.into_iter().map(|(tab, label)| {
                        let is_active = self.active_tab == tab;
                        div()
                            .id(SharedString::from(format!("tab-btn-{}", label)))
                            .cursor_pointer()
                            .px_4()
                            .py_1p5()
                            .rounded(px(9999.))
                            .font_medium()
                            .text_sm()
                            .map(|this| {
                                if is_active {
                                    this.bg(rgba_hex(0x26262e)).text_color(TEXT_MAIN)
                                } else {
                                    this.text_color(TEXT_MUTED).hover(|s| s.bg(rgba_hex(0x1c1c22)).text_color(TEXT_BODY))
                                }
                            })
                            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                this.active_tab = tab;
                                cx.notify();
                            }))
                            .child(label)
                    })),
            )
            .child(
                // Right Status Badge
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .size(px(6.))
                            .rounded_full()
                            .bg(ACCENT_GREEN),
                    )
                    .child(
                        div()
                            .text_xs()
                            .font_medium()
                            .text_color(TEXT_MUTED)
                            .child("Silero VAD · 300ms buffer"),
                    ),
            )
    }

    // ═══════════════════════════════════════════════════════════════
    // TAB: DIRECT (LIVE STAGE & HISTORY - SCREENSHOT MATCH)
    // ═══════════════════════════════════════════════════════════════
    fn render_direct_tab(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let count_str = format!("{} entrées", self.history_entries.len());

        v_flex()
            .size_full()
            .gap_4()
            .p_6()
            .child(
                // Dual Cards Row (Both cards locked to 50% so Card 2 never disappears)
                h_flex()
                    .w_full()
                    .gap_5()
                    .child(
                        // Card 1: Voix captée (Source Speech)
                        v_flex()
                            .flex_1()
                            .w(relative(0.5))
                            .min_w_0()
                            .h(px(220.))
                            .overflow_hidden()
                            .rounded(px(16.))
                            .bg(BG_CARD)
                            .border_1()
                            .border_color(BORDER_DARK)
                            .p_5()
                            .justify_between()
                            .child(
                                // Header
                                h_flex()
                                    .items_center()
                                    .justify_between()
                                    .child(
                                        h_flex()
                                            .gap_2()
                                            .items_center()
                                            .child(
                                                div()
                                                    .px_2()
                                                    .py_0p5()
                                                    .rounded(px(6.))
                                                    .bg(rgba_hex(0x26262c))
                                                    .text_xs()
                                                    .font_bold()
                                                    .text_color(rgba_hex(0xe5e2e1))
                                                    .child(self.latest_source_lang.clone()),
                                            )
                                            .child(
                                                div()
                                                    .text_sm()
                                                    .text_color(rgba_hex(0xa0a0ab))
                                                    .child("Voix captée"),
                                            ),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(TEXT_DIM)
                                            .font_family(cx.theme().mono_font_family.clone())
                                            .child(format!("{} ms", self.latest_stt_ms)),
                                    ),
                            )
                            .child(
                                // Text body (Responsive size + vertical scroll if long)
                                div()
                                    .flex_1()
                                    .w_full()
                                    .min_w_0()
                                    .min_h_0()
                                    .my_1()
                                    .overflow_y_scrollbar()
                                    .child(
                                        div()
                                            .w_full()
                                            .min_w_0()
                                            .whitespace_normal()
                                            .font_medium()
                                            .text_color(TEXT_MAIN)
                                            .map(|this| {
                                                let len = self.latest_source_text.chars().count();
                                                if len > 120 {
                                                    this.text_sm().line_height(px(20.))
                                                } else if len > 60 {
                                                    this.text_base().line_height(px(24.))
                                                } else if len > 30 {
                                                    this.text_lg().line_height(px(28.))
                                                } else {
                                                    this.text_2xl().line_height(px(34.))
                                                }
                                            })
                                            .child(self.latest_source_text.clone()),
                                    ),
                            )
                            .child(
                                // Divider line & Footer
                                v_flex()
                                    .w_full()
                                    .gap_3()
                                    .child(div().w_full().h(px(1.)).bg(BORDER_DARK))
                                    .child(
                                        h_flex()
                                            .items_center()
                                            .justify_between()
                                            .child(
                                                h_flex()
                                                    .gap_2()
                                                    .items_center()
                                                    .child(div().size(px(6.)).rounded_full().bg(ACCENT_GREEN))
                                                    .child(
                                                        div()
                                                            .text_xs()
                                                            .text_color(ACCENT_GREEN)
                                                            .child(format!("{:.1}% de confiance", self.latest_confidence)),
                                                    ),
                                            )
                                            .child(
                                                div()
                                                    .text_xs()
                                                    .text_color(TEXT_DIM)
                                                    .child(if self.configured_source_lang == "auto" {
                                                        format!("Source : Auto ({})", self.latest_source_lang)
                                                    } else {
                                                        format!("Source : {}", self.configured_source_lang.to_uppercase())
                                                    }),
                                            ),
                                    ),
                            ),
                    )
                    .child(
                        // Card 2: Traduction instantanée (Gradient Glass Card)
                        v_flex()
                            .flex_1()
                            .w(relative(0.5))
                            .min_w_0()
                            .h(px(220.))
                            .overflow_hidden()
                            .rounded(px(16.))
                            .bg(linear_gradient(
                                135.,
                                linear_color_stop(hsla(305. / 360., 0.52, 0.22, 0.95), 0.),
                                linear_color_stop(hsla(188. / 360., 0.65, 0.20, 0.95), 1.),
                            ))
                            .border_1()
                            .border_color(hsla(188. / 360., 0.5, 0.35, 0.45))
                            .p_5()
                            .justify_between()
                            .child(
                                // Header
                                h_flex()
                                    .items_center()
                                    .justify_between()
                                    .child(
                                        h_flex()
                                            .gap_2()
                                            .items_center()
                                            .child(
                                                div()
                                                    .px_2()
                                                    .py_0p5()
                                                    .rounded(px(6.))
                                                    .bg(rgba_hex(0x26262c))
                                                    .text_xs()
                                                    .font_bold()
                                                    .text_color(rgba_hex(0xe5e2e1))
                                                    .child(self.latest_target_lang.clone()),
                                            )
                                            .child(
                                                div()
                                                    .text_sm()
                                                    .text_color(rgba_hex(0xe2e8f0))
                                                    .child("Traduction instantanée"),
                                            ),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(rgba_hex(0x67e8f9))
                                            .font_family(cx.theme().mono_font_family.clone())
                                            .child(format!("{} ms", self.latest_trans_ms)),
                                    ),
                            )
                            .child(
                                // Text body (Responsive size + vertical scroll if long)
                                div()
                                    .flex_1()
                                    .w_full()
                                    .min_w_0()
                                    .min_h_0()
                                    .my_1()
                                    .overflow_y_scrollbar()
                                    .child(
                                        div()
                                            .w_full()
                                            .min_w_0()
                                            .whitespace_normal()
                                            .font_medium()
                                            .text_color(TEXT_MAIN)
                                            .map(|this| {
                                                let len = self.latest_trans_text.chars().count();
                                                if len > 120 {
                                                    this.text_sm().line_height(px(20.))
                                                } else if len > 60 {
                                                    this.text_base().line_height(px(24.))
                                                } else if len > 30 {
                                                    this.text_lg().line_height(px(28.))
                                                } else {
                                                    this.text_2xl().line_height(px(34.))
                                                }
                                            })
                                            .child(self.latest_trans_text.clone()),
                                    ),
                            )
                            .child(
                                // Footer
                                h_flex()
                                    .items_center()
                                    .justify_between()
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(rgba_hex(0x94a3b8))
                                            .child(format!("Cible : {} • OBS synchronisé", self.configured_target_lang.to_uppercase())),
                                    )
                                    .child(
                                        h_flex()
                                            .gap_2()
                                            .items_center()
                                            .child(
                                                div()
                                                    .id("btn-speak")
                                                    .p_1()
                                                    .rounded(px(6.))
                                                    .cursor_pointer()
                                                    .hover(|s| s.bg(rgba_hex(0x38bdf820)))
                                                    .child(
                                                        Icon::new(IconName::Volume2)
                                                            .small()
                                                            .text_color(rgba_hex(0xe2e8f0)),
                                                    ),
                                            )
                                            .child(
                                                div()
                                                    .id("btn-copy-trans")
                                                    .p_1()
                                                    .rounded(px(6.))
                                                    .cursor_pointer()
                                                    .hover(|s| s.bg(rgba_hex(0x38bdf820)))
                                                    .on_click(cx.listener(Self::copy_translation))
                                                    .child(
                                                        Icon::new(IconName::Copy)
                                                            .small()
                                                            .text_color(rgba_hex(0xe2e8f0)),
                                                    ),
                                            ),
                                    ),
                            ),
                    ),
            )
            .child(
                // History Section (Historique)
                v_flex()
                    .flex_1()
                    .min_h(px(200.))
                    .rounded(px(16.))
                    .bg(BG_CARD)
                    .border_1()
                    .border_color(BORDER_DARK)
                    .p_4()
                    .gap_3()
                    .child(
                        // Historique Header
                        h_flex()
                            .items_center()
                            .justify_between()
                            .child(
                                h_flex()
                                    .gap_3()
                                    .items_center()
                                    .child(
                                        div()
                                            .text_sm()
                                            .font_semibold()
                                            .text_color(TEXT_MAIN)
                                            .child("Historique"),
                                    )
                                    .child(
                                        div()
                                            .px_2p5()
                                            .py_0p5()
                                            .rounded(px(9999.))
                                            .bg(rgba_hex(0x26262c))
                                            .text_xs()
                                            .text_color(rgba_hex(0xa0a0ab))
                                            .child(count_str),
                                    ),
                            )
                            .child(
                                h_flex()
                                    .gap_2()
                                    .items_center()
                                    .child(
                                        div()
                                            .id("btn-export-srt")
                                            .cursor_pointer()
                                            .px_3()
                                            .py_1()
                                            .rounded(px(6.))
                                            .bg(rgba_hex(0x1f1f24))
                                            .border_1()
                                            .border_color(BORDER_DARK)
                                            .hover(|s| s.bg(BG_CARD_HOVER))
                                            .on_click(cx.listener(Self::export_srt))
                                            .child(
                                                h_flex()
                                                    .gap_1p5()
                                                    .items_center()
                                                    .child(Icon::new(IconName::Download).xsmall().text_color(TEXT_BODY))
                                                    .child(div().text_xs().font_medium().text_color(TEXT_BODY).child("SRT")),
                                            ),
                                    )
                                    .child(
                                        div()
                                            .id("btn-clear-hist")
                                            .cursor_pointer()
                                            .px_3()
                                            .py_1()
                                            .rounded(px(6.))
                                            .bg(rgba_hex(0x1f1f24))
                                            .border_1()
                                            .border_color(BORDER_DARK)
                                            .hover(|s| s.bg(BG_CARD_HOVER))
                                            .on_click(cx.listener(Self::clear_history))
                                            .child(
                                                div().text_xs().font_medium().text_color(TEXT_BODY).child("Effacer"),
                                            ),
                                    ),
                            ),
                    )
                    .child(
                        // Scrollable List of History Cards
                        div()
                            .flex_1()
                            .min_h_0()
                            .child(
                                v_flex()
                                    .size_full()
                                    .gap_2()
                                    .overflow_y_scrollbar()
                                    .when(self.history_entries.is_empty(), |this| {
                                        this.items_center().justify_center().py_8().child(
                                            div().text_xs().text_color(TEXT_MUTED).child("Aucune transcription enregistrée pour le moment."),
                                        )
                                    })
                                    .children(self.history_entries.iter().map(|entry| {
                                        let duration_sec = (entry.total_duration_ms as f64) / 1000.0;
                                        h_flex()
                                            .w_full()
                                            .items_start()
                                            .justify_between()
                                            .gap_4()
                                            .px_4()
                                            .py_3()
                                            .rounded(px(10.))
                                            .bg(rgba_hex(0x1a1a1e))
                                            .border_1()
                                            .border_color(BORDER_SUBTLE)
                                            .hover(|s| s.bg(rgba_hex(0x202026)))
                                            .child(
                                                // Left side: Timestamp (pinned width, top-aligned)
                                                div()
                                                    .w(px(68.))
                                                    .flex_shrink_0()
                                                    .pt(px(2.))
                                                    .text_xs()
                                                    .font_family(cx.theme().mono_font_family.clone())
                                                    .text_color(TEXT_DIM)
                                                    .child(entry.timestamp.clone()),
                                            )
                                            .child(
                                                // Middle: Full text (original + translated) with multi-line wrapping
                                                v_flex()
                                                    .flex_1()
                                                    .min_w_0()
                                                    .gap_1()
                                                    .child(
                                                        div()
                                                            .w_full()
                                                            .min_w_0()
                                                            .whitespace_normal()
                                                            .text_sm()
                                                            .font_medium()
                                                            .text_color(TEXT_MAIN)
                                                            .line_height(px(20.))
                                                            .child(entry.original_text.clone()),
                                                    )
                                                    .child(
                                                        h_flex()
                                                            .w_full()
                                                            .min_w_0()
                                                            .items_start()
                                                            .gap_1p5()
                                                            .child(
                                                                div()
                                                                    .flex_shrink_0()
                                                                    .pt(px(2.))
                                                                    .text_xs()
                                                                    .text_color(ACCENT_CYAN)
                                                                    .child("→"),
                                                            )
                                                            .child(
                                                                div()
                                                                    .flex_1()
                                                                    .min_w_0()
                                                                    .whitespace_normal()
                                                                    .text_sm()
                                                                    .text_color(rgba_hex(0x93c5fd))
                                                                    .line_height(px(20.))
                                                                    .child(entry.translated_text.clone()),
                                                            ),
                                                    ),
                                            )
                                            .child(
                                                // Right side: Duration + status dot (NEVER shrunk or pushed off!)
                                                h_flex()
                                                    .flex_shrink_0()
                                                    .pt(px(2.))
                                                    .gap_2()
                                                    .items_center()
                                                    .child(
                                                        div()
                                                            .text_xs()
                                                            .text_color(TEXT_DIM)
                                                            .font_family(cx.theme().mono_font_family.clone())
                                                            .child(format!("{:.1}s", duration_sec)),
                                                    )
                                                    .child(div().size(px(6.)).rounded_full().bg(ACCENT_CYAN)),
                                            )
                                    })),
                            ),
                    ),
            )
    }

    // ═══════════════════════════════════════════════════════════════
    // TAB: MODÈLES (CATALOG, GPU/CPU HARDWARE & REMOTE SPEACHES)
    // ═══════════════════════════════════════════════════════════════
    fn render_modeles_tab(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let active_hw_summary = crate::stt::active_hardware_summary(
            self.use_gpu,
            &self.gpu_backend,
            self.gpu_device,
        );
        let cur_backend = if !self.use_gpu {
            "cpu".to_string()
        } else {
            self.gpu_backend.to_ascii_lowercase()
        };
        let cur_dev = self.gpu_device;
        let cur_flash = self.flash_attn;
        let cur_threads = self.n_threads;

        #[cfg(target_os = "macos")]
        let backend_options: &[(&str, &str, &str)] = &[
            ("auto", "Auto (GPU Recommandé)", "Sélectionne automatiquement Metal GPU"),
            ("metal", "Apple Metal GPU", "Accélération matérielle unifiée Apple Silicon"),
            ("cpu", "CPU Uniquement", "Exécution multithread sur processeur"),
        ];
        #[cfg(not(target_os = "macos"))]
        let backend_options: &[(&str, &str, &str)] = &[
            ("auto", "Auto (GPU Recommandé)", "Détecte automatiquement votre GPU NVIDIA / AMD / Intel"),
            ("vulkan", "Vulkan GPU (NVIDIA / AMD)", "Accélération GPU universelle directe sans CUDA externe"),
            ("cuda", "NVIDIA CUDA GPU", "Kernels CUDA natifs dédiés aux cartes NVIDIA GeForce / RTX"),
            ("cpu", "CPU Uniquement", "Force le calcul multithread sur processeur"),
        ];

        v_flex()
            .size_full()
            .p_6()
            .gap_5()
            .overflow_y_scrollbar()
            .child(
                // Section 1: Engine Switch Cards
                v_flex()
                    .gap_3()
                    .child(
                        div().text_base().font_semibold().text_color(TEXT_MAIN).child("Moteur de Reconnaissance Vocale (ASR)"),
                    )
                    .child(
                        h_flex()
                            .gap_4()
                            .child(
                                // Whisper Local GPU/CPU Card
                                v_flex()
                                    .id("card-whisper-local")
                                    .flex_1()
                                    .p_4()
                                    .rounded(px(12.))
                                    .border_1()
                                    .cursor_pointer()
                                    .map(|this| {
                                        if self.active_engine == "whisper_local" {
                                            this.bg(rgba_hex(0x19221f)).border_color(ACCENT_GREEN)
                                        } else {
                                            this.bg(BG_CARD).border_color(BORDER_DARK).hover(|s| s.bg(BG_CARD_HOVER))
                                        }
                                    })
                                    .on_click(cx.listener(|this, _: &ClickEvent, window, cx| {
                                        if this.active_engine != "whisper_local" {
                                            let catalog = this.models_catalog.clone();
                                            if let Some(target) = catalog.iter().find(|m| m.default_engine == "whisper_local" && is_model_installed(m)) {
                                                this.switch_local_whisper(target.clone(), window, cx);
                                            } else if let Some(target) = catalog.iter().find(|m| m.default_engine == "whisper_local") {
                                                this.switch_local_whisper(target.clone(), window, cx);
                                            }
                                        }
                                    }))
                                    .gap_2()
                                    .child(
                                        h_flex()
                                            .justify_between()
                                            .items_center()
                                            .child(
                                                div().text_sm().font_semibold().text_color(TEXT_MAIN).child(
                                                    format!("Whisper Local ({})", active_hw_summary),
                                                ),
                                            )
                                            .child(
                                                if self.active_engine == "whisper_local" {
                                                    div().px_2().py_0p5().rounded(px(9999.)).bg(rgba_hex(0x166534)).text_xs().text_color(rgba_hex(0xbbf7d0)).child("Actif")
                                                } else {
                                                    div().px_2().py_0p5().rounded(px(9999.)).bg(rgba_hex(0x26262c)).text_xs().text_color(TEXT_MUTED).child("Cliquer pour activer")
                                                },
                                            ),
                                    )
                                    .child(
                                        div().text_xs().text_color(TEXT_MUTED).child("100% Hors-ligne, ultra-rapide sur NVIDIA GPU (Vulkan / CUDA), Metal GPU ou CPU multithread."),
                                    ),
                            )
                            .child(
                                // Speaches / Local Server Card
                                v_flex()
                                    .id("card-remote-asr")
                                    .flex_1()
                                    .p_4()
                                    .rounded(px(12.))
                                    .border_1()
                                    .cursor_pointer()
                                    .map(|this| {
                                        if self.active_engine == "remote_asr" {
                                            this.bg(rgba_hex(0x181f28)).border_color(ACCENT_CYAN)
                                        } else {
                                            this.bg(BG_CARD).border_color(BORDER_DARK).hover(|s| s.bg(BG_CARD_HOVER))
                                        }
                                    })
                                    .on_click(cx.listener(|this, _: &ClickEvent, window, cx| {
                                        if this.active_engine != "remote_asr" {
                                            let catalog = this.models_catalog.clone();
                                            if let Some(target) = catalog.iter().find(|m| m.default_engine != "whisper_local") {
                                                this.switch_to_server_model(target.clone(), window, cx);
                                            } else {
                                                this.start_local_server(window, cx);
                                                let url = this.api_url_input.clone();
                                                this.switch_remote_asr(url, None, window, cx);
                                            }
                                        }
                                    }))
                                    .gap_2()
                                    .child(
                                        h_flex()
                                            .justify_between()
                                            .items_center()
                                            .child(
                                                div().text_sm().font_semibold().text_color(TEXT_MAIN).child("Serveur Local / OpenAI API (Speaches)"),
                                            )
                                            .child(
                                                if self.active_engine == "remote_asr" {
                                                    div().px_2().py_0p5().rounded(px(9999.)).bg(rgba_hex(0x1e40af)).text_xs().text_color(rgba_hex(0xbfdbfe)).child("Actif")
                                                } else {
                                                    div().px_2().py_0p5().rounded(px(9999.)).bg(rgba_hex(0x26262c)).text_xs().text_color(TEXT_MUTED).child("Cliquer pour activer")
                                                },
                                            ),
                                    )
                                    .child(
                                        div().text_xs().text_color(TEXT_MUTED).child("Serveur ASR local haute performance (Faster-Whisper, Qwen, Voxtral, Moonshine)."),
                                    ),
                            ),
                    ),
            )
            .child(
                // Section 1.5: Accélération Matérielle (GPU NVIDIA / Vulkan / CUDA / Metal / CPU)
                v_flex()
                    .p_5()
                    .rounded(px(14.))
                    .bg(BG_CARD)
                    .border_1()
                    .border_color(BORDER_DARK)
                    .gap_4()
                    .child(
                        h_flex()
                            .justify_between()
                            .items_center()
                            .child(
                                v_flex()
                                    .gap_0p5()
                                    .child(
                                        div()
                                            .text_sm()
                                            .font_semibold()
                                            .text_color(TEXT_MAIN)
                                            .child("Accélération Matérielle (GPU NVIDIA / Vulkan / CUDA / CPU)"),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(TEXT_MUTED)
                                            .child("Choisissez le backend matériel, la carte graphique dédiée et les options d'inférence locale."),
                                    ),
                            )
                            .child(
                                div()
                                    .px_3()
                                    .py_1()
                                    .rounded(px(8.))
                                    .bg(if self.use_gpu && crate::stt::has_compiled_gpu_backend() {
                                        rgba_hex(0x143322)
                                    } else {
                                        rgba_hex(0x26262c)
                                    })
                                    .border_1()
                                    .border_color(if self.use_gpu && crate::stt::has_compiled_gpu_backend() {
                                        ACCENT_GREEN
                                    } else {
                                        BORDER_DARK
                                    })
                                    .text_xs()
                                    .font_bold()
                                    .text_color(if self.use_gpu && crate::stt::has_compiled_gpu_backend() {
                                        ACCENT_GREEN
                                    } else {
                                        TEXT_BODY
                                    })
                                    .child(format!("Actif : {}", active_hw_summary)),
                            ),
                    )
                    .child(
                        // Row 1: Backend Mode Selector
                        h_flex()
                            .gap_3()
                            .children(backend_options.iter().map(|(val, title, hint)| {
                                let is_sel = cur_backend == *val;
                                let val_str = (*val).to_string();
                                div()
                                    .flex_1()
                                    .id(SharedString::from(format!("gpu-backend-{}", val)))
                                    .cursor_pointer()
                                    .p_3()
                                    .rounded(px(8.))
                                    .map(|this| {
                                        if is_sel {
                                            this.bg(rgba_hex(0x1e3a5f)).border_1().border_color(ACCENT_CYAN)
                                        } else {
                                            this.bg(rgba_hex(0x1a1a1e)).border_1().border_color(BORDER_SUBTLE).hover(|s| s.bg(BG_CARD_HOVER))
                                        }
                                    })
                                    .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                                        let next_use_gpu = val_str != "cpu";
                                        this.apply_gpu_settings(
                                            next_use_gpu,
                                            val_str.clone(),
                                            cur_dev,
                                            cur_flash,
                                            cur_threads,
                                            window,
                                            cx,
                                        );
                                    }))
                                    .child(
                                        v_flex()
                                            .gap_0p5()
                                            .child(
                                                div()
                                                    .text_xs()
                                                    .font_semibold()
                                                    .text_color(if is_sel { ACCENT_CYAN } else { TEXT_MAIN })
                                                    .child(*title),
                                            )
                                            .child(div().text_xs().text_color(TEXT_MUTED).child(*hint)),
                                    )
                            })),
                    )
                    .when(!self.gpu_devices.is_empty(), |this| {
                        this.child(
                            v_flex()
                                .gap_2()
                                .child(
                                    div()
                                        .text_xs()
                                        .font_semibold()
                                        .text_color(TEXT_BODY)
                                        .child("Carte(s) Graphique(s) Détectée(s) :"),
                                )
                                .child(
                                    h_flex()
                                        .gap_3()
                                        .children(self.gpu_devices.iter().map(|gpu| {
                                            let is_sel = self.use_gpu && self.gpu_device == gpu.index;
                                            let idx = gpu.index;
                                            let backend_lower = gpu.backend.to_ascii_lowercase();
                                            let vram_str = if gpu.vram_total_mb > 0 {
                                                format!(
                                                    "{} Mo libres / {} Mo VRAM",
                                                    gpu.vram_free_mb, gpu.vram_total_mb
                                                )
                                            } else {
                                                format!("Périphérique #{} ({})", gpu.index, gpu.dev_name)
                                            };
                                            div()
                                                .flex_1()
                                                .id(SharedString::from(format!("gpu-dev-{}", idx)))
                                                .cursor_pointer()
                                                .p_3()
                                                .rounded(px(8.))
                                                .map(|el| {
                                                    if is_sel {
                                                        el.bg(rgba_hex(0x163520)).border_1().border_color(ACCENT_GREEN)
                                                    } else {
                                                        el.bg(rgba_hex(0x1a1a1e)).border_1().border_color(BORDER_SUBTLE).hover(|s| s.bg(BG_CARD_HOVER))
                                                    }
                                                })
                                                .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                                                    let next_backend = if this.gpu_backend == "cpu" {
                                                        backend_lower.clone()
                                                    } else {
                                                        this.gpu_backend.clone()
                                                    };
                                                    this.apply_gpu_settings(
                                                        true,
                                                        next_backend,
                                                        idx,
                                                        cur_flash,
                                                        cur_threads,
                                                        window,
                                                        cx,
                                                    );
                                                }))
                                                .child(
                                                    h_flex()
                                                        .justify_between()
                                                        .items_center()
                                                        .child(
                                                            v_flex()
                                                                .gap_0p5()
                                                                .child(
                                                                    div()
                                                                        .text_xs()
                                                                        .font_semibold()
                                                                        .text_color(if is_sel { ACCENT_GREEN } else { TEXT_MAIN })
                                                                        .child(format!(
                                                                            "GPU #{}: {} ({})",
                                                                            gpu.index, gpu.description, gpu.backend
                                                                        )),
                                                                )
                                                                .child(
                                                                    div()
                                                                        .text_xs()
                                                                        .text_color(TEXT_MUTED)
                                                                        .child(vram_str),
                                                                ),
                                                        )
                                                        .child(
                                                            if is_sel {
                                                                div()
                                                                    .px_2()
                                                                    .py_0p5()
                                                                    .rounded(px(6.))
                                                                    .bg(rgba_hex(0x166534))
                                                                    .text_xs()
                                                                    .text_color(rgba_hex(0xbbf7d0))
                                                                    .child("Sélectionné")
                                                            } else {
                                                                div()
                                                            },
                                                        ),
                                                )
                                        })),
                                ),
                        )
                    })
                    .child(
                        // Row 3: Flash Attention & CPU Threads
                        h_flex()
                            .gap_4()
                            .items_center()
                            .justify_between()
                            .child(
                                // Flash Attention Toggle
                                div()
                                    .id("btn-toggle-flash-attn")
                                    .cursor_pointer()
                                    .px_3p5()
                                    .py_2()
                                    .rounded(px(8.))
                                    .map(|this| {
                                        if self.flash_attn {
                                            this.bg(rgba_hex(0x1e3a5f)).border_1().border_color(ACCENT_CYAN)
                                        } else {
                                            this.bg(rgba_hex(0x1a1a1e)).border_1().border_color(BORDER_SUBTLE).hover(|s| s.bg(BG_CARD_HOVER))
                                        }
                                    })
                                    .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                                        let next_flash = !this.flash_attn;
                                        let use_gpu = this.use_gpu;
                                        let backend = this.gpu_backend.clone();
                                        this.apply_gpu_settings(
                                            use_gpu,
                                            backend,
                                            cur_dev,
                                            next_flash,
                                            cur_threads,
                                            window,
                                            cx,
                                        );
                                    }))
                                    .child(
                                        h_flex()
                                            .gap_2()
                                            .items_center()
                                            .child(
                                                div()
                                                    .size(px(8.))
                                                    .rounded_full()
                                                    .bg(if self.flash_attn { ACCENT_CYAN } else { rgba_hex(0x6b7280) }),
                                            )
                                            .child(
                                                div()
                                                    .text_xs()
                                                    .font_semibold()
                                                    .text_color(if self.flash_attn { ACCENT_CYAN } else { TEXT_BODY })
                                                    .child(if self.flash_attn {
                                                        "⚡ Flash Attention GPU : Activé"
                                                    } else {
                                                        "⚡ Flash Attention GPU : Désactivé"
                                                    }),
                                            ),
                                    ),
                            )
                            .child(
                                // CPU Threads Selector
                                h_flex()
                                    .gap_2()
                                    .items_center()
                                    .child(
                                        div()
                                            .text_xs()
                                            .font_medium()
                                            .text_color(TEXT_MUTED)
                                            .child("Threads CPU :"),
                                    )
                                    .children([2usize, 4, 8, 12, 16].into_iter().map(|th| {
                                        let is_sel = self.n_threads == th;
                                        div()
                                            .id(SharedString::from(format!("cpu-threads-{}", th)))
                                            .cursor_pointer()
                                            .px_2p5()
                                            .py_1()
                                            .rounded(px(6.))
                                            .map(|this| {
                                                if is_sel {
                                                    this.bg(rgba_hex(0x1e3a5f)).border_1().border_color(ACCENT_CYAN)
                                                } else {
                                                    this.bg(rgba_hex(0x1a1a1e)).border_1().border_color(BORDER_SUBTLE).hover(|s| s.bg(BG_CARD_HOVER))
                                                }
                                            })
                                            .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                                                let use_gpu = this.use_gpu;
                                                let backend = this.gpu_backend.clone();
                                                this.apply_gpu_settings(
                                                    use_gpu,
                                                    backend,
                                                    cur_dev,
                                                    cur_flash,
                                                    th,
                                                    window,
                                                    cx,
                                                );
                                            }))
                                            .child(
                                                div()
                                                    .text_xs()
                                                    .font_bold()
                                                    .text_color(if is_sel { ACCENT_CYAN } else { TEXT_BODY })
                                                    .child(format!("{}", th)),
                                            )
                                    })),
                            ),
                    ),
            )
            .child(
                // Section 2: Local Server Management
                v_flex()
                    .p_5()
                    .rounded(px(14.))
                    .bg(BG_CARD)
                    .border_1()
                    .border_color(BORDER_DARK)
                    .gap_4()
                    .child(
                        h_flex()
                            .justify_between()
                            .items_center()
                            .child(
                                div().text_sm().font_semibold().text_color(TEXT_MAIN).child("Gestionnaire de Serveur ASR Local"),
                            )
                            .child(
                                div().text_xs().text_color(TEXT_MUTED).child(format!("Statut : {}", self.local_server_status)),
                            ),
                    )
                    .child(
                        h_flex()
                            .gap_3()
                            .child(
                                div()
                                    .id("btn-start-local")
                                    .cursor_pointer()
                                    .px_4()
                                    .py_2()
                                    .rounded(px(8.))
                                    .bg(rgba_hex(0x22c55e))
                                    .border_1()
                                    .border_color(ACCENT_GREEN)
                                    .hover(|s| s.bg(rgba_hex(0x16a34a)))
                                    .on_click(cx.listener(|this, _: &ClickEvent, window, cx| {
                                        this.start_local_server(window, cx);
                                    }))
                                    .child(
                                        div().text_xs().font_semibold().text_color(rgba_hex(0xffffff)).child("🚀 Démarrer Serveur Local (Port 8000)"),
                                    ),
                            )
                            .child(
                                div()
                                    .id("btn-stop-local")
                                    .cursor_pointer()
                                    .px_4()
                                    .py_2()
                                    .rounded(px(8.))
                                    .bg(rgba_hex(0x7f1d1d))
                                    .border_1()
                                    .border_color(ACCENT_RED)
                                    .hover(|s| s.bg(rgba_hex(0x991b1b)))
                                    .on_click(cx.listener(|this, _: &ClickEvent, window, cx| {
                                        this.stop_local_asr_server(cx);
                                        window.push_notification(Notification::info("Serveur ASR local arrêté."), cx);
                                    }))
                                    .child(
                                        div().text_xs().font_semibold().text_color(rgba_hex(0xffffff)).child("⏹️ Arrêter Serveur"),
                                    ),
                            )
                            .child(
                                div()
                                    .id("btn-test-server")
                                    .cursor_pointer()
                                    .px_4()
                                    .py_2()
                                    .rounded(px(8.))
                                    .bg(rgba_hex(0x38bdf8))
                                    .border_1()
                                    .border_color(ACCENT_CYAN)
                                    .hover(|s| s.bg(rgba_hex(0x0284c7)))
                                    .on_click(cx.listener(|this, _: &ClickEvent, window, cx| {
                                        let url = this.api_url_input.clone();
                                        let key = if this.api_key_input.is_empty() { None } else { Some(this.api_key_input.clone()) };
                                        this.test_server_connection(url, key, window, cx);
                                    }))
                                    .child(
                                        div().text_xs().font_semibold().text_color(rgba_hex(0xffffff)).child("Tester Connexion ASR"),
                                    ),
                            )
                            .child(
                                div()
                                    .id("btn-use-server")
                                    .cursor_pointer()
                                    .px_4()
                                    .py_2()
                                    .rounded(px(8.))
                                    .bg(rgba_hex(0x26262c))
                                    .border_1()
                                    .border_color(BORDER_DARK)
                                    .hover(|s| s.bg(BG_CARD_HOVER))
                                    .on_click(cx.listener(|this, _: &ClickEvent, window, cx| {
                                        let url = this.api_url_input.clone();
                                        let key = if this.api_key_input.is_empty() { None } else { Some(this.api_key_input.clone()) };
                                        this.switch_remote_asr(url, key, window, cx);
                                    }))
                                    .child(
                                        div().text_xs().font_semibold().text_color(TEXT_MAIN).child("Utiliser ce Serveur pour ASR"),
                                    ),
                            ),
                    ),
            )
            .child(
                // Section 3: Model Catalog Table
                v_flex()
                    .p_5()
                    .rounded(px(14.))
                    .bg(BG_CARD)
                    .border_1()
                    .border_color(BORDER_DARK)
                    .gap_4()
                    .child(
                        h_flex()
                            .justify_between()
                            .items_center()
                            .child(
                                div().text_sm().font_semibold().text_color(TEXT_MAIN).child("Catalogue des Modèles Vocaux (Whisper Local & Serveur ASR)"),
                            )
                            .child(
                                if self.is_downloading {
                                    div().text_xs().text_color(ACCENT_YELLOW).child(format!("Téléchargement en cours : {:.1}% - {}", self.download_pct, self.download_msg))
                                } else {
                                    div().text_xs().text_color(TEXT_MUTED).child(format!(
                                        "Activation automatique du serveur ASR ou du moteur {}",
                                        active_hw_summary
                                    ))
                                },
                            ),
                    )
                    .child(
                        v_flex()
                            .gap_2()
                            .children(self.models_catalog.iter().map(|meta| {
                                let is_whisper = meta.default_engine == "whisper_local";
                                let is_active = self.active_model_id == meta.id && (
                                    (is_whisper && self.active_engine == "whisper_local") ||
                                    (!is_whisper && self.active_engine == "remote_asr")
                                );
                                let installed = if is_whisper { is_model_installed(meta) } else { true };
                                let meta_clone1 = meta.clone();
                                let meta_clone2 = meta.clone();
                                let size_str = if meta.file_size_mb > 0 {
                                    format!("{} Mo", meta.file_size_mb)
                                } else {
                                    "Endpoint ASR".to_string()
                                };
                                let model_id_key = meta.id.clone();
                                let badge_text = meta.badge.clone();

                                h_flex()
                                    .items_center()
                                    .justify_between()
                                    .p_3()
                                    .rounded(px(8.))
                                    .bg(rgba_hex(0x1a1a1e))
                                    .border_1()
                                    .border_color(BORDER_SUBTLE)
                                    .child(
                                        h_flex()
                                            .gap_3()
                                            .items_center()
                                            .child(
                                                div().size(px(8.)).rounded_full().bg(if is_active { ACCENT_GREEN } else if installed { ACCENT_CYAN } else { rgba_hex(0x6b7280) }),
                                            )
                                            .child(
                                                v_flex()
                                                    .gap_0p5()
                                                    .child(
                                                        h_flex()
                                                            .gap_2()
                                                            .items_center()
                                                            .child(div().text_sm().font_medium().text_color(TEXT_MAIN).child(meta.name.clone()))
                                                            .child(
                                                                div()
                                                                    .px_1p5()
                                                                    .py_0p5()
                                                                    .rounded(px(4.))
                                                                    .bg(if is_whisper { rgba_hex(0x14532d) } else { rgba_hex(0x1e3a5f) })
                                                                    .text_xs()
                                                                    .text_color(if is_whisper { rgba_hex(0x86efac) } else { rgba_hex(0x93c5fd) })
                                                                    .child(badge_text),
                                                            ),
                                                    )
                                                    .child(div().text_xs().text_color(TEXT_MUTED).child(meta.description.clone())),
                                            ),
                                    )
                                    .child(
                                        h_flex()
                                            .gap_3()
                                            .items_center()
                                            .child(div().text_xs().font_family(cx.theme().mono_font_family.clone()).text_color(TEXT_DIM).child(size_str))
                                            .child(
                                                if is_active {
                                                    div().px_3().py_1().rounded(px(6.)).bg(rgba_hex(0x166534)).text_xs().font_semibold().text_color(rgba_hex(0xbbf7d0)).child("Actif").into_any_element()
                                                } else if !is_whisper {
                                                    // Non-whisper model: auto-starts local ASR server
                                                    div()
                                                        .id(SharedString::from(format!("act-srv-{}", model_id_key)))
                                                        .cursor_pointer()
                                                        .px_3()
                                                        .py_1()
                                                        .rounded(px(6.))
                                                        .bg(rgba_hex(0x1e3a5f))
                                                        .border_1()
                                                        .border_color(ACCENT_CYAN)
                                                        .hover(|s| s.bg(rgba_hex(0x2563eb)))
                                                        .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                                                            this.switch_to_server_model(meta_clone1.clone(), window, cx);
                                                        }))
                                                        .child(div().text_xs().font_semibold().text_color(rgba_hex(0xbfdbfe)).child("Activer (Serveur ASR)"))
                                                        .into_any_element()
                                                } else if installed {
                                                    // Whisper local model installed
                                                    div()
                                                        .id(SharedString::from(format!("act-{}", model_id_key)))
                                                        .cursor_pointer()
                                                        .px_3()
                                                        .py_1()
                                                        .rounded(px(6.))
                                                        .bg(rgba_hex(0x1e3a5f))
                                                        .border_1()
                                                        .border_color(ACCENT_CYAN)
                                                        .hover(|s| s.bg(rgba_hex(0x2563eb)))
                                                        .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                                                            this.switch_local_whisper(meta_clone1.clone(), window, cx);
                                                        }))
                                                        .child(div().text_xs().font_semibold().text_color(rgba_hex(0xbfdbfe)).child("Activer (Local GPU)"))
                                                        .into_any_element()
                                                } else {
                                                    // Whisper local model needs download
                                                    let is_this_downloading = self.is_downloading && self.download_msg.contains(&meta.name);
                                                    div()
                                                        .id(SharedString::from(format!("dl-{}", model_id_key)))
                                                        .cursor_pointer()
                                                        .px_3()
                                                        .py_1()
                                                        .rounded(px(6.))
                                                        .bg(rgba_hex(0x26262c))
                                                        .border_1()
                                                        .border_color(BORDER_DARK)
                                                        .hover(|s| s.bg(BG_CARD_HOVER))
                                                        .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                                                            this.download_model(meta_clone2.clone(), window, cx);
                                                        }))
                                                        .child(div().text_xs().font_semibold().text_color(TEXT_BODY).child(
                                                            if is_this_downloading {
                                                                format!("Téléchargement ({:.0}%)", self.download_pct)
                                                            } else {
                                                                "Télécharger".to_string()
                                                            }
                                                        ))
                                                        .into_any_element()
                                                },
                                            ),
                                    )
                            })),
                    ),
            )
    }

    // ═══════════════════════════════════════════════════════════════
    // TAB: LANGUES (SOURCE & TARGET TRANSLATION)
    // ═══════════════════════════════════════════════════════════════
    fn render_langues_tab(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let source_languages = [
            ("auto", "⚡ Détection Auto (Multilingue)"),
            ("fr", "Français"),
            ("en", "English"),
            ("es", "Español"),
            ("de", "Deutsch"),
            ("it", "Italiano"),
            ("ja", "日本語"),
            ("zh", "简体中文"),
            ("pt", "Português"),
            ("ru", "Русский"),
            ("ar", "العربية"),
            ("ko", "한국어"),
        ];

        let target_languages = [
            ("en", "English"),
            ("fr", "Français"),
            ("es", "Español"),
            ("de", "Deutsch"),
            ("it", "Italiano"),
            ("ja", "日本語"),
            ("zh", "简体中文"),
            ("pt", "Português"),
            ("ru", "Русский"),
            ("ar", "العربية"),
            ("ko", "한국어"),
        ];

        let cur_source = self.configured_source_lang.to_lowercase();
        let cur_target = self.configured_target_lang.to_lowercase();

        v_flex()
            .size_full()
            .p_6()
            .gap_6()
            .overflow_y_scrollbar()
            .child(
                h_flex()
                    .gap_6()
                    .child(
                        // Source Language Picker
                        v_flex()
                            .flex_1()
                            .p_5()
                            .rounded(px(14.))
                            .bg(BG_CARD)
                            .border_1()
                            .border_color(BORDER_DARK)
                            .gap_3()
                            .child(div().text_sm().font_semibold().text_color(TEXT_MAIN).child("Langue Source (Voix parlée)"))
                            .child(
                                v_flex()
                                    .gap_1p5()
                                    .children(source_languages.iter().map(|(code, name)| {
                                        let is_sel = cur_source == *code;
                                        let code_str = code.to_string();
                                        let cur_tgt = cur_target.clone();
                                        div()
                                            .id(SharedString::from(format!("src-lang-{}", code)))
                                            .cursor_pointer()
                                            .p_2p5()
                                            .rounded(px(8.))
                                            .map(|this| {
                                                if is_sel {
                                                    this.bg(rgba_hex(0x1e3a5f)).border_1().border_color(ACCENT_CYAN)
                                                } else {
                                                    this.bg(rgba_hex(0x19191d)).hover(|s| s.bg(BG_CARD_HOVER))
                                                }
                                            })
                                            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                                this.set_languages(code_str.clone(), cur_tgt.clone(), cx);
                                            }))
                                            .child(
                                                h_flex()
                                                    .justify_between()
                                                    .items_center()
                                                    .child(div().text_sm().font_medium().text_color(if is_sel { TEXT_MAIN } else { TEXT_BODY }).child(*name))
                                                    .child(div().text_xs().font_bold().text_color(if is_sel { ACCENT_CYAN } else { TEXT_DIM }).child(if *code == "auto" { "AUTO".to_string() } else { code.to_uppercase() })),
                                            )
                                    })),
                            ),
                    )
                    .child(
                        // Target Language Picker
                        v_flex()
                            .flex_1()
                            .p_5()
                            .rounded(px(14.))
                            .bg(BG_CARD)
                            .border_1()
                            .border_color(BORDER_DARK)
                            .gap_3()
                            .child(div().text_sm().font_semibold().text_color(TEXT_MAIN).child("Langue Cible (Traduction)"))
                            .child(
                                v_flex()
                                    .gap_1p5()
                                    .children(target_languages.iter().map(|(code, name)| {
                                        let is_sel = cur_target == *code;
                                        let code_str = code.to_string();
                                        let cur_src = cur_source.clone();
                                        div()
                                            .id(SharedString::from(format!("tgt-lang-{}", code)))
                                            .cursor_pointer()
                                            .p_2p5()
                                            .rounded(px(8.))
                                            .map(|this| {
                                                if is_sel {
                                                    this.bg(rgba_hex(0x2a1b38)).border_1().border_color(ACCENT_MAGENTA)
                                                } else {
                                                    this.bg(rgba_hex(0x19191d)).hover(|s| s.bg(BG_CARD_HOVER))
                                                }
                                            })
                                            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                                this.set_languages(cur_src.clone(), code_str.clone(), cx);
                                            }))
                                            .child(
                                                h_flex()
                                                    .justify_between()
                                                    .items_center()
                                                    .child(div().text_sm().font_medium().text_color(if is_sel { TEXT_MAIN } else { TEXT_BODY }).child(*name))
                                                    .child(div().text_xs().font_bold().text_color(if is_sel { ACCENT_MAGENTA } else { TEXT_DIM }).child(code.to_uppercase())),
                                            )
                                    })),
                            ),
                    ),
            )
    }

    // ═══════════════════════════════════════════════════════════════
    // TAB: AUDIO & VAD
    // ═══════════════════════════════════════════════════════════════
    fn render_audio_tab(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let vol_pct = (self.current_volume * 100.0).clamp(0.0, 100.0);

        v_flex()
            .size_full()
            .p_6()
            .gap_5()
            .overflow_y_scrollbar()
            .child(
                v_flex()
                    .p_5()
                    .rounded(px(14.))
                    .bg(BG_CARD)
                    .border_1()
                    .border_color(BORDER_DARK)
                    .gap_3()
                    .child(div().text_sm().font_semibold().text_color(TEXT_MAIN).child("Périphérique d'Entrée Audio (Microphone)"))
                    .child(
                        v_flex()
                            .gap_2()
                            .children(self.available_devices.iter().map(|dev| {
                                let is_active = self.selected_device == *dev;
                                let dev_name = dev.clone();
                                let dev_key = dev.clone();
                                div()
                                    .id(SharedString::from(format!("dev-{}", dev_key)))
                                    .cursor_pointer()
                                    .p_3()
                                    .rounded(px(8.))
                                    .map(|this| {
                                        if is_active {
                                            this.bg(rgba_hex(0x1e3a5f)).border_1().border_color(ACCENT_CYAN)
                                        } else {
                                            this.bg(rgba_hex(0x1a1a1e)).hover(|s| s.bg(BG_CARD_HOVER))
                                        }
                                    })
                                    .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                                        this.selected_device = dev_name.clone();
                                        let mut cfg = this.state.config.blocking_write();
                                        cfg.audio.device_name = Some(dev_name.clone());
                                        let _ = cfg.save_default();
                                        cx.notify();
                                    }))
                                    .child(
                                        h_flex()
                                            .justify_between()
                                            .items_center()
                                            .child(div().text_sm().font_medium().text_color(TEXT_MAIN).child(dev.clone()))
                                            .child(
                                                if is_active {
                                                    div().px_2().py_0p5().rounded(px(6.)).bg(rgba_hex(0x166534)).text_xs().text_color(rgba_hex(0xbbf7d0)).child("Sélectionné")
                                                } else {
                                                    div()
                                                },
                                            ),
                                    )
                            })),
                    ),
            )
            .child(
                // Live VU Meter Bar
                v_flex()
                    .p_5()
                    .rounded(px(14.))
                    .bg(BG_CARD)
                    .border_1()
                    .border_color(BORDER_DARK)
                    .gap_3()
                    .child(
                        h_flex()
                            .justify_between()
                            .items_center()
                            .child(div().text_sm().font_semibold().text_color(TEXT_MAIN).child("Niveau Microphone en Temps Réel"))
                            .child(div().text_xs().font_family(cx.theme().mono_font_family.clone()).text_color(TEXT_DIM).child(format!("{:.1}%", vol_pct))),
                    )
                    .child(
                        div()
                            .w_full()
                            .h(px(12.))
                            .rounded_full()
                            .bg(rgba_hex(0x141418))
                            .border_1()
                            .border_color(BORDER_DARK)
                            .p(px(2.))
                            .child(
                                div()
                                    .h_full()
                                    .w(px(vol_pct * 4.0))
                                    .rounded_full()
                                    .bg(if vol_pct > 80.0 { ACCENT_RED } else if vol_pct > 40.0 { ACCENT_YELLOW } else { ACCENT_GREEN }),
                            ),
                    ),
            )
            .child(
                // Section 3: VAD Sensibilité (Seuil RMS)
                v_flex()
                    .p_5()
                    .rounded(px(14.))
                    .bg(BG_CARD)
                    .border_1()
                    .border_color(BORDER_DARK)
                    .gap_3()
                    .child(
                        h_flex()
                            .justify_between()
                            .items_center()
                            .child(
                                v_flex()
                                    .child(div().text_sm().font_semibold().text_color(TEXT_MAIN).child("Sensibilité Détection Vocale (Seuil d'Énergie RMS)"))
                                    .child(div().text_xs().text_color(TEXT_MUTED).child("Niveau sonore minimal requis pour démarrer la détection de parole.")),
                            )
                            .child(
                                div()
                                    .px_3()
                                    .py_1()
                                    .rounded(px(8.))
                                    .bg(rgba_hex(0x1a2634))
                                    .border_1()
                                    .border_color(ACCENT_CYAN)
                                    .text_xs()
                                    .font_bold()
                                    .font_family(cx.theme().mono_font_family.clone())
                                    .text_color(ACCENT_CYAN)
                                    .child(format!("{:.1}% RMS", self.vad_threshold * 100.0)),
                            ),
                    )
                    .child(
                        h_flex()
                            .gap_3()
                            .children([
                                (0.008f32, "0.8% - Haute", "Chuchotements / Micro éloigné"),
                                (0.015f32, "1.5% - Recommandé", "Standard optimal voix parlée"),
                                (0.025f32, "2.5% - Moyenne", "Filtre les bruits modérés"),
                                (0.040f32, "4.0% - Faible", "Micro très fort / bruit élevé"),
                            ].into_iter().map(|(val, label, hint)| {
                                let is_sel = (self.vad_threshold - val).abs() < 0.002;
                                div()
                                    .flex_1()
                                    .id(SharedString::from(format!("vad-th-{:.3}", val)))
                                    .cursor_pointer()
                                    .p_3()
                                    .rounded(px(8.))
                                    .map(|this| {
                                        if is_sel {
                                            this.bg(rgba_hex(0x1e3a5f)).border_1().border_color(ACCENT_CYAN)
                                        } else {
                                            this.bg(rgba_hex(0x1a1a1e)).border_1().border_color(BORDER_SUBTLE).hover(|s| s.bg(BG_CARD_HOVER))
                                        }
                                    })
                                    .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                                        this.vad_threshold = val;
                                        let mut cfg = this.state.config.blocking_write();
                                        cfg.audio.vad_threshold = val;
                                        let _ = cfg.save_default();
                                        window.push_notification(Notification::info(format!("Sensibilité VAD ajustée à {:.1}%", val * 100.0)), cx);
                                        cx.notify();
                                    }))
                                    .child(
                                        v_flex()
                                            .gap_0p5()
                                            .child(div().text_xs().font_semibold().text_color(if is_sel { ACCENT_CYAN } else { TEXT_MAIN }).child(label))
                                            .child(div().text_xs().text_color(TEXT_MUTED).child(hint)),
                                    )
                            })),
                    ),
            )
            .child(
                // Section 4: Silence de Fin de Phrase
                v_flex()
                    .p_5()
                    .rounded(px(14.))
                    .bg(BG_CARD)
                    .border_1()
                    .border_color(BORDER_DARK)
                    .gap_3()
                    .child(
                        h_flex()
                            .justify_between()
                            .items_center()
                            .child(
                                v_flex()
                                    .child(div().text_sm().font_semibold().text_color(TEXT_MAIN).child("Silence de Fin de Phrase (Coupure de segment)"))
                                    .child(div().text_xs().text_color(TEXT_MUTED).child("Durée de silence requise pour clore une phrase et lancer la transcription.")),
                            )
                            .child(
                                div()
                                    .px_3()
                                    .py_1()
                                    .rounded(px(8.))
                                    .bg(rgba_hex(0x1e2e1e))
                                    .border_1()
                                    .border_color(ACCENT_GREEN)
                                    .text_xs()
                                    .font_bold()
                                    .font_family(cx.theme().mono_font_family.clone())
                                    .text_color(ACCENT_GREEN)
                                    .child(format!("{} ms", self.vad_silence_ms)),
                            ),
                    )
                    .child(
                        h_flex()
                            .gap_3()
                            .children([
                                (400u64, "400 ms - Rapide", "Réactivité maximale streaming"),
                                (700u64, "700 ms - Recommandé", "Rythme naturel de parole"),
                                (1000u64, "1000 ms - Fluide", "Permet des pauses courtes"),
                                (1500u64, "1500 ms - Phrases longues", "Regroupe les longues phrases"),
                            ].into_iter().map(|(val, label, hint)| {
                                let is_sel = self.vad_silence_ms == val;
                                div()
                                    .flex_1()
                                    .id(SharedString::from(format!("vad-sil-{}", val)))
                                    .cursor_pointer()
                                    .p_3()
                                    .rounded(px(8.))
                                    .map(|this| {
                                        if is_sel {
                                            this.bg(rgba_hex(0x163520)).border_1().border_color(ACCENT_GREEN)
                                        } else {
                                            this.bg(rgba_hex(0x1a1a1e)).border_1().border_color(BORDER_SUBTLE).hover(|s| s.bg(BG_CARD_HOVER))
                                        }
                                    })
                                    .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                                        this.vad_silence_ms = val;
                                        let mut cfg = this.state.config.blocking_write();
                                        cfg.audio.vad_silence_duration_ms = val;
                                        let _ = cfg.save_default();
                                        window.push_notification(Notification::info(format!("Silence de fin de phrase : {} ms", val)), cx);
                                        cx.notify();
                                    }))
                                    .child(
                                        v_flex()
                                            .gap_0p5()
                                            .child(div().text_xs().font_semibold().text_color(if is_sel { ACCENT_GREEN } else { TEXT_MAIN }).child(label))
                                            .child(div().text_xs().text_color(TEXT_MUTED).child(hint)),
                                    )
                            })),
                    ),
            )
            .child(
                // Section 5: Durée Minimale & Maximale de Parole
                h_flex()
                    .gap_5()
                    .child(
                        // Durée min de parole
                        v_flex()
                            .flex_1()
                            .p_5()
                            .rounded(px(14.))
                            .bg(BG_CARD)
                            .border_1()
                            .border_color(BORDER_DARK)
                            .gap_3()
                            .child(
                                h_flex()
                                    .justify_between()
                                    .items_center()
                                    .child(div().text_sm().font_semibold().text_color(TEXT_MAIN).child("Durée Minimale de Parole"))
                                    .child(div().text_xs().font_bold().font_family(cx.theme().mono_font_family.clone()).text_color(TEXT_MUTED).child(format!("{} ms", self.vad_min_speech_ms))),
                            )
                            .child(div().text_xs().text_color(TEXT_MUTED).child("Filtre les bruits brefs (toux, clics souris, frappe clavier)."))
                            .child(
                                h_flex()
                                    .gap_2()
                                    .children([
                                        (200u64, "200 ms"),
                                        (350u64, "350 ms (Recommandé)"),
                                        (500u64, "500 ms"),
                                    ].into_iter().map(|(val, label)| {
                                        let is_sel = self.vad_min_speech_ms == val;
                                        div()
                                            .flex_1()
                                            .id(SharedString::from(format!("vad-min-{}", val)))
                                            .cursor_pointer()
                                            .p_2p5()
                                            .rounded(px(6.))
                                            .map(|this| {
                                                if is_sel {
                                                    this.bg(rgba_hex(0x1e3a5f)).border_1().border_color(ACCENT_CYAN)
                                                } else {
                                                    this.bg(rgba_hex(0x1a1a1e)).border_1().border_color(BORDER_SUBTLE).hover(|s| s.bg(BG_CARD_HOVER))
                                                }
                                            })
                                            .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                                                this.vad_min_speech_ms = val;
                                                let mut cfg = this.state.config.blocking_write();
                                                cfg.audio.vad_min_speech_ms = val;
                                                let _ = cfg.save_default();
                                                window.push_notification(Notification::info(format!("Durée min de parole : {} ms", val)), cx);
                                                cx.notify();
                                            }))
                                            .child(div().text_center().text_xs().font_semibold().text_color(if is_sel { ACCENT_CYAN } else { TEXT_BODY }).child(label))
                                    })),
                            ),
                    )
                    .child(
                        // Durée max continue
                        v_flex()
                            .flex_1()
                            .p_5()
                            .rounded(px(14.))
                            .bg(BG_CARD)
                            .border_1()
                            .border_color(BORDER_DARK)
                            .gap_3()
                            .child(
                                h_flex()
                                    .justify_between()
                                    .items_center()
                                    .child(div().text_sm().font_semibold().text_color(TEXT_MAIN).child("Durée Maximale par Segment"))
                                    .child(div().text_xs().font_bold().font_family(cx.theme().mono_font_family.clone()).text_color(TEXT_MUTED).child(format!("{} s", self.vad_max_speech_ms / 1000))),
                            )
                            .child(div().text_xs().text_color(TEXT_MUTED).child("Force le découpage et l'envoi en cas de parole continue sans pause."))
                            .child(
                                h_flex()
                                    .gap_2()
                                    .children([
                                        (8000u64, "8 s"),
                                        (15000u64, "15 s (Recommandé)"),
                                        (25000u64, "25 s"),
                                    ].into_iter().map(|(val, label)| {
                                        let is_sel = self.vad_max_speech_ms == val;
                                        div()
                                            .flex_1()
                                            .id(SharedString::from(format!("vad-max-{}", val)))
                                            .cursor_pointer()
                                            .p_2p5()
                                            .rounded(px(6.))
                                            .map(|this| {
                                                if is_sel {
                                                    this.bg(rgba_hex(0x1e3a5f)).border_1().border_color(ACCENT_CYAN)
                                                } else {
                                                    this.bg(rgba_hex(0x1a1a1e)).border_1().border_color(BORDER_SUBTLE).hover(|s| s.bg(BG_CARD_HOVER))
                                                }
                                            })
                                            .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                                                this.vad_max_speech_ms = val;
                                                let mut cfg = this.state.config.blocking_write();
                                                cfg.audio.vad_max_speech_ms = val;
                                                let _ = cfg.save_default();
                                                window.push_notification(Notification::info(format!("Durée max de segment : {} s", val / 1000)), cx);
                                                cx.notify();
                                            }))
                                            .child(div().text_center().text_xs().font_semibold().text_color(if is_sel { ACCENT_CYAN } else { TEXT_BODY }).child(label))
                                    })),
                            ),
                    ),
            )
    }

    // ═══════════════════════════════════════════════════════════════
    // TAB: INCRUSTATION (OBS STUDIO OVERLAY)
    // ═══════════════════════════════════════════════════════════════
    fn render_incrustation_tab(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let port = self.state.config.blocking_read().server.port;
        let obs_url = format!("http://127.0.0.1:{}/obs", port);

        v_flex()
            .size_full()
            .p_6()
            .gap_5()
            .overflow_y_scrollbar()
            .child(
                // OBS Preview Card
                v_flex()
                    .p_6()
                    .rounded(px(16.))
                    .bg(rgba_hex(0x0a0a0d))
                    .border_1()
                    .border_color(ACCENT_CYAN)
                    .gap_4()
                    .items_center()
                    .justify_center()
                    .child(
                        div().text_xs().font_bold().text_color(ACCENT_CYAN).child("APERÇU EN DIRECT DU SOUS-TITRE OBS"),
                    )
                    .child(
                        v_flex()
                            .w(px(640.))
                            .p_4()
                            .rounded(px(12.))
                            .bg(rgba_hex(0x000000))
                            .border_1()
                            .border_color(ACCENT_CYAN)
                            .gap_2()
                            .items_center()
                            .child(
                                div()
                                    .text_xl()
                                    .font_bold()
                                    .text_color(TEXT_MAIN)
                                    .child(self.latest_source_text.clone()),
                            )
                            .child(
                                div()
                                    .text_lg()
                                    .font_medium()
                                    .text_color(rgba_hex(0x38bdf8))
                                    .child(self.latest_trans_text.clone()),
                            ),
                    ),
            )
            .child(
                // Action Buttons
                h_flex()
                    .gap_4()
                    .child(
                        div()
                            .id("btn-copy-obs-incrust")
                            .cursor_pointer()
                            .px_5()
                            .py_3()
                            .rounded(px(10.))
                            .bg(rgba_hex(0x1e3a5f))
                            .border_1()
                            .border_color(ACCENT_CYAN)
                            .hover(|s| s.bg(rgba_hex(0x2563eb)))
                            .on_click(cx.listener(Self::copy_obs_url))
                            .child(
                                h_flex()
                                    .gap_2()
                                    .items_center()
                                    .child(Icon::new(IconName::Copy).small().text_color(TEXT_MAIN))
                                    .child(div().text_sm().font_semibold().text_color(TEXT_MAIN).child(format!("Copier l'URL OBS ({})", obs_url))),
                            ),
                    )
                    .child(
                        div()
                            .id("btn-send-test-sub")
                            .cursor_pointer()
                            .px_5()
                            .py_3()
                            .rounded(px(10.))
                            .bg(rgba_hex(0x166534))
                            .border_1()
                            .border_color(ACCENT_GREEN)
                            .hover(|s| s.bg(rgba_hex(0x15803d)))
                            .on_click(cx.listener(Self::send_test_subtitle))
                            .child(
                                h_flex()
                                    .gap_2()
                                    .items_center()
                                    .child(Icon::new(IconName::Play).small().text_color(TEXT_MAIN))
                                    .child(div().text_sm().font_semibold().text_color(TEXT_MAIN).child("Envoyer un sous-titre de test à OBS")),
                            ),
                    ),
            )
            .child(
                // Instructions Card
                v_flex()
                    .p_5()
                    .rounded(px(14.))
                    .bg(BG_CARD)
                    .border_1()
                    .border_color(BORDER_DARK)
                    .gap_2()
                    .child(div().text_sm().font_semibold().text_color(TEXT_MAIN).child("Comment ajouter dans OBS Studio :"))
                    .child(div().text_xs().text_color(TEXT_MUTED).child("1. Dans OBS, ajoutez une source de type 'Navigateur' (Browser Source)."))
                    .child(div().text_xs().text_color(TEXT_MUTED).child(format!("2. Collez l'URL : {}", obs_url)))
                    .child(div().text_xs().text_color(TEXT_MUTED).child("3. Définissez la taille sur 1920 x 1080 (le fond est 100% transparent par défaut)."))
                    .child(div().text_xs().text_color(TEXT_MUTED).child("4. Vos sous-titres apparaîtront en direct avec animations fluides 120 FPS !")),
            )
    }

    // ═══════════════════════════════════════════════════════════════
    // RENDER: FOOTER / STATUS BAR
    // ═══════════════════════════════════════════════════════════════
    fn render_footer(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let port = self.state.config.blocking_read().server.port;
        let hw_label = crate::stt::active_hardware_summary(
            self.use_gpu,
            &self.gpu_backend,
            self.gpu_device,
        );

        h_flex()
            .w_full()
            .items_center()
            .justify_between()
            .px_6()
            .py_2()
            .bg(BG_MAIN)
            .border_t_1()
            .border_color(BORDER_SUBTLE)
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(div().size(px(5.)).rounded_full().bg(ACCENT_GREEN))
                    .child(
                        div()
                            .text_xs()
                            .font_family(cx.theme().mono_font_family.clone())
                            .text_color(TEXT_DIM)
                            .child(format!("{} · {} Threads CPU", hw_label, self.n_threads)),
                    ),
            )
            .child(
                h_flex()
                    .gap_3()
                    .items_center()
                    .child(
                        div()
                            .text_xs()
                            .font_family(cx.theme().mono_font_family.clone())
                            .text_color(TEXT_DIM)
                            .child(format!("OBS: http://127.0.0.1:{}/obs", port)),
                    )
                    .child(
                        div()
                            .text_xs()
                            .font_medium()
                            .text_color(TEXT_DIM)
                            .child("VoiceBridge"),
                    ),
            )
    }
}

impl Focusable for VoiceBridgeApp {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for VoiceBridgeApp {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .size_full()
            .bg(BG_MAIN)
            .child(self.render_titlebar(cx))
            .child(self.render_action_row(cx))
            .child(self.render_tabs_bar(cx))
            .child(
                div().flex_1().min_h_0().child(match self.active_tab {
                    GuiTab::Direct => self.render_direct_tab(cx).into_any_element(),
                    GuiTab::Modeles => self.render_modeles_tab(cx).into_any_element(),
                    GuiTab::Langues => self.render_langues_tab(cx).into_any_element(),
                    GuiTab::AudioVad => self.render_audio_tab(cx).into_any_element(),
                    GuiTab::Incrustation => self.render_incrustation_tab(cx).into_any_element(),
                }),
            )
            .child(self.render_footer(cx))
    }
}

/// Locates `scripts/asr_server.py` across dev workspaces, macOS `.app` bundles (`Contents/Resources`),
/// Linux `.deb` FHS installs (`/usr/share/voicebridge`), Windows `Program Files`, or extracts the
/// embedded copy into the writable application data directory for standalone single-file `.exe` runs.
fn resolve_asr_script_path() -> PathBuf {
    let local_rel = PathBuf::from("scripts").join("asr_server.py");
    if local_rel.is_file() {
        return local_rel;
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            // Windows installer / portable directory beside binary
            let beside_exe = exe_dir.join("scripts").join("asr_server.py");
            if beside_exe.is_file() {
                return beside_exe;
            }
            // macOS VoiceBridge.app/Contents/Resources/scripts/asr_server.py
            if let Some(contents_dir) = exe_dir.parent() {
                let bundle_res = contents_dir
                    .join("Resources")
                    .join("scripts")
                    .join("asr_server.py");
                if bundle_res.is_file() {
                    return bundle_res;
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        let deb_path = PathBuf::from("/usr/share/voicebridge/scripts/asr_server.py");
        if deb_path.is_file() {
            return deb_path;
        }
    }

    // Fallback for standalone single-file executables: materialize embedded asr_server.py
    const EMBEDDED_ASR_SERVER_PY: &str = include_str!("../../scripts/asr_server.py");
    if std::fs::create_dir_all("scripts").is_ok() {
        let _ = std::fs::write(&local_rel, EMBEDDED_ASR_SERVER_PY);
    }
    local_rel
}

/// Resolves a working Python 3 interpreter across macOS, Linux, and Windows.
/// Checks local virtual environments first, then probes system commands with `--version`
/// (avoiding Windows Microsoft Store `python3.exe` stubs that exit with code 9009).
fn resolve_python_command() -> (String, Vec<String>) {
    #[cfg(windows)]
    let venv_candidates = [
        std::path::Path::new(".venv").join("Scripts").join("python.exe"),
        std::path::Path::new("venv").join("Scripts").join("python.exe"),
    ];
    #[cfg(not(windows))]
    let venv_candidates = [
        std::path::Path::new(".venv").join("bin").join("python3"),
        std::path::Path::new(".venv").join("bin").join("python"),
        std::path::Path::new("venv").join("bin").join("python3"),
        std::path::Path::new("venv").join("bin").join("python"),
    ];

    for venv_py in &venv_candidates {
        if venv_py.is_file() {
            return (venv_py.to_string_lossy().into_owned(), Vec::new());
        }
    }

    #[cfg(windows)]
    let system_candidates: &[(&str, &[&str])] = &[
        ("python", &[]),
        ("py", &["-3"]),
        ("python3", &[]),
    ];
    #[cfg(not(windows))]
    let system_candidates: &[(&str, &[&str])] = &[
        ("python3", &[]),
        ("python", &[]),
    ];

    for (bin, prefix_args) in system_candidates {
        let mut probe = std::process::Command::new(bin);
        probe
            .args(*prefix_args)
            .arg("--version")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            probe.creation_flags(CREATE_NO_WINDOW);
        }

        if let Ok(status) = probe.status() {
            if status.success() {
                return (
                    (*bin).to_string(),
                    prefix_args.iter().map(|s| (*s).to_string()).collect(),
                );
            }
        }
    }

    #[cfg(windows)]
    {
        ("python".to_string(), Vec::new())
    }
    #[cfg(not(windows))]
    {
        ("python3".to_string(), Vec::new())
    }
}

#[cfg(unix)]
unsafe extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
}

/// Gracefully stops the local Python ASR server (sending SIGTERM first on Unix so Uvicorn and
/// Python's `multiprocessing.resource_tracker` release semaphores cleanly before falling back to kill).
fn terminate_asr_child(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        unsafe {
            const SIGTERM: i32 = 15;
            let _ = kill(child.id() as i32, SIGTERM);
        }
        for _ in 0..30 {
            if let Ok(Some(_)) = child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(15));
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

impl Drop for VoiceBridgeApp {
    fn drop(&mut self) {
        self.state.is_shutting_down.store(true, Ordering::SeqCst);
        self.state.is_recording.store(false, Ordering::SeqCst);

        if let Ok(mut guard) = self.local_asr_child.lock() {
            if let Some(mut child) = guard.take() {
                terminate_asr_child(&mut child);
                tracing::info!("Serveur ASR local arrêté lors de la fermeture de VoiceBridge.");
            }
        }

        // Explicitly release Metal GPU residency sets before AppKit's `-[NSApplication terminate:]`
        // calls C `exit()`, avoiding `GGML_ASSERT([rsets->data count] == 0)` in `ggml_metal_rsets_free`.
        self.state.shutdown_sync();
    }
}

// Main GUI Entrypoint
pub fn run_gui(
    state: AppState,
    tokio_handle: tokio::runtime::Handle,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    struct AppAssets;

    impl gpui_kit::AssetSource for AppAssets {
        fn load(
            &self,
            path: &str,
        ) -> Result<std::option::Option<std::borrow::Cow<'static, [u8]>>, gpui_kit::private::anyhow::Error> {
            if path == "docs/logo.png" || path == "logo.png" {
                const LOGO_PNG: &[u8] = include_bytes!("../../docs/logo.png");
                return Ok(Some(std::borrow::Cow::Borrowed(LOGO_PNG)));
            }
            AllAssets.load(path)
        }

        fn list(
            &self,
            path: &str,
        ) -> Result<Vec<SharedString>, gpui_kit::private::anyhow::Error> {
            let mut list = AllAssets.list(path)?;
            if path.is_empty() || path == "docs" {
                list.push(SharedString::from("docs/logo.png"));
            }
            Ok(list)
        }
    }

    let app = gpui_kit::application().with_assets(AppAssets);

    app.run(move |cx| {
        gpui_kit::init(cx);
        gpui_kit::component::theme::Theme::change(gpui_kit::component::theme::ThemeMode::Dark, None, cx);

        let state_for_quit = state.clone();
        cx.on_app_quit(move |_| {
            let st = state_for_quit.clone();
            async move {
                st.shutdown_sync();
            }
        })
        .detach();

        cx.on_window_closed(|cx, _| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();

        let window_options = WindowOptions {
            window_bounds: Some(WindowBounds::centered(size(px(1120.), px(780.)), cx)),
            window_min_size: Some(size(px(860.), px(580.))),
            ..TitleBar::window_options()
        };

        gpui_kit::open_window(window_options, cx, |window, cx| {
            cx.new(|cx| VoiceBridgeApp::new(state, tokio_handle, window, cx))
        })
        .expect("Failed to open VoiceBridge GPUI window");
    });

    Ok(())
}
