use super::state::{AppState, PipelineEvent, SystemStatus};
use crate::audio::{AudioDeviceManager, DeviceInfo};
use crate::config::AppConfig;
use crate::stt::{
    download_model_with_progress, find_model, get_available_models, is_model_installed,
    DownloadStatus, ModelMetadata,
};
use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        Html,
    },
    routing::{get, post},
    Json, Router,
};
use futures_util::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use tower_http::cors::CorsLayer;

const INDEX_HTML: &str = include_str!("../web/index.html");
const OBS_HTML: &str = include_str!("../web/obs.html");
const LOGO_PNG: &[u8] = include_bytes!("../../docs/logo.png");

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index_handler))
        .route("/obs", get(obs_handler))
        .route("/logo.png", get(logo_handler))
        .route("/api/stream", get(sse_handler))
        .route("/api/status", get(status_handler))
        .route("/api/history", get(history_handler))
        .route("/api/config", get(get_config_handler).post(update_config_handler))
        .route("/api/models", get(get_models_handler))
        .route("/api/models/switch", post(switch_model_handler))
        .route("/api/models/download-progress", get(download_progress_handler))
        .route("/api/devices", get(get_devices_handler))
        .route("/api/toggle-mic", post(toggle_mic_handler))
        .route("/api/test-stt", post(test_stt_handler))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn index_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn obs_handler() -> Html<&'static str> {
    Html(OBS_HTML)
}

async fn logo_handler() -> ([(axum::http::header::HeaderName, &'static str); 1], &'static [u8]) {
    ([(axum::http::header::CONTENT_TYPE, "image/png")], LOGO_PNG)
}

async fn status_handler(State(state): State<AppState>) -> Json<SystemStatus> {
    Json(state.get_status().await)
}

async fn history_handler(State(state): State<AppState>) -> Json<Vec<PipelineEvent>> {
    let history = state.history.read().await;
    let list: Vec<PipelineEvent> = history.iter().cloned().collect();
    Json(list)
}

async fn toggle_mic_handler(State(state): State<AppState>) -> Json<SystemStatus> {
    let current = state.is_recording.load(Ordering::Relaxed);
    state.is_recording.store(!current, Ordering::Relaxed);
    Json(state.get_status().await)
}

async fn get_config_handler(State(state): State<AppState>) -> Json<AppConfig> {
    let cfg = state.config.read().await.clone();
    Json(cfg)
}

async fn update_config_handler(
    State(state): State<AppState>,
    Json(mut new_config): Json<AppConfig>,
) -> Json<SystemStatus> {
    // Synchronize chosen source language between STT and Translation
    let chosen_source_lang = if !new_config.stt.language.is_empty() {
        new_config.stt.language.clone()
    } else if !new_config.translation.source_lang.is_empty() {
        new_config.translation.source_lang.clone()
    } else {
        "auto".to_string()
    };
    new_config.stt.language = chosen_source_lang.clone();
    new_config.translation.source_lang = chosen_source_lang.clone();

    let hw_changed = {
        let old = state.config.read().await;
        old.stt.use_gpu != new_config.stt.use_gpu
            || old.stt.gpu_backend != new_config.stt.gpu_backend
            || old.stt.gpu_device != new_config.stt.gpu_device
            || old.stt.flash_attn != new_config.stt.flash_attn
    };

    // Immediately update live Whisper engine in memory
    {
        let mut engine = state.whisper_engine.lock().await;
        engine.set_language(&chosen_source_lang);
        engine.set_n_threads(new_config.stt.n_threads);
        if hw_changed {
            let _ = engine.apply_hardware_settings(
                new_config.stt.use_gpu,
                &new_config.stt.gpu_backend,
                new_config.stt.gpu_device,
                new_config.stt.flash_attn,
                new_config.stt.n_threads,
            );
        }
        tracing::info!("Whisper STT config updated (lang='{}', use_gpu={}, backend='{}', device={})",
            chosen_source_lang,
            new_config.stt.use_gpu,
            new_config.stt.gpu_backend,
            new_config.stt.gpu_device
        );
    }

    {
        let mut cfg = state.config.write().await;
        *cfg = new_config.clone();
    }
    {
        let mut trans = state.translation_service.write().await;
        trans.enabled = new_config.translation.enabled;
        trans.source_lang = new_config.translation.source_lang.clone();
        trans.target_lang = new_config.translation.target_lang.clone();
        trans.engine = new_config.translation.engine.clone();
        trans.local_api_url = new_config.translation.local_api_url.clone();
        trans.model_name = new_config.translation.model_name.clone();
        trans.api_key = new_config.translation.api_key.clone();
    }
    let _ = new_config.save_to_file("config.json");
    Json(state.get_status().await)
}

#[derive(Serialize)]
pub struct ModelCatalogItem {
    #[serde(flatten)]
    pub metadata: ModelMetadata,
    pub is_installed: bool,
    pub is_active: bool,
}

async fn get_models_handler(State(state): State<AppState>) -> Json<Vec<ModelCatalogItem>> {
    let cfg = state.config.read().await;
    let items = get_available_models()
        .into_iter()
        .map(|m| {
            let installed = is_model_installed(&m);
            let active = cfg.stt.model_id == m.id;
            ModelCatalogItem {
                metadata: m,
                is_installed: installed,
                is_active: active,
            }
        })
        .collect();
    Json(items)
}

async fn download_progress_handler(State(state): State<AppState>) -> Json<DownloadStatus> {
    let status = state.download_status.read().await.clone();
    Json(status)
}

#[derive(Deserialize)]
pub struct SwitchModelRequest {
    pub model_id: String,
    pub api_url: Option<String>,
    pub api_key: Option<String>,
}

#[derive(Serialize)]
pub struct SwitchModelResponse {
    pub success: bool,
    pub status: String, // "ready", "downloading", "offline", "error"
    pub message: String,
    pub model_id: String,
}

async fn switch_model_handler(
    State(state): State<AppState>,
    Json(payload): Json<SwitchModelRequest>,
) -> Json<SwitchModelResponse> {
    let model = match find_model(&payload.model_id) {
        Some(m) => m,
        None => {
            return Json(SwitchModelResponse {
                success: false,
                status: "error".to_string(),
                message: format!("Modèle inconnu : {}", payload.model_id),
                model_id: payload.model_id,
            });
        }
    };

    if model.is_local_gpu {
        let model_path = model.default_model_name.clone();

        if is_model_installed(&model) {
            // Already installed: hot reload directly on configured GPU/CPU
            let mut engine = state.whisper_engine.lock().await;
            match engine.reload_model(&model_path) {
                Ok(_) => {
                    let active_hw = engine.active_hardware_label().to_string();
                    drop(engine);
                    let mut cfg = state.config.write().await;
                    cfg.stt.engine_type = "whisper_local".to_string();
                    cfg.stt.model_id = payload.model_id.clone();
                    cfg.stt.model_path = model_path;
                    if let Some(ref url) = model.download_url {
                        cfg.stt.model_url = url.clone();
                    }
                    cfg.stt.model_name = model.name.clone();
                    let _ = cfg.save_to_file("config.json");

                    return Json(SwitchModelResponse {
                        success: true,
                        status: "ready".to_string(),
                        message: format!(
                            "{} activé ({}).",
                            model.name,
                            active_hw
                        ),
                        model_id: payload.model_id,
                    });
                }
                Err(e) => {
                    return Json(SwitchModelResponse {
                        success: false,
                        status: "error".to_string(),
                        message: format!("Erreur rechargement GPU : {}", e),
                        model_id: payload.model_id,
                    });
                }
            }
        }

        // Needs download first
        let is_dl = state.download_status.read().await.is_downloading;
        if is_dl {
            return Json(SwitchModelResponse {
                success: false,
                status: "downloading".to_string(),
                message: "Un téléchargement est déjà en cours.".to_string(),
                model_id: payload.model_id,
            });
        }

        let download_url = match model.download_url {
            Some(u) => u,
            None => {
                return Json(SwitchModelResponse {
                    success: false,
                    status: "error".to_string(),
                    message: "Aucune URL de téléchargement pour ce modèle.".to_string(),
                    model_id: payload.model_id,
                });
            }
        };

        let status_tracker = state.download_status.clone();
        let whisper = state.whisper_engine.clone();
        let config = state.config.clone();
        let model_id = payload.model_id.clone();
        let model_name = model.name.clone();
        let path_clone = model_path.clone();
        let url_clone = download_url.clone();

        tokio::spawn(async move {
            let hw = crate::stt::active_whisper_backend_label();
            tracing::info!("Starting background download of {}...", model_name);
            match download_model_with_progress(&path_clone, &download_url, &model_id, status_tracker.clone()).await {
                Ok(_) => {
                    tracing::info!("Model download complete. Reloading on {}...", hw);
                    let mut engine = whisper.lock().await;
                    if let Err(e) = engine.reload_model(&path_clone) {
                        tracing::error!("Failed to reload model after download: {}", e);
                        let mut st = status_tracker.write().await;
                        st.error = Some(format!("Erreur ({}) : {}", hw, e));
                        return;
                    }

                    let mut cfg = config.write().await;
                    cfg.stt.engine_type = "whisper_local".to_string();
                    cfg.stt.model_id = model_id;
                    cfg.stt.model_path = path_clone;
                    cfg.stt.model_url = url_clone;
                    cfg.stt.model_name = model_name.clone();
                    let _ = cfg.save_to_file("config.json");
                    tracing::info!("Model {} is now active on {}.", model_name, hw);
                }
                Err(e) => {
                    tracing::error!("Failed to download model: {}", e);
                }
            }
        });

        return Json(SwitchModelResponse {
            success: true,
            status: "downloading".to_string(),
            message: format!("Téléchargement de {} ({} Mo) lancé...", model.name, model.file_size_mb),
            model_id: payload.model_id,
        });
    }

    // Remote ASR Model (Qwen3, Voxtral, Canary, VibeVoice, etc.)
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap_or_default();

    let target_url = payload
        .api_url
        .as_deref()
        .filter(|u| !u.trim().is_empty())
        .unwrap_or_else(|| {
            if !model.default_api_url.is_empty() {
                &model.default_api_url
            } else {
                "http://127.0.0.1:8000/v1/audio/transcriptions"
            }
        });

    let mut req = client.get(target_url);
    if let Some(ref key) = payload.api_key {
        if !key.trim().is_empty() {
            req = req.bearer_auth(key);
        }
    }

    match req.send().await {
        Ok(_) => {
            let mut cfg = state.config.write().await;
            cfg.stt.engine_type = "remote_api".to_string();
            cfg.stt.model_id = payload.model_id.clone();
            cfg.stt.model_name = model.default_model_name.clone();
            cfg.stt.api_url = target_url.to_string();
            cfg.stt.api_key = payload.api_key.clone();
            let _ = cfg.save_to_file("config.json");

            Json(SwitchModelResponse {
                success: true,
                status: "ready".to_string(),
                message: format!("Modèle distant {} connecté et actif sur {}.", model.name, target_url),
                model_id: payload.model_id,
            })
        }
        Err(e) => {
            tracing::warn!("Remote ASR unreachable on {}: {}", target_url, e);
            Json(SwitchModelResponse {
                success: false,
                status: "offline".to_string(),
                message: format!(
                    "Serveur ASR distant injoignable sur {} ({}). Lancez votre serveur (vLLM / Speaches) ou conservez Whisper {} local.",
                    target_url,
                    e,
                    crate::stt::active_whisper_backend_label()
                ),
                model_id: payload.model_id,
            })
        }
    }
}


async fn get_devices_handler() -> Json<Vec<DeviceInfo>> {
    Json(AudioDeviceManager::list_input_devices())
}

#[derive(Serialize)]
struct TestSttResponse {
    success: bool,
    message: String,
}

async fn test_stt_handler(
    State(state): State<AppState>,
) -> Json<TestSttResponse> {
    let cfg = state.config.read().await.clone();
    if cfg.stt.engine_type == "whisper_local" {
        let hw = crate::stt::active_hardware_summary(
            cfg.stt.use_gpu,
            &cfg.stt.gpu_backend,
            cfg.stt.gpu_device,
        );
        return Json(TestSttResponse {
            success: true,
            message: format!("Whisper local ({}) est actif et prêt.", hw),
        });
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap_or_default();

    match client.get(&cfg.stt.api_url).send().await {
        Ok(resp) => Json(TestSttResponse {
            success: true,
            message: format!("Serveur ASR joignable (HTTP {})", resp.status()),
        }),
        Err(e) => Json(TestSttResponse {
            success: false,
            message: format!("Impossible de joindre {}: {}", cfg.stt.api_url, e),
        }),
    }
}

async fn sse_handler(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.event_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|msg| {
        match msg {
            Ok(event) => {
                if let Ok(json) = serde_json::to_string(&event) {
                    Some(Ok(Event::default().data(json)))
                } else {
                    None
                }
            }
            Err(_) => None,
        }
    });

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}
