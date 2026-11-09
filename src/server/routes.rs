use super::state::{AppState, PipelineEvent, SystemStatus};
use crate::audio::{AudioDeviceManager, DeviceInfo};
use crate::config::AppConfig;
use crate::controller::ActivationOutcome;
use crate::stt::{get_available_models, is_model_installed, DownloadStatus, ModelMetadata};
use axum::{
    extract::{Request, State},
    http::{header, HeaderMap, Method, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        Html, IntoResponse, Response,
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
        .route("/api/translation/engine", post(translation_engine_handler))
        .route("/api/devices", get(get_devices_handler))
        .route("/api/toggle-mic", post(toggle_mic_handler))
        .route("/api/test-stt", post(test_stt_handler))
        // No CORS layer on purpose: the UI and the OBS overlay are served from this
        // same origin, and other websites must not be able to drive the API.
        .layer(middleware::from_fn_with_state(state.clone(), local_request_guard))
        .with_state(state)
}

// ─────────────────────────────────────────────────────────────────────────────
// Security: only same-origin requests addressed to this machine are accepted.
// ─────────────────────────────────────────────────────────────────────────────

fn host_without_port(host: &str) -> &str {
    if let Some(rest) = host.strip_prefix('[') {
        return rest.split(']').next().unwrap_or(rest);
    }
    host.rsplit_once(':').map(|(h, _)| h).unwrap_or(host)
}

/// Host names this server legitimately answers to (blocks DNS-rebinding attacks).
fn is_allowed_host(host: &str, configured_host: &str) -> bool {
    let h = host_without_port(host).to_ascii_lowercase();
    if matches!(h.as_str(), "localhost" | "127.0.0.1" | "::1") {
        return true;
    }
    if !configured_host.is_empty() && h == configured_host.to_ascii_lowercase() {
        return true;
    }
    // When bound to all interfaces, accept private LAN addresses (e.g. OBS on another PC).
    if matches!(configured_host, "0.0.0.0" | "::")
        && let Ok(ip) = h.parse::<std::net::IpAddr>()
    {
        return crate::netpolicy::is_local_ip(ip);
    }
    false
}

/// Decide whether a request may proceed. Returns an error message otherwise.
pub(crate) fn check_request(method: &Method, headers: &HeaderMap, configured_host: &str) -> Result<(), &'static str> {
    let host = headers.get(header::HOST).and_then(|v| v.to_str().ok()).unwrap_or("");
    if !is_allowed_host(host, configured_host) {
        return Err("Hôte non autorisé");
    }
    let state_changing = !matches!(*method, Method::GET | Method::HEAD | Method::OPTIONS);
    // Browsers tag cross-site requests; reject them for every API call.
    if let Some(site) = headers.get("sec-fetch-site").and_then(|v| v.to_str().ok())
        && !matches!(site, "same-origin" | "none")
        && (state_changing || site == "cross-site")
    {
        return Err("Requête inter-sites refusée");
    }
    if state_changing {
        match headers.get(header::ORIGIN).and_then(|v| v.to_str().ok()) {
            // Non-browser clients (curl, scripts) send no Origin header.
            None => {}
            Some(origin) => {
                let origin_host = origin
                    .strip_prefix("http://")
                    .or_else(|| origin.strip_prefix("https://"))
                    .unwrap_or("");
                if origin_host != host {
                    return Err("Origine non autorisée");
                }
            }
        }
    }
    Ok(())
}

async fn local_request_guard(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let configured_host = state.config.read().await.server.host.clone();
    if let Err(reason) = check_request(req.method(), req.headers(), &configured_host) {
        tracing::warn!(
            "Requête HTTP refusée ({}) : {} {} (Host={:?}, Origin={:?})",
            reason,
            req.method(),
            req.uri().path(),
            req.headers().get(header::HOST),
            req.headers().get(header::ORIGIN)
        );
        return (StatusCode::FORBIDDEN, reason).into_response();
    }
    next.run(req).await
}

// ─────────────────────────────────────────────────────────────────────────────

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
    Json(history.iter().cloned().collect())
}

async fn toggle_mic_handler(State(state): State<AppState>) -> Json<SystemStatus> {
    state.is_recording.fetch_xor(true, Ordering::Relaxed);
    Json(state.get_status().await)
}

async fn get_config_handler(State(state): State<AppState>) -> Json<AppConfig> {
    Json(state.config.read().await.redacted())
}

#[derive(Serialize)]
struct ConfigUpdateResponse {
    success: bool,
    message: String,
    status: SystemStatus,
}

async fn update_config_handler(
    State(state): State<AppState>,
    Json(new_config): Json<AppConfig>,
) -> (StatusCode, Json<ConfigUpdateResponse>) {
    let res = state.apply_config(new_config).await;
    let status = state.get_status().await;
    match res {
        Ok(()) => (
            StatusCode::OK,
            Json(ConfigUpdateResponse { success: true, message: "Réglages appliqués.".into(), status }),
        ),
        Err(message) => (
            StatusCode::BAD_REQUEST,
            Json(ConfigUpdateResponse { success: false, message, status }),
        ),
    }
}

#[derive(Serialize)]
pub struct ModelCatalogItem {
    #[serde(flatten)]
    pub metadata: ModelMetadata,
    pub is_installed: bool,
    pub is_active: bool,
}

async fn get_models_handler(State(state): State<AppState>) -> Json<Vec<ModelCatalogItem>> {
    let cfg = state.config.read().await.clone();
    let items = get_available_models()
        .into_iter()
        .map(|m| {
            let installed = is_model_installed(&m);
            let active = if m.kind == crate::stt::models::KIND_TRANSLATION {
                cfg.translation.engine == crate::stt::models::ENGINE_MADLAD
                    && cfg.translation.madlad_model_dir == m.default_model_name
            } else {
                cfg.stt.model_id == m.id && cfg.stt.engine_type == m.default_engine
            };
            ModelCatalogItem { metadata: m, is_installed: installed, is_active: active }
        })
        .collect();
    Json(items)
}

async fn download_progress_handler(State(state): State<AppState>) -> Json<DownloadStatus> {
    Json(state.download_status.read().await.clone())
}

#[derive(Deserialize)]
pub struct SwitchModelRequest {
    pub model_id: String,
    pub api_url: Option<String>,
    pub api_key: Option<String>,
}

async fn switch_model_handler(
    State(state): State<AppState>,
    Json(payload): Json<SwitchModelRequest>,
) -> Json<ActivationOutcome> {
    Json(
        state
            .activate_model(&payload.model_id, payload.api_url, payload.api_key)
            .await,
    )
}

#[derive(Deserialize)]
pub struct TranslationEngineRequest {
    pub engine: String,
}

async fn translation_engine_handler(
    State(state): State<AppState>,
    Json(payload): Json<TranslationEngineRequest>,
) -> Json<ActivationOutcome> {
    Json(state.set_translation_engine(&payload.engine).await)
}

async fn get_devices_handler() -> Json<Vec<DeviceInfo>> {
    Json(AudioDeviceManager::list_input_devices())
}

#[derive(Serialize)]
struct TestSttResponse {
    success: bool,
    message: String,
}

async fn test_stt_handler(State(state): State<AppState>) -> Json<TestSttResponse> {
    let cfg = state.config.read().await.clone();
    match cfg.stt.engine_type.as_str() {
        crate::stt::models::ENGINE_WHISPER => {
            let hw = crate::stt::active_hardware_summary(cfg.stt.use_gpu, &cfg.stt.gpu_backend, cfg.stt.gpu_device);
            let loaded = state.whisper_engine.lock().await.is_loaded();
            return Json(TestSttResponse {
                success: loaded,
                message: if loaded {
                    format!("Whisper local ({hw}) est actif et prêt.")
                } else {
                    "Aucun modèle Whisper n'est chargé.".to_string()
                },
            });
        }
        crate::stt::models::ENGINE_KYUTAI | crate::stt::models::ENGINE_HIBIKI => {
            let running = state.streaming_worker.lock().map(|w| w.is_some()).unwrap_or(false);
            return Json(TestSttResponse {
                success: running,
                message: if running {
                    format!("{} est chargé et écoute.", cfg.stt.model_name)
                } else {
                    format!("{} n'est pas chargé.", cfg.stt.model_name)
                },
            });
        }
        _ => {}
    }
    if let Err(e) = crate::netpolicy::ensure_local_endpoint(&cfg.stt.api_url) {
        return Json(TestSttResponse { success: false, message: e });
    }
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap_or_default();
    match client.get(&cfg.stt.api_url).send().await {
        Ok(resp) => Json(TestSttResponse {
            success: true,
            message: format!("Serveur ASR local joignable (HTTP {})", resp.status()),
        }),
        Err(e) => Json(TestSttResponse {
            success: false,
            message: format!("Impossible de joindre {} : {}", cfg.stt.api_url, e),
        }),
    }
}

async fn sse_handler(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.event_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|msg| match msg {
        Ok(event) => serde_json::to_string(&event)
            .ok()
            .map(|json| Ok(Event::default().data(json))),
        Err(_) => None,
    });
    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(
                axum::http::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    #[test]
    fn same_origin_ui_requests_are_allowed() {
        let h = headers(&[("host", "127.0.0.1:3000"), ("origin", "http://127.0.0.1:3000"), ("sec-fetch-site", "same-origin")]);
        assert!(check_request(&Method::POST, &h, "127.0.0.1").is_ok());
        let h = headers(&[("host", "localhost:3000")]);
        assert!(check_request(&Method::GET, &h, "127.0.0.1").is_ok());
    }

    #[test]
    fn cross_site_requests_are_rejected() {
        // A malicious page trying to reconfigure the ASR endpoint.
        let h = headers(&[("host", "127.0.0.1:3000"), ("origin", "https://evil.example"), ("sec-fetch-site", "cross-site")]);
        assert!(check_request(&Method::POST, &h, "127.0.0.1").is_err());
        // Even without Fetch Metadata, a foreign Origin is rejected.
        let h = headers(&[("host", "127.0.0.1:3000"), ("origin", "https://evil.example")]);
        assert!(check_request(&Method::POST, &h, "127.0.0.1").is_err());
        // Cross-site reads (e.g. SSE / config) are refused too.
        let h = headers(&[("host", "127.0.0.1:3000"), ("sec-fetch-site", "cross-site")]);
        assert!(check_request(&Method::GET, &h, "127.0.0.1").is_err());
    }

    #[test]
    fn dns_rebinding_hosts_are_rejected() {
        let h = headers(&[("host", "attacker.example:3000")]);
        assert!(check_request(&Method::GET, &h, "127.0.0.1").is_err());
        let h = headers(&[("host", "192.168.1.10:3000")]);
        assert!(check_request(&Method::GET, &h, "127.0.0.1").is_err());
        assert!(check_request(&Method::GET, &h, "0.0.0.0").is_ok());
    }

    #[test]
    fn obs_browser_source_is_allowed() {
        // OBS loads the overlay directly (no Origin, Sec-Fetch-Site: none).
        let h = headers(&[("host", "localhost:3000"), ("sec-fetch-site", "none")]);
        assert!(check_request(&Method::GET, &h, "127.0.0.1").is_ok());
    }
}
