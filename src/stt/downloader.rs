//! Model downloader: multi-file models, pinned revisions, streaming SHA-256 and size
//! verification, resumable `.part` files and atomic installation.
//!
//! Downloading model weights is the only network access VoiceBridge performs, and it
//! only ever talks to the Hugging Face URLs pinned in the catalog.

use crate::stt::models::{find_model_by_path, is_model_file_valid, ModelFile, ModelMetadata};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadStatus {
    pub is_downloading: bool,
    pub model_id: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub percent: f32,
    pub speed_mb_s: f32,
    pub status_message: String,
    pub error: Option<String>,
}

impl Default for DownloadStatus {
    fn default() -> Self {
        Self {
            is_downloading: false,
            model_id: String::new(),
            downloaded_bytes: 0,
            total_bytes: 0,
            percent: 0.0,
            speed_mb_s: 0.0,
            status_message: "Prêt".to_string(),
            error: None,
        }
    }
}

pub type SharedDownloadStatus = Arc<RwLock<DownloadStatus>>;

/// Atomically reserve the downloader. Returns `false` if a download is already running.
pub async fn try_begin_download(status: &SharedDownloadStatus, model_id: &str, label: &str) -> bool {
    let mut st = status.write().await;
    if st.is_downloading {
        return false;
    }
    *st = DownloadStatus {
        is_downloading: true,
        model_id: model_id.to_string(),
        status_message: format!("Préparation du téléchargement de {label}..."),
        ..DownloadStatus::default()
    };
    true
}

/// Release the downloader with a final status.
pub async fn finish_download(status: &SharedDownloadStatus, result: &Result<(), String>) {
    let mut st = status.write().await;
    st.is_downloading = false;
    st.speed_mb_s = 0.0;
    match result {
        Ok(()) => {
            st.percent = 100.0;
            st.error = None;
            st.status_message = "Téléchargement terminé et vérifié.".to_string();
        }
        Err(e) => {
            st.error = Some(e.clone());
            st.status_message = format!("Échec du téléchargement : {e}");
        }
    }
}

fn http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent(concat!("VoiceBridge/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(15))
        .read_timeout(Duration::from_secs(60))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| e.to_string())
}

fn file_is_complete(file: &ModelFile) -> bool {
    match std::fs::metadata(&file.path) {
        Ok(m) if m.is_file() => file.size_bytes == 0 || m.len() == file.size_bytes,
        _ => false,
    }
}

/// Download every missing file of a catalog model. The caller must hold the downloader
/// reservation (see [`try_begin_download`]) and release it with [`finish_download`].
pub async fn download_model_files(
    meta: &ModelMetadata,
    status: &SharedDownloadStatus,
) -> Result<(), String> {
    if meta.files.is_empty() {
        return Err(format!("{} n'est pas un modèle téléchargeable.", meta.name));
    }
    let client = http_client()?;
    let total: u64 = meta.files.iter().map(|f| f.size_bytes).sum();
    let mut done: u64 = meta
        .files
        .iter()
        .filter(|f| file_is_complete(f))
        .map(|f| f.size_bytes)
        .sum();
    {
        let mut st = status.write().await;
        st.total_bytes = total;
        st.downloaded_bytes = done;
    }
    let started = Instant::now();
    let mut transferred: u64 = 0;
    for file in &meta.files {
        if file_is_complete(file) {
            continue;
        }
        let bytes =
            download_one(&client, file, &meta.name, status, done, total, started, &mut transferred).await?;
        done += bytes;
    }
    tracing::info!("Modèle {} téléchargé et vérifié.", meta.name);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn download_one(
    client: &reqwest::Client,
    file: &ModelFile,
    model_name: &str,
    status: &SharedDownloadStatus,
    done_before: u64,
    total: u64,
    started: Instant,
    transferred: &mut u64,
) -> Result<u64, String> {
    let path = Path::new(&file.path);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("impossible de créer {} : {e}", parent.display()))?;
    }
    let part_path = format!("{}.part", file.path);
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or(&file.path).to_string();

    // Resume a previous partial download when possible.
    let mut hasher = Sha256::new();
    let mut existing: u64 = 0;
    if let Ok(meta) = tokio::fs::metadata(&part_path).await {
        if file.size_bytes > 0 && meta.len() < file.size_bytes {
            existing = meta.len();
            let mut f = tokio::fs::File::open(&part_path).await.map_err(|e| e.to_string())?;
            let mut buf = vec![0u8; 1 << 20];
            loop {
                let n = f.read(&mut buf).await.map_err(|e| e.to_string())?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
            }
        } else {
            let _ = tokio::fs::remove_file(&part_path).await;
        }
    }

    let mut req = client.get(&file.url);
    if existing > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={existing}-"));
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("{file_name} : connexion impossible ({e})"))?;
    let status_code = resp.status();
    if existing > 0 && status_code != reqwest::StatusCode::PARTIAL_CONTENT {
        // Server ignored the range request: start over.
        existing = 0;
        hasher = Sha256::new();
    }
    if !status_code.is_success() {
        return Err(format!("{file_name} : HTTP {status_code}"));
    }

    let mut out = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(existing > 0)
        .truncate(existing == 0)
        .open(&part_path)
        .await
        .map_err(|e| format!("impossible d'écrire {part_path} : {e}"))?;

    let mut written = existing;
    let mut stream = resp.bytes_stream();
    let mut last_update = Instant::now();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("{file_name} : transfert interrompu ({e})"))?;
        hasher.update(&chunk);
        out.write_all(&chunk).await.map_err(|e| format!("écriture disque : {e}"))?;
        written += chunk.len() as u64;
        *transferred += chunk.len() as u64;

        if last_update.elapsed() >= Duration::from_millis(200) {
            last_update = Instant::now();
            let now = done_before + written;
            let speed = (*transferred as f32 / (1024.0 * 1024.0)) / started.elapsed().as_secs_f32().max(0.001);
            let pct = if total > 0 { now as f32 / total as f32 * 100.0 } else { 0.0 };
            let mut st = status.write().await;
            st.downloaded_bytes = now;
            st.speed_mb_s = speed;
            st.percent = pct.min(99.9);
            st.status_message = format!(
                "{model_name} — {file_name} : {:.1}% ({} / {} Mo) à {:.1} Mo/s",
                pct,
                now / (1024 * 1024),
                total / (1024 * 1024),
                speed
            );
        }
    }
    out.flush().await.map_err(|e| e.to_string())?;
    drop(out);

    if file.size_bytes > 0 && written != file.size_bytes {
        let _ = tokio::fs::remove_file(&part_path).await;
        return Err(format!(
            "{file_name} : taille incorrecte ({written} octets reçus, {} attendus)",
            file.size_bytes
        ));
    }
    if let Some(expected) = &file.sha256 {
        let actual = format!("{:x}", hasher.finalize());
        if !actual.eq_ignore_ascii_case(expected) {
            let _ = tokio::fs::remove_file(&part_path).await;
            return Err(format!("{file_name} : empreinte SHA-256 invalide (fichier corrompu)"));
        }
    }
    tokio::fs::rename(&part_path, &file.path)
        .await
        .map_err(|e| format!("installation de {file_name} : {e}"))?;
    Ok(written)
}

/// Make sure the Whisper model used at startup is present, downloading it when needed.
pub async fn ensure_model_exists(
    model_path: &str,
    fallback_url: &str,
    status: &SharedDownloadStatus,
) -> Result<(), String> {
    let meta = match find_model_by_path(model_path) {
        Some(mut m) => {
            // Respect a custom location for a catalog model.
            if let Some(f) = m.files.first_mut() {
                f.path = model_path.to_string();
            }
            m.default_model_name = model_path.to_string();
            m
        }
        None => {
            if is_model_file_valid(Path::new(model_path), 0) {
                return Ok(());
            }
            if fallback_url.trim().is_empty() {
                return Err(format!("modèle introuvable : {model_path}"));
            }
            ModelMetadata {
                id: "custom_whisper".into(),
                name: model_path.into(),
                provider: String::new(),
                description: String::new(),
                kind: crate::stt::models::KIND_STT.into(),
                default_engine: crate::stt::models::ENGINE_WHISPER.into(),
                default_model_name: model_path.into(),
                default_api_url: String::new(),
                download_url: Some(fallback_url.into()),
                files: vec![ModelFile {
                    url: fallback_url.into(),
                    path: model_path.into(),
                    size_bytes: 0,
                    sha256: None,
                }],
                file_size_mb: 0,
                supports_diarization: false,
                is_local_gpu: true,
                license: String::new(),
                badge: String::new(),
                languages: Vec::new(),
                auto_start_server: false,
            }
        }
    };
    if crate::stt::models::is_model_installed(&meta) {
        return Ok(());
    }
    // A wrong-size file of a catalog model is replaced.
    if Path::new(model_path).exists() {
        tracing::warn!("Fichier modèle '{}' incomplet ou invalide, nouveau téléchargement...", model_path);
        let _ = tokio::fs::remove_file(model_path).await;
    }
    tracing::info!("Modèle Whisper absent ({}), téléchargement...", model_path);
    if !try_begin_download(status, &meta.id, &meta.name).await {
        return Err("un téléchargement est déjà en cours".into());
    }
    let res = download_model_files(&meta, status).await;
    finish_download(status, &res).await;
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reservation_is_exclusive_and_released_on_error() {
        let status: SharedDownloadStatus = Arc::new(RwLock::new(DownloadStatus::default()));
        assert!(try_begin_download(&status, "a", "A").await);
        assert!(!try_begin_download(&status, "b", "B").await);
        finish_download(&status, &Err("boom".into())).await;
        let st = status.read().await.clone();
        assert!(!st.is_downloading);
        assert_eq!(st.error.as_deref(), Some("boom"));
        assert!(try_begin_download(&status, "b", "B").await);
    }
}
