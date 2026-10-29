use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
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

pub async fn ensure_model_exists(
    model_path: &str,
    fallback_url: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let path = Path::new(model_path);
    if path.exists() {
        return Ok(());
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    tracing::info!(
        "Whisper model not found at '{}'. Automatically downloading from '{}'...",
        model_path,
        fallback_url
    );

    let client = reqwest::Client::builder()
        .user_agent("VoiceBridge/1.0 (Mozilla/5.0 Apple-Silicon)")
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?;
    let response = client.get(fallback_url).send().await?;

    if !response.status().is_success() {
        return Err(format!(
            "Failed to download model: HTTP status {}",
            response.status()
        )
        .into());
    }

    let bytes = response.bytes().await?;
    let mut file = File::create(path)?;
    file.write_all(&bytes)?;

    tracing::info!(
        "Successfully downloaded model to '{}' ({} MB)",
        model_path,
        bytes.len() / (1024 * 1024)
    );

    Ok(())
}

/// Download model with live byte-level progress reporting
pub async fn download_model_with_progress(
    model_path: &str,
    url: &str,
    model_id: &str,
    status_tracker: Arc<RwLock<DownloadStatus>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let path = Path::new(model_path);
    if path.exists() {
        let mut st = status_tracker.write().await;
        st.is_downloading = false;
        st.percent = 100.0;
        st.status_message = "Modèle déjà téléchargé et prêt.".to_string();
        st.error = None;
        return Ok(());
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    {
        let mut st = status_tracker.write().await;
        st.is_downloading = true;
        st.model_id = model_id.to_string();
        st.downloaded_bytes = 0;
        st.total_bytes = 0;
        st.percent = 0.0;
        st.speed_mb_s = 0.0;
        st.status_message = "Connexion aux serveurs HuggingFace...".to_string();
        st.error = None;
    }

    let client = reqwest::Client::builder()
        .user_agent("VoiceBridge/1.0 (Mozilla/5.0 Apple-Silicon)")
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?;
    let resp = client.get(url).send().await?;

    if !resp.status().is_success() {
        let err_msg = format!("HTTP {}", resp.status());
        let mut st = status_tracker.write().await;
        st.is_downloading = false;
        st.error = Some(err_msg.clone());
        return Err(err_msg.into());
    }

    let total_size = resp.content_length().unwrap_or(0);
    {
        let mut st = status_tracker.write().await;
        st.total_bytes = total_size;
        st.status_message = format!("Téléchargement en cours ({} Mo)...", total_size / (1024 * 1024));
    }

    let temp_path = format!("{}.part", model_path);
    let mut file = File::create(&temp_path)?;
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;
    let start_time = Instant::now();
    let mut last_update = Instant::now();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;

        if last_update.elapsed().as_millis() > 150 {
            let elapsed_sec = start_time.elapsed().as_secs_f32().max(0.001);
            let speed_mb_s = (downloaded as f32 / (1024.0 * 1024.0)) / elapsed_sec;
            let percent = if total_size > 0 {
                (downloaded as f32 / total_size as f32) * 100.0
            } else {
                0.0
            };

            let mut st = status_tracker.write().await;
            st.downloaded_bytes = downloaded;
            st.speed_mb_s = speed_mb_s;
            st.percent = percent.min(99.9);
            st.status_message = format!(
                "{:.1}% ({} Mo / {} Mo) à {:.1} Mo/s",
                percent,
                downloaded / (1024 * 1024),
                total_size / (1024 * 1024),
                speed_mb_s
            );

            last_update = Instant::now();
        }
    }

    fs::rename(&temp_path, model_path)?;

    {
        let mut st = status_tracker.write().await;
        st.is_downloading = false;
        st.downloaded_bytes = downloaded;
        st.percent = 100.0;
        st.status_message = "Téléchargement terminé avec succès ! Chargement GPU Metal...".to_string();
    }

    Ok(())
}
