pub mod downloader;
pub mod engine;
pub mod models;

pub use downloader::{download_model_with_progress, ensure_model_exists, DownloadStatus};
pub use engine::{register_atexit_whisper_cleanup, RemoteAsrClient, WhisperEngine};
pub use models::{
    find_model, find_model_by_path, get_available_models, is_model_installed,
    resolve_model_download_url, ModelMetadata,
};
