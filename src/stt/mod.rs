pub mod downloader;
pub mod engine;
pub mod models;

pub use downloader::{download_model_with_progress, ensure_model_exists, DownloadStatus};
pub use engine::{RemoteAsrClient, WhisperEngine};
pub use models::{find_model, get_available_models, is_model_installed, ModelMetadata};
