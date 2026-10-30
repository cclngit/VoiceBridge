pub mod downloader;
pub mod engine;
pub mod models;

pub use downloader::{download_model_with_progress, ensure_model_exists, DownloadStatus};
pub use engine::{register_atexit_whisper_cleanup, RemoteAsrClient, WhisperEngine};
pub use models::{
    active_hardware_summary, active_whisper_backend_label, compiled_gpu_backends, find_model,
    find_model_by_path, get_available_models, has_compiled_gpu_backend, is_model_installed,
    list_gpu_devices, resolve_gpu_selection, resolve_model_download_url, GpuDeviceInfo,
    ModelMetadata,
};
