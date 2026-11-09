pub mod downloader;
pub mod engine;
pub mod kyutai_engine;
pub mod models;

pub use downloader::DownloadStatus;
pub use engine::{register_atexit_whisper_cleanup, RemoteAsrClient, WhisperEngine};
pub use models::{
    active_hardware_summary, active_whisper_backend_label, compiled_gpu_backends, find_model,
    get_available_models, has_compiled_gpu_backend, is_model_installed, list_gpu_devices,
    resolve_gpu_selection, GpuDeviceInfo, ModelMetadata,
};
