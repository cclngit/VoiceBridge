use serde::{Deserialize, Serialize};
use std::ffi::CStr;

/// Information about a physical GPU device detected at runtime by `ggml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    /// 0-based GPU index passed to `WhisperContextParameters::gpu_device`.
    pub index: i32,
    /// Backend family name (`CUDA`, `Vulkan`, `Metal`, `ROCm`, `GPU`).
    pub backend: String,
    /// Internal ggml device identifier (`CUDA0`, `Vulkan0`, `MTL0`, etc.).
    pub dev_name: String,
    /// Hardware description (`NVIDIA GeForce RTX 5060`, `Apple M3 Max`, etc.).
    pub description: String,
    /// Free VRAM in MB.
    pub vram_free_mb: u64,
    /// Total VRAM in MB.
    pub vram_total_mb: u64,
}

/// Returns the list of compiled GPU backend identifiers.
pub fn compiled_gpu_backends() -> Vec<&'static str> {
    let mut list = Vec::new();
    if cfg!(target_os = "macos") {
        list.push("metal");
    }
    if cfg!(feature = "cuda") {
        list.push("cuda");
    }
    if cfg!(feature = "vulkan") {
        list.push("vulkan");
    }
    if cfg!(feature = "hipblas") {
        list.push("rocm");
    }
    list
}

/// Returns the compiled hardware GPU backend label for `whisper.cpp`, if any.
pub const fn compiled_gpu_backend_label() -> Option<&'static str> {
    if cfg!(target_os = "macos") {
        Some("Metal GPU")
    } else if cfg!(all(feature = "cuda", feature = "vulkan")) {
        Some("CUDA / Vulkan GPU")
    } else if cfg!(feature = "cuda") {
        Some("CUDA GPU")
    } else if cfg!(feature = "vulkan") {
        Some("Vulkan GPU")
    } else if cfg!(feature = "hipblas") {
        Some("ROCm GPU")
    } else {
        None
    }
}

/// Returns whether a GPU backend (`metal`, `cuda`, `vulkan`, or `hipblas`) was compiled in.
pub const fn has_compiled_gpu_backend() -> bool {
    compiled_gpu_backend_label().is_some()
}

/// Enumerates all physical GPU / iGPU devices visible to `ggml` at runtime.
pub fn list_gpu_devices() -> Vec<GpuDeviceInfo> {
    let mut devices = Vec::new();
    if !has_compiled_gpu_backend() {
        return devices;
    }

    unsafe {
        let count = whisper_rs::whisper_rs_sys::ggml_backend_dev_count();
        let mut gpu_idx: i32 = 0;

        for i in 0..count {
            let dev = whisper_rs::whisper_rs_sys::ggml_backend_dev_get(i);
            if dev.is_null() {
                continue;
            }
            let dev_type = whisper_rs::whisper_rs_sys::ggml_backend_dev_type(dev);
            if dev_type == whisper_rs::whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
                || dev_type == whisper_rs::whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU
            {
                let name_ptr = whisper_rs::whisper_rs_sys::ggml_backend_dev_name(dev);
                let dev_name = if !name_ptr.is_null() {
                    CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
                } else {
                    format!("GPU{}", gpu_idx)
                };

                let desc_ptr = whisper_rs::whisper_rs_sys::ggml_backend_dev_description(dev);
                let description = if !desc_ptr.is_null() {
                    CStr::from_ptr(desc_ptr).to_string_lossy().trim().to_string()
                } else {
                    dev_name.clone()
                };

                let reg = whisper_rs::whisper_rs_sys::ggml_backend_dev_backend_reg(dev);
                let backend = if !reg.is_null() {
                    let reg_name_ptr = whisper_rs::whisper_rs_sys::ggml_backend_reg_name(reg);
                    if !reg_name_ptr.is_null() {
                        CStr::from_ptr(reg_name_ptr).to_string_lossy().into_owned()
                    } else {
                        infer_backend_from_dev_name(&dev_name)
                    }
                } else {
                    infer_backend_from_dev_name(&dev_name)
                };

                let mut free_bytes: usize = 0;
                let mut total_bytes: usize = 0;
                whisper_rs::whisper_rs_sys::ggml_backend_dev_memory(
                    dev,
                    &mut free_bytes,
                    &mut total_bytes,
                );

                devices.push(GpuDeviceInfo {
                    index: gpu_idx,
                    backend,
                    dev_name,
                    description,
                    vram_free_mb: (free_bytes as u64) / (1024 * 1024),
                    vram_total_mb: (total_bytes as u64) / (1024 * 1024),
                });

                gpu_idx += 1;
            }
        }
    }

    devices
}

fn infer_backend_from_dev_name(dev_name: &str) -> String {
    let lower = dev_name.to_ascii_lowercase();
    if lower.starts_with("cuda") {
        "CUDA".to_string()
    } else if lower.starts_with("vulkan") || lower.starts_with("vk") {
        "Vulkan".to_string()
    } else if lower.starts_with("mtl") || lower.starts_with("metal") {
        "Metal".to_string()
    } else if lower.starts_with("hip") || lower.starts_with("rocm") {
        "ROCm".to_string()
    } else {
        "GPU".to_string()
    }
}

/// Resolves user GPU settings (`use_gpu`, `gpu_backend`, `gpu_device`) into
/// `(effective_use_gpu, effective_gpu_device_index, display_label)`.
pub fn resolve_gpu_selection(
    use_gpu: bool,
    gpu_backend: &str,
    gpu_device: i32,
) -> (bool, i32, String) {
    let backend_req = gpu_backend.trim().to_ascii_lowercase();
    if !use_gpu || backend_req == "cpu" || !has_compiled_gpu_backend() {
        return (false, 0, "CPU".to_string());
    }

    let devices = list_gpu_devices();
    if devices.is_empty() {
        // If runtime enumeration found no devices yet, still allow compiled backend attempt on index 0
        if let Some(compiled) = compiled_gpu_backend_label() {
            return (true, gpu_device.max(0), compiled.to_string());
        }
        return (false, 0, "CPU".to_string());
    }

    // Filter by requested backend if user explicitly selected "cuda", "vulkan", "metal", or "rocm"
    let chosen = if backend_req != "auto" && !backend_req.is_empty() {
        devices
            .iter()
            .find(|d| {
                d.index == gpu_device && d.backend.to_ascii_lowercase().contains(&backend_req)
            })
            .or_else(|| {
                devices
                    .iter()
                    .find(|d| d.backend.to_ascii_lowercase().contains(&backend_req))
            })
            .or_else(|| devices.iter().find(|d| d.index == gpu_device))
            .unwrap_or(&devices[0])
    } else {
        devices
            .iter()
            .find(|d| d.index == gpu_device)
            .unwrap_or(&devices[0])
    };

    let label = if !chosen.description.is_empty()
        && !chosen.description.eq_ignore_ascii_case(&chosen.backend)
    {
        format!("{} ({} GPU)", chosen.description, chosen.backend)
    } else {
        format!("{} GPU", chosen.backend)
    };

    (true, chosen.index, label)
}

/// Returns the human-readable default Whisper execution backend (`Metal GPU`, `CUDA GPU`, `Vulkan GPU`, `ROCm GPU`, or `CPU`).
pub const fn active_whisper_backend_label() -> &'static str {
    match compiled_gpu_backend_label() {
        Some(gpu) => gpu,
        None => "CPU",
    }
}

/// Returns a rich hardware label including the detected GPU card model when available (e.g., `NVIDIA GeForce RTX 5060 (Vulkan GPU)`).
pub fn active_hardware_summary(use_gpu: bool, gpu_backend: &str, gpu_device: i32) -> String {
    let (_, _, label) = resolve_gpu_selection(use_gpu, gpu_backend, gpu_device);
    label
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub description: String,
    pub default_engine: String,
    pub default_model_name: String,
    pub default_api_url: String,
    pub download_url: Option<String>,
    pub file_size_mb: u32,
    pub supports_diarization: bool,
    pub is_local_gpu: bool,
    pub license: String,
    pub badge: String,
}

pub fn get_available_models() -> Vec<ModelMetadata> {
    let hw = active_whisper_backend_label();
    vec![
        // Local Whisper Models (Metal / CUDA / Vulkan / ROCm GPU or multithreaded CPU)
        ModelMetadata {
            id: "whisper_tiny".to_string(),
            name: format!("Whisper Tiny ({})", hw),
            provider: "OpenAI / whisper.cpp".to_string(),
            description: "Modèle ultra-rapide (75 Mo). Tourne à plus de 10x la vitesse réelle en local. Idéal pour une latence minimale.".to_string(),
            default_engine: "whisper_local".to_string(),
            default_model_name: "models/ggml-tiny.bin".to_string(),
            default_api_url: "".to_string(),
            download_url: Some("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin".to_string()),
            file_size_mb: 75,
            supports_diarization: false,
            is_local_gpu: true,
            license: "MIT".to_string(),
            badge: "Local 75 Mo".to_string(),
        },
        ModelMetadata {
            id: "whisper_base".to_string(),
            name: format!("Whisper Base ({})", hw),
            provider: "OpenAI / whisper.cpp".to_string(),
            description: "Le compromis idéal rapidité/précision (142 Mo). Très bonne reconnaissance des accents et de la ponctuation.".to_string(),
            default_engine: "whisper_local".to_string(),
            default_model_name: "models/ggml-base.bin".to_string(),
            default_api_url: "".to_string(),
            download_url: Some("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin".to_string()),
            file_size_mb: 142,
            supports_diarization: false,
            is_local_gpu: true,
            license: "MIT".to_string(),
            badge: "Local 142 Mo".to_string(),
        },
        ModelMetadata {
            id: "whisper_small".to_string(),
            name: format!("Whisper Small ({})", hw),
            provider: "OpenAI / whisper.cpp".to_string(),
            description: "Haute précision (466 Mo). Excellent pour le vocabulaire technique et les phrases complexes avec peu de fautes.".to_string(),
            default_engine: "whisper_local".to_string(),
            default_model_name: "models/ggml-small.bin".to_string(),
            default_api_url: "".to_string(),
            download_url: Some("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin".to_string()),
            file_size_mb: 466,
            supports_diarization: false,
            is_local_gpu: true,
            license: "MIT".to_string(),
            badge: "Local 466 Mo".to_string(),
        },
        ModelMetadata {
            id: "whisper_small_q5".to_string(),
            name: format!("Whisper Small Q5_1 ({})", hw),
            provider: "OpenAI / whisper.cpp".to_string(),
            description: "Version quantisée ultra-compacte (190 Mo). Offre 95% de la précision de Small tout en occupant 2.5x moins de mémoire RAM.".to_string(),
            default_engine: "whisper_local".to_string(),
            default_model_name: "models/ggml-small-q5_1.bin".to_string(),
            default_api_url: "".to_string(),
            download_url: Some("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin".to_string()),
            file_size_mb: 190,
            supports_diarization: false,
            is_local_gpu: true,
            license: "MIT".to_string(),
            badge: "Local 190 Mo".to_string(),
        },
        ModelMetadata {
            id: "whisper_medium".to_string(),
            name: format!("Whisper Medium ({})", hw),
            provider: "OpenAI / whisper.cpp".to_string(),
            description: "Qualité studio (1.5 Go). Compréhension phénoménale du français, des accents et des termes techniques.".to_string(),
            default_engine: "whisper_local".to_string(),
            default_model_name: "models/ggml-medium.bin".to_string(),
            default_api_url: "".to_string(),
            download_url: Some("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin".to_string()),
            file_size_mb: 1530,
            supports_diarization: false,
            is_local_gpu: true,
            license: "MIT".to_string(),
            badge: "Studio 1.5 Go".to_string(),
        },
        ModelMetadata {
            id: "whisper_medium_q5".to_string(),
            name: format!("Whisper Medium Q5_0 ({})", hw),
            provider: "OpenAI / whisper.cpp".to_string(),
            description: "Modèle Medium compressé en 515 Mo. La richesse du Medium avec la légèreté et la vitesse d'un petit modèle !".to_string(),
            default_engine: "whisper_local".to_string(),
            default_model_name: "models/ggml-medium-q5_0.bin".to_string(),
            default_api_url: "".to_string(),
            download_url: Some("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin".to_string()),
            file_size_mb: 515,
            supports_diarization: false,
            is_local_gpu: true,
            license: "MIT".to_string(),
            badge: "Local 515 Mo".to_string(),
        },
        ModelMetadata {
            id: "whisper_turbo".to_string(),
            name: format!("Whisper Large-v3-Turbo ({})", hw),
            provider: "OpenAI / whisper.cpp".to_string(),
            description: "Le sommet de la vitesse et précision Whisper (1.5 Go). Architecture distillée Turbo optimisée pour l'inférence locale.".to_string(),
            default_engine: "whisper_local".to_string(),
            default_model_name: "models/ggml-large-v3-turbo.bin".to_string(),
            default_api_url: "".to_string(),
            download_url: Some("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin".to_string()),
            file_size_mb: 1550,
            supports_diarization: false,
            is_local_gpu: true,
            license: "MIT".to_string(),
            badge: "Turbo 1.5 Go".to_string(),
        },
        ModelMetadata {
            id: "whisper_turbo_q5".to_string(),
            name: format!("Whisper Large-v3-Turbo Q5_0 ({})", hw),
            provider: "OpenAI / whisper.cpp".to_string(),
            description: "La puissance ultime de Turbo en 547 Mo seulement ! Idéal pour les machines de 8 à 16 Go de RAM.".to_string(),
            default_engine: "whisper_local".to_string(),
            default_model_name: "models/ggml-large-v3-turbo-q5_0.bin".to_string(),
            default_api_url: "".to_string(),
            download_url: Some("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin".to_string()),
            file_size_mb: 547,
            supports_diarization: false,
            is_local_gpu: true,
            license: "MIT".to_string(),
            badge: "⚡ Turbo 547 Mo".to_string(),
        },
        ModelMetadata {
            id: "whisper_large_v3".to_string(),
            name: format!("Whisper Large-v3 ({})", hw),
            provider: "OpenAI / whisper.cpp".to_string(),
            description: "La référence absolue mondiale OpenAI (2.9 Go). 128 bandes Mel, vocabulaire intégral et taux d'erreur minimum absolu.".to_string(),
            default_engine: "whisper_local".to_string(),
            default_model_name: "models/ggml-large-v3.bin".to_string(),
            default_api_url: "".to_string(),
            download_url: Some("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin".to_string()),
            file_size_mb: 2900,
            supports_diarization: false,
            is_local_gpu: true,
            license: "MIT".to_string(),
            badge: "Gold Standard 2.9 Go".to_string(),
        },
        ModelMetadata {
            id: "whisper_large_v3_q5".to_string(),
            name: format!("Whisper Large-v3 Q5_0 ({})", hw),
            provider: "OpenAI / whisper.cpp".to_string(),
            description: "Large-v3 complet quantisé en 1 Go. Qualité maximale avec une consommation mémoire et disque divisée par 3.".to_string(),
            default_engine: "whisper_local".to_string(),
            default_model_name: "models/ggml-large-v3-q5_0.bin".to_string(),
            default_api_url: "".to_string(),
            download_url: Some("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_0.bin".to_string()),
            file_size_mb: 1080,
            supports_diarization: false,
            is_local_gpu: true,
            license: "MIT".to_string(),
            badge: "Flagship 1 Go".to_string(),
        },
        ModelMetadata {
            id: "distil_whisper_large_v3".to_string(),
            name: format!("Distil-Whisper Large-v3 [Anglais] ({})", hw),
            provider: "Hugging Face / whisper.cpp".to_string(),
            description: "Modèle distillé par Hugging Face (756 Mo) optimisé exclusivement pour l'anglais. Non adapté au français.".to_string(),
            default_engine: "whisper_local".to_string(),
            default_model_name: "models/ggml-distil-large-v3.bin".to_string(),
            default_api_url: "".to_string(),
            download_url: Some("https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin".to_string()),
            file_size_mb: 756,
            supports_diarization: false,
            is_local_gpu: true,
            license: "MIT".to_string(),
            badge: "Anglais uniquement • 756 Mo".to_string(),
        },

        // Advanced Open-Source Models (Local Host via Speaches / vLLM / Ollama)
        ModelMetadata {
            id: "qwen3_asr".to_string(),
            name: "Qwen3-ASR-1.7B".to_string(),
            provider: "Alibaba Cloud".to_string(),
            description: "Modèle SOTA ultra-précis. Exécutable 100% en local sur votre machine via Speaches (http://127.0.0.1:8000) ou vLLM.".to_string(),
            default_engine: "remote_api".to_string(),
            default_model_name: "Qwen/Qwen3-ASR-1.7B".to_string(),
            default_api_url: "http://127.0.0.1:8000/v1/audio/transcriptions".to_string(),
            download_url: None,
            file_size_mb: 3400,
            supports_diarization: false,
            is_local_gpu: false,
            license: "Apache 2.0".to_string(),
            badge: "Local Speaches / vLLM".to_string(),
        },
        ModelMetadata {
            id: "voxtral_mini".to_string(),
            name: "Voxtral Mini Realtime (4B)".to_string(),
            provider: "Mistral Architecture".to_string(),
            description: "Optimisé spécifiquement pour les agents vocaux et le streaming à ultra-faible latence en local.".to_string(),
            default_engine: "remote_api".to_string(),
            default_model_name: "mistralai/Voxtral-Mini-4B".to_string(),
            default_api_url: "http://127.0.0.1:8000/v1/audio/transcriptions".to_string(),
            download_url: None,
            file_size_mb: 4200,
            supports_diarization: false,
            is_local_gpu: false,
            license: "Apache 2.0".to_string(),
            badge: "Streaming 4B Local".to_string(),
        },
        ModelMetadata {
            id: "voxtral_small".to_string(),
            name: "Voxtral Small (24B)".to_string(),
            provider: "Mistral Architecture".to_string(),
            description: "Modèle multimodal haute capacité pour saisir les intonations fines et contextes complexes en local.".to_string(),
            default_engine: "remote_api".to_string(),
            default_model_name: "mistralai/Voxtral-Small-24B".to_string(),
            default_api_url: "http://127.0.0.1:8000/v1/audio/transcriptions".to_string(),
            download_url: None,
            file_size_mb: 24000,
            supports_diarization: false,
            is_local_gpu: false,
            license: "Apache 2.0".to_string(),
            badge: "24B Puissance Local".to_string(),
        },
        ModelMetadata {
            id: "canary_qwen".to_string(),
            name: "Canary-Qwen-2.5B".to_string(),
            provider: "NVIDIA NeMo".to_string(),
            description: "Précision phénoménale développée par NVIDIA pour les transcriptions et traductions multilingues professionnelles.".to_string(),
            default_engine: "remote_api".to_string(),
            default_model_name: "nvidia/canary-qwen-2.5b".to_string(),
            default_api_url: "http://127.0.0.1:8000/v1/audio/transcriptions".to_string(),
            download_url: None,
            file_size_mb: 5000,
            supports_diarization: false,
            is_local_gpu: false,
            license: "CC-BY-4.0".to_string(),
            badge: "NVIDIA Local".to_string(),
        },
        ModelMetadata {
            id: "vibevoice_asr".to_string(),
            name: "VibeVoice-ASR (Diarisation native)".to_string(),
            provider: "Microsoft".to_string(),
            description: "Intègre nativement la diarisation pour séparer automatiquement les voix de chaque participant sans script externe.".to_string(),
            default_engine: "remote_api".to_string(),
            default_model_name: "microsoft/vibevoice-asr".to_string(),
            default_api_url: "http://127.0.0.1:8000/v1/audio/transcriptions".to_string(),
            download_url: None,
            file_size_mb: 4000,
            supports_diarization: true,
            is_local_gpu: false,
            license: "MIT".to_string(),
            badge: "Diarisation Multi-Voix".to_string(),
        },
        ModelMetadata {
            id: "moonshine".to_string(),
            name: "Moonshine (Ultra-léger CPU)".to_string(),
            provider: "Useful Sensors".to_string(),
            description: "Modèle ultra-compact et économe, optimisé pour tourner à pleine vitesse sur processeur sans GPU dédié.".to_string(),
            default_engine: "remote_api".to_string(),
            default_model_name: "usefulsensors/moonshine-base".to_string(),
            default_api_url: "http://127.0.0.1:8000/v1/audio/transcriptions".to_string(),
            download_url: None,
            file_size_mb: 120,
            supports_diarization: false,
            is_local_gpu: false,
            license: "MIT".to_string(),
            badge: "CPU Léger Local".to_string(),
        },
        ModelMetadata {
            id: "custom_api".to_string(),
            name: "Serveur ASR Personnalisé (Local / Distant)".to_string(),
            provider: "Speaches / vLLM / Ollama / OpenAI / Groq".to_string(),
            description: "Connectez n'importe quel serveur local (127.0.0.1) ou distant compatible /v1/audio/transcriptions.".to_string(),
            default_engine: "remote_api".to_string(),
            default_model_name: "whisper-1".to_string(),
            default_api_url: "http://127.0.0.1:8000/v1/audio/transcriptions".to_string(),
            download_url: None,
            file_size_mb: 0,
            supports_diarization: false,
            is_local_gpu: false,
            license: "Ouvert".to_string(),
            badge: "Endpoint Compatible".to_string(),
        },
    ]
}

pub fn find_model(model_id: &str) -> Option<ModelMetadata> {
    get_available_models().into_iter().find(|m| m.id == model_id)
}

pub fn find_model_by_path(model_path: &str) -> Option<ModelMetadata> {
    let target_name = std::path::Path::new(model_path).file_name();
    get_available_models().into_iter().find(|m| {
        m.is_local_gpu
            && (m.default_model_name == model_path
                || (target_name.is_some()
                    && std::path::Path::new(&m.default_model_name).file_name() == target_name))
    })
}

pub fn resolve_model_download_url(
    model_id: &str,
    model_path: &str,
    configured_url: &str,
) -> String {
    if let Some(url) = find_model_by_path(model_path).and_then(|m| m.download_url) {
        return url;
    }
    if let Some(m) = find_model(model_id) {
        if m.default_model_name == model_path {
            if let Some(url) = m.download_url {
                return url;
            }
        }
    }
    configured_url.to_string()
}

pub fn is_model_file_valid(path: &std::path::Path, expected_size_mb: u32) -> bool {
    let meta = match std::fs::metadata(path) {
        Ok(m) if m.is_file() => m,
        _ => return false,
    };
    if expected_size_mb > 0 {
        // Require at least 70% of catalog size to catch truncated downloads or wrong GGML files
        let min_bytes = (expected_size_mb as u64) * 1024 * 1024 * 7 / 10;
        meta.len() >= min_bytes
    } else {
        meta.len() > 1024 * 1024
    }
}

pub fn is_model_installed(model: &ModelMetadata) -> bool {
    if !model.is_local_gpu {
        return false;
    }
    is_model_file_valid(
        std::path::Path::new(&model.default_model_name),
        model.file_size_mb,
    )
}
