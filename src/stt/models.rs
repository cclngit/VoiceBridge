use serde::{Deserialize, Serialize};

/// Returns the compiled hardware GPU backend label for `whisper.cpp`, if any.
pub const fn compiled_gpu_backend_label() -> Option<&'static str> {
    if cfg!(target_os = "macos") {
        Some("Metal GPU")
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

/// Returns the human-readable default Whisper execution backend (`Metal GPU`, `CUDA GPU`, `Vulkan GPU`, `ROCm GPU`, or `CPU`).
pub const fn active_whisper_backend_label() -> &'static str {
    match compiled_gpu_backend_label() {
        Some(gpu) => gpu,
        None => "CPU",
    }
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
            name: format!("Distil-Whisper Large-v3 ({})", hw),
            provider: "Hugging Face / whisper.cpp".to_string(),
            description: "Modèle distillé par Hugging Face (756 Mo). 6x plus rapide que Large-v3 avec moins de 1% d'écart de WER.".to_string(),
            default_engine: "whisper_local".to_string(),
            default_model_name: "models/ggml-distil-large-v3.bin".to_string(),
            default_api_url: "".to_string(),
            download_url: Some("https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin".to_string()),
            file_size_mb: 756,
            supports_diarization: false,
            is_local_gpu: true,
            license: "MIT".to_string(),
            badge: "Distil 756 Mo".to_string(),
        },

        // Advanced Open-Source Models (Local Host via Speaches / vLLM / Ollama)
        ModelMetadata {
            id: "qwen3_asr".to_string(),
            name: "Qwen3-ASR-1.7B".to_string(),
            provider: "Alibaba Cloud".to_string(),
            description: "Modèle SOTA ultra-précis. Exécutable 100% en local sur votre Mac via Speaches (http://127.0.0.1:8000) ou vLLM.".to_string(),
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

