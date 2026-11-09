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

pub const ENGINE_WHISPER: &str = "whisper_local";
pub const ENGINE_REMOTE: &str = "remote_asr";
pub const ENGINE_KYUTAI: &str = "kyutai_streaming";
pub const ENGINE_HIBIKI: &str = "hibiki_streaming";
pub const ENGINE_MADLAD: &str = "madlad";

pub const KIND_STT: &str = "stt";
pub const KIND_SPEECH_TRANSLATION: &str = "speech_translation";
pub const KIND_TRANSLATION: &str = "translation";

pub const KYUTAI_STT_MODEL_ID: &str = "kyutai_stt_1b";
pub const HIBIKI_1B_MODEL_ID: &str = "kyutai_hibiki_1b";
pub const HIBIKI_2B_MODEL_ID: &str = "kyutai_hibiki_2b";
pub const MADLAD_MODEL_ID: &str = "madlad400_3b";
pub const MADLAD_DEFAULT_DIR: &str = "models/madlad400-3b-mt";

/// One file of a downloadable model, pinned to an exact Hugging Face revision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFile {
    pub url: String,
    /// Local path (relative to the working directory).
    pub path: String,
    /// Exact size in bytes (0 = unknown).
    pub size_bytes: u64,
    /// Expected SHA-256 (lower-case hex) when published by the hub.
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub description: String,
    /// `stt`, `speech_translation` (audio in, translated text out) or `translation` (text MT).
    pub kind: String,
    pub default_engine: String,
    /// Model file (Whisper) or model directory (Kyutai / Hibiki / MADLAD).
    pub default_model_name: String,
    pub default_api_url: String,
    pub download_url: Option<String>,
    pub files: Vec<ModelFile>,
    pub file_size_mb: u32,
    pub supports_diarization: bool,
    /// Runs in-process on this machine (GPU or CPU).
    pub is_local_gpu: bool,
    pub license: String,
    pub badge: String,
    /// Spoken languages supported (empty = multilingual).
    pub languages: Vec<String>,
    /// Server models only: served by the bundled `scripts/asr_server.py` (started
    /// automatically). Otherwise the user runs their own local server (vLLM, Speaches...).
    #[serde(default)]
    pub auto_start_server: bool,
}

const HF: &str = "https://huggingface.co";
const WHISPER_REPO: &str = "ggerganov/whisper.cpp";
const WHISPER_REV: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";
const DISTIL_REPO: &str = "distil-whisper/distil-large-v3-ggml";
const DISTIL_REV: &str = "0d78dd96ed9fc152325f63b53788fec3b43de031";
const KYUTAI_STT_REPO: &str = "kyutai/stt-1b-en_fr-candle";
const KYUTAI_STT_REV: &str = "095e38f6242006a93c2541149b181988397f5c7c";
const HIBIKI_1B_REPO: &str = "kyutai/hibiki-1b-rs-bf16";
const HIBIKI_1B_REV: &str = "7ab972c47bd4e857818cc8358867035e5632eb4c";
const HIBIKI_2B_REPO: &str = "kyutai/hibiki-2b-rs-bf16";
const HIBIKI_2B_REV: &str = "d378308f65760d4125280ffe76ea7fec05fd908b";
const MADLAD_REPO: &str = "jbochi/madlad400-3b-mt";
const MADLAD_REV: &str = "bb45f1851bf13a0b8f4fcd9ecadf2cdb7cf22439";

fn hf_file(repo: &str, rev: &str, name: &str, local: &str, size: u64, sha: Option<&str>) -> ModelFile {
    ModelFile {
        url: format!("{HF}/{repo}/resolve/{rev}/{name}"),
        path: local.to_string(),
        size_bytes: size,
        sha256: sha.map(str::to_string),
    }
}

fn total_mb(files: &[ModelFile]) -> u32 {
    (files.iter().map(|f| f.size_bytes).sum::<u64>() / (1024 * 1024)) as u32
}

#[allow(clippy::too_many_arguments)]
fn whisper(
    id: &str,
    title: &str,
    description: &str,
    repo: &str,
    rev: &str,
    file: &str,
    size: u64,
    sha: &str,
    badge: &str,
    languages: &[&str],
) -> ModelMetadata {
    let hw = active_whisper_backend_label();
    let local = format!("models/{file}");
    let files = vec![hf_file(repo, rev, file, &local, size, Some(sha))];
    ModelMetadata {
        id: id.to_string(),
        name: format!("{title} ({hw})"),
        provider: if repo == WHISPER_REPO { "OpenAI / whisper.cpp" } else { "Hugging Face / whisper.cpp" }.to_string(),
        description: description.to_string(),
        kind: KIND_STT.to_string(),
        default_engine: ENGINE_WHISPER.to_string(),
        default_model_name: local,
        default_api_url: String::new(),
        download_url: Some(files[0].url.clone()),
        file_size_mb: total_mb(&files),
        files,
        supports_diarization: false,
        is_local_gpu: true,
        license: "MIT".to_string(),
        badge: badge.to_string(),
        languages: languages.iter().map(|s| s.to_string()).collect(),
        auto_start_server: false,
    }
}

fn hibiki(id: &str, size_label: &str, repo: &str, rev: &str, lm_file: &str, lm_size: u64, lm_sha: &str) -> ModelMetadata {
    let hw = candle_backend_label();
    let dir = format!("models/{}", repo.trim_start_matches("kyutai/"));
    let files = vec![
        hf_file(repo, rev, "config.toml", &format!("{dir}/config.toml"), 0, None),
        hf_file(repo, rev, lm_file, &format!("{dir}/{lm_file}"), lm_size, Some(lm_sha)),
        hf_file(
            repo,
            rev,
            "mimi-dbaa9758@125.safetensors",
            &format!("{dir}/mimi-dbaa9758@125.safetensors"),
            384_644_900,
            Some("31c14cf365353131094e8248150c6fe58e8642cf91899c50d9e450f861630e55"),
        ),
        hf_file(
            repo,
            rev,
            "tokenizer_spm_48k_multi6_2.model",
            &format!("{dir}/tokenizer_spm_48k_multi6_2.model"),
            857_314,
            Some("c22110fb855aa049e17346ea2e88355bdd664f06cbfd09948380ab5e85b39697"),
        ),
    ];
    ModelMetadata {
        id: id.to_string(),
        name: format!("Kyutai Hibiki {size_label} — Traduction simultanée FR → EN ({hw})"),
        provider: "Kyutai Labs / Candle".to_string(),
        description: "Traduction simultanée parole française → texte anglais en temps réel (Moshi multistream, codec Mimi 12,5 Hz). Nécessite un GPU Metal/CUDA pour tenir le temps réel.".to_string(),
        kind: KIND_SPEECH_TRANSLATION.to_string(),
        default_engine: ENGINE_HIBIKI.to_string(),
        default_model_name: dir,
        default_api_url: String::new(),
        download_url: Some(files[1].url.clone()),
        file_size_mb: total_mb(&files),
        files,
        supports_diarization: false,
        is_local_gpu: true,
        license: "CC-BY-4.0".to_string(),
        badge: format!("🌐 Simultanée FR→EN • {size_label}"),
        languages: vec!["fr".to_string()],
        auto_start_server: false,
    }
}

/// Label of the accelerator used by the Candle models (Kyutai / Hibiki / MADLAD).
pub fn candle_backend_label() -> &'static str {
    if cfg!(target_os = "macos") {
        "Metal GPU"
    } else if cfg!(feature = "cuda") {
        "CUDA GPU"
    } else {
        "CPU"
    }
}

pub fn get_available_models() -> Vec<ModelMetadata> {
    let candle_hw = candle_backend_label();
    let kyutai_dir = "models/kyutai-stt-1b-en_fr-candle";
    let kyutai_files = vec![
        hf_file(KYUTAI_STT_REPO, KYUTAI_STT_REV, "config.json", &format!("{kyutai_dir}/config.json"), 1315, None),
        hf_file(
            KYUTAI_STT_REPO,
            KYUTAI_STT_REV,
            "model.safetensors",
            &format!("{kyutai_dir}/model.safetensors"),
            1_978_522_200,
            Some("b9e97c53229dce728d65c76bfa892f7b563c69d671899f0ebc6518582dddec6f"),
        ),
        hf_file(
            KYUTAI_STT_REPO,
            KYUTAI_STT_REV,
            "mimi-pytorch-e351c8d8@125.safetensors",
            &format!("{kyutai_dir}/mimi-pytorch-e351c8d8@125.safetensors"),
            384_644_900,
            Some("09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50"),
        ),
        hf_file(
            KYUTAI_STT_REPO,
            KYUTAI_STT_REV,
            "tokenizer_en_fr_audio_8000.model",
            &format!("{kyutai_dir}/tokenizer_en_fr_audio_8000.model"),
            120_378,
            Some("cd87dd5d17169151782ac700280ec057e5d658a9afbe238a048ea5ff318cce69"),
        ),
    ];
    let madlad_files = vec![
        hf_file(MADLAD_REPO, MADLAD_REV, "config.json", &format!("{MADLAD_DEFAULT_DIR}/config.json"), 749, None),
        hf_file(
            MADLAD_REPO,
            MADLAD_REV,
            "tokenizer.json",
            &format!("{MADLAD_DEFAULT_DIR}/tokenizer.json"),
            16_629_031,
            Some("a2799ccc696b752ba00c34f58726bfe253a04921ceb6cfc620400f560474790b"),
        ),
        hf_file(
            MADLAD_REPO,
            MADLAD_REV,
            "model-q4k.gguf",
            &format!("{MADLAD_DEFAULT_DIR}/model-q4k.gguf"),
            1_654_597_280,
            Some("ea6e5531a3e95213c7f0635988d119e078a655c09306e47851e15d4c0c3f9c37"),
        ),
    ];

    let mut models = vec![
        // ── Kyutai streaming STT (Candle, in-process) ──────────────────────────────
        ModelMetadata {
            id: KYUTAI_STT_MODEL_ID.to_string(),
            name: format!("Kyutai STT 1B Streaming FR/EN ({candle_hw})"),
            provider: "Kyutai Labs / Candle".to_string(),
            description: "Reconnaissance vocale en flux continu (Delayed Streams Modeling, codec Mimi 12,5 Hz) : mots affichés ~0,5 s après avoir été prononcés, fin de phrase détectée par VAD sémantique. Français et anglais.".to_string(),
            kind: KIND_STT.to_string(),
            default_engine: ENGINE_KYUTAI.to_string(),
            default_model_name: kyutai_dir.to_string(),
            default_api_url: String::new(),
            download_url: Some(kyutai_files[1].url.clone()),
            file_size_mb: total_mb(&kyutai_files),
            files: kyutai_files,
            supports_diarization: false,
            is_local_gpu: true,
            license: "CC-BY-4.0".to_string(),
            badge: "🚀 Streaming 0,5 s • FR/EN".to_string(),
            languages: vec!["fr".to_string(), "en".to_string()],
            auto_start_server: false,
        },
        // ── Kyutai Hibiki simultaneous speech translation ──────────────────────────
        hibiki(
            HIBIKI_1B_MODEL_ID,
            "1B",
            HIBIKI_1B_REPO,
            HIBIKI_1B_REV,
            "hibiki-rs-dc2cf5a5@80.safetensors",
            3_600_041_272,
            "59deb424db9738d88dde02e914f40dbb03e34c65b581d2e50e3b2ad698eb1e30",
        ),
        hibiki(
            HIBIKI_2B_MODEL_ID,
            "2B",
            HIBIKI_2B_REPO,
            HIBIKI_2B_REV,
            "hibiki-rs-220b12c0@200.safetensors",
            6_198_951_736,
            "6ea6dfd50d009bbe7dcdf73876fdbdc7e50152e4c1fab687d44cc1d7f887b586",
        ),
        // ── Local text translation ──────────────────────────────────────────────────
        ModelMetadata {
            id: MADLAD_MODEL_ID.to_string(),
            name: format!("MADLAD-400 3B — Traduction locale 400 langues ({candle_hw})"),
            provider: "Google Research / Candle".to_string(),
            description: "Traduction automatique 100 % locale (T5 3B quantifié 4 bits) utilisée pour traduire les transcriptions Whisper / Kyutai. Aucune donnée n'est envoyée sur Internet.".to_string(),
            kind: KIND_TRANSLATION.to_string(),
            default_engine: ENGINE_MADLAD.to_string(),
            default_model_name: MADLAD_DEFAULT_DIR.to_string(),
            default_api_url: String::new(),
            download_url: Some(madlad_files[2].url.clone()),
            file_size_mb: total_mb(&madlad_files),
            files: madlad_files,
            supports_diarization: false,
            is_local_gpu: true,
            license: "Apache 2.0".to_string(),
            badge: "🌍 Traduction locale • 1,6 Go".to_string(),
            languages: Vec::new(),
            auto_start_server: false,
        },
    ];

    // ── Local Whisper models (whisper.cpp: Metal / CUDA / Vulkan / ROCm or CPU) ──────
    models.extend([
        whisper("whisper_tiny", "Whisper Tiny", "Modèle ultra-rapide (75 Mo). Idéal pour une latence minimale.", WHISPER_REPO, WHISPER_REV, "ggml-tiny.bin", 77_691_713, "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21", "Local 75 Mo", &[]),
        whisper("whisper_base", "Whisper Base", "Bon compromis rapidité/précision (142 Mo).", WHISPER_REPO, WHISPER_REV, "ggml-base.bin", 147_951_465, "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe", "Local 142 Mo", &[]),
        whisper("whisper_small", "Whisper Small", "Haute précision (466 Mo), bon sur le vocabulaire technique.", WHISPER_REPO, WHISPER_REV, "ggml-small.bin", 487_601_967, "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b", "Local 466 Mo", &[]),
        whisper("whisper_small_q5", "Whisper Small Q5_1", "Version quantisée compacte de Small (181 Mo).", WHISPER_REPO, WHISPER_REV, "ggml-small-q5_1.bin", 190_085_487, "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb", "Local 181 Mo", &[]),
        whisper("whisper_medium", "Whisper Medium", "Très bonne compréhension du français et des accents (1,5 Go).", WHISPER_REPO, WHISPER_REV, "ggml-medium.bin", 1_533_763_059, "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208", "Studio 1.5 Go", &[]),
        whisper("whisper_medium_q5", "Whisper Medium Q5_0", "Medium quantisé (514 Mo).", WHISPER_REPO, WHISPER_REV, "ggml-medium-q5_0.bin", 539_212_467, "19fea4b380c3a618ec4723c3eef2eb785ffba0d0538cf43f8f235e7b3b34220f", "Local 514 Mo", &[]),
        whisper("whisper_turbo", "Whisper Large-v3-Turbo", "Large-v3 distillé Turbo (1,5 Go).", WHISPER_REPO, WHISPER_REV, "ggml-large-v3-turbo.bin", 1_624_555_275, "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69", "Turbo 1.5 Go", &[]),
        whisper("whisper_turbo_q5", "Whisper Large-v3-Turbo Q5_0", "Turbo quantisé (547 Mo), idéal pour 8 à 16 Go de RAM.", WHISPER_REPO, WHISPER_REV, "ggml-large-v3-turbo-q5_0.bin", 574_041_195, "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2", "⚡ Turbo 547 Mo", &[]),
        whisper("whisper_large_v3", "Whisper Large-v3", "Le modèle de référence OpenAI (2,9 Go).", WHISPER_REPO, WHISPER_REV, "ggml-large-v3.bin", 3_095_033_483, "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2", "Gold Standard 2.9 Go", &[]),
        whisper("whisper_large_v3_q5", "Whisper Large-v3 Q5_0", "Large-v3 quantisé (1 Go).", WHISPER_REPO, WHISPER_REV, "ggml-large-v3-q5_0.bin", 1_081_140_203, "d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1", "Flagship 1 Go", &[]),
        whisper("distil_whisper_large_v3", "Distil-Whisper Large-v3 [Anglais]", "Modèle distillé optimisé exclusivement pour l'anglais (1,4 Go).", DISTIL_REPO, DISTIL_REV, "ggml-distil-large-v3.bin", 1_519_521_155, "2883a11b90fb10ed592d826edeaee7d2929bf1ab985109fe9e1e7b4d2b69a298", "Anglais uniquement", &["en"]),
    ]);

    // ── Local OpenAI-compatible ASR servers ─────────────────────────────────────────
    // `bundled` models are served by scripts/asr_server.py (faster-whisper-server) and
    // started automatically; the others need the user's own local server (vLLM, Speaches...).
    #[allow(clippy::too_many_arguments)]
    let server = |id: &str, name: &str, provider: &str, description: &str, model: &str, size: u32, diar: bool, license: &str, badge: &str, bundled: bool| ModelMetadata {
        id: id.to_string(),
        name: name.to_string(),
        provider: provider.to_string(),
        description: description.to_string(),
        kind: KIND_STT.to_string(),
        default_engine: ENGINE_REMOTE.to_string(),
        default_model_name: model.to_string(),
        default_api_url: "http://127.0.0.1:8000/v1/audio/transcriptions".to_string(),
        download_url: None,
        files: Vec::new(),
        file_size_mb: size,
        supports_diarization: diar,
        is_local_gpu: false,
        license: license.to_string(),
        badge: badge.to_string(),
        languages: Vec::new(),
        auto_start_server: bundled,
    };
    const EXTERNAL: &str = " Nécessite votre propre serveur local (vLLM / Speaches) servant ce modèle sur 127.0.0.1:8000 : il n'est pas lancé automatiquement.";
    models.extend([
        server("faster_whisper_turbo", "faster-whisper Large-v3-Turbo (serveur Python local)", "SYSTRAN / faster-whisper-server", "Whisper Turbo via CTranslate2, lancé automatiquement par scripts/asr_server.py (Python requis).", "deepdml/faster-whisper-large-v3-turbo", 1600, false, "MIT", "Serveur local intégré", true),
        server("faster_whisper_base", "faster-whisper Base (serveur Python local)", "SYSTRAN / faster-whisper-server", "Whisper Base via CTranslate2, lancé automatiquement par scripts/asr_server.py (Python requis).", "Systran/faster-whisper-base", 145, false, "MIT", "Serveur local intégré", true),
        server("qwen3_asr", "Qwen3-ASR-1.7B", "Alibaba Cloud", &format!("Modèle ASR très précis.{EXTERNAL}"), "Qwen/Qwen3-ASR-1.7B", 3400, false, "Apache 2.0", "Serveur externe", false),
        server("voxtral_mini", "Voxtral Mini (3B)", "Mistral AI", &format!("Modèle audio optimisé pour la transcription à faible latence.{EXTERNAL}"), "mistralai/Voxtral-Mini-3B-2507", 9400, false, "Apache 2.0", "Serveur externe", false),
        server("voxtral_small", "Voxtral Small (24B)", "Mistral AI", &format!("Modèle audio haute capacité (GPU puissant requis).{EXTERNAL}"), "mistralai/Voxtral-Small-24B-2507", 48000, false, "Apache 2.0", "Serveur externe", false),
        server("canary_qwen", "Canary-Qwen-2.5B", "NVIDIA NeMo", &format!("Transcription anglaise de précision.{EXTERNAL}"), "nvidia/canary-qwen-2.5b", 5000, false, "CC-BY-4.0", "Serveur externe", false),
        server("vibevoice_asr", "VibeVoice-ASR (diarisation)", "Microsoft", &format!("Transcription avec séparation des locuteurs.{EXTERNAL}"), "microsoft/VibeVoice-ASR", 4000, true, "MIT", "Serveur externe", false),
        server("custom_api", "Serveur ASR personnalisé (local / réseau local)", "Speaches / vLLM / faster-whisper-server", "Tout serveur compatible /v1/audio/transcriptions sur cette machine ou votre réseau local.", "whisper-1", 0, false, "Ouvert", "Endpoint local", false),
    ]);
    models
}

pub fn find_model(model_id: &str) -> Option<ModelMetadata> {
    get_available_models().into_iter().find(|m| m.id == model_id)
}

/// Find a local Whisper model of the catalog from its file path.
pub fn find_model_by_path(model_path: &str) -> Option<ModelMetadata> {
    let target_name = std::path::Path::new(model_path).file_name();
    get_available_models().into_iter().find(|m| {
        m.is_local_gpu
            && m.default_engine == ENGINE_WHISPER
            && (m.default_model_name == model_path
                || (target_name.is_some()
                    && std::path::Path::new(&m.default_model_name).file_name() == target_name))
    })
}

/// Size sanity check for model files that are not in the catalog (custom Whisper paths).
pub fn is_model_file_valid(path: &std::path::Path, expected_size_mb: u32) -> bool {
    let meta = match std::fs::metadata(path) {
        Ok(m) if m.is_file() => m,
        _ => return false,
    };
    if expected_size_mb > 0 {
        let min_bytes = (expected_size_mb as u64) * 1024 * 1024 * 7 / 10;
        meta.len() >= min_bytes
    } else {
        meta.len() > 1024 * 1024
    }
}

fn is_file_complete(file: &ModelFile) -> bool {
    match std::fs::metadata(&file.path) {
        Ok(m) if m.is_file() => file.size_bytes == 0 || m.len() == file.size_bytes,
        _ => false,
    }
}

/// A local model is installed when every one of its files is present with its exact size
/// (SHA-256 is verified at download time).
pub fn is_model_installed(model: &ModelMetadata) -> bool {
    model.is_local_gpu && !model.files.is_empty() && model.files.iter().all(is_file_complete)
}

pub fn is_model_id_installed(model_id: &str) -> bool {
    find_model(model_id).is_some_and(|m| is_model_installed(&m))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_ids_are_unique_and_local_models_are_pinned() {
        let models = get_available_models();
        let mut ids = std::collections::HashSet::new();
        for m in &models {
            assert!(ids.insert(m.id.clone()), "duplicate id {}", m.id);
            if m.is_local_gpu {
                assert!(!m.files.is_empty(), "{} has no files", m.id);
                for f in &m.files {
                    assert!(f.url.starts_with("https://huggingface.co/"), "{}", f.url);
                    assert!(!f.url.contains("/resolve/main/"), "{} is not pinned", f.url);
                    assert!(f.path.starts_with("models/"), "{}", f.path);
                    if f.size_bytes > 10 * 1024 * 1024 {
                        assert!(f.sha256.as_ref().is_some_and(|h| h.len() == 64), "{} lacks sha256", f.path);
                    }
                }
            } else {
                assert!(crate::netpolicy::is_local_endpoint(&m.default_api_url), "{}", m.id);
            }
        }
        for id in [KYUTAI_STT_MODEL_ID, HIBIKI_1B_MODEL_ID, HIBIKI_2B_MODEL_ID, MADLAD_MODEL_ID] {
            assert!(ids.contains(id), "missing {id}");
        }
    }

    #[test]
    fn whisper_model_is_found_by_path() {
        let m = find_model_by_path("models/ggml-large-v3-turbo-q5_0.bin").unwrap();
        assert_eq!(m.id, "whisper_turbo_q5");
        assert!(find_model_by_path("models/kyutai-stt-1b-en_fr-candle").is_none());
    }
}
