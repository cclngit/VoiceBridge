fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=packaging/windows/voicebridge.rc");
    println!("cargo:rerun-if-changed=packaging/windows/VoiceBridge.ico");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "windows" {
        embed_resource::compile(
            "packaging/windows/voicebridge.rc",
            embed_resource::ParamsIncludeDirs(&["packaging/windows"]),
        )
        .manifest_optional()
        .unwrap();

        // CUDA build: Candle (cudarc) and whisper.cpp (ggml-cuda) link the CUDA *driver* API,
        // i.e. `nvcuda.dll`, which only an NVIDIA display driver installs. Delay-load it so the
        // single Windows installer still starts on AMD / Intel / driverless PCs; the code only
        // touches the driver API after `ml::nvidia_driver_present()` (or cudart) found a GPU.
        // The redistributable CUDA runtime DLLs (cudart, cuBLAS, cuRAND) ship next to the .exe
        // instead (see packaging/windows/collect_cuda_dlls.ps1).
        let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
        if target_env == "msvc" && std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
            println!("cargo:rustc-link-arg=/DELAYLOAD:nvcuda.dll");
            println!("cargo:rustc-link-lib=delayimp");
        }
    }
}

