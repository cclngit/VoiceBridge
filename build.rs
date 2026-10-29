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
    }
}

