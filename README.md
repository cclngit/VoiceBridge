<p align="center">
  <img src="docs/logo.png" alt="VoiceBridge Logo" width="160" style="border-radius: 28px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
</p>

# VoiceBridge — Speech To Text To Translate (Rust)

<p align="center">
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-2024_Edition-orange.svg?logo=rust" alt="Rust"></a>
  <a href="https://github.com/longbridge/gpui-kit"><img src="https://img.shields.io/badge/GUI-GPUI_Accelerated-6366f1.svg" alt="GUI"></a>
  <a href="https://developer.apple.com/metal/"><img src="https://img.shields.io/badge/macOS-Metal_GPU-success.svg?logo=apple" alt="macOS"></a>
  <img src="https://img.shields.io/badge/Windows-DirectX_%7C_CUDA_%7C_ARM64-0078D4.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iI2ZmZmZmZiI%2BPHBhdGggZD0iTTIgMmg5LjV2OS41SDJWMnptMTAuNSAwSDIydjkuNWgtOS41VjJ6TTIgMTIuNWg5LjVWMjJIMnYtOS41em0xMC41IDBIMjJWMjJoLTkuNXYtOS41eiIvPjwvc3ZnPg%3D%3D" alt="Windows">
  <img src="https://img.shields.io/badge/Linux-CUDA_%7C_Vulkan_%7C_ARM64-0ea5e9.svg?logo=linux&logoColor=white" alt="Linux">
  <a href="https://obsproject.com/"><img src="https://img.shields.io/badge/OBS-Browser_Source_Overlay-blue.svg" alt="OBS Studio"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

Ultra-fast, 100% offline real-time speech recognition and multilingual translation pipeline written in **Rust** for **macOS, Windows, and Linux** (`x86_64` & `ARM64`). Powered by a hardware-accelerated native **GPUI** desktop interface (Metal / DirectX / Vulkan), a multi-model STT/ASR engine (Whisper GPU/CPU, Qwen3-ASR, Voxtral, Canary, Moonshine, VibeVoice), a transparent **OBS Studio** subtitle overlay, zero-disk in-memory audio streaming, and automatic local ASR server lifecycle management.

---

## ⚡ Key Highlights & Architecture

| Feature | Legacy Version (Python) | VoiceBridge (Rust) |
| :--- | :--- | :--- |
| **OS Support** | Environment-dependent scripts | **Native Cross-Platform**: macOS (CoreAudio/Metal), Windows (WASAPI/DirectX), Linux (ALSA/Wayland/X11) |
| **Desktop Interface** | None or basic web page | **60–120 FPS native GPUI app**, Cyber Neon Dark design, instant responsiveness |
| **STT / ASR Models** | Whisper Tiny only | **Multi-Model**: Whisper (Metal / CUDA / Vulkan / ROCm / CPU), Qwen3-ASR, Voxtral, Canary, VibeVoice, Moonshine |
| **ASR Server Lifecycle** | Manual / Separate scripts | **100% automatic start & shutdown** based on selected model (`.venv`, `python3`, `python`, `py -3`) |
| **OBS Studio Overlay** | Opaque static web page | **Dedicated transparent OBS overlay (`/obs`)** with customizable themes (Dual, Glow, Netflix, Boxed) |
| **VAD Customization** | Fixed hardcoded parameters | **Live-tunable RMS sensitivity, end-of-speech silence, and segment durations** |
| **Disk I/O** | Continuous WAV file reads/writes | **Zero disk writes**: 100% in-memory streaming (*in-memory ring buffers*) |
| **Hardware Acceleration** | PyTorch / CTranslate2 CPU | **Metal GPU (Apple Silicon)** by default on macOS, optional **CUDA / Vulkan / ROCm** + multithreaded CPU on Windows & Linux |
| **Footprint & Deployment** | 8–12 GB Python environment | **Single self-contained binary** with HTML/CSS/PNG assets embedded at compile time |

---

## 📁 Repository Structure

```text
VoiceBridge/
├── Cargo.toml                  # Cargo manifest & dependencies (gpui-kit, whisper-rs, axum, cpal, tokio)
├── Cargo.lock                  # Reproducible dependency lockfile
├── config.json                 # Active application configuration (auto-saved)
├── config.example.json         # Default configuration template
├── LICENSE                     # MIT License
├── README.md                   # Main project documentation
│
├── .github/workflows/
│   └── ci.yml                  # Multi-OS & multi-arch CI/CD (.dmg macOS, .deb Linux, .exe Windows)
├── build.rs                    # Embeds Windows application icon (.ico) into VoiceBridge.exe
├── packaging/                  # Native installer scripts & manifests
│   ├── macos/                  # Info.plist and build_dmg.sh (VoiceBridge.app + .dmg)
│   ├── linux/                  # voicebridge.desktop and build_deb.sh (.deb package via dpkg-deb)
│   └── windows/                # VoiceBridge.ico, voicebridge.rc, and installer.iss (Inno Setup .exe)
├── docs/                       # Architecture & design specifications
│   ├── DESIGN.md               # Design system (Cyber Neon Dark palette, typography, tokens)
│   └── screen.png              # Application screenshot
│
├── models/                     # Local Whisper GGML model binaries (.bin)
│   ├── .gitkeep                # Keeps directory tracked in Git
│   └── README.md               # Whisper model guide & manual download commands
│
├── scripts/                    # Utility scripts for the local ASR server
│   ├── asr_server.py           # OpenAI-compatible ASR server (PyAV shim, automatic model mapping)
│   ├── start_local_asr.sh      # Quick-start Bash script for macOS / Linux (port 8000)
│   ├── start_local_asr.bat     # Quick-start Batch script for Windows (port 8000)
│   └── README.md               # Scripts usage guide
│
└── src/                        # Modular Rust source code
    ├── main.rs                 # Entry point, Tokio runtime orchestration, and GPUI launch
    ├── config.rs               # Configuration schema & JSON persistence
    ├── audio/                  # CPAL audio module
    │   ├── capture.rs          # Microphone stream capture & in-memory buffer management
    │   ├── resampler.rs        # Real-time 16 kHz mono resampling
    │   └── mod.rs              # Device manager exports
    ├── vad/                    # Voice Activity Detection (VAD)
    │   ├── energy_vad.rs       # Adaptive RMS energy segmentation with pre-speech ring buffer
    │   └── mod.rs              # SpeechSegmenter export
    ├── stt/                    # Speech-To-Text engine
    │   ├── engine.rs           # Local Whisper GPU/CPU inference & Remote ASR client
    │   ├── models.rs           # Full catalog of local Whisper & server ASR models
    │   ├── downloader.rs       # Async streaming model downloader with live progress tracking
    │   └── mod.rs              # STT types & engine exports
    ├── translate/              # Machine Translation engine
    │   ├── engine.rs           # Smart translation (OpenAI / Ollama / LibreTranslate / Fallback)
    │   └── mod.rs              # TranslationService export
    ├── server/                 # Embedded HTTP & SSE server (Axum)
    │   ├── routes.rs           # REST endpoints, live stream SSE, and HTML handlers
    │   ├── state.rs            # Thread-safe shared state across GUI, audio, and network
    │   └── mod.rs              # Router export
    ├── web/                    # Embedded web interfaces (compiled into the binary)
    │   ├── index.html          # Web control & monitoring dashboard
    │   └── obs.html            # Transparent Browser Source overlay for OBS Studio
    └── gui/                    # Native GPUI desktop interface
        ├── app.rs              # Views (Live, Models, Languages, Audio/VAD, OBS), VU meter & event loop
        └── mod.rs              # run_gui entry point
```

---

## 🎙️ Speech Models & Automatic ASR Server Management

VoiceBridge supports two model categories with seamless switching:

### 1. Local Whisper Models (GPU / CPU Acceleration)

- **Whisper Tiny, Base, Small, Medium, Distil-Large-v3, Large-v3-Turbo, Large-v3**: Run 100% offline on your GPU (Metal, CUDA, Vulkan, ROCm) or multithreaded CPU via `whisper.cpp`.
- Downloadable in one click from the **AI Models** tab. Once downloaded, the model is automatically hot-loaded into memory and activated.
- **Resource efficiency**: Whenever a local Whisper model is active, the background Python ASR server is **automatically shut down** to free RAM and CPU/GPU resources.

### 2. Local OpenAI-Compatible ASR Server Models

- **Qwen3-ASR-1.7B, Voxtral, Canary, VibeVoice, Moonshine**:
- **Automatic launch**: Selecting a server-backed model automatically spawns `scripts/asr_server.py` on port `8000` and routes audio segments to it.

---

## 🎛️ Audio & VAD (Voice Activity Detection) Settings

The **Audio & VAD** tab lets you fine-tune speech segmentation on the fly:

- **VAD Sensitivity (RMS Threshold)**: `0.8%` (Whispers), `1.5%` (Recommended standard), `2.5%` (Medium), `4.0%` (Loud mic or noisy environment).
- **End-of-Sentence Silence**: `400 ms` (Fast / Streaming), `700 ms` (Natural), `1000 ms` (Smooth), `1500 ms` (Long sentences).
- **Minimum Speech Duration**: `200 ms`, `350 ms`, `500 ms` (filters out clicks and brief noise).
- **Maximum Continuous Duration**: `8 s`, `15 s`, `25 s` (forces a split during uninterrupted speech).
- **Hot Reload**: Parameters propagate to the live audio segmenter within 500 ms without interrupting capture.

---

## 📺 OBS Studio Integration (Streaming & Podcasts)

To display live translated subtitles in OBS Studio:

1. In **OBS Studio**, add a **`Browser`** source (*Browser Source*).
2. Set the URL to: **`http://localhost:3000/obs`**
3. Recommended dimensions: **Width `1920`, Height `250`**.
4. The background is **100% transparent** and composites cleanly without chroma keying.

### Customizable URL Parameters

- `mode=dual`: Small original transcription on top + large translation below (default).
- `mode=translated`: Translation only.
- `mode=source`: Original transcription only.
- `theme=glow` (Cyber Neon), `theme=netflix` (Cinematic), `theme=boxed` (Studio crisp box).
- `size=34`: Font size in pixels.
- `fade=6`: Auto-fade after 6 seconds of silence (`0` to keep visible).

Example:

```text
http://localhost:3000/obs?mode=dual&theme=glow&size=34&fade=6
```

---

## 🚀 Building & Running (macOS, Windows, Linux)

### Prerequisites by Operating System

- **Common**:
  - **Rust** (stable, 2024 edition): [rustup.rs](https://rustup.rs/)
  - **CMake** & **Clang / LLVM** (required to compile `whisper.cpp` via `whisper-rs`)
  - **Python 3** *(optional)*: only required for models backed by the local ASR server (`scripts/asr_server.py`)
- **macOS (Apple Silicon ARM64 & Intel / Universal2)**:
  - Xcode Command Line Tools: `xcode-select --install` *(Metal GPU enabled automatically)*
- **Windows 10 / 11 (x86_64 & ARM64 Copilot+ / Snapdragon X)**:
  - *Visual Studio Build Tools* (*Desktop development with C++* workload + *C++ Clang Compiler for Windows* on ARM64 + Windows SDK)
  - *CMake* and *LLVM* (`winget install Kitware.CMake LLVM.LLVM`)
- **Linux (Ubuntu / Debian — x86_64 & ARM64 / aarch64)**:
  ```bash
  sudo apt-get update && sudo apt-get install -y \
    build-essential cmake pkg-config clang libclang-dev libssl-dev \
    libasound2-dev libwayland-dev libxkbcommon-dev libxkbcommon-x11-dev \
    libxcb1-dev libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev \
    libxcb-shm0-dev libxcb-randr0-dev libxcb-xkb-dev libxcb-xinput-dev \
    libx11-dev libx11-xcb-dev libxrandr-dev libxi-dev libxcursor-dev \
    libfontconfig1-dev libfreetype-dev libdbus-1-dev libvulkan-dev
  ```

### Build Commands

```bash
# macOS (Metal GPU enabled by default) or Windows / Linux (multithreaded CPU by default):
cargo run --release

# Windows / Linux with NVIDIA CUDA GPU acceleration (requires CUDA Toolkit):
cargo run --release --features cuda

# Windows / Linux with universal Vulkan GPU acceleration (NVIDIA / AMD / Intel):
cargo run --release --features vulkan

# Linux with AMD ROCm / hipBLAS GPU acceleration:
cargo run --release --features hipblas

# Headless / server-only mode (without desktop GUI, e.g. remote server / Docker):
cargo run --release -- --headless
```

### Running the Compiled Binary Directly

```bash
# macOS / Linux:
./target/release/VoiceBridge

# Windows (PowerShell / CMD):
.\target\release\VoiceBridge.exe
```

> **CI/CD & Native Multi-Arch Installers (`.dmg`, `.deb`, `.exe`)**: The GitHub Actions workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) automatically builds, tests, and packages native installers for every target architecture on every push and `v*` tag:
> - **macOS ARM64 (`.dmg`)**: `VoiceBridge-macos-arm64.dmg` (ad-hoc signed `VoiceBridge.app` + `.icns` icon + `Applications` symlink)
> - **macOS Universal2 (`.dmg`)**: `VoiceBridge-macos-universal.dmg` (universal `arm64` + `x86_64` binary via `lipo`)
> - **Linux x86_64 (`.deb`)**: `VoiceBridge-linux-amd64.deb` (Debian/Ubuntu package built via `dpkg-deb` with `.desktop` launcher)
> - **Linux ARM64 (`.deb`)**: `VoiceBridge-linux-arm64.deb` (Debian/Ubuntu `arm64` package built via `dpkg-deb`)
> - **Windows x86_64 (`.exe`)**: `VoiceBridge-windows-x86_64-setup.exe` (Inno Setup graphical installer) + `VoiceBridge-windows-x86_64.exe` (standalone portable executable)
> - **Windows ARM64 (`.exe`)**: `VoiceBridge-windows-arm64-setup.exe` (native Windows on ARM / Copilot+ / Snapdragon X installer) + `VoiceBridge-windows-arm64.exe` (standalone portable executable)

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
