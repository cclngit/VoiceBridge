## 🚀 Public Release — VoiceBridge v0.2.0

**VoiceBridge** is a high-performance, 100% offline real-time Speech-To-Text and multilingual translation pipeline rewritten in **Rust** (2024 Edition) with a native hardware-accelerated **GPUI** desktop interface (*Cyber Neon Dark* theme) and a transparent **OBS Studio** Browser Source overlay.

### ✨ Highlights

- **🖥️ Native Cross-Platform & Multi-Architecture (`x86_64` & `ARM64`)**
  - **macOS** (`ARM64` Apple Silicon & `Universal2` Apple Silicon + Intel): CoreAudio capture + native **Metal GPU** acceleration + `.dmg` Disk Image (`VoiceBridge.app`).
  - **Windows 10 / 11** (`x86_64` & `ARM64` Copilot+ / Snapdragon X): WASAPI audio capture + DirectX GPUI + built-in **Vulkan GPU** acceleration (NVIDIA GeForce RTX / AMD Radeon / Intel Arc — zero external CUDA toolkit required) & optional **CUDA** + Inno Setup `.exe` installer & standalone portable `.exe`.
  - **Linux** (`amd64` & `arm64`): ALSA audio capture + Wayland/X11 GPUI + built-in **Vulkan GPU** (`amd64`) & optional **CUDA / ROCm (`hipBLAS`)** GPU acceleration + Debian/Ubuntu `.deb` package.

- **🔥 Configurable Hardware Acceleration (GPU NVIDIA / Vulkan / CUDA / Metal / CPU)**
  - **Live GPU Enumeration**: Automatically detects physical GPUs via `ggml-backend`, displaying the exact GPU model name, active backend (`Vulkan`, `CUDA`, `Metal`, `ROCm`), and free/total VRAM in both the native GPUI desktop interface and Web UI.
  - **Hot-Swappable Execution Mode**: Switch on the fly between `Auto (GPU Priority)`, `Vulkan GPU`, `NVIDIA CUDA GPU`, `Apple Metal GPU`, or `CPU Only (Multithread)` without restarting the application.
  - **Multi-GPU & Performance Tuning**: Select a specific GPU index (`--gpu-device`), toggle **Flash Attention** on GPU (`--flash-attn`), or adjust **CPU Threads** (`2`, `4`, `8`, `12`, `16`) via the native GUI (`Modèles` tab), Web UI settings, `config.json`, or CLI flags (`--gpu-backend`, `--no-gpu`, `--threads`).

- **🎙️ Multi-Model STT / ASR Engine & 1-Click Model Downloader**
  - **Local Whisper (`whisper.cpp` / `whisper-rs`)**: *Tiny, Base, Small, Medium, Distil-Large-v3, Large-v3-Turbo, Large-v3* with built-in async downloader, integrity verification, and hot-reloading.
  - **Automatic Local ASR Server Lifecycle**: Seamless automatic startup and shutdown of the local OpenAI-compatible ASR server (`scripts/asr_server.py`) for *Qwen3-ASR-1.7B, Voxtral, Canary, VibeVoice,* and *Moonshine*.

- **⚡ Zero-Disk Streaming & Live-Tunable VAD**
  - 100% in-memory audio ring buffers and real-time 16 kHz mono resampling (zero temporary WAV writes on disk).
  - Live-adjustable Voice Activity Detection (RMS sensitivity threshold, end-of-sentence silence, min/max speech segment duration) with < 500 ms hot reload.

- **🌍 Smart Multilingual Translation & 📺 OBS Studio Overlay**
  - Pluggable translation backends (*OpenAI-compatible APIs, local Ollama LLMs, LibreTranslate,* and resilient offline fallback).
  - Embedded HTTP + Server-Sent Events (`Axum`) server with web dashboard (`http://localhost:3000`) and 100% transparent **OBS Studio** subtitle overlay (`http://localhost:3000/obs`) featuring *Dual, Translated, Source* modes and *Glow, Netflix, Boxed* themes.
  - Headless server mode (`--headless`) and automatic port fallback (`3000–3010`).

---

### 📦 Installation Packages

| Platform | Architecture | Package | Description |
| :--- | :--- | :--- | :--- |
| **macOS** | Apple Silicon (`arm64`) | `VoiceBridge-macos-arm64.dmg` | Signed `.dmg` with `VoiceBridge.app` & `/Applications` drag-and-drop (Metal GPU) |
| **macOS** | Universal (`arm64` + `x86_64`) | `VoiceBridge-macos-universal.dmg` | Universal2 `.dmg` for both Apple Silicon & Intel Macs (Metal GPU) |
| **Windows** | 64-bit (`x86_64`) | `VoiceBridge-windows-x86_64-setup.exe` | Graphical Windows Setup installer with NVIDIA/AMD/Intel Vulkan GPU acceleration |
| **Windows** | 64-bit (`x86_64`) | `VoiceBridge-windows-x86_64.exe` | Single-file portable executable with NVIDIA/AMD/Intel Vulkan GPU acceleration |
| **Windows** | Windows on ARM (`arm64`) | `VoiceBridge-windows-arm64-setup.exe` | Native ARM64 Setup installer (Snapdragon X / Copilot+ PCs) |
| **Windows** | Windows on ARM (`arm64`) | `VoiceBridge-windows-arm64.exe` | Native ARM64 single-file portable executable |
| **Linux** | 64-bit (`amd64` / `x86_64`) | `VoiceBridge-linux-amd64.deb` | Debian / Ubuntu package (`sudo dpkg -i VoiceBridge-linux-amd64.deb`, Vulkan GPU) |
| **Linux** | ARM64 (`arm64` / `aarch64`) | `VoiceBridge-linux-arm64.deb` | Debian / Ubuntu ARM64 package (`sudo dpkg -i VoiceBridge-linux-arm64.deb`) |
