## 🚀 Initial Public Release — VoiceBridge

**VoiceBridge** is a high-performance, 100% offline real-time Speech-To-Text and multilingual translation pipeline rewritten in **Rust** (2024 Edition) with a native hardware-accelerated **GPUI** desktop interface (*Cyber Neon Dark* theme) and a transparent **OBS Studio** Browser Source overlay.

### ✨ Highlights

- **🖥️ Native Cross-Platform & Multi-Architecture (`x86_64` & `ARM64`)**
  - **macOS** (`ARM64` Apple Silicon & `Universal2` Apple Silicon + Intel): CoreAudio capture + native **Metal GPU** acceleration + `.dmg` Disk Image (`VoiceBridge.app`).
  - **Windows 10 / 11** (`x86_64` & `ARM64` Copilot+ / Snapdragon X): WASAPI audio capture + DirectX GPUI + optional **CUDA / Vulkan** GPU acceleration + Inno Setup `.exe` installer & standalone portable `.exe`.
  - **Linux** (`amd64` & `arm64`): ALSA audio capture + Wayland/X11 GPUI + optional **CUDA / Vulkan / ROCm (`hipBLAS`)** GPU acceleration + Debian/Ubuntu `.deb` package.

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
| **macOS** | Apple Silicon (`arm64`) | `VoiceBridge-macos-arm64.dmg` | Signed `.dmg` with `VoiceBridge.app` & `/Applications` drag-and-drop |
| **macOS** | Universal (`arm64` + `x86_64`) | `VoiceBridge-macos-universal.dmg` | Universal2 `.dmg` for both Apple Silicon & Intel Macs |
| **Windows** | 64-bit (`x86_64`) | `VoiceBridge-windows-x86_64-setup.exe` | Graphical Windows Setup installer (Start Menu & Desktop shortcuts) |
| **Windows** | 64-bit (`x86_64`) | `VoiceBridge-windows-x86_64.exe` | Single-file portable executable (no installation required) |
| **Windows** | Windows on ARM (`arm64`) | `VoiceBridge-windows-arm64-setup.exe` | Native ARM64 Setup installer (Snapdragon X / Copilot+ PCs) |
| **Windows** | Windows on ARM (`arm64`) | `VoiceBridge-windows-arm64.exe` | Native ARM64 single-file portable executable |
| **Linux** | 64-bit (`amd64` / `x86_64`) | `VoiceBridge-linux-amd64.deb` | Debian / Ubuntu package (`sudo dpkg -i VoiceBridge-linux-amd64.deb`) |
| **Linux** | ARM64 (`arm64` / `aarch64`) | `VoiceBridge-linux-arm64.deb` | Debian / Ubuntu ARM64 package (`sudo dpkg -i VoiceBridge-linux-arm64.deb`) |

