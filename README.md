<p align="center">
  <img src="docs/logo.png" alt="VoiceBridge Logo" width="160" style="border-radius: 28px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
</p>

# VoiceBridge — Speech To Text To Translate (Rust)

<p align="center">
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-2024_Edition-orange.svg?logo=rust" alt="Rust"></a>
  <a href="https://github.com/longbridge/gpui-kit"><img src="https://img.shields.io/badge/GUI-GPUI_Accelerated-6366f1.svg" alt="GUI"></a>
  <a href="https://developer.apple.com/metal/"><img src="https://img.shields.io/badge/macOS-Metal_GPU-success.svg?logo=apple" alt="Metal GPU"></a>
  <img src="https://img.shields.io/badge/Windows_%26_Linux-CUDA_%7C_Vulkan_%7C_CPU-0ea5e9.svg" alt="Windows & Linux">
  <a href="https://obsproject.com/"><img src="https://img.shields.io/badge/OBS-Browser_Source_Overlay-blue.svg" alt="OBS Studio"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

Pipeline de reconnaissance vocale et traduction multilingue en temps réel, ultra-rapide et 100% hors-ligne, écrit en **Rust** et compatible **macOS, Windows et Linux**. Doté d'une interface graphique native propulsée par **GPUI** (accélération matérielle Metal / DirectX / Vulkan), d'un moteur multi-modèles (Whisper GPU/CPU, Qwen3-ASR, Voxtral, Canary, Moonshine, VibeVoice), d'une incrustation transparente pour **OBS Studio**, d'un streaming audio zéro-disque en mémoire et d'un cycle de vie automatique du serveur ASR.

---

## ⚡ Points Forts & Architecture

| Fonctionnalité | Ancienne version (Python) | VoiceBridge (Rust) |
| :--- | :--- | :--- |
| **Compatibilité OS** | Scripts dépendants de l'environnement | **Multi-plateforme natif** : macOS (CoreAudio/Metal), Windows (WASAPI/DirectX), Linux (ALSA/Wayland/X11) |
| **Interface Graphique** | Aucune ou page web basique | **GUI native GPUI 60–120 FPS**, design Cyber Neon Dark, réactivité instantanée |
| **Modèles STT / ASR** | Whisper Tiny uniquement | **Multi-Modèles** : Whisper (Metal / CUDA / Vulkan / ROCm / CPU), Qwen3-ASR, Voxtral, Canary, VibeVoice, Moonshine |
| **Cycle de vie Serveur ASR** | Manuel / Scripts séparés | **Démarrage & arrêt 100% automatique** selon le modèle sélectionné (`.venv`, `python3`, `python`, `py -3`) |
| **Incrustation OBS Studio** | Page web statique opaque | **Overlay OBS transparent dédié (`/obs`)** avec thèmes (Dual, Glow, Netflix, Studio) |
| **Personnalisation VAD** | Réglages figés | **Sensibilité RMS, silence de fin de phrase et durées de segment** modifiables en direct |
| **Entrées/Sorties Disque** | Écritures/lectures continues de WAV | **Zéro écriture disque** : streaming 100% en mémoire (*in-memory ring buffers*) |
| **Accélération Matérielle** | PyTorch / CTranslate2 CPU | **Metal GPU (Apple Silicon)** par défaut sur macOS, features **CUDA / Vulkan / ROCm** + CPU multithread sur Windows & Linux |
| **Autonomie & Poids** | 8 à 12 Go d'environnement Python | **Binaire unique autonome** avec assets HTML/CSS/PNG intégrés à la compilation |

---

## 📁 Organisation du Répertoire

```text
VoiceBridge/
├── Cargo.toml                  # Configuration Cargo et dépendances (gpui-kit, whisper-rs, axum, cpal, tokio)
├── Cargo.lock                  # Verrouillage reproductible des versions de dépendances
├── config.json                 # Configuration active de l'application (sauvegarde automatique)
├── config.example.json         # Modèle de configuration par défaut
├── LICENSE                     # Licence MIT
├── README.md                   # Documentation principale du projet
│
├── .github/workflows/
│   └── ci.yml                  # Build & packaging multi-OS (.dmg macOS, .deb Linux, .exe Windows)
├── build.rs                    # Intégration de l'icône Windows (.ico) dans VoiceBridge.exe
├── packaging/                  # Scripts et manifestes d'installateurs natifs
│   ├── macos/                  # Info.plist et build_dmg.sh (VoiceBridge.app + .dmg)
│   ├── linux/                  # voicebridge.desktop et build_deb.sh (paquet .deb via dpkg-deb)
│   └── windows/                # VoiceBridge.ico, voicebridge.rc et installer.iss (Inno Setup .exe)
├── docs/                       # Spécifications et documentation d'architecture
│   ├── DESIGN.md               # Système de design (palette Cyber Neon Dark, typographie, tokens)
│   └── screen.png              # Capture d'écran de l'interface
│
├── models/                     # Modèles Whisper GGML binaires (.bin)
│   ├── .gitkeep                # Maintien du dossier dans Git
│   └── README.md               # Guide et commandes de téléchargement des modèles Whisper
│
├── scripts/                    # Scripts utilitaires pour le serveur ASR local
│   ├── asr_server.py           # Serveur ASR OpenAI-compatible (shim PyAV, auto-mappage des modèles)
│   ├── start_local_asr.sh      # Script bash de démarrage rapide macOS / Linux (port 8000)
│   ├── start_local_asr.bat     # Script batch de démarrage rapide Windows (port 8000)
│   └── README.md               # Guide d'utilisation des scripts
│
└── src/                        # Code source Rust modulaire
    ├── main.rs                 # Point d'entrée de l'application, orchestration Tokio et lancement GPUI
    ├── config.rs               # Schéma et gestion de la persistance de configuration (JSON)
    ├── audio/                  # Module audio CPAL
    │   ├── capture.rs          # Capture du flux micro et gestion du buffer en mémoire
    │   ├── resampler.rs        # Rééchantillonnage 16 kHz mono temps réel
    │   └── mod.rs              # Export des gestionnaires de périphériques
    ├── vad/                    # Détection d'Activité Vocale (VAD)
    │   ├── energy_vad.rs       # Segmentation par énergie RMS adaptative avec buffer pré-parole
    │   └── mod.rs              # Export du SpeechSegmenter
    ├── stt/                    # Moteur de Reconnaissance Vocale (Speech-To-Text)
    │   ├── engine.rs           # Inférence Whisper Metal GPU / CPU et client Remote ASR
    │   ├── models.rs           # Catalogue complet des 11 modèles vocaux
    │   ├── downloader.rs       # Téléchargement asynchrone avec suivi en temps réel
    │   └── mod.rs              # Export des types et moteurs ASR
    ├── translate/              # Moteur de Traduction Automatique
    │   ├── engine.rs           # Traduction intelligente (OpenAI / Ollama / LibreTranslate / Fallback)
    │   └── mod.rs              # Export du TranslationEngine
    ├── server/                 # Serveur HTTP & SSE embarqué (Axum)
    │   ├── routes.rs           # Endpoints REST, SSE pour le live stream et handlers HTML
    │   ├── state.rs            # État partagé thread-safe entre GUI, audio et réseau
    │   └── mod.rs              # Export du routeur
    ├── web/                    # Interfaces web embarquées (compilées dans le binaire)
    │   ├── index.html          # Dashboard web de contrôle et monitoring
    │   └── obs.html            # Overlay transparent pour source navigateur OBS Studio
    └── gui/                    # Interface graphique native GPUI
        ├── app.rs              # Vues (Direct, Modèles, Langues, Audio/VAD, OBS), VU-mètre et boucle d'événements
        └── mod.rs              # Point d'entrée run_gui
```

---

## 🎙️ Modèles Vocaux & Gestion Automatique du Serveur ASR

VoiceBridge prend en charge deux catégories de modèles, avec bascule intelligente :

### 1. Modèles Whisper Locaux (Accélération Metal GPU / CPU)

- **Whisper Tiny, Base, Small, Turbo, Large-v3** : Exécution 100% hors-ligne sur votre processeur ou puce graphique Apple Silicon via `whisper.cpp`.
- Téléchargeables en un clic depuis l'onglet **Modèles IA**. Dès la fin du téléchargement, le modèle est automatiquement chargé en mémoire et activé.
- **Économie de ressources** : Lorsque vous utilisez un modèle Whisper local, le serveur ASR Python est **automatiquement éteint** pour libérer la RAM et le processeur.

### 2. Modèles Serveur ASR Local (OpenAI-compatible)

- **Qwen3-ASR-1.7B, Voxtral, Canary, VibeVoice, Moonshine** :
- **Lancement automatique** : Dès qu'un modèle serveur est sélectionné, VoiceBridge lance automatiquement le serveur local `scripts/asr_server.py` sur le port `8000`.

---

## 🎛️ Réglages Audio & VAD (Détection Vocale)

L'onglet **Audio & VAD** permet d'ajuster le comportement du découpage vocal à la volée :

- **Sensibilité VAD (Seuil RMS)** : 0.8% (Chuchotements), 1.5% (Standard recommandé), 2.5% (Moyen), 4.0% (Micro fort ou environnement bruyant).
- **Silence de fin de phrase** : 400 ms (Rapide / Streaming), 700 ms (Naturel), 1000 ms (Fluide), 1500 ms (Phrases longues).
- **Durée minimale de parole** : 200 ms, 350 ms, 500 ms (élimine les clics et bruits de bouche).
- **Durée maximale continue** : 8 s, 15 s, 25 s (force le découpage en cas de parole ininterrompue).
- **Mise à jour à chaud** : Les réglages sont transmis au découpeur audio sous 500 ms sans interruption du flux.

---

## 📺 Intégration OBS Studio (Streaming & Podcasts)

Pour afficher vos sous-titres traduits en direct dans OBS Studio :

1. Dans **OBS Studio**, ajoutez une source **`Source Navigateur`** (*Browser Source*).
2. Entrez l'URL : **`http://localhost:3000/obs`**
3. Configurez les dimensions recommandées : **Largeur 1920, Hauteur 250**.
4. Le fond est **100% transparent** et s'incruste sans nécessiter de fond vert.

### Paramètres URL personnalisables

- `mode=dual` : Texte original en petit au-dessus + traduction en grand au-dessous (défaut).
- `mode=translated` : Traduction seule.
- `mode=source` : Transcription originale seule.
- `theme=glow` (Neon Cyber), `theme=netflix` (Cinématique), `theme=boxed` (Studio avec contour net).
- `size=34` : Taille de police en pixels.
- `fade=6` : Disparition automatique après 6 secondes de silence (`0` pour laisser affiché).

Exemple :

```text
http://localhost:3000/obs?mode=dual&theme=glow&size=34&fade=6
```

---

## 🚀 Compilation & Lancement (macOS, Windows, Linux)

### Prérequis par système d'exploitation

- **Commun** :
  - **Rust** (stable, édition 2024) : [rustup.rs](https://rustup.rs/)
  - **CMake** & **Clang / LLVM** (requis pour compiler `whisper.cpp` via `whisper-rs`)
  - **Python 3** *(optionnel)* : uniquement pour les modèles basés sur le serveur ASR local (`scripts/asr_server.py`)
- **macOS (Apple Silicon ARM64 & Intel / Universal2)** :
  - Xcode Command Line Tools : `xcode-select --install` *(Metal GPU activé automatiquement)*
- **Windows 10 / 11 (x86_64 & ARM64 Copilot+ / Snapdragon X)** :
  - *Visual Studio Build Tools* (charge de travail *Développement Desktop en C++* + *Compilateur C++ Clang pour Windows* sur ARM64 + Windows SDK)
  - *CMake* et *LLVM* (`winget install Kitware.CMake LLVM.LLVM`)
- **Linux (Ubuntu / Debian — x86_64 & ARM64 / aarch64)** :
  ```bash
  sudo apt-get update && sudo apt-get install -y \
    build-essential cmake pkg-config clang libclang-dev libssl-dev \
    libasound2-dev libwayland-dev libxkbcommon-dev libxkbcommon-x11-dev \
    libxcb1-dev libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev \
    libxcb-shm0-dev libxcb-randr0-dev libxcb-xkb-dev libxcb-xinput-dev \
    libx11-dev libx11-xcb-dev libxrandr-dev libxi-dev libxcursor-dev \
    libfontconfig1-dev libfreetype-dev libdbus-1-dev libvulkan-dev
  ```

### Commandes de compilation

```bash
# macOS (Metal GPU activé par défaut) ou Windows / Linux (CPU multithread par défaut) :
cargo run --release

# Windows / Linux avec accélération GPU NVIDIA CUDA (nécessite CUDA Toolkit) :
cargo run --release --features cuda

# Windows / Linux avec accélération GPU universelle Vulkan (NVIDIA / AMD / Intel) :
cargo run --release --features vulkan

# Linux avec accélération GPU AMD ROCm / hipBLAS :
cargo run --release --features hipblas

# Mode headless / serveur uniquement (sans interface graphique de bureau, ex. VPS / Docker) :
cargo run --release -- --headless
```

### Exécution directe du binaire compilé

```bash
# macOS / Linux :
./target/release/VoiceBridge

# Windows (PowerShell / CMD) :
.\target\release\VoiceBridge.exe
```

> **CI / Installateurs natifs multi-OS & multi-architectures (`.dmg`, `.deb`, `.exe`)** : Le workflow GitHub Actions [`.github/workflows/ci.yml`](.github/workflows/ci.yml) compile, teste et génère automatiquement les installateurs natifs pour chaque architecture à chaque push ou tag `v*` :
> - **macOS ARM64 (`.dmg`)** : `VoiceBridge-macos-arm64.dmg` (`VoiceBridge.app` signé ad-hoc + icône `.icns` + raccourci `Applications`)
> - **macOS Universal2 (`.dmg`)** : `VoiceBridge-macos-universal.dmg` (binaire universel `arm64` + `x86_64` via `lipo`)
> - **Linux x86_64 (`.deb`)** : `VoiceBridge-linux-amd64.deb` (paquet Debian/Ubuntu généré via `dpkg-deb` avec lanceur `.desktop`)
> - **Linux ARM64 (`.deb`)** : `VoiceBridge-linux-arm64.deb` (paquet Debian/Ubuntu `arm64` généré via `dpkg-deb`)
> - **Windows x86_64 (`.exe`)** : `VoiceBridge-windows-x86_64-setup.exe` (installateur graphique Inno Setup) + `VoiceBridge-windows-x86_64.exe` (exécutable portable autonome)
> - **Windows ARM64 (`.exe`)** : `VoiceBridge-windows-arm64-setup.exe` (installateur natif Windows on ARM / Copilot+ / Snapdragon X) + `VoiceBridge-windows-arm64.exe` (exécutable portable autonome)

---

## 📜 Licence

Ce projet est distribué sous licence MIT — voir le fichier [LICENSE](LICENSE) pour plus de détails.
