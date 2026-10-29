# VoiceBridge Pro — Speech To Text To Translate (Rust)

[![Rust](https://img.shields.io/badge/Rust-2024_Edition-orange.svg?logo=rust)](https://www.rust-lang.org/)
[![GUI](https://img.shields.io/badge/GUI-GPUI_Accelerated-6366f1.svg)](https://github.com/longbridge/gpui-kit)
[![Metal GPU](https://img.shields.io/badge/macOS-Metal_GPU-success.svg?logo=apple)](https://developer.apple.com/metal/)
[![OBS Studio](https://img.shields.io/badge/OBS-Browser_Source_Overlay-blue.svg)](https://obsproject.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Pipeline de reconnaissance vocale et traduction multilingue en temps réel, ultra-rapide et 100% hors-ligne, écrit en **Rust**. Doté d'une interface graphique native propulsée par **GPUI** (accélération matérielle Metal / Vulkan / DirectX), d'un moteur multi-modèles (Whisper Metal GPU, Qwen3-ASR, Voxtral, Canary, Moonshine, VibeVoice), d'une incrustation transparente pour **OBS Studio**, d'un streaming audio zéro-disque en mémoire et d'un cycle de vie automatique du serveur ASR.

---

## ⚡ Points Forts & Architecture

| Fonctionnalité | Ancienne version (Python) | VoiceBridge Pro (Rust) |
| :--- | :--- | :--- |
| **Interface Graphique** | Aucune ou page web basique | **GUI native GPUI 60 FPS**, design Cyber Neon Dark, réactivité instantanée |
| **Modèles STT / ASR** | Whisper Tiny uniquement | **Multi-Modèles** : Whisper (Metal GPU Apple Silicon), Qwen3-ASR, Voxtral, Canary, VibeVoice, Moonshine |
| **Cycle de vie Serveur ASR** | Manuel / Scripts séparés | **Démarrage & arrêt 100% automatique** selon le modèle sélectionné |
| **Incrustation OBS Studio** | Page web statique opaque | **Overlay OBS transparent dédié (`/obs`)** avec thèmes (Dual, Glow, Netflix, Studio) |
| **Personnalisation VAD** | Réglages figés | **Sensibilité RMS, silence de fin de phrase et durées de segment** modifiables en direct |
| **Entrées/Sorties Disque** | Écritures/lectures continues de WAV | **Zéro écriture disque** : streaming 100% en mémoire (*in-memory ring buffers*) |
| **Accélération Matérielle** | PyTorch / CTranslate2 CPU | **Metal GPU natif Apple Silicon (M1 à M4)** via `whisper.cpp` + CPU multithreadé |
| **Autonomie & Poids** | 8 à 12 Go d'environnement Python | **Binaire unique autonome** avec assets HTML/CSS intégrés à la compilation |

---

## 📁 Organisation du Répertoire

```text
SpeechToTextToTranslate/
├── Cargo.toml                  # Configuration Cargo et dépendances (gpui-kit, whisper-rs, axum, cpal, tokio)
├── Cargo.lock                  # Verrouillage reproductible des versions de dépendances
├── config.json                 # Configuration active de l'application (sauvegarde automatique)
├── config.example.json         # Modèle de configuration par défaut
├── LICENSE                     # Licence MIT
├── README.md                   # Documentation principale du projet
│
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
│   ├── start_local_asr.sh      # Script bash de démarrage rapide (port 8000)
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

VoiceBridge Pro prend en charge deux catégories de modèles, avec bascule intelligente :

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

### Paramètres URL personnalisables :
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

## 🚀 Compilation & Lancement

### Prérequis
- **Rust** (stable, édition 2024) : `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Python 3** (optionnel, uniquement pour les modèles nécessitant le serveur ASR local)

### Commandes
```bash
# Compilation et lancement en mode release (optimisé GPU) :
cargo run --release

# Lancement direct du binaire compilé :
./target/release/SpeechToTextToTranslate

# Mode headless / serveur uniquement (sans interface graphique de bureau) :
./target/release/SpeechToTextToTranslate --headless
```

---

## 📜 Licence

Ce projet est distribué sous licence MIT — voir le fichier [LICENSE](LICENSE) pour plus de détails.
