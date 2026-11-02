## 🚀 VoiceBridge v0.3.0 — Notes de version (Rupture Technologique Streaming)

Cette version majeure marque le passage de VoiceBridge d'une architecture par fenêtres audio (chunks batch) à une **architecture de streaming audio-texte continue à ultra-faible latence**, inspirée des travaux de recherche de Kyutai Labs (*Moshi*, *Mimi*, *Delayed Streams Modeling* et *Hibiki* - [arXiv:2410.00037](https://arxiv.org/abs/2410.00037)).

---

### 🌟 Nouveautés Majeures

#### 1. Codec Audio Neuronal Causal Mimi (24 kHz / 12,5 Hz)
- **Traitement causal par trames de 80 ms** : L'audio capturé est converti en flux 24 kHz et traité par pas de 1 920 échantillons (80 ms), avec seulement **80 ms de lookahead** (latence algorithmique aller-retour < 160 ms).
- **Split RVQ (Residual Vector Quantization)** :
  - **Codebook 0 (Sémantique)** : Distillation des représentations phonétiques et linguistiques de WavLM.
  - **Codebooks 1 à 7 (Acoustique)** : Modélisation des détails acoustiques pour la reconstruction haute fidélité (débit ultra-compact de 1,1 kbps).
- **Buffer audio causal** : Garantie d'absence de gigue (*zero-underflow*) et élimination des découpages artificiels.

#### 2. Moteur STT Streaming sub-seconde & Semantic VAD (Kyutai 1B FR/EN)
- **Latence fixe et bornée à 500 ms** : Grâce au paradigme *Delayed Streams Modeling*, les tokens textuels sont décodés avec un décalage de seulement 6 pas (480 ms ~ 500 ms) par rapport à la voix, divisant par 4 la latence perçue par rapport aux découpages Whisper classiques (1 500 ms à 3 500 ms).
- **Semantic VAD (Détection Vocale Sémantique)** : Le modèle analyse directement les probabilités des tokens linguistiques pour distinguer le début et la fin de parole réelle des respirations, bruits de micro ou hésitations ("euh...").
- **Horodatages au mot près (*Word-Level Timestamps*)** : Chaque mot est émis avec ses marqueurs temporels `start_ms` et `end_ms` précis à 80 ms près.
- **Coexistence harmonieuse des moteurs** : Possibilité de basculer instantanément dans l'interface entre *Kyutai Streaming*, *Whisper Local GPU/CPU* et *Remote API (Speaches/vLLM)*.

#### 3. Traduction Simultanée en Direct (Inspirée d'Hibiki)
- **Politique adaptative Wait-$k$** : La traduction n'attend plus la fin de la phrase. Les mots et clauses traduits sont émis au fil de l'eau dès que le contexte grammatical source est suffisant.
- **Streaming mot à mot bilingue** : Traduction simultanée continue optimisée pour le Français et l'Anglais.

#### 4. Overlay OBS Studio & Interface GPUI Temps Réel
- **Affichage 'Word Ticker' dynamique** : Dans `obs.html`, les mots apparaissent mot par mot avec une animation fluide d'apparition progressive (`@keyframes fadeInWord`) et un indicateur pulsé en direct.
- **Protocole d'événements streaming sub-100ms** : Événements SSE / WebSocket enrichis pour diffuser les tokens instantanément aux clients OBS et à l'interface GPUI.

#### 5. Framework Machine Learning Candle en Rust Natif (CUDA & Metal)
- **Intégration directe sans bridge externe** : Utilisation des crates Hugging Face `candle-core`, `candle-nn` et `candle-transformers` directement en Rust.
- **Accélération matérielle complète** :
  - **NVIDIA CUDA** sur Windows x64 (RTX 5060, RTX 40xx/30xx) avec allocation mémoire VRAM contenue (~1,2 Go).
  - **Apple Silicon Metal** sur macOS.
  - **Fallback CPU optimisé multithread**.

---

### 🧪 Validation & Tests Unitaires
- Suite de tests unitaires portée à **13 tests automatisés** couvrant l'ensemble du pipeline :
  - Mathématiques et débits du codec Mimi
  - Précision du buffer causal 24 kHz (1 920 samples)
  - Alignement temporel des tokens et transitions du Semantic VAD
  - Streaming simultané Wait-$k$ Hibiki
  - Rétrocompatibilité totale des modèles Whisper locaux et de la traduction de secours.
