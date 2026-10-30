## 🚀 VoiceBridge v0.2.1 — Notes de version (Changements)

Cette mise à jour corrective résout définitivement les problèmes de détection de langue, de changement de langue et de transcription audio.

### 🐛 Corrections de bugs & Améliorations

- **Correction critique de la détection automatique de langue (Whisper STT)**
  - Correction du mode `auto` : passage de `params.set_detect_language(false)` avec `params.set_language(Some("auto"))`. Auparavant, activer `detect_language` provoquait un arrêt immédiat de `whisper.cpp` après la classification sans exécuter le décodage du texte, ce qui rendait le français muet en détection auto.
  - La détection automatique identifie désormais la langue parlée (FR, EN, ES, DE, IT...) et transcrit immédiatement le texte dans la langue source.

- **Isolement et étiquetage de `Distil-Whisper Large-v3 [Anglais]`**
  - Distil-Whisper est un modèle optimisé par Hugging Face exclusivement sur l'anglais. Il est désormais clairement étiqueté `[Anglais]` dans le catalogue pour éviter qu'un utilisateur francophone ne l'utilise par mégarde (ce qui provoquait des transcriptions phonétiques en anglais comme *"And it's for that"* pour *"Et c'est pour ça"*).
  - Le modèle par défaut et recommandé est désormais **Whisper Large-v3-Turbo Q5_0** (100% multilingue, très haute précision sur le français et ultra-rapide sur GPU).

- **Nouveau moteur de traduction direct Google Chrome Neural MT & GTX**
  - Ajout de l'endpoint direct de traduction neuronale Google Chrome (`dict-chrome-ex`) avec fallback Google GTX pour toutes les paires de langues (EN, FR, ES, etc.).
  - Résout définitivement les erreurs de quota HTTP 429 rencontrées avec MyMemory.

- **Assainissement automatique de la configuration (`AppConfig::sanitize`)**
  - Correction du bug où la langue revenait automatiquement au français ou restait bloquée sur un serveur local éteint (`remote_asr`).
  - La routine d'assainissement normalise automatiquement la configuration au démarrage :
    - Bascule vers Whisper local GPU si aucun serveur Speaches n'est joignable sur le port 8000.
    - Empêche les boucles de traduction identiques (ex: `source FR -> target FR`).
    - Synchronise les changements de langue instantanément entre l'interface GPUI, le service de traduction et le moteur Whisper en mémoire.

- **Affichage dynamique dans la barre de titre**
  - La barre supérieure de l'application affiche désormais le vrai modèle actif ainsi que le GPU matériel détecté (ex: `NVIDIA GeForce RTX 5060 (Vulkan GPU) / Whisper Large-v3-Turbo Q5_0`).
