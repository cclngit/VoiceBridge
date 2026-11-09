## VoiceBridge — prochaine version (non publiée)

### ⚠️ Correctifs importants par rapport à la v0.3.0

La v0.3.0 annonçait un codec Mimi, un moteur Kyutai STT et une traduction Hibiki qui étaient **simulés** (sorties générées à partir de l'énergie du signal et d'une liste de phrases fixes). Ils sont remplacés par les **vrais modèles Kyutai**, exécutés via la crate officielle `moshi`.

### 🌟 Nouveautés

- **Kyutai STT 1B streaming (FR/EN)** — poids `kyutai/stt-1b-en_fr-candle`, codec Mimi réel, VAD sémantique pour la fin de phrase, mots affichés en direct (~0,5 s).
- **Kyutai Hibiki 1B / 2B** — traduction simultanée parole française → texte anglais.
- **Traduction 100 % locale MADLAD-400 3B** (400 langues) intégrée au binaire, ou LLM local (LM Studio / Ollama). Les services en ligne (Google, MyMemory, LibreTranslate) sont supprimés.
- **`--input-wav`** : sous-titrer un enregistrement au lieu du micro.
- Téléchargements de modèles multi-fichiers, figés sur une révision Hugging Face, vérifiés par **SHA-256**, avec reprise des fichiers partiels.

### 🔒 Sécurité

- API web limitée aux requêtes de même origine vers `localhost` (suppression du CORS permissif, protection CSRF et DNS-rebinding) : un site web ne peut plus lire les clés API ni rediriger l'audio du micro.
- Endpoints ASR / LLM limités aux adresses locales (loopback / réseau local). Clés API masquées dans l'API.

### 🐛 Corrections

- Inférence Whisper déplacée hors du runtime async ; sous-titres traités dans l'ordre.
- Échec de rechargement d'un modèle Whisper : l'ancien modèle est restauré.
- Téléchargement : plus de blocage « téléchargement en cours » après une erreur ; fichiers tronqués rejetés.
- `--config` respecté pour toutes les sauvegardes (écriture atomique).
- VAD : la durée minimale ne compte que la parole réelle ; silence final rogné.
- Rééchantillonnage avec filtre anti-repliement et sans perte entre blocs (`rubato`).
- `scripts/asr_server.py` ne substitue plus silencieusement faster-whisper aux modèles Qwen3 / Voxtral / Canary / VibeVoice.
- Suppression des valeurs factices de l'interface (« 98,4 % de confiance », latences de démonstration) et des polices chargées depuis Google Fonts.

### 🧪 Tests

31 tests unitaires hors ligne (sécurité HTTP, politique réseau, VAD, rééchantillonnage, SentencePiece, catalogue, configuration, téléchargeur) et 4 tests sur les vrais modèles (`cargo test --release -- --ignored`).
