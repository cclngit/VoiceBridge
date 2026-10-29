# Scripts Utilitaires (VoiceBridge)

Ce répertoire contient les scripts utilitaires du serveur ASR local :

## 1. `asr_server.py`

Serveur ASR local compatible OpenAI (`/v1/audio/transcriptions`), basé sur `faster-whisper-server`.

- **Gestion automatique** : Ce script est démarré et arrêté automatiquement par l'interface graphique VoiceBridge lorsque vous sélectionnez un modèle nécessitant le serveur ASR (Qwen3-ASR, Voxtral, Moonshine, VibeVoice, Canary, etc.).
- **Compatibilité PyAV** : Intègre un shim universel assurant la compatibilité avec toutes les versions récentes de PyAV (v14 à v19+).
- **Mappage intelligent** : Résout automatiquement les requêtes de modèles vers les architectures CTranslate2 optimales.

Lancement manuel :

```bash
python3 scripts/asr_server.py [modele]
# Exemple :
python3 scripts/asr_server.py Systran/faster-whisper-base
```

## 2. `start_local_asr.sh`

Script shell pour lancer le serveur ASR local en ligne de commande :

```bash
./scripts/start_local_asr.sh [modele]
```

Par défaut, écoute sur `http://127.0.0.1:8000`.
