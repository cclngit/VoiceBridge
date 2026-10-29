#!/usr/bin/env bash
# ==============================================================================
# VoiceBridge — Serveur ASR Local OpenAI-Compatible (127.0.0.1:8000)
# ==============================================================================

set -e

echo ""
echo "🚀 Serveur ASR Local OpenAI-Compatible sur http://127.0.0.1:8000..."
echo "------------------------------------------------------------------"

MODEL="${1:-Systran/faster-whisper-base}"

echo "🧠 Modèle ASR sélectionné : $MODEL"
echo "✨ Lancement sur http://127.0.0.1:8000/v1/audio/transcriptions..."
echo ""

exec python3 scripts/asr_server.py --host 127.0.0.1 --port 8000 "$MODEL"
