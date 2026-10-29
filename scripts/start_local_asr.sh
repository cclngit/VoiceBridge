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

export PYTHONWARNINGS="ignore::UserWarning:multiprocessing.resource_tracker"

if [ -x ".venv/bin/python3" ]; then
    PYTHON_BIN=".venv/bin/python3"
elif [ -x "venv/bin/python3" ]; then
    PYTHON_BIN="venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "❌ Erreur : Python 3 introuvable (.venv, python3, python)." >&2
    exit 1
fi

exec "$PYTHON_BIN" scripts/asr_server.py --host 127.0.0.1 --port 8000 "$MODEL"
