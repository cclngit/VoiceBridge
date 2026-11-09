#!/usr/bin/env python3
"""
VoiceBridge — Local ASR Server Wrapper
Ensures 100% PyAV compatibility across all versions on macOS / Linux / Windows.
"""
import os
import sys
import av

# PyAV compatibility shim for av 14 to 19+
_orig_av_open = av.open
def _safe_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _orig_av_open(*args, **kwargs)
av.open = _safe_av_open

if __name__ == "__main__":
    host = "127.0.0.1"
    port = "8000"
    model = "Systran/faster-whisper-base"

    # Models this wrapper can actually serve (CTranslate2 / faster-whisper checkpoints).
    # Other ASR models (Qwen3-ASR, Voxtral, Canary, VibeVoice...) need their own server
    # such as vLLM: they are never silently replaced by a different model.
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        requested = sys.argv[1]
        if os.path.exists(requested) or "faster-whisper" in requested.lower():
            model = requested
        else:
            sys.stderr.write(
                f"[VoiceBridge ASR Server] '{requested}' n'est pas un modèle faster-whisper : "
                "lancez un serveur dédié (vLLM / Speaches) pour ce modèle.\n"
            )
            sys.exit(2)
        print(f"⚡ [VoiceBridge ASR Server] Modèle : '{model}' sur http://{host}:{port}")

    # Check if host or port was provided in sys.argv
    has_host = "--host" in sys.argv
    has_port = "--port" in sys.argv

    if not has_host and not has_port:
        sys.argv = [sys.argv[0], "--host", host, "--port", port, model]

    from faster_whisper_server.cli import main
    main()
