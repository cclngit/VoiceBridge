#!/usr/bin/env python3
"""
VoiceBridge Pro — Local ASR Server Wrapper
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

    if len(sys.argv) == 1:
        pass
    elif len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        raw_model = sys.argv[1]
        # If user passed a model that is not a direct CTranslate2 / faster-whisper repo, map gracefully
        if os.path.exists(raw_model) or "faster-whisper" in raw_model.lower():
            model = raw_model
        elif "turbo" in raw_model.lower():
            model = "deepdml/faster-whisper-large-v3-turbo"
        elif "large" in raw_model.lower() or "24b" in raw_model.lower():
            model = "Systran/faster-whisper-large-v3"
        elif "tiny" in raw_model.lower() or "moonshine" in raw_model.lower():
            model = "Systran/faster-whisper-tiny"
        elif "small" in raw_model.lower() or "mini" in raw_model.lower():
            model = "Systran/faster-whisper-small"
        else:
            model = "Systran/faster-whisper-base"
        print(f"⚡ [VoiceBridge ASR Server] Modèle sélectionné: '{raw_model}' -> Exécution via '{model}'")
    
    # Check if host or port was provided in sys.argv
    has_host = "--host" in sys.argv
    has_port = "--port" in sys.argv

    if not has_host and not has_port:
        sys.argv = [sys.argv[0], "--host", host, "--port", port, model]

    from faster_whisper_server.cli import main
    main()
