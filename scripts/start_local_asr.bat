@echo off
REM ==============================================================================
REM VoiceBridge — Serveur ASR Local OpenAI-Compatible (127.0.0.1:8000) - Windows
REM ==============================================================================
setlocal

echo.
echo [VoiceBridge] Serveur ASR Local OpenAI-Compatible sur http://127.0.0.1:8000...
echo ------------------------------------------------------------------

set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=Systran/faster-whisper-base"

echo Modele ASR selectionne : %MODEL%
echo Lancement sur http://127.0.0.1:8000/v1/audio/transcriptions...
echo.

set "PYTHONWARNINGS=ignore::UserWarning:multiprocessing.resource_tracker"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\asr_server.py --host 127.0.0.1 --port 8000 "%MODEL%"
    exit /b %ERRORLEVEL%
)

if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" scripts\asr_server.py --host 127.0.0.1 --port 8000 "%MODEL%"
    exit /b %ERRORLEVEL%
)

python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python scripts\asr_server.py --host 127.0.0.1 --port 8000 "%MODEL%"
    exit /b %ERRORLEVEL%
)

py -3 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py -3 scripts\asr_server.py --host 127.0.0.1 --port 8000 "%MODEL%"
    exit /b %ERRORLEVEL%
)

python3 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python3 scripts\asr_server.py --host 127.0.0.1 --port 8000 "%MODEL%"
    exit /b %ERRORLEVEL%
)

echo [Erreur] Python 3 introuvable dans le PATH (.venv, python, py -3).
exit /b 1

