# Modèles Whisper GGML (VoiceBridge)

Ce répertoire héberge les modèles Whisper au format binaire GGML (`.bin`) utilisés pour l'inférence locale haute performance (accélération matérielle Apple Silicon Metal GPU et CPU AVX2).

## Téléchargement

Les modèles peuvent être téléchargés directement en un clic depuis l'interface graphique de VoiceBridge (Onglet **Modèles IA**).

Vous pouvez également télécharger manuellement les modèles officiels whisper.cpp :

| Modèle | Précision / Taille | Commande de téléchargement |
| :--- | :--- | :--- |
| **Whisper Tiny** | 75 Mo | `curl -L -o models/ggml-tiny.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin` |
| **Whisper Base** | 142 Mo | `curl -L -o models/ggml-base.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin` |
| **Whisper Small** | 466 Mo | `curl -L -o models/ggml-small.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin` |
| **Whisper Turbo Q5** | 547 Mo | `curl -L -o models/ggml-large-v3-turbo-q5_0.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin` |
| **Whisper Turbo** | 1.5 Go | `curl -L -o models/ggml-large-v3-turbo.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin` |
| **Whisper Large-v3** | 2.9 Go | `curl -L -o models/ggml-large-v3.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin` |

*Note : Les fichiers binaires `.bin` sont exclus du suivi Git via `.gitignore`.*
