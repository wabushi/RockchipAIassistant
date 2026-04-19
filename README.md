This is a repo for an AI assistant running on ROCK-5A device.
Due to the size of the AI model files - the are added to the .gitignore file - but they are available publicly.

Directory tree:
```text
slm/
├── README.md
├── Notes.txt
├── models/
│   └── qwen2.5-1.5b-instruct-q4_k_m.gguf
├── tts/
│   ├── llm_tts_simple.py
│   ├── voices/
│   │   ├── en_US-lessac-medium.onnx
│   │   └── en_US-lessac-medium.onnx.json
│   └── output/
└── llama.cpp/
    ├── build/
    ├── include/
    ├── src/
    └── examples/
```

**Clone repos:**
git clone https://github.com/ggml-org/llama.cpp.git

**Download the SLM AI models:**
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/blob/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
## System Requirements (APT)

```bash
sudo apt update
sudo apt install -y \
  git \
  wget \
  curl \
  python3 \
  python3-venv \
  python3-pip \
  build-essential \
  cmake \
  alsa-utils \
  ffmpeg
```

---

## Python Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install piper-tts
```
**Get onnx voice model from piper here (.onnx & .json files required):**
https://huggingface.co/rhasspy/piper-voices/tree/main

**To run the AI ASSISTANT:**
1. start the virtual environment:
   ```bash
   cd tts
   source venv/bin/activate
   ```
2. ```bash
   python3 tts/llm_tts_simple.py
   ```
