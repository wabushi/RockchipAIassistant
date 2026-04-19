#!/usr/bin/env python3
import os
import re
import sys
import wave
import subprocess
from pathlib import Path

import pexpect


LLAMA_BIN = os.path.expanduser("~/slm/llama.cpp/build/bin/llama-cli")
MODEL_PATH = os.path.expanduser("~/slm/models/qwen2.5-1.5b-instruct-q4-k-m.gguf")

PIPER_BIN = os.path.expanduser("~/slm/tts/venv/bin/piper")
PIPER_MODEL = os.path.expanduser("~/slm/tts/voices/en_US-lessac-medium.onnx")
APLAY_BIN = "/usr/bin/aplay"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_WAV = OUTPUT_DIR / "llm_tts_output.wav"
LOG_FILE = OUTPUT_DIR / "run.log"
RAW_FILE = OUTPUT_DIR / "llm_last_raw.txt"
CLEAN_FILE = OUTPUT_DIR / "llm_last_clean.txt"

# Add a short silence to the start of every WAV so playback hardware
# does not clip the first spoken syllables.
LEADING_SILENCE_MS = 350


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG_FILE.open("a", encoding="utf-8", errors="replace") as f:
        f.write(msg + "\n")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def reset_outputs() -> None:
    ensure_output_dir()
    for p in [OUTPUT_WAV, LOG_FILE, RAW_FILE, CLEAN_FILE]:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass
    LOG_FILE.write_text("", encoding="utf-8")


def check_file(path_str: str, label: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise RuntimeError(f"{label} is not a file: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"{label} is empty: {path}")
    return path


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1B[@-_][0-?]*[ -/]*[@-~]", "", text)


def clean_response(raw_text: str, prompt: str) -> str:
    text = strip_ansi(raw_text or "")
    text = text.replace("\r", "")

    # Remove performance/status lines
    text = re.sub(r"\[\s*Prompt:.*?\]", " ", text)
    text = re.sub(r"\[\s*Generation:.*?\]", " ", text)

    # Remove echoed user prompt
    p = prompt.strip()
    if p:
        ep = re.escape(p)
        text = re.sub(rf"(^|\n)\s*{ep}\s*($|\n)", "\n", text, flags=re.MULTILINE)

    # Remove startup/banner/help noise if it leaks into a turn
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(">"):
            continue
        if s.startswith("build"):
            continue
        if s.startswith("model"):
            continue
        if s.startswith("modalities"):
            continue
        if s.startswith("available commands"):
            continue
        if s.startswith("/exit"):
            continue
        if s.startswith("/regen"):
            continue
        if s.startswith("/clear"):
            continue
        if s.startswith("/read"):
            continue
        if s.startswith("/glob"):
            continue
        if all(ch in "▄▀█ \t" for ch in s):
            continue
        lines.append(s)

    text = "\n".join(lines)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def sanitize_for_tts(text: str) -> str:
    text = text.strip()

    # Remove slash-based noise (///, /, etc.)
    text = re.sub(r"[\\/]{2,}", " ", text)
    text = re.sub(r"(^|\s)[\\/]+($|\s)", " ", text)

    # Remove bracketed tags like [TTS], [INFO], etc.
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # Remove leftover commands like /exit /clear
    text = re.sub(r"/\w+", " ", text)

    # Remove weird punctuation-only chunks
    text = re.sub(r"[^\w\s.,!?'\-]", " ", text)

    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepend_silence_to_wav(wav_path: Path, silence_ms: int = LEADING_SILENCE_MS) -> None:
    """
    Reads the existing WAV, prepends digital silence, and writes it back.
    This is usually the most reliable fix for clipped first words on ALSA devices.
    """
    with wave.open(str(wav_path), "rb") as wf:
        params = wf.getparams()
        audio_data = wf.readframes(wf.getnframes())

    framerate = params.framerate
    nchannels = params.nchannels
    sampwidth = params.sampwidth

    silence_frames = int(framerate * silence_ms / 1000)
    silence_bytes = b"\x00" * silence_frames * nchannels * sampwidth

    with wave.open(str(wav_path), "wb") as wf:
        wf.setparams(params)
        wf.writeframes(silence_bytes + audio_data)


def synthesize_and_play(text: str) -> None:
    text = sanitize_for_tts(text)

    if not text:
        raise RuntimeError("No text to synthesize after cleaning")

    if OUTPUT_WAV.exists():
        OUTPUT_WAV.unlink()

    log(f"[TTS ] {text}")

    res = subprocess.run(
        [
            PIPER_BIN,
            "--model",
            PIPER_MODEL,
            "--output_file",
            str(OUTPUT_WAV),
        ],
        input=text,
        text=True,
        capture_output=True,
    )

    if res.returncode != 0:
        if res.stderr:
            log("[PIPER STDERR]")
            log(res.stderr)
        raise RuntimeError("Piper synthesis failed")

    check_file(str(OUTPUT_WAV), "Generated WAV")

    prepend_silence_to_wav(OUTPUT_WAV, LEADING_SILENCE_MS)
    log(f"[INFO] Prepended {LEADING_SILENCE_MS} ms of silence to WAV")

    play_res = subprocess.run(
        [APLAY_BIN, str(OUTPUT_WAV)],
        capture_output=True,
        text=True,
    )

    if play_res.returncode != 0:
        if play_res.stderr:
            log("[APLAY STDERR]")
            log(play_res.stderr)
        raise RuntimeError("Audio playback failed")


class LlamaSession:
    def __init__(self):
        reset_outputs()

        check_file(LLAMA_BIN, "llama-cli binary")
        check_file(MODEL_PATH, "GGUF model")
        check_file(PIPER_BIN, "Piper binary")
        check_file(PIPER_MODEL, "Piper model")
        check_file(APLAY_BIN, "aplay binary")

        self.child = None

    def start(self) -> None:
        cmd = (
            f'{LLAMA_BIN} '
            f'-m "{MODEL_PATH}" '
            f'-t 6 '
            f'-c 8192 '
            f'--keep 512 '
            f'-n -1 '
            f'--repeat-penalty 1.15'
        )

        log("[INFO] Starting llama.cpp session...")
        log(f"[CMD ] {cmd}")

        self.child = pexpect.spawn(
            cmd,
            encoding="utf-8",
            timeout=120,
        )

        self.child.expect(r"\n> ")
        log("[INFO] llama.cpp is ready")

    def ask(self, prompt: str) -> str:
        if not self.child:
            raise RuntimeError("llama session is not running")

        prompt = prompt.strip()
        if not prompt:
            return ""

        log(f"[USER ] {prompt}")

        self.child.sendline(prompt)

        self.child.expect(r"\n> ")
        raw = self.child.before or ""

        RAW_FILE.write_text(raw, encoding="utf-8", errors="replace")
        cleaned = clean_response(raw, prompt)
        CLEAN_FILE.write_text(cleaned, encoding="utf-8", errors="replace")

        log("[RAW ]")
        log(raw.strip() if raw.strip() else "<EMPTY>")
        log("[CLEAN]")
        log(cleaned if cleaned else "<EMPTY>")

        return cleaned

    def close(self) -> None:
        if self.child is not None:
            try:
                self.child.sendline("/exit")
            except Exception:
                pass
            try:
                self.child.close(force=True)
            except Exception:
                pass


def main() -> None:
    session = LlamaSession()

    try:
        session.start()

        if len(sys.argv) > 1:
            prompt = " ".join(sys.argv[1:])
            reply = session.ask(prompt)
            if reply:
                synthesize_and_play(reply)
            else:
                log("[WARN] Empty cleaned response")
            return

        while True:
            try:
                prompt = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not prompt:
                continue

            if prompt.lower() in {"exit", "quit"}:
                break

            reply = session.ask(prompt)
            if reply:
                synthesize_and_play(reply)
            else:
                log("[WARN] Empty cleaned response")

    finally:
        session.close()


if __name__ == "__main__":
    main()
