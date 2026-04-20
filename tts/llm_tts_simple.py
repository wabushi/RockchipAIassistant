#!/usr/bin/env python3
import os
import re
import sys
import wave
import shutil
import subprocess
import threading
import select
import termios
import tty
from queue import Queue
from pathlib import Path

import pexpect


LLAMA_BIN = os.path.expanduser("~/slm/llama.cpp/build/bin/llama-cli")
MODEL_PATH = os.path.expanduser("~/slm/models/qwen2.5-3b-instruct-q4_k_m.gguf")

PIPER_BIN = os.path.expanduser("~/slm/tts/venv/bin/piper")
PIPER_MODEL = os.path.expanduser("~/slm/tts/voices/en_US-danny-low.onnx")
APLAY_BIN = "/usr/bin/aplay"

SCRIPT_DIR = Path(__file__).resolve().parent
SOUNDS_DIR = SCRIPT_DIR / "sounds"
LOADING_SOUND = str(SOUNDS_DIR / "loading.wav")
BOOT_TEXT = "Booting Up AI"
READY_TEXT = "AI Assistant Ready"
TERMINATING_TEXT = "Terminating AI assistant"
TERMINATED_TEXT = "AI Assistant terminated!"

OUTPUT_DIR = SCRIPT_DIR / "output"
AUDIO_DIR = OUTPUT_DIR / "audio_chunks"

LOG_FILE = OUTPUT_DIR / "run.log"
RAW_FILE = OUTPUT_DIR / "llm_last_raw.txt"
CLEAN_FILE = OUTPUT_DIR / "llm_last_clean.txt"
STARTUP_FILE = Path(os.path.expanduser("~/slm/llm_startup.txt"))
ABSORBED_STARTUP_FILE = OUTPUT_DIR / "absorbed_startup.txt"

LEADING_SILENCE_MS = 350
PIPER_TIMEOUT_SEC = 120
APLAY_TIMEOUT_SEC = 120
PREBUFFER_SENTENCE_COUNT = 2
IDLE_TIMEOUT_SEC = 60


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG_FILE.open("a", encoding="utf-8", errors="replace") as f:
        f.write(msg + "\n")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SOUNDS_DIR.mkdir(parents=True, exist_ok=True)


def reset_outputs() -> None:
    ensure_output_dir()

    for p in [LOG_FILE, RAW_FILE, CLEAN_FILE, ABSORBED_STARTUP_FILE]:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    if AUDIO_DIR.exists():
        try:
            shutil.rmtree(AUDIO_DIR)
        except Exception:
            pass
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

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


def strip_control_garbage(text: str) -> str:
    if not text:
        return ""

    text = text.replace("^H", "")
    text = text.replace("|", " ")
    text = text.replace("\b", "")
    text = text.replace("\x7f", "")
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)

    return text


def sanitize_for_tts(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[\\/]{2,}", " ", text)
    text = re.sub(r"(^|\s)[\\/]+($|\s)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"/\w+", " ", text)
    text = re.sub(r"[^\w\s.,!?'\-:;]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    parts = re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", text)
    sentences = [p.strip() for p in parts if p.strip()]

    merged = []
    for s in sentences:
        if merged and len(s.split()) <= 2 and not re.search(r"[.!?]$", s):
            merged[-1] = f"{merged[-1]} {s}".strip()
        else:
            merged.append(s)

    return merged


def prepend_silence_to_wav(wav_path: Path, silence_ms: int = LEADING_SILENCE_MS) -> None:
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


def synthesize_sentence_to_wav(sentence: str, wav_path: Path, add_leading_silence: bool = True) -> None:
    sentence = sanitize_for_tts(sentence)
    if not sentence:
        raise RuntimeError("Empty sentence passed to Piper")

    if wav_path.exists():
        wav_path.unlink()

    log(f"[TTS ] {sentence}")

    res = subprocess.run(
        [
            PIPER_BIN,
            "--model",
            PIPER_MODEL,
            "--output_file",
            str(wav_path),
        ],
        input=sentence,
        text=True,
        capture_output=True,
        timeout=PIPER_TIMEOUT_SEC,
    )

    if res.returncode != 0:
        if res.stderr:
            log("[PIPER STDERR]")
            log(res.stderr)
        raise RuntimeError("Piper synthesis failed")

    check_file(str(wav_path), "Generated WAV")

    if add_leading_silence:
        prepend_silence_to_wav(wav_path, LEADING_SILENCE_MS)


def play_wav_blocking(wav_path: Path) -> None:
    check_file(str(wav_path), "Playback WAV")

    res = subprocess.run(
        [APLAY_BIN, str(wav_path)],
        capture_output=True,
        text=True,
        timeout=APLAY_TIMEOUT_SEC,
    )

    if res.returncode != 0:
        if res.stderr:
            log("[APLAY STDERR]")
            log(res.stderr)
        raise RuntimeError("Audio playback failed")


def get_voice_model_name() -> str:
    return Path(PIPER_MODEL).stem


def get_boot_wav_path() -> Path:
    return SOUNDS_DIR / f"{get_voice_model_name()}_BOOT.wav"


def get_ready_wav_path() -> Path:
    return SOUNDS_DIR / f"{get_voice_model_name()}_READY.wav"


def get_terminating_wav_path() -> Path:
    return SOUNDS_DIR / f"{get_voice_model_name()}_TERMINATING.wav"


def get_terminated_wav_path() -> Path:
    return SOUNDS_DIR / f"{get_voice_model_name()}_TERMINATED.wav"


def ensure_phrase_wav(text: str, wav_path: Path) -> None:
    if not wav_path.exists():
        log(f"[INFO] Missing WAV, generating: {wav_path}")
        synthesize_sentence_to_wav(text, wav_path, add_leading_silence=True)
    else:
        log(f"[INFO] Reusing existing WAV: {wav_path}")


def ensure_and_play_boot_sound() -> None:
    boot_wav = get_boot_wav_path()
    ensure_phrase_wav(BOOT_TEXT, boot_wav)
    log("[INFO] Playing boot WAV")
    play_wav_blocking(boot_wav)


def ensure_and_play_ready_sound() -> None:
    ready_wav = get_ready_wav_path()
    ensure_phrase_wav(READY_TEXT, ready_wav)
    log("[INFO] Playing ready WAV")
    play_wav_blocking(ready_wav)


def ensure_and_play_terminating_sound() -> None:
    term_wav = get_terminating_wav_path()
    ensure_phrase_wav(TERMINATING_TEXT, term_wav)
    log("[INFO] Playing terminating WAV")
    play_wav_blocking(term_wav)


def ensure_and_play_terminated_sound() -> None:
    terminated_wav = get_terminated_wav_path()
    ensure_phrase_wav(TERMINATED_TEXT, terminated_wav)
    log("[INFO] Playing terminated WAV")
    play_wav_blocking(terminated_wav)


def parse_startup_file(file_path: Path) -> tuple[str, str]:
    if not file_path.exists():
        return "", ""

    raw = file_path.read_text(encoding="utf-8", errors="replace")

    static_match = re.search(
        r"Static:\s*(.*?)(?=\n\s*Dynamic:\s*|\Z)",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    dynamic_match = re.search(
        r"Dynamic:\s*(.*)$",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )

    static_text = static_match.group(1).strip() if static_match else ""
    dynamic_text = dynamic_match.group(1).strip() if dynamic_match else ""

    return static_text, dynamic_text


def write_startup_file(file_path: Path, static_text: str, dynamic_text: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"Static:\n{static_text.strip()}\n\nDynamic:\n{dynamic_text.strip()}\n"
    file_path.write_text(content, encoding="utf-8")


def build_absorbed_startup_text(static_text: str, dynamic_text: str) -> str:
    parts = []
    if static_text.strip():
        parts.append(static_text.strip())
    if dynamic_text.strip():
        parts.append(dynamic_text.strip())
    return "\n".join(parts).strip()


def prepare_absorbed_startup_file() -> Path | None:
    static_text, dynamic_text = parse_startup_file(STARTUP_FILE)
    absorbed = build_absorbed_startup_text(static_text, dynamic_text)

    if not absorbed:
        return None

    ABSORBED_STARTUP_FILE.parent.mkdir(parents=True, exist_ok=True)
    ABSORBED_STARTUP_FILE.write_text(absorbed + "\n", encoding="utf-8")
    return ABSORBED_STARTUP_FILE


def clean_response(raw_text: str, prompt: str) -> str:
    text = strip_ansi(raw_text or "")
    text = strip_control_garbage(text)
    text = text.replace("\r", "")

    text = re.sub(r"\[\s*Prompt:.*?\]", " ", text)
    text = re.sub(r"\[\s*Generation:.*?\]", " ", text)

    p = prompt.strip()
    if p:
        ep = re.escape(p)
        text = re.sub(rf"(^|\n)\s*{ep}\s*($|\n)", "\n", text, flags=re.MULTILINE)

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

    if text.startswith("Assistant:"):
        text = text[len("Assistant:"):].strip()

    return text


class LoadingSoundPlayer:
    def __init__(self, sound_path: str):
        self.sound_path = Path(sound_path)
        self._stop_event = threading.Event()
        self._thread = None
        self._current_proc = None
        self._running = False

    def start(self) -> None:
        if not self.sound_path.exists():
            log(f"[WARN] Loading sound not found: {self.sound_path}")
            return
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._running = True
        log(f"[INFO] Loading sound started: {self.sound_path}")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._current_proc = subprocess.Popen(
                    [APLAY_BIN, str(self.sound_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._current_proc.wait()
            except Exception as e:
                log(f"[WARN] Loading sound playback failed: {e}")
                break
            finally:
                self._current_proc = None

    def stop(self) -> None:
        if not self._running:
            return

        self._stop_event.set()

        if self._current_proc is not None:
            try:
                self._current_proc.terminate()
            except Exception:
                pass
            try:
                self._current_proc.wait(timeout=1)
            except Exception:
                try:
                    self._current_proc.kill()
                except Exception:
                    pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)

        self._running = False
        log("[INFO] Loading sound stopped")


def speak_text_progressively(text: str, loading_player: LoadingSoundPlayer | None = None) -> None:
    text = sanitize_for_tts(text)
    if not text:
        raise RuntimeError("No text to synthesize after cleaning")

    sentences = split_into_sentences(text)
    if not sentences:
        raise RuntimeError("No sentences found for TTS")

    total_sentences = len(sentences)
    prebuffer_target = min(PREBUFFER_SENTENCE_COUNT, total_sentences)

    log(f"[INFO] Split reply into {total_sentences} sentence(s)")
    log(f"[INFO] Prebuffer target: {prebuffer_target} sentence(s)")

    synthesis_queue: Queue = Queue()
    stop_event = threading.Event()
    prebuffer_ready = threading.Event()
    producer_error = {"msg": None}
    synthesized_count = {"value": 0}
    counter_lock = threading.Lock()

    def producer() -> None:
        try:
            for idx, sentence in enumerate(sentences, start=1):
                if stop_event.is_set():
                    break

                wav_path = AUDIO_DIR / f"chunk_{idx:03d}.wav"
                log(f"[INFO] Synthesizing sentence {idx}/{total_sentences}")
                synthesize_sentence_to_wav(sentence, wav_path)
                synthesis_queue.put((idx, sentence, wav_path))
                log(f"[INFO] Queued sentence {idx}/{total_sentences}")

                with counter_lock:
                    synthesized_count["value"] += 1
                    current_count = synthesized_count["value"]

                if current_count >= prebuffer_target:
                    prebuffer_ready.set()

            synthesis_queue.put(None)
        except Exception as e:
            producer_error["msg"] = str(e)
            prebuffer_ready.set()
            synthesis_queue.put(("ERROR", str(e), None))

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    try:
        prebuffer_ready.wait()

        if producer_error["msg"] is not None:
            raise RuntimeError(producer_error["msg"])

        if loading_player is not None:
            try:
                loading_player.stop()
            except Exception:
                pass

        log("[INFO] Prebuffer complete, starting playback")

        while True:
            item = synthesis_queue.get()

            if item is None:
                break

            if item[0] == "ERROR":
                raise RuntimeError(item[1])

            idx, sentence, wav_path = item
            log(f"[INFO] Playing sentence {idx}/{total_sentences}")
            play_wav_blocking(wav_path)

    finally:
        stop_event.set()
        producer_thread.join(timeout=1.0)
        if loading_player is not None:
            try:
                loading_player.stop()
            except Exception:
                pass


class LlamaSession:
    def __init__(self):
        reset_outputs()

        check_file(LLAMA_BIN, "llama-cli binary")
        check_file(MODEL_PATH, "GGUF model")
        check_file(PIPER_BIN, "Piper binary")
        check_file(PIPER_MODEL, "Piper model")
        check_file(APLAY_BIN, "aplay binary")

        static_text, dynamic_text = parse_startup_file(STARTUP_FILE)
        self.static_text = static_text
        self.dynamic_text = dynamic_text

        self.child = None

    def start(self) -> None:
        cmd = (
            f'"{LLAMA_BIN}" '
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
            timeout=180,
        )

        self.child.expect(r"\n> ")
        _ = self.child.before

        absorbed_file = prepare_absorbed_startup_file()
        if absorbed_file is not None:
            log(f"[INFO] Absorbing startup context from: {STARTUP_FILE}")
            self.child.sendline(f"/read {absorbed_file}")
            self.child.expect(r"\n> ")
            _ = self.child.before
            log("[INFO] Startup context absorbed")
        else:
            log(f"[INFO] No startup context found in: {STARTUP_FILE}")

        log("[INFO] Session ready")

    def _send_and_capture(self, prompt: str) -> str:
        if not self.child:
            raise RuntimeError("llama session is not running")

        self.child.sendline(prompt)
        self.child.expect(r"\n> ")
        return self.child.before or ""

    def ask(self, prompt: str) -> tuple[str, LoadingSoundPlayer | None]:
        if not self.child:
            raise RuntimeError("llama session is not running")

        prompt = prompt.strip()
        if not prompt:
            return "", None

        log(f"[USER ] {prompt}")

        loading_player = LoadingSoundPlayer(LOADING_SOUND)
        loading_player.start()

        raw = self._send_and_capture(prompt)
        cleaned = clean_response(raw, prompt)

        RAW_FILE.write_text(raw, encoding="utf-8", errors="replace")
        CLEAN_FILE.write_text(cleaned, encoding="utf-8", errors="replace")

        log("[RAW ]")
        log(raw.strip() if raw.strip() else "<EMPTY>")
        log("[CLEAN]")
        log(cleaned if cleaned else "<EMPTY>")

        return cleaned, loading_player

    def ask_silent(self, prompt: str) -> str:
        if not self.child:
            raise RuntimeError("llama session is not running")

        raw = self._send_and_capture(prompt)
        cleaned = clean_response(raw, prompt)
        return cleaned.strip()

    def handle_terminate_and_persist(self, startup_file: Path) -> None:
        log("[INFO] Terminate detected: collecting dynamic memory from SLM")

        username = self.ask_silent("What is the name of the user?")
        learned = self.ask_silent(
            "What have you learned about the user this far according to this conversation?"
        )

        if not username:
            username = "Unknown"
        if not learned:
            learned = "No reliable information learned yet."

        username = strip_control_garbage(username)
        learned = strip_control_garbage(learned)

        username = username.splitlines()[0].strip()
        username = re.split(r"\bUser:\b|\bAssistant:\b", username)[0].strip()
        username = re.sub(r"^\s*The\s+name\s+of\s+the\s+user\s*:\s*", "", username, flags=re.IGNORECASE)
        username = re.sub(r"^\s*The\s+name\s+of\s+the\s+user\s+is\s+", "", username, flags=re.IGNORECASE)
        username = re.sub(r"^\s*Your\s+name\s+is\s+", "", username, flags=re.IGNORECASE)
        username = re.sub(r"^\s*The\s+user'?s\s+name\s+is\s+", "", username, flags=re.IGNORECASE)
        username = re.sub(r"[.>]+$", "", username).strip()
        username = re.sub(r"\s+", " ", username).strip()

        learned = re.sub(r"[>]+$", "", learned).strip()
        learned = re.sub(r"\s+", " ", learned).strip()

        new_dynamic_text = "\n".join([
            f"The name of the User: {username}",
            learned,
        ])

        write_startup_file(startup_file, self.static_text, new_dynamic_text)
        log(f"[INFO] Updated startup file: {startup_file}")
        ensure_and_play_terminated_sound()

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


def is_terminate_command(prompt: str) -> bool:
    normalized = prompt.strip().lower()
    normalized = re.sub(r"[^\w]", "", normalized)
    return normalized == "terminate"


def read_prompt_with_idle_timeout(timeout_sec: int = IDLE_TIMEOUT_SEC) -> str | None:
    """
    Returns:
      - str: the typed prompt (without trailing newline)
      - None: if idle timeout expires before the first typed character
    Behavior:
      - Timer starts when prompt is available.
      - As soon as the first character is typed, the idle timeout is cancelled.
      - Then the user may finish the line normally.
    """
    prompt_label = "\nYou: "
    sys.stdout.write(prompt_label)
    sys.stdout.flush()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    buffer: list[str] = []
    started_typing = False

    try:
        tty.setcbreak(fd)

        while True:
            timeout = None if started_typing else timeout_sec
            ready, _, _ = select.select([sys.stdin], [], [], timeout)

            if not ready:
                sys.stdout.write("\n")
                sys.stdout.flush()
                return None

            ch = sys.stdin.read(1)

            if ch in ("\n", "\r"):
                sys.stdout.write("\n")
                sys.stdout.flush()
                return "".join(buffer).strip()

            if ch in ("\x7f", "\b"):
                if buffer:
                    buffer.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue

            if ch == "\x03":
                raise KeyboardInterrupt

            if ch == "\x04":
                raise EOFError

            if not started_typing:
                started_typing = True

            buffer.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main() -> None:
    reset_outputs()
    check_file(LLAMA_BIN, "llama-cli binary")
    check_file(MODEL_PATH, "GGUF model")
    check_file(PIPER_BIN, "Piper binary")
    check_file(PIPER_MODEL, "Piper model")
    check_file(APLAY_BIN, "aplay binary")

    ensure_and_play_boot_sound()

    startup_loading_player = LoadingSoundPlayer(LOADING_SOUND)
    startup_loading_player.start()

    session = LlamaSession()

    try:
        session.start()

        try:
            startup_loading_player.stop()
        except Exception:
            pass

        ensure_and_play_ready_sound()

        if len(sys.argv) > 1:
            prompt = " ".join(sys.argv[1:]).strip()

            if is_terminate_command(prompt):
                ensure_and_play_terminating_sound()
                session.handle_terminate_and_persist(STARTUP_FILE)
                return

            reply, loading_player = session.ask(prompt)
            if reply and reply.strip():
                log(f"[INFO] Speaking: {reply}")
                speak_text_progressively(reply, loading_player)
            else:
                if loading_player is not None:
                    try:
                        loading_player.stop()
                    except Exception:
                        pass
                log("[WARN] Empty cleaned response")
            return

        while True:
            try:
                prompt = read_prompt_with_idle_timeout(IDLE_TIMEOUT_SEC)
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if prompt is None:
                log(f"[INFO] No input detected for {IDLE_TIMEOUT_SEC} seconds")
                ensure_and_play_terminating_sound()
                session.handle_terminate_and_persist(STARTUP_FILE)
                break

            if not prompt:
                continue

            if prompt.lower() in {"exit", "quit"}:
                break

            if is_terminate_command(prompt):
                ensure_and_play_terminating_sound()
                session.handle_terminate_and_persist(STARTUP_FILE)
                break

            reply, loading_player = session.ask(prompt)
            if reply and reply.strip():
                log(f"[INFO] Speaking: {reply}")
                speak_text_progressively(reply, loading_player)
            else:
                if loading_player is not None:
                    try:
                        loading_player.stop()
                    except Exception:
                        pass
                log("[WARN] Empty cleaned response")

    finally:
        try:
            startup_loading_player.stop()
        except Exception:
            pass
        session.close()


if __name__ == "__main__":
    main()
