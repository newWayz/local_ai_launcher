"""
Parakeet TDT V3 Speech-to-Text backend using Intel OpenVINO NPU acceleration.
Exposes an OpenAI-compatible REST API at /v1/audio/transcriptions.
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
import ctypes
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import pyperclip
except ImportError:
    pyperclip = None

try:
    import openvino as ov
except ImportError:
    ov = None

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("parakeet-npu")

AI_WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
_candidate_ov_dir = AI_WORKSPACE_DIR / "models" / "openvino" / "parakeet-tdt-v3-ov"
DEFAULT_MODEL_DIR = _candidate_ov_dir if _candidate_ov_dir.exists() else (AI_WORKSPACE_DIR / "models" / "parakeet-tdt-v3-ov")
REPO_ID = "FluidInference/parakeet-tdt-0.6b-v3-ov"

BLANK_TOKEN_ID = 8192
SAMPLE_RATE = 16000
CHUNK_SECONDS = 15.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)  # 240,000 samples
MEL_FRAMES_PER_CHUNK = 1501  # Fixed expected shape for mel spectrogram


def ensure_parakeet_models(model_dir: Path) -> bool:
    """Ensure all required Parakeet TDT V3 OpenVINO model files are present."""
    model_dir.mkdir(parents=True, exist_ok=True)
    required_files = [
        "parakeet_encoder.xml",
        "parakeet_encoder.bin",
        "parakeet_decoder.xml",
        "parakeet_decoder.bin",
        "parakeet_joint.xml",
        "parakeet_joint.bin",
        "parakeet_melspectogram.xml",
        "parakeet_melspectogram.bin",
        "parakeet_v3_vocab.json"
    ]

    missing = [f for f in required_files if not (model_dir / f).exists()]
    if not missing:
        return True

    logger.info(f"Missing Parakeet TDT V3 model files: {missing}. Downloading from {REPO_ID}...")
    hf_exe = AI_WORKSPACE_DIR / ".venv" / "Scripts" / "hf.exe"
    if hf_exe.exists():
        import subprocess
        try:
            cmd = [str(hf_exe), "download", REPO_ID, "--local-dir", str(model_dir)]
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            return True
        except Exception as e:
            logger.error(f"Failed to download via hf.exe: {e}")

    try:
        from huggingface_hub import snapshot_download
        logger.info("Downloading via huggingface_hub...")
        snapshot_download(repo_id=REPO_ID, local_dir=str(model_dir))
        return True
    except Exception as e:
        logger.error(f"Failed to download models: {e}")
        return False


class ParakeetTDTEngine:
    """Inference engine for Parakeet TDT V3 on Intel NPU / OpenVINO."""

    def __init__(self, model_dir: Path, target_device: str = "NPU"):
        self.model_dir = Path(model_dir)
        self.target_device = target_device
        self.lock = threading.Lock()

        if ov is None:
            raise RuntimeError("OpenVINO is not installed. Install via `uv pip install openvino`.")

        self.core = ov.Core()
        self.available_devices = self.core.available_devices
        logger.info(f"OpenVINO Core initialized. Available devices: {self.available_devices}")

        # Configure model caching so subsequent launches compile instantly (<500ms)
        cache_dir = self.model_dir / ".cache_npu"
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.core.set_property(self.target_device, {"CACHE_DIR": str(cache_dir)})
            logger.info(f"Configured {self.target_device} model cache at {cache_dir}")
        except Exception as e:
            logger.warning(f"Could not set CACHE_DIR on {self.target_device}: {e}")

        self._load_vocab()
        self._compile_models()

    def _load_vocab(self):
        vocab_path = self.model_dir / "parakeet_v3_vocab.json"
        if not vocab_path.exists():
            vocab_path = self.model_dir / "parakeet_vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found in {self.model_dir}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        # Mapping token_id (int) -> string
        self.id_to_token = {int(k): v for k, v in raw_vocab.items()}
        logger.info(f"Loaded {len(self.id_to_token)} vocabulary tokens.")

    def _compile_models(self):
        logger.info("Compiling OpenVINO models...")

        # 1. Mel Spectrogram (CPU for instant FFT featurization)
        mel_xml = self.model_dir / "parakeet_melspectogram.xml"
        logger.info("Loading melspectogram model on CPU...")
        self.compiled_mel = self.core.compile_model(self.core.read_model(mel_xml), "CPU")

        # 2. FastConformer Encoder on NPU
        enc_device = self.target_device if self.target_device in self.available_devices else "CPU"
        enc_xml = self.model_dir / "parakeet_encoder.xml"
        logger.info(f"Compiling FastConformer encoder to {enc_device}...")
        t0 = time.time()
        try:
            self.compiled_encoder = self.core.compile_model(self.core.read_model(enc_xml), enc_device)
            logger.info(f"Encoder compiled to {enc_device} in {time.time() - t0:.2f}s")
        except Exception as e:
            logger.warning(f"Failed to compile encoder to {enc_device}: {e}. Falling back to CPU...")
            self.compiled_encoder = self.core.compile_model(self.core.read_model(enc_xml), "CPU")

        # 3. Decoder LSTM (CPU for 1-step token evaluations)
        dec_xml = self.model_dir / "parakeet_decoder.xml"
        logger.info("Compiling decoder model on CPU...")
        self.compiled_decoder = self.core.compile_model(self.core.read_model(dec_xml), "CPU")

        # 4. Joint Network (CPU for low-latency argmax pairing)
        joint_xml = self.model_dir / "parakeet_joint.xml"
        logger.info("Compiling joint network on CPU...")
        self.compiled_joint = self.core.compile_model(self.core.read_model(joint_xml), "CPU")

        logger.info("[SUCCESS] Parakeet TDT V3 models compiled and ready for inference.")

    def decode_tokens(self, token_ids: List[int]) -> str:
        """Convert token IDs into normalized text."""
        pieces = []
        for tid in token_ids:
            if tid == BLANK_TOKEN_ID:
                continue
            token_str = self.id_to_token.get(tid, "")
            # Skip special control tokens (e.g. <|startoftranscript|>, <pad>, etc.)
            if token_str.startswith("<|") and token_str.endswith("|>"):
                continue
            if token_str == "<unk>":
                continue
            pieces.append(token_str)

        raw_text = "".join(pieces)
        # SentencePiece underscore ' ' represents word boundary space
        text = raw_text.replace(" ", " ").strip()
        # Clean up double spaces
        while "  " in text:
            text = text.replace("  ", " ")
        return text

    def transcribe_chunk(self, chunk_samples: np.ndarray) -> List[int]:
        """Transcribe a single 15-second audio chunk (16kHz float32, length <= 240,000)."""
        actual_len = len(chunk_samples)
        if actual_len < CHUNK_SAMPLES:
            # Pad with silence to 240,000
            padded = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
            padded[:actual_len] = chunk_samples
            chunk_samples = padded

        sig_input = chunk_samples.reshape(1, CHUNK_SAMPLES).astype(np.float32)
        len_input = np.array([actual_len], dtype=np.int64)

        # 1. Mel Spectrogram
        mel_res = self.compiled_mel([sig_input, len_input])
        mel_features = mel_res[0]  # Shape [1, 128, 1501]

        # 2. Encoder
        mel_len_input = np.array([MEL_FRAMES_PER_CHUNK], dtype=np.int32)
        enc_res = self.compiled_encoder([mel_features, mel_len_input])
        enc_out = enc_res[0]  # Shape [1, 1024, 188]

        # 3. TDT Greedy Transducer Loop
        T_enc = enc_out.shape[2]  # ~188 frames
        # Scale down valid frames if audio was padded
        if actual_len < CHUNK_SAMPLES:
            valid_ratio = actual_len / CHUNK_SAMPLES
            valid_frames = max(1, int(T_enc * valid_ratio))
        else:
            valid_frames = T_enc

        h = np.zeros((2, 1, 640), dtype=np.float32)
        c = np.zeros((2, 1, 640), dtype=np.float32)
        last_token = BLANK_TOKEN_ID

        emitted_tokens = []
        cached_dec_out = None
        cached_for_token = None

        t = 0
        max_steps = valid_frames * 3
        step = 0

        while t < valid_frames and step < max_steps:
            step += 1
            # Run decoder only when token changes
            if cached_for_token != last_token:
                target_arr = np.array([[last_token]], dtype=np.int64)
                dec_res = self.compiled_decoder([target_arr, h, c])
                dec_out = dec_res[0]  # [1, 1, 640]
                h = dec_res[1]        # [2, 1, 640]
                c = dec_res[2]        # [2, 1, 640]
                cached_dec_out = dec_out
                cached_for_token = last_token
            else:
                dec_out = cached_dec_out

            # Extract encoder frame at time t
            enc_frame = enc_out[:, :, t:t+1].transpose(0, 2, 1)  # [1, 1, 1024]

            # Joint network
            joint_res = self.compiled_joint([enc_frame, dec_out])
            logits = joint_res[0][0, 0, 0]  # Shape [8198]

            token_logits = logits[:8193]
            duration_logits = logits[8193:8198]

            token_id = int(np.argmax(token_logits))
            duration = int(np.argmax(duration_logits))  # 0 to 4 frames skip

            if token_id != BLANK_TOKEN_ID:
                emitted_tokens.append(token_id)
                last_token = token_id

            # Advance acoustic frames by predicted duration
            t += duration
            if duration == 0 and token_id == BLANK_TOKEN_ID:
                t += 1

        return emitted_tokens

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe an arbitrary-length 16kHz mono audio float32 array."""
        with self.lock:
            if len(audio_data) == 0:
                return ""

            # If audio is within 1 chunk
            if len(audio_data) <= CHUNK_SAMPLES:
                tokens = self.transcribe_chunk(audio_data)
                return self.decode_tokens(tokens)

            # Slicing into overlapping 15-second chunks (12s stride, 3s overlap)
            stride_samples = int(12.0 * SAMPLE_RATE)
            all_tokens = []
            offset = 0

            while offset < len(audio_data):
                chunk = audio_data[offset:offset + CHUNK_SAMPLES]
                chunk_tokens = self.transcribe_chunk(chunk)
                all_tokens.extend(chunk_tokens)
                offset += stride_samples

            return self.decode_tokens(all_tokens)


def load_audio_from_bytes(file_bytes: bytes) -> np.ndarray:
    """Decode audio bytes to 16kHz mono float32 numpy array."""
    import io
    if sf is not None:
        try:
            with io.BytesIO(file_bytes) as bio:
                data, samplerate = sf.read(bio, dtype="float32")
                # Downmix to mono if stereo
                if len(data.shape) > 1 and data.shape[1] > 1:
                    data = np.mean(data, axis=1)

                # Resample to 16000 Hz if necessary
                if samplerate != SAMPLE_RATE:
                    duration = len(data) / samplerate
                    num_target_samples = int(duration * SAMPLE_RATE)
                    orig_indices = np.linspace(0, len(data) - 1, num=len(data))
                    new_indices = np.linspace(0, len(data) - 1, num=num_target_samples)
                    data = np.interp(new_indices, orig_indices, data).astype(np.float32)

                return data
        except Exception as e:
            logger.warning(f"soundfile failed to decode audio ({e}). Falling back to wave parser.")

    # Fallback: standard wave parser
    import wave
    try:
        with io.BytesIO(file_bytes) as bio:
            with wave.open(bio, "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)

            if sampwidth == 2:
                samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            elif sampwidth == 1:
                samples = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                samples = np.frombuffer(raw_data, dtype=np.float32)

            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)

            if framerate != SAMPLE_RATE:
                orig_indices = np.linspace(0, len(samples) - 1, num=len(samples))
                new_indices = np.linspace(0, len(samples) - 1, num=int(len(samples) * SAMPLE_RATE / framerate))
                samples = np.interp(new_indices, orig_indices, samples).astype(np.float32)

            return samples
    except wave.Error as e:
        logger.error(f"Failed to decode WAV audio with wave parser: {e}")
        raise ValueError(f"Unsupported or corrupted audio format: {e}")


# Global engine instance
engine: Optional[ParakeetTDTEngine] = None
app = FastAPI(title="Parakeet TDT V3 NPU Voice Transcription")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "parakeet-tdt-0.6b-v3",
        "device": getattr(engine, "target_device", "NPU") if engine else "uninitialized",
        "npu_boost": True
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "parakeet-tdt-0.6b-v3",
                "object": "model",
                "owned_by": "FluidInference",
                "permission": []
            },
            {
                "id": "whisper-1",
                "object": "model",
                "owned_by": "OpenAI",
                "permission": []
            }
        ]
    }


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None)
):
    """OpenAI-compatible speech-to-text transcription endpoint."""
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Parakeet TDT V3 engine not initialized")

    try:
        content = await file.read()
        if not content:
            return JSONResponse({"text": ""})

        t0 = time.time()
        audio = load_audio_from_bytes(content)
        audio_dur = len(audio) / SAMPLE_RATE

        text = engine.transcribe(audio)
        latency_ms = (time.time() - t0) * 1000.0
        rtfx = (audio_dur / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

        logger.info(f"Transcribed {audio_dur:.2f}s audio in {latency_ms:.1f}ms ({rtfx:.1f}x RTF) -> '{text}'")
        return JSONResponse({"text": text})
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Bad audio request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


HOTKEY_VK_MAP = {
    "right_alt": 0xA5,     # VK_RMENU
    "ralt": 0xA5,
    "alt_r": 0xA5,
    "right_ctrl": 0xA3,    # VK_RCONTROL
    "rctrl": 0xA3,
    "ctrl_r": 0xA3,
    "f8": 0x77,            # VK_F8
    "f9": 0x78,            # VK_F9
    "f10": 0x79,           # VK_F10
    "f11": 0x7A,           # VK_F11
    "f12": 0x7B,           # VK_F12
    "scroll_lock": 0x91,   # VK_SCROLL
    "hangul": 0x15,        # VK_HANGUL
}


class GlobalHotkeyDictation:
    """
    Background worker that listens for a global hotkey toggle (default: Right Alt).
    - First press: begins microphone recording (plays high-pitched beep).
    - Second press: stops recording (plays low-pitched beep), transcribes on NPU,
      copies to clipboard, and pastes via simulated Ctrl+V into the active window.
    """

    def __init__(
        self,
        engine: "ParakeetTDTEngine",
        hotkey_name: str = "right_alt",
        add_trailing_space: bool = True
    ):
        self.engine = engine
        self.hotkey_name = hotkey_name.lower().strip()
        self.vk_code = HOTKEY_VK_MAP.get(self.hotkey_name, 0xA5)
        self.add_trailing_space = add_trailing_space

        self.is_recording = False
        self.is_processing = False
        self.start_time = 0.0
        self._last_toggle_time = 0.0
        self._state_lock = threading.Lock()

        self.audio_chunks: List[np.ndarray] = []
        self.stream = None
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

    def start(self):
        if sys.platform != "win32":
            logger.info("[Dictation] Global hotkey dictation is only supported on Windows.")
            return

        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._poll_hotkey, daemon=True, name="DictationHotkeyListener")
        self.worker_thread.start()
        logger.info(f"[Dictation] Global dictation toggle active on [{self.hotkey_name.upper()}]. Press to record, press again to stop & paste.")

    def stop(self):
        self.stop_event.set()
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        logger.info("[Dictation] Global dictation listener stopped.")

    def _play_beep(self, freq: int, dur: int):
        def _beep():
            try:
                import winsound
                winsound.Beep(freq, dur)
            except Exception:
                pass
        threading.Thread(target=_beep, daemon=True).start()

    def _poll_hotkey(self):
        was_pressed = False
        user32 = ctypes.windll.user32

        while not self.stop_event.is_set():
            try:
                state = user32.GetAsyncKeyState(self.vk_code)
                is_pressed = bool(state & 0x8000)

                if is_pressed and not was_pressed:
                    self._handle_toggle()

                was_pressed = is_pressed

                # Auto-stop safety timeout: max 120s per recording session
                if self.is_recording and (time.time() - self.start_time > 120.0):
                    logger.warning("[Dictation] Max recording duration limit (120s) reached. Stopping recording.")
                    self._handle_toggle()

            except Exception as e:
                logger.debug(f"[Dictation] Hotkey polling loop exception: {e}")

            time.sleep(0.02)

    def _handle_toggle(self):
        now = time.time()
        # Debounce rapid accidental double-presses within 300ms
        if now - self._last_toggle_time < 0.3:
            return
        self._last_toggle_time = now

        with self._state_lock:
            if self.is_processing:
                logger.info("[Dictation] Busy transcribing previous speech. Please wait a moment.")
                return

            if not self.is_recording:
                self._start_recording()
            else:
                self._stop_recording()

    def _start_recording(self):
        self.is_recording = True
        self.audio_chunks = []
        self.start_time = time.time()
        self._play_beep(1200, 80)

        try:
            import sounddevice as sd

            def audio_callback(indata, frames, time_info, status):
                if self.is_recording:
                    self.audio_chunks.append(indata.copy())

            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=audio_callback
            )
            self.stream.start()
            logger.info(f"[Dictation] Listening... (Speak now, press [{self.hotkey_name.upper()}] to stop and paste)")
        except Exception as e:
            logger.error(f"[Dictation] Failed to open microphone: {e}")
            self.is_recording = False
            if self.stream:
                try:
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None

    def _stop_recording(self):
        self.is_recording = False
        self.is_processing = True
        self._play_beep(800, 80)

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.debug(f"[Dictation] Error closing audio stream: {e}")
            self.stream = None

        chunks = list(self.audio_chunks)
        self.audio_chunks = []
        logger.info("[Dictation] Audio captured. Transcribing on NPU...")

        threading.Thread(
            target=self._process_and_paste,
            args=(chunks,),
            daemon=True,
            name="DictationProcessor"
        ).start()

    def _process_and_paste(self, chunks: list):
        try:
            if not chunks:
                logger.info("[Dictation] No audio frames captured.")
                return

            audio = np.concatenate(chunks, axis=0).flatten()
            duration = len(audio) / SAMPLE_RATE

            if duration < 0.25:
                logger.info(f"[Dictation] Audio too short ({duration:.2f}s). Discarded.")
                return

            peak = float(np.max(np.abs(audio)))
            if peak < 0.003:
                logger.info(f"[Dictation] Silence detected (peak={peak:.4f}). Discarded.")
                return

            t0 = time.time()
            text = self.engine.transcribe(audio).strip()
            elapsed_ms = (time.time() - t0) * 1000.0

            if not text:
                logger.info(f"[Dictation] Empty transcription ({elapsed_ms:.1f}ms).")
                return

            logger.info(f"[Dictation] Transcribed in {elapsed_ms:.1f}ms: \"{text}\"")

            self._paste_text(text)

        except Exception as e:
            logger.error(f"[Dictation] Transcription/paste error: {e}", exc_info=True)
        finally:
            with self._state_lock:
                self.is_processing = False

    def _paste_text(self, text: str):
        try:
            import pyperclip
            text_to_paste = (text + " ") if (self.add_trailing_space and not text.endswith(" ")) else text
            pyperclip.copy(text_to_paste)

            user32 = ctypes.windll.user32

            # Wait until the hotkey is physically released to avoid modifier interference (e.g. Ctrl+Alt+V)
            t_release_wait = time.time()
            while (user32.GetAsyncKeyState(self.vk_code) & 0x8000) and (time.time() - t_release_wait < 1.0):
                time.sleep(0.02)
            time.sleep(0.05)

            # Ensure any Alt key state is released
            KEYEVENTF_KEYUP = 0x0002
            user32.keybd_event(0x12, 0, KEYEVENTF_KEYUP, 0)  # VK_MENU
            time.sleep(0.02)

            # Simulate Ctrl + V
            VK_CONTROL = 0x11
            VK_V = 0x56
            user32.keybd_event(VK_CONTROL, 0, 0, 0)
            user32.keybd_event(VK_V, 0, 0, 0)
            time.sleep(0.02)
            user32.keybd_event(VK_V, 0, KEYEVENTF_KEYUP, 0)
            user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)

            logger.info(f"[Dictation] Pasted \"{text_to_paste}\" into active window.")
        except Exception as e:
            logger.error(f"[Dictation] Auto-paste failed: {e}")


def run_test_compilation(model_dir: Path, device: str = "NPU"):
    """Quick verification routine to compile and test the model."""
    logger.info(f"--- Running Parakeet TDT V3 test compilation on {device} ---")
    if not ensure_parakeet_models(model_dir):
        logger.error("Model download/verification failed.")
        sys.exit(1)

    eng = ParakeetTDTEngine(model_dir, target_device=device)
    # Test with 1 second of synthetic silence
    test_audio = np.zeros(16000, dtype=np.float32)
    t0 = time.time()
    res = eng.transcribe(test_audio)
    logger.info(f"Test transcription completed in {(time.time() - t0) * 1000:.1f}ms. Result: '{res}'")
    logger.info("Test compilation SUCCESSFUL!")


def main():
    parser = argparse.ArgumentParser(description="Parakeet TDT V3 NPU Transcription Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default 127.0.0.1)")
    parser.add_argument("--device", type=str, default="NPU", help="Target OpenVINO device: NPU, CPU, AUTO")
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR), help="Path to parakeet model directory")
    parser.add_argument("--test-compile", action="store_true", help="Run self-test compilation and exit")
    parser.add_argument("--hotkey", type=str, default="right_alt", help="Global toggle hotkey for dictation (default: right_alt, 'none' to disable)")
    parser.add_argument("--no-trailing-space", action="store_true", help="Disable adding a trailing space after pasted text")
    args = parser.parse_args()

    model_path = Path(args.model_dir)

    if args.test_compile:
        run_test_compilation(model_path, device=args.device)
        return

    port = args.port
    try:
        p = int(port)
        if not (1 <= p <= 65535):
            logger.warning(f"Invalid port {port}, defaulting to 8000")
            port = 8000
        else:
            port = p
    except (ValueError, TypeError):
        port = 8000

    logger.info(f"Starting Parakeet TDT V3 NPU Server on {args.host}:{port} (device: {args.device})...")
    ensure_parakeet_models(model_path)

    global engine
    engine = ParakeetTDTEngine(model_path, target_device=args.device)

    hotkey_worker = None
    if args.hotkey and str(args.hotkey).strip().lower() not in ("none", "false", "off", "0", "-1", "disabled", "no"):
        hotkey_worker = GlobalHotkeyDictation(
            engine=engine,
            hotkey_name=args.hotkey,
            add_trailing_space=not args.no_trailing_space
        )
        hotkey_worker.start()

    try:
        uvicorn.run(app, host=args.host, port=port, log_level="info")
    finally:
        if hotkey_worker:
            hotkey_worker.stop()


if __name__ == "__main__":
    main()
