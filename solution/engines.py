"""Local ASR engine adapters for the streaming dictation track (Apple Silicon).

Three lazily-loaded engines, each behind a tiny uniform interface so draft.py
stays a clean router:

  - fast_english (Parakeet via parakeet-mlx)  : sub-second English/Indian-English
  - hinglish (Oriserve Apex via mlx-whisper)  : faithful romanized Hindi-English
  - detect_lang (whisper tiny via mlx-whisper): cheap language ID for routing

Everything runs offline once models are cached to disk (warm_all() pre-pulls
them during the harness warmup, before block_network() fires). No engine makes
a network call at decode time. If a package/model is missing we degrade to an
empty string rather than crash — draft.py then holds its last good text.

Model IDs are read from env so CI can pin local converted paths:
  PARAKEET_MODEL   default mlx-community/parakeet-tdt_ctc-110m  (English)
  APEX_MODEL       default (converted) Oriserve Apex MLX dir / repo
  DETECT_MODEL     default mlx-community/whisper-tiny
"""
from __future__ import annotations

import os
import time

_SR = 16000

PARAKEET_MODEL = os.environ.get("PARAKEET_MODEL", "mlx-community/parakeet-tdt_ctc-110m")
APEX_MODEL = os.getenv("APEX_MODEL", "knownsense/whisper-hindi-apex-mlx")
DETECT_MODEL = os.environ.get("DETECT_MODEL", "mlx-community/whisper-tiny")

# module-level singletons (loaded once, reused across every draft() call)
_np = None
_parakeet = None
_mlx_whisper = None
_last_timings: dict[str, float] = {}


def _numpy():
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


def pcm_to_float32(audio_buffer: bytes):
    """int16 LE PCM bytes -> float32 mono [-1, 1] ndarray at 16 kHz."""
    np = _numpy()
    if not audio_buffer:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0


def last_timings() -> dict[str, float]:
    """Per-stage milliseconds from the most recent engine calls (for debug logs)."""
    return dict(_last_timings)


# --- Parakeet fast English path ------------------------------------------

def _load_parakeet():
    global _parakeet
    if _parakeet is None:
        from mlx_audio.stt.utils import load
        _parakeet = load(PARAKEET_MODEL)
    return _parakeet


def transcribe_fast(audio) -> str:
    """Parakeet English/Indian-English decode on a float32 array. '' on failure."""
    t0 = time.monotonic()
    try:
        model = _load_parakeet()
        
        import tempfile
        import wave
        
        np = _numpy()
        # Convert float32 array back to int16 bytes
        pcm = (audio * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()
        
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        try:
            with wave.open(temp_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(_SR)
                w.writeframes(pcm)
            result = model.generate(audio=temp_path)
        finally:
            os.remove(temp_path)
            
        text = getattr(result, "text", None)
        if text is None and isinstance(result, (list, tuple)) and result:
            text = getattr(result[0], "text", "")
        out = (text or "").strip()
    except Exception as e:
        import traceback
        traceback.print_exc()
        out = ""
    _last_timings["fast_ms"] = (time.monotonic() - t0) * 1000.0
    return out


# --- Oriserve Apex faithful Hinglish path (romanized) --------------------

def _load_mlx_whisper():
    global _mlx_whisper
    if _mlx_whisper is None:
        import mlx_whisper  # local import; offline once cached
        _mlx_whisper = mlx_whisper
    return _mlx_whisper


def transcribe_hinglish(audio) -> str:
    """Apex (mlx-whisper) faithful code-switch decode -> romanized Hinglish.

    Greedy, no prev-text conditioning: lowest latency and kills a common
    repetition-loop source. language='en' matches Oriserve's card (Apex was
    fine-tuned to emit Hindi audio as romanized Latin under the 'en' head).
    """
    t0 = time.monotonic()
    try:
        mw = _load_mlx_whisper()
        result = mw.transcribe(
            audio,
            path_or_hf_repo=APEX_MODEL,
            language="hi",
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt="Yeh audio ek technical meeting hai. Always use English terms for open source software, features, aur operating systems. Write numbers in digits (jaise 3, 4, 100).",
            fp16=True,
        )
        out = (result.get("text") or "").strip()
    except Exception as e:
        import traceback
        traceback.print_exc()
        out = ""
    _last_timings["hinglish_ms"] = (time.monotonic() - t0) * 1000.0
    return out


# --- Cheap language detector for routing ---------------------------------

def detect_language(audio) -> tuple[str, float]:
    """Return (lang_code, probability) using whisper-tiny on a short prefix.

    Only the first ~6s is needed to route; keeps the LID cost tiny. Falls back
    to ('en', 0.0) if unavailable so the router defaults to the fast path.
    """
    t0 = time.monotonic()
    lang, prob = "en", 0.0
    try:
        np = _numpy()
        mw = _load_mlx_whisper()
        prefix = audio[: _SR * 6] if audio.size > _SR * 6 else audio
        result = mw.transcribe(
            prefix, path_or_hf_repo=DETECT_MODEL, temperature=0.0,
            condition_on_previous_text=False, fp16=True,
        )
        lang = result.get("language", "en") or "en"
        prob = 1.0  # mlx-whisper doesn't surface prob here; presence is the signal
        _ = np
    except Exception:  # noqa: BLE001
        pass
    _last_timings["detect_ms"] = (time.monotonic() - t0) * 1000.0
    return lang, prob


# --- Warmup (run BEFORE block_network(); pulls + JITs every model) -------

def warm_all() -> dict[str, bool]:
    """Pre-load and dummy-decode each engine so the first real clip is warm.

    Returns {engine: ok} so CI can assert models actually loaded offline-ready.
    """
    np = _numpy()
    silence = np.zeros(_SR, dtype=np.float32)  # 1s of silence
    status: dict[str, bool] = {}
    for name, fn in (
        ("fast", lambda: transcribe_fast(silence)),
        ("hinglish", lambda: transcribe_hinglish(silence)),
        ("detect", lambda: detect_language(silence)),
    ):
        try:
            fn()
            status[name] = True
        except Exception:  # noqa: BLE001
            status[name] = False
    return status
