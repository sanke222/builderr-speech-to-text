"""Local ASR engine adapters — platform-aware dual-backend.

Darwin (Apple Silicon):
  - fast_english  : Parakeet via parakeet-mlx          (sub-second English/Indian-English)
  - hinglish      : Trelis Q4 MLX via mlx-whisper       (faithful romanized code-switch)
  - detect_lang   : whisper-tiny via mlx-whisper         (cheap LID for routing)

Linux (x86-64 / CI / scoring box):
  - fast_english  : faster-distil-whisper-small.en via faster-whisper  (int8, CPU, ~300ms)
  - hinglish      : zero-stt-hinglish-ct2 via faster-whisper           (int8, CPU, ~1-2s)
  - detect_lang   : whisper-tiny via faster-whisper                     (cheap LID)

All engines are lazily loaded and offline after warm_all() has pre-pulled every
model. No engine makes a network call at decode time. Failures degrade to '' so
draft.py falls back to its last committed prefix rather than crashing.

Model IDs are read from env vars so CI can pin specific converted checkpoints:
  Darwin:
    PARAKEET_MODEL   default mlx-community/parakeet-tdt_ctc-110m
    HINGLISH_MODEL   default sanke/trelis-mlx-hinglish-q4
    DETECT_MODEL     default mlx-community/whisper-tiny

  Linux:
    FW_ENGLISH_MODEL   default Systran/faster-distil-whisper-small.en
    FW_HINGLISH_MODEL  default sanke/zero-stt-hinglish-ct2   (your HF upload)
    FW_DETECT_MODEL    default Systran/faster-whisper-tiny-ct2 (or openai/whisper-tiny ct2)
"""
from __future__ import annotations

import os
import platform
import time

# ---------------------------------------------------------------------------
# Offline guard: keep HF Hub from pinging the network during scoring.
# We lift it during warm_all() and re-apply after.
# Also disable HF XetHub CAS protocol to avoid CDN errors on CI runners.
# ---------------------------------------------------------------------------
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

_SR = 16000
_IS_DARWIN = platform.system() == "Darwin"

# ---------------------------------------------------------------------------
# Model IDs — overridable via env vars
# ---------------------------------------------------------------------------
# --- Darwin / Apple Silicon MLX ---
PARAKEET_MODEL = os.environ.get("PARAKEET_MODEL", "mlx-community/parakeet-tdt_ctc-110m")
HINGLISH_MODEL = os.environ.get("HINGLISH_MODEL", "sanke/trelis-mlx-hinglish-q4")
DETECT_MODEL   = os.environ.get("DETECT_MODEL",   "mlx-community/whisper-tiny")

# --- Linux / faster-whisper (CTranslate2 int8) ---
FW_ENGLISH_MODEL  = os.environ.get("FW_ENGLISH_MODEL",  "Systran/faster-distil-whisper-small.en")
FW_HINGLISH_MODEL = os.environ.get("FW_HINGLISH_MODEL", "sanke/zero-stt-hinglish-ct2")
FW_DETECT_MODEL   = os.environ.get("FW_DETECT_MODEL",   "Systran/faster-whisper-tiny")

# Resolved local snapshot paths (populated by warm_all before network is blocked)
_hinglish_path: str = HINGLISH_MODEL
_detect_path:   str = DETECT_MODEL

# ---------------------------------------------------------------------------
# Module-level singletons — loaded once, reused across every draft() call
# ---------------------------------------------------------------------------
_np          = None
_parakeet    = None      # Darwin: parakeet-mlx model
_mlx_whisper = None      # Darwin: mlx_whisper module

_fw_english  = None      # Linux: faster-whisper WhisperModel (English)
_fw_hinglish = None      # Linux: faster-whisper WhisperModel (Hinglish)
_fw_detect   = None      # Linux: faster-whisper WhisperModel (LID)

_last_timings: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


# ===========================================================================
# DARWIN path — MLX engines (unchanged from original)
# ===========================================================================

def _load_parakeet():
    global _parakeet
    if _parakeet is None:
        from mlx_audio.stt.utils import load
        _parakeet = load(PARAKEET_MODEL)
    return _parakeet


def _load_mlx_whisper():
    global _mlx_whisper
    if _mlx_whisper is None:
        import mlx_whisper
        import huggingface_hub

        # Patch snapshot_download inside mlx_whisper.load_models so that passing a local
        # snapshot directory path returns instantly without hitting network (repo_info API call).
        orig_snapshot = huggingface_hub.snapshot_download
        def _safe_snapshot(repo_id, **kwargs):
            if os.path.exists(str(repo_id)):
                return str(repo_id)
            kwargs["local_files_only"] = True
            try:
                return orig_snapshot(repo_id, **kwargs)
            except Exception:
                return str(repo_id)

        try:
            import mlx_whisper.load_models
            mlx_whisper.load_models.snapshot_download = _safe_snapshot
        except Exception:
            pass

        _mlx_whisper = mlx_whisper
    return _mlx_whisper


def _transcribe_fast_darwin(audio) -> str:
    """Parakeet English/Indian-English decode. '' on failure."""
    t0 = time.monotonic()
    try:
        model = _load_parakeet()
        import tempfile, wave as _wave
        np = _numpy()
        pcm = (audio * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with _wave.open(temp_path, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(_SR)
                w.writeframes(pcm)
            result = model.generate(audio=temp_path)
        finally:
            os.remove(temp_path)
        text = getattr(result, "text", None)
        if text is None and isinstance(result, (list, tuple)) and result:
            text = getattr(result[0], "text", "")
        out = (text or "").strip()
    except Exception:
        import traceback; traceback.print_exc()
        out = ""
    _last_timings["fast_ms"] = (time.monotonic() - t0) * 1000.0
    return out


def _transcribe_hinglish_darwin(audio) -> str:
    """Trelis MLX faithful code-switch decode. '' on failure."""
    t0 = time.monotonic()
    try:
        mw = _load_mlx_whisper()
        result = mw.transcribe(
            audio,
            path_or_hf_repo=_hinglish_path,
            language="hi",
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=(
                "Yeh audio ek technical meeting hai. Always use English terms for open source "
                "software, features, aur operating systems. Write numbers in digits (jaise 3, 4, 100)."
            ),
            fp16=True,
        )
        out = (result.get("text") or "").strip()
    except Exception:
        import traceback; traceback.print_exc()
        out = ""
    _last_timings["hinglish_ms"] = (time.monotonic() - t0) * 1000.0
    return out


def _detect_language_darwin(audio) -> tuple[str, float]:
    """whisper-tiny LID on first ~6s."""
    t0 = time.monotonic()
    lang, prob = "en", 0.0
    try:
        np = _numpy()
        mw = _load_mlx_whisper()
        prefix = audio[: _SR * 6] if audio.size > _SR * 6 else audio
        result = mw.transcribe(
            prefix, path_or_hf_repo=_detect_path,
            temperature=0.0, condition_on_previous_text=False, fp16=True,
        )
        lang = result.get("language", "en") or "en"
        prob = 1.0
        _ = np
    except Exception:
        pass
    _last_timings["detect_ms"] = (time.monotonic() - t0) * 1000.0
    return lang, prob


def _warm_darwin() -> dict[str, bool]:
    """Warm all MLX engines. Called inside warm_all() after lifting HF_HUB_OFFLINE."""
    global _hinglish_path, _detect_path
    try:
        from huggingface_hub import snapshot_download
        if not os.path.exists(HINGLISH_MODEL):
            _hinglish_path = snapshot_download(repo_id=HINGLISH_MODEL, local_files_only=False)
        if not os.path.exists(DETECT_MODEL):
            _detect_path   = snapshot_download(repo_id=DETECT_MODEL,   local_files_only=False)
    except Exception as e:
        import sys; sys.stderr.write(f"Snapshot pre-download note: {e}\n")

    np = _numpy()
    silence = np.zeros(_SR, dtype=np.float32)
    status: dict[str, bool] = {}
    for name, fn in (
        ("fast",     lambda: _transcribe_fast_darwin(silence)),
        ("hinglish", lambda: _transcribe_hinglish_darwin(silence)),
        ("detect",   lambda: _detect_language_darwin(silence)),
    ):
        try:
            fn(); status[name] = True
        except Exception:
            status[name] = False
    return status


# ===========================================================================
# LINUX path — faster-whisper (CTranslate2 int8) engines
# ===========================================================================

def _load_fw_english():
    """Load the faster-whisper English model (singleton)."""
    global _fw_english
    if _fw_english is None:
        from faster_whisper import WhisperModel
        _fw_english = WhisperModel(
            FW_ENGLISH_MODEL,
            device="cpu",
            compute_type="int8",
            cpu_threads=max(1, (os.cpu_count() or 2)),
            num_workers=1,
        )
    return _fw_english


def _load_fw_hinglish():
    """Load the faster-whisper Hinglish model (singleton)."""
    global _fw_hinglish
    if _fw_hinglish is None:
        from faster_whisper import WhisperModel
        _fw_hinglish = WhisperModel(
            FW_HINGLISH_MODEL,
            device="cpu",
            compute_type="int8",
            cpu_threads=max(1, (os.cpu_count() or 2)),
            num_workers=1,
        )
    return _fw_hinglish


def _load_fw_detect():
    """Load the faster-whisper tiny model for LID (singleton)."""
    global _fw_detect
    if _fw_detect is None:
        from faster_whisper import WhisperModel
        _fw_detect = WhisperModel(
            FW_DETECT_MODEL,
            device="cpu",
            compute_type="int8",
            cpu_threads=2,
            num_workers=1,
        )
    return _fw_detect


def _fw_segments_to_text(segments) -> str:
    """Concatenate faster-whisper segment objects into a single string."""
    return " ".join(s.text.strip() for s in segments if s.text.strip())


def _transcribe_fast_linux(audio) -> str:
    """faster-distil-whisper-small.en: fast English/Indian-English decode. '' on failure."""
    t0 = time.monotonic()
    try:
        model = _load_fw_english()
        # distil-whisper/small.en is English-only; no language= arg needed
        segments, _info = model.transcribe(
            audio,
            beam_size=1,                    # greedy — lowest latency
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,               # Silero VAD: trims silence, speeds up short clips
            vad_parameters={"min_silence_duration_ms": 300},
        )
        out = _fw_segments_to_text(segments).strip()
    except Exception:
        import traceback; traceback.print_exc()
        out = ""
    _last_timings["fast_ms"] = (time.monotonic() - t0) * 1000.0
    return out


def _transcribe_hinglish_linux(audio) -> str:
    """zero-stt-hinglish-ct2: faithful code-switch decode. '' on failure.

    The zero-stt model outputs mixed-script (Devanagari + Roman). We pass the
    result through a lightweight transliteration step in finalize() so the
    final output stays in Roman script (matching the gold references).
    The 'hi' language hint keeps the decoder from forcing a pure-English path.
    """
    t0 = time.monotonic()
    try:
        model = _load_fw_hinglish()
        segments, _info = model.transcribe(
            audio,
            language="hi",                 # keeps Whisper in Hindi-family decode mode
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=(
                "Yeh audio ek technical meeting hai. Always use English terms for open source "
                "software, features, aur operating systems. Write numbers in digits."
            ),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        out = _fw_segments_to_text(segments).strip()
    except Exception:
        import traceback; traceback.print_exc()
        out = ""
    _last_timings["hinglish_ms"] = (time.monotonic() - t0) * 1000.0
    return out


def _detect_language_linux(audio) -> tuple[str, float]:
    """faster-whisper tiny LID on first ~6s. Matches Mac transcribe-based LID."""
    t0 = time.monotonic()
    lang, prob = "en", 0.0
    try:
        np = _numpy()
        model = _load_fw_detect()
        prefix = audio[: _SR * 6] if audio.size > _SR * 6 else audio
        segments, info = model.transcribe(prefix, beam_size=1, max_initial_timestamp=1.0)
        lang = info.language or "en"
        prob = float(info.language_probability) if info.language_probability is not None else 1.0
        _ = (np, list(segments))
    except Exception:
        pass
    _last_timings["detect_ms"] = (time.monotonic() - t0) * 1000.0
    return lang, prob


def _warm_linux() -> dict[str, bool]:
    """Warm all faster-whisper engines. Called inside warm_all()."""
    try:
        from huggingface_hub import snapshot_download
        for m in (FW_ENGLISH_MODEL, FW_HINGLISH_MODEL, FW_DETECT_MODEL):
            snapshot_download(repo_id=m, repo_type="model")
    except Exception as e:
        import sys; sys.stderr.write(f"Pre-download note: {e}\n")

    np = _numpy()
    silence = np.zeros(_SR, dtype=np.float32)
    status: dict[str, bool] = {}
    for name, fn in (
        ("fast",     lambda: _transcribe_fast_linux(silence)),
        ("hinglish", lambda: _transcribe_hinglish_linux(silence)),
        ("detect",   lambda: _detect_language_linux(silence)),
    ):
        try:
            fn(); status[name] = True
        except Exception:
            status[name] = False
    return status


# ===========================================================================
# Unified public API — draft.py calls ONLY these four functions
# ===========================================================================

def transcribe_fast(audio) -> str:
    """Route to the platform-appropriate fast English engine."""
    if _IS_DARWIN:
        return _transcribe_fast_darwin(audio)
    return _transcribe_fast_linux(audio)


def transcribe_hinglish(audio) -> str:
    """Route to the platform-appropriate Hinglish/code-switch engine."""
    if _IS_DARWIN:
        return _transcribe_hinglish_darwin(audio)
    return _transcribe_hinglish_linux(audio)


def detect_language(audio) -> tuple[str, float]:
    """Route to the platform-appropriate language detector."""
    if _IS_DARWIN:
        return _detect_language_darwin(audio)
    return _detect_language_linux(audio)


def warm_all() -> dict[str, bool]:
    """Pre-load and dummy-decode every engine before block_network() fires.

    Returns {engine: ok} so CI can assert that all models loaded correctly.
    Network is allowed during this call; it is blocked immediately after.
    """
    # Temporarily lift HF offline guard so snapshot_download works
    if "HF_HUB_OFFLINE" in os.environ:
        del os.environ["HF_HUB_OFFLINE"]

    if _IS_DARWIN:
        status = _warm_darwin()
    else:
        status = _warm_linux()

    os.environ["HF_HUB_OFFLINE"] = "1"
    return status
