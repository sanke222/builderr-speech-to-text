"""Batch transcription contract for the builderr local-dictation challenge.

    python -m solution.transcribe --input clip.wav --mode auto --output result.json

Platform-aware dual-lane engine:
  Darwin (Apple Silicon):
    fast/auto  → Parakeet-110m via MLX          (sub-second English/Indian-English)
    hinglish   → Trelis Q4 via mlx-whisper       (faithful romanized code-switch)

  Linux (x86-64):
    fast/auto  → faster-distil-whisper-small.en  (int8 CTranslate2)
    hinglish   → zero-stt-hinglish-ct2           (int8 CTranslate2)

  auto mode → whisper-tiny LID routes to the best lane automatically.
  verbatim  → hinglish engine with no finalization (raw faithful transcript).

All models run offline once cached. warm_all() must be called before
block_network() fires. No hardcoded phrase fixes.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import soundfile as sf

from solution import engines
from solution.finalizer import finalize


def _load_audio(wav_path: str) -> np.ndarray:
    """Load a WAV file and return float32 mono 16 kHz."""
    audio, sr = sf.read(wav_path, dtype="float32")
    # stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # resample to 16 kHz if needed
    if sr != 16000:
        import soxr  # lazy; only needed for non-16k files
        audio = soxr.resample(audio, sr, 16000)
    return audio


def transcribe(wav_path: str, mode: str = "auto") -> dict:
    t0 = time.time()

    audio = _load_audio(wav_path)
    candidates: list[dict] = []
    model_ids: list[str] = []
    timings: dict[str, float] = {}

    # --- routing ---
    if mode == "fast":
        route = "fast"
        route_reason = "mode=fast"
        lang = "en"
    elif mode == "hinglish":
        route = "hinglish"
        route_reason = "mode=hinglish"
        lang = "hi"
    elif mode == "verbatim":
        route = "hinglish"
        route_reason = "mode=verbatim"
        lang = "hi"
    else:
        # auto: use LID to decide
        lang, _prob = engines.detect_language(audio)
        timings["detect_ms"] = engines.last_timings().get("detect_ms", 0)
        if lang in {"hi", "ur", "mr", "ne", "sa"}:
            route = "hinglish"
            route_reason = f"auto:lang={lang}"
        else:
            route = "fast"
            route_reason = f"auto:lang={lang}"

    # --- decode on the chosen lane, with cross-lane fallback ---
    asr_t0 = time.time()

    import platform as _plat
    _darwin = _plat.system() == "Darwin"
    _fast_id    = engines.PARAKEET_MODEL   if _darwin else engines.FW_ENGLISH_MODEL
    _hing_id    = engines.HINGLISH_MODEL   if _darwin else engines.FW_HINGLISH_MODEL
    _fast_name  = "parakeet-fast"          if _darwin else "fw-distil-small-en"
    _hing_name  = "trelis-hinglish"        if _darwin else "fw-zero-stt-hinglish"

    if route == "hinglish":
        text = engines.transcribe_hinglish(audio)
        timings["hinglish_ms"] = engines.last_timings().get("hinglish_ms", 0)
        model_ids.append(_hing_id)
        candidates.append({"engine": _hing_name, "text": text})

        if not text:
            # fallback to fast lane
            text = engines.transcribe_fast(audio)
            timings["fast_ms"] = engines.last_timings().get("fast_ms", 0)
            model_ids.append(_fast_id)
            candidates.append({"engine": f"{_fast_name}-fallback", "text": text})
    else:
        text = engines.transcribe_fast(audio)
        timings["fast_ms"] = engines.last_timings().get("fast_ms", 0)
        model_ids.append(_fast_id)
        candidates.append({"engine": _fast_name, "text": text})

        if not text:
            # fallback to hinglish lane
            text = engines.transcribe_hinglish(audio)
            timings["hinglish_ms"] = engines.last_timings().get("hinglish_ms", 0)
            model_ids.append(_hing_id)
            candidates.append({"engine": f"{_hing_name}-fallback", "text": text})

    asr_ms = (time.time() - asr_t0) * 1000

    # --- finalize (digits, loop guard, negation-safe) ---
    pp_t0 = time.time()
    if mode == "verbatim":
        final_text = text or ""
    else:
        final_text = finalize(text or "")
    pp_ms = (time.time() - pp_t0) * 1000

    total_ms = (time.time() - t0) * 1000
    timings.update({"total": round(total_ms), "asr": round(asr_ms), "postprocess": round(pp_ms)})

    return {
        "text": final_text,
        "mode_used": mode,
        "language_guess": lang,
        "timings_ms": timings,
        "raw_candidates": candidates,
        "model_ids": model_ids,
        "local_only": True,
        "route": route,
        "route_reason": route_reason,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="auto", choices=["auto", "fast", "hinglish", "verbatim"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = transcribe(args.input, args.mode)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.output}  ({result['timings_ms']['total']}ms, "
          f"route={result['route']}, local_only={result['local_only']})")


if __name__ == "__main__":
    main()
