"""The ONE function the STREAMING dictation track scores: draft().

    draft(audio_buffer, is_final) -> (text_so_far, stable_chars)

Architecture (matches the reference-bot shape, built to beat it):

  1. Fast lane  — Parakeet (parakeet-mlx) for English / Indian-English. Cheap,
     sub-second, runs on the rolling buffer.
  2. Router     — whisper-tiny language ID on the prefix decides plain-English
     (keep fast lane) vs Hindi-mixed (escalate). We only pay for the heavy model
     when the clip needs it, so English clips stay fast.
  3. Hinglish   — Oriserve Apex (mlx-whisper) faithful romanized code-switch,
     ONLY on escalated / final clips.
  4. Finalizer  — faithful cleanup (digits, negation kept, loop guard). Never
     translates the mix; never blanks.

Latency trick: partials aren't scored, so we use them as free compute. During
speech we decode the rolling buffer on the fast lane and commit a stable common
prefix. On is_final we produce the faithful final; end-to-final latency is
measured from the last frame, so keeping the heavy decode scoped to the tail /
final keeps the paste fast.
"""
from __future__ import annotations

import re
import threading
import time

from solution import engines
from solution.finalizer import finalize

_SR = 16000
_MIN_AUDIO_BYTES = int(_SR * 0.75) * 2  # ~0.75s before the first draft

# per-clip state (harness calls draft_reset() between clips)
_prev_text: str = ""
_committed: str = ""
_route: str = "fast"        # 'fast' | 'hinglish', decided once per clip
_route_reason: str = ""
_lang: str = ""
_finalized_s: float = 0.0
_chunk_future = None
_chunk_lock = threading.Lock()


def _join(a: str, b: str) -> str:
    if not a: return b
    if not b: return a
    return a + " " + b


def _bg_decode(audio, route):
    if route == "hinglish":
        text = engines.transcribe_hinglish(audio) or engines.transcribe_fast(audio)
    else:
        import numpy as np
        padded_audio = np.concatenate([audio, np.zeros(int(_SR * 0.5), dtype=np.float32)])
        text = engines.transcribe_fast(padded_audio) or engines.transcribe_hinglish(audio)
    return finalize(text)


def draft_reset() -> None:
    """Called by the sealed harness at the start of each clip. Clear state."""
    global _prev_text, _committed, _route, _route_reason, _lang, _finalized_s, _chunk_future
    _prev_text = ""
    _committed = ""
    _route = "fast"
    _route_reason = ""
    _lang = ""
    _finalized_s = 0.0
    _chunk_future = None


def route_debug() -> dict:
    """Expose the last routing decision for the debug harness (unscored)."""
    return {"route": _route, "reason": _route_reason, "lang": _lang,
            "timings_ms": engines.last_timings()}


def _decide_route(audio) -> None:
    """Set the per-clip route once, from a cheap language-ID pass."""
    global _route, _route_reason, _lang
    if _route_reason:  # already decided for this clip
        return
    lang, _prob = engines.detect_language(audio)
    _lang = lang
    if lang in {"hi", "ur", "mr", "ne", "sa"}:  # Hindi-family -> code-switch path
        _route, _route_reason = "hinglish", f"lang={lang}"
    else:
        _route, _route_reason = "fast", f"lang={lang}"


def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    global _prev_text, _committed, _finalized_s, _chunk_future
    if not is_final and len(audio_buffer) < _MIN_AUDIO_BYTES:
        return (_committed, len(_committed))

    audio = engines.pcm_to_float32(audio_buffer)
    if audio.size == 0:
        return (_committed, len(_committed))

    _decide_route(audio)

    # 1. Commit any completed background chunk
    with _chunk_lock:
        if _chunk_future and _chunk_future.done():
            try:
                res = _chunk_future.result()
                if res:
                    _committed = _join(_committed, res)
            except Exception:
                pass
            _chunk_future = None

    if is_final:
        # wait a tiny bit for any in-flight chunk
        t0 = time.monotonic()
        while _chunk_future and time.monotonic() - t0 < 0.5:
            with _chunk_lock:
                if _chunk_future.done():
                    try:
                        res = _chunk_future.result()
                        if res:
                            _committed = _join(_committed, res)
                    except Exception:
                        pass
                    _chunk_future = None
                    break
            time.sleep(0.02)
        
        # slice remaining tail and decode
        tail = audio[int(_finalized_s * _SR):]
        if tail.size > int(0.2 * _SR):
            text = _bg_decode(tail, _route)
            if text:
                _committed = _join(_committed, text)
        
        if not _committed:
            return (_prev_text, len(_prev_text))
        return (_committed, len(_committed))

    # --- background chunking ---
    pending_audio = audio[int(_finalized_s * _SR):]
    if not _chunk_future and pending_audio.size > _SR * 3.5:
        regions = engines.speech_regions(pending_audio)
        if len(regions) > 1:
            boundary = regions[-2][1] + 0.15
            if boundary > 2.0:
                chunk = pending_audio[:int(boundary * _SR)]
                _chunk_future = engines._DECODE_EXECUTOR.submit(_bg_decode, chunk, _route)
                _finalized_s += boundary
                pending_audio = pending_audio[int(boundary * _SR):]

    # --- partial (unscored): cheap fast-lane decode ---
    text = engines.transcribe_fast(pending_audio)
    if not text:
        return (_committed, len(_committed))
    
    _prev_text = text
    return (_join(_committed, text), len(_committed))


def _common_word_prefix(left: str, right: str) -> str:
    lw, rw = _words(left), _words(right)
    out: list[str] = []
    for a, b in zip(lw, rw):
        if a.lower() != b.lower():
            break
        out.append(b)
    return " ".join(out)


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'.-]+", text, flags=re.UNICODE)
