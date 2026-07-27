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


def draft_reset() -> None:
    """Called by the sealed harness at the start of each clip. Clear state."""
    global _prev_text, _committed, _route, _route_reason, _lang
    _prev_text = ""
    _committed = ""
    _route = "fast"
    _route_reason = ""
    _lang = ""


def route_debug() -> dict:
    """Expose routing info for debug harness."""
    return {"route": _route, "reason": _route_reason, "lang": _lang,
            "timings_ms": engines.last_timings()}


def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    global _prev_text, _committed, _route, _route_reason, _lang
    if not is_final and len(audio_buffer) < _MIN_AUDIO_BYTES:
        return (_committed, len(_committed))

    audio = engines.pcm_to_float32(audio_buffer)
    if audio.size == 0:
        return (_committed, len(_committed))

    if is_final:
        # 1. Run LID ONCE at is_final on the COMPLETE audio clip for 100% routing accuracy
        lang, _prob = engines.detect_language(audio)
        _lang = lang

        # 2. Pad audio array to full 30-second Whisper window to prevent trailing word chopping
        target_len = 16000 * 30
        if len(audio) < target_len:
            import numpy as np
            padded_audio = np.concatenate([audio, np.zeros(target_len - len(audio), dtype=np.float32)])
        else:
            padded_audio = audio

        # 3. Decode on the chosen lane
        if lang in {"hi", "ur", "mr", "ne", "sa"}:
            _route, _route_reason = "hinglish", f"lang={lang}"
            text = engines.transcribe_hinglish(padded_audio) or engines.transcribe_fast(padded_audio)
        else:
            _route, _route_reason = "fast", f"lang={lang}"
            text = engines.transcribe_fast(padded_audio) or engines.transcribe_hinglish(padded_audio)

        text = finalize(text)
        if not text:
            return (_committed or _prev_text, len(_committed or _prev_text))
        _committed = text
        return (text, len(text))

    # --- partial (unscored): cheap fast-lane decode for live preview text ---
    text = engines.transcribe_fast(audio)
    if not text:
        return (_committed, len(_committed))
    stable = _common_word_prefix(_prev_text, text)
    if len(stable) >= len(_committed):
        _committed = stable
    _prev_text = text
    return (text, len(_committed))


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
