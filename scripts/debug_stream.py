"""Neat per-clip debug harness for the streaming engine (Apple Silicon / CI).

Runs each sample clip straight through draft() (bypassing the WebSocket harness)
so we can SEE, per clip:

  - which lane the router chose and why (language ID)
  - the raw fast-lane (Parakeet) text
  - the raw Hinglish (Apex) text when escalated
  - the finalized text we would paste
  - the gold + the exact-token AND phonetic scores (so the Devanagari-vs-romanized
    gap is visible instead of guessed)
  - per-stage timings (detect / fast / hinglish ms) and total wall time

This is a DEBUG tool, not the scorer. preview_stream.py is the real scorecard
(runs the sealed WebSocket harness). Usage:

    python scripts/debug_stream.py                 # samples/manifest.json
    python scripts/debug_stream.py path/to/manifest.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import wave

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from scorecard import phonetic_token_f1, phonetic_wer, token_f1, wer  # noqa: E402
from solution import draft, engines  # noqa: E402


def _read_wav_pcm(path: str) -> bytes:
    """Load a WAV as 16 kHz mono s16le PCM bytes (what the harness feeds)."""
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, f"{path}: expected 16k, got {w.getframerate()}"
        assert w.getnchannels() == 1, f"{path}: expected mono"
        return w.readframes(w.getnframes())


def _best(gold: str, alts: list[str], pred: str):
    """Score pred against gold (exact) and best alternative (phonetic)."""
    exact_m, exact_w = token_f1(gold, pred), wer(gold, pred)
    ph_m = max([phonetic_token_f1(a, pred) for a in alts], default=0.0)
    ph_w = min([phonetic_wer(a, pred) for a in alts], default=1.0)
    return exact_m, exact_w, ph_m, ph_w


def run(manifest_path: str) -> None:
    rows = json.load(open(manifest_path))
    base = os.path.dirname(os.path.abspath(manifest_path))

    print("== warmup (pull + JIT models; must finish before offline) ==")
    status = engines.warm_all()
    print(f"   engines ready: {status}\n")

    for r in rows:
        clip_id = r.get("clip_id", "?")
        gold = r.get("gold", "")
        alts = r.get("gold_alternatives") or []
        audio_path = r.get("audio") or os.path.join(base, f"{clip_id}.wav")
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(HERE, audio_path)

        pcm = _read_wav_pcm(audio_path)
        audio = engines.pcm_to_float32(pcm)

        draft.draft_reset()
        t0 = time.monotonic()
        # route decision + raw lane outputs (explicit, for the table)
        draft._decide_route(audio)  # noqa: SLF001 - debug introspection
        dbg = draft.route_debug()
        raw_fast = engines.transcribe_fast(audio)
        raw_hing = engines.transcribe_hinglish(audio) if dbg["route"] == "hinglish" else ""
        # the actual final draft() would emit
        final_text, _ = draft.draft(pcm, is_final=True)
        total_ms = (time.monotonic() - t0) * 1000.0

        ex_m, ex_w, ph_m, ph_w = _best(gold, alts, final_text)
        t = dbg["timings_ms"]
        print(f"── {clip_id}  [{r.get('language','?')}/{r.get('category','?')}]")
        print(f"   route      : {dbg['route']}  ({dbg['reason']})")
        print(f"   raw fast   : {raw_fast[:100]!r}")
        if raw_hing:
            print(f"   raw apex   : {raw_hing[:100]!r}")
        print(f"   FINAL      : {final_text[:100]!r}")
        print(f"   gold       : {gold[:100]!r}")
        if alts:
            print(f"   gold(alt)  : {alts[0][:100]!r}")
        print(f"   score      : exact meaning {ex_m:.3f} / WER {ex_w:.3f}   "
              f"phonetic meaning {ph_m:.3f} / WER {ph_w:.3f}")
        print(f"   timings ms : detect {t.get('detect_ms',0):.0f}  "
              f"fast {t.get('fast_ms',0):.0f}  hinglish {t.get('hinglish_ms',0):.0f}  "
              f"total {total_ms:.0f}\n")


if __name__ == "__main__":
    manifest = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "samples/manifest.json")
    run(manifest)
