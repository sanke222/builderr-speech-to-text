"""Isolated MLX decode worker — runs as a CHILD PROCESS of the engine.

Why a subprocess: MLX/Metal faults are process-fatal (segfaults, GPU-watchdog
kills) — no in-process guard can survive them. This worker is single-threaded,
so all Metal work is strictly serial by construction, and if Metal ever takes
it down it takes down ONLY this child; the sealed server detects the death and
falls back gracefully.

Protocol (newline-delimited JSON over stdio):
  parent -> child : {"wav": "/abs/path.wav", "lang": "hi", "repo": "..."}
  child  -> parent: {"text": "..."}            on success
                    {"error": "..."}           on a caught failure
  On start the child prints {"ready": true} after the model is loaded and a
  tiny warm decode has compiled the Metal kernels.
"""
from __future__ import annotations

import json
import os
import sys
import wave
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np

SR = 16000

def _read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as r:
        raw = r.readframes(r.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _decode(mdir: str, audio: np.ndarray, lang: str) -> str:
    import mlx_whisper
    out = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=mdir,
        language=lang,
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt=(
            "Yeh audio ek technical meeting hai. Always use English terms for open source "
            "software, features, aur operating systems. Write numbers in digits (jaise 3, 4, 100)."
        ),
        fp16=True,
    )
    segs = out.get("segments") or []
    if not segs:
        return (out.get("text") or "").strip()
    keep = []
    for s in segs:
        if s.get("no_speech_prob", 0.0) > 0.65 and s.get("avg_logprob", 0.0) < -1.0:
            continue
        keep.append((s.get("text") or "").strip())
    return " ".join(k for k in keep if k).strip()


def main() -> None:
    # We read the model repo from sys.argv because the parent passes it on startup
    repo = sys.argv[1] if len(sys.argv) > 1 else "sanke/trelis-mlx-hinglish-q4"
    
    # warm decode: loads weights + compiles Metal kernels before READY
    _decode(repo, np.zeros(SR // 2, dtype=np.float32), "en")
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            audio = _read_wav(req["wav"])
            # The parent can override the repo per request if needed
            req_repo = req.get("repo", repo)
            text = _decode(req_repo, audio, req.get("lang") or "hi")
            resp = {"text": text}
        except Exception as exc:  # noqa: BLE001
            resp = {"error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
