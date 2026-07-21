"""Prove the streaming engine runs with the network blocked (like scoring).

Warms up ALL models (network allowed), then calls offline_guard.block_network()
and re-runs draft() on every sample. Any non-loopback socket now raises
NetworkBlocked -> we fail loudly. A loopback ASR server would still be allowed.

Exit 0 = offline-clean. Exit 1 = something touched the network after warmup.
"""
from __future__ import annotations

import json
import os
import sys
import wave

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from offline_guard import NetworkBlocked, block_network  # noqa: E402
from solution import draft, engines  # noqa: E402


def _pcm(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        return w.readframes(w.getnframes())


def main() -> int:
    manifest = os.path.join(HERE, "samples/manifest.json")
    rows = json.load(open(manifest))

    print("warmup (network allowed): pulling + JITing models ...")
    status = engines.warm_all()
    print(f"  engines ready: {status}")
    if not any(status.values()):
        print("FAIL: no engine loaded during warmup (models not cached).")
        return 1

    print("block_network(): outbound non-loopback now raises")
    block_network()

    ok = True
    for r in rows:
        clip_id = r.get("clip_id", "?")
        audio_path = r.get("audio") or f"samples/{clip_id}.wav"
        pcm = _pcm(os.path.join(HERE, audio_path))
        try:
            draft.draft_reset()
            text, _ = draft.draft(pcm, is_final=True)
            print(f"  offline OK  {clip_id[:36]:36s} -> {text[:48]!r}")
        except NetworkBlocked as e:
            print(f"  NETWORK!    {clip_id}: {e}")
            ok = False
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR       {clip_id}: {type(e).__name__}: {e}")
            ok = False

    print("OFFLINE-CLEAN" if ok else "OFFLINE-DIRTY (a call hit the network)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
