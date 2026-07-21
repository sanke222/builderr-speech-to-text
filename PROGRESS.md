# Session progress — streaming dictation engine

## What was built

### solution/finalizer.py
Faithful post-processing that prevents the scorecard's hard caps from firing:
- Spelled-out numbers → digits (`twenty five` → `25`) so `critical_flip` doesn't cap at 50
- Repetition-loop collapse (same 3-gram ×4+) so the loop cap (30) never fires
- Negation tokens (English + Hindi: `not/nahi/mat/नहीं`) are never touched
- No translation, no meaning rewrite — purely structural cleanup
- Verified against the real `scorecard.py`: a number-flip goes from cap-50 to pass

### solution/engines.py
Lazy-loaded MLX engine adapters (Apple Silicon, offline once cached):
- `transcribe_fast()` — Parakeet via `parakeet-mlx` (English/Indian-English, sub-second)
- `transcribe_hinglish()` — Oriserve Apex via `mlx-whisper` (romanized code-switch, greedy, no prev-text conditioning)
- `detect_language()` — whisper-tiny prefix LID for routing decisions
- `warm_all()` — pre-pulls and JITs all three models during warmup (before `block_network()` fires)
- `pcm_to_float32()` — int16 LE PCM → float32 for all engines
- Per-call timing dict exposed for debug logs

Model IDs overridable via env vars (`PARAKEET_MODEL`, `APEX_MODEL`, `DETECT_MODEL`).

### solution/draft.py (rewritten)
Router + streaming orchestration implementing the `draft()` contract:
- **Partial calls**: cheap Parakeet decode on rolling buffer, commits stable common-word prefix
- **Route decision**: whisper-tiny LID on first call per clip → `fast` (English) or `hinglish` (Hindi-family)
- **Final call**: faithful decode on the chosen lane (Apex for Hinglish, Parakeet for English), with cross-lane fallback if primary returns empty, then `finalize()`
- Never blanks: degrades to last committed prefix rather than returning empty
- `route_debug()` exposes lane + reason + timings for the debug harness (unscored)

### scripts/debug_stream.py
Per-clip debug table (bypasses WebSocket harness, calls `draft()` directly):
- Lane chosen + why (language ID result)
- Raw Parakeet text
- Raw Apex text (only when escalated)
- Finalized text
- Gold + gold_alternatives
- Exact-token AND phonetic meaning/WER scores (so Devanagari-vs-romanized gap is visible)
- Per-stage timings: detect / fast / hinglish / total ms

### scripts/offline_check.py
Offline proof script:
- Warms all engines (network allowed)
- Calls `block_network()` from `offline_guard.py`
- Re-runs all 6 samples — any non-loopback socket raises `NetworkBlocked` and exits 1
- Exit 0 = offline-clean, safe to submit

### .github/workflows/mac-stream.yml
Main CI workflow on `macos-14` (Apple Silicon M1), triggers on push to `main` or manually:
1. Install `mlx-whisper`, `parakeet-mlx`, `websockets`, `numpy`, `soundfile`
2. HF model cache (keyed by model IDs)
3. Warmup step — asserts all engines load
4. `scripts/offline_check.py` — offline proof
5. `scripts/debug_stream.py` — per-clip debug table → `debug_table.txt`
6. `python preview_stream.py` — official streaming scorecard → `streaming_score.txt`
7. Both logs uploaded as artifacts

### .github/workflows/convert-apex.yml
One-shot conversion workflow (run once, manually):
- Converts `Oriserve/Whisper-Hindi2Hinglish-Apex` → MLX 4-bit on the Mac runner
- Smoke-tests on the 3 Hinglish samples (prints transcripts to verify faithfulness)
- Uploads to your HF repo using `HF_TOKEN` secret
- After this runs, set `APEX_MODEL` in `mac-stream.yml` to your uploaded repo ID

## Current gap

`APEX_MODEL` defaults to `mlx-community/whisper-large-v3-turbo` (base turbo, not the Oriserve fine-tune).
Run `convert-apex.yml` once with your HF repo name → update `APEX_MODEL` → Hinglish lane uses the real fine-tune.

## Architecture shape

```
draft(pcm, is_final)
  │
  ├─ partial: Parakeet rolling decode → commit stable prefix (free compute during speech)
  │
  └─ final:
       detect_language(prefix) → route
       ├─ fast     → transcribe_fast()   [Parakeet]
       └─ hinglish → transcribe_hinglish() [Apex]  + fast fallback
                  → finalize() → paste
```

## Key scoring facts

| Cap | Trigger | Our defence |
|-----|---------|-------------|
| 0   | blank final | degrade to committed prefix |
| 30  | repetition loop | `collapse_repetition()` in finalizer |
| 50  | critical fact flip (number/negation/entity) | `digitize_numbers()` + never drop negation |
| 70  | median latency > 4s | router (Apex only on Hinglish clips) + greedy decode |

Latency scoring: ≤1000ms = full 30pts, linear decay to 0 at 5000ms.
