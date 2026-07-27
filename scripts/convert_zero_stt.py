#!/usr/bin/env python3
"""Convert shunyalabs/zero-stt-hinglish -> CTranslate2 int8 for faster-whisper.

Run this on a Linux machine (or in the GitHub Actions convert-zero-stt workflow).
The output directory is a drop-in for faster-whisper WhisperModel():

    WhisperModel("./zero-stt-hinglish-ct2", device="cpu", compute_type="int8")

Usage:
    python scripts/convert_zero_stt.py [--output-dir ./zero-stt-hinglish-ct2]
                                       [--hf-repo yourname/zero-stt-hinglish-ct2]
                                       [--upload]

Deps (install before running):
    pip install ctranslate2 transformers huggingface_hub torch

After conversion you can upload to your HF with:
    huggingface-cli upload <your-hf-username>/zero-stt-hinglish-ct2 ./zero-stt-hinglish-ct2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


SOURCE_MODEL = "shunyalabs/zero-stt-hinglish"
DEFAULT_OUTPUT = "./zero-stt-hinglish-ct2"


def _convert(output_dir: str) -> None:
    print(f"[convert] {SOURCE_MODEL} -> {output_dir}  (int8, cpu)")
    # ct2-transformers-converter is the official CTranslate2 CLI tool.
    # It reads Hugging Face Transformers weights and writes a ct2 directory.
    cmd = [
        sys.executable, "-m", "ctranslate2.tools.transformers",
        "--model",       SOURCE_MODEL,
        "--output_dir",  output_dir,
        "--quantization", "int8",
        "--copy_files",  "tokenizer.json", "tokenizer_config.json",
                         "vocab.json", "merges.txt",
                         "preprocessor_config.json",
                         "added_tokens.json", "special_tokens_map.json",
        "--force",
    ]
    # Fallback: use the ct2-transformers-converter entry-point if available
    try:
        subprocess.run(
            ["ct2-transformers-converter",
             "--model",       SOURCE_MODEL,
             "--output_dir",  output_dir,
             "--quantization", "int8",
             "--copy_files",  "tokenizer.json", "tokenizer_config.json",
                              "vocab.json", "merges.txt",
                              "preprocessor_config.json",
                              "added_tokens.json", "special_tokens_map.json",
             "--force"],
            check=True,
        )
    except FileNotFoundError:
        # Entry-point not on PATH; run via python -m
        subprocess.run(cmd, check=True)
    print(f"[convert] done → {output_dir}")


def _smoke_test(output_dir: str) -> None:
    """Run a quick sanity decode on the 3 Hinglish sample clips."""
    print("[smoke] running faster-whisper sanity check ...")
    import glob
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[smoke] faster-whisper not installed — skipping smoke test")
        return

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav_files = sorted(glob.glob(os.path.join(here, "samples", "openslr104*.wav")))
    if not wav_files:
        print("[smoke] no sample clips found — skipping smoke test")
        return

    model = WhisperModel(output_dir, device="cpu", compute_type="int8")
    for wav in wav_files:
        segs, info = model.transcribe(
            wav,
            language="hi",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segs)
        print(f"  {os.path.basename(wav)} -> {text[:100]!r}  [lang={info.language}]")
    print("[smoke] done")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                    help=f"Where to write the CTranslate2 directory (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--skip-smoke", action="store_true",
                    help="Skip the faster-whisper smoke test after conversion")
    args = ap.parse_args()

    _convert(args.output_dir)

    if not args.skip_smoke:
        _smoke_test(args.output_dir)

    print(f"\n[done] Model saved to: {os.path.abspath(args.output_dir)}")
    print("To upload manually:  huggingface-cli upload <yourname/zero-stt-hinglish-ct2> "
          f"{args.output_dir}")


if __name__ == "__main__":
    main()
