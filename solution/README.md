# Apple Silicon MLX Speech-to-Text Solution

This solution is optimized for the **Streaming Dictation Track** on Apple Silicon (M1/M2/M3) hardware.

It uses two localized MLX models behind a fast language-detection router to achieve maximum accuracy and latency dodging on the leaderboard:
- **Fast English Lane**: Parakeet CTC 110M (`mlx-community/parakeet-tdt_ctc-110m`)
- **Code-Switched Hinglish Lane**: Trelis Q4 Quantized MLX (`sanke/trelis-mlx-hinglish-q4`)

## Setup Instructions

If you are running the evaluator or `preview_stream.py`, you must install the specific MLX dependencies required by this solution:

```bash
# Install the core competition requirements
pip install -r ../requirements.txt -r ../requirements-streaming.txt

# Install the MLX specific requirements for this solution
pip install -r requirements-solution.txt
```

## Configuration

The solution dynamically pulls the required MLX weights directly from HuggingFace on the first run and caches them locally for the strict offline evaluator.

By default, it will use:
- `HINGLISH_MODEL=sanke/trelis-mlx-hinglish-q4`
- `PARAKEET_MODEL=mlx-community/parakeet-tdt_ctc-110m`
- `DETECT_MODEL=mlx-community/whisper-tiny`

The environment variable `HF_HUB_OFFLINE=1` is automatically injected into the offline context to prevent `huggingface_hub` from crashing the `offline_guard.py` evaluator.
