# Clef Crane Game

A browser demo that sends live game observations and JPEG screenshots to the pinned Cloudflare Clef model and applies its discrete action choices to the physics crane game.

## Requirements

- Linux on a CUDA-capable NVIDIA GPU with BF16 support and a compatible NVIDIA driver.
- Python 3.12 and `uv`.
- Internet access to download the pinned model (about 55 GB) and the browser's Three.js / cannon-es modules from esm.sh.
- Chromium or Firefox with WebGL enabled.

The configured PyTorch wheels target CUDA 13.0 on Linux. There is no CPU or hosted-model fallback.

## Run

```sh
uv sync --python 3.12
uv run --no-sync python main.py
```

Wait for the startup log `Clef model loaded and ready`, then open <http://127.0.0.1:8000/>. Startup probes CUDA/BF16 and loads `Cloudflare/clef` at revision `2f3de3dd85f379784083b0814d997ab627200f0c` into the standard Hugging Face cache. The browser enables automatic operation only after `/api/health` confirms that model is ready. Manual WASD/arrow and Space controls remain available while auto is stopped.
The service binds to `0.0.0.0:8000` (accessible on the host's network interfaces). Stop it with Ctrl-C. No API key is required.

During release, the claw opens more gently to reduce sideways kicks from the pads. Prizes still fall and collide under the existing physics; grip eligibility and scoring are unchanged.

