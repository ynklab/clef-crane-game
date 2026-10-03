# Clef Crane Game

A browser demo that sends only the phase, held-prize flag, a JPEG screenshot, and recent action history to the pinned Cloudflare Clef model, and applies its discrete action choices to the physics crane game.

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

The human player's view remains the existing oblique 3D camera, with no compass or directional-arrow overlay (removed as ineffective); manual WASD/arrow and Space controls still map to world axes W = -Z, A = -X, S = +Z, D = +X, the player just has to learn that mapping rather than read it off an on-screen arrow.

## Minimal decide payload

`POST /api/decide` sends Clef only `{phase, has_prize, image, history}` — the
idle/carrying phase, a strict boolean for whether the claw is holding a prize,
a screenshot, and up to 5 recent `{action, step}` entries (oldest first). No
coordinates, prize lists, velocities, scores, or numeric navigation offsets are
sent to the model or included in its prompt; Clef must judge claw/prize/tray
alignment purely from the screenshot.

The screenshot sent to the model is captured from a dedicated straight-down
top-down camera, not the oblique camera the human player sees. An oblique
view has perspective and occlusion that make "is the claw over the prize"
ambiguous from a single frame — nearer objects can visually overlap farther
ones even when they are not aligned in the horizontal plane the crane moves
in. Removing that perspective makes screen-space position a direct, unambiguous
proxy for horizontal (x/z) alignment: on the top-down image, up is W (-Z),
down is S (+Z), left is A (-X), and right is D (+X), matching the instructions
given to the model. The claw marker visible in that capture sits at the claw's
actual world position (it moves with the claw, it is not fixed at the image
center), and prizes are rendered unobstructed in the capture. This top-down
capture is used only for the model's input image; the human player's
on-screen view, controls, and HUD remain the existing oblique camera with no
compass overlay, unchanged. The W/A/S/D <-> screen-direction mapping used by
the model comes from the fixed instructions given to it for every request,
not from any arrows or overlay drawn into the image.
The observation state handed to the model otherwise stays minimal — `{phase, has_prize, image, history}` — with no
coordinates or geometric measurements added for the model to reason with.
Extra fields (e.g. the old coordinate-based payload shape) are rejected with
HTTP 422 instead of being silently accepted. This does not guarantee
grab/release success or scoring — Clef's visual judgment of alignment can be
wrong, and missed prizes are reported as failures, not hidden.

## If a running tab gets HTTP 422 from `/api/decide`

The server enforces one strict request schema with no compatibility shim for
older payload shapes. A browser tab left open from before a server/client
update keeps running whatever copy of `clef-demo.js` it already loaded and
will keep posting that older shape, which the current server now rejects
with 422 — this looks like a server bug but is purely a stale client.
**Reload the page** (not just re-click Start) to fetch the current
`index.html` and `clef-demo.js`. `index.html`, `/index.html`, and
`clef-demo.js` are now served with `Cache-Control: no-store`, and the
`clef-demo.js` import in `index.html` carries a content-hash query string
(`?v=<hash>`) that changes whenever `clef-demo.js` changes, so a reload can
never resolve to a cached copy from before a deploy. If 422s persist after a
full reload, check the server log line `422 POST /api/decide rejected
fields: [...]`, which reports only the rejected field locations and pydantic
error types (never the image/base64 or request body) for diagnosis.


During release, the claw opens more gently to reduce sideways kicks from the pads. Prizes still fall and collide under the existing physics; grip eligibility and scoring are unchanged.

