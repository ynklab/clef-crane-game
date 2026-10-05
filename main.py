"""FastAPI server that proxies the crane game in ``index.html`` to a
llama.cpp ``llama-server`` instance serving the Clef decision model,
answering a joint action/step choice per turn.

Routes: GET /, /index.html, /clef-demo.js, GET /api/health, POST /api/decide.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from pydantic.functional_validators import field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("clef_server")

MODEL_REPO = "Cloudflare/clef"
MODEL_REVISION = "2f3de3dd85f379784083b0814d997ab627200f0c"
HOST = "0.0.0.0"
PORT = 8000

UPSTREAM_BASE_URL = os.environ.get("CLEF_UPSTREAM_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
UPSTREAM_HEALTH_URL = f"{UPSTREAM_BASE_URL}/health"
UPSTREAM_SYSTEMONE_URL = f"{UPSTREAM_BASE_URL}/v1/systemone"
UPSTREAM_TIMEOUT_SECONDS = float(os.environ.get("CLEF_UPSTREAM_TIMEOUT_SECONDS", "60"))

STATIC_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = STATIC_DIR / "index.html"
CLEF_DEMO_JS_PATH = STATIC_DIR / "clef-demo.js"

# The browser must never keep running an old copy of clef-demo.js against a
# newer, stricter /api/decide schema: a stale in-page module (or a disk-cached
# response) that still posts a retired payload shape produces 422s that look
# like a server bug but are purely a client-version mismatch. Two independent
# protections close that gap: (1) every static response is served
# Cache-Control: no-store so the browser cannot silently reuse a prior body
# across reloads, and (2) the clef-demo.js import in index.html is suffixed
# with a content hash, so any change to clef-demo.js changes the import URL
# itself and a reloaded page can never resolve to a stale cached JS body.
NO_STORE_HEADERS = {"Cache-Control": "no-store"}
_INDEX_HTML_IMPORT_NEEDLE = "from './clef-demo.js';"


def _content_version(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


CLEF_DEMO_JS_VERSION = _content_version(CLEF_DEMO_JS_PATH)
_INDEX_HTML_IMPORT_REPLACEMENT = f"from './clef-demo.js?v={CLEF_DEMO_JS_VERSION}';"


def _render_index_html() -> str:
    text = INDEX_HTML_PATH.read_text(encoding="utf-8")
    if _INDEX_HTML_IMPORT_NEEDLE not in text:
        raise RuntimeError(
            "index.html is missing the expected clef-demo.js import; cannot version it"
        )
    return text.replace(_INDEX_HTML_IMPORT_NEEDLE, _INDEX_HTML_IMPORT_REPLACEMENT, 1)


IMAGE_DATA_URL_PREFIX = "data:image/jpeg;base64,"
MAX_IMAGE_DATA_URL_CHARS = 2 * 1024 * 1024
MAX_IMAGE_SIDE = 768
MAX_HISTORY = 5

GOAL = (
    "Operate the physics crane game using the screenshot, phase, and has_prize flag. The "
    "screenshot is a straight-down top view of the play area: up on the screen is the W "
    "direction, down on the screen is S, left on the screen is A, and right on the screen is "
    "D. The small claw marker at the center of the crane shows the claw's current position; "
    "prizes are drawn unobstructed and in full view. In the idle phase, move the claw marker "
    "so it is over an unscored prize visible in the screenshot, then grab. In the carrying "
    "phase, trust has_prize over the phase label: if has_prize is true, move the held prize "
    "so it is over the center of the green DROP ZONE visible in the screenshot and release "
    "there; if has_prize is false, release immediately to start another attempt. Prefer fine "
    "steps once close to the target and also right after an overshoot or a direction reversal, "
    "so as not to repeatedly undo the last move; use coarser steps while still clearly far from "
    "the target. Wait only when the scene shows necessary physical settling, not when a useful "
    "move, grab, or release is visibly available. Use the recent action history to keep "
    "pursuing the same target instead of oscillating. Choose exactly one allowed action."
)

STEP_INSTRUCTIONS_BY_PHASE = {
    "idle": (
        "Choose fine=0.1 seconds when the claw marker is close to the target prize or just "
        "overshot or reversed direction, medium=0.25 seconds at a moderate distance, and "
        "coarse=0.6 seconds when the target prize is far across the play area. This duration "
        "does not limit grab."
    ),
    "carrying": (
        "Choose fine=0.1 seconds when the held prize is close to the green DROP ZONE or just "
        "overshot or reversed direction, medium=0.25 seconds at a moderate distance, and "
        "coarse=0.6 seconds when it is far from the DROP ZONE. This duration does not limit "
        "release."
    ),
}


ACTION_CRITERIA_BY_PHASE: dict[str, dict[str, str]] = {
    "idle": {
        "left": "Move the claw marker toward the target prize in the screen-left (A) direction.",
        "right": "Move the claw marker toward the target prize in the screen-right (D) direction.",
        "forward": "Move the claw marker toward the target prize in the screen-up (W) direction.",
        "backward": "Move the claw marker toward the target prize in the screen-down (S) direction.",
        "grab": "Begin the grab sequence once the claw marker is over an unscored prize.",
        "wait": "Take no action and let physics settle for the chosen step duration.",
    },
    "carrying": {
        "left": "Move the held prize toward the green DROP ZONE in the screen-left (A) direction.",
        "right": "Move the held prize toward the green DROP ZONE in the screen-right (D) direction.",
        "forward": "Move the held prize toward the green DROP ZONE in the screen-up (W) direction.",
        "backward": "Move the held prize toward the green DROP ZONE in the screen-down (S) direction.",
        "release": "Begin release once the held prize is over the center of the green DROP ZONE, or immediately if has_prize is false.",
        "wait": "Take no action and let physics settle for the chosen step duration.",
    },
}

STEP_CRITERIA: dict[str, str] = {
    "fine": "0.1 seconds (12 ticks); use when close to the target, or right after an overshoot or direction reversal.",
    "medium": "0.25 seconds (30 ticks); use at a moderate distance from the target.",
    "coarse": "0.6 seconds (72 ticks); use only while still clearly far from the target.",
}




def build_questions(phase: str) -> dict[str, Any]:
    return {
        "action": {
            "type": "choice",
            "instructions": GOAL,
            "criteria": ACTION_CRITERIA_BY_PHASE[phase],
        },
        "step": {
            "type": "choice",
            "instructions": STEP_INSTRUCTIONS_BY_PHASE[phase],
            "criteria": dict(STEP_CRITERIA),
        },
    }


PhaseLiteral = Literal["idle", "carrying"]
ActionLiteral = Literal["left", "right", "forward", "backward", "grab", "release", "wait"]
StepLiteral = Literal["fine", "medium", "coarse"]


class HistoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ActionLiteral
    step: StepLiteral


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: PhaseLiteral
    has_prize: StrictBool
    image: str
    history: Annotated[list[HistoryEntry], Field(max_length=MAX_HISTORY)]

    @field_validator("image")
    @classmethod
    def _check_image_prefix(cls, value: str) -> str:
        if len(value) > MAX_IMAGE_DATA_URL_CHARS:
            raise ValueError(f"image data URL exceeds {MAX_IMAGE_DATA_URL_CHARS} characters")
        if not value.startswith(IMAGE_DATA_URL_PREFIX):
            raise ValueError("image must be a data:image/jpeg;base64, URL")
        return value


class DecideResponse(BaseModel):
    model: str
    revision: str
    answers: dict[str, Any]
    elapsed_ms: float


def _decode_image(data_url: str) -> None:
    encoded = data_url[len(IMAGE_DATA_URL_PREFIX) :]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"image is not valid base64: {exc}") from exc
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise ValueError(f"image could not be decoded: {exc}") from exc
    if image.format != "JPEG":
        raise ValueError(f"image must be JPEG, got {image.format!r}")
    if max(image.size) > MAX_IMAGE_SIDE:
        raise ValueError(f"image long side {max(image.size)} exceeds {MAX_IMAGE_SIDE}")
    # The validated original data URL is forwarded to llama-server; no image copy is needed here.


INFERENCE_LOCK = threading.Lock()
HTTP_CLIENT: httpx.Client | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global HTTP_CLIENT
    HTTP_CLIENT = httpx.Client(timeout=UPSTREAM_TIMEOUT_SECONDS)
    yield
    HTTP_CLIENT.close()
    HTTP_CLIENT = None


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Diagnose client/server schema mismatches (e.g. a stale cached client
    # still posting a retired payload shape) without ever logging the image
    # data URL, base64, or any other field value: only the rejected field
    # path and the pydantic error type are safe to log.
    locations = [{"loc": error.get("loc"), "type": error.get("type")} for error in exc.errors()]
    logger.warning("422 %s %s rejected fields: %s", request.method, request.url.path, locations)
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": exc.errors()}),
    )


@app.get("/")
def get_root() -> HTMLResponse:
    return HTMLResponse(_render_index_html(), headers=NO_STORE_HEADERS)


@app.get("/index.html")
def get_index_html() -> HTMLResponse:
    return HTMLResponse(_render_index_html(), headers=NO_STORE_HEADERS)


@app.get("/clef-demo.js")
def get_clef_demo_js() -> FileResponse:
    return FileResponse(
        CLEF_DEMO_JS_PATH, media_type="text/javascript", headers=NO_STORE_HEADERS
    )


@app.get("/api/health")
def get_health() -> dict[str, Any]:
    if HTTP_CLIENT is None:
        raise HTTPException(status_code=503, detail="upstream model server is not available")
    try:
        response = HTTP_CLIENT.get(UPSTREAM_HEALTH_URL)
    except httpx.HTTPError as exc:
        logger.warning("upstream health check failed: %s", exc)
        raise HTTPException(
            status_code=503, detail="upstream model server is not reachable"
        ) from exc
    if response.status_code != 200:
        raise HTTPException(status_code=503, detail="upstream model server is not ready")
    return {
        "ready": True,
        "model": MODEL_REPO,
        "revision": MODEL_REVISION,
        "device": "llama.cpp",
    }


@app.post("/api/decide", response_model=DecideResponse)
def post_decide(request: DecideRequest) -> DecideResponse:
    try:
        _decode_image(request.image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if HTTP_CLIENT is None:
        raise HTTPException(status_code=503, detail="upstream model server is not available")

    if not INFERENCE_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="an inference is already in progress")
    try:
        history = [entry.model_dump(mode="json") for entry in request.history]
        state = {
            "goal": GOAL,
            "phase": request.phase,
            "has_prize": request.has_prize,
            "history": history,
        }
        questions = build_questions(request.phase)
        payload = {
            "state": state,
            "questions": questions,
            "images": [request.image],
        }
        started = time.perf_counter()
        try:
            response = HTTP_CLIENT.post(UPSTREAM_SYSTEMONE_URL, json=payload)
        except httpx.HTTPError as exc:
            logger.exception("upstream Clef request failed")
            raise HTTPException(
                status_code=502, detail="upstream model server request failed"
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if response.status_code == 503:
            raise HTTPException(status_code=503, detail="upstream model server is not ready")
        if response.status_code != 200:
            logger.error(
                "upstream Clef request returned status %s", response.status_code
            )
            raise HTTPException(
                status_code=502,
                detail=f"upstream model server returned status {response.status_code}",
            )
        try:
            result = response.json()
        except ValueError as exc:
            logger.exception("upstream Clef response was not valid JSON")
            raise HTTPException(
                status_code=502, detail="upstream model server returned an invalid response"
            ) from exc
        answers = result.get("answers") if isinstance(result, dict) else None
        if not isinstance(answers, dict):
            logger.error("upstream Clef response is missing an 'answers' object")
            raise HTTPException(
                status_code=502, detail="upstream model server response is missing answers"
            )
    finally:
        INFERENCE_LOCK.release()

    return DecideResponse(
        model=MODEL_REPO,
        revision=MODEL_REVISION,
        answers=answers,
        elapsed_ms=elapsed_ms,
    )


def main() -> None:
    uvicorn.run(app, host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
