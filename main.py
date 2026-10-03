"""FastAPI server that loads the pinned Cloudflare/clef model and drives the
crane game in ``index.html`` by answering a joint action/step choice per turn.

Routes: GET /, /index.html, /clef-demo.js, GET /api/health, POST /api/decide.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import math
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field
from pydantic.functional_validators import AfterValidator, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("clef_server")

MODEL_REPO = "Cloudflare/clef"
MODEL_REVISION = "2f3de3dd85f379784083b0814d997ab627200f0c"
HOST = "0.0.0.0"
PORT = 8000

STATIC_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = STATIC_DIR / "index.html"
CLEF_DEMO_JS_PATH = STATIC_DIR / "clef-demo.js"

IMAGE_DATA_URL_PREFIX = "data:image/jpeg;base64,"
MAX_IMAGE_DATA_URL_CHARS = 2 * 1024 * 1024
MAX_IMAGE_SIDE = 768
MAX_HISTORY = 5
MIN_PRIZES = 1
MAX_PRIZES = 12

GOAL = (
    "Operate the physics crane game to increase score by delivering unscored prizes to the "
    "green tray centered at x=3.55, z=2.8. Use world coordinates, not screen directions. "
    "navigation.prizes gives each unscored prize's dx and dz from the actual claw; "
    "navigation.tray gives the dx and dz from the held prize (or claw if no prize is held) to "
    "the tray. In idle, pursue the nearest reachable unscored prize. If abs(dx) and abs(dz) are "
    "both at most 0.45, choose grab. Otherwise move on the axis with the larger absolute "
    "offset, toward its sign: dx>0 right, dx<0 left, dz>0 backward, dz<0 forward. Continue "
    "toward the same prize; do not reverse while that offset keeps the same sign. In carrying, "
    "if held_prize_id is not null, follow navigation.tray and release when both tray "
    "offsets are at most 0.45; if held_prize_id is null, release to start another attempt. "
    "Choose fine steps near the target and coarse steps only when the selected axis offset "
    "is greater than "
    "1.5. Wait only for necessary physical settling, not when a useful move, grab, or release "
    "is available. Never infer successful grip or scoring from phase alone. Choose exactly one "
    "allowed action."
)

STEP_INSTRUCTIONS_BY_PHASE = {
    "idle": (
        "Choose fine=0.1 seconds when the chosen prize's selected-axis offset is at most 0.55; "
        "medium=0.25 seconds when it is over 0.55 through 1.5; coarse=0.6 seconds only when it "
        "exceeds 1.5. Use the dx/dz offsets in navigation.prizes. This duration does not limit "
        "grab or release."
    ),
    "carrying": (
        "Choose fine=0.1 seconds when navigation.tray's selected-axis offset is at most 0.55; "
        "medium=0.25 seconds when it is over 0.55 through 1.5; coarse=0.6 seconds only when it "
        "exceeds 1.5. This duration does not limit grab or release."
    ),
}


ACTION_CRITERIA_BY_PHASE: dict[str, dict[str, str]] = {
    "idle": {
        "left": "Move the claw toward the target prize in -x (dx<0), using keyboard A.",
        "right": "Move the claw toward the target prize in +x (dx>0), using keyboard D.",
        "forward": "Move the claw toward the target prize in -z (dz<0), using keyboard W.",
        "backward": "Move the claw toward the target prize in +z (dz>0), using keyboard S.",
        "grab": "Begin the grab sequence when both target offsets satisfy abs(dx)<=0.45 and abs(dz)<=0.45.",
        "wait": "Take no action and let physics settle for the chosen step duration.",
    },
    "carrying": {
        "left": "Move the held prize toward the tray in -x when navigation.tray.dx<0 (keyboard A).",
        "right": "Move the held prize toward the tray in +x when navigation.tray.dx>0 (keyboard D).",
        "forward": "Move the held prize toward the tray in -z when navigation.tray.dz<0 (keyboard W).",
        "backward": "Move the held prize toward the tray in +z when navigation.tray.dz>0 (keyboard S).",
        "release": "Begin release when the held prize is over the green tray.",
        "wait": "Take no action and let physics settle for the chosen step duration.",
    },
}

STEP_CRITERIA: dict[str, str] = {
    "fine": "0.1 seconds (12 ticks); small target-axis offset.",
    "medium": "0.25 seconds (30 ticks); medium target-axis offset.",
    "coarse": "0.6 seconds (72 ticks); large target-axis offset only.",
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


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("value must be a finite number")
    return value


FiniteFloat = Annotated[float, AfterValidator(_finite)]
PhaseLiteral = Literal["idle", "carrying"]
ActionLiteral = Literal["left", "right", "forward", "backward", "grab", "release", "wait"]
StepLiteral = Literal["fine", "medium", "coarse"]


class Vec3(BaseModel):
    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat


class Vec2XZ(BaseModel):
    x: FiniteFloat
    z: FiniteFloat


class ClawState(BaseModel):
    position: Vec3
    target: Vec2XZ
    finger_angle: FiniteFloat
    held_prize_id: int | None


class SphereShape(BaseModel):
    type: Literal["sphere"]
    radius: FiniteFloat


class BoxShape(BaseModel):
    type: Literal["box"]
    half_extents: Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]


Shape = Annotated[SphereShape | BoxShape, Field(discriminator="type")]


class Prize(BaseModel):
    id: int
    position: Vec3
    velocity: Vec3
    shape: Shape
    scored: bool


class AxisBounds(BaseModel):
    x: Annotated[list[FiniteFloat], Field(min_length=2, max_length=2)]
    z: Annotated[list[FiniteFloat], Field(min_length=2, max_length=2)]


class Speeds(BaseModel):
    idle: FiniteFloat
    carrying: FiniteFloat


class ScoreBounds(BaseModel):
    x: Annotated[list[FiniteFloat], Field(min_length=2, max_length=2)]
    z: Annotated[list[FiniteFloat], Field(min_length=2, max_length=2)]
    y_max: FiniteFloat
    speed_max: FiniteFloat


class Tray(BaseModel):
    center: Vec2XZ
    score_bounds: ScoreBounds


class Observation(BaseModel):
    phase: PhaseLiteral
    phase_time: FiniteFloat
    score: int
    claw: ClawState
    prizes: Annotated[list[Prize], Field(min_length=MIN_PRIZES, max_length=MAX_PRIZES)]
    bounds: AxisBounds
    speeds: Speeds
    tray: Tray
def build_navigation(observation: Observation) -> dict[str, Any]:
    claw = observation.claw.position
    held_prize = next(
        (prize for prize in observation.prizes if prize.id == observation.claw.held_prize_id),
        None,
    )
    tray_origin = (
        held_prize.position if observation.phase == "carrying" and held_prize is not None else claw
    )
    return {
        "prizes": [
            {
                "prize_id": prize.id,
                "dx": round(prize.position.x - claw.x, 3),
                "dz": round(prize.position.z - claw.z, 3),
                "distance_xz": round(
                    math.hypot(prize.position.x - claw.x, prize.position.z - claw.z), 3
                ),
            }
            for prize in observation.prizes
            if not prize.scored
        ],
        "tray": {
            "dx": round(observation.tray.center.x - tray_origin.x, 3),
            "dz": round(observation.tray.center.z - tray_origin.z, 3),
        },
    }


class HistoryClaw(BaseModel):
    x: FiniteFloat
    z: FiniteFloat


class HistorySnapshot(BaseModel):
    phase: PhaseLiteral
    claw: HistoryClaw
    score: int


class HistoryEntry(BaseModel):
    action: ActionLiteral
    step: StepLiteral
    before: HistorySnapshot
    after: HistorySnapshot


class DecideRequest(BaseModel):
    observation: Observation
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


def _decode_image(data_url: str) -> Image.Image:
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
    return image.convert("RGB")


@dataclass
class ModelState:
    model: Any
    processor: Any
    systemone: Callable[..., dict[str, Any]]


MODEL_STATE: ModelState | None = None
INFERENCE_LOCK = threading.Lock()


def _load_model() -> ModelState:
    logger.info("probing CUDA availability")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this server requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the GPU does not support bfloat16")
    probe = torch.ones((8, 8), device="cuda", dtype=torch.bfloat16)
    probe_result = (probe @ probe)[0, 0].item()
    if probe_result != 8:
        raise RuntimeError(f"bf16 matmul probe failed: expected 8, got {probe_result}")
    logger.info("CUDA bf16 probe passed on %s", torch.cuda.get_device_name(0))

    from huggingface_hub import snapshot_download

    logger.info("downloading %s at revision %s", MODEL_REPO, MODEL_REVISION)
    model_path = snapshot_download(MODEL_REPO, revision=MODEL_REVISION)
    if model_path not in sys.path:
        sys.path.insert(0, model_path)
    from joint_schema_model import load_release_model, systemone  # type: ignore[import-not-found]

    logger.info("loading Clef model weights from %s", model_path)
    model, processor = load_release_model(
        model_path, device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    logger.info("Clef model loaded and ready")
    return ModelState(model=model, processor=processor, systemone=systemone)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_STATE
    try:
        MODEL_STATE = _load_model()
    except Exception:
        logger.exception("failed to load the Clef model; server will not start")
        raise
    yield
    MODEL_STATE = None


app = FastAPI(lifespan=lifespan)


@app.get("/")
def get_root() -> FileResponse:
    return FileResponse(INDEX_HTML_PATH)


@app.get("/index.html")
def get_index_html() -> FileResponse:
    return FileResponse(INDEX_HTML_PATH)


@app.get("/clef-demo.js")
def get_clef_demo_js() -> FileResponse:
    return FileResponse(CLEF_DEMO_JS_PATH, media_type="text/javascript")


@app.get("/api/health")
def get_health() -> dict[str, Any]:
    if MODEL_STATE is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "ready": True,
        "model": MODEL_REPO,
        "revision": MODEL_REVISION,
        "device": "cuda",
    }


@app.post("/api/decide", response_model=DecideResponse)
def post_decide(request: DecideRequest) -> DecideResponse:
    try:
        image = _decode_image(request.image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if MODEL_STATE is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    if not INFERENCE_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="an inference is already in progress")
    try:
        observation = request.observation.model_dump(mode="json")
        history = [entry.model_dump(mode="json") for entry in request.history]
        navigation = build_navigation(request.observation)
        state = {
            "goal": GOAL,
            "observation": observation,
            "navigation": navigation,
            "history": history,
        }
        questions = build_questions(request.observation.phase)
        started = time.perf_counter()
        try:
            result = MODEL_STATE.systemone(
                MODEL_STATE.model,
                MODEL_STATE.processor,
                {
                    "model": "clef",
                    "state": state,
                    "images": [image],
                    "questions": questions,
                },
            )
        except Exception as exc:
            logger.exception("Clef inference failed")
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    finally:
        INFERENCE_LOCK.release()

    return DecideResponse(
        model=MODEL_REPO,
        revision=MODEL_REVISION,
        answers=result["answers"],
        elapsed_ms=elapsed_ms,
    )


def main() -> None:
    uvicorn.run(app, host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
