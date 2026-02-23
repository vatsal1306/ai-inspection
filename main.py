"""
main.py
=======

AI Inspection FastAPI service implementing:

GET    /health
POST   /jobs
GET    /jobs/{job_id}
DELETE /jobs/{job_id}

Job lifecycle:
- POST /jobs creates a job (running) and spawns a background pipeline execution.
- GET /jobs/{id} returns:
    running -> result={}
    done    -> result={ overlay_url, prediction:{...status pass/fail...} }
    error   -> result={ error: "..." }
- DELETE /jobs/{id} marks cancel requested and returns status=error with {error:"canceled"}.

Uses:
- FastAPI lifespan for single-time initialization (models, S3 client, downloader, semaphore).  [oai_citation:8‡FastAPI](https://fastapi.tiangolo.com/advanced/events/?utm_source=chatgpt.com)
- BackgroundTasks for running the job after the response returns.  [oai_citation:9‡FastAPI](https://fastapi.tiangolo.com/tutorial/background-tasks/?utm_source=chatgpt.com)
- Requests retry/backoff session inside RobustDownloader.  [oai_citation:10‡urllib3](https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html?utm_source=chatgpt.com)
- Boto3 presigned URL generation for overlay.  [oai_citation:11‡Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-presigned-urls.html?highlight=presigned&utm_source=chatgpt.com)

Run:
  uvicorn main:app --host 0.0.0.0 --port 9002 --reload
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl, model_validator

from src.inspection_pipeline import (
    DownloadError,
    DetectionError,
    SegmentationError,
    MeasurementError,
    RobustDownloader,
    S3OverlayStore,
    Sam2BoxSegmenter,
    GroundingDinoDetector,
    MaskMeasurer,
    PixelToMmScaler,
    TargetSpec,
    UploadError,
    check_pass_fail,
    pil_to_jpeg_bytes,
)

load_dotenv()

# -------------------------
# Config (hardcoded now, changeable later via env/DB)
# -------------------------
APP_PORT = int(os.getenv("APP_PORT", "9002"))

# Hardcode for now, but configurable via env:
BUS_LENGTH_MM = float(os.getenv("BUS_LENGTH_MM", "9000"))
BUS_HEIGHT_MM = float(os.getenv("BUS_HEIGHT_MM", "3200"))

# Model ids (configurable)
DINO_MODEL_ID = os.getenv("DINO_MODEL_ID", "IDEA-Research/grounding-dino-base")
SAM2_MODEL_ID = os.getenv("SAM2_MODEL_ID", "facebook/sam2-hiera-large")

# Detection/segmentation thresholds
DINO_THRESHOLD = float(os.getenv("DINO_THRESHOLD", "0.35"))
SAM_MASK_THRESHOLD = float(os.getenv("SAM_MASK_THRESHOLD", "0.5"))

# S3 overlay output
OVERLAY_BUCKET = os.getenv("OVERLAY_BUCKET", "ai-inspection-model")
OVERLAY_PREFIX = os.getenv("OVERLAY_PREFIX", "overlays")
OVERLAY_PRESIGN_EXPIRES = int(os.getenv("OVERLAY_PRESIGN_EXPIRES", "3600"))
AWS_REGION = os.getenv("AWS_REGION", None)

# Concurrency limit (simple)
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))

# Housekeeping TTL
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", str(24 * 60 * 60)))
CLEANUP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "60"))

# -------------------------
# API contract models
# -------------------------
JobType = Literal["image_measure"]
JobStatus = Literal["running", "done", "error"]
ObjectType = Literal["tyre", "driver_door"]
Unit = Literal["mm", "cm", "m", "inch"]


class Target(BaseModel):
    unit: Unit
    exact: Optional[float] = Field(default=None, gt=0)
    min: Optional[float] = Field(default=None, gt=0)
    max: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_exact_or_range(self) -> "Target":
        has_exact = self.exact is not None
        has_min = self.min is not None
        has_max = self.max is not None
        if has_exact and (has_min or has_max):
            raise ValueError("target must contain either `exact` OR (`min` and `max`), not both")
        if has_exact:
            return self
        if not (has_min and has_max):
            raise ValueError("target range must contain both `min` and `max`")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("target range invalid: min > max")
        return self


class MeasureObject(BaseModel):
    type: ObjectType
    target: Target


class CreateJobRequest(BaseModel):
    job_type: JobType = "image_measure"
    image: HttpUrl
    object: MeasureObject


class CreateJobResponse(BaseModel):
    job_id: str
    job_type: JobType
    status: JobStatus
    job_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: JobType
    status: JobStatus
    result: Dict[str, Any]


# -------------------------
# In-memory job store
# -------------------------
@dataclass
class JobRecord:
    job_id: str
    job_type: JobType
    status: JobStatus = "running"
    result: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_requested: bool = False


JOBS: Dict[str, JobRecord] = {}
JOBS_LOCK = asyncio.Lock()


def now() -> float:
    return time.time()


def make_job_id() -> str:
    # Traceable, stable, safe for filenames
    return uuid.uuid4().hex


def job_url(job_id: str) -> str:
    return f"/jobs/{job_id}"


async def cleanup_loop(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        cutoff = now() - JOB_TTL_SECONDS
        async with JOBS_LOCK:
            dead = [jid for jid, rec in JOBS.items() if rec.updated_at < cutoff]
            for jid in dead:
                JOBS.pop(jid, None)


def pick_torch_device() -> torch.device:
    # Prefer CUDA if present; otherwise MPS; otherwise CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan initializer:
    - Create shared downloader (requests session with retry/backoff)
    - Load ML models once
    - Create S3 store (if configured)
    - Create concurrency semaphore
    - Start cleanup loop
    """
    app.state.stop_event = asyncio.Event()
    app.state.cleanup_task = asyncio.create_task(cleanup_loop(app.state.stop_event))

    app.state.semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

    # Robust downloader (handles MinIO/browser cases)
    app.state.downloader = RobustDownloader(timeout_s=20.0, retries=3, backoff_factor=0.5)

    # ML models: load once
    device = pick_torch_device()
    app.state.device = device

    app.state.detector = GroundingDinoDetector(model_id=DINO_MODEL_ID, device=device)
    app.state.segmenter = Sam2BoxSegmenter(model_id=SAM2_MODEL_ID, device=device)

    # Measurer + scaler
    app.state.measurer = MaskMeasurer(smooth=False)  # keep simple; can make env-configurable
    app.state.scaler = PixelToMmScaler(bus_length_mm=BUS_LENGTH_MM, bus_height_mm=BUS_HEIGHT_MM)

    # S3 overlay store (optional but you want it)
    if not OVERLAY_BUCKET:
        # Keep service running, but jobs will error at upload step.
        app.state.overlay_store = None
    else:
        app.state.overlay_store = S3OverlayStore(
            bucket=OVERLAY_BUCKET,
            prefix=OVERLAY_PREFIX,
            presign_expiry_s=OVERLAY_PRESIGN_EXPIRES,
            region=AWS_REGION,
        )

    yield

    # Shutdown
    app.state.stop_event.set()
    app.state.cleanup_task.cancel()
    try:
        await app.state.cleanup_task
    except Exception:
        pass


app = FastAPI(title="AI Inspection API", version="1.0", lifespan=lifespan)

# CORS (safe default; tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


async def set_job_error(job_id: str, message: str) -> None:
    async with JOBS_LOCK:
        rec = JOBS.get(job_id)
        if rec:
            rec.status = "error"
            rec.result = {"error": message}
            rec.updated_at = now()


async def set_job_done(job_id: str, result: Dict[str, Any]) -> None:
    async with JOBS_LOCK:
        rec = JOBS.get(job_id)
        if rec:
            rec.status = "done"
            rec.result = result
            rec.updated_at = now()


async def is_canceled(job_id: str) -> bool:
    async with JOBS_LOCK:
        rec = JOBS.get(job_id)
        return bool(rec and rec.cancel_requested)


async def run_image_measure_job(app: FastAPI, job_id: str, req: CreateJobRequest) -> None:
    """
    Background worker for image_measure.
    Produces API result exactly matching your documentation:
      result = { overlay_url, prediction: { type, dimensions:{width,height,unit}, status: pass/fail } }
    """
    # Simple concurrency guardrail
    async with app.state.semaphore:
        try:
            # Cancellation check early
            if await is_canceled(job_id):
                await set_job_error(job_id, "canceled")
                return

            # 1) Download input image to temp file
            with tempfile.TemporaryDirectory(prefix=f"{job_id}_") as tmpd:
                tmp_path = Path(tmpd) / "input"
                tmp_path, _ctype = app.state.downloader.download_to_path(str(req.image), tmp_path)

                if await is_canceled(job_id):
                    await set_job_error(job_id, "canceled")
                    return

                # 2) Load image
                img = Image.open(tmp_path).convert("RGB")

                # 3) Detect boxes: always ask for bus+tyre+door (bus used for scaling)
                det_labels = ["bus", "tyre", "door"]
                detections = app.state.detector.detect(img, labels=det_labels, threshold=DINO_THRESHOLD)

                if await is_canceled(job_id):
                    await set_job_error(job_id, "canceled")
                    return

                # 4) Segment masks for all detections
                masks = app.state.segmenter.segment(img, detections, mask_threshold=SAM_MASK_THRESHOLD)

                if await is_canceled(job_id):
                    await set_job_error(job_id, "canceled")
                    return

                # 5) Measure pixel bboxes + overlay (px overlay)
                measured, overlay_px = app.state.measurer.measure(img, detections, masks)

                # 6) Scale px -> mm using bus bbox
                sx, sy = app.state.scaler.compute_scale(measured)

                # 7) Choose the requested object instance to report
                wanted = req.object.type  # "tyre" or "driver_door"
                wanted_label = "tyre" if wanted == "tyre" else "door"

                # Build a list of candidate detections for that label (sorted by score desc)
                candidates = [d for d in detections if str(d.get("label", "")).lower() == wanted_label]
                candidates.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
                if not candidates:
                    raise MeasurementError(f"No detections found for requested object type: {wanted}")

                # Match candidate detection box to closest measured instance by IoU
                def iou(a, b) -> float:
                    ax0, ay0, ax1, ay1 = a
                    bx0, by0, bx1, by1 = b
                    inter_x0 = max(ax0, bx0)
                    inter_y0 = max(ay0, by0)
                    inter_x1 = min(ax1, bx1)
                    inter_y1 = min(ay1, by1)
                    iw = max(0, inter_x1 - inter_x0)
                    ih = max(0, inter_y1 - inter_y0)
                    inter = iw * ih
                    if inter == 0:
                        return 0.0
                    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
                    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
                    denom = area_a + area_b - inter
                    return float(inter) / float(denom) if denom > 0 else 0.0

                best_det = candidates[0]
                best_box = tuple(int(round(x)) for x in best_det["box_xyxy"])

                best_measured = None
                best_i = -1.0
                for m in measured:
                    score = iou(best_box, m.bbox_px)
                    if score > best_i:
                        best_i = score
                        best_measured = m
                if best_measured is None or best_i < 0.1:
                    # if IoU match is poor, fall back to first measured instance of label
                    best_measured = next((m for m in measured if m.label.lower() == wanted_label), None)
                if best_measured is None:
                    raise MeasurementError(f"Could not match a measured mask for object type: {wanted}")

                dims_mm = app.state.scaler.to_mm(best_measured, sx, sy)

                # 8) Decide pass/fail
                target = TargetSpec(
                    unit=req.object.target.unit,
                    exact=req.object.target.exact,
                    min=req.object.target.min,
                    max=req.object.target.max,
                )

                if wanted == "tyre":
                    # For tyres, calculate diameter
                    measured_val = dims_mm["height"]
                    status_pf = "pass" if check_pass_fail(measured_val, target) else "fail"

                    dimensions_out = {
                        "exact": float(measured_val),
                        "unit": "mm",
                    }
                    txt = f"{wanted} diameter={measured_val:}mm => {status_pf}"
                else:
                    # Policy: compare HEIGHT (proxy for door height)
                    measured_val = dims_mm["height"]
                    status_pf = "pass" if check_pass_fail(measured_val, target) else "fail"

                    dimensions_out = {
                        "width": float(dims_mm["width"]),
                        "height": float(dims_mm["height"]),
                        "unit": "mm",
                    }
                    txt = f"{wanted} w={dims_mm['width']:.1f}mm h={dims_mm['height']:.1f}mm => {status_pf}"

                # 9) Create final overlay showing mm dimensions + pass/fail for requested object only
                overlay_final = overlay_px.copy()
                draw = ImageDraw.Draw(overlay_final)
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                draw.text((20, 20), txt, fill=(255, 255, 255), font=font)

                # 10) Upload overlay to S3 and presign
                if app.state.overlay_store is None:
                    raise UploadError("OVERLAY_BUCKET not configured; cannot upload overlay")

                overlay_bytes = pil_to_jpeg_bytes(overlay_final, quality=90)
                overlay_url = app.state.overlay_store.put_overlay_and_presign(job_id=job_id, image_bytes=overlay_bytes)

                result = {
                    "overlay_url": overlay_url,
                    "prediction": {
                        "type": wanted,
                        "dimensions": dimensions_out,
                        "status": status_pf,
                    },
                }

                await set_job_done(job_id, result)

        except DownloadError as e:
            await set_job_error(job_id, str(e))
        except (DetectionError, SegmentationError, MeasurementError, UploadError) as e:
            await set_job_error(job_id, str(e))
        except Exception as e:
            # final safety net: never crash the worker
            await set_job_error(job_id, f"Unhandled error: {e}")


@app.post("/jobs", response_model=CreateJobResponse, status_code=202)
async def create_job(req: CreateJobRequest, background_tasks: BackgroundTasks) -> CreateJobResponse:
    job_id = make_job_id()
    rec = JobRecord(job_id=job_id, job_type=req.job_type, status="running", result={})
    async with JOBS_LOCK:
        JOBS[job_id] = rec

    # Fire-and-forget background task
    background_tasks.add_task(run_image_measure_job, app, job_id, req)

    return CreateJobResponse(job_id=job_id, job_type=req.job_type, status="running", job_url=job_url(job_id))


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str) -> JobStatusResponse:
    async with JOBS_LOCK:
        rec = JOBS.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="job not found")

    # Enforce result shape per your contract
    if rec.status == "running":
        result = {}
    elif rec.status == "error":
        result = {"error": str(rec.result.get("error", "unknown error"))}
    else:
        result = rec.result or {}

    return JobStatusResponse(job_id=rec.job_id, job_type=rec.job_type, status=rec.status, result=result)


@app.delete("/jobs/{job_id}", response_model=JobStatusResponse)
async def cancel_job(job_id: str) -> JobStatusResponse:
    async with JOBS_LOCK:
        rec = JOBS.get(job_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="job not found")
        rec.cancel_requested = True
        rec.updated_at = now()
        # Your spec wants cancel to return status=error + {error:"canceled"}
        rec.status = "error"
        rec.result = {"error": "canceled"}

    return JobStatusResponse(job_id=rec.job_id, job_type=rec.job_type, status="error", result={"error": "canceled"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)
