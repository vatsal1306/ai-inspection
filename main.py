"""
AI Inspection API (FastAPI) — single-file scaffold
=================================================
Implements your current contract with robust validation + placeholder processing.

Endpoints
---------
GET    /health
POST   /jobs
GET    /jobs/{job_id}
DELETE /jobs/{job_id}

Notes
-----
- Uses FastAPI "lifespan" for startup/shutdown orchestration.  [oai_citation:0‡FastAPI](https://fastapi.tiangolo.com/advanced/events/?utm_source=chatgpt.com)
- Uses FastAPI BackgroundTasks to run placeholder work asynchronously after POST returns.  [oai_citation:1‡FastAPI](https://fastapi.tiangolo.com/tutorial/background-tasks/?utm_source=chatgpt.com)
- Uses Pydantic v2 model_validator for "exact vs range" target validation.  [oai_citation:2‡Pydantic Documentation](https://docs.pydantic.dev/latest/concepts/validators/?utm_source=chatgpt.com)

Run
---
pip install fastapi uvicorn pydantic requests
uvicorn main:app --host 0.0.0.0 --port 9002 --reload
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl, model_validator

# =========================
# API contract (Pydantic)
# =========================
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

        # Range mode
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


# =========================
# In-memory job store
# =========================
@dataclass
class JobRecord:
    job_id: str
    job_type: JobType
    status: JobStatus = "running"
    # As per spec:
    # - running -> {}
    # - error   -> {"error": "..."}
    # - done    -> { overlay_url, prediction: {...} }
    result: Dict[str, Any] = field(default_factory=dict)

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # best-effort cancellation
    cancel_requested: bool = False


JOBS: Dict[str, JobRecord] = {}
JOBS_LOCK = asyncio.Lock()

# housekeeping
JOB_TTL_SECONDS = 24 * 60 * 60  # 1 day
CLEANUP_INTERVAL_SECONDS = 60  # clean once per minute


def _now() -> float:
    return time.time()


def _make_job_id() -> str:
    return uuid.uuid4().hex


def _job_url(job_id: str) -> str:
    # Frontend asked for job URL in response payload.
    # Keep it as relative path so it works behind any host:port.
    return f"/jobs/{job_id}"


# =========================
# Placeholder "pipeline"
# =========================
def _convert_to_mm(value: float, unit: Unit) -> float:
    # Simple unit conversion to mm for demo output
    if unit == "mm":
        return value
    if unit == "cm":
        return value * 10.0
    if unit == "m":
        return value * 1000.0
    if unit == "inch":
        return value * 25.4
    raise ValueError(f"Unsupported unit: {unit}")


def _is_pass(pred_mm: float, target: Target) -> bool:
    if target.exact is not None:
        exact_mm = _convert_to_mm(target.exact, target.unit)
        # Placeholder tolerance: exact match +/- 2%
        tol = 0.02 * exact_mm
        return (exact_mm - tol) <= pred_mm <= (exact_mm + tol)

    # Range
    assert target.min is not None and target.max is not None
    min_mm = _convert_to_mm(target.min, target.unit)
    max_mm = _convert_to_mm(target.max, target.unit)
    return min_mm <= pred_mm <= max_mm


def _fake_predict_dimensions_mm(obj_type: ObjectType) -> Dict[str, float]:
    """
    Placeholder: returns width/height in mm.
    Replace this with your DINO+SAM2 measurement outputs.
    """
    if obj_type == "tyre":
        # e.g., tyre bbox proxy dims in mm
        return {"width": 260.4, "height": 705.2}
    # driver_door
    return {"width": 800.0, "height": 1750.0}


def _fake_overlay_url(job_id: str) -> str:
    """
    Placeholder overlay URL. In your real pipeline:
    - upload overlay image to S3
    - return signed URL
    """
    return f"https://example.com/overlays/{job_id}.jpg"


def _download_check(url: str, timeout_s: int = 15) -> None:
    """
    Optional robustness: validate that the signed URL is reachable.
    For scaffolding we do a HEAD; if S3 doesn't allow HEAD in your setup,
    swap to GET with stream=True.
    """
    try:
        r = requests.head(url, timeout=timeout_s, allow_redirects=True)
        if r.status_code >= 400:
            raise RuntimeError(f"image URL not accessible (HTTP {r.status_code})")
    except Exception as e:
        raise RuntimeError(f"failed to access image URL: {e}") from e


async def run_image_measure_job(job_id: str, payload: CreateJobRequest) -> None:
    """
    Background worker for image_measure jobs.
    Replace the inside of this function with your actual pipeline later.
    """
    # ---- Step 0: optional input URL accessibility check
    # If you don't want any network call here yet, delete this.
    # try:
    #     # Run blocking IO in a thread to not block the event loop.
    #     await asyncio.to_thread(_download_check, str(payload.image))
    # except Exception as e:
    #     async with JOBS_LOCK:
    #         rec = JOBS.get(job_id)
    #         if rec is None:
    #             return
    #         rec.status = "error"
    #         rec.result = {"error": str(e)}
    #         rec.updated_at = _now()
    #     return

    # ---- Simulate compute time (placeholder)
    # Replace this with your real pipeline duration.
    for _ in range(10):
        await asyncio.sleep(0.3)
        async with JOBS_LOCK:
            rec = JOBS.get(job_id)
            if rec is None:
                return
            if rec.cancel_requested:
                rec.status = "error"
                rec.result = {"error": "canceled"}
                rec.updated_at = _now()
                return

    # ---- Produce placeholder measurement result
    dims_mm = _fake_predict_dimensions_mm(payload.object.type)

    # Determine pass/fail using a single scalar:
    # For "tyre", assume target relates to "height" (proxy for diameter)
    # For "driver_door", assume target relates to "height"
    scalar_mm = dims_mm["height"]
    status_pf = "pass" if _is_pass(scalar_mm, payload.object.target) else "fail"

    done_payload = {
        "overlay_url": _fake_overlay_url(job_id),
        "prediction": {
            "type": payload.object.type,
            "dimensions": {
                "width": dims_mm["width"],
                "height": dims_mm["height"],
                "unit": "mm",
            },
            "status": status_pf,
        },
    }

    async with JOBS_LOCK:
        rec = JOBS.get(job_id)
        if rec is None:
            return
        if rec.cancel_requested:
            rec.status = "error"
            rec.result = {"error": "canceled"}
        else:
            rec.status = "done"
            rec.result = done_payload
        rec.updated_at = _now()


# =========================
# Lifespan: cleanup loop
# =========================
async def _cleanup_loop(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        cutoff = _now() - JOB_TTL_SECONDS
        async with JOBS_LOCK:
            dead = [jid for jid, rec in JOBS.items() if rec.updated_at < cutoff]
            for jid in dead:
                JOBS.pop(jid, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.stop_event = asyncio.Event()
    app.state.cleanup_task = asyncio.create_task(_cleanup_loop(app.state.stop_event))
    yield
    # Shutdown
    app.state.stop_event.set()
    app.state.cleanup_task.cancel()
    try:
        await app.state.cleanup_task
    except Exception:
        pass


app = FastAPI(title="AI Inspection API", version="0.1.0", lifespan=lifespan)

# Allow all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Routes
# =========================
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs", response_model=CreateJobResponse, status_code=202)
async def create_job(req: CreateJobRequest, background_tasks: BackgroundTasks) -> CreateJobResponse:
    job_id = _make_job_id()

    rec = JobRecord(job_id=job_id, job_type=req.job_type, status="running", result={})
    async with JOBS_LOCK:
        JOBS[job_id] = rec

    # Spawn processing (placeholder). FastAPI BackgroundTasks runs after response is sent.  [oai_citation:3‡FastAPI](https://fastapi.tiangolo.com/tutorial/background-tasks/?utm_source=chatgpt.com)
    background_tasks.add_task(run_image_measure_job, job_id, req)

    return CreateJobResponse(
        job_id=job_id,
        job_type=req.job_type,
        status="running",
        job_url=_job_url(job_id),
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str) -> JobStatusResponse:
    async with JOBS_LOCK:
        rec = JOBS.get(job_id)

    if rec is None:
        raise HTTPException(status_code=404, detail="job not found")

    # Enforce the response contract strictly
    if rec.status == "running":
        result = {}
    elif rec.status == "error":
        # ensure error dict shape
        if "error" not in rec.result:
            result = {"error": "unknown error"}
        else:
            result = {"error": str(rec.result.get("error"))}
    else:
        # done
        result = rec.result or {}

    return JobStatusResponse(
        job_id=rec.job_id,
        job_type=rec.job_type,
        status=rec.status,
        result=result,
    )


@app.delete("/jobs/{job_id}", response_model=JobStatusResponse)
async def cancel_job(job_id: str) -> JobStatusResponse:
    async with JOBS_LOCK:
        rec = JOBS.get(job_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="job not found")
        # Best-effort cancel:
        rec.cancel_requested = True
        rec.updated_at = _now()

        # As per your spec, cancel returns status=error with {"error":"canceled"}.
        rec.status = "error"
        rec.result = {"error": "canceled"}

    return JobStatusResponse(
        job_id=rec.job_id,
        job_type=rec.job_type,
        status="error",
        result={"error": "canceled"},
    )
