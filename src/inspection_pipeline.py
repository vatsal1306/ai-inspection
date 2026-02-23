"""
inspection_pipeline.py
======================

End-to-end inspection pipeline components used by the FastAPI service.

This module provides:
- Robust URL download with retries/backoff
- GroundingDINO detection
- SAM2 segmentation
- Mask cleanup + pixel measurement
- Pixel -> mm scaling using bus length/height (hardcoded via env in API layer)
- Overlay generation
- S3 upload + presigned URL for overlay

Design goals:
- Keep ML model objects instantiated once and reused (performance + stability).
- Provide clear exceptions for each pipeline stage.
- Avoid brittle subprocess/script invocation (use direct function calls).

"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import boto3
import cv2
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from requests.adapters import HTTPAdapter
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers import Sam2Model, Sam2Processor
from urllib3.util.retry import Retry


# -------------------------
# Exceptions (stage-specific)
# -------------------------
class PipelineError(RuntimeError):
    """Base exception for pipeline failures."""


class DownloadError(PipelineError):
    """Raised when input URL cannot be downloaded."""


class DetectionError(PipelineError):
    """Raised when detection fails or yields unusable outputs."""


class SegmentationError(PipelineError):
    """Raised when segmentation fails or yields unusable outputs."""


class MeasurementError(PipelineError):
    """Raised when mask measurement or scaling fails."""


class UploadError(PipelineError):
    """Raised when uploading overlay or generating presigned URL fails."""


# -------------------------
# Utilities: units + target checking
# -------------------------
Unit = Literal["mm", "cm", "m", "inch"]


def to_mm(value: float, unit: Unit) -> float:
    """Convert a scalar value to millimeters."""
    if unit == "mm":
        return value
    if unit == "cm":
        return value * 10.0
    if unit == "m":
        return value * 1000.0
    if unit == "inch":
        return value * 25.4
    raise ValueError(f"Unsupported unit: {unit}")


@dataclass(frozen=True)
class TargetSpec:
    """Target constraint: either exact or [min,max], expressed in a unit."""
    unit: Unit
    exact: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None

    def is_exact(self) -> bool:
        return self.exact is not None

    def validate(self) -> None:
        if self.exact is not None and (self.min is not None or self.max is not None):
            raise ValueError("TargetSpec: either exact OR (min,max), not both")
        if self.exact is None:
            if self.min is None or self.max is None:
                raise ValueError("TargetSpec: range requires both min and max")
            if self.min > self.max:
                raise ValueError("TargetSpec: min > max")


def check_pass_fail(measured_mm: float, target: TargetSpec) -> bool:
    """
    Compare a measured mm scalar against a target spec.
    - exact: +/- 2% tolerance (placeholder policy, easy to change)
    - range: inclusive min/max
    """
    target.validate()
    if target.exact is not None:
        exact_mm = to_mm(target.exact, target.unit)
        tol = 0.02 * exact_mm
        return (exact_mm - tol) <= measured_mm <= (exact_mm + tol)
    assert target.min is not None and target.max is not None
    return to_mm(target.min, target.unit) <= measured_mm <= to_mm(target.max, target.unit)


# -------------------------
# Robust downloader (MinIO/S3/browser-friendly)
# -------------------------
class RobustDownloader:
    """
    Robust URL downloader:
    - Browser-like headers
    - allow_redirects=True
    - HEAD-first (fast) then GET fallback if HEAD blocked
    - Streaming download
    - Retries with exponential backoff using urllib3 Retry via requests adapter
    """

    def __init__(
            self,
            timeout_s: float = 20.0,
            retries: int = 3,
            backoff_factor: float = 0.5,
            status_forcelist: Iterable[int] = (429, 500, 502, 503, 504),
    ) -> None:
        self.timeout_s = timeout_s

        self.session = requests.Session()
        retry = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            backoff_factor=backoff_factor,
            status_forcelist=set(status_forcelist),
            allowed_methods=frozenset(["HEAD", "GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Browser-ish headers to avoid MinIO/S3 quirks
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0 Safari/537.36"
            ),
            "Accept": "*/*",
            "Accept-Encoding": "identity",
            "Connection": "keep-alive",
        }

    def check_accessible(self, url: str) -> Tuple[int, str]:
        """Return (http_status, content_type). Raise DownloadError on failure."""
        try:
            r = self.session.head(url, headers=self.headers, timeout=self.timeout_s, allow_redirects=True)
        except requests.RequestException:
            r = None

        if r is None or r.status_code >= 400:
            # Fallback GET check
            try:
                g = self.session.get(url, headers=self.headers, timeout=self.timeout_s, stream=True,
                                     allow_redirects=True)
                status = g.status_code
                ctype = g.headers.get("Content-Type", "")
                g.close()
            except requests.RequestException as e:
                raise DownloadError(f"URL unreachable via GET fallback: {e}") from e
            if status >= 400:
                raise DownloadError(f"URL not accessible (HTTP {status})")
            return status, ctype

        return r.status_code, r.headers.get("Content-Type", "")

    def download_to_path(self, url: str, out_path: Path) -> Tuple[Path, str]:
        """
        Download URL to disk. Returns (path, content_type).
        Raises DownloadError on failure.
        """
        status, ctype = self.check_accessible(url)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.session.get(
                    url,
                    headers=self.headers,
                    timeout=self.timeout_s,
                    stream=True,
                    allow_redirects=True,
            ) as r:
                if r.status_code >= 400:
                    raise DownloadError(f"Download failed (HTTP {r.status_code})")
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 128):
                        if chunk:
                            f.write(chunk)
        except requests.RequestException as e:
            raise DownloadError(f"Download request failed: {e}") from e

        return out_path, ctype


# -------------------------
# S3 uploader + presigner
# -------------------------
class S3OverlayStore:
    """
    Upload overlay bytes to S3 and return a presigned GET URL.

    Uses boto3 presigned URL generation.  [oai_citation:7‡Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-presigned-urls.html?highlight=presigned&utm_source=chatgpt.com)
    """

    def __init__(self, bucket: str, prefix: str, presign_expiry_s: int = 3600, region: Optional[str] = None) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")

        client_kwargs: Dict[str, Any] = {}
        if region:
            client_kwargs["region_name"] = region
        self.s3 = boto3.client("s3", **client_kwargs)

        self.presign_expiry_s = presign_expiry_s

    def put_overlay_and_presign(self, job_id: str, image_bytes: bytes, content_type: str = "image/jpeg") -> str:
        """
        Upload overlay as s3://bucket/prefix/{job_id}.jpg and return a presigned GET URL.
        """
        key = f"{self.prefix}/{job_id}.jpg" if self.prefix else f"{job_id}.jpg"
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=image_bytes,
                ContentType=content_type,
            )
        except Exception as e:
            raise UploadError(f"Failed to upload overlay to S3: {e}") from e

        try:
            url = self.s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=self.presign_expiry_s,
            )
        except Exception as e:
            raise UploadError(f"Failed to generate presigned GET URL: {e}") from e

        return url


# -------------------------
# Detector (GroundingDINO)
# -------------------------
class GroundingDinoDetector:
    """
    GroundingDINO wrapper for zero-shot text label detection.
    Keeps processor + model loaded and reused.

    Output detections list:
    { "label": str, "score": float, "box_xyxy": [x0,y0,x1,y1] }
    """

    def __init__(self, model_id: str, device: torch.device) -> None:
        self.model_id = model_id
        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, image: Image.Image, labels: List[str], threshold: float) -> List[Dict[str, Any]]:
        if not labels:
            raise DetectionError("No labels provided to detector")

        candidate_labels = [s.strip().lower() for s in labels if s.strip()]
        text_labels = [candidate_labels]  # batch size 1

        inputs = self.processor(images=image, text=text_labels, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=[(image.height, image.width)],
            text_labels=text_labels,
        )[0]

        boxes = results["boxes"].detach().cpu().tolist()
        scores = results["scores"].detach().cpu().tolist()

        if results.get("text_labels", None) is not None:
            labels_out = [str(x) for x in results["text_labels"]]
        else:
            idxs = results["labels"].detach().cpu().tolist()
            labels_out = [candidate_labels[i] if 0 <= i < len(candidate_labels) else str(i) for i in idxs]

        dets = [{"label": lab, "score": float(sc), "box_xyxy": box} for lab, sc, box in zip(labels_out, scores, boxes)]
        if not dets:
            raise DetectionError("No detections produced by GroundingDINO")

        return dets


# -------------------------
# Segmenter (SAM2)
# -------------------------
class Sam2BoxSegmenter:
    """
    SAM2 box-prompt segmenter.
    Keeps processor + model loaded and reused.

    For each input box, returns a binary mask (H, W) aligned with original image.
    """

    def __init__(self, model_id: str, device: torch.device) -> None:
        self.model_id = model_id
        self.device = device

        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(device)
        self.model.eval()

    @torch.no_grad()
    def segment(self, image: Image.Image, detections: List[Dict[str, Any]], mask_threshold: float = 0.5) -> List[
        np.ndarray]:
        if not detections:
            raise SegmentationError("No detections provided to segmenter")

        boxes = [d["box_xyxy"] for d in detections]
        inputs = self.processor(images=image, input_boxes=[boxes], return_tensors="pt").to(self.device)

        outputs = self.model(**inputs, multimask_output=False)

        all_masks = self.processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            mask_threshold=mask_threshold,
            binarize=True,
        )
        masks_t = all_masks[0]  # (N,H,W) or (N,1,H,W)
        if masks_t.ndim == 4 and masks_t.shape[1] == 1:
            masks_t = masks_t.squeeze(1)
        if masks_t.ndim != 3:
            raise SegmentationError(f"Unexpected SAM2 mask shape: {tuple(masks_t.shape)}")

        masks = masks_t.numpy().astype(bool)
        if masks.shape[0] != len(detections):
            raise SegmentationError("Mask count does not match detection count")

        return [masks[i] for i in range(masks.shape[0])]


# -------------------------
# Measurement + overlay
# -------------------------
def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only largest connected component in a boolean mask."""
    mask_u8 = (mask.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    return labels == largest_idx


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Axis-aligned bbox in xyxy (xmin,ymin,xmax,ymax) where xmax/ymax are exclusive."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    xmin, xmax = int(xs.min()), int(xs.max()) + 1
    ymin, ymax = int(ys.min()), int(ys.max()) + 1
    return xmin, ymin, xmax, ymax


def draw_bbox_and_label(img: Image.Image, bbox: Tuple[int, int, int, int], text: str) -> None:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    xmin, ymin, xmax, ymax = bbox
    draw.rectangle([xmin, ymin, xmax, ymax], width=3)
    try:
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    except Exception:
        tw, th = (len(text) * 6, 12)
    pad = 2
    tx, ty = xmin, max(0, ymin - (th + 2 * pad))
    draw.rectangle([tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad], fill=(0, 0, 0))
    draw.text((tx + pad, ty + pad), text, fill=(255, 255, 255), font=font)


def overlay_mask_alpha(base_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    out = base_rgb.copy()
    green = np.array([0, 255, 0], dtype=np.float32)
    m = mask.astype(bool)
    out[m] = (out[m].astype(np.float32) * (1 - alpha) + green * alpha).astype(np.uint8)
    return out


@dataclass
class MeasuredInstance:
    label: str
    bbox_px: Tuple[int, int, int, int]
    width_px: int
    height_px: int


class MaskMeasurer:
    """
    Convert detections+masks into measurement-ready objects.
    Keeps logic consistent with your earlier CLI steps (largest CC + bbox).
    """

    def __init__(self, smooth: bool = False, smooth_kernel: int = 5, smooth_iters: int = 1) -> None:
        self.smooth = smooth
        self.smooth_kernel = smooth_kernel
        self.smooth_iters = smooth_iters

    def _optional_smooth(self, mask: np.ndarray) -> np.ndarray:
        if not self.smooth:
            return mask
        mask_u8 = (mask.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.smooth_kernel, self.smooth_kernel))
        closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=self.smooth_iters)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=self.smooth_iters)
        return opened > 127

    def measure(self, image: Image.Image, detections: List[Dict[str, Any]], masks: List[np.ndarray]) -> Tuple[
        List[MeasuredInstance], Image.Image]:
        if len(detections) != len(masks):
            raise MeasurementError("detections and masks length mismatch")

        overlay = np.array(image.convert("RGB"))
        measured: List[MeasuredInstance] = []

        for det, mask in zip(detections, masks):
            m = keep_largest_connected_component(mask)
            m = self._optional_smooth(m)
            bbox = mask_bbox(m)
            if bbox is None:
                continue

            xmin, ymin, xmax, ymax = bbox
            wpx, hpx = xmax - xmin, ymax - ymin
            measured.append(MeasuredInstance(det["label"], bbox, wpx, hpx))
            overlay = overlay_mask_alpha(overlay, m, alpha=0.20)

        viz = Image.fromarray(overlay)
        for inst in measured:
            draw_bbox_and_label(viz, inst.bbox_px, f"{inst.label} w={inst.width_px}px h={inst.height_px}px")

        if not measured:
            raise MeasurementError("No measurable instances (all masks empty?)")

        return measured, viz


class PixelToMmScaler:
    """
    Convert measured pixel bbox dimensions into mm using bus bbox scaling:
      sx = bus_length_mm / bus_width_px
      sy = bus_height_mm / bus_height_px

    This matches your earlier approach.
    """

    def __init__(self, bus_length_mm: float, bus_height_mm: float) -> None:
        if bus_length_mm <= 0 or bus_height_mm <= 0:
            raise ValueError("bus_length_mm and bus_height_mm must be > 0")
        self.bus_length_mm = bus_length_mm
        self.bus_height_mm = bus_height_mm

    def compute_scale(self, measured: List[MeasuredInstance]) -> Tuple[float, float]:
        bus = next((m for m in measured if m.label.lower() == "bus"), None)
        if bus is None:
            raise MeasurementError("Bus detection missing; required for scaling")
        if bus.width_px <= 0 or bus.height_px <= 0:
            raise MeasurementError("Invalid bus pixel dimensions")
        sx = self.bus_length_mm / float(bus.width_px)
        sy = self.bus_height_mm / float(bus.height_px)
        return sx, sy

    def to_mm(self, inst: MeasuredInstance, sx: float, sy: float) -> Dict[str, float]:
        return {
            "width": inst.width_px * sx,
            "height": inst.height_px * sy,
        }


def pil_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    """Serialize PIL image to JPEG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
