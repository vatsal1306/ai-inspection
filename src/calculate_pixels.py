import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_mask_png(path: Path) -> np.ndarray:
    """Load 8-bit mask PNG (0..255) -> bool array."""
    m = np.array(Image.open(path).convert("L"))
    return (m > 127)


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a boolean mask."""
    mask_u8 = (mask.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask  # no foreground or single component
    # stats: [label, x, y, w, h, area] with row 0 = background
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    return (labels == largest_idx)


def optional_smooth(mask: np.ndarray, k: int = 5, iters: int = 1) -> np.ndarray:
    """
    Optional boundary smoothing using morphological close then open.
    k: kernel size (odd-ish usually)
    """
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=iters)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iters)
    return (opened > 127)


def mask_bbox(mask: np.ndarray):
    """Return bbox as (xmin, ymin, xmax, ymax) in pixel coords, inclusive-exclusive style for drawing."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    xmin, xmax = int(xs.min()), int(xs.max()) + 1
    ymin, ymax = int(ys.min()), int(ys.max()) + 1
    return xmin, ymin, xmax, ymax


def bbox_dims(b):
    """Return width,height from bbox."""
    xmin, ymin, xmax, ymax = b
    return (xmax - xmin), (ymax - ymin)


def overlay_mask_alpha(base_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """
    Simple green alpha overlay on mask pixels.
    base_rgb: HxWx3 uint8
    mask: HxW bool
    """
    out = base_rgb.copy()
    green = np.array([0, 255, 0], dtype=np.float32)
    m = mask.astype(bool)
    out[m] = (out[m].astype(np.float32) * (1 - alpha) + green * alpha).astype(np.uint8)
    return out


def draw_bbox_and_label(pil_img: Image.Image, bbox, text: str):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    xmin, ymin, xmax, ymax = bbox
    draw.rectangle([xmin, ymin, xmax, ymax], width=3)

    # background for readability
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    pad = 2
    tx, ty = xmin, max(0, ymin - (th + 2 * pad))
    draw.rectangle([tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad], fill=(0, 0, 0))
    draw.text((tx + pad, ty + pad), text, fill=(255, 255, 255), font=font)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Original image path")
    ap.add_argument("--sam2_results_json", required=True, help="Step 3 output JSON path")
    ap.add_argument("--out_dir", default="out", help="Output directory")
    ap.add_argument("--smooth", action="store_true", help="Enable optional morphological smoothing (off by default)")
    ap.add_argument("--smooth_kernel", type=int, default=5, help="Kernel size for smoothing")
    ap.add_argument("--smooth_iters", type=int, default=1, help="Iterations for smoothing")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image)
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)

    data = json.loads(Path(args.sam2_results_json).read_text())
    dets = data["detections_used"]

    # Load + clean masks
    cleaned = []
    for d in dets:
        mpath = Path(d["mask_path"])
        m = load_mask_png(mpath)
        m = keep_largest_connected_component(m)

        if args.smooth:
            m = optional_smooth(m, k=args.smooth_kernel, iters=args.smooth_iters)

        bb = mask_bbox(m)
        if bb is None:
            # keep record but mark as invalid
            cleaned.append({**d, "valid": False, "bbox_xyxy": None, "width_px": None, "height_px": None})
            continue

        w, h = bbox_dims(bb)
        cleaned.append({
            **d,
            "valid": True,
            "bbox_xyxy": [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])],
            "width_px": int(w),
            "height_px": int(h),
        })

    # Save cleaned JSON with pixel dimensions
    out_json = out_dir / f"{image_path.stem}_dims_px.json"
    out_json.write_text(json.dumps({
        "image": str(image_path),
        "notes": {
            "largest_cc": True,
            "smoothing_applied": bool(args.smooth),
            "smoothing_kernel": args.smooth_kernel if args.smooth else None,
            "smoothing_iters": args.smooth_iters if args.smooth else None,
            "dimension_definition": "axis-aligned bbox width/height in pixels from cleaned mask",
        },
        "objects": cleaned
    }, indent=2))

    # Visualization: alpha overlay of all valid masks + bbox + label
    overlay = img_np.copy()
    for obj in cleaned:
        if not obj["valid"]:
            continue
        m = load_mask_png(Path(obj["mask_path"]))
        m = keep_largest_connected_component(m)
        if args.smooth:
            m = optional_smooth(m, k=args.smooth_kernel, iters=args.smooth_iters)
        overlay = overlay_mask_alpha(overlay, m, alpha=0.25)

    viz = Image.fromarray(overlay)
    for obj in cleaned:
        if not obj["valid"]:
            continue
        bb = obj["bbox_xyxy"]
        label = obj["label"]
        wpx, hpx = obj["width_px"], obj["height_px"]
        txt = f"{label}  w={wpx}px  h={hpx}px"
        draw_bbox_and_label(viz, bb, txt)

    out_viz = out_dir / f"{image_path.stem}_dims_overlay.jpg"
    viz.save(out_viz)

    print(f"Saved: {out_json}")
    print(f"Saved: {out_viz}")


if __name__ == "__main__":
    main()
