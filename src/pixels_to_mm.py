import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_bbox_and_label(pil_img: Image.Image, bbox, text: str):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    xmin, ymin, xmax, ymax = bbox
    draw.rectangle([xmin, ymin, xmax, ymax], width=3)

    # label background
    try:
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    except Exception:
        # fallback rough size
        tw, th = (len(text) * 6, 12)

    pad = 2
    tx, ty = xmin, max(0, ymin - (th + 2 * pad))
    draw.rectangle([tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad], fill=(0, 0, 0))
    draw.text((tx + pad, ty + pad), text, fill=(255, 255, 255), font=font)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Original image path")
    ap.add_argument("--dims_px_json", required=True, help="Step 4 output: *_dims_px.json")
    ap.add_argument("--bus_length_mm", type=float, required=True)
    ap.add_argument("--bus_height_mm", type=float, required=True)
    ap.add_argument("--out_dir", default="out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image)
    img = Image.open(image_path).convert("RGB")

    dims = json.loads(Path(args.dims_px_json).read_text())
    objs = dims["objects"]

    # Find bus object
    bus = None
    for o in objs:
        if o.get("valid") and str(o.get("label", "")).lower() == "bus":
            bus = o
            break
    if bus is None:
        raise RuntimeError("Could not find a valid 'bus' object in dims_px_json.")

    bus_w_px = float(bus["width_px"])
    bus_h_px = float(bus["height_px"])
    if bus_w_px <= 0 or bus_h_px <= 0:
        raise RuntimeError("Bus pixel dimensions are invalid.")

    sx = args.bus_length_mm / bus_w_px
    sy = args.bus_height_mm / bus_h_px

    # Convert all
    out_objects = []
    for o in objs:
        o2 = dict(o)
        if o.get("valid"):
            w_px = float(o["width_px"])
            h_px = float(o["height_px"])
            o2["width_mm"] = w_px * sx
            o2["height_mm"] = h_px * sy
        else:
            o2["width_mm"] = None
            o2["height_mm"] = None
        out_objects.append(o2)

    out_json = out_dir / f"{image_path.stem}_dims_real.json"
    out_json.write_text(json.dumps({
        "image": str(image_path),
        "bus_reference": {
            "bus_length_mm": args.bus_length_mm,
            "bus_height_mm": args.bus_height_mm,
            "bus_width_px": bus_w_px,
            "bus_height_px": bus_h_px,
            "sx_mm_per_px": sx,
            "sy_mm_per_px": sy,
            "method": "anisotropic scaling using bus bbox width/height",
        },
        "objects": out_objects
    }, indent=2))

    # Overlay real dimensions
    viz = img.copy()
    for o in out_objects:
        if not o.get("valid"):
            continue
        bb = o["bbox_xyxy"]
        lab = o["label"]
        wmm = o["width_mm"]
        hmm = o["height_mm"]
        txt = f"{lab}  w={wmm:.1f}mm  h={hmm:.1f}mm"
        draw_bbox_and_label(viz, bb, txt)

    out_viz = out_dir / f"{image_path.stem}_dims_real_overlay.jpg"
    viz.save(out_viz)

    print(f"Saved: {out_json}")
    print(f"Saved: {out_viz}")
    print(f"sx(mm/px)={sx:.6f}  sy(mm/px)={sy:.6f}")


if __name__ == "__main__":
    main()