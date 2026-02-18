import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def pick_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def draw_boxes(image: Image.Image, boxes, labels, scores, out_path: Path) -> None:
    im = image.convert("RGB")
    draw = ImageDraw.Draw(im)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for box, lab, score in zip(boxes, labels, scores):
        x0, y0, x1, y1 = map(float, box)
        draw.rectangle([x0, y0, x1, y1], width=3)

        txt = f"{lab} {score:.2f}"
        draw.text((x0, max(0.0, y0 - 12)), txt, font=font)

    im.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["bus", "tire", "door"],
        help="Candidate labels for open-vocab detection (lowercase words/phrases).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Score threshold for keeping boxes (single threshold in your HF version).",
    )
    parser.add_argument(
        "--model_id",
        default="IDEA-Research/grounding-dino-base",
        help="GroundingDINO model id",
    )
    parser.add_argument("--out_dir", default="out")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")

    device = pick_device()
    print(f"Using device: {device}")

    # Normalize labels
    candidate_labels = [s.strip().lower() for s in args.labels if s.strip()]
    if not candidate_labels:
        raise ValueError("No labels provided. Use --labels bus wheel tire door ...")

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device)
    model.eval()

    # IMPORTANT:
    # In your Transformers version, post_process_grounded_object_detection expects:
    #   outputs, threshold=..., target_sizes=..., text_labels=...
    # To ensure label names are returned, we pass text_labels explicitly.
    text_labels = [candidate_labels]  # batch of 1 image

    inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=args.threshold,
        target_sizes=[(image.height, image.width)],
        text_labels=text_labels,
    )

    res = results[0]

    # Expected keys per HF docs: "scores", "boxes", "labels", "text_labels"
    boxes = res["boxes"].detach().cpu().tolist()
    scores = res["scores"].detach().cpu().tolist()

    # Some unified post-processors return numeric label indices in "labels"
    # and optionally strings in "text_labels". We'll handle both.
    if res.get("text_labels", None) is not None:
        labels_out = [str(x) for x in res["text_labels"]]
    else:
        # labels are indices into candidate_labels
        idxs = res["labels"]
        idxs = idxs.detach().cpu().tolist() if torch.is_tensor(idxs) else list(idxs)
        labels_out = [candidate_labels[i] if 0 <= i < len(candidate_labels) else str(i) for i in idxs]

    out_json = out_dir / f"{image_path.stem}_dino.json"
    payload = {
        "image": str(image_path),
        "model_id": args.model_id,
        "device": str(device),
        "labels": candidate_labels,
        "threshold": args.threshold,
        "detections": [
            {"label": lab, "score": float(sc), "box_xyxy": box}
            for lab, sc, box in zip(labels_out, scores, boxes)
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2))

    out_img = out_dir / f"{image_path.stem}_dino_boxes.jpg"
    draw_boxes(image, boxes, labels_out, scores, out_img)

    print(f"Saved: {out_json}")
    print(f"Saved: {out_img}")
    print(f"Detections: {len(boxes)}")


if __name__ == "__main__":
    main()
