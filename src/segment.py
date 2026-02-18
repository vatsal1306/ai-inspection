import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Processor, Sam2Model


def pick_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def save_mask_png(mask: np.ndarray, out_path: Path) -> None:
    mask_u8 = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_u8, mode="L").save(out_path)


def overlay_masks_on_image(image: Image.Image, masks: list[np.ndarray], out_path: Path) -> None:
    im = np.array(image.convert("RGB")).astype(np.uint8)
    overlay = im.copy()
    for m in masks:
        m = m.astype(bool)
        overlay[m] = (overlay[m] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--detections_json", required=True)
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--model_id", default="facebook/sam2-hiera-large")
    parser.add_argument("--score_threshold", type=float, default=0.0)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")

    det = json.loads(Path(args.detections_json).read_text())
    detections = [d for d in det["detections"] if float(d.get("score", 0.0)) >= args.score_threshold]
    if not detections:
        raise RuntimeError("No detections to segment. Lower --score_threshold or check Step 2 output.")

    boxes = [d["box_xyxy"] for d in detections]  # xyxy in pixel coords
    labels = [d.get("label", "obj") for d in detections]
    dino_scores = [float(d.get("score", 0.0)) for d in detections]

    # SAM2 expects batched boxes: [batch, num_boxes, 4]
    input_boxes = [boxes]

    device = pick_device()
    print(f"Using device: {device}")

    processor = Sam2Processor.from_pretrained(args.model_id)
    model = Sam2Model.from_pretrained(args.model_id).to(device)
    model.eval()

    inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    # Post-process masks back to original image size (SAM2 signature)
    # Returns a list (len=batch_size); each element is (num_objects, H, W)
    all_masks = processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        inputs["original_sizes"].detach().cpu(),
        mask_threshold=args.mask_threshold,
        binarize=True,
    )

    # all_masks is list(batch); pick first image
    masks_t = all_masks[0]  # torch.Tensor
    print("Post-processed masks shape:", tuple(masks_t.shape))  # should be (N, H, W)
    print("Image size (HxW):", image.height, image.width)
    # Expected shapes often: (N, 1, H, W) or (N, H, W)
    # masks_t can be (N, H, W) OR (N, 1, H, W) OR (N, 3, H, W)
    if masks_t.ndim == 4:
        # If it's multimask (3), choose the best mask per object.
        # Prefer iou_scores if present; otherwise fallback to largest-area mask.
        if masks_t.shape[1] == 3 and hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
            # outputs.iou_scores likely shape: (B, N, 3); take B=0
            scores = outputs.iou_scores.detach().cpu()[0]  # (N, 3)
            best = torch.argmax(scores, dim=1)  # (N,)
            masks_t = masks_t[torch.arange(masks_t.shape[0]), best]  # (N, H, W)
        else:
            # squeeze singleton channel if present
            if masks_t.shape[1] == 1:
                masks_t = masks_t.squeeze(1)  # (N, H, W)
            else:
                # fallback: pick largest area across channel dimension
                areas = masks_t.flatten(2).sum(-1)  # (N, C)
                best = torch.argmax(areas, dim=1)
                masks_t = masks_t[torch.arange(masks_t.shape[0]), best]  # (N, H, W)
    elif masks_t.ndim != 3:
        raise RuntimeError(f"Unexpected mask tensor shape: {tuple(masks_t.shape)}")

    masks = masks_t.numpy().astype(bool)  # (N, H, W)

    # masks = processed.detach().cpu().numpy()
    # masks = masks > args.mask_threshold

    stem = image_path.stem
    masks_dir = out_dir / f"{stem}_sam2_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    overlay_list = []

    for i, (lab, sc, box) in enumerate(zip(labels, dino_scores, boxes)):
        m = masks[i]
        overlay_list.append(m)
        if m.ndim == 3:
            m = m.squeeze(0)
        if m.ndim != 2:
            raise RuntimeError(f"Mask[{i}] is not 2D after squeeze: shape={m.shape}")
        safe_lab = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in lab)
        mask_path = masks_dir / f"{i:02d}_{safe_lab}.png"
        save_mask_png(m, mask_path)

        saved.append({
            "index": i,
            "label": lab,
            "dino_score": sc,
            "box_xyxy": box,
            "mask_path": str(mask_path),
        })

    overlay_path = out_dir / f"{stem}_sam2_overlay.jpg"
    overlay_masks_on_image(image, overlay_list, overlay_path)

    out_json = out_dir / f"{stem}_sam2_results.json"
    out_json.write_text(json.dumps({
        "image": str(image_path),
        "sam2_model_id": args.model_id,
        "device": str(device),
        "mask_threshold": args.mask_threshold,
        "detections_used": saved
    }, indent=2))

    print(f"Saved masks to: {masks_dir}")
    print(f"Saved overlay to: {overlay_path}")
    print(f"Saved results to: {out_json}")


if __name__ == "__main__":
    main()
