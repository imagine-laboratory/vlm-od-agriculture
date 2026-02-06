import os
import csv
import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.ops as ops

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# -----------------------------
# Args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_json", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="sam2_auto_masks.csv")

    # SAM2 specific
    parser.add_argument("--sam2_checkpoint", type=str, required=True)
    parser.add_argument("--model_cfg", type=str, required=True)

    return parser.parse_args()


# -----------------------------
# Model
# -----------------------------
def load_model(device, model_cfg, checkpoint):
    print("Loading SAM2 model for automatic mask generation...")
    sam2 = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    return mask_generator


# -----------------------------
# Mask → Bounding Box
# -----------------------------
def mask_to_xywh(mask):
    """Convert binary mask to COCO-style xywh box"""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


# -----------------------------
# SAM2 Auto Segmentation
# -----------------------------
def run_sam2_auto(mask_generator, image_np):
    """
    Returns:
        boxes_xywh (list[list[float]])
        scores (list[float]) — using mask area as pseudo-score
    """
    masks = mask_generator.generate(image_np)

    boxes_xywh = []
    scores = []

    for ann in masks:
        mask = ann["segmentation"]
        box = mask_to_xywh(mask)
        if box is None:
            continue

        boxes_xywh.append(box)
        scores.append(float(ann["area"]))  # SAM2 doesn't give confidence, using area instead

    return boxes_xywh, scores


# -----------------------------
# Main COCO Loop
# -----------------------------
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    mask_generator = load_model(device, args.model_cfg, args.sam2_checkpoint)

    print(f"Loading COCO annotations from {args.coco_json}")
    with open(args.coco_json, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    rows = []

    print(f"Running SAM2 automatic segmentation on {len(images)} images...\n")

    for img_info in tqdm(images):
        file_name = img_info["file_name"]
        image_path = os.path.join(args.image_dir, file_name)

        if not os.path.exists(image_path):
            print(f"⚠️ Missing image: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        boxes, scores = run_sam2_auto(mask_generator, image_np)

        rows.append({
            "filename": file_name,
            "detections": json.dumps(boxes),
            "scores": json.dumps(scores),
        })

    print(f"\nSaving CSV results to {args.output}")

    with open(args.output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["filename", "detections", "scores"])
        writer.writeheader()
        writer.writerows(rows)

    print("✅ Done!")


if __name__ == "__main__":
    main()

"""
python /home/danny.xie/data/dxie/vlm-agriculture/inference_sam2.py \
  --coco_json /home/danny.xie/data/dxie/vlm-agriculture/seed/test_fold_0.json \
  --image_dir /home/danny.xie/data/dxie/vlm-agriculture/dataset/train \
  --sam2_checkpoint /home/danny.xie/data/dxie/vlm-agriculture/sam2/checkpoints/sam2.1_hiera_base_plus.pt \
  --model_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
  --device cuda \
  --output sam2_auto_masks.csv

"""