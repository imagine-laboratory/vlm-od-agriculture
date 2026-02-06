import os
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from tqdm import tqdm
from torchvision.ops import box_convert

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox


# ==============================
# CONFIG — ORIGINAL VS TARGET SIZE
# ==============================
ORIG_W, ORIG_H = 1920, 1088
TARGET_W, TARGET_H = 1920, 1088


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_json", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bpe_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="sam3_segmented_outputs")
    parser.add_argument("--conf", type=float, default=0.5)
    return parser.parse_args()


def load_model(device, bpe_path, conf):
    print("Loading SAM3 model...")
    model = build_sam3_image_model(bpe_path=bpe_path)
    model = model.to(device)
    model.eval()
    return Sam3Processor(model, confidence_threshold=conf)


# ==============================
# SCALE BBOX FROM 512 → 1920x1088
# ==============================
def scale_bbox_to_target(bbox):
    x, y, w, h = bbox

    scale_x = TARGET_W / ORIG_W
    scale_y = TARGET_H / ORIG_H

    return [
        x * scale_x,
        y * scale_y,
        w * scale_x,
        h * scale_y,
    ]


def clip_bbox_xywh(bbox, img_w, img_h):
    x, y, w, h = bbox
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return [x, y, w, h]


def draw_bbox_safe(image_np, bbox):
    x, y, w, h = map(int, bbox)
    h_img, w_img = image_np.shape[:2]

    x1 = max(0, min(x, w_img - 1))
    y1 = max(0, min(y, h_img - 1))
    x2 = max(0, min(x + w, w_img - 1))
    y2 = max(0, min(y + h, h_img - 1))

    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image_np


def overlay_mask(image_np, mask):
    if mask.ndim == 3:
        mask = mask[0]

    mask = cv2.resize(
        mask.astype(np.float32),
        (image_np.shape[1], image_np.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    mask = mask > 0.5
    color = np.array([255, 0, 0])
    alpha = 0.5

    image_np[mask] = image_np[mask] * (1 - alpha) + color * alpha
    return image_np


def segment_from_box(processor, image, bbox_xywh):
    width, height = image.size

    box_xywh = torch.tensor(bbox_xywh).view(1, 4)
    box_cxcywh = box_convert(box_xywh, in_fmt="xywh", out_fmt="cxcywh")
    norm_box = normalize_bbox(box_cxcywh, width, height).flatten().tolist()

    state = processor.set_image(image)
    processor.reset_all_prompts(state)
    state = processor.add_geometric_prompt(state=state, box=norm_box, label=True)

    if "masks" not in state or len(state["masks"]) == 0:
        return None

    return state["masks"][0].detach().cpu().numpy()


def load_coco_boxes(path):
    with open(path, "r") as f:
        coco = json.load(f)

    detections_per_image = {}
    id_to_filename = {}

    if isinstance(coco, dict) and "images" in coco:
        for img in coco["images"]:
            id_to_filename[img["id"]] = img["file_name"]

    if isinstance(coco, list):
        for det in coco:
            detections_per_image.setdefault(det["image_id"], []).append(det)
    elif "annotations" in coco:
        for ann in coco["annotations"]:
            detections_per_image.setdefault(ann["image_id"], []).append({
                "bbox": ann["bbox"],
                "score": ann.get("score", 1.0)
            })

    return detections_per_image, id_to_filename


# ==============================
# MAIN
# ==============================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    processor = load_model(args.device, args.bpe_path, args.conf)
    detections_per_image, id_to_filename = load_coco_boxes(args.coco_json)

    print(f"Processing {len(detections_per_image)} images...")

    for image_id, detections in tqdm(detections_per_image.items()):
        file_name = id_to_filename.get(image_id)
        if file_name is None:
            continue

        image_path = os.path.join(args.image_dir, file_name)
        if not os.path.exists(image_path):
            continue

        # 🔹 Load and RESIZE image to 1920x1088
        image = Image.open(image_path).convert("RGB")
        image = image.resize((TARGET_W, TARGET_H), Image.BILINEAR)
        image_np = np.array(image).copy()

        for det in detections:
            if det.get("score", 1.0) < args.conf:
                continue

            # 🔹 Scale bbox from 512 space → 1920x1088
            bbox = scale_bbox_to_target(det["bbox"])
            bbox = clip_bbox_xywh(bbox, TARGET_W, TARGET_H)

            mask = segment_from_box(processor, image, bbox)
            if mask is None:
                continue

            image_np = overlay_mask(image_np, mask)
            image_np = draw_bbox_safe(image_np, bbox)

        out_path = os.path.join(args.output_dir, file_name)
        plt.figure(figsize=(12, 7))
        plt.imshow(image_np)
        plt.axis("off")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    print("✅ Done! Visualizations saved.")


if __name__ == "__main__":
    main()



"""
python /home/danny.xie/data/dxie/vlm-agriculture/plot_images_segmentation_sam3.py \
  --coco_json /home/danny.xie/data/dxie/vlm-agriculture/dataset/high_resolution/train/_annotations.coco.json \
  --image_dir /home/danny.xie/data/dxie/vlm-agriculture/dataset/high_resolution/train \
  --bpe_path /home/danny.xie/data/dxie/vlm-agriculture/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
  --device cuda \
  --output_dir sam3_segmented_outputs

"""