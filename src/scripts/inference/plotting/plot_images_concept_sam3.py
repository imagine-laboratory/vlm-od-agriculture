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


# ==============================
# IMAGE SIZE
# ==============================
TARGET_W, TARGET_H = 1920, 1088


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_json", type=str, required=True, help="Used only to get image ids & filenames")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bpe_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="sam3_text_outputs")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--text_prompt", type=str, default="pineapple")
    return parser.parse_args()


def load_model(device, bpe_path, conf):
    print("Loading SAM3 model...")
    model = build_sam3_image_model(bpe_path=bpe_path)
    model = model.to(device)
    model.eval()
    return Sam3Processor(model, confidence_threshold=conf)


# ==============================
# TEXT PROMPT DETECTION
# ==============================
def run_text_detection(processor, image, target_text):
    """
    Runs SAM3 text-prompt detection

    Returns:
        boxes_xywh (list[list[float]])
        scores (list[float])
        masks (list[np.ndarray])
    """
    state = processor.set_image(image)
    processor.reset_all_prompts(state)

    state = processor.set_text_prompt(state=state, prompt=target_text)

    if "boxes" not in state or len(state["boxes"]) == 0:
        return [], [], []

    boxes = state["boxes"]   # normalized cxcywh
    scores = state["scores"]
    masks = state["masks"]

    width, height = image.size
    boxes_xywh = []
    final_masks = []

    for box, mask in zip(boxes, masks):
        box = torch.tensor(box).view(1, 4)
        box_xywh = box_convert(box, in_fmt="cxcywh", out_fmt="xywh")[0]

        # scale to pixel coords
        box_xywh[0] *= width
        box_xywh[1] *= height
        box_xywh[2] *= width
        box_xywh[3] *= height

        boxes_xywh.append(box_xywh.tolist())
        final_masks.append(mask.detach().cpu().numpy())

    return boxes_xywh, scores, final_masks


# ==============================
# VISUALIZATION HELPERS
# ==============================
def clip_bbox_xywh(bbox, img_w, img_h):
    x, y, w, h = bbox
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return [x, y, w, h]


def draw_bbox_safe(image_np, bbox, score=None):
    x, y, w, h = map(int, bbox)
    h_img, w_img = image_np.shape[:2]

    x1 = max(0, min(x, w_img - 1))
    y1 = max(0, min(y, h_img - 1))
    x2 = max(0, min(x + w, w_img - 1))
    y2 = max(0, min(y + h, h_img - 1))

    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if score is not None:
        cv2.putText(
            image_np,
            f"{score:.2f}",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return image_np


def overlay_mask(image_np, mask):
    if mask.ndim == 3:
        mask = mask[0]

    mask = cv2.resize(
        mask.astype(np.float32),
        (image_np.shape[1], image_np.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    mask = mask > 0.5
    color = np.array([255, 0, 0])
    alpha = 0.5

    image_np[mask] = image_np[mask] * (1 - alpha) + color * alpha
    return image_np


# ==============================
# LOAD IMAGE FILENAMES FROM COCO JSON
# ==============================
def load_image_list(path):
    with open(path, "r") as f:
        coco = json.load(f)

    id_to_filename = {}

    if isinstance(coco, dict) and "images" in coco:
        for img in coco["images"]:
            id_to_filename[img["id"]] = img["file_name"]

    return id_to_filename


# ==============================
# MAIN
# ==============================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    processor = load_model(args.device, args.bpe_path, args.conf)
    id_to_filename = load_image_list(args.coco_json)

    print(f"Processing {len(id_to_filename)} images using text prompt: '{args.text_prompt}'")

    for image_id, file_name in tqdm(id_to_filename.items()):
        image_path = os.path.join(args.image_dir, file_name)
        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path).convert("RGB")
        image = image.resize((TARGET_W, TARGET_H), Image.BILINEAR)
        image_np = np.array(image).copy()

        boxes, scores, masks = run_text_detection(processor, image, args.text_prompt)

        for bbox, score, mask in zip(boxes, scores, masks):
            if score < args.conf:
                continue

            bbox = clip_bbox_xywh(bbox, TARGET_W, TARGET_H)
            image_np = overlay_mask(image_np, mask)
            image_np = draw_bbox_safe(image_np, bbox, score)

        out_path = os.path.join(args.output_dir, file_name)
        plt.figure(figsize=(12, 7))
        plt.imshow(image_np)
        plt.axis("off")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    print("✅ Done! Text-based detections saved.")


if __name__ == "__main__":
    main()



"""
python /home/danny.xie/data/dxie/vlm-agriculture/plot_images_concept_sam3.py \
  --coco_json /home/danny.xie/data/dxie/vlm-agriculture/dataset/high_resolution/train/_annotations.coco.json \
  --image_dir /home/danny.xie/data/dxie/vlm-agriculture/dataset/high_resolution/train \
  --bpe_path /home/danny.xie/data/dxie/vlm-agriculture/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
  --device cuda \
  --output_dir sam3_segmented_outputs_concept \
  --text_prompt "pineapple" 

"""