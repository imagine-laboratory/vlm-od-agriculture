import os
import csv
import json
import torch
import argparse
import torchvision
from PIL import Image
from tqdm import tqdm

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# -----------------------------
# Args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_json", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target", type=str, default="pineapple")
    parser.add_argument("--output", type=str, default="sam3_coco_predictions.csv")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--bpe_path", type=str, required=True,
                        help="Path to sam3 BPE vocab file")
    return parser.parse_args()


# -----------------------------
# Model
# -----------------------------
def load_model(device, bpe_path, conf):
    print("Loading SAM3 model...")
    model = build_sam3_image_model(bpe_path=bpe_path)
    model = model.to(device)
    model.eval()

    processor = Sam3Processor(model, confidence_threshold=conf)
    return processor


# -----------------------------
# SAM3 Detection
# -----------------------------
def run_detection(processor, image, target):
    """
    Returns:
        boxes_xywh (list[list[float]]): bounding boxes
        scores (list[float]): confidence scores
    """
    state = processor.set_image(image)
    processor.reset_all_prompts(state)

    state = processor.set_text_prompt(state=state, prompt=target)

    boxes_xywh = []
    scores = []

    if "boxes" not in state:
        return boxes_xywh, scores

    for box, score in zip(state["boxes"], state["scores"]):
        box = box.detach().cpu()  # xyxy tensor

        xywh = torchvision.ops.box_convert(
            box.unsqueeze(0), in_fmt="xyxy", out_fmt="xywh"
        ).squeeze(0)

        boxes_xywh.append(xywh.tolist())
        scores.append(float(score.item()))

    return boxes_xywh, scores


# -----------------------------
# Main COCO Loop
# -----------------------------
def main():
    args = parse_args()

    processor = load_model(args.device, args.bpe_path, args.conf)

    print(f"Loading COCO annotations from {args.coco_json}")
    with open(args.coco_json, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    rows = []

    print(f"Running SAM3 detection on {len(images)} images...\n")

    for img_info in tqdm(images):
        file_name = img_info["file_name"]
        image_path = os.path.join(args.image_dir, file_name)

        if not os.path.exists(image_path):
            print(f"⚠️ Missing image: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")

        boxes, scores = run_detection(processor, image, args.target)

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
python inference_sam3.py \
  --coco_json /home/danny.xie/data/dxie/vlm-agriculture/seed/test_fold_0.json \
  --image_dir /home/danny.xie/data/dxie/vlm-agriculture/dataset/train \
  --bpe_path /home/danny.xie/data/dxie/vlm-agriculture/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
  --target pineapple \
  --output /home/danny.xie/data/dxie/vlm-agriculture/outputs/SAM3/test_fold_0.csv
"""