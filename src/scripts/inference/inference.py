import os
import json
import csv
import time
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import supervision as sv
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen VLM COCO Inference")

    parser.add_argument("--coco_ann", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit")
    parser.add_argument("--target", type=str, default=None, help="Single class to detect (optional)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--limit_images", type=int, default=None)
    parser.add_argument("--per_image_class_limit", type=int, default=None)

    return parser.parse_args()


def load_model(model_id, device):
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id).to(device)
    return processor, model


def load_coco(coco_ann_path):
    with open(coco_ann_path) as f:
        coco = json.load(f)

    images_info = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    name_to_catid = {v: k for k, v in categories.items()}

    image_targets = defaultdict(set)
    for ann in coco["annotations"]:
        image_targets[ann["image_id"]].add(categories[ann["category_id"]])

    print(f"Loaded {len(images_info)} images | {len(categories)} categories")
    return images_info, categories, name_to_catid, image_targets


def qwen_detect(processor, model, image: Image, target: str, device, max_tokens):
    prompt = f"Detect every {target} in this image and output bounding boxes in JSON."

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_tokens)

    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, output)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def annotate_image(image: Image, detections: sv.Detections):
    thickness = sv.calculate_optimal_line_thickness(image.size)
    text_scale = sv.calculate_optimal_text_scale(image.size)

    box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.DEFAULT, thickness=thickness)
    label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.DEFAULT, text_scale=text_scale)

    img = box_annotator.annotate(image.copy(), detections)
    return label_annotator.annotate(img, detections)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    processor, model = load_model(args.model_id, args.device)
    images_info, categories, name_to_catid, image_targets = load_coco(args.coco_ann)

    csv_rows = []
    coco_predictions = []

    image_items = list(images_info.items())
    if args.limit_images:
        image_items = image_items[:args.limit_images]

    for image_id, img_info in image_items:
        img_path = os.path.join(args.image_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        print(f"\nProcessing {img_info['file_name']}")
        image = Image.open(img_path).convert("RGB")

        if args.target:
            targets = [args.target]
        else:
            targets = list(image_targets[image_id])
            if args.per_image_class_limit:
                targets = targets[:args.per_image_class_limit]

        all_detections = []
        start_total = time.perf_counter()

        for target in targets:
            try:
                response = qwen_detect(processor, model, image, target, args.device, args.max_tokens)

                detections = sv.Detections.from_vlm(
                    vlm=sv.VLM.QWEN_3_VL,
                    result=response,
                    resolution_wh=image.size
                )

                all_detections.append(detections)

                for box, score in zip(detections.xyxy, detections.confidence):
                    x1, y1, x2, y2 = box
                    coco_predictions.append({
                        "image_id": image_id,
                        "category_id": name_to_catid.get(target, 1),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score)
                    })

            except Exception as e:
                print(f"  Failed on {target}: {e}")

        total_time = time.perf_counter() - start_total

        if all_detections:
            merged = sv.Detections.merge(all_detections)
            annotated = annotate_image(image, merged)
            annotated.save(os.path.join(args.output_dir, f"{image_id}_annotated.jpg"))
            status = "success"
        else:
            status = "failed"

        csv_rows.append([img_info["file_name"], status, round(total_time, 3)])

    # Save outputs
    with open(os.path.join(args.output_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "status", "total_time"])
        writer.writerows(csv_rows)

    with open(os.path.join(args.output_dir, args.output_file), "w") as f:
        json.dump(coco_predictions, f)

    print("\n✅ All done!")


if __name__ == "__main__":
    main()
