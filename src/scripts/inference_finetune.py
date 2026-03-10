import os
import json
import argparse
from PIL import Image, ImageDraw

import torch
from unsloth import FastVisionModel
from transformers import AutoProcessor, AutoModelForImageTextToText


# ============================================
# ARGUMENTS
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL inference using COCO JSON"
    )

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--coco_json", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)

    parser.add_argument("--target_class", type=str, default="Pineapple")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=4096)

    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--save_coco_format", action="store_true")
    parser.add_argument("--path_save", default="/home/danny.xie/data/dxie/vlm-agriculture/train-qwen/tmp_visual")

    return parser.parse_args()


args = parse_args()


# ============================================
# LOAD MODEL
# ============================================

print("Loading model...")

def load_model(model_id, device):
    print("Loading Qwen VL model...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model

#model, tokenizer = FastVisionModel.from_pretrained(
#    args.model_path,
#    load_in_4bit=True,
#)
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, model = load_model(args.model_path, device)

#FastVisionModel.for_inference(model)

print("Model loaded.")

# ============================================
# PROMPT
# ============================================

def create_prompt(target):
    return (
        f"Outline the position of {target} and output all the coordinates in JSON format."
    )


# ============================================
# SAFE JSON PARSER
# ============================================

def extract_json(text):
    try:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == -1:
            return []
        return json.loads(text[start:end])
    except Exception:
        return []


# ============================================
# LOAD COCO
# ============================================

print("Loading COCO file...")

with open(args.coco_json, "r") as f:
    coco = json.load(f)

images_info = coco["images"]

print(f"Total images in COCO: {len(images_info)}")


# ============================================
# INFERENCE
# ============================================

all_coco_predictions = []

for image_info in images_info:

    image_id = image_info["id"]
    file_name = image_info["file_name"]

    image_path = os.path.join(args.image_folder, file_name)

    if not os.path.exists(image_path):
        print(f"Skipping missing: {image_path}")
        continue

    print(f"\nProcessing: {file_name}")

    image = Image.open(image_path).convert("RGB")

    original_width, original_height = image.size

    image_resized = image.resize(
        (args.resolution, args.resolution),
        Image.Resampling.LANCZOS
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": create_prompt(args.target_class)},
                {"type": "image", "image": image_resized},
            ],
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.2,
            do_sample=False,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    parsed = extract_json(response)

    print("Detections:", parsed)

    # ----------------------------------------
    # Convert back to original resolution
    # ----------------------------------------

    scale_x = original_width / args.resolution
    scale_y = original_height / args.resolution

    converted_predictions = []

    for obj in parsed:

        if "bbox_2d" not in obj:
            continue

        x1, y1, x2, y2 = obj["bbox_2d"]

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        width = x2 - x1
        height = y2 - y1

        converted_predictions.append({
            "image_id": image_id,
            "category_id": 1,  # assume single-class detection
            "bbox": [x1, y1, width, height],
            "score": 1.0  # Qwen does not output confidence by default
        })

    all_coco_predictions.extend(converted_predictions)

    # ----------------------------------------
    # Visualization
    # ----------------------------------------

    if args.save_vis and len(converted_predictions) > 0:

        draw = ImageDraw.Draw(image)

        for pred in converted_predictions:
            x, y, w, h = pred["bbox"]
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Create new save path
        save_path = os.path.join(args.path_save, base_name + "_vis.jpg")
        image.save(save_path)
        print("Saved:", save_path)


# ============================================
# SAVE COCO FORMAT PREDICTIONS
# ============================================

if args.save_coco_format:

    output_path = "coco_predictions.json"

    with open(output_path, "w") as f:
        json.dump(all_coco_predictions, f, indent=4)

    print("\nSaved COCO predictions:", output_path)

print("\nDone.")