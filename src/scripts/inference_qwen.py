import torch
import json
import argparse
import os
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm


# -----------------------------
# Args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--coco_gt", type=str, required=True,
                        help="Path to COCO ground truth annotation file")

    parser.add_argument("--image_folder", type=str, required=True,
                        help="Path to folder containing images")

    parser.add_argument("--output_json", type=str,
                        default="predictions.json")

    parser.add_argument("--save_vis_folder", type=str, default=None,
                        help="Folder to save images with predicted bounding boxes")

    parser.add_argument("--model_id", type=str,
                        default="unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit")

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--max_tokens", type=int, default=4096)

    parser.add_argument("--target", type=str, default="pineapple",
                        help="Target object to detect")

    parser.add_argument("--category_id", type=int, default=1,
                        help="COCO category_id for predictions")

    parser.add_argument("--resize", type=int, default=1000,
                        help="Resize image to square size before LLM")

    return parser.parse_args()


# -----------------------------
# Model Loader
# -----------------------------
def load_model(model_id, device):
    print("Loading Qwen VL model...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


# -----------------------------
# LLM Detection
# -----------------------------
def detect_objects(processor, model, image, device, max_tokens, target):

    prompt = (
        #f"Locate every instance that belongs to the following categories: \"{target}\". Report bbox coordinates in JSON format."
        #f"Detect all the {target}."
        #f"Carefully detect all the {target}."
        f"Outline the position of {target} and output all the coordinates in JSON format."
    )

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
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            top_p=0.8,
            top_k=20,
            temperature=0.1,
            repetition_penalty=1.0,
            #presence_penalty=0.5,
        )

    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, output)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    return response


# -----------------------------
# Extract JSON safely
# -----------------------------
def extract_json(text):
    start = text.find("[")
    end = text.rfind("]") + 1

    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end])
        except Exception:
            pass

    return []


# -----------------------------
# Convert resized boxes → COCO format
# -----------------------------
def convert_to_coco(boxes, image_id, category_id, orig_w, orig_h, model_size=1000):

    coco_preds = []

    scale_x = orig_w / model_size
    scale_y = orig_h / model_size

    for box in boxes:

        if "bbox_2d" not in box:
            continue

        x1, y1, x2, y2 = box["bbox_2d"]

        # Scale back to original resolution
        x1 = float(x1) * scale_x
        x2 = float(x2) * scale_x
        y1 = float(y1) * scale_y
        y2 = float(y2) * scale_y

        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)

        coco_preds.append({
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x1, y1, width, height],
            "score": 1.0
        })

    return coco_preds


# -----------------------------
# Draw and Save Bounding Boxes
# -----------------------------
def save_image_with_boxes(image, boxes, save_path, label_name="object"):

    draw = ImageDraw.Draw(image)

    for box in boxes:

        x, y, w, h = box["bbox"]

        x2 = x + w
        y2 = y + h

        # Draw rectangle
        draw.rectangle([x, y, x2, y2], outline="red", width=3)

        # Draw label
        draw.text((x, max(0, y - 12)), label_name, fill="red")

    image.save(save_path)


# -----------------------------
# Main
# -----------------------------
def main():

    args = parse_args()

    # -----------------------------
    # Load COCO Ground Truth
    # -----------------------------
    print("Loading COCO ground truth...")
    with open(args.coco_gt, "r") as f:
        coco_gt = json.load(f)

    images_info = coco_gt["images"]
    print(f"Found {len(images_info)} images in COCO GT")

    # -----------------------------
    # Load Model
    # -----------------------------
    processor, model = load_model(args.model_id, args.device)

    all_predictions = []

    if args.save_vis_folder is not None:
        os.makedirs(args.save_vis_folder, exist_ok=True)

    # -----------------------------
    # Iterate over COCO images
    # -----------------------------
    for img_info in tqdm(images_info):

        image_id = img_info["id"]
        file_name = img_info["file_name"]

        image_path = os.path.join(args.image_folder, file_name)

        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            continue

        # Load original image
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        # Resize BEFORE sending to LLM
        image_resized = image.resize(
            (512, 512),
            Image.BICUBIC
        )

        # Run detection
        raw_output = detect_objects(
            processor,
            model,
            image_resized,
            args.device,
            args.max_tokens,
            args.target
        )

        boxes = extract_json(raw_output)

        # Convert to COCO format (original resolution)
        coco_preds = convert_to_coco(
            boxes,
            image_id,
            args.category_id,
            orig_w,
            orig_h,
            model_size=args.resize
        )

        all_predictions.extend(coco_preds)

        # Save visualization if enabled
        if args.save_vis_folder is not None:

            save_path = os.path.join(args.save_vis_folder, file_name)

            save_image_with_boxes(
                image.copy(),
                coco_preds,
                save_path,
                label_name=args.target
            )

    # -----------------------------
    # Save predictions JSON
    # -----------------------------
    with open(args.output_json, "w") as f:
        json.dump(all_predictions, f, indent=2)

    print(f"\n✅ Saved {len(all_predictions)} predictions to {args.output_json}")


if __name__ == "__main__":
    main()
"""
python /home/danny.xie/data/dxie/vlm-agriculture/inference_qwen.py \
    --coco_gt /home/danny.xie/data/dxie/vlm-agriculture/dataset/Video_dataset_kfolds/COCO/kfold_10_imgs/fold_1/train.json \
    --image_folder /home/danny.xie/data/dxie/vlm-agriculture/dataset/high_resolution/train \
    --output_json /home/danny.xie/data/dxie/vlm-agriculture/Qwen3-vl/tmp.json \
    --target pineapple \
    --category_id 1 \
    --resize 512 \
    --model_id unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit \
    --save_vis_folder /home/danny.xie/data/dxie/vlm-agriculture/Qwen3-vl/tmp

    kfold_10_imgs
    kfold_50_imgs
    kfold_all_imgs
    """