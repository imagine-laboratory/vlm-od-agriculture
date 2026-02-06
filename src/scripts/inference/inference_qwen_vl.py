import os
import json
import csv
import time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import supervision as sv

# -----------------------------
# Load Model
# -----------------------------

MODEL_ID = "unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID).to("cuda")

# -----------------------------
# Qwen Detection Function
# -----------------------------

def qwen_detect(image: Image, target: str, max_new_tokens: int = 1024):

    prompt = f"Outline the position of {target} and output all the coordinates in JSON format."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return text

# -----------------------------
# Annotation Function
# -----------------------------

COLOR = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
    "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

def annotate_image(image: Image, detections: sv.Detections, smart_position=True) -> Image:
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

    color_by_class = detections.class_id is not None
    box_annotator = sv.BoxAnnotator(
        color=COLOR,
        color_lookup=sv.ColorLookup.CLASS if color_by_class else sv.ColorLookup.INDEX,
        thickness=thickness
    )
    label_annotator = sv.LabelAnnotator(
        color=COLOR,
        color_lookup=sv.ColorLookup.CLASS if color_by_class else sv.ColorLookup.INDEX,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=thickness - 1,
        smart_position=smart_position
    )

    annotated = box_annotator.annotate(image.copy(), detections)
    return label_annotator.annotate(annotated, detections)

# -----------------------------
# Folder Processing
# -----------------------------

INPUT_FOLDER = "/home/danny.xie/data/dxie/dataset/pineapples/DJI_20240308111117_0010_V_2/train/images/"
OUTPUT_FOLDER = "/home/danny.xie/data/dxie/vlm-agriculture/Qwen3-VL-32B/piña/DJI_20240308111117_0010_V_2"
TARGET = "piña"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

results_csv_path = os.path.join(OUTPUT_FOLDER, "detection_results.csv")

csv_rows = []
header = ["image_name", "success_attempt", "status", "total_time_seconds"]

for img_name in os.listdir(INPUT_FOLDER):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(INPUT_FOLDER, img_name)
    print(f"\nProcessing: {img_name}")

    image = Image.open(img_path).convert("RGB")
    attempts = 0
    success_attempt = None
    detections = None

    timing_info = {"attempts": []}
    t0_total = time.perf_counter()

    # ----------------------------------------
    # Retry Loop (try 3 times)
    # ----------------------------------------
    while attempts < 3:
        attempts += 1
        print(f"  Attempt {attempts}...")

        t0 = time.perf_counter()
        attempt_record = {"attempt": attempts, "time_seconds": None, "status": None}

        try:
            response = qwen_detect(image, TARGET)

            try:
                detections = sv.Detections.from_vlm(
                    vlm=sv.VLM.QWEN_3_VL,
                    result=response,
                    resolution_wh=image.size
                )
                success_attempt = attempts
                attempt_record["status"] = "success"

                break

            except Exception as det_err:
                print(f"  Detection parsing failed: {det_err}")
                attempt_record["status"] = "parse_error"

        except Exception as gen_err:
            print(f"  Generation failed: {gen_err}")
            attempt_record["status"] = "generation_error"

        attempt_record["time_seconds"] = time.perf_counter() - t0
        timing_info["attempts"].append(attempt_record)

    total_time = time.perf_counter() - t0_total

    # ----------------------------------------
    # Save Output Files
    # ----------------------------------------
    base_name = os.path.splitext(img_name)[0]

    # Save total timing
    timing_info["total_time_seconds"] = total_time
    timing_json_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_timing.json")
    with open(timing_json_path, "w") as f:
        json.dump(timing_info, f, indent=4)

    # Save detections JSON
    if detections is not None:
        detections_json_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_detections.json")
        with open(detections_json_path, "w") as f:
            f.write(response)

        # Save annotated image
        annotated = annotate_image(image, detections)
        annotated.thumbnail((1000, 1000))
        out_img = os.path.join(OUTPUT_FOLDER, f"{base_name}_annotated.jpg")
        annotated.save(out_img)

        status = "success"
    else:
        status = "failed"

    # Add to CSV
    csv_rows.append([img_name, success_attempt, status, round(total_time, 4)])

# Save CSV summary
with open(results_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(csv_rows)

print("\nProcessing complete!")
print(f"Summary saved to: {results_csv_path}")