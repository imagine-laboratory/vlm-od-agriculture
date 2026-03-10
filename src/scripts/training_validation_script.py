import os
import json
import argparse
from collections import defaultdict
from PIL import Image

import torch
from datasets import Dataset

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


# ============================================
# ARGUMENTS
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Qwen3-VL for object detection using COCO format"
    )

    parser.add_argument("--model_name", type=str,
                        default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit")

    parser.add_argument("--coco_train_json", type=str, required=True)
    parser.add_argument("--coco_val_json", type=str, required=True)

    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="qwen_detection")

    parser.add_argument("--target_class", type=str, default="Pineapple")
    parser.add_argument("--qwen_resolution", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=512)

    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)

    # ==============================
    # Fine-tuning control flags
    # ==============================

    parser.add_argument("--finetune_vision", action="store_true",
                        help="Enable LoRA on vision encoder")

    parser.add_argument("--finetune_language", action="store_true",
                        help="Enable LoRA on language model")

    parser.add_argument("--finetune_attention", action="store_true",
                        help="Enable LoRA on attention modules")

    parser.add_argument("--finetune_mlp", action="store_true",
                        help="Enable LoRA on MLP modules")

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    return parser.parse_args()


args = parse_args()


# ============================================
# LOAD MODEL
# ============================================

print("Loading model...")

model, tokenizer = FastVisionModel.from_pretrained(
    args.model_name,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=args.finetune_vision,
    finetune_language_layers=args.finetune_language,
    finetune_attention_modules=args.finetune_attention,
    finetune_mlp_modules=args.finetune_mlp,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
    random_state=3407,
)

FastVisionModel.for_training(model)


# ============================================
# COCO LOADER
# ============================================

def load_coco_dataset(coco_json_path):

    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    annotations_per_image = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_per_image[ann["image_id"]].append(ann)

    return images, categories, annotations_per_image


# ============================================
# BBOX CONVERSION
# ============================================

def convert_bbox_to_qwen(bbox, width, height):

    x, y, w, h = bbox

    scale_x = args.qwen_resolution / width
    scale_y = args.qwen_resolution / height

    x1 = int(x * scale_x)
    y1 = int(y * scale_y)
    x2 = int((x + w) * scale_x)
    y2 = int((y + h) * scale_y)

    #x1 = int(x)
    #y1 = int(y)
    #x2 = int((x + w))
    #y2 = int((y + h))

    return [x1, y1, x2, y2]


# ============================================
# PROMPT
# ============================================

def create_prompt(target):
    return f"Outline the position of {target} and output all the coordinates in JSON format."


# ============================================
# DATASET CONVERSION
# ============================================

def convert_dataset(images, categories, annotations_per_image):

    dataset = []

    for image_id, anns in annotations_per_image.items():

        image_info = images[image_id]

        image_path = os.path.join(
            args.image_folder,
            image_info["file_name"]
        )

        if not os.path.exists(image_path):
            continue

        width = image_info["width"]
        height = image_info["height"]

        objects = []

        for ann in anns:

            label = categories[ann["category_id"]]

            if label != args.target_class:
                continue

            bbox = convert_bbox_to_qwen(
                ann["bbox"],
                width,
                height
            )

            objects.append({
                "bbox_2d": bbox,
                "label": label
            })

        if len(objects) == 0:
            continue

        answer = json.dumps(objects)
        prompt = create_prompt(args.target_class)

        image = Image.open(image_path).convert("RGB")
        image = image.resize(
            (args.resolution, args.resolution),
            Image.Resampling.LANCZOS
        )

        dataset.append({
            "image": image,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
        })
    return dataset


# ============================================
# LOAD TRAIN + VAL
# ============================================

print("Loading COCO datasets...")

train_images, train_categories, train_annotations = load_coco_dataset(args.coco_train_json)
val_images, val_categories, val_annotations = load_coco_dataset(args.coco_val_json)

print("Converting train dataset...")
train_converted = convert_dataset(train_images, train_categories, train_annotations)

print("Converting val dataset...")
val_converted = convert_dataset(val_images, val_categories, val_annotations)

print("Train samples:", len(train_converted))
print("Val samples:", len(val_converted))


# ============================================
# HF DATASET
# ============================================

def to_hf_dataset(converted):

    data_dict = {
        "image": [],
        "messages": [],
    }

    for sample in converted:
        data_dict["image"].append(sample["image"])
        data_dict["messages"].append(sample["messages"])

    return Dataset.from_dict(data_dict)


train_dataset = to_hf_dataset(train_converted)
val_dataset   = to_hf_dataset(val_converted)


# ============================================
# TRAINER
# ============================================

print("Training...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    data_collator=UnslothVisionDataCollator(
        model,
        tokenizer
    ),

    args=SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=3,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        #num_train_epochs = 1,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=args.output_dir,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=2048,
    )
)

trainer.train()


# ============================================
# SAVE
# ============================================

print("Saving final model...")

model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

print("Done.")

"""
python /home/danny.xie/data/dxie/vlm-agriculture/train-qwen/training_validation_script.py \
  --model_name unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit \
  --coco_train_json /home/danny.xie/data/dxie/vlm-agriculture/dataset/Video-split-train-test-val/ds_1/train_full.json \
  --coco_val_json /home/danny.xie/data/dxie/vlm-agriculture/dataset/Video-split-train-test-val/ds_1/val.json \
  --image_folder /home/danny.xie/data/dxie/vlm-agriculture/dataset/Video-split-train-test-val/images \
  --output_dir /home/danny.xie/data/dxie/vlm-agriculture/train-qwen/qwen3-vl-models/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit/ds_1/ \
  --target_class pineapple \
  --resolution 512 \
  --max_steps 30 \
  --batch_size 2 \
  --grad_accum 4 \
  --learning_rate 2e-4 \
  --finetune_vision \
  --finetune_language \
  --finetune_attention \
  --finetune_mlp \
  --lora_r 32 \
  --lora_alpha 32 \
  --lora_dropout 0.0
"""