import torch
import json
import argparse
from collections import defaultdict
from transformers import set_seed
import wandb

set_seed(42)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("bf16 supported:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
COCO_JSON_PATH  = "/home/danny.xie/data/dxie/vlm-agriculture/dataset/Video-split-train-test-val/ds_1/train_full.json"   # <-- update
COCO_JSON_PATH_VAL = "/home/danny.xie/data/dxie/vlm-agriculture/dataset/Video-split-train-test-val/ds_1/val.json"   # <-- update
IMAGES_DIR      = "/home/danny.xie/data/dxie/vlm-agriculture/dataset/Video-split-train-test-val/images/"            # <-- update
QWEN_RESOLUTION = 1000
MAX_TARGET_CHARS = 5000
MAX_IMAGE_SIDE   = 1024
MAX_IMAGE_PIXELS = 640
MAX_LEN          = 4096
TASK             = "detection"


# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL on COCO with LoRA")

    # Data / model
    parser.add_argument("--coco_json_path",  type=str, default=COCO_JSON_PATH)
    parser.add_argument("--val_json_path",   type=str, default=COCO_JSON_PATH_VAL)
    parser.add_argument("--images_dir",      type=str, default=IMAGES_DIR)
    parser.add_argument("--model_id",        type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--max_samples",     type=int, default=800)
    parser.add_argument("--max_val_samples", type=int, default=200)
    parser.add_argument("--task",            type=str, default=TASK)
    parser.add_argument("--output_dir",      type=str, default="/home/danny.xie/data/dxie/vlm-agriculture/train-qwen/models")

    # Training
    parser.add_argument("--num_train_epochs",             type=int,   default=1)
    parser.add_argument("--per_device_train_batch_size",  type=int,   default=2)
    parser.add_argument("--gradient_accumulation_steps",  type=int,   default=4)
    parser.add_argument("--learning_rate",                type=float, default=1e-4)
    parser.add_argument("--warmup_steps",                 type=int,   default=10)
    parser.add_argument("--weight_decay",                 type=float, default=0.01)
    parser.add_argument("--max_grad_norm",                type=float, default=1.0)
    parser.add_argument("--max_len",                      type=int,   default=MAX_LEN)
    parser.add_argument("--early_stopping_patience",      type=int,   default=5)

    # Eval / checkpointing
    parser.add_argument("--eval_strategy", type=str, default="epoch")   # "no", "steps", "epoch"
    parser.add_argument("--save_strategy", type=str, default="epoch")   # "no", "steps", "epoch"
    parser.add_argument("--load_best_model_at_end", type=bool, default=True)
    parser.add_argument("--metric_for_best_model",  type=str, default="eval_loss")

    # LoRA
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # W&B
    parser.add_argument("--wandb_project", type=str, default="qwen3vl-coco-detection")
    parser.add_argument("--wandb_entity",  type=str, default="imagine-laboratory-conare",
                        help="W&B entity (username or team). Omit to use the default.")

    return parser.parse_args()


# ─────────────────────────────────────────────
# W&B INIT
# ─────────────────────────────────────────────
def init_wandb(args: argparse.Namespace) -> None:
    """Initialise a Weights & Biases run with the full argument config."""
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
    )


# ─────────────────────────────────────────────
# YOUR PROVIDED HELPERS (unchanged)
# ─────────────────────────────────────────────
def create_prompt(target: str) -> str:
    return (
        f"Outline the position of {target} and output all the coordinates in JSON format."
    )


def convert_bbox_to_qwen(bbox: list, width: int, height: int, qwen_resolution: int) -> list:
    x, y, w, h = bbox
    scale_x = qwen_resolution / width
    scale_y = qwen_resolution / height
    x1 = int(x * scale_x)
    y1 = int(y * scale_y)
    x2 = int((x + w) * scale_x)
    y2 = int((y + h) * scale_y)
    return [x1, y1, x2, y2]


def load_coco_dataset(coco_json_path: str):
    with open(coco_json_path, "r") as f:
        coco = json.load(f)
    images     = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    annotations_per_image: defaultdict = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_per_image[ann["image_id"]].append(ann)
    return images, categories, annotations_per_image


# ─────────────────────────────────────────────
# BUILD A HF-COMPATIBLE DATASET FROM COCO
# ─────────────────────────────────────────────
import os
from PIL import Image
from datasets import Dataset


def build_hf_dataset_from_coco(
    coco_json_path: str,
    images_dir: str,
    max_samples: int = 800,
) -> Dataset:
    images_meta, categories, annotations_per_image = load_coco_dataset(coco_json_path)

    records = []
    for img_id, img_meta in images_meta.items():
        anns = annotations_per_image.get(img_id, [])
        if not anns:
            continue

        img_path = os.path.join(images_dir, img_meta["file_name"])
        if not os.path.exists(img_path):
            continue

        w, h = img_meta["width"], img_meta["height"]

        category_names = [categories[a["category_id"]] for a in anns]
        bboxes_qwen    = [
            convert_bbox_to_qwen(a["bbox"], w, h, QWEN_RESOLUTION) for a in anns
        ]

        records.append({
            "image_path":     img_path,
            "image_id":       img_id,
            "category_names": category_names,
            "bboxes_qwen":    bboxes_qwen,
        })

        if len(records) >= max_samples:
            break

    def load_image(rec):
        rec["image"] = Image.open(rec["image_path"]).convert("RGB")
        return rec

    ds = Dataset.from_list(records)
    ds = ds.map(load_image)
    return ds


# ─────────────────────────────────────────────
# PROMPT / TARGET BUILDERS
# ─────────────────────────────────────────────
def build_prompt() -> str:
    target_str     = "pineapple"
    return create_prompt(target_str)


def build_target(example: dict) -> str:
    detections = [
        {"bbox": bbox, "label": name}
        for name, bbox in zip(example["category_names"], example["bboxes_qwen"])
    ]
    return json.dumps(detections, separators=(",", ":"))


def clamp_text(s: str, max_chars: int = MAX_TARGET_CHARS) -> str:
    s = (s or "").strip()
    return s if len(s) <= max_chars else s[:max_chars].rstrip()


# ─────────────────────────────────────────────
# IMAGE RESIZE
# ─────────────────────────────────────────────
def _resize_pil(pil: Image.Image, max_side=MAX_IMAGE_SIDE, max_pixels=MAX_IMAGE_PIXELS) -> Image.Image:
    pil = pil.convert("RGB")
    w, h = pil.size
    scale_side = min(1.0, max_side / float(max(w, h)))
    scale_area = (max_pixels / float(w * h)) ** 0.5 if (w * h) > max_pixels else 1.0
    scale = min(scale_side, scale_area)
    if scale < 1.0:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        pil = pil.resize((nw, nh), resample=Image.BICUBIC)
    return pil

def resize_keep_ratio(pil, target_w=640):
    pil = pil.convert("RGB")
    w, h = pil.size
    scale = target_w / w
    new_h = int(round(h * scale))
    return pil.resize((target_w, new_h), resample=Image.BICUBIC)

# ─────────────────────────────────────────────
# MESSAGES FORMATTER
# ─────────────────────────────────────────────
def to_messages(example: dict) -> dict:
    prompt = build_prompt()
    target = clamp_text(build_target(example))
    example["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target}],
        },
    ]
    return example


# ─────────────────────────────────────────────
# COLLATOR
# ─────────────────────────────────────────────
from typing import List, Dict, Any

def collate_fn(batch: List[Dict[str, Any]], processor=None, max_len=MAX_LEN):
    full_texts = [
        processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        for ex in batch
    ]
    prompt_texts = [
        processor.apply_chat_template(ex["messages"][:-1], tokenize=False, add_generation_prompt=True)
        for ex in batch
    ]
    images = [resize_keep_ratio(ex["image"]) for ex in batch]

    enc = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )

    input_ids = enc["input_ids"]
    pad_id    = processor.tokenizer.pad_token_id

    prompt_ids = processor.tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    )["input_ids"]

    prompt_lens = (prompt_ids != pad_id).sum(dim=1)
    labels      = input_ids.clone()
    bs, seqlen  = labels.shape

    for i in range(bs):
        pl = min(int(prompt_lens[i].item()), seqlen)
        labels[i, :pl] = -100

    labels[labels == pad_id] = -100
    enc["labels"] = labels
    return enc


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    # ── W&B init ──────────────────────────────
    init_wandb(args)

    # ── Model + processor ─────────────────────
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, EarlyStoppingCallback

    model     = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    # ── Dataset ───────────────────────────────
    train_ds = build_hf_dataset_from_coco(
        coco_json_path=args.coco_json_path,
        images_dir=args.images_dir,
        max_samples=args.max_samples,
    )
    train_ds = train_ds.shuffle(seed=42).map(to_messages)
    print("Train samples:", len(train_ds))

    val_ds = build_hf_dataset_from_coco(
        coco_json_path=args.val_json_path,
        images_dir=args.images_dir,
        max_samples=args.max_val_samples,
    )
    val_ds = val_ds.map(to_messages)
    print("Val samples:  ", len(val_ds))

    # ── Baseline check ────────────────────────
    def run_inference(model_, example, max_new_tokens=4096):
        prompt = build_prompt()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": resize_keep_ratio(example["image"])}, #_resize_pil(example["image"])},
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
        ).to(model_.device)

        with torch.inference_mode():
            out = model_.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)

        gen = out[0][inputs["input_ids"].shape[1]:]
        return processor.decode(gen, skip_special_tokens=True)

    baseline_ex     = train_ds.shuffle(seed=120).select(range(5))[3]
    baseline_output = run_inference(model, baseline_ex)
    baseline_target = clamp_text(build_target(baseline_ex), 1500)
    print("\n--- BASELINE EXAMPLE ---\n", baseline_ex)
    print("\n--- BASELINE OUTPUT ---\n", baseline_output)
    print("\n--- TARGET (dataset) ---\n", baseline_target)

    wandb.log({
        "baseline/output": baseline_output,
        "baseline/target": baseline_target,
    })

    # ── LoRA + Trainer ────────────────────────
    from peft import LoraConfig, TaskType, get_peft_model
    from trl  import SFTTrainer, SFTConfig
    from functools import partial

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "lm_head",

            # attention
            "q_proj", "k_proj", "v_proj", "o_proj",
            "attn.qkv", "attn.proj",

            # mlp
            "linear_fc1", "linear_fc2",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    #model = get_peft_model(model, lora)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.4f}%")


    output_dir = f"{args.output_dir}/qwen3vl-{args.task}-lora"

    sft_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=False,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        fp16=False,
        lr_scheduler_type="cosine",
        logging_steps=1,
        report_to="wandb",       # ← hand off step-level metrics to W&B
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=partial(collate_fn, processor=processor, max_len=args.max_len),
        peft_config=lora,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    )

    trainer.train()

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # ── W&B finish ────────────────────────────
    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()