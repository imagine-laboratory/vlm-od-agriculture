import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ==================================================
# Counting Metrics
# ==================================================
def compute_count_metrics(coco_gt, detections, iou_threshold=0.5):

    preds_by_image = {}
    for det in detections:
        preds_by_image.setdefault(det["image_id"], []).append(det)

    total_tp, total_fp, total_fn = 0, 0, 0
    abs_errors, biases = [], []

    for image_id in coco_gt.getImgIds():

        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))

        gt_boxes = [ann["bbox"] for ann in gt_anns]
        pred_boxes = [d["bbox"] for d in preds_by_image.get(image_id, [])]

        gt_count = len(gt_boxes)
        pred_count = len(pred_boxes)

        gt_xyxy = (
            torchvision.ops.box_convert(
                torch.tensor(gt_boxes, dtype=torch.float32),
                "xywh",
                "xyxy",
            )
            if gt_count > 0
            else torch.empty((0, 4))
        )

        pred_xyxy = (
            torchvision.ops.box_convert(
                torch.tensor(pred_boxes, dtype=torch.float32),
                "xywh",
                "xyxy",
            )
            if pred_count > 0
            else torch.empty((0, 4))
        )

        matched_gt = set()
        matched_pred = set()

        if len(gt_xyxy) > 0 and len(pred_xyxy) > 0:

            iou_matrix = torchvision.ops.box_iou(pred_xyxy, gt_xyxy)

            for p_idx in range(iou_matrix.shape[0]):

                best_iou, g_idx = torch.max(iou_matrix[p_idx], dim=0)

                if (
                    best_iou >= iou_threshold
                    and g_idx.item() not in matched_gt
                ):
                    matched_gt.add(g_idx.item())
                    matched_pred.add(p_idx)

        tp = len(matched_pred)
        fp = pred_count - tp
        fn = gt_count - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

        abs_errors.append(abs(pred_count - gt_count))
        biases.append(pred_count - gt_count)

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MAE": np.mean(abs_errors),
        "Bias": np.mean(biases),
    }


# ==================================================
# COCO Evaluation
# ==================================================
def evaluate_coco(gt_path, pred_path, iou_threshold=0.5):

    coco_gt = COCO(gt_path)

    with open(pred_path, "r") as f:
        detections = json.load(f)

    if len(detections) == 0:
        return None

    coco_dt = coco_gt.loadRes(detections)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.maxDets = [1, 10, 1000]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats

    results = {
        "mAP_50_95": stats[0],
        "mAP_50": stats[1],
        "mAP_75": stats[2],
        "AR_1": stats[6],
        "AR_10": stats[7],
        "AR_1000": stats[8],
    }

    results.update(
        compute_count_metrics(
            coco_gt,
            detections,
            iou_threshold=iou_threshold,
        )
    )

    return results


# ==================================================
# Experiment Runner
# ==================================================
def run_evaluation(args):

    all_results = []

    for model in args.models:

        for fold in args.folds:

            print(f"\n Evaluating {model} | {fold}")

            gt_path = (
                Path(args.dataset_root)
                / fold
                / args.gt_filename
            )

            pred_path = (
                Path(args.results_root)
                / model
                / fold
                / args.pred_filename
            )

            if not pred_path.exists():
                print(f"Missing predictions: {pred_path}")
                continue

            metrics = evaluate_coco(
                str(gt_path),
                str(pred_path),
                args.iou_threshold,
            )

            if metrics is None:
                continue

            metrics["model"] = model
            metrics["fold"] = fold

            all_results.append(metrics)

    df = pd.DataFrame(all_results)

    output_path = Path(args.output_csv)

    df.to_csv(output_path, index=False)

    print(f"\n Saved results to: {output_path}")
    print(df.groupby("model").mean(numeric_only=True))


# ==================================================
# Args
# ==================================================
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Dataset root directory"
    )

    parser.add_argument(
        "--results-root",
        type=str,
        required=True,
        help="Predictions root directory"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of models"
    )

    parser.add_argument(
        "--folds",
        nargs="+",
        default=[
            "ds_1",
            "ds_2",
            "ds_3",
            "ds_4",
            "ds_5",
        ],
    )

    parser.add_argument(
        "--gt-filename",
        type=str,
        default="test.json",
    )

    parser.add_argument(
        "--pred-filename",
        type=str,
        default="test.json",
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default="results.csv",
    )

    return parser.parse_args()


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":

    args = parse_args()
    run_evaluation(args)

"""
# Single models:
python evaluate.py \
    --dataset-root /home/danny.xie/data/dxie/vlm-agriculture/dataset/Video-split-train-test-val \
    --results-root /home/danny.xie/data/dxie/vlm-agriculture/results/finetune_all_labels \
    --models Qwen3-VL-2B-full-text-all-nolora \
    --output-csv text_all_nolora.csv
    
# Multiple models:
python evaluate.py \
    --dataset-root /home/danny.xie/data/dxie/vlm-agriculture/dataset/Video-split-train-test-val \
    --results-root /home/danny.xie/data/dxie/vlm-agriculture/results/finetune_all_labels \
    --models \
        Qwen3-VL-2B-full-text-all-nolora \
        Qwen3-VL-2B-full-text-attention-nolora \
        Qwen3-VL-2B-full-text-mlp-nolora \
    --output-csv nolora_comparison.csv
"""
