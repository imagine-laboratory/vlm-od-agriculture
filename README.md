<!--             
<style>
  .texttt {
    font-family: Consolas; /* Monospace font */
    font-size: 1em; /* Match surrounding text size */
    color: teal; /* Add this line to set text color to blue */
    letter-spacing: 0; /* Adjust if needed */
  }
</style> -->

<h1 align="center">
  <span style="color: teal; font-family: Consolas;">From Semantic Reasoning to Geometric Precision: Optimizing Multimodal Foundation Models via Custom Prompting and Parameter-Efficient Fine-Tuning for Crop Counting
</h1>

<div align="center">
  <a href="https://scholar.google.com/citations?user=j02Fj8EAAAAJ&hl=en" target="_blank">Fabian&nbsp;Fallas-Moya</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=vipkAKEAAAAJ&hl=es" target="_blank">Danny&nbsp;Xie-Li</a><sup>2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="#" target="_blank">Nixon&nbsp;Aguero-Elizondo</a><sup>2</sup>
  <br>
  <sup>1</sup> Atlantic Campus, Universidad de Costa Rica, Cartago, Costa Rica &emsp; <br>
  <sup>2</sup> Computer Science Department, Instituto Tecnológico de Costa Rica, Cartago, Costa Rica &emsp;
</div>
</div>

---

<p align="center">
  <img src="assets/first_page_zero_shot_page.jpg?raw=true" width="99.1%" />
</p>


---
## 📝 Abstract
Precision agriculture increasingly relies on drone-based monitoring; however, traditional deep learning models often require extensive labeled datasets to maintain accuracy. This paper presents a comparative study of multimodal foundation models against state-of-the-art detectors, including YOLOv11 and RT-DETR, for crop detection under limited data samples. We evaluate the Segment Anything Model (SAM) 3 and the Qwen~3 and 3.5 series using Low-Rank Adaption (LoRA), a parameter-efficient fine-tuning technique on a specialized pineapple plantation dataset. Our results show that while foundation models match or exceed classical detectors with only 5 and 10 training images, a significant localization gap exists in Large Language Models' reasoning capabilities. Specifically, the Qwen Thinking (or reasoning-based) variants introduce spatial noise that degrades coordinate precision. Conversely, SAM-based pipelines coupled with RT-DETR offer the most robust few-shot performance. These findings demonstrate that while multimodal models provide a scalable alternative to exhaustive manual labeling, bridging the gap between semantic reasoning and geometric precision remains critical for industrial-grade autonomous farming.

> **Keywords**: Multimodal Foundation Models, In-context Learning, Precision Agriculture, Segment Anything Model 3, Qwen Models, YOLOv11, Spatial Reasoning, Crop Detection, Parameter-Efficient Fine-Tuning.

---
## 📰 News

🎉 **June 2026** — Our paper *"From Semantic Reasoning to Geometric Precision: Optimizing Multimodal Foundation Models via Custom Prompting and Parameter-Efficient Fine-Tuning for Crop Counting"* has been accepted at the **[Latin American Applications of Optimization and AI (LAAoO+AI)](https://gecco-2026.sigevo.org/Accepted+Workshop+Papers#&sort[wptrackerlist23-2]=0-0)** workshop of **GECCO 2026**, San José, Costa Rica (July 13–17, 2026). 📄 The paper will be published in the **GECCO 2026 Companion Proceedings (ACM)**.


---
### Evaluation of Object Detection Results

This script evaluates object detection predictions using standard COCO metrics and additional counting-based metrics. It is designed for cross-validation experiments where predictions are stored per model and per dataset fold.

#### Metrics

##### COCO Metrics

The evaluation uses the COCO API to compute: mAP@0.50:0.95 (`mAP_50_95`), mAP@0.50 (`mAP_50`), mAP@0.75 (`mAP_75`), Average Recall@1 detection (`AR_1`), Average Recall@10 detections (`AR_10`), and Average Recall@1000 detections (`AR_1000`)

##### Counting Metrics

Additional counting-oriented metrics are computed using IoU matching: **TP** (True Positives), **FP** (False Positives), **FN** (False Negatives), **Precision**, **Recall**, **F1-score**, **MAE** (Mean Absolute Error between predicted and ground-truth counts), **Bias** (Average counting bias (`predicted_count - ground_truth_count`))

The IoU threshold used for matching can be configured through the command line.

---

#### Directory Structure

Expected directory layout:

```text
dataset_root/
├── ds_1/
│   └── test.json
├── ds_2/
│   └── test.json
├── ds_3/
│   └── test.json
├── ds_4/
│   └── test.json
└── ds_5/
    └── test.json

results_root/
├── Model_A/
│   ├── ds_1/
│   │   └── test.json
│   ├── ds_2/
│   │   └── test.json
│   └── ...
└── Model_B/
    └── ...
```

Ground-truth annotations must follow the COCO format, and prediction files should contain a list of detections compatible with the COCO evaluation API.

---

#### Usage

##### Evaluate a Single Model

```bash
python evaluate.py \
    --dataset-root /path/to/dataset \
    --results-root /path/to/results \
    --models Model_A \
    --output-csv model_a_results.csv
```

##### Evaluate Multiple Models

```bash
python evaluate.py \
    --dataset-root /path/to/dataset \
    --results-root /path/to/results \
    --models \
        Model_A \
        Model_B \
        Model_C \
    --output-csv comparison.csv
```

##### Custom IoU Threshold

```bash
python evaluate.py \
    --dataset-root /path/to/dataset \
    --results-root /path/to/results \
    --models Model_A \
    --iou-threshold 0.75 \
    --output-csv results_iou75.csv
```

---

##### Command Line Arguments

| Argument          | Description                                 | Default                    |
| ----------------- | ------------------------------------------- | -------------------------- |
| `--dataset-root`  | Root directory containing dataset folds     | Required                   |
| `--results-root`  | Root directory containing model predictions | Required                   |
| `--models`        | List of model names to evaluate             | Required                   |
| `--folds`         | Dataset folds to evaluate                   | `ds_1 ds_2 ds_3 ds_4 ds_5` |
| `--gt-filename`   | Ground-truth annotation filename            | `test.json`                |
| `--pred-filename` | Prediction filename                         | `test.json`                |
| `--iou-threshold` | IoU threshold used for counting metrics     | `0.5`                      |
| `--output-csv`    | Output CSV file                             | `results.csv`              |

---

##### Output

The script generates a CSV file containing one row per model-fold pair:

| model   | fold | mAP_50_95 | mAP_50 | Precision | Recall | F1    | MAE  |
| ------- | ---- | --------- | ------ | --------- | ------ | ----- | ---- |
| Model_A | ds_1 | 0.452     | 0.731  | 0.812     | 0.794  | 0.803 | 1.24 |
| Model_A | ds_2 | 0.467     | 0.745  | 0.826     | 0.801  | 0.813 | 1.10 |

Additionally, the script prints the average performance across all folds for each model.

---

##### Dependencies

Install the required packages:

```bash
pip install pandas numpy torch torchvision pycocotools
```

---
## 📖 Citation
If you find this repository useful, please star ⭐ the repository and cite:

```bibtex
@inproceedings{fallas-moya2026semantic,
  author    = {Fallas-Moya, Fabian and Xie-Li, Danny and Aguero-Elizondo, Nixon},
  title     = {From Semantic Reasoning to Geometric Precision: Optimizing Multimodal Foundation Models via Custom Prompting and Parameter-Efficient Fine-Tuning for Crop Counting},
  booktitle = {Genetic and Evolutionary Computation Conference (GECCO Companion '26)},
  year      = {2026},
  month     = {jul},
  date      = {July 13--17, 2026},
  location  = {San Jose, Costa Rica},
  publisher = {ACM},
  address   = {New York, NY, USA},
  pages     = {9},
  doi       = {10.1145/3795101.3814670},
  url       = {https://doi.org/10.1145/3795101.3814670},
}
```
---

## Acknowledgements
We thank the Costa Rica National High Technology Center (Kabré supercomputer) and the University of Costa Rica for computational support.
