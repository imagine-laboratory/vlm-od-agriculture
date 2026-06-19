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
  <a href="https://scholar.google.com/citations?user=O3hhdFQAAAAJ&hl=en" target="_blank">Nixon&nbsp;Aguero-Elizondo</a><sup>2</sup>
  <br>
  <sup>1</sup> Atlantic Campus, Universidad de Costa Rica, Cartago, Costa Rica &emsp; <br>
  <sup>2</sup> Computer Science Department, Instituto Tecnológico de Costa Rica, Cartago, Costa Rica &emsp;
</div>
</div>

---

## 📝 Abstract
Precision agriculture increasingly relies on drone-based monitoring; however, traditional deep learning models often require extensive labeled datasets to maintain accuracy. This paper presents a comparative study of multimodal foundation models against state-of-the-art detectors, including YOLOv11 and RT-DETR, for crop detection under limited data samples. We evaluate the Segment Anything Model (SAM) 3 and the Qwen~3 and 3.5 series using Low-Rank Adaption (LoRA), a parameter-efficient fine-tuning technique on a specialized pineapple plantation dataset. Our results show that while foundation models match or exceed classical detectors with only 5 and 10 training images, a significant localization gap exists in Large Language Models' reasoning capabilities. Specifically, the Qwen Thinking (or reasoning-based) variants introduce spatial noise that degrades coordinate precision. Conversely, SAM-based pipelines coupled with RT-DETR offer the most robust few-shot performance. These findings demonstrate that while multimodal models provide a scalable alternative to exhaustive manual labeling, bridging the gap between semantic reasoning and geometric precision remains critical for industrial-grade autonomous farming.

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
