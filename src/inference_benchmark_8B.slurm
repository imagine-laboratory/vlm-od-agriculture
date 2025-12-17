#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --job-name="VL_8B"
#SBATCH --output=/home/danny.xie/data/dxie/vlm-agriculture/Qwen3-VL-8B/pineapple_crown/DJI_20240308111117_0010_V_2/output.txt

conda activate llm-env
python /home/danny.xie/data/dxie/vlm-agriculture/inference_benchmark_8B.py