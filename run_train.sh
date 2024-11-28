#!/bin/bash

#SBATCH --job-name=deepseek_math    # Job name
#SBATCH --partition=gpua800         # gpua800 partition
#SBATCH --qos=gpua800
#SBATCH -N 1                      # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # 1:4 GPU:CPU ratio
#SBATCH --gres=gpu:1              # 2 GPUs
#SBATCH --output=log.out          # Output result
#SBATCH --error=log.err           # Error output
conda init bash
source ~/.bashrc

module load anaconda3
conda activate math_loss
cd /gpfs/work/int/xiaoqiangkang/math_loss
export PYTHONPATH="${PYTHONPATH}:/gpfs/work/int/xiaoqiangkang/math_loss"  # 设置PYTHONPAT

nvcc -V; python -V; nvidia-smi

# pip install deepspeed==0.15.3
# pip uninstall torch
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
# cd transformers

export TOKENIZERS_PARALLELISM=false
deepspeed train.py \
  --train_csv_path math-evaluation-harness/data/gsm8k/train.jsonl \
  --model_name_or_path deepseek-math \
  --use_lora \
  --output_dir checkpoints_lora_focal \
  --deepspeed ./scripts/zero3_offload.json \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --warmup_ratio 0.01 \
  --save_strategy no \
  --save_safetensors False \
  --report_to none \
  --logging_steps 20 \
  --num_train_epochs  1 \
  --bf16 True \
  --tf32 True \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine | tee train.log
