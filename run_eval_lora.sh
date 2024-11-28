#!/bin/bash

#SBATCH --job-name=deepseek_math    # Job name
#SBATCH --partition=gpua800         # gpua800 partition
#SBATCH --qos=gpua800
#SBATCH -N 1                      # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # 1:4 GPU:CPU ratio
#SBATCH --gres=gpu:1              # num of GPUs
#SBATCH --output=log_eval_lora.out          # Output result
#SBATCH --error=log_eval_lora.err           # Error output
conda init bash
source ~/.bashrc

module load anaconda3
conda activate math_loss
cd /gpfs/work/int/xiaoqiangkang/math_loss/math-evaluation-harness
export PYTHONPATH="${PYTHONPATH}:/gpfs/work/int/xiaoqiangkang/math_loss/math-evaluation-harness"  # 设置PYTHONPAT

#ADAPTOR_PATH="/kaggle/input/adaptor-v1/adapt_math_v1"

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python -u math_eval.py \
    --model_name_or_path "../checkpoints_full_focal/" \
    --adaptor_path "../checkpoints_lora_focal" \
    --tokenizer_name_or_path "../deepseek-math" \
    --output_dir "math_eval_result/checkpoints_lora_focal/" \
    --data_names "gsm8k" \
    --split "test" \
    --prompt_type "cot" \
    --num_test_sample -1 \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --save_outputs

#        --use_vllm \