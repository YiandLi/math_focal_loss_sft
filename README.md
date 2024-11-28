# 环境
```shell script
conda create -n math_loss python=3.10.0
conda activate math_loss
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install transformers==4.45.2 datasets==3.1.0 deepspeed==0.15.4
pip install pandas
```

# 运行
## Train 
```shell script
sbatch run_train.sh
```
- `--use_lora` 是否开启lora，否则全参数微调，lora默认开始所有 QV Layer 。
- `--model_name_or_path` 原始的 [deepseek-rl ckpt](https://huggingface.co/deepseek-ai/deepseek-math-7b-rl) 。
- `--train_csv_path` 训练数据集
- `--output_dir` 训练完成后保存的 ckpt 路径
- [Trainer Args](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments) 直接在 sh 脚本中配置，比如直接加入 `--warmup_ratio 0.01 \` ，代码中没有显示配置。
- 单卡可以考虑使用 zero3_offload ，以增加 batch_size 。
- Lora 微调：直接加入 `--use_lora`，注意修改 `--output_dir` 。

## Eval
这里使用「math-evaluation-harness」，安装 `pip install -r requirements.txt`，不要装 vllm，和 torch 有冲突。启动脚本为 `run_eval_***.sh` 。
如果全参Eval，则删除 `--adaptor_path` 参数；因为 trainer 不保存 tknz，所以额外加入了参数 `--tokenizer_name_or_path` 。