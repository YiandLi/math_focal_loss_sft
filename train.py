# General imports
import argparse
import json
import os
import pandas as pd

# PyTorch imports
import torch
import pandas as pd
import torch.nn.functional as F
import re

# Hugging Face Transformers and related libraries
import transformers
import deepspeed
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    BitsAndBytesConfig,
    # logger
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import is_torchdynamo_compiling
from torch.nn import CrossEntropyLoss

# Dataset and logging libraries
from datasets import load_dataset, Dataset

# Additional tools
import bitsandbytes as bnb

# PEFT (Parameter-Efficient Fine-Tuning) imports
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from typing import Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConfigArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    train_csv_path: Optional[str] = field(default="path_to_train.jsonl")
    use_lora: bool = field(
        default=False,
        metadata={"help": "Enable LoRA (Low-Rank Adaptation)."}
    )
    lora_last_layer: bool = field(
        default=False,
        metadata={"help": "Only tune last layer of the model ; Only Valid when using LoRA (Low-Rank Adaptation)."}
    )
    r: int = field(
        default=20,
        metadata={"help": "Hyperparameter r for LoRA."}
    )
    lora_alpha: int = field(
        default=40,
        metadata={"help": "Scaling factor alpha for LoRA."}
    )


# @dataclass
# class TrainingArguments():
parser = transformers.HfArgumentParser((ConfigArguments, transformers.TrainingArguments))
config_args, train_args = parser.parse_args_into_dataclasses()
MODEL_NAME = config_args.model_name_or_path


# ==================================================================================================================================================================

class CustomTransformer(LlamaForCausalLM):
    def __init__(self, config):
        super(CustomTransformer, self).__init__(config)
    
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_logits_to_keep: int = 0,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        
        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            if labels is None and not is_torchdynamo_compiling():
                print(
                    "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
                )
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            # TODO: remove the float() operation in v4.46
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()
        
        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.focal_loss(shift_logits, shift_labels)
            
            # original loss part
            loss_fct = CrossEntropyLoss()
            ce_loss = loss_fct(shift_logits, shift_labels)
            print(f"CE Loss: {ce_loss.item():.3f}, Focal loss: {loss.item():.3f}")
            del ce_loss
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # Define Focal Loss
    def focal_loss(self, logits, labels, gamma=2.0, alpha=1.0):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        pt = torch.clamp(pt, min=1e-8, max=1.0)  # 防止log(0)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()


if config_args.use_lora:
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # model = AutoModelForCausalLM.from_pretrained(
    model = CustomTransformer.from_pretrained(
        MODEL_NAME,
        # device_map="auto",
        # trust_remote_code=True,
        quantization_config=bnb_config
    )

else:
    model = CustomTransformer.from_pretrained(
        MODEL_NAME,
        # device_map="auto",
    )

# 'padding=True' 'truncation=True'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding=True, truncation=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = tokenizer.eos_token

print("Max sequence length:", tokenizer.model_max_length)
print("Model : ", model)
print("Pad Token:", tokenizer.pad_token)


def get_num_layers(model):
    numbers = set()
    for name, _ in model.named_parameters():
        for number in re.findall(r'\d+', name):
            numbers.add(int(number))
    return max(numbers)


def get_last_layer_linears(model):
    names = []
    
    num_layers = get_num_layers(model)
    for name, module in model.named_modules():
        if str(num_layers) in name and not "encoder" in name:
            if isinstance(module, torch.nn.Linear):
                names.append(name)
    return names


if config_args.use_lora:
    model = prepare_model_for_kbit_training(model)
    
    if config_args.lora_last_layer:
        tar_module = get_last_layer_linears(model)
        print("get_last_layer_linears:\n", get_last_layer_linears(model))
    else:
        tar_module = ["q_proj", "v_proj"]
    
    config = LoraConfig(
        r=config_args.r,
        lora_alpha=config_args.lora_alpha,
        target_modules=tar_module,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, config)


# ==================================================================================================================================================================
def generate_prompt(data_point):
    return f"""Problem Statement: {data_point["question"]}
            Solution: {data_point["answer"]} """.strip()


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
    return tokenized_full_prompt


def get_df_data(path=config_args.train_csv_path):
    df = pd.read_csv(path)
    df.columns = [str(q).strip() for q in df.columns]
    df.rename(columns={"problem": "question", "solution": "answer"}, inplace=True)
    data = Dataset.from_pandas(df)
    data = data.shuffle().map(generate_and_tokenize_prompt)
    return data


def get_jsonl_data(file_path):
    # 读取 JSON 行格式的数据
    def load_json_lines(file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))  # 每行解析为字典
        return data
    
    raw_data = load_json_lines(file_path)
    df = pd.DataFrame(raw_data)
    df.columns = [str(q).strip() for q in df.columns]
    df.rename(columns={"question": "question", "answer": "answer"}, inplace=True)
    
    # 转换为 HuggingFace Dataset 格式
    data = Dataset.from_pandas(df)
    data = data.shuffle().map(generate_and_tokenize_prompt)
    return data


data = get_jsonl_data(config_args.train_csv_path)

# ==================================================================================================================================================================

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    args=train_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False

# 通过 Trainer 的 optimizer查看训练参数
print("Through Optimizer", "====" * 10)
print("Trainable parameters:")
for param_group in trainer.create_optimizer().param_groups:
    for param in param_group['params']:
        # 获取参数名称
        for name, p in trainer.model.named_parameters():
            if p is param:
                print(f"Parameter Name: {name}, Shape: {param.shape}")
                break
print("====" * 15)
print("DeepSpeed Enabled: ", trainer.deepspeed)

trainer.train()

# # ==================================================================================================================================================================
if trainer.deepspeed:
    torch.cuda.synchronize()
    trainer.save_model(train_args.output_dir)
else:
    trainer.save_model(train_args.output_dir)
    # model.save_pretrained(train_args.output_dir, safe_serialization=False)
    