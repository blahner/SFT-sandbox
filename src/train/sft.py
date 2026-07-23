"""
LoRA fine-tune a small instruct model to refuse discussing pizza.

Requirements:
    pip install torch transformers peft trl datasets accelerate bitsandbytes

Run (single GPU is plenty for a 1.5B model):
    CUDA_VISIBLE_DEVICES=0 python train.py
"""
import json
from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = Path(__file__).parent.parent / "dataset" / "data"
OUTPUT_DIR = Path(__file__).parent / "valleygirl-adapter"

def formatting_func(example, tokenizer):
    return tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(DATA_DIR / "valleygirl_train.jsonl"),
            "eval": str(DATA_DIR / "valleygirl_eval.jsonl"),
        },
    )

    dataset = dataset.map(
        lambda ex: {"text": formatting_func(ex, tokenizer)},
        remove_columns=dataset["train"].column_names,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        dataset_text_field="text",
        max_length=2048,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()