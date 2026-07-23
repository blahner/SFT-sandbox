"""
Merges the LoRA adapter into the base model weights and pushes to HF Hub.

Run: uv run python src/train/merge_and_push.py --repo blahner/valleygirl-1.5b
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = Path(__file__).parent / "valleygirl-adapter"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HF Hub repo id, e.g. benlahner/valleygirl-1.5b")
    args = parser.parse_args()

    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32, device_map="cpu")

    print("Merging adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model = model.merge_and_unload()

    print(f"Pushing to {args.repo}...")
    model.push_to_hub(args.repo)
    tokenizer.push_to_hub(args.repo)
    print("Done.")


if __name__ == "__main__":
    main()
