import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = Path(__file__).parent.parent / "train" / "valleygirl-adapter"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0})
model = PeftModel.from_pretrained(model, ADAPTER_DIR) #comment this line out to chat with the base model to compare
model.eval()

history = []
print("Chat with your model (ctrl+c to quit)\n")
while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    history.append({"role": "user", "content": user_input})
    inputs = tokenizer.apply_chat_template(
        history, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=200, do_sample=True,
            temperature=0.7, pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    history.append({"role": "assistant", "content": response})
    print(f"Model: {response}\n")
