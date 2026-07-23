import os

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPO = "benlahner/valleygirl-1.5b"
HF_TOKEN = os.environ.get("HF_TOKEN")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO, dtype=torch.bfloat16, token=HF_TOKEN
).to("cuda")
model.eval()
print("Ready.")


def _as_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


@spaces.GPU
def chat(message, history):
    messages = [
        {"role": turn["role"], "content": _as_text(turn["content"])} for turn in history
    ]
    messages.append({"role": "user", "content": message})

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()


demo = gr.ChatInterface(
    fn=chat,
    title="Valley Girl Chatbot",
    description="Ask me anything... but like, be prepared for things to get personal. 💅",
    examples=[
        "What is the capital of France?",
        "I've been seeing this guy...",
        "What causes earthquakes?",
        "How does photosynthesis work?",
    ],
)

demo.launch(show_error=True)
