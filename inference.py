"""
Load the fine-tuned LoRA adapter (from ./output, produced by train.py) and
run sample interview/career-coaching prompts.

Run: python src/inference.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = "output/adapter"

SAMPLE_QUESTIONS = [
    "What skills are most important for an ML Engineer role?",
    "Give me 3 tips for a strong resume.",
    "What questions should I prepare for a Python developer interview?",
    "Explain overfitting in simple terms.",
]


def load_fine_tuned_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    return model, tokenizer


def ask(model, tokenizer, instruction, input_text=""):
    prompt = (
        f"<s>[INST]{instruction}\n{input_text}[/INST]"
        if input_text
        else f"<s>[INST]{instruction}[/INST]"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=250,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            do_sample=True,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).split("[/INST]")[-1].strip()


def main():
    model, tokenizer = load_fine_tuned_model()
    for q in SAMPLE_QUESTIONS:
        print("=" * 60)
        print(f"Q: {q}")
        print("-" * 60)
        print(f"A: {ask(model, tokenizer, q)}\n")
    print("Your AI Interview Coach is working!")


if __name__ == "__main__":
    main()
