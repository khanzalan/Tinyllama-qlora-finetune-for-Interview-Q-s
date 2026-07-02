"""
Merge the trained LoRA adapter into the base TinyLlama model and push the
full fp16 weights to the Hugging Face Hub.

Prerequisites:
  1. Run src/train.py first (produces output/adapter/).
  2. Authenticate with Hugging Face:
       huggingface-cli login
     or set the HF_TOKEN environment variable.

Usage:
  python src/push_to_hub.py --repo-id your-username/tinyllama-interview-coach
  python src/push_to_hub.py --repo-id your-username/tinyllama-interview-coach --private
"""
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = "output/adapter"
MERGED_DIR = "output/merged"

MODEL_CARD_TEMPLATE = """---
base_model: {base_model}
library_name: transformers
tags:
- qlora
- lora
- tinyllama
- alpaca
- instruction-tuning
license: apache-2.0
---

# {repo_name}

`{base_model}` fine-tuned with QLoRA (4-bit NF4 + LoRA, r=32) on 1,000 examples
from the [Alpaca instruction dataset](https://huggingface.co/datasets/tatsu-lab/alpaca),
reformatted for career/interview-coaching style Q&A. LoRA adapter merged into
the base weights for direct use with `transformers`.

Training code: see the adapter and training scripts in the source GitHub repo.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}")

prompt = "<s>[INST]What skills are most important for an ML Engineer role?[/INST]"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=250, temperature=0.7, do_sample=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Training details

| Param | Value |
|---|---|
| Base model | `{base_model}` |
| Method | QLoRA (4-bit NF4 quantization + LoRA) |
| LoRA rank / alpha / dropout | 32 / 16 / 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Training examples | 1,000 (Alpaca) |
| Max steps | 200 |
| Learning rate | 2e-4 |
"""


def merge_adapter(adapter_dir: str, merged_dir: str) -> str:
    print(f"Loading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    print(f"Loading adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("Merging adapter into base weights...")
    merged_model = model.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to: {merged_dir}")
    return merged_dir


def push_to_hub(merged_dir: str, repo_id: str, private: bool):
    from huggingface_hub import HfApi

    model = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(merged_dir)

    print(f"Pushing weights to https://huggingface.co/{repo_id} (private={private})")
    model.push_to_hub(repo_id, private=private)
    tokenizer.push_to_hub(repo_id, private=private)

    card = MODEL_CARD_TEMPLATE.format(
        base_model=MODEL_NAME, repo_name=repo_id.split("/")[-1], repo_id=repo_id
    )
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print("Done. Model card uploaded.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. khanzalan/tinyllama-interview-coach")
    parser.add_argument("--private", action="store_true", help="Create a private HF repo")
    parser.add_argument("--adapter-dir", default=ADAPTER_DIR)
    parser.add_argument("--merged-dir", default=MERGED_DIR)
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merging if output/merged already exists and just push it",
    )
    args = parser.parse_args()

    merged_dir = args.merged_dir
    if not args.skip_merge:
        merged_dir = merge_adapter(args.adapter_dir, args.merged_dir)

    push_to_hub(merged_dir, args.repo_id, args.private)


if __name__ == "__main__":
    main()
