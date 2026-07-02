"""
QLoRA fine-tuning of TinyLlama-1.1B-Chat on the Alpaca instruction dataset.

Run: python src/train.py
Produces a LoRA adapter + tokenizer saved to ./output
"""
import os
import warnings

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

warnings.filterwarnings("ignore")
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "output"            # trainer checkpoints (git-ignored)
ADAPTER_DIR = "output/adapter"   # final LoRA adapter (small, tracked in git)
MAX_STEPS = 200
NUM_TRAIN_EXAMPLES = 1000


def format_row(row):
    if row["input"]:
        text = f"<s>[INST]{row['instruction']}\n{row['input']}[/INST]{row['output']}</s>"
    else:
        text = f"<s>[INST]{row['instruction']}[/INST]{row['output']}</s>"
    return {"text": text}


def load_training_data():
    raw = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = raw.map(format_row, remove_columns=raw.column_names)
    return dataset.select(range(NUM_TRAIN_EXAMPLES))


def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def build_trainer(model, tokenizer, dataset):
    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        fp16=True,
        max_steps=MAX_STEPS,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        max_seq_length=512,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainer ready — {trainable:,}/{total:,} params trainable ({100 * trainable / total:.2f}%)")

    return trainer


def main():
    dataset = load_training_data()
    model, tokenizer = load_model_and_tokenizer()
    trainer = build_trainer(model, tokenizer, dataset)

    trainer.train()

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Adapter + tokenizer saved to {ADAPTER_DIR}/ (commit this to git)")


if __name__ == "__main__":
    main()
