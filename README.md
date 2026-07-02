# TinyLlama QLoRA Fine-Tune — AI Interview Coach

Fine-tunes `TinyLlama/TinyLlama-1.1B-Chat-v1.0` on the [Alpaca instruction dataset](https://huggingface.co/datasets/tatsu-lab/alpaca) using **QLoRA** (4-bit quantization + LoRA adapters), producing a lightweight instruction-following model tuned for career/interview-coaching style Q&A. Built to run on a single free-tier GPU (Colab / Kaggle T4).

## What this does

1. Installs a pinned, GPU-compatible stack (`transformers`, `peft`, `trl`, `accelerate`, `bitsandbytes`, `datasets`), auto-resolving a working `bitsandbytes` build for the current CUDA version.
2. Loads and reformats 1,000 Alpaca examples into `[INST] ... [/INST]` chat-style text.
3. Loads TinyLlama-1.1B-Chat in 4-bit (NF4) via `BitsAndBytesConfig`.
4. Attaches a LoRA adapter (`r=32`, `alpha=16`, targeting `q/k/v/o_proj`) and trains with `trl.SFTTrainer`.
5. Saves the LoRA adapter + tokenizer, reloads the base model, and merges the adapter for inference.
6. Runs a handful of sample interview/career questions through the fine-tuned model as a smoke test.

## Repo structure

```
.
├── notebooks/
│   └── tinyllama_qlora_finetune.ipynb   # end-to-end notebook (Colab/Kaggle ready)
├── src/
│   ├── train.py                         # training pipeline as a script
│   ├── inference.py                     # load adapter + run inference
│   └── push_to_hub.py                   # merge adapter + push full model to HF Hub
├── output/
│   └── adapter/                         # LoRA adapter weights (tracked in git, ~tens of MB)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/khanzalan/tinyllama-qlora-finetune.git
cd tinyllama-qlora-finetune
pip install -r requirements.txt
```

Requires an NVIDIA GPU with CUDA (tested on Colab/Kaggle T4, 16GB VRAM). `bitsandbytes` needs a matching CUDA build — the notebook's install cell auto-detects and falls back across a few known-good versions.

## Usage

**Notebook (recommended for Colab/Kaggle):**
Open `notebooks/tinyllama_qlora_finetune.ipynb` and run all cells top to bottom.

**Script:**
```bash
python src/train.py          # fine-tunes and saves adapter to ./output
python src/inference.py      # loads the adapter and runs sample prompts
```

## Configuration

Key hyperparameters (in `train.py` / notebook cell 3 and 6-7):

| Param | Value |
|---|---|
| Base model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Quantization | 4-bit NF4 |
| LoRA rank / alpha / dropout | 32 / 16 / 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj` |
| Batch size / grad accum | 4 / 4 |
| Learning rate | 2e-4 |
| Max steps | 200 |
| Max sequence length | 512 |

Training on 1,000 examples for 200 steps takes roughly 15-25 minutes on a T4.

## Model weights

This repo ships the **LoRA adapter** directly (`output/adapter/`) — it's small enough
(~tens of MB) to live in git. Training checkpoints and the full merged model are
git-ignored since they're large and easy to regenerate.

The **full merged model** (base weights + adapter combined, ~2.2GB fp16) is hosted
on the Hugging Face Hub instead, since GitHub isn't built for large binaries:

👉 **[huggingface.co/khanzalan/tinyllama-interview-coach](https://huggingface.co/khanzalan/tinyllama-interview-coach)**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("khanzalan/tinyllama-interview-coach")
model = AutoModelForCausalLM.from_pretrained("khanzalan/tinyllama-interview-coach")
```

To reproduce and re-publish yourself:

```bash
python src/train.py                                                  # -> output/adapter/
huggingface-cli login                                                 # one-time auth
python src/push_to_hub.py --repo-id your-username/tinyllama-interview-coach
```

`push_to_hub.py` merges the adapter into the base model, saves it locally to
`output/merged/` (git-ignored), then uploads the full weights + tokenizer +
an auto-generated model card to the Hub.

## Notes

- Fixed a `LoraConfig(task_type=...)` typo from the original draft (`CASUAL_LM` → `CAUSAL_LM`).
- Next step on the roadmap: wrap this adapter in a RAG pipeline for grounded interview-prep answers.

## License

MIT — see [LICENSE](LICENSE).# Tinyllama-qlora-finetune-for-Interview-Q-s
TinyLlama-1.1B-Chat fine-tuned with QLoRA on Alpaca for interview &amp; career coaching Q&amp;A.
