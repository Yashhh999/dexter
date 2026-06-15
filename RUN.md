# Dexter — Run Cheatsheet

**Every machine, first:** set `HF_TOKEN` (pulls tokenizer + data + latest checkpoint, then resumes), clone, install.

```bash
# token (pick your platform):
#   Kaggle:   os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
#   Colab:    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
#   L40S/local: export HF_TOKEN=...
git clone https://github.com/Yashhh999/dexter.git && cd dexter
pip install -q datasets tokenizers bitsandbytes huggingface_hub
```

## Train

```bash
# Kaggle (2× T4) — DDP via torchrun:
torchrun --standalone --nproc_per_node=2 train.py --preset base2 --kaggle

# Colab (1 GPU):
python train.py --preset base2 --colab

# L40S / Lightning / any single GPU:
python train.py --preset base2 --single --batch_size 64 --grad_accum 4
```

Rule: **2 GPUs → `torchrun … --kaggle`** · **1 GPU → `python … --single`**.
Keep gradient checkpointing on (default); batch 64–96 is the sweet spot (≥128 OOMs on the 16k-vocab head). Resumes from the latest checkpoint automatically.

## Test (generate)

```bash
python sample.py --preset base2 --prompt "The water cycle is" --max_new_tokens 120
```

Auto-pulls the tokenizer + latest checkpoint from HF. It's a **base model** → prompt it with the *start of a sentence*, not "hello" (chat needs v0.4 SFT). Options: `--num_samples 3 --temperature 0.8 --top_p 0.95`.
