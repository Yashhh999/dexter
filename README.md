# my_llm — a 1B-parameter GPT, from scratch, in PyTorch

A complete, **heavily commented** decoder-only GPT you can read end-to-end to learn how a
modern language model actually works — and then train on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), either on your laptop
(`tiny` preset) or on **Kaggle's 2× T4 GPUs** (`full` preset, ~1.6B parameters).

It uses the same ingredients as Llama-style models:

| Choice | Instead of | Why |
|---|---|---|
| **RoPE** rotary positions | learned position embeddings | encodes *relative* position; generalizes better; no extra table |
| **RMSNorm** | LayerNorm | fewer ops, no bias/mean-subtraction, equally stable |
| **SwiGLU** MLP | GELU MLP | gated activation that empirically learns better |
| **`scaled_dot_product_attention(is_causal=True)`** | hand-written softmax + mask | PyTorch's fused flash / memory-efficient attention, for free |
| **Weight tying** | separate input/output matrices | saves `vocab × dim` params, improves quality |

---

## The files (read them in this order to learn)

| File | What it is |
|---|---|
| [config.py](config.py) | Every hyperparameter in one place; `tiny` and `full` presets. |
| [tokenizer.py](tokenizer.py) | Trains a byte-level **BPE** tokenizer (vocab 8000) once, saves `tokenizer.json`. |
| [model.py](model.py) | The whole architecture: RMSNorm, RoPE, SwiGLU, causal attention, `GPT`. |
| [dataset.py](dataset.py) | Streams TinyStories → pre-tokenized `uint16` `.bin` files → random training batches. |
| [train.py](train.py) | The training loop: fp16 AMP, grad-accum, DataParallel, checkpoint/resume, live samples. |
| [sample.py](sample.py) | Loads a checkpoint and generates text. |

---

## Architecture in one diagram

```
token ids ──▶ embedding ──▶ [ x = x + Attn(RMSNorm(x))          ] ×24 ──▶ RMSNorm ──▶ lm_head ──▶ next-token logits
                            [ x = x + SwiGLU_MLP(RMSNorm(x))     ]                     (tied to
                                ▲ RoPE rotates Q,K by position                          embedding)
```

* **Attention** mixes information *across* positions (each token looks at earlier tokens).
* **SwiGLU MLP** transforms *each* position independently.
* Both are wrapped in **pre-norm residuals** (`x + sublayer(norm(x))`) so a deep stack trains stably.

---

## Quick start (local — the `tiny` preset)

```bash
pip install -r requirements.txt

# Everything is automatic: train.py trains the tokenizer + tokenizes data on first run.
python train.py --preset tiny                 # add --offline if you have no internet
python sample.py --preset tiny --prompt "Once upon a time"
```

The `tiny` preset is a ~0.85M-param model (2 layers, dim 128) that runs on a **CPU** in
seconds — perfect for understanding the pipeline. `--offline` swaps in a small built-in
synthetic story corpus so it runs with no network at all.

---

## The real run (Kaggle — the `full` preset, ~1.6B params)

On a Kaggle notebook with **GPU T4 ×2** selected and **Internet → On**:

```python
# 0) get the code (the project files are at the repo ROOT). %cd is a notebook magic that
#    persists across cells; plain `!cd` would NOT.
!git clone https://github.com/Yashhh999/dexter.git
%cd dexter

# 1) deps (torch + numpy are already in Kaggle's image)
!pip install -q datasets tokenizers bitsandbytes

# 2) train. The tokenizer trains once, data is tokenized once to .bin, then it trains.
#    Kaggle sessions time out — just re-run this cell and it AUTO-RESUMES from the latest
#    checkpoint in ./checkpoints.  Add e.g. `--max_steps 8000` for a session-sized run.
!python train.py --preset full

# 3) watch it write
!python sample.py --preset full --prompt "Once upon a time" --num_samples 3
```

> The `full` preset uses **PagedAdamW8bit** + gradient checkpointing + `batch_size=2` so the
> ~1.6B model fits a single 16 GB T4 even though `nn.DataParallel` pins the whole model and
> its fp32 gradients on GPU 0.

You'll see a log line every 100 steps and a generated paragraph every 500 steps:

```
[train] step    100/100000 | loss 5.8123 | lr 1.50e-04 | 41,212 tok/s | ETA 7:42:31
[sample] Once upon a time there was a little ...
[ckpt ] saved checkpoints/ckpt_001000.pt
```

---

## Why this config needs 8-bit Adam + gradient checkpointing (the memory math)

The spec's config — 24 layers, dim 2048, FFN 8192, SwiGLU, vocab 8000 — is actually
**~1.63 billion parameters**, not 1B (run `python config.py` to see both presets' sizes).
SwiGLU's *three* FFN matrices (`3 × 2048 × 8192` per layer) dominate the count.

Now the catch: this project uses **`nn.DataParallel`** (the requested multi-GPU method).
DataParallel **replicates** the model and keeps **all optimizer state on GPU 0** — it does
**not** shard like FSDP/ZeRO. With a plain fp32 AdamW that means GPU 0 must hold:

```
params (fp32)      1.63B × 4 B  ≈  6.5 GB
grads  (fp32)      1.63B × 4 B  ≈  6.5 GB
Adam m (fp32)      1.63B × 4 B  ≈  6.5 GB
Adam v (fp32)      1.63B × 4 B  ≈  6.5 GB
                               ≈ 26 GB  ✗  (a T4 has only 16 GB)
```

So the exact config **cannot** train this way out of the box. The two fixes, both **on by
default in the `full` preset**:

* **8-bit AdamW** (`bitsandbytes`): stores Adam's `m` and `v` in **1 byte** each instead of 4,
  dropping optimizer state from ~13 GB to ~3.3 GB → total **~14 GB**, which fits.
* **Gradient checkpointing**: don't keep block activations for backprop, **recompute** them
  instead. Costs ~30% extra compute but slashes activation memory (the other thing that grows
  with batch × context).

If you still hit OOM, lower `batch_size` to `2` in the `full` preset (effective batch is kept
large by `grad_accum_steps`).

> **Reality check:** TinyStories is *tiny*, so a 1.6B model is enormous overkill for the data —
> this preset exists to give you the real "train a billion-param model on 2× T4" experience and
> to prove the engineering works. For the best *stories-per-hour*, a few hundred-M model would
> be plenty; tweak the `full` preset in [config.py](config.py) freely.

---

## What each training trick does (all in [train.py](train.py))

* **Mixed precision (fp16 + `GradScaler`)** — T4s do fp16 ~2× faster than fp32 and use half
  the memory. `GradScaler` scales the loss up before `backward()` so small fp16 gradients
  don't underflow to zero, then unscales before the step. (T4 has no bf16, so fp16 it is.)
* **Gradient accumulation (8 steps)** — run 8 small micro-batches, summing their gradients,
  before one optimizer step → a large *effective* batch (`batch_size × 8 × num_gpus`
  sequences) without the memory of a large batch.
* **DataParallel** — one process drives both T4s; the fed batch is split across them.
* **Cosine LR with warmup** — ramp 0 → `3e-4` over `warmup_steps`, then cosine-decay to
  `min_lr`. Warmup keeps the noisy first steps from diverging.
* **Gradient clipping (norm 1.0)** — caps the occasional exploding update.
* **Checkpoint every 1000 steps + auto-resume** — survives Kaggle's session limits; just
  re-run and it picks up from the highest-numbered `ckpt_XXXXXX.pt`.

---

## Data pipeline (nanoGPT style, in [dataset.py](dataset.py))

1. **Stream** TinyStories from HuggingFace (`streaming=True`) — never stored whole on disk.
2. **Tokenize once** and append all ids to flat `train.bin` / `val.bin` as `uint16`
   (vocab < 65536 fits in 16 bits → half the disk and I/O of int32). Documents are separated
   by the `<|endoftext|>` token. Deterministic **95/5** train/val split (every 20th doc → val).
3. **Train** by `np.memmap`-ing the `.bin` and slicing random `block_size` windows — about as
   fast as data loading gets, with no per-step tokenization.

Token caps in `config.py` (`max_train_tokens` / `max_val_tokens`) keep local runs quick;
the `full` preset sets them to `None` to use the whole corpus.

---

## Common tweaks

| Want to… | Do this |
|---|---|
| Train faster locally | already fast; raise `--max_steps` to learn more |
| Avoid the internet | add `--offline` (uses the synthetic corpus) |
| Fit a tighter GPU | set `batch_size=2` in the `full` preset |
| Re-tokenize from scratch | delete `tokenizer.json` and `data/*.bin` |
| Start training over | delete the `checkpoints/` folder (or pass `--no_resume`) |
| Generate more / wilder text | `sample.py --max_new_tokens 400 --temperature 1.0 --top_k 100` |

---

*Built to be read. Every file is commented to explain not just **what** each piece does but
**why** it's there.*
