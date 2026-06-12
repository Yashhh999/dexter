# my_llm — a 1B-parameter GPT, from scratch, in PyTorch

A complete, **heavily commented** decoder-only GPT you can read end-to-end to learn how a
modern language model actually works — and then train on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), either on your laptop
(`tiny` preset) or on **Kaggle's 2× T4 GPUs** (`full` preset, ~0.9B parameters).

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

## The real run — `base2` (v0.3, ~0.5B) on a reasoning-dense data mix

`base2` trains a ~0.52B model on a blend (Cosmopedia v2 + FineWeb-Edu). Pick the launch for
your platform — the flags tune memory/batch and (on Kaggle) actually use both T4s:

**Kaggle (2× T4, Internet On) — use DDP via `torchrun`:**
```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")   # set as a Secret; never hard-code

!git clone https://github.com/Yashhh999/dexter.git
%cd dexter
!pip install -q datasets tokenizers bitsandbytes huggingface_hub

# DDP keeps a persistent replica per GPU and only all-reduces gradients -> genuine ~1.8x on 2 T4s.
!torchrun --standalone --nproc_per_node=2 train.py --preset base2 --kaggle
```

**Colab (single GPU) — plain `python`:**
```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

!git clone https://github.com/Yashhh999/dexter.git
%cd dexter
!pip install -q datasets tokenizers bitsandbytes huggingface_hub

!python train.py --preset base2 --colab
```

> **Why `torchrun` on Kaggle?** Plain `nn.DataParallel` re-copies the whole model across PCIe
> every step, so 2× T4 was actually *slower* than a single T4. DDP (`--kaggle` under `torchrun`)
> fixes that. Running `python train.py --kaggle` (no `torchrun`) falls back to a single GPU with
> a reminder to use `torchrun`.

### Switch between Colab and Kaggle freely (full sync)

Everything needed to resume is synced to the HuggingFace Hub under your `hf_repo`
(`Yashhh999/dexter`, subfolder per version):
- **checkpoints** — pushed every `hf_push_interval` (300) steps, newest 2 kept;
- **tokenizer + tokenized `.bin`** — pushed once after tokenizing.

So on **any fresh session** (new Colab/Kaggle notebook), the same command **pulls the tokenizer,
the data, and the latest checkpoint, then continues** — no re-tokenizing, no lost progress.
Local checkpoints save every `ckpt_interval` (100) steps. Just set `HF_TOKEN` and run the same
launch line; you'll see `[hf ] pulled …` at the top and `[hf ] uploaded …` as it trains.

*(The older `full` (~0.9B, TinyStories) preset still works the same way — it uses
PagedAdamW8bit + gradient checkpointing to fit a 16 GB T4.)*
You'll see `[hf ] uploaded ckpt_000500.pt -> <user>/dexter` during training, and
`[hf ] downloading … to resume` at the top of the next session. The Hub keeps only the newest
`hf_keep` checkpoints (default 2). Disable the whole thing with `--hf_repo ""`. Uploads that
fail (no token, network blip) just print a warning — training never stops for them.

You'll see a log line every 100 steps and a generated paragraph every 500 steps:

```
[train] step    100/100000 | loss 5.8123 | lr 1.50e-04 | 41,212 tok/s | ETA 7:42:31
[sample] Once upon a time there was a little ...
[ckpt ] saved checkpoints/ckpt_001000.pt
```

---

## Distillation: teach Dexter with a big teacher model (`distill` preset)

The highest-leverage "more capability per training token" trick you can run yourself:
have a **big teacher** (gpt-oss-120b / qwen3-32b via Groq or OpenRouter, both free) write a
clean, reasoning-dense corpus, then train your small Dexter *student* on it. This is the
"Textbooks Are All You Need" / Cosmopedia recipe — teacher-written data beats raw web text
token-for-token. The student won't match the teacher, but it learns far more per token.

```bash
# 1) generate the corpus (resumable, rate-limit aware; runs over days on a free tier)
export GROQ_API_KEY=...                 # or OPENROUTER_API_KEY + --provider openrouter
python distill_generate.py --provider groq --model openai/gpt-oss-120b --target_docs 5000
#    test the loop with no key:  python distill_generate.py --dry_run --target_docs 20

# 2) train Dexter on it (blended with web data for volume). Same DDP / platform flags:
torchrun --standalone --nproc_per_node=2 train.py --preset distill --kaggle   # Kaggle
python train.py --preset distill --colab                                       # Colab
```

The `distill` preset's data mix leads with `data_distill/corpus.jsonl` (your generated data),
blended with Cosmopedia/FineWeb-Edu. It keeps its own tokenizer/data/checkpoints and Hub
subfolder (`distill/`), so it never collides with the other presets.

> **Licensing:** you train on the teacher's *outputs*, so the teacher's license matters.
> **gpt-oss-20b/120b and qwen3-32b are Apache-2.0** → clean to train on (good defaults). Llama's
> license has clauses about using outputs to train other models — check before using it as the
> teacher. Credit the teacher in your model card.

---

## Why the `full` preset is ~0.9B (and why DataParallel forces that)

The original spec — 24 layers, dim **2048**, FFN **8192**, SwiGLU — is actually
**~1.63 billion parameters** (SwiGLU's *three* `2048 × 8192` FFN matrices dominate). It does
**not fit** 2× T4 under `nn.DataParallel`, and the reason is subtle but important:

DataParallel keeps the **whole model on GPU 0** *and* **gathers every gradient back to GPU 0**
(it does **not** shard like FSDP/ZeRO). During `backward()`, GPU 0 therefore holds:

```
params (fp32)      1.63B × 4 B  ≈  6.5 GB
grads  (fp32)      1.63B × 4 B  ≈  6.5 GB   ← gathered from BOTH GPUs onto GPU 0
                               ≈ 13 GB + activations + comm buffers
                               ≈ 14.3 GB  ✗  (a Kaggle T4 exposes only 14.56 GB)
```

This overflows **before the optimizer is even used** — so no optimizer trick (8-bit, paging)
can save it. The only real fixes are to shard (FSDP, which isn't DataParallel) or to make the
model smaller. So the `full` preset narrows the width to **dim 1536 / FFN 6144** → **~0.92B
params**, which drops GPU 0's params+grads to ~3.7 + 3.7 = **~7.4 GB**. Depth (24 layers) and
context (1024) are unchanged — it's still a deep, real billion-class model.

On top of that, two memory savers keep it comfortable (both **on by default**):

* **8-bit / Paged AdamW** (`bitsandbytes`): stores Adam's `m`/`v` in 1 byte each (or pages them
  to CPU), so optimizer state is ~2 GB (or ~0) on GPU instead of ~7 GB of fp32 AdamW state.
  **You must `pip install bitsandbytes`** — without it the code falls back to fp32 AdamW, whose
  ~7 GB of optimizer state will push you back into OOM.
* **Gradient checkpointing**: recompute block activations in backward instead of storing them.
  ~30% extra compute, big activation-memory saving.

If you still hit OOM, drop `batch_size` to `1` in the `full` preset (effective batch is kept
large by `grad_accum_steps`).

> **Reality check:** TinyStories is *tiny*, so even ~0.9B is enormous overkill for the data —
> this preset exists to give you the real "train a billion-class model on 2× T4" experience.
> For the best *stories-per-hour*, a few-hundred-M model would be plenty; tweak the `full`
> preset in [config.py](config.py) freely. To run the original 1.63B numbers you'd need a
> sharding strategy (FSDP) rather than DataParallel, or a single GPU with ≥40 GB.

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


Run cmd :- 
# Kaggle (2× T4):
!torchrun --standalone --nproc_per_node=2 train.py --preset base2 --kaggle
# Colab (1 GPU):
!python train.py --preset base2 --colab
