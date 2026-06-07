"""
config.py
=========

ALL hyperparameters for the project live here, in one place, so you never have to
hunt through the training code to change a number.

We expose ONE dataclass, `Config`, that bundles together three groups of settings:

  1. Model architecture   (how big / what shape the transformer is)
  2. Training              (optimizer, schedule, batching, logging, checkpoints)
  3. Data / tokenizer      (where files live, how much data to use)

and a single helper `get_config(preset)` that returns a ready-to-use Config for one
of two presets:

  * "tiny"  -> a microscopic model that trains on a CPU/laptop in seconds.
              Use this to LEARN and to smoke-test that the whole pipeline works.

  * "full"  -> the real ~0.9B-parameter model meant for Kaggle's 2x T4 GPUs.

Why a dataclass instead of a dict?  A dataclass gives you autocompletion, type
hints, and a clear list of every knob in one screen.  It is also trivially
serializable, which we exploit when we save it inside a checkpoint.
"""

from __future__ import annotations  # lets us write type hints like "Config" inside the class

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Config:
    # A short name so logs / checkpoints can tell which preset they came from.
    preset_name: str = "tiny"

    # =====================================================================================
    # 1) MODEL ARCHITECTURE
    # =====================================================================================
    # These five numbers define the *shape* of the transformer.  Everything else in the
    # model (parameter counts, memory use, FLOPs) is derived from them.

    vocab_size: int = 4096          # number of distinct tokens the model knows.
                                    #   Must match the trained BPE tokenizer.  TinyStories
                                    #   has a small vocabulary, so a small vocab is plenty
                                    #   and keeps the (vocab x n_embd) embedding table cheap.

    n_layer: int = 2                # number of transformer "blocks" stacked on top of each
                                    #   other.  Depth.  More layers -> more reasoning steps.

    n_head: int = 4                 # number of attention heads per block.  The model splits
                                    #   its n_embd-dim vectors into n_head independent
                                    #   "views" of size (n_embd // n_head) and attends with
                                    #   each.  n_embd MUST be divisible by n_head.

    n_embd: int = 128               # the model's "width": the size of every token's hidden
                                    #   vector as it flows through the network.

    ffn_hidden: int = 256           # inner width of the feed-forward (MLP) sub-layer.
                                    #   In a classic transformer this is 4 * n_embd.  We make
                                    #   it explicit so the "full" preset can match the spec's
                                    #   FFN dim of 8192 exactly.  With SwiGLU there are THREE
                                    #   matrices of this size, which is why the FFN dominates
                                    #   the parameter count (see README for the math).

    block_size: int = 128           # context length: the maximum number of tokens the model
                                    #   can look at when predicting the next one.  Also the
                                    #   length of every training example.

    # --- positional + normalization knobs ------------------------------------------------
    rope_theta: float = 10000.0     # base frequency for RoPE (Rotary Positional Embeddings).
                                    #   10000 is the standard value from the RoPE/Llama papers.

    norm_eps: float = 1e-5          # tiny constant added inside RMSNorm to avoid divide-by-0.

    dropout: float = 0.0            # dropout probability.  0.0 is normal for big-model
                                    #   pretraining (we have far more data than parameters,
                                    #   so we don't need this kind of regularization).

    # =====================================================================================
    # 2) TRAINING
    # =====================================================================================
    # --- batching ------------------------------------------------------------------------
    batch_size: int = 16            # number of sequences per micro-step PER GPU.  Keep this
                                    #   small for big models; we recover a large *effective*
                                    #   batch with gradient accumulation below.

    grad_accum_steps: int = 2       # accumulate gradients over this many micro-steps before
                                    #   each optimizer update.  Effective batch (in sequences)
                                    #   = batch_size * grad_accum_steps * num_gpus.  This lets
                                    #   a 16 GB GPU "simulate" a much larger batch than fits in
                                    #   memory at once.

    # --- optimizer (AdamW) ---------------------------------------------------------------
    lr: float = 1e-3                # peak learning rate (reached at the end of warmup).
    min_lr: float = 1e-4            # final learning rate at the end of cosine decay.
    warmup_steps: int = 50          # linearly ramp lr 0 -> lr over this many steps.  Warmup
                                    #   stops the very first (noisy) updates from blowing up.
    max_steps: int = 200            # total number of optimizer steps to train for.
    weight_decay: float = 0.1       # L2-style regularization, applied only to matrices
                                    #   (not to biases / norm weights -- see model.py).
    beta1: float = 0.9              # AdamW momentum term.
    beta2: float = 0.95             # AdamW variance term (0.95 is the LLM convention, a bit
                                    #   lower than the 0.999 default -> reacts faster).
    grad_clip: float = 1.0          # clip the global gradient norm to this value each step.
                                    #   The single most important trick for stable training.

    # --- precision / memory --------------------------------------------------------------
    amp_dtype: str = "float16"      # autocast dtype on GPU.  T4 GPUs (Turing) support fast
                                    #   float16 but NOT bfloat16, so we use float16 + a
                                    #   GradScaler.  On CPU we ignore this and run float32.

    use_8bit_adam: bool = False     # if True (and CUDA + bitsandbytes are available) use an
                                    #   8-bit AdamW.  This stores Adam's two state tensors in
                                    #   1 byte each instead of 4, cutting optimizer memory ~4x
                                    #   -- a trick that helps the model fit on a 14.5 GB T4.
                                    #   State stays ON the GPU -> fast.

    use_paged_adam: bool = False    # if True, use the *Paged* 8-bit AdamW, which offloads
                                    #   optimizer state to CPU RAM and pages it over PCIe every
                                    #   step.  Saves more GPU memory but is MUCH SLOWER.  Only
                                    #   enable if plain 8-bit AdamW still OOMs.

    use_grad_checkpoint: bool = False  # if True, don't keep activations for the backward
                                    #   pass; recompute them instead.  Trades ~30% extra
                                    #   compute for a large activation-memory saving.

    # --- intervals (all measured in optimizer steps) -------------------------------------
    log_interval: int = 10          # print loss / lr / tokens-per-sec / ETA this often.
    sample_interval: int = 100      # generate a sample paragraph this often (watch it learn).
    eval_interval: int = 100        # estimate validation loss this often.
    eval_iters: int = 20            # number of batches to average for each val-loss estimate.
    ckpt_interval: int = 100        # write a checkpoint this often.
    keep_last_checkpoints: int = 2  # disk saver: after each save, keep only the newest N
                                    #   checkpoints and delete older ones.  Each checkpoint is
                                    #   several GB for the full model, so without this a long
                                    #   run fills the disk.  Set to 0 to keep ALL checkpoints.

    # =====================================================================================
    # 3) DATA / TOKENIZER
    # =====================================================================================
    dataset_name: str = "roneneldan/TinyStories"  # HuggingFace dataset id.

    data_dir: str = "data"          # where the pre-tokenized train.bin / val.bin live.
    ckpt_dir: str = "checkpoints"   # where training checkpoints are written.
    tokenizer_path: str = "tokenizer.json"  # where the trained BPE tokenizer is saved.

    # Caps so local runs finish quickly.  None = "use the entire corpus" (the Kaggle setting).
    max_train_tokens: Optional[int] = 2_000_000   # stop writing train.bin after this many tokens.
    max_val_tokens: Optional[int] = 100_000       # stop writing val.bin after this many tokens.
    tokenizer_train_docs: int = 20_000            # how many documents to train the BPE on.

    offline: bool = False           # if True (or if streaming fails) use a small built-in
                                    #   synthetic corpus instead of downloading TinyStories.
                                    #   Lets the pipeline run with no internet.

    # ------------------------------------------------------------------------------------
    def num_params_estimate(self) -> int:
        """
        Quick analytical estimate of the parameter count from the shape numbers, so the
        README's claims are reproducible and so you can sanity-check a preset *without*
        building the model.  The real count is printed by model.GPT.num_params().

        Breakdown per the architecture we build in model.py (weights are tied, so the
        output head shares the embedding table and is not counted twice):

          embedding         : vocab_size * n_embd
          per attention     : 4 * n_embd^2            (fused QKV = 3, output proj = 1)
          per SwiGLU MLP     : 3 * n_embd * ffn_hidden (gate, up, down)
          norms (RMSNorm)    : ~2 * n_embd per block + n_embd final   (tiny, included)
        """
        embed = self.vocab_size * self.n_embd
        per_attn = 4 * self.n_embd * self.n_embd
        per_mlp = 3 * self.n_embd * self.ffn_hidden
        per_norm = 2 * self.n_embd
        per_block = per_attn + per_mlp + per_norm
        total = embed + self.n_layer * per_block + self.n_embd  # + final norm
        return total

    def to_dict(self) -> dict:
        """Plain dict for saving inside a checkpoint or printing."""
        return asdict(self)


# =========================================================================================
# PRESETS
# =========================================================================================
def get_config(preset: str = "tiny", **overrides) -> Config:
    """
    Return a Config for the requested preset, then apply any keyword overrides on top
    (used by the CLIs, e.g. `--max_steps 20`).

    Two presets:
      "tiny" -> learn / smoke-test locally (CPU friendly).
      "full" -> the real ~0.9B model for Kaggle 2x T4.
    """
    preset = preset.lower()

    if preset == "tiny":
        # Defaults of the dataclass ARE the tiny preset, so we just stamp the name.
        cfg = Config(preset_name="tiny")

    elif preset == "full":
        cfg = Config(
            preset_name="full",
            # ---- architecture: a ~0.9B model sized to actually FIT 2x T4 under DataParallel.
            #   WHY not the original 2048-wide / 8192-FFN (=1.63B) spec?  DataParallel keeps
            #   the whole model on GPU 0 AND gathers every gradient there, so GPU 0 needs
            #   ~params + ~grads = (6.5 + 6.5) = ~13 GB for 1.63B -- which overflows a 14.56 GB
            #   T4 *during backward*, before the optimizer is even touched.  Narrowing the
            #   width to 1536 (and FFN to 6144) drops this to ~3.7 + 3.7 = ~7.4 GB, leaving
            #   room for activations + the 8-bit optimizer.  Depth (24 layers) and context
            #   (1024) are unchanged, so it's still a deep, real billion-class model.
            vocab_size=8000,        # matches the BPE tokenizer we train for Kaggle.
            n_layer=24,             # 24 transformer blocks deep (unchanged).
            n_head=12,              # 12 attention heads (1536 / 12 = 128 dims per head).
            n_embd=1536,            # model width (narrowed from 2048 to fit GPU 0).
            ffn_hidden=6144,        # SwiGLU inner dim (4 x n_embd).
            block_size=1024,        # 1024-token context window (unchanged).
            dropout=0.0,

            # ---- batching: small micro-batch, big effective batch via accumulation ----
            batch_size=2,           # per-GPU micro-batch.  Kept at 2 so that params + fp32
                                    #   grads + activations fit a single 16 GB T4 (remember
                                    #   DataParallel pins the whole model on GPU 0).  Raise to
                                    #   4 only if you see spare VRAM in `nvidia-smi`.
            grad_accum_steps=8,     # -> effective batch = 2 * 8 * 2 GPUs = 32 sequences
                                    #    = 32 * 1024 = 32,768 tokens per optimizer step.

            # ---- optimizer / schedule (the spec: AdamW, lr 3e-4, cosine decay) ----
            lr=3e-4,
            min_lr=3e-5,            # decay down to 10% of peak.
            warmup_steps=200,
            max_steps=100_000,      # set as high as you have time for; resume picks up where
                                    #   you left off.  ~100k steps is a long, meaningful run.
            weight_decay=0.1,
            beta2=0.95,
            grad_clip=1.0,

            # ---- precision / memory: the tricks that make ~0.9B fit on 2x T4 ----
            amp_dtype="float16",        # T4 = fast fp16, no bf16.
            use_8bit_adam=True,         # 8-bit/Paged AdamW -> optimizer state ~1.8 GB not ~7.4 GB.
            use_grad_checkpoint=True,   # recompute activations -> big activation-memory save.

            # ---- intervals (the spec's cadence) ----
            log_interval=100,
            sample_interval=500,
            eval_interval=1000,
            eval_iters=50,
            ckpt_interval=1000,

            # ---- data: use the WHOLE corpus on Kaggle (caps off) ----
            max_train_tokens=None,
            max_val_tokens=None,
            tokenizer_train_docs=200_000,  # more docs -> a better-quality BPE vocabulary.
        )

    else:
        raise ValueError(f"Unknown preset {preset!r}. Choose 'tiny' or 'full'.")

    # Apply CLI / programmatic overrides last so they always win.
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Config has no field {key!r} to override.")
        setattr(cfg, key, value)

    # A couple of cheap sanity checks that catch the most common misconfigurations early.
    assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
    assert (cfg.n_embd // cfg.n_head) % 2 == 0, "head dim must be even for RoPE"
    return cfg


if __name__ == "__main__":
    # Run `python config.py` to print both presets and their estimated sizes.
    for name in ("tiny", "full"):
        c = get_config(name)
        n = c.num_params_estimate()
        print(f"[{name:4s}] ~{n/1e6:8.2f}M params | "
              f"{c.n_layer}L {c.n_head}H {c.n_embd}d ffn={c.ffn_hidden} "
              f"ctx={c.block_size} vocab={c.vocab_size}")
