"""
train.py
========

The main training loop -- where the model actually learns.  Run it like:

    python train.py --preset tiny          # local CPU/GPU smoke run
    python train.py --preset full          # the real ~1.6B run on Kaggle's 2x T4

It wires together everything else in the project and implements all the standard
"make a big model train well on small hardware" tricks, each explained inline:

  * Mixed precision (fp16 autocast + GradScaler)  -> ~2x faster, ~half the memory, on T4.
  * Gradient accumulation                          -> a big *effective* batch from small
                                                      micro-batches that fit in VRAM.
  * nn.DataParallel                                -> use BOTH T4 GPUs from one process.
  * Cosine LR schedule with linear warmup          -> stable start, smooth annealing.
  * Gradient clipping                              -> prevents the occasional exploding step.
  * Checkpoint every N steps + auto-resume         -> survive Kaggle's session time limits.
  * Periodic loss / lr / tokens-per-sec / ETA logs and generated samples (watch it learn).
"""

from __future__ import annotations

import argparse
import math
import os
import time
import glob

import torch

from config import get_config
from model import GPT
from dataset import prepare_data, get_batch
from tokenizer import train_tokenizer


# =========================================================================================
# Learning-rate schedule: linear warmup, then cosine decay down to min_lr.
# =========================================================================================
def get_lr(step: int, cfg) -> float:
    # 1) Warmup: ramp linearly from 0 to the peak lr.  Protects the fragile early steps.
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    # 2) After the schedule ends, hold at the floor.
    if step >= cfg.max_steps:
        return cfg.min_lr
    # 3) Cosine decay from lr down to min_lr over the remaining steps.
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # goes 1 -> 0.
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


# =========================================================================================
# Checkpoint helpers
# =========================================================================================
def latest_checkpoint(ckpt_dir: str):
    """Return the path of the highest-numbered ckpt_XXXXXX.pt, or None if there are none."""
    paths = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt"))
    if not paths:
        return None
    # Sort by the integer step embedded in the filename.
    def step_of(p):
        return int(os.path.basename(p).split("_")[1].split(".")[0])
    return max(paths, key=step_of)


def save_checkpoint(path, raw_model, optimizer, scaler, step, cfg):
    """Write everything needed to resume EXACTLY where we left off."""
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "config": cfg.to_dict(),
    }, path)


# =========================================================================================
# Validation-loss estimate (no gradients, averaged over several batches)
# =========================================================================================
@torch.no_grad()
def estimate_val_loss(model, cfg, device, device_batch, autocast_ctx):
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = get_batch("val", cfg, device, batch_size=device_batch)
        with autocast_ctx():
            loss = model(x, y)
            if loss.dim() > 0:        # DataParallel returns one loss per GPU.
                loss = loss.mean()
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# =========================================================================================
# Main
# =========================================================================================
def main():
    parser = argparse.ArgumentParser(description="Train the from-scratch GPT.")
    parser.add_argument("--preset", default="tiny", choices=["tiny", "full"])
    parser.add_argument("--max_steps", type=int, default=None, help="override cfg.max_steps")
    parser.add_argument("--batch_size", type=int, default=None, help="override per-GPU micro-batch")
    parser.add_argument("--offline", action="store_true", help="use the synthetic corpus (no internet)")
    parser.add_argument("--no_resume", action="store_true", help="ignore existing checkpoints")
    parser.add_argument("--device", default=None, help="cuda | cpu (auto if unset)")
    args = parser.parse_args()

    # ---- assemble config from preset + CLI overrides --------------------------------------
    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.offline:
        overrides["offline"] = True
    cfg = get_config(args.preset, **overrides)

    # ---- device + precision setup ---------------------------------------------------------
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    num_gpus = torch.cuda.device_count() if device_type == "cuda" else 0

    # Mixed precision (fp16) ONLY makes sense on a GPU.  On CPU we run plain float32 so the
    # local smoke test works without GradScaler/autocast complaints.
    use_amp = (device_type == "cuda" and cfg.amp_dtype == "float16")
    amp_dtype = torch.float16 if use_amp else torch.float32

    def autocast_ctx():
        # A no-op-ish context on CPU; real fp16 autocast on GPU.
        return torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp)

    # GradScaler keeps fp16 gradients from underflowing to zero by scaling the loss up before
    # backward and unscaling before the optimizer step.  Disabled (pass-through) off-GPU, so
    # the "cuda" device tag here is harmless on CPU.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    torch.manual_seed(1337)
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True   # faster matmuls on Ampere+ (harmless on T4)
        torch.backends.cudnn.allow_tf32 = True

    print(f"[setup] preset={cfg.preset_name} device={device} num_gpus={num_gpus} "
          f"amp={use_amp} 8bit_adam={cfg.use_8bit_adam} grad_ckpt={cfg.use_grad_checkpoint}")

    # ---- data + tokenizer (train-once-or-load, then tokenize-once-or-load) ----------------
    tok = train_tokenizer(cfg)   # we need it to DECODE generated samples below.
    prepare_data(cfg)

    # ---- build the model ------------------------------------------------------------------
    model = GPT(cfg).to(device)
    print(f"[model] {model.num_params()/1e6:.2f}M parameters "
          f"({model.num_params(non_embedding=True)/1e6:.2f}M non-embedding)")

    # The "raw" (unwrapped) model is what we save, generate with, and build the optimizer on.
    raw_model = model
    # Effective micro-batch fed to model(): per-GPU size * number of GPUs.  DataParallel
    # scatters it back into cfg.batch_size sequences per GPU.
    device_batch = cfg.batch_size * max(1, num_gpus)
    if num_gpus > 1:
        print(f"[model] wrapping in nn.DataParallel across {num_gpus} GPUs "
              f"(fed batch {device_batch} -> {cfg.batch_size}/GPU).")
        model = torch.nn.DataParallel(model)

    # ---- optimizer ------------------------------------------------------------------------
    optimizer = raw_model.configure_optimizers(cfg, device_type)

    # ---- resume from the latest checkpoint, if any ----------------------------------------
    start_step = 0
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ckpt_path = None if args.no_resume else latest_checkpoint(cfg.ckpt_dir)
    if ckpt_path is not None:
        print(f"[resume] loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"] + 1
        print(f"[resume] continuing from step {start_step}")
    else:
        print("[resume] no checkpoint found; starting from scratch.")

    # ---- the training loop ----------------------------------------------------------------
    model.train()
    # Tokens processed per optimizer step (for the tokens/sec readout).
    tokens_per_step = device_batch * cfg.block_size * cfg.grad_accum_steps
    t0 = time.time()

    for step in range(start_step, cfg.max_steps):
        # Set this step's learning rate on every parameter group.
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # --- periodic validation -----------------------------------------------------------
        if step % cfg.eval_interval == 0:
            val_loss = estimate_val_loss(model, cfg, device, device_batch, autocast_ctx)
            print(f"[eval ] step {step:6d} | val_loss {val_loss:.4f}")

        # --- periodic text sample (watch the model learn) ---------------------------------
        if step % cfg.sample_interval == 0:
            generate_sample(raw_model, tok, cfg, device)

        # --- one optimizer step = grad_accum_steps micro-batches accumulated together ------
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = get_batch("train", cfg, device, batch_size=device_batch)
            with autocast_ctx():
                loss = model(x, y)
                if loss.dim() > 0:           # DataParallel -> one loss per GPU; average them.
                    loss = loss.mean()
                # Divide by grad_accum so the SUM of micro-batch grads == the mean over the
                # full effective batch (keeps the lr meaning the same as a single big batch).
                loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()    # accumulates into .grad (scaled up for fp16).
            loss_accum += loss.item()

        # Unscale grads back to true magnitude, then clip their global norm for stability.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        # scaler.step skips the update if it detects inf/nan grads (an fp16 overflow), then
        # scaler.update adjusts the scale factor for next time.
        scaler.step(optimizer)
        scaler.update()

        # --- logging -----------------------------------------------------------------------
        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            steps_done = step - start_step + 1
            tok_per_sec = tokens_per_step * steps_done / dt
            steps_left = cfg.max_steps - step - 1
            eta_sec = (dt / steps_done) * steps_left
            print(f"[train] step {step:6d}/{cfg.max_steps} | loss {loss_accum:.4f} | "
                  f"lr {lr:.2e} | {tok_per_sec:,.0f} tok/s | ETA {fmt_eta(eta_sec)}")

        # --- checkpoint --------------------------------------------------------------------
        if step > 0 and step % cfg.ckpt_interval == 0:
            path = os.path.join(cfg.ckpt_dir, f"ckpt_{step:06d}.pt")
            save_checkpoint(path, raw_model, optimizer, scaler, step, cfg)
            print(f"[ckpt ] saved {path}")

    # Always save a final checkpoint at the end of the run.
    final = os.path.join(cfg.ckpt_dir, f"ckpt_{cfg.max_steps:06d}.pt")
    save_checkpoint(final, raw_model, optimizer, scaler, cfg.max_steps - 1, cfg)
    print(f"[done ] training complete; final checkpoint {final}")


# =========================================================================================
# Helpers used by main()
# =========================================================================================
def fmt_eta(seconds: float) -> str:
    """Pretty-print a seconds count as h:mm:ss."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


@torch.no_grad()
def generate_sample(raw_model, tok, cfg, device, max_new_tokens: int = 100):
    """Generate a short paragraph from the end-of-text seed and print it, so you can
    literally watch the model's writing improve as training progresses."""
    was_training = raw_model.training
    raw_model.eval()
    # Seed with a single <|endoftext|> token = "start a fresh document".
    idx = torch.tensor([[tok.eot_id]], dtype=torch.long, device=device)
    out = raw_model.generate(idx, max_new_tokens=max_new_tokens,
                             temperature=0.8, top_k=200, eot_id=tok.eot_id)
    text = tok.decode(out[0].tolist())
    print("-" * 70)
    print("[sample]", text.strip()[:500])
    print("-" * 70)
    if was_training:
        raw_model.train()


if __name__ == "__main__":
    main()
