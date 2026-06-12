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
from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist

from config import get_config
from model import GPT
from dataset import prepare_data, get_batch
from tokenizer import train_tokenizer


# =========================================================================================
# Multi-GPU: DistributedDataParallel (the right way to use Kaggle's 2x T4)
# =========================================================================================
def setup_distributed():
    """
    If launched under torchrun (which sets RANK / WORLD_SIZE / LOCAL_RANK), start the process
    group so we can use DistributedDataParallel.  DDP keeps a persistent model replica on each
    GPU and only all-reduces gradients -- unlike nn.DataParallel, which re-broadcasts the whole
    model across PCIe EVERY step (that's why 2x T4 with DataParallel was slower than 1 T4).

    Uses the nccl backend on GPU, or gloo on CPU so the DDP wiring can be smoke-tested with no
    GPUs (`torchrun --nproc_per_node=2 train.py --preset tiny --offline`).

    Returns (is_ddp, rank, world_size, local_rank, device_or_None, is_master).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # Generous timeout: non-master ranks sit at a barrier while the master tokenizes the
        # corpus (can take a long while for the big v0.3 mix), so the default ~30 min isn't safe.
        dist.init_process_group(backend=backend, timeout=timedelta(hours=2))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"
        return True, rank, world_size, local_rank, device, (rank == 0)
    return False, 0, 1, 0, None, True


def apply_platform(cfg, args):
    """
    Tune the config for the target platform.  Neither a single GPU nor DDP has nn.DataParallel's
    gradient-gather buffer, so both can drop the slow PAGED optimizer for the fast on-GPU 8-bit
    one.  We also size the per-GPU batch sensibly per platform (CLI --batch_size still wins).
    """
    if not (args.colab or args.kaggle):
        return
    cfg.use_paged_adam = False     # no DataParallel gather buffer -> non-paged fits and is fast
    if args.batch_size is None:    # don't clobber an explicit --batch_size
        if cfg.preset_name == "base2":
            cfg.batch_size = 12 if args.colab else 8
    tag = "colab (single GPU)" if args.colab else "kaggle (2x T4, DDP)"
    print(f"[platform] tuned for {tag}: non-paged adam, batch_size={cfg.batch_size}/GPU")


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


def prune_checkpoints(ckpt_dir: str, keep: int):
    """Keep only the `keep` most recent checkpoints; delete the rest.

    Each checkpoint is several GB (full-model weights + optimizer state), so on a long run
    they'd pile up and fill the disk.  keep<=0 disables pruning (keeps everything).
    """
    if keep <= 0:
        return
    paths = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt"))

    def step_of(p):
        return int(os.path.basename(p).split("_")[1].split(".")[0])

    paths = sorted(paths, key=step_of)          # oldest -> newest
    for old in paths[:-keep]:                    # everything except the newest `keep`
        try:
            os.remove(old)
            print(f"[ckpt ] pruned old {os.path.basename(old)}")
        except OSError:
            pass


# =========================================================================================
# HuggingFace Hub sync -- push checkpoints to a model repo so they survive Kaggle sessions
# =========================================================================================
def _ckpt_step(name: str) -> int:
    """Parse the step number out of a 'ckpt_XXXXXX.pt' filename (ignores any subfolder)."""
    return int(os.path.basename(name).split("_")[1].split(".")[0])


def _hf_prefix(cfg) -> str:
    """Repo path prefix that namespaces a version's checkpoints, e.g. 'v03/' (or '' = root)."""
    return f"{cfg.hf_subfolder}/" if cfg.hf_subfolder else ""


def _hf_list_ckpts(api, cfg):
    """Sorted (oldest->newest) list of this version's checkpoint paths on the Hub."""
    prefix = _hf_prefix(cfg)
    files = [f for f in api.list_repo_files(cfg.hf_repo, repo_type="model")
             if f.startswith(prefix + "ckpt_") and f.endswith(".pt")]
    return sorted(files, key=_ckpt_step)


def hf_upload_checkpoint(cfg, local_path: str):
    """
    Upload one checkpoint to the Hub repo `cfg.hf_repo` (under cfg.hf_subfolder so versions
    don't collide), then delete old ones there so only the newest `cfg.hf_keep` remain.
    Auth comes from the HF_TOKEN environment variable (set it as a Kaggle Secret; never
    hard-code it).

    Failures are deliberately NON-FATAL: a flaky upload should never kill a training run, so
    we just warn and carry on -- the local checkpoint is still safe on disk.
    """
    if not cfg.hf_repo:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(repo_id=cfg.hf_repo, repo_type="model", exist_ok=True)
        name = os.path.basename(local_path)
        path_in_repo = _hf_prefix(cfg) + name
        api.upload_file(path_or_fileobj=local_path, path_in_repo=path_in_repo,
                        repo_id=cfg.hf_repo, repo_type="model",
                        commit_message=f"checkpoint {path_in_repo}")
        print(f"[hf   ] uploaded {path_in_repo} -> {cfg.hf_repo}")
        # Prune old checkpoints on the Hub (keep the newest cfg.hf_keep).
        remote = _hf_list_ckpts(api, cfg)
        for old in (remote[:-cfg.hf_keep] if cfg.hf_keep > 0 else []):
            api.delete_file(path_in_repo=old, repo_id=cfg.hf_repo, repo_type="model")
            print(f"[hf   ] pruned {old} from the Hub")
    except Exception as e:
        print(f"[hf   ] upload failed ({e}); continuing training (local checkpoint is safe).")


def hf_download_latest(cfg):
    """
    If this version's subfolder on the Hub has checkpoints, download the newest one so training
    can resume across a fresh session (e.g. a brand-new Kaggle/Colab notebook with no local
    files). Returns the local path (wherever it landed), or None if nothing was fetched.
    """
    if not cfg.hf_repo:
        return None
    try:
        from huggingface_hub import HfApi, hf_hub_download
        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        remote = _hf_list_ckpts(api, cfg)
        if not remote:
            return None
        newest = remote[-1]
        print(f"[hf   ] no local checkpoint; downloading {newest} from {cfg.hf_repo} ...")
        cached = hf_hub_download(repo_id=cfg.hf_repo, filename=newest, repo_type="model",
                                 local_dir=cfg.ckpt_dir, token=token)
        # Flatten into ckpt_dir root (the subfolder path would hide it from latest_checkpoint()
        # and from the other DDP ranks).
        flat = os.path.join(cfg.ckpt_dir, os.path.basename(newest))
        if os.path.abspath(cached) != os.path.abspath(flat):
            import shutil
            shutil.copy(cached, flat)
        return flat
    except Exception as e:
        print(f"[hf   ] could not fetch from the Hub ({e}); starting fresh.")
        return None


def hf_upload_data(cfg):
    """
    Upload the tokenizer + tokenized train.bin/val.bin to the Hub (under {hf_subfolder}/data/)
    so a fresh session can PULL them instead of re-tokenizing the slow corpus.  Run once after
    a fresh tokenization.  Non-fatal on failure (the data is still safe locally).
    """
    if not cfg.hf_repo:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(repo_id=cfg.hf_repo, repo_type="model", exist_ok=True)
        prefix = _hf_prefix(cfg) + "data/"
        for local in (cfg.tokenizer_path,
                      os.path.join(cfg.data_dir, "train.bin"),
                      os.path.join(cfg.data_dir, "val.bin")):
            if os.path.exists(local):
                dest = prefix + os.path.basename(local)
                api.upload_file(path_or_fileobj=local, path_in_repo=dest,
                                repo_id=cfg.hf_repo, repo_type="model",
                                commit_message=f"data {os.path.basename(local)}")
                print(f"[hf   ] uploaded {dest} -> {cfg.hf_repo}")
    except Exception as e:
        print(f"[hf   ] data upload failed ({e}); continuing (data is still local).")


def hf_download_data(cfg):
    """
    If the tokenizer/.bin are missing locally but present on the Hub, download them so training
    resumes on a fresh session WITHOUT re-tokenizing.  Non-fatal: on any failure we just fall
    through and the normal train_tokenizer()/prepare_data() will regenerate them.
    """
    if not cfg.hf_repo:
        return
    train_bin = os.path.join(cfg.data_dir, "train.bin")
    if os.path.exists(cfg.tokenizer_path) and os.path.exists(train_bin):
        return  # already have everything locally
    try:
        import shutil
        from huggingface_hub import HfApi, hf_hub_download
        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        remote = set(api.list_repo_files(cfg.hf_repo, repo_type="model"))
        prefix = _hf_prefix(cfg) + "data/"
        os.makedirs(cfg.data_dir, exist_ok=True)

        def pull(remote_name, local_target):
            if prefix + remote_name not in remote:
                return False
            cached = hf_hub_download(repo_id=cfg.hf_repo, filename=prefix + remote_name,
                                     repo_type="model", token=token)
            shutil.copy(cached, local_target)
            print(f"[hf   ] pulled {prefix}{remote_name} -> {local_target}")
            return True

        pull(os.path.basename(cfg.tokenizer_path), cfg.tokenizer_path)
        got = pull("train.bin", train_bin)
        pull("val.bin", os.path.join(cfg.data_dir, "val.bin"))
        if got:
            print("[hf   ] pulled tokenized data from the Hub (no re-tokenization needed).")
    except Exception as e:
        print(f"[hf   ] could not fetch data from the Hub ({e}); will tokenize locally.")


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
    parser.add_argument("--preset", default="tiny", choices=["tiny", "full", "base2"])
    parser.add_argument("--max_steps", type=int, default=None, help="override cfg.max_steps")
    parser.add_argument("--batch_size", type=int, default=None, help="override per-GPU micro-batch")
    parser.add_argument("--log_interval", type=int, default=None,
                        help="print a train log every N steps (lower = faster feedback)")
    parser.add_argument("--ckpt_interval", type=int, default=None,
                        help="save a checkpoint every N steps (lower = lose less on a crash)")
    parser.add_argument("--offline", action="store_true", help="use the synthetic corpus (no internet)")
    parser.add_argument("--no_resume", action="store_true", help="ignore existing checkpoints")
    parser.add_argument("--hf_repo", default=None,
                        help='HuggingFace repo for checkpoint sync, e.g. "Yashhh999/dexter" '
                             '(pass "" to disable). Needs the HF_TOKEN env var.')
    parser.add_argument("--colab", action="store_true",
                        help="optimize for a single Colab GPU (no DataParallel/DDP)")
    parser.add_argument("--kaggle", action="store_true",
                        help="optimize for Kaggle 2x T4 via DDP "
                             "(launch with: torchrun --nproc_per_node=2 train.py ... --kaggle)")
    parser.add_argument("--device", default=None, help="cuda | cpu (auto if unset)")
    args = parser.parse_args()
    if args.colab and args.kaggle:
        raise SystemExit("Pick one of --colab / --kaggle, not both.")

    # ---- assemble config from preset + CLI overrides --------------------------------------
    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.log_interval is not None:
        overrides["log_interval"] = args.log_interval
    if args.ckpt_interval is not None:
        overrides["ckpt_interval"] = args.ckpt_interval
    if args.hf_repo is not None:
        overrides["hf_repo"] = args.hf_repo
    if args.offline:
        overrides["offline"] = True
    cfg = get_config(args.preset, **overrides)
    apply_platform(cfg, args)   # --colab / --kaggle tuning (after preset + CLI overrides)

    # ---- distributed (DDP) + device setup -------------------------------------------------
    is_ddp, rank, world_size, local_rank, ddp_device, master = setup_distributed()
    device = ddp_device if is_ddp else (args.device or
             ("cuda" if torch.cuda.is_available() else "cpu"))
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
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

    if master:
        print(f"[setup] preset={cfg.preset_name} device={device} ddp={is_ddp} world={world_size} "
              f"amp={use_amp} 8bit_adam={cfg.use_8bit_adam} paged={cfg.use_paged_adam} "
              f"grad_ckpt={cfg.use_grad_checkpoint}")

    # ---- data + tokenizer: MASTER prepares (pull/tokenize/upload); other ranks wait --------
    # Only the master trains the tokenizer / tokenizes (otherwise every rank would redo it and
    # race on the files).  After the barrier the files exist, so the others just load them.
    tok = None
    if master:
        if cfg.hf_repo:
            hf_download_data(cfg)          # pull tokenizer + .bin from the Hub if present
        tok = train_tokenizer(cfg)         # train-or-load the tokenizer
        fresh = prepare_data(cfg)          # tokenize-or-skip the .bin files
        if cfg.hf_repo and fresh:
            hf_upload_data(cfg)            # push tokenizer + .bin so future sessions just pull
    if is_ddp:
        dist.barrier()
        if not master:
            tok = train_tokenizer(cfg)     # load the files master created (shared filesystem)

    # ---- build the model ------------------------------------------------------------------
    raw_model = GPT(cfg).to(device)        # identical init on every rank (same seed 1337)
    if master:
        print(f"[model] {raw_model.num_params()/1e6:.2f}M parameters "
              f"({raw_model.num_params(non_embedding=True)/1e6:.2f}M non-embedding)")

    # ---- parallelism: DDP (best for 2x T4) > single GPU > nn.DataParallel (legacy fallback) -
    model = raw_model
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(raw_model, device_ids=[local_rank] if device_type == "cuda" else None)
        device_batch, world = cfg.batch_size, world_size     # each rank does cfg.batch_size
        if master:
            print(f"[model] DistributedDataParallel across {world_size} processes "
                  f"({cfg.batch_size}/GPU).")
    elif args.colab or args.kaggle:
        if args.kaggle and num_gpus > 1:                     # --kaggle without torchrun
            print("[platform] --kaggle but NOT launched with torchrun -> using ONE GPU. For "
                  "real 2x T4 speed run:\n"
                  "           torchrun --standalone --nproc_per_node=2 train.py ... --kaggle")
        device_batch, world = cfg.batch_size, 1
    elif num_gpus > 1 and device_type == "cuda":
        model = torch.nn.DataParallel(raw_model)
        device_batch, world = cfg.batch_size * num_gpus, 1
        print(f"[model] nn.DataParallel across {num_gpus} GPUs (fed batch {device_batch}).")
    else:
        device_batch, world = cfg.batch_size, 1

    # ---- optimizer ------------------------------------------------------------------------
    optimizer = raw_model.configure_optimizers(cfg, device_type)

    # ---- resume from the latest checkpoint, if any ----------------------------------------
    start_step = 0
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    # Master pulls the newest checkpoint from the Hub if there's nothing local; then ALL ranks
    # load the SAME file, so their weights AND optimizer state stay identical (required for DDP
    # -- otherwise the per-rank optimizer states would drift the replicas apart).
    if master and not args.no_resume and cfg.hf_repo and latest_checkpoint(cfg.ckpt_dir) is None:
        hf_download_latest(cfg)            # flattened into ckpt_dir so latest_checkpoint finds it
    if is_ddp:
        dist.barrier()
    ckpt_path = None if args.no_resume else latest_checkpoint(cfg.ckpt_dir)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"] + 1
        if master:
            print(f"[resume] continuing from step {start_step} ({ckpt_path})")
    elif master:
        print("[resume] no checkpoint found; starting from scratch.")

    # ---- the training loop ----------------------------------------------------------------
    model.train()
    # Different data per rank: the model was initialized identically (seed 1337), now we offset
    # the RNG so each rank samples DIFFERENT windows from the memmap.
    torch.manual_seed(1337 + rank)
    # Total tokens processed per optimizer step across ALL ranks (for the tokens/sec readout).
    tokens_per_step = device_batch * world * cfg.block_size * cfg.grad_accum_steps
    t0 = time.time()

    for step in range(start_step, cfg.max_steps):
        # Set this step's learning rate on every parameter group.
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # --- periodic validation (master only; uses the unwrapped model) ------------------
        if master and step % cfg.eval_interval == 0:
            val_loss = estimate_val_loss(raw_model, cfg, device, cfg.batch_size, autocast_ctx)
            print(f"[eval ] step {step:6d} | val_loss {val_loss:.4f}")

        # --- periodic text sample (master only) -------------------------------------------
        if master and step % cfg.sample_interval == 0:
            generate_sample(raw_model, tok, cfg, device)

        # --- one optimizer step = grad_accum_steps micro-batches accumulated together ------
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(cfg.grad_accum_steps):
            x, y = get_batch("train", cfg, device, batch_size=device_batch)
            # Under DDP, suppress the gradient all-reduce on every micro-step except the last,
            # so we sync once per optimizer step instead of grad_accum_steps times.
            sync_ctx = (model.no_sync() if (is_ddp and micro < cfg.grad_accum_steps - 1)
                        else nullcontext())
            with sync_ctx:
                with autocast_ctx():
                    loss = model(x, y)
                    if loss.dim() > 0:       # DataParallel -> one loss per GPU; average them.
                        loss = loss.mean()
                    # Divide by grad_accum so the SUM of micro-batch grads == the mean over the
                    # full effective batch (keeps the lr meaning the same as a big batch).
                    loss = loss / cfg.grad_accum_steps
                scaler.scale(loss).backward()  # accumulates into .grad (scaled up for fp16).
            loss_accum += loss.item()

        # Unscale grads back to true magnitude, then clip their global norm for stability.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        # scaler.step skips the update if it detects inf/nan grads (an fp16 overflow), then
        # scaler.update adjusts the scale factor for next time.
        scaler.step(optimizer)
        scaler.update()

        # --- logging (master only) ---------------------------------------------------------
        if master and step % cfg.log_interval == 0:
            dt = time.time() - t0
            steps_done = step - start_step + 1
            tok_per_sec = tokens_per_step * steps_done / dt
            steps_left = cfg.max_steps - step - 1
            eta_sec = (dt / steps_done) * steps_left
            print(f"[train] step {step:6d}/{cfg.max_steps} | loss {loss_accum:.4f} | "
                  f"lr {lr:.2e} | {tok_per_sec:,.0f} tok/s | ETA {fmt_eta(eta_sec)}")

        # --- checkpoint: local (master only) -----------------------------------------------
        if master and step > 0 and step % cfg.ckpt_interval == 0:
            path = os.path.join(cfg.ckpt_dir, f"ckpt_{step:06d}.pt")
            save_checkpoint(path, raw_model, optimizer, scaler, step, cfg)
            print(f"[ckpt ] saved {path}")
            prune_checkpoints(cfg.ckpt_dir, cfg.keep_last_checkpoints)

        # --- checkpoint: HuggingFace Hub (master only) -------------------------------------
        # Less often than local saves (uploads cost bandwidth), but enough to survive the
        # session being deleted.  Pushes whatever the newest local checkpoint is.
        if master and cfg.hf_repo and step > 0 and step % cfg.hf_push_interval == 0:
            hf_upload_checkpoint(cfg, latest_checkpoint(cfg.ckpt_dir))

    # Always save a final checkpoint at the end of the run (master only).
    if master:
        final = os.path.join(cfg.ckpt_dir, f"ckpt_{cfg.max_steps:06d}.pt")
        save_checkpoint(final, raw_model, optimizer, scaler, cfg.max_steps - 1, cfg)
        prune_checkpoints(cfg.ckpt_dir, cfg.keep_last_checkpoints)
        hf_upload_checkpoint(cfg, final)
        print(f"[done ] training complete; final checkpoint {final}")
    if is_ddp:
        dist.destroy_process_group()


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
                             temperature=0.8, top_k=200, top_p=0.95,
                             repetition_penalty=1.1, eot_id=tok.eot_id)
    text = tok.decode(out[0].tolist())
    print("-" * 70)
    print("[sample]", text.strip()[:500])
    print("-" * 70)
    if was_training:
        raw_model.train()


if __name__ == "__main__":
    main()
