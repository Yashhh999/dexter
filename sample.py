"""
sample.py
=========

Load a trained checkpoint and generate text from it.  Use this after (or during) training
to see what the model has learned.

    python sample.py --preset tiny --prompt "Once upon a time"
    python sample.py --preset full --prompt "The little dog" --num_samples 3 --temperature 0.8

By default it loads the LATEST checkpoint in the preset's checkpoint directory; pass
--ckpt to load a specific one.

The checkpoint stores the full Config it was trained with, so we rebuild the exact same
model shape automatically -- you don't have to remember the architecture flags.
"""

from __future__ import annotations

import argparse
import glob
import os

import torch

from config import Config, get_config
from model import GPT
from tokenizer import Tokenizer


def latest_checkpoint(ckpt_dir: str):
    paths = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt"))
    if not paths:
        return None
    return max(paths, key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))


def pull_from_hf(cfg, want_ckpt=True):
    """
    On a fresh session (e.g. a new Lightning/Colab box), fetch the tokenizer and the latest
    checkpoint from the Hub if they're not already local -- so sampling works WITHOUT manual
    downloads (this is what fixes 'tokenizer not found').  Needs HF_TOKEN only for a private
    repo.  Non-fatal: any failure just falls through to the normal local-file path.
    """
    if not getattr(cfg, "hf_repo", ""):
        return
    try:
        import shutil
        from huggingface_hub import HfApi, hf_hub_download
        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        prefix = (cfg.hf_subfolder + "/") if cfg.hf_subfolder else ""
        files = api.list_repo_files(cfg.hf_repo, repo_type="model")

        # tokenizer (small): {prefix}data/<tokenizer filename>
        tok_remote = prefix + "data/" + os.path.basename(cfg.tokenizer_path)
        if not os.path.exists(cfg.tokenizer_path) and tok_remote in files:
            shutil.copy(hf_hub_download(cfg.hf_repo, tok_remote, repo_type="model", token=token),
                        cfg.tokenizer_path)
            print(f"[sample] pulled tokenizer {tok_remote}")

        # latest checkpoint: {prefix}ckpt_XXXXXX.pt  (flattened into ckpt_dir)
        if want_ckpt and latest_checkpoint(cfg.ckpt_dir) is None:
            ckpts = sorted([f for f in files if f.startswith(prefix + "ckpt_") and f.endswith(".pt")],
                           key=lambda f: int(os.path.basename(f).split("_")[1].split(".")[0]))
            if ckpts:
                os.makedirs(cfg.ckpt_dir, exist_ok=True)
                local = hf_hub_download(cfg.hf_repo, ckpts[-1], repo_type="model",
                                        token=token, local_dir=cfg.ckpt_dir)
                flat = os.path.join(cfg.ckpt_dir, os.path.basename(ckpts[-1]))
                if os.path.abspath(local) != os.path.abspath(flat):
                    shutil.copy(local, flat)
                print(f"[sample] pulled checkpoint {ckpts[-1]}")
    except Exception as e:
        print(f"[sample] HF auto-pull skipped ({e})")


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--preset", default="tiny", choices=["tiny", "full", "base2", "distill"],
                        help="which preset's checkpoint/tokenizer dir to use by default")
    parser.add_argument("--ckpt", default=None, help="path to a specific checkpoint .pt")
    parser.add_argument("--prompt", default="", help="text to continue (empty = free generation)")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help=">1 = more random, <1 = more focused, 0 = greedy")
    parser.add_argument("--top_k", type=int, default=200, help="sample only from the top-k tokens")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="nucleus sampling: keep the smallest set summing to >= top_p (0/1 = off)")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help=">1 discourages repeating tokens (1.0 = off)")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--device", default=None, help="cuda | cpu (auto if unset)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Use the preset only to locate the checkpoint dir + tokenizer path; the ACTUAL model
    # shape comes from the checkpoint itself.
    base_cfg = get_config(args.preset)
    # Fresh session? pull the checkpoint + tokenizer from the Hub so we don't hit
    # "tokenizer not found" / "no checkpoint" (set HF_TOKEN for a private repo).
    pull_from_hf(base_cfg, want_ckpt=(args.ckpt is None))
    ckpt_path = args.ckpt or latest_checkpoint(base_cfg.ckpt_dir)
    if ckpt_path is None:
        raise SystemExit(f"No checkpoint found in '{base_cfg.ckpt_dir}'. Train first.")

    print(f"[sample] loading {ckpt_path}")
    # mmap=True keeps the (multi-GB) checkpoint memory-mapped from disk instead of reading it
    # all into RAM. For generation we only touch the model weights, so the big optimizer state
    # is never paged in -> peak RAM stays ~= model size (matters on a laptop). map_location is
    # "cpu" here; the model itself was already moved to `device`, and load_state_dict copies
    # the weights onto it. Falls back to a normal load on older torch without mmap support.
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
    except (TypeError, RuntimeError):
        ckpt = torch.load(ckpt_path, map_location="cpu")

    # Rebuild the exact Config the model was trained with (falls back to the preset if an
    # older checkpoint didn't store one).
    cfg = Config(**ckpt["config"]) if "config" in ckpt else base_cfg

    # Rebuild + load the model.
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[sample] model {model.num_params()/1e6:.2f}M params, step {ckpt.get('step', '?')}")

    # Load the tokenizer used during training.
    tok = Tokenizer(cfg.tokenizer_path)

    # Encode the prompt.  An empty prompt starts from the end-of-text seed = "begin a story".
    if args.prompt:
        start_ids = [tok.eot_id] + tok.encode(args.prompt)
    else:
        start_ids = [tok.eot_id]
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)

    for s in range(args.num_samples):
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eot_id=tok.eot_id,
        )
        text = tok.decode(out[0].tolist())
        print(f"\n===== sample {s + 1}/{args.num_samples} " + "=" * 40)
        print(text.strip())


if __name__ == "__main__":
    main()
