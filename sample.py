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
