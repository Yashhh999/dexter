"""
chat.py
=======

Talk to the SFT model (v0.4) WITHOUT hand-typing the prompt scaffolding.  It wraps whatever
you type in the exact `### Instruction / ### Response` template the model was fine-tuned on
(see dataset.py), generates, and prints just the response.

    python chat.py                              # latest checkpoint in checkpoints_sft/
    python chat.py --ckpt checkpoints_sft/ckpt_002000.pt
    python chat.py --temperature 0.5            # more focused answers

Type a message and press enter; type 'exit' / 'quit' (or Ctrl-D) to leave.  Each turn is
INDEPENDENT (no conversation memory) -- that matches the single-turn SFT data; multi-turn
chat would need history-threading we deliberately keep out of this v1.

Honest expectation: this is a 0.5B model.  It follows instructions and answers in the right
*shape*; facts and math will often be wrong (the scale ceiling).  Read it as "can it follow
the format and stay on topic", not "is it correct".
"""

from __future__ import annotations

import argparse
import os

import torch

from config import Config, get_config
from model import GPT
from tokenizer import Tokenizer
# Reuse the exact template the model trained on (keeps prompt == training distribution) and the
# checkpoint/HF helpers, so chat.py never drifts from dataset.py / sample.py.
from dataset import SFT_TEMPLATE
from sample import latest_checkpoint, pull_from_hf


def build_prompt(user_message: str) -> str:
    """Wrap the user's message in the training template, leaving the response empty for the
    model to complete: '### Instruction:\\n<msg>\\n\\n### Response:\\n'."""
    return SFT_TEMPLATE.format(instr=user_message.strip(), resp="")


def main():
    parser = argparse.ArgumentParser(description="Chat with the SFT (v0.4) model.")
    parser.add_argument("--preset", default="sft",
                        choices=["tiny", "full", "base2", "distill", "sft"],
                        help="which preset's checkpoint/tokenizer dir to use (default: sft)")
    parser.add_argument("--ckpt", default=None, help="path to a specific checkpoint .pt")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7,
                        help=">1 = more random, <1 = more focused, 0 = greedy")
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--device", default=None, help="cuda | cpu (auto if unset)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Locate + load the checkpoint (pull from HF on a fresh box, exactly like sample.py).
    base_cfg = get_config(args.preset)
    pull_from_hf(base_cfg, want_ckpt=(args.ckpt is None))
    ckpt_path = args.ckpt or latest_checkpoint(base_cfg.ckpt_dir)
    if ckpt_path is None:
        raise SystemExit(f"No checkpoint found in '{base_cfg.ckpt_dir}'. Train the SFT model first "
                         f"(python train.py --preset sft --init_from <base ckpt>).")

    print(f"[chat] loading {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
    except (TypeError, RuntimeError):
        ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg = Config(**ckpt["config"]) if "config" in ckpt else base_cfg
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = Tokenizer(cfg.tokenizer_path)
    print(f"[chat] {model.num_params()/1e6:.2f}M params, step {ckpt.get('step', '?')} | "
          f"device={device}. Type 'exit' to quit.\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            break

        # Seed with <eot> (= "start a fresh document", matching how every SFT example was framed
        # after the previous document's trailing <eot>), then the templated instruction.
        start_ids = [tok.eot_id] + tok.encode(build_prompt(user))
        idx = torch.tensor([start_ids], dtype=torch.long, device=device)
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eot_id=tok.eot_id,
        )

        # Decode ONLY the newly generated tokens (everything after the prompt) -> just the answer.
        gen_ids = out[0].tolist()[len(start_ids):]
        reply = tok.decode(gen_ids).strip()
        # If the model ran on and started a new instruction, cut it off at that boundary.
        for marker in ("### Instruction", "### Response"):
            if marker in reply:
                reply = reply.split(marker)[0].strip()
        print(f"dexter> {reply}\n")


if __name__ == "__main__":
    main()
