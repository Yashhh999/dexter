"""
dataset.py
==========

Turns raw TinyStories text into something a GPU can train on FAST, using the "nanoGPT"
data pattern:

  1. STREAM the dataset from HuggingFace so we never download/store the whole thing on disk.
  2. TOKENIZE every document once and append a single contiguous stream of token ids to a
     flat binary file (train.bin / val.bin) of 16-bit integers.
  3. At training time, MEMORY-MAP that file and grab random windows from it.  This is about
     as fast as data loading gets: no per-step tokenization, no Python dataset objects,
     just a memmap + a couple of slices.

Why uint16?  Our vocabulary is <= 8000 tokens, which fits comfortably in a 16-bit integer
(0..65535).  That halves the file size (and disk I/O) versus int32.

Documents are separated by the tokenizer's <|endoftext|> id so the model learns where one
story stops and the next starts.

If there is no internet (or cfg.offline=True) we fall back to a small built-in synthetic
"stories" generator, so the entire pipeline still runs end-to-end for learning/testing.
"""

from __future__ import annotations

import os
import random
from typing import Iterator, Optional

import numpy as np
import torch

from config import Config
from tokenizer import train_tokenizer


# =========================================================================================
# Text source: real (streamed) TinyStories, with an offline synthetic fallback
# =========================================================================================
def _synthetic_texts() -> Iterator[str]:
    """
    A tiny procedural generator of TinyStories-flavored sentences.  It is intentionally
    simple and repetitive -- just enough vocabulary and structure for the pipeline to run
    and for a small model to show *visible* learning (it will start reproducing these
    patterns).  Used only when the real dataset can't be reached.
    """
    names = ["Tom", "Lily", "Sam", "Mia", "Ben", "Anna", "Max", "Sara", "Leo", "Zoe"]
    animals = ["dog", "cat", "bird", "fish", "frog", "bunny", "duck", "bear", "fox", "owl"]
    places = ["park", "house", "garden", "forest", "beach", "river", "school", "farm"]
    adjs = ["big", "small", "happy", "sad", "red", "blue", "soft", "shiny", "funny", "kind"]
    verbs = ["ran", "played", "jumped", "looked", "smiled", "walked", "found", "helped"]
    rng = random.Random(1234)  # fixed seed -> reproducible "corpus".
    while True:  # infinite stream; callers cap how much they consume.
        name = rng.choice(names)
        animal = rng.choice(animals)
        place = rng.choice(places)
        adj = rng.choice(adjs)
        verb = rng.choice(verbs)
        story = (
            f"Once upon a time, there was a {adj} {animal}. "
            f"{name} and the {animal} {verb} in the {place}. "
            f"The {animal} was very {rng.choice(adjs)} and {name} was happy. "
            f"They {rng.choice(verbs)} all day and then went home to sleep."
        )
        yield story


def text_iter(cfg: Config) -> Iterator[str]:
    """
    Yield raw document strings from the TinyStories TRAIN corpus, one at a time.

    prepare_data() below is what performs the 95/5 train/val split over this single stream,
    so we only need one source here.
    """
    if cfg.offline:
        print("[data] offline=True -> using built-in synthetic corpus.")
        yield from _synthetic_texts()
        return

    try:
        from datasets import load_dataset
        # streaming=True returns an IterableDataset: examples are fetched lazily over the
        # network, so nothing is written to disk and memory stays flat.
        ds = load_dataset(cfg.dataset_name, split="train", streaming=True)
        for example in ds:
            text = example.get("text", "")
            if text:
                yield text
    except Exception as e:
        # No internet / datasets not installed / dataset moved -> degrade gracefully.
        print(f"[data] could not stream '{cfg.dataset_name}' ({e}); "
              f"using synthetic corpus instead.")
        yield from _synthetic_texts()


# =========================================================================================
# Pre-tokenization:  text stream  ->  train.bin / val.bin
# =========================================================================================
def _bin_paths(cfg: Config):
    return (os.path.join(cfg.data_dir, "train.bin"),
            os.path.join(cfg.data_dir, "val.bin"))


def prepare_data(cfg: Config) -> None:
    """
    Tokenize the corpus once and write train.bin + val.bin.  Safe to call every run: if both
    files already exist we return immediately.

    Split rule: every 20th document goes to validation (=> a deterministic 95/5 split).
    Caps (cfg.max_train_tokens / cfg.max_val_tokens) let local runs finish quickly; set them
    to None (the "full" preset) to consume the entire corpus.
    """
    train_path, val_path = _bin_paths(cfg)
    if os.path.exists(train_path) and os.path.exists(val_path):
        tt = os.path.getsize(train_path) // 2  # 2 bytes per uint16 token.
        vt = os.path.getsize(val_path) // 2
        print(f"[data] found existing .bin files (train={tt:,} tok, val={vt:,} tok); skipping.")
        return

    os.makedirs(cfg.data_dir, exist_ok=True)

    # We need the tokenizer first (train-once-or-load happens inside train_tokenizer).
    tok = train_tokenizer(cfg)
    eot = tok.eot_id

    train_cap = cfg.max_train_tokens if cfg.max_train_tokens is not None else float("inf")
    val_cap = cfg.max_val_tokens if cfg.max_val_tokens is not None else float("inf")

    print(f"[data] tokenizing -> {train_path} / {val_path} "
          f"(caps: train={train_cap}, val={val_cap}) ...")

    # Open both output files and stream tokens into them in buffered chunks (one big write
    # per buffer is far faster than millions of tiny writes).
    train_buf, val_buf = [], []
    train_count, val_count = 0, 0
    FLUSH_EVERY = 1_000_000  # flush a buffer to disk once it holds ~1M tokens.

    def flush(buf, f):
        if buf:
            np.array(buf, dtype=np.uint16).tofile(f)
            buf.clear()

    with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
        for doc_index, text in enumerate(text_iter(cfg)):
            ids = tok.encode(text)
            ids.append(eot)  # mark end-of-document.

            is_val = (doc_index % 20 == 0)  # 1 in 20 docs -> validation (5%).

            if is_val:
                if val_count < val_cap:
                    val_buf.extend(ids)
                    val_count += len(ids)
                    if len(val_buf) >= FLUSH_EVERY:
                        flush(val_buf, f_val)
            else:
                if train_count < train_cap:
                    train_buf.extend(ids)
                    train_count += len(ids)
                    if len(train_buf) >= FLUSH_EVERY:
                        flush(train_buf, f_train)

            # Stop once BOTH splits have hit their caps.
            if train_count >= train_cap and val_count >= val_cap:
                break

            if doc_index % 50_000 == 0 and doc_index > 0:
                print(f"[data]   {doc_index:,} docs | train={train_count:,} tok | "
                      f"val={val_count:,} tok")

        flush(train_buf, f_train)
        flush(val_buf, f_val)

    print(f"[data] done. train={train_count:,} tokens, val={val_count:,} tokens.")


# =========================================================================================
# Batch sampling for the training loop
# =========================================================================================
def get_batch(split: str, cfg: Config, device: str, batch_size: Optional[int] = None):
    """
    Return one training batch (x, y), each shaped (batch_size, block_size), int64.

      x = a random window of `block_size` consecutive tokens.
      y = the same window shifted right by one  -> the "next token" label for every position.

    `batch_size` defaults to cfg.batch_size.  train.py passes a LARGER value
    (cfg.batch_size * num_gpus) so nn.DataParallel can split it back into cfg.batch_size
    sequences per GPU -- that's why cfg.batch_size is documented as the *per-GPU* micro-batch.

    We re-open the memmap every call.  np.memmap doesn't load the file into RAM -- the OS
    pages in only the slices we actually touch -- so this stays cheap even for huge .bin
    files, and recreating it each call avoids a known memmap memory leak.
    """
    bs = batch_size if batch_size is not None else cfg.batch_size
    train_path, val_path = _bin_paths(cfg)
    path = train_path if split == "train" else val_path
    data = np.memmap(path, dtype=np.uint16, mode="r")

    # Pick `bs` random start positions such that a full window fits.
    max_start = len(data) - cfg.block_size - 1
    assert max_start > 0, (
        f"{path} has only {len(data)} tokens, too few for block_size={cfg.block_size}. "
        f"Increase the token caps in config.py."
    )
    ix = torch.randint(max_start, (bs,))

    # Slice out the windows and convert to int64 tensors (the dtype embeddings expect).
    x = torch.stack([torch.from_numpy(data[i:i + cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + cfg.block_size].astype(np.int64)) for i in ix])

    if device.startswith("cuda"):
        # pin_memory + non_blocking lets the host->GPU copy overlap with compute.
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


if __name__ == "__main__":
    # Prepare the tiny-preset data offline and show one batch:  python dataset.py
    cfg = Config()
    cfg.offline = True
    prepare_data(cfg)
    xb, yb = get_batch("train", cfg, "cpu")
    print("x:", xb.shape, "y:", yb.shape)
    print("x[0,:12]:", xb[0, :12].tolist())
