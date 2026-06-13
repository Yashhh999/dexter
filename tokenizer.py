"""
tokenizer.py
============

A language model does not see text -- it sees integers ("token ids").  The tokenizer
is the dictionary that converts between the two:

    "Once upon a time"  --encode-->  [402, 89, 12, 771]   (what the model trains on)
    [402, 89, 12, 771]  --decode-->  "Once upon a time"   (what we read back)

We train a **Byte-Pair Encoding (BPE)** tokenizer FROM SCRATCH on TinyStories using
HuggingFace's fast `tokenizers` library.  BPE starts from raw bytes and repeatedly
merges the most frequent adjacent pair into a new token, until it reaches the target
vocabulary size.  Common words/word-pieces end up as single tokens; rare strings fall
back to smaller pieces or individual bytes.

Why **byte-level** BPE (the GPT-2 style)?  Because the base alphabet is the 256 possible
bytes, the tokenizer can represent *any* string (emoji, weird unicode, etc.) and can
never produce an "unknown token".  Robust and simple.

Why a small vocab (8000)?  TinyStories uses a tiny slice of English.  A small vocabulary
is more than enough, and it keeps the model's (vocab x n_embd) embedding/output matrix
small -- which matters when n_embd is 2048.

The trained tokenizer is saved to `tokenizer.json` and re-loaded thereafter, so the
(somewhat slow) training only ever happens ONCE.
"""

from __future__ import annotations

import os
from typing import Iterable, List

# The `tokenizers` library (Rust-backed, very fast).  These are its building blocks:
from tokenizers import Tokenizer as HFTokenizer        # the core object
from tokenizers import models, pre_tokenizers, decoders, trainers

from config import Config

# The single special token we need: a marker placed BETWEEN documents so the model learns
# where one story ends and the next begins.  "EOT" = end of text.  It doubles as the seed
# token we feed at generation time and as the natural place to stop generating.
END_OF_TEXT = "<|endoftext|>"


def train_tokenizer(cfg: Config) -> "Tokenizer":
    """
    Train a byte-level BPE tokenizer on TinyStories text and save it to cfg.tokenizer_path.

    If the file already exists we skip training and just load it -- so this is safe to call
    at the start of every run.
    """
    if os.path.exists(cfg.tokenizer_path):
        print(f"[tokenizer] found existing '{cfg.tokenizer_path}', loading (no retrain).")
        return Tokenizer(cfg.tokenizer_path)

    _src = "the dataset mix" if getattr(cfg, "dataset_mix", None) else cfg.dataset_name
    print(f"[tokenizer] training a {cfg.vocab_size}-token byte-level BPE on {_src} ...")

    # 1) The MODEL: an (initially empty) BPE.  unk_token=None because byte-level BPE never
    #    needs an unknown token -- every byte is representable.
    tok = HFTokenizer(models.BPE(unk_token=None))

    # 2) The PRE-TOKENIZER: ByteLevel maps the raw text into the byte alphabet and splits on
    #    a GPT-2-style regex (spaces become a visible 'Ġ' byte so word boundaries survive a
    #    round trip).  add_prefix_space=False = don't force a leading space.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 3) The DECODER: the exact inverse of the pre-tokenizer, so decode() reproduces the
    #    original spacing/bytes faithfully.
    tok.decoder = decoders.ByteLevel()

    # 4) The TRAINER: this is what actually runs the BPE merge algorithm.
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        special_tokens=[END_OF_TEXT],                 # reserve id(s) for our special token.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # seed all 256 byte symbols so
                                                      # even unseen bytes are always covered.
        show_progress=True,
    )

    # 5) Feed it text.  We pull from the SAME streaming iterator the dataset uses, capped to
    #    cfg.tokenizer_train_docs documents (training on the whole corpus is unnecessary --
    #    a representative sample gives essentially the same merges far faster).
    #    Imported here (not at top) to avoid a circular import: dataset.py imports us.
    from dataset import text_iter

    def doc_stream() -> Iterable[str]:
        for i, text in enumerate(text_iter(cfg)):
            if i >= cfg.tokenizer_train_docs:
                break
            yield text

    tok.train_from_iterator(doc_stream(), trainer=trainer)

    # 6) Save the whole thing (vocab + merges + pre/decoder config) to a single JSON file.
    tok.save(cfg.tokenizer_path)
    print(f"[tokenizer] done -> saved to '{cfg.tokenizer_path}' "
          f"(real vocab size = {tok.get_vocab_size()}).")

    return Tokenizer(cfg.tokenizer_path)


class Tokenizer:
    """
    A thin, friendly wrapper around the saved HuggingFace tokenizer.

    It exposes exactly what the rest of the project needs:
        .encode(text) -> list[int]
        .decode(ids)  -> str
        .eot_id       -> int id of the <|endoftext|> token
        .vocab_size   -> int
    """

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No tokenizer at '{path}'. Run train_tokenizer(cfg) (or train.py) first."
            )
        self._tok = HFTokenizer.from_file(path)
        self.eot_id = self._tok.token_to_id(END_OF_TEXT)
        self.vocab_size = self._tok.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        """Text -> list of token ids (no special tokens added automatically; we add EOT
        ourselves between documents in dataset.py)."""
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: List[int]) -> str:
        """List of token ids -> text.  We keep special tokens out of the readable output."""
        return self._tok.decode(ids, skip_special_tokens=True)


if __name__ == "__main__":
    # Quick manual test:  python tokenizer.py
    # Trains (or loads) the tiny-preset tokenizer and round-trips a sentence.
    cfg = Config()  # tiny defaults
    cfg.offline = True  # don't require internet for this little demo
    t = train_tokenizer(cfg)
    sample = "Once upon a time, a little dog ran in the park."
    ids = t.encode(sample)
    print("text :", sample)
    print("ids  :", ids)
    print("back :", t.decode(ids))
    print("eot  :", t.eot_id, "| vocab:", t.vocab_size)
