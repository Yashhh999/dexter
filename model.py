"""
model.py
========

The transformer itself: a decoder-only GPT, built from scratch, using the same modern
ingredients as Llama-style models.  Read this file top-to-bottom and you will have seen
every line of math in a 2024-era language model.

The pipeline for one forward pass:

    token ids (B, T)
        |  embedding lookup
        v
    hidden states (B, T, n_embd)
        |  repeated n_layer times:
        |      x = x + Attention(RMSNorm(x))     <- mixes information ACROSS positions
        |      x = x + SwiGLU_MLP(RMSNorm(x))    <- transforms EACH position independently
        v
    RMSNorm
        |  linear "lm_head" (weights TIED to the embedding table)
        v
    logits (B, T, vocab_size)   -> a probability distribution over the next token

Key modern choices and WHY:

  * RMSNorm instead of LayerNorm  -> fewer ops (no mean-subtraction, no bias), just as
    stable in practice.  Used by Llama/Mistral/etc.

  * RoPE (rotary) instead of learned position embeddings -> positions are injected by
    *rotating* the query/key vectors by an angle proportional to their position.  This
    encodes RELATIVE position directly into the attention dot-product and generalizes
    better.  There is no separate position-embedding table.

  * SwiGLU MLP instead of GELU MLP -> a gated activation (SiLU(gate) * up) that empirically
    learns better.  It uses THREE weight matrices instead of two, which is why the MLP
    dominates the parameter count.

  * scaled_dot_product_attention with is_causal=True -> PyTorch's fused attention kernel.
    It automatically uses FlashAttention / a memory-efficient kernel when available, so we
    get fast, low-memory causal attention for free without writing the masking ourselves.
    (On Kaggle's T4 / Turing GPUs the true FlashAttention-2 kernel isn't supported, but the
    fused *memory-efficient* backend is -- still a big win over a naive implementation.)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import Config


# =========================================================================================
# RMSNorm  --  Root-Mean-Square Layer Normalization
# =========================================================================================
class RMSNorm(nn.Module):
    """
    Normalize each token vector by its root-mean-square, then scale by a learned per-channel
    weight.  Unlike LayerNorm there is no mean subtraction and no bias term.

        rms(x) = sqrt(mean(x_i^2) + eps)
        out    = (x / rms(x)) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # starts as identity scaling.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability even when the model runs in float16,
        # then cast back to the input dtype.
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


# =========================================================================================
# RoPE  --  Rotary Positional Embeddings
# =========================================================================================
def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    """
    Precompute the cos/sin tables RoPE needs, for every position up to seq_len.

    Intuition: split each head's vector into pairs of numbers and treat each pair as a 2D
    point.  For position `p` we rotate the pair by an angle p * freq, where each pair has
    its own frequency.  Nearby positions get similar rotations; far-apart positions get
    very different ones -- so the attention dot product naturally encodes *relative* offset.

    Returns cos, sin each of shape (seq_len, head_dim).
    """
    # One frequency per pair of dimensions: high freq for early dims, low freq for later dims.
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()        # (seq_len,)
    freqs = torch.outer(positions, inv_freq)                        # (seq_len, head_dim/2)
    # Duplicate so the table lines up with the "rotate_half" layout used below.
    emb = torch.cat((freqs, freqs), dim=-1)                         # (seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper for RoPE: turn [a, b] (split in half along the last dim) into [-b, a]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """
    Rotate the query and key vectors by their position's angle.

    q, k    : (B, n_head, T, head_dim)
    cos,sin : (T, head_dim)  -> broadcast over batch and head dims.
    """
    cos = cos[None, None, :, :]   # (1, 1, T, head_dim)
    sin = sin[None, None, :, :]
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# =========================================================================================
# Causal self-attention
# =========================================================================================
class CausalSelfAttention(nn.Module):
    """
    Each token builds a Query, looks at the Keys of itself and all EARLIER tokens (causal),
    and pulls a weighted mix of their Values.  This is how information moves between
    positions.  RoPE is applied to Q and K before the dot product.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

        # One fused linear produces Q, K and V together (3 * n_embd outputs) -- cheaper than
        # three separate matmuls.  bias=False is the modern convention (norms handle shifts).
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        # Projects the attention output back to model width.
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q,K,V and split.  Each is (B, T, C).
        q, k, v = self.qkv(x).split(C, dim=2)

        # Reshape (B, T, C) -> (B, n_head, T, head_dim): give each head its own slice.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Inject position information by rotating Q and K.
        q, k = apply_rope(q, k, cos, sin)

        # The fused attention kernel.  is_causal=True applies the lower-triangular mask
        # (token t may only attend to <= t) WITHOUT us materializing a (T x T) mask, and
        # it scales by 1/sqrt(head_dim) internally.  dropout only active in training.
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # Merge the heads back together: (B, n_head, T, head_dim) -> (B, T, C).
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


# =========================================================================================
# SwiGLU feed-forward network
# =========================================================================================
class SwiGLU(nn.Module):
    """
    The per-position MLP.  "SwiGLU" = a SiLU-gated linear unit:

        out = W_down( SiLU(W_gate(x)) * W_up(x) )

    The `gate` branch (squashed by SiLU) acts as a soft, learned mask on the `up` branch.
    Three matrices total -> in the "full" preset (n_embd=2048, ffn_hidden=8192) this is the
    single biggest chunk of parameters in each block.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu(z) = z * sigmoid(z); the gate decides how much of `up` passes through.
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# =========================================================================================
# Transformer block
# =========================================================================================
class Block(nn.Module):
    """
    One transformer layer, in the modern PRE-NORM residual form:

        x = x + Attention(RMSNorm(x))     # communicate across positions
        x = x + SwiGLU(RMSNorm(x))        # think at each position

    Pre-norm (normalize the *input* of each sub-layer, then add the result back) keeps the
    residual "highway" clean and makes deep stacks train stably.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


# =========================================================================================
# The full GPT model
# =========================================================================================
class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Token embedding table: row `i` is the starting vector for token id `i`.
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        # The stack of transformer blocks.
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd, cfg.norm_eps)  # final norm before the output head.
        # The output "head": projects hidden states to a score per vocabulary token.
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # WEIGHT TYING: share the SAME matrix for input embedding and output projection.
        # Saves vocab_size * n_embd parameters and is known to improve quality.
        self.lm_head.weight = self.tok_emb.weight

        # RoPE tables are not learned; precompute once and store as (non-persistent) buffers.
        cos, sin = build_rope_cache(
            cfg.block_size, cfg.n_embd // cfg.n_head, cfg.rope_theta,
            device="cpu", dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Initialize all weights (see _init_weights for the scheme + reasoning).
        self.apply(self._init_weights)
        # Special scaled init for the residual "output" projections: shrink them by
        # 1/sqrt(2 * n_layer) so the residual stream doesn't blow up as depth grows (GPT-2).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w_down.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        # Small Gaussian init -- the standard GPT recipe.
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        """Total trainable parameters.  Because the head is tied to the embedding, counting
        all parameters does NOT double-count the vocabulary matrix."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        idx     : (B, T) int64 token ids.
        targets : (B, T) int64 next-token labels, or None.

        Returns:
          * if targets given      -> a scalar cross-entropy loss  (what we train on).
          * if targets is None    -> logits for the LAST position only, shape (B, 1, vocab)
                                     (all `generate` ever needs -- saves a huge matmul).

        Returning ONLY the loss during training is deliberate: under nn.DataParallel the
        forward output is gathered onto GPU 0, and gathering a scalar loss is far cheaper
        than gathering a giant (B, T, vocab) logits tensor.
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"

        # Slice the precomputed RoPE tables to the current sequence length, on the right device.
        cos = self.rope_cos[:T].to(idx.device)
        sin = self.rope_sin[:T].to(idx.device)

        x = self.drop(self.tok_emb(idx))  # (B, T, n_embd)

        for block in self.blocks:
            if self.cfg.use_grad_checkpoint and self.training:
                # Don't store this block's activations; recompute them in the backward pass.
                # Trades compute for a large activation-memory saving (needed for 1.6B on T4).
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)

        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)                       # (B, T, vocab)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),          # (B*T, vocab)
                targets.view(-1),                          # (B*T,)
                ignore_index=-1,                           # -1 labels (padding) are skipped.
            )
            return loss

        # Inference path: we only care about the prediction at the final position.
        logits = self.lm_head(x[:, [-1], :])               # (B, 1, vocab)
        return logits

    # -------------------------------------------------------------------------------------
    def configure_optimizers(self, cfg: Config, device_type: str):
        """
        Build the optimizer with two parameter groups:

          * decay group    : 2D matrices (all the Linear / embedding weights) get weight decay.
          * no-decay group : 1D tensors (RMSNorm weights) do NOT -- decaying them just biases
                             the normalization and hurts.

        If cfg.use_8bit_adam and we're on CUDA with bitsandbytes installed, we use an 8-bit
        AdamW: it keeps Adam's two running-statistics tensors in 1 byte each instead of 4,
        roughly quartering optimizer memory -- the difference between fitting the 1.6B model
        on a 16 GB T4 or not.
        """
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        if cfg.use_8bit_adam and device_type == "cuda":
            try:
                import bitsandbytes as bnb
                if getattr(cfg, "use_paged_adam", False):
                    # PAGED: optimizer state lives in CPU RAM and is paged onto the GPU only
                    # for each update -> lowest GPU memory, but SLOW (PCIe transfer every step).
                    # Only worth it if plain 8-bit AdamW still OOMs.
                    opt = bnb.optim.PagedAdamW8bit(groups, lr=cfg.lr,
                                                   betas=(cfg.beta1, cfg.beta2))
                    print("[optim] using bitsandbytes PagedAdamW8bit (state paged to CPU; slower).")
                else:
                    # NON-PAGED: 8-bit optimizer state stays on the GPU (1 byte per moment).
                    # ~4x less optimizer memory than fp32 AdamW, and MUCH faster than paging.
                    # This fits comfortably now that the model is ~0.9B.
                    opt = bnb.optim.AdamW8bit(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
                    print("[optim] using bitsandbytes 8-bit AdamW (state on GPU; fast).")
                return opt
            except Exception as e:  # bitsandbytes missing or no GPU support -> fall back.
                print(f"[optim] 8-bit AdamW unavailable ({e}); falling back to torch AdamW.")

        # Use the fused AdamW kernel on CUDA when available (a bit faster); plain otherwise.
        use_fused = (device_type == "cuda")
        try:
            opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                                    fused=use_fused)
        except TypeError:
            opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
        print("[optim] using torch AdamW.")
        return opt

    # -------------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 eot_id: Optional[int] = None) -> torch.Tensor:
        """
        Autoregressively extend `idx` (B, T) by sampling one token at a time.

          temperature : >1 = more random/creative, <1 = more greedy/repetitive, 0-ish = argmax.
          top_k       : if set, sample only from the k most likely tokens (cuts off the
                       long tail of nonsense).
          eot_id      : if all sequences emit this token, stop early.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Never feed more than block_size tokens of context to the model.
            idx_cond = idx[:, -self.cfg.block_size:]
            logits = self(idx_cond)              # (B, 1, vocab) from the inference path above.
            logits = logits[:, -1, :]            # (B, vocab): scores for the next token.

            if temperature <= 0:
                # Greedy: always take the single most likely token.
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k)
                    # Mask everything below the k-th best down to -inf so it can't be sampled.
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat((idx, idx_next), dim=1)

            if eot_id is not None and (idx_next == eot_id).all():
                break
        return idx


if __name__ == "__main__":
    # Smoke test the architecture with random inputs (no data needed):  python model.py
    from config import get_config

    # Count the FULL (~1.6B) model on the "meta" device: shapes are tracked but no real
    # memory is allocated, so we can verify the parameter count without needing ~6.5 GB RAM.
    with torch.device("meta"):
        mfull = GPT(get_config("full"))
    print(f"[full] real param count: {mfull.num_params()/1e6:8.2f}M "
          f"(non-embedding {mfull.num_params(non_embedding=True)/1e6:8.2f}M)")

    # Actually build + run the TINY model to exercise the forward/loss path for real.
    cfg = get_config("tiny")
    m = GPT(cfg)
    print(f"[tiny] real param count: {m.num_params()/1e6:8.2f}M "
          f"(non-embedding {m.num_params(non_embedding=True)/1e6:8.2f}M)")
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    loss = m(x, x)  # use inputs as their own targets just to exercise the loss path.
    print(f"[tiny] forward OK, loss = {loss.item():.4f}")
