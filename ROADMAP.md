# Dexter — Roadmap (v0.1 → v1.0)

A from-scratch language model, built and trained by us. The goal for **v1.0** is a small but
**well-rounded** model: 🧠 logically capable, 🎨 creative, and 🔀 versatile.

> **Honest framing:** Dexter is trained from scratch on free/cheap GPUs, so v1.0 targets a
> *capable small model* (SmolLM / Phi-small class) — basic, sometimes-shaky reasoning, not a
> frontier model like GPT-4. Great **data** maximizes what small scale gives us; it doesn't
> break the scaling ceiling. Every version below pulls **one new lever** so each release is a
> visible jump.

---

## Current state

- **v0.1** — from-scratch ~0.9B GPT (24 layers, dim 1536, RoPE + RMSNorm + SwiGLU, 1024 ctx),
  trained on **TinyStories**. Writes fluent, coherent simple English. No real logic yet; narrow.

---

## The plan

| Version | The one lever (what we do / feed) | Builds |
|---|---|---|
| **v0.1** *(now)* | From-scratch base on simple stories → fluent English. | 🎨 seed |
| **v0.2** | **Converge + control:** finish training, add top-p + repetition-penalty + KV-cache decoding, light instruction-tuning → polished, follows prompts. | 🎨 🔀 |
| **v0.3** | **Broaden the brain (NEW bigger base):** bigger tokenizer (~16–32k), retrain from scratch on a *reasoning-dense* mix — educational/textbook + code + stories. | 🧠 🔀 **(big leap)** |
| **v0.4** | **Teach it to reason:** fine-tune on step-by-step solutions, math-with-working, chain-of-thought, varied instructions → it *explains* its logic. | 🧠 |
| **v0.5** | **Widen creativity + tasks:** add genres, dialogue, Q&A, summarization to the mix; scale params/context if compute allows. | 🎨 🔀 |
| **v0.6 – v0.9** | **Align + iterate:** preference tuning (pick better outputs / DPO), grow the eval suite, fix weak spots, scale with budget. | all three, sharper |
| **v1.0** | **Well-rounded small model:** basic multi-step logic + varied creative writing + follows many instructions. Evaluated, model card, live demo. | 🧠 🎨 🔀 ✅ |

---

## Which versions need a *retrain from scratch*?

The rule: **retrain from scratch only when the tokenizer/vocabulary or architecture changes.**
Otherwise we keep the existing weights and continue/fine-tune (cheap).

| Version | Method | Why |
|---|---|---|
| v0.1 | **from scratch** | the original base (done) |
| v0.2 | continue + fine-tune | same tokenizer/arch → reuse weights |
| **v0.3** | **from scratch (retrain)** | **bigger tokenizer + new data domain → old weights can't transfer** |
| v0.4 | fine-tune | reuse v0.3 weights |
| v0.5 | continue-pretrain (or fresh if vocab grows again) | usually same tokenizer |
| v0.6 – v0.9 | fine-tune / DPO | reuse weights |
| v1.0 | polish | reuse weights |

➡️ **v0.3 is the one expensive "start over" step** (new vocab = old weights are meaningless).
It's also the biggest capability jump, so it's worth it — but budget the GPU hours/time for it.
Keep the **v0.1 tokenizer + weights archived** before v0.3 so v0.1/v0.2 stay reproducible.

---

## How the three goals get built

- **🧠 Logic** = v0.3 (reasoning-dense *data*) + v0.4 (reasoning *SFT*). Logic isn't a switch —
  it's fed in via data that contains step-by-step reasoning (textbooks, code, worked problems).
- **🎨 Creative** = v0.1 seeds it (stories), v0.5 widens it (genres/dialogue), and good decoding
  (v0.2: top-p + repetition penalty) makes it *feel* creative.
- **🔀 Versatile** = instruction-tuning (v0.2/v0.4) + a broad data mix (v0.3/v0.5) → the model
  learns many task types, not one.

---

## Eval discipline (start at v0.2, never skip)

Build a fixed evaluation set early and run it on **every** version:

- **~15 fixed prompts** covering each goal — at least one **logic** prompt, one **creative**
  prompt, one **"follow this instruction"** prompt.
- **Held-out perplexity** (one comparable number).
- Save outputs **side-by-side per version**.

This is how we *prove* each release is better instead of guessing — and it catches regressions.

---

## Data ingredients (for v0.3+, mix — don't feed sequentially)

Reasoning/knowledge foundation comes from *reasoning-dense, educational* data:

- educational/textbook-style synthetic data (clean, "how/why" explanations) — the core
- high-quality educational web text — knowledge + light reasoning
- a **code** slice — excellent for logic/structure
- some **stories** — keeps creativity alive
- *(for the SFT phase)* math-with-solutions + chain-of-thought traces → explicit reasoning

> Cite every dataset's license in the model card. Using public datasets keeps Dexter
> **100% ours** (we train the weights from scratch) — unlike fine-tuning someone else's model.

---

## Guiding principle

> **One lever per release.** Better decoding → reasoning data → reasoning SFT → diversity →
> alignment. One change at a time means every update feels big *and* we always know **why** it
> got better.

**Immediate next step:** finish **v0.1**, then ship **v0.2** (decoding polish + instruction
control) — cheap, fast, and it makes Dexter feel real. v0.3 is where 🧠 logic genuinely enters.
