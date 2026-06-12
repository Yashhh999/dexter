"""
distill_generate.py
====================

Generate a high-quality synthetic training corpus for Dexter using a big "teacher" model
through an OpenAI-compatible API (Groq or OpenRouter -- both have free tiers).

This is **sequence-level knowledge distillation**, a.k.a. the "Textbooks Are All You Need" /
Cosmopedia recipe: a 70B-120B teacher writes clean, dense, self-contained documents, and we
later train the small Dexter *student* on them.  The student will NOT match the teacher (it's
~0.5B vs 120B), but learning from teacher-written data beats raw web text token-for-token --
that's the "more capability per training token" you were after.

Why this is genuinely yours: YOU generate the data (your prompts, your curation), and you
train the weights from scratch on it.  The model is your work; just credit the teacher in the
model card (see the licensing note at the bottom).

------------------------------------------------------------------------------------------
USAGE
  export GROQ_API_KEY=...                       # or OPENROUTER_API_KEY for --provider openrouter
  python distill_generate.py --provider groq --model openai/gpt-oss-120b --target_docs 2000

  # resumes automatically (appends to the same .jsonl); rate-limit aware; Ctrl-C safe.
  # test the whole loop without an API key / network:
  python distill_generate.py --dry_run --target_docs 20
------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

# Both Groq and OpenRouter speak the OpenAI "chat/completions" protocol, so one client works
# for both -- we only swap the base URL and which env var holds the key.
PROVIDERS = {
    "groq":       ("https://api.groq.com/openai/v1/chat/completions",  "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions",    "OPENROUTER_API_KEY"),
}

# -----------------------------------------------------------------------------------------
# Diversity pools.  The single biggest driver of good *pretraining* data is VARIETY, so we
# randomize the topic, the audience, the style and the format on every call (the Cosmopedia
# trick).  A few dozen of each gives tens of thousands of unique combinations.
# -----------------------------------------------------------------------------------------
TOPICS = [
    "how rainbows form", "why the sky is blue", "the water cycle", "how plants make food",
    "the life of a honeybee", "how computers store numbers", "what makes a bridge strong",
    "why ice floats", "how the heart pumps blood", "the phases of the moon", "how magnets work",
    "what causes the seasons", "how sound travels", "why we dream", "how a seed becomes a tree",
    "the basics of fractions", "how to compare two numbers", "what a prime number is",
    "how volcanoes erupt", "why leaves change color", "how birds fly", "the parts of a story",
    "how to make a simple budget", "what gravity does", "how a battery works", "the food chain",
    "how rain is measured", "what an ecosystem is", "how to solve a simple word problem",
    "why we need sleep", "how a rocket lifts off", "what friction is", "how shadows are made",
    "the idea of cause and effect", "how to write a clear paragraph", "what a fraction of a pizza means",
    "how bees help flowers", "why exercise is good for you", "how a thermometer works",
    "the difference between weather and climate",
]
AUDIENCES = [
    "a curious 8-year-old", "a 12-year-old student", "a high-school beginner",
    "a complete beginner adult", "a hobbyist who likes clear explanations",
]
STYLES = [
    "a clear, self-contained textbook passage", "an engaging explainer with a real example",
    "a short story that gently teaches the idea", "a friendly step-by-step walkthrough",
    "a question-and-answer that thinks out loud",
]

# Each recipe is a (system, user-template) pair.  Templates are filled from the pools above.
RECIPES = {
    "textbook": (
        "You are an expert educator. You write clear, accurate, self-contained explanations "
        "in simple language. No preamble, no meta-commentary -- output only the passage.",
        "Write {style} for {audience} about: {topic}. Be concrete, include one simple example, "
        "and explain the reasoning step by step. About 250-450 words.",
    ),
    "qa_reason": (
        "You write question-and-answer pairs that demonstrate careful, step-by-step reasoning. "
        "Output exactly in the form 'Question: ...' then 'Answer: ...'.",
        "Pose a thoughtful question about {topic} suitable for {audience}, then answer it by "
        "reasoning step by step and explaining WHY each step follows. About 200-350 words.",
    ),
    "story": (
        "You are a skilled children's author. Output only the story -- no title, no notes.",
        "Write a short, vivid, COMPLETE story (about 250 words) that naturally involves "
        "{topic}. It should make sense from beginning to end and be suitable for {audience}.",
    ),
    "howto": (
        "You write practical, numbered how-to guides in plain language. Output only the guide.",
        "Write a simple step-by-step guide for {audience} explaining {topic}. Number the steps "
        "and end with one short 'why this works' sentence. About 200-350 words.",
    ),
    "dialogue": (
        "You write short, informative dialogues where a curious learner asks and a patient "
        "teacher explains. Output only the dialogue.",
        "Write a short dialogue where {audience} asks about {topic} and a teacher explains it "
        "clearly with an example. About 200-350 words.",
    ),
}


def build_messages(recipe_name: str):
    """Pick random pool values and render this recipe's system+user messages."""
    system, user_tmpl = RECIPES[recipe_name]
    user = user_tmpl.format(
        topic=random.choice(TOPICS),
        audience=random.choice(AUDIENCES),
        style=random.choice(STYLES),
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_teacher(url, model, key, messages, max_tokens, timeout=120):
    """One chat completion with retry/backoff on rate-limit (429) and transient 5xx errors.
    Returns the generated text, or None if it keeps failing."""
    import requests  # imported here so --dry_run works with no extra deps installed
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.9,
               "top_p": 0.95, "max_tokens": max_tokens}
    for attempt in range(6):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:
            wait = min(60, 2 ** attempt)
            print(f"[gen] network error ({e}); retry in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        if r.status_code in (429, 500, 502, 503, 529):
            # Respect the server's Retry-After if it sent one, else exponential backoff.
            wait = int(r.headers.get("retry-after", min(60, 2 ** attempt)))
            print(f"[gen] HTTP {r.status_code} (rate limit / busy); backing off {wait}s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"API error {r.status_code}: {r.text[:300]}")
    return None


def count_existing(path: str) -> int:
    """How many documents are already in the output file (so we can resume / append)."""
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    p = argparse.ArgumentParser(description="Generate distilled training data from a teacher LLM.")
    p.add_argument("--provider", default="groq", choices=list(PROVIDERS))
    p.add_argument("--model", default="openai/gpt-oss-120b",
                   help="teacher model id (Apache-licensed teachers like gpt-oss-* / qwen3-* "
                        "are cleanest to train on -- see the licensing note in this file)")
    p.add_argument("--out", default="data_distill/corpus.jsonl")
    p.add_argument("--target_docs", type=int, default=2000, help="stop once the file has this many docs")
    p.add_argument("--recipes", default=",".join(RECIPES),
                   help="comma-separated subset of: " + ",".join(RECIPES))
    p.add_argument("--max_tokens", type=int, default=900)
    p.add_argument("--sleep", type=float, default=1.0, help="seconds between calls (free-tier friendly)")
    p.add_argument("--dry_run", action="store_true", help="fabricate text instead of calling the API")
    args = p.parse_args()

    recipes = [r.strip() for r in args.recipes.split(",") if r.strip() in RECIPES]
    assert recipes, "no valid recipes selected"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    url, key_env = PROVIDERS[args.provider]
    key = os.environ.get(key_env)
    if not args.dry_run and not key:
        raise SystemExit(f"Set {key_env} (your {args.provider} API key) -- or use --dry_run to test.")

    have = count_existing(args.out)
    print(f"[distill] out={args.out} | already have {have} docs | target {args.target_docs} | "
          f"teacher={args.model} ({args.provider})")

    written, total_chars = 0, 0
    t0 = time.time()
    # Open in append mode so re-running resumes (and a Ctrl-C never loses what's on disk).
    with open(args.out, "a", encoding="utf-8") as f:
        while have + written < args.target_docs:
            recipe = random.choice(recipes)
            messages = build_messages(recipe)

            if args.dry_run:
                text = (f"[dry-run/{recipe}] " + messages[1]["content"] + " ") * 8
            else:
                text = call_teacher(url, args.model, key, messages, args.max_tokens)
                if not text:
                    print("[distill] a call failed after retries; pausing 30s then continuing.")
                    time.sleep(30)
                    continue

            f.write(json.dumps({"text": text, "recipe": recipe}, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            total_chars += len(text)

            if written % 20 == 0 or args.dry_run:
                rate = written / max(1e-6, time.time() - t0)
                print(f"[distill] {have + written}/{args.target_docs} docs | "
                      f"~{total_chars // max(1, written)} chars/doc | {rate:.2f} docs/s")
            if not args.dry_run:
                time.sleep(args.sleep)

    approx_tokens = total_chars // 4   # rough chars->tokens
    print(f"[distill] done. wrote {written} new docs (~{approx_tokens:,} new tokens) to {args.out}")
    print("[distill] next: train Dexter on it ->  python train.py --preset distill")


if __name__ == "__main__":
    main()


# =========================================================================================
# LICENSING NOTE (read before publishing a model trained on this data)
# -----------------------------------------------------------------------------------------
# You are training on a teacher's OUTPUTS, so the teacher's license/terms matter:
#   * gpt-oss-20b / gpt-oss-120b  -> Apache-2.0  -> outputs are clean to train on. (good default)
#   * qwen3-32b                   -> Apache-2.0  -> clean.
#   * llama-3.x                   -> Llama Community License has clauses about using outputs to
#                                    train other models; check it before using Llama as the teacher.
# Also respect the API provider's Terms of Service (Groq / OpenRouter).
# Credit the teacher in your model card, e.g. "training data distilled from gpt-oss-120b".
# =========================================================================================
