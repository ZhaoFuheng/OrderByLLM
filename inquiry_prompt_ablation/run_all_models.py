#!/usr/bin/env python3
"""Run the inquiry-prompt ablation (V5b prompt, combined NQ + SituatedQA benchmark) on
all three models -- GPT-5-mini, Haiku-4.5, and Llama-3.1-70B -- and print one combined
table (+ LaTeX). Writes results_<model>.json for each model.

This is a thin driver over run_inquiry_ablation.py; it just fixes the model list.

Usage:
    python inquiry_prompt_ablation/run_all_models.py
    python inquiry_prompt_ablation/run_all_models.py --concurrency 8
"""
import argparse
import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_inquiry_ablation as A  # noqa: E402

# (model short-name, provider) -- llama3.1-70b is served via Fireworks (see
# order_by.clients.FIREWORKS_MODEL_MAP); swap to "hf" to use the HuggingFace router.
MODELS = [
    ("openai-gpt-5-mini", "openai"),
    ("claude-haiku-4-5", "anthropic"),
    ("llama3.1-70b", "fireworks"),
]


async def main_async(concurrency):
    A.load_env()
    data = A._load(str(A.DEFAULT_DATA))
    npar = sum(1 for d in data if d["gold"] == "parametric")
    nnon = sum(1 for d in data if d["gold"] == "non_parametric")
    print(f"Inquiry-prompt ablation [V5b] on {len(data)} questions "
          f"({npar} parametric NQ / {nnon} non_parametric SituatedQA)\n")

    rows = []
    for model, provider in MODELS:
        try:
            m, n_err = await A.run_model(model, provider, data, concurrency)
        except Exception as e:
            print(f"[{model}] SKIPPED -- provider error: {type(e).__name__}: {e}")
            continue
        if m["n"] == 0:   # every call errored -> model unreachable (e.g. deployment down)
            print(f"[{model}] SKIPPED -- all {n_err} calls failed (model not deployed/reachable)")
            continue
        rows.append((model, m))
        print(f"[{model}] acc={m['accuracy']*100:.1f}%"
              + (f"   ({n_err} classification errors, excluded)" if n_err else ""))

    if rows:
        A.print_table(rows)
    else:
        print("\nNo model completed successfully.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    asyncio.run(main_async(args.concurrency))


if __name__ == "__main__":
    main()
