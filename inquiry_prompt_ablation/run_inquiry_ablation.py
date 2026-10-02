#!/usr/bin/env python3
"""Inquiry-prompt ablation: can the inquiry prompt separate PARAMETRIC (world-factual)
questions from NON-PARAMETRIC (situation-dependent) ones?

Benchmark: data/nq_situatedqa_combined.jsonl -- a balanced 274-question set:
  * 137 PARAMETRIC questions from Natural Questions (NQ-open): stable world facts
    with a single correct answer (e.g. "who wrote the novel Pride and Prejudice").
  * 137 NON-PARAMETRIC questions from SituatedQA-geo (is_dependent=yes): questions
    whose answer depends on the asker's situation (e.g. "when did we last win a
    national championship" -- who is "we"?).
Each line has a `label` field in {parametric, non_parametric}.

Prompt: V5b (DEFAULT_PROMPT below) -- a keyword-free, "situation"-based adaptation of
the deployed inquiry prompt (prompts.all_prompts.direct_inquiry_factual_knowledge_prompt),
tuned so it generalizes across models without naming domain-specific keywords. We map
isFactualKnowledge = Yes -> parametric, No -> non_parametric, and compare to `label`.
Reports accuracy, per-class precision/recall/F1, and the confusion matrix.

Shared diskcache keyed by (prompt, model), like the rest of the system.

Usage:
    python inquiry_prompt_ablation/run_inquiry_ablation.py                                   # gpt-5-mini + haiku
    python inquiry_prompt_ablation/run_inquiry_ablation.py --model llama3.1-70b --provider fireworks
    python inquiry_prompt_ablation/run_inquiry_ablation.py --model openai-gpt-5-mini --provider openai \
                                                            --model claude-haiku-4-5 --provider anthropic
To sweep GPT-5-mini + Haiku + Llama-3.1-70B in one go, use run_all_models.py.
"""
import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from order_by.cache import cache  # noqa: E402  (shared LLM response cache)
from order_by.clients import PROVIDERS, build_client  # noqa: E402
from order_by.optimizer import SeenStatus  # noqa: E402  (inquiry-prompt schema)
from order_by.utils import hash_prompt, load_env_file, resolve  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "nq_situatedqa_combined.jsonl"
_CLASSES = ["parametric", "non_parametric"]
DEFAULT_MODELS = [("openai-gpt-5-mini", "openai"), ("claude-haiku-4-5", "anthropic")]
_DISPLAY = {"openai-gpt-5-mini": "GPT-5-mini", "claude-haiku-4-5": "Haiku-4.5",
            "llama3.1-70b": "Llama-3.1-70B"}

# ---------------------------------------------------------------------------
# V5b: keyword-free ("situation"-based) inquiry prompt.  Yes => PARAMETRIC
# (single world-factual answer); No => NON-PARAMETRIC (situation-dependent).
# ---------------------------------------------------------------------------
DEFAULT_PROMPT = """Can the following question be answered using factual world knowledge?

Question:
{question}

First decide: does the question have a single correct answer that is the same for everyone, or does its answer depend on the asker's situation so that it differs from person to person?
A question is situation-dependent if it leaves some reference unstated that the asker has in mind, so that people in different situations would give different correct answers. Look carefully -- a question can look like a simple fact but still assume the asker's own situation.
Return 'Yes' only if the answer is a single fixed fact -- the same regardless of the asker's situation -- that can be found by retrieving explicit factual world information.
Return 'No' if the answer depends on the asker's situation, or if the question is general, subjective, or requires interpretation or nuanced reasoning.
If the question involves relevance matching, interpretation, semantic judgment, nuanced reasoning, or any subjective preference, return 'No'.
If you return 'Yes', also provide a short, specific web search query that would help retrieve the needed facts.
If you return 'No', do not provide a web search query.
"""


def _load(path):
    """Load the combined benchmark. Accepts the `label` field (parametric /
    non_parametric); falls back to SituatedQA's `is_dependent` (yes=non_parametric)."""
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "label" in r:
                gold = r["label"]
            else:
                dep = str(r.get("is_dependent", "")).strip().lower()
                gold = "non_parametric" if dep == "yes" else "parametric"
            q = (r.get("question") or "").strip()
            if not q or gold not in _CLASSES:
                continue
            data.append({"question": q, "gold": gold, "id": r.get("id"),
                         "src": r.get("source", "")})
    return data


def _to_pred(is_factual):
    v = getattr(is_factual, "value", is_factual)   # status enum -> "Yes"/"No"; str passthrough
    return "parametric" if str(v).strip().lower() == "yes" else "non_parametric"


def _cached_val(entry):
    """Yes/No value from a cache entry (new {"isFactualKnowledge": "Yes"} or old
    {"parsed": {...}} form)."""
    if not isinstance(entry, dict):
        return None
    if "isFactualKnowledge" in entry:
        return entry["isFactualKnowledge"]
    return (entry.get("parsed") or {}).get("isFactualKnowledge")


def _hit(ex, val):
    v = getattr(val, "value", val)
    return {**ex, "pred": _to_pred(val), "raw": str(v), "cached": True}


async def _classify(client, model, ex, sem, prompt_template=DEFAULT_PROMPT, use_cache=True):
    prompt = prompt_template.format(question=ex["question"])
    key = hash_prompt(prompt, model)   # shared diskcache, keyed by (prompt, model)
    if use_cache and key in cache:
        val = _cached_val(cache[key])
        if val is not None:
            return _hit(ex, val)
    last = None
    async with sem:
        if use_cache and key in cache:  # re-check in case a concurrent task populated it
            val = _cached_val(cache[key])
            if val is not None:
                return _hit(ex, val)
        for _ in range(4):
            try:
                resp = await resolve(client.beta.chat.completions.parse(
                    model=model,
                    messages=[{"role": "system", "content": "You are a helpful agent. Think step by step. Output a JSON object."},
                              {"role": "user", "content": prompt}],
                    temperature=0.0, response_format=SeenStatus, max_completion_tokens=2048))
                parsed = resp.choices[0].message.parsed
                val = parsed.isFactualKnowledge.value   # "Yes" / "No"
                if use_cache:
                    cache[key] = {"isFactualKnowledge": val}
                return {**ex, "pred": _to_pred(val), "raw": val, "cached": False}
            except Exception as e:
                last = e
        return {**ex, "pred": "ERROR", "raw": str(last)}


def _metrics(results):
    cm = {g: {p: 0 for p in _CLASSES} for g in _CLASSES}
    n = correct = 0
    for r in results:
        if r["pred"] not in _CLASSES:
            continue
        cm[r["gold"]][r["pred"]] += 1
        n += 1
        correct += (r["gold"] == r["pred"])
    out = {"n": n, "accuracy": correct / n if n else 0.0, "confusion": cm, "per_class": {}}
    for c in _CLASSES:
        tp = cm[c][c]
        fp = sum(cm[g][c] for g in _CLASSES if g != c)
        fn = sum(cm[c][p] for p in _CLASSES if p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out["per_class"][c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


async def run_model(model, provider, data, concurrency=16, prompt_template=DEFAULT_PROMPT,
                    save=True):
    """Classify every question with one model; return (metrics, n_errors). Writes
    results_<model>.json when save=True."""
    client = build_client(provider)
    sem = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(*[
        _classify(client, model, ex, sem, prompt_template) for ex in data])
    n_err = sum(1 for r in results if r["pred"] == "ERROR")
    m = _metrics(results)
    if save and m["n"] > 0:   # don't write a results file if every call errored (model unreachable)
        (HERE / f"results_{model}.json").write_text(json.dumps(
            {"dataset": "nq_situatedqa_combined", "prompt": "V5b", "model": model,
             "metrics": m}, indent=2))
    return m, n_err


def print_table(rows):
    """rows: list of (model, metrics). Prints console table + LaTeX."""
    P = lambda m, c: m['per_class'][c]['precision'] * 100
    Rc = lambda m, c: m['per_class'][c]['recall'] * 100
    F = lambda m, c: m['per_class'][c]['f1'] * 100

    print("\n" + "=" * 88)
    print(f"{'Model':<16s}{'Acc':>7s}   {'Param P':>8s}{'Param R':>8s}{'Param F1':>9s}   "
          f"{'Non-P P':>8s}{'Non-P R':>8s}{'Non-P F1':>9s}")
    print("-" * 88)
    for model, m in rows:
        print(f"{_DISPLAY.get(model, model):<16s}{m['accuracy']*100:>6.1f}%   "
              f"{P(m,'parametric'):>8.1f}{Rc(m,'parametric'):>8.1f}{F(m,'parametric'):>9.1f}   "
              f"{P(m,'non_parametric'):>8.1f}{Rc(m,'non_parametric'):>8.1f}{F(m,'non_parametric'):>9.1f}")
    print("=" * 88)

    print("\n% ---- LaTeX ----")
    print("\\begin{tabular}{lccccccc}")
    print("\\toprule")
    print("\\multirow{2}{*}{Model} & \\multirow{2}{*}{Acc.} & \\multicolumn{3}{c}{Parametric (NQ)} "
          "& \\multicolumn{3}{c}{Non-parametric (SituatedQA)} \\\\")
    print("\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}")
    print(" & & P & R & F1 & P & R & F1 \\\\")
    print("\\midrule")
    for model, m in rows:
        print(f"{_DISPLAY.get(model, model)} & {m['accuracy']*100:.1f}\\% & "
              f"{P(m,'parametric'):.1f} & {Rc(m,'parametric'):.1f} & {F(m,'parametric'):.1f} & "
              f"{P(m,'non_parametric'):.1f} & {Rc(m,'non_parametric'):.1f} & {F(m,'non_parametric'):.1f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


def load_env():
    load_env_file(PROJECT_ROOT / ".env")


async def main_async(args):
    load_env()
    data = _load(args.data)
    dist = Counter(d["gold"] for d in data)
    models = list(zip(args.model, args.provider)) if args.model else DEFAULT_MODELS
    print(f"Inquiry-prompt ablation [V5b] on {len(data)} questions  "
          f"(parametric={dist['parametric']}, non_parametric={dist['non_parametric']})\n")

    rows = []
    for model, provider in models:
        m, n_err = await run_model(model, provider, data, args.concurrency)
        print(f"[{model}] acc={m['accuracy']*100:.1f}%"
              + (f"   [warn] {n_err} classification errors (excluded)" if n_err else ""))
        rows.append((model, m))
    print_table(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--model", action="append", default=[],
                    help="Model short-name (repeatable). Pair each with --provider.")
    ap.add_argument("--provider", action="append", default=[], choices=PROVIDERS,
                    help="Provider for the corresponding --model (repeatable).")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    if len(args.model) != len(args.provider):
        ap.error("each --model needs a matching --provider")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
