#!/usr/bin/env python3
"""Verify the LLM-call-count and cost formulas on DL20 (or HellaSwag).

For every query and each algorithm in
    [ext_point_4, point, quick, quick_3, ext_bubble_4, ext_merge_4]
we compare:

  * ACTUAL #calls / cost  — run the algorithm on the FULL query (served from the
    sort_cache; the returned api-call count and cached tokens give the truth).
  * PREDICTED #calls      — the closed-form formula (Table 1 in the paper):
        point            -> N
        ext_point_4      -> ceil(N/m)
        quick / quick_3  -> quick_calls_formula(N, v, K)
        ext_bubble_4     -> bubble_calls_formula(N, m, K)
        ext_merge_4      -> merge_calls_formula(N, m, K)
  * PREDICTED cost        — the optimizer's estimator: run a 20-item SAMPLE, take
        its (price, #calls), and scale by the call formula
        (this mirrors OrderByOptimizer.estimated_total_price).

It prints (and saves) a table of the mean/std of the relative estimation error
for both #calls and cost, per algorithm.

The seed-0 shuffle (seeds=[0], hit_depth=100) is reproduced exactly so every call
hits cache; a call-counter reports any cache miss (= real API spend) so you know
the run stayed free.

Usage:
    python verify_num_calls_and_cost/verify.py                       # llama, all queries
    python verify_num_calls_and_cost/verify.py --model openai-gpt-5-mini --provider openai
    python verify_num_calls_and_cost/verify.py --limit 3             # smoke test
"""
import argparse
import asyncio
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "test"))

import csv                                            # noqa: E402
import run_experiment as R                            # noqa: E402 (test/run_experiment)
from benchmarks import load_dl20, load_hellaswag      # noqa: E402
from order_by.clients import PROVIDERS, build_client  # noqa: E402
from order_by.utils import (                          # noqa: E402
    load_env_file, tokens2price,
    quick_calls_formula, bubble_calls_formula, merge_calls_formula,
)

ALGS = ["ext_point_4", "point", "quick", "quick_3", "ext_bubble_4", "ext_merge_4"]
# optimizer-style algorithm name -> standalone algorithm name in test/run_experiment.py
_STANDALONE_NAME = {
    "ext_point_4": "external_pointwise_4", "point": "pointwise",
    "quick": "quick_sort", "quick_3": "quick_sort3",
    "ext_bubble_4": "external_bubble_sort_4", "ext_merge_4": "external_merge_sort_4",
}
M = 4          # batch / memory size
K = 10         # LIMIT k
SAMPLE = 20    # sample size used by the optimizer's cost estimator
SEED = 0
HIT_DEPTH = 100
RUN_FILE = "data/run.msmarco-v1-passage.bm25-default.dl20.txt"


# ── call-counter client wrapper (detects cache misses = real API spend) ───────
class _CountCompletions:
    def __init__(self, real, c): self._real, self._c = real, c
    async def create(self, *a, **k):
        self._c["n"] += 1
        return await self._real.create(*a, **k)
    def __getattr__(self, n): return getattr(self._real, n)

class _CountChat:
    def __init__(self, real, c): self.completions = _CountCompletions(real.completions, c); self._real = real
    def __getattr__(self, n): return getattr(self._real, n)

class _CountClient:
    def __init__(self, real, c): self.chat = _CountChat(real.chat, c); self._real = real
    def __getattr__(self, n): return getattr(self._real, n)


# ── predicted #calls (closed form) ────────────────────────────────────────────
def predicted_calls(alg, n):
    if alg == "point":         return float(n)
    if alg == "ext_point_4":   return float(math.ceil(n / M))
    if alg == "quick":         return float(quick_calls_formula(n, 1, K))
    if alg == "quick_3":       return float(quick_calls_formula(n, 3, K))
    if alg == "ext_bubble_4":  return float(bubble_calls_formula(n, M, K))
    if alg == "ext_merge_4":   return float(merge_calls_formula(n, M, K))
    raise ValueError(alg)


# ── run one algorithm on `data`; return (num_calls, in_tok, out_tok) ──────────
async def run_alg(alg, data, query, client, model):
    r = await R.passage_algorithm(_STANDALONE_NAME[alg], data, query, client, model)
    if alg in ("point", "ext_point_4"):
        return r[2], r[3], r[4]     # (ids, scores, calls, in, out[, texts])
    return r[1], r[2], r[3]         # (sorted, calls, in, out)


# ── predicted cost via the optimizer's sample-based estimator ─────────────────
def predicted_cost(alg, n, model, sample_price, sample_calls, quick_cost=None):
    s = max(sample_calls, 1)
    if alg in ("point", "ext_point_4"):
        return sample_price * (n / SAMPLE)            # scale by data ratio (== call ratio)
    if alg == "quick":
        correction = 1.0                                 # was expected_sample/s_calls
        return sample_price * quick_calls_formula(n, 1, K) * correction / s
    if alg == "quick_3":
        # quick_1's per-call price x quick_3's call count. Since quick_cost =
        # (p1/c1)*F(N,1) and F(N,3)=3*F(N,1), this equals 3*quick_cost — i.e. the
        # per-call cost is taken from quick_1 (same comparison prompt), NOT a v3 sample.
        return 3.0 * quick_cost
    if alg == "ext_bubble_4":
        return sample_price * bubble_calls_formula(n, M, K) / s
    if alg == "ext_merge_4":
        return sample_price * merge_calls_formula(n, M, K) / s
    raise ValueError(alg)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1-70b")
    ap.add_argument("--provider", default="fireworks", choices=PROVIDERS)
    ap.add_argument("--dataset", default="dl20", choices=["dl20", "hellaswag"])
    ap.add_argument("--limit", type=int, default=None, help="cap #queries (smoke test)")
    ap.add_argument("--hit-depth", type=int, default=HIT_DEPTH,
                    help="[dl20] docs per query (N). Smaller N => cheaper fresh runs; the "
                         "formula check is valid at any N. Default 100.")
    ap.add_argument("--run-file", default=None, help="[dl20] BM25 run file")
    ap.add_argument("--hellaswag-dir", default="data/hellaswag", help="[hellaswag] data dir")
    ap.add_argument("--hellaswag-num-queries", type=int, default=100,
                    help="[hellaswag] pooled queries; pool N = 4x this (100 -> N=400). "
                         "Must match the cached experiment (default 100).")
    args = ap.parse_args()

    load_env_file(PROJECT_ROOT / ".env")
    counter = {"n": 0}
    client = _CountClient(build_client(args.provider), counter)
    model = args.model

    if args.dataset == "dl20":
        bench = load_dl20(R._resolve(args.run_file or RUN_FILE), args.hit_depth)
    else:
        bench = load_hellaswag(R._resolve(args.hellaswag_dir), args.hellaswag_num_queries)
    prepared = bench.shuffled(SEED)   # the canonical shuffle, so every call hits cache

    if args.limit is not None:
        prepared = prepared[:args.limit]

    n0 = len(prepared[0][2]) if prepared else 0
    print(f"[{model}] {args.dataset}: {len(prepared)} queries, N={n0}, algs={ALGS}", flush=True)

    rows = []  # per (qid, alg)
    sem = asyncio.Semaphore(10 if args.provider in ("fireworks", "openai") else 5)

    async def _one_query(qid, query, top):
        async with sem:
            n = len(top)
            sample = top[:SAMPLE]
            quick_cost = None
            for alg in ["quick"] + [a for a in ALGS if a != "quick"]:  # quick first (needed by quick_3)
                # sample run -> cost estimator inputs
                s_calls, s_in, s_out = await run_alg(alg, sample[:], query, client, model)
                s_price = tokens2price(model, s_in, s_out)
                # full run -> actuals
                a_calls, a_in, a_out = await run_alg(alg, top[:], query, client, model)
                a_cost = tokens2price(model, a_in, a_out)
                p_calls = predicted_calls(alg, n)
                p_cost = predicted_cost(alg, n, model, s_price, s_calls, quick_cost=quick_cost)
                if alg == "quick":
                    quick_cost = p_cost
                rows.append({
                    "qid": str(qid), "alg": alg, "N": n,
                    "pred_calls": p_calls, "act_calls": a_calls,
                    "pred_cost": p_cost, "act_cost": a_cost,
                    "calls_err_pct": 100.0 * (p_calls - a_calls) / a_calls if a_calls else float("nan"),
                    "cost_err_pct":  100.0 * (p_cost - a_cost) / a_cost if a_cost else float("nan"),
                })

    await asyncio.gather(*[_one_query(q, qy, tp) for q, qy, tp in prepared])
    # Queries finish in whatever order the API answers; write the rows in query
    # order so the CSV is reproducible run to run.
    order = {str(q): i for i, (q, _, _) in enumerate(prepared)}
    rows.sort(key=lambda r: (order[r["qid"]], ALGS.index(r["alg"])))

    out_dir = Path(__file__).resolve().parent

    def _mean(xs): return statistics.mean(xs) if xs else float("nan")
    def _std(xs):  return statistics.pstdev(xs) if len(xs) > 1 else 0.0

    # ── aggregate: mean/std of the relative errors, per algorithm ─────────────
    summary = []
    for alg in ALGS:
        g = [r for r in rows if r["alg"] == alg]
        if not g:
            continue
        ce = [r["calls_err_pct"] for r in g if r["calls_err_pct"] == r["calls_err_pct"]]
        ce_cost = [r["cost_err_pct"] for r in g if r["cost_err_pct"] == r["cost_err_pct"]]
        summary.append({
            "alg": alg,
            "n_q": len(g),
            "avg_N": _mean([r["N"] for r in g]),
            "pred_calls": _mean([r["pred_calls"] for r in g]),
            "act_calls": _mean([r["act_calls"] for r in g]),
            "calls_err%_mean": _mean(ce),
            "calls_err%_std": _std(ce),
            "pred_cost$": _mean([r["pred_cost"] for r in g]),
            "act_cost$": _mean([r["act_cost"] for r in g]),
            "cost_err%_mean": _mean(ce_cost),
            "cost_err%_std": _std(ce_cost),
        })

    cols = ["alg", "n_q", "avg_N", "pred_calls", "act_calls", "calls_err%_mean",
            "calls_err%_std", "pred_cost$", "act_cost$", "cost_err%_mean", "cost_err%_std"]
    widths = {c: max(len(c), 12) for c in cols}
    widths["alg"] = 14

    def _fmt(c, v):
        if c in ("alg",):  return f"{v:<{widths[c]}}"
        if c in ("n_q",):  return f"{v:>{widths[c]}d}"
        if "cost$" in c:   return f"{v:>{widths[c]}.5f}"
        return f"{v:>{widths[c]}.2f}"

    print(f"\n================ estimation-error summary ({args.dataset}, per algorithm) ================")
    print("  ".join(f"{c:>{widths[c]}}" if c != "alg" else f"{c:<{widths[c]}}" for c in cols))
    for r in summary:
        print("  ".join(_fmt(c, r[c]) for c in cols))
    print(f"\ncache misses (real API calls made): {counter['n']}"
          + ("  <-- WARNING: some prompts were NOT cached (real spend)" if counter["n"] else "  (all cache hits, $0)"))

    with (out_dir / f"per_query_{model.replace('/', '-')}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with (out_dir / f"summary_{model.replace('/', '-')}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(summary)
    print(f"\nWrote per-query rows and summary CSV to {out_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
