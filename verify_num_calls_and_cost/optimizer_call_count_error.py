#!/usr/bin/env python3
"""LLM-call-count estimation error of the optimizer's cost model.

For every algorithm the optimizer predicts the number of LLM calls of a full run
from a closed-form complexity formula (order_by/utils.py: quick_calls_formula,
merge_calls_formula, bubble_calls_formula; the value-based paths are linear in the
input). This script compares that prediction with the number of calls a full run
actually issues, replayed from the response cache, and reports the mean absolute
percentage error per algorithm and model:

    |predicted_calls - actual_calls| / actual_calls

Everything replays from the cache (no API key, no API calls); a query whose run is
not cached is skipped and counted as a miss. Per-query data is written to
verify_num_calls_and_cost/call_counts_<dataset>.json.

Usage:
    python verify_num_calls_and_cost/optimizer_call_count_error.py --dataset dl20
    python verify_num_calls_and_cost/optimizer_call_count_error.py --dataset nfcorpus
    python verify_num_calls_and_cost/optimizer_call_count_error.py --dataset hellaswag
"""
import argparse
import asyncio
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "test"))
sys.path.insert(0, str(HERE))

import optimizer_cost_estimation_error as CE  # noqa: E402  (shared loaders / optimizer builder)
from order_by.replay import NoNetworkClient, RealCall, install_memcache  # noqa: E402
from order_by.utils import bubble_calls_formula, merge_calls_formula, quick_calls_formula  # noqa: E402

MODELS = [("Llama-3.1 70B", "llama3.1-70b"), ("GPT-5 mini", "openai-gpt-5-mini"), ("Haiku-4.5", "claude-haiku-4-5")]
ALGS = ["ext_point_4", "point", "ext_merge_4", "ext_bubble_4", "quick", "quick_3"]
TOP_K = 10


def predicted_calls(alg: str, n: int) -> float:
    if alg == "point":
        return n
    if alg == "ext_point_4":
        return math.ceil(n / 4)
    if alg == "quick":
        return quick_calls_formula(n, 1, TOP_K)
    if alg == "quick_3":
        return 3 * quick_calls_formula(n, 1, TOP_K)
    if alg == "ext_merge_4":
        return merge_calls_formula(n, 4, TOP_K)
    if alg == "ext_bubble_4":
        return bubble_calls_formula(n, 4, TOP_K)
    raise ValueError(alg)


async def run(args):
    install_memcache()
    client = NoNetworkClient()
    prepared = CE._prepared(args.dataset, args)
    out = {}
    for disp, key in MODELS:
        errs = {a: [] for a in ALGS}
        rows = []
        for qid, query, top in prepared:
            o = CE._build_opt(args.dataset, query, top, client, key)
            n = len(top)
            for alg in ALGS:
                try:
                    r = await o.map_algname_2_alg(top[:], alg, final_decision=False)
                except RealCall:
                    r = None
                pred = predicted_calls(alg, n)
                if r is None:
                    rows.append({"qid": qid, "alg": alg, "actual": None, "predicted": pred})
                    continue
                actual = r[1]
                rows.append({"qid": qid, "alg": alg, "actual": actual, "predicted": pred})
                errs[alg].append((pred - actual) / actual * 100)
        out[key] = rows
        print(f"\n== {disp} ({args.dataset}, {len(prepared)} queries) -- LLM-call-count estimation error ==")
        for a in ALGS:
            v = errs[a]
            if v:
                print(f"  {a:<13s} mean|err|% = {statistics.mean(abs(x) for x in v):5.1f}%  "
                      f"bias% = {statistics.mean(v):+6.1f}%  (n={len(v)}; miss={len(prepared) - len(v)})")
            else:
                print(f"  {a:<13s} no cached replays")
        sys.stdout.flush()
    path = HERE / f"call_counts_{args.dataset}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nper-query data -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["dl20", "hellaswag", "nfcorpus"], default="dl20")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N queries.")
    ap.add_argument("--dl20-run-file", default="data/run.msmarco-v1-passage.bm25-default.dl20.txt")
    ap.add_argument("--hellaswag-dir", default="data/hellaswag")
    ap.add_argument("--nfcorpus-dir", default="data/nfcorpus")
    ap.add_argument("--nfcorpus-queries", type=lambda s: [q.strip() for q in s.split(",") if q.strip()],
                    default=["PLAIN-1018", "PLAIN-102", "PLAIN-1050"],
                    help="NFCorpus query ids (default: the three used in the paper; '' selects the first --nfcorpus-limit).")
    ap.add_argument("--nfcorpus-limit", type=int, default=3)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
