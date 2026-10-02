#!/usr/bin/env python3
"""Verify that the DL20 run of a model is FULLY cached for every algorithm, so the
latency replay only reads cache (adds formula latency) and never calls the API.

Runs each standalone algorithm on the requested queries in cache-only mode (see
order_by/replay.py): a raised RealCall means a genuine cache miss (prompt/params
diverge from the canonical run, or the entry was never populated).

Usage:
    python latency/check_cache_coverage.py --max-queries 3
    python latency/check_cache_coverage.py --max-queries 54 --algorithms quick_sort3
"""
import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "test"))

import run_experiment as RE  # noqa: E402
from benchmarks import load_dl20  # noqa: E402
from order_by.replay import NoNetworkClient, RealCall, install_memcache  # noqa: E402


async def main_async(args):
    bench = load_dl20(RE._resolve(args.dl20_run_file), args.hit_depth)
    prepared = bench.shuffled(0)[: args.max_queries]  # canonical seed-0 shuffle

    install_memcache()
    client = NoNetworkClient()
    algos = args.algorithms.split(",") if args.algorithms else RE.PASSAGE_ALGORITHMS

    print(f"Checking cache coverage on {len(prepared)} DL20 queries (seed 0), model={args.model}\n")
    all_ok = True
    for name in algos:
        miss_qids = []
        for qid, query, top in prepared:
            try:
                await RE.passage_algorithm(name, top, query, client, args.model)
            except RealCall:
                miss_qids.append(qid)
        if miss_qids:
            all_ok = False
            print(f"  {name:<24s} MISS on {len(miss_qids)}/{len(prepared)} queries "
                  f"(e.g. {miss_qids[:3]})")
        else:
            print(f"  {name:<24s} OK  (fully cached on all {len(prepared)} queries)")
    print("\n" + ("ALL CACHED — safe for cache-only latency replay."
                  if all_ok else "MISSES FOUND — populate cache before latency replay."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--algorithms", default=None)
    ap.add_argument("--max-queries", type=int, default=3)
    ap.add_argument("--hit-depth", type=int, default=100)
    ap.add_argument("--dl20-run-file", default="data/run.msmarco-v1-passage.bm25-default.dl20.txt")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
