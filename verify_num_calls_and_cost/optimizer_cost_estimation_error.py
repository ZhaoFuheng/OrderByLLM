#!/usr/bin/env python3
"""Error between the OPTIMIZER's cost estimate and the ACTUAL cost, per query.

The optimizer estimates each algorithm's full cost by running it on a 20-item
sample and scaling by the call-count formula (OrderByOptimizer.estimated_total_price).
We reproduce exactly that per query: run each algorithm on the optimizer's sample
through its own dispatcher and apply estimated_total_price to the sample's cost.

Actual per-query costs come from test/<dataset>/results_<model>.json. For each
algorithm/model we report the mean absolute percentage error
    |estimate - actual| / actual
averaged over queries.

Cache-only: reads route through RAM and a no-network client refuses real calls, so
a genuine miss aborts loudly (the sample runs are already cached from the real
optimizer runs).

Usage:
    python verify_num_calls_and_cost/optimizer_cost_estimation_error.py --dataset dl20 --limit 2 --debug
    python verify_num_calls_and_cost/optimizer_cost_estimation_error.py --dataset dl20
"""
import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "test"))

import run_optimizer as RO            # noqa: E402
import order_by.optimizer as opt      # noqa: E402
from benchmarks import load_dl20, load_hellaswag, load_nfcorpus  # noqa: E402
from order_by.replay import NoNetworkClient, RealCall, install_memcache  # noqa: E402
from order_by.utils import tokens2price  # noqa: E402

MODELS = [("Llama-3.1 70B", "llama3.1-70b"),
          ("GPT-5 mini", "openai-gpt-5-mini"),
          ("Haiku-4.5", "claude-haiku-4-5")]
# optimizer alg-cost key -> standalone results.json algorithm name
EST_TO_ACTUAL = {
    "point": "pointwise",
    "quick": "quick_sort",
    "quick_3": "quick_sort3",
    "ext_merge_4": "external_merge_sort_4",
    "ext_bubble_4": "external_bubble_sort_4",
    "ext_point_4": "external_pointwise_4",
    "ext_point_8": "external_pointwise_4",   # optimizer may sample ext_point at batch 8
}
BIG = 1e9


def _new_quick_calls_formula(n, vote, limit_k):
    """Alternative partial-quicksort LLM-call model to test:
        vote * (2n + 2k*ln(n/k) + 2k + k*log2(k)),  k = min(limit_k, n).
    Full sort (k >= n) keeps vote * n * log2(n)."""
    import math
    if n <= 1:
        return 0.0
    k = min(limit_k, n)
    if k >= n:
        return vote * n * math.log2(max(n, 2))
    return vote * (2 * n + 2 * k * math.log(n / k) + 2 * k + k * math.log2(max(k, 2)))


def _actual_costs(dataset, model_key):
    p = PROJECT_ROOT / "test" / dataset / f"results_{model_key}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    out = {}  # alg -> {qid: cost}
    for m in d.get("metrics", []):
        by = {}
        for seed in m.get("per_query_costs", {}).values():
            for qid, c in seed.items():
                by.setdefault(qid, []).append(float(c))
        out[m["algorithm"]] = {q: sum(v) / len(v) for q, v in by.items()}
    return out


def _prepared(dataset, args):
    """Queries with the canonical seed-0 shuffle of run_optimizer.py, so the
    optimizer's sample runs hit the cache."""
    if dataset == "dl20":
        bench = load_dl20(RO._resolve(args.dl20_run_file), 100)
    elif dataset == "nfcorpus":
        bench = load_nfcorpus(RO._resolve(args.nfcorpus_dir))
        bench.limit(args.nfcorpus_limit)
    else:
        bench = load_hellaswag(RO._resolve(args.hellaswag_dir), 100)
    prepared = [(str(qid), query, top) for qid, query, top in bench.shuffled(0)]
    return prepared[: args.limit] if args.limit else prepared


def _build_opt(dataset, query, top, client, model):
    # All three datasets share the passage optimizer builder (same prompts/schema).
    oargs = SimpleNamespace(
        model=model, judge_model=model, judge_client=None, proxy_policy="borda",
        total_ranking_budget=BIG, sample_size=20, ext_point_batch=8, rrf_k=60,
        ensemble_max_lists=0, dl20_run_file="", hit_depth=100)
    return RO._build_passage_optimizer(query, top[:], client, oargs, BIG)


async def _sample(o, alg, sample):
    """Run one algorithm on the 20-item sample via the optimizer's own dispatcher.
    Returns (sorted, calls, in_tok, out_tok), or None if that sample isn't cached."""
    try:
        return await o.map_algname_2_alg(sample[:], alg, final_decision=False)
    except RealCall:
        return None


async def estimate_query(dataset, query, top, client, model):
    """Reproduce the optimizer's FINAL cost estimate for each algorithm: run it on the
    first-20 sample and apply estimated_total_price with that algorithm's OWN sample
    cost/calls (mirrors physical_order_by_impl's candidate re-estimation). quick_3 =
    3x quick. Batch samples uncached at the model's budgets are skipped (None)."""
    o = _build_opt(dataset, query, top, client, model)
    sample = list(o.data[: int(o.sample_size)])
    ss = int(o.sample_size)
    price = lambda r: tokens2price(model, r[2], r[3])
    est = {}
    rep = await _sample(o, "ext_point_4", sample)
    if rep:
        est["ext_point_4"] = o.estimated_total_price("ext_point_4", price(rep), ss)
    rp = await _sample(o, "point", sample)
    if rp:
        est["point"] = o.estimated_total_price("point", price(rp), ss)
    rq = await _sample(o, "quick", sample)
    if rq:
        est["quick"] = o.estimated_total_price("quick", price(rq), ss, actual_sample_api_calls=rq[1])
        est["quick_3"] = o.estimated_total_price("quick_3", None, None, None)
    rm = await _sample(o, "ext_merge_4", sample)
    if rm:
        est["ext_merge_4"] = o.estimated_total_price("ext_merge_4", price(rm), ss, actual_sample_api_calls=rm[1])
    rb = await _sample(o, "ext_bubble_4", sample)
    if rb:
        est["ext_bubble_4"] = o.estimated_total_price("ext_bubble_4", price(rb), ss, actual_sample_api_calls=rb[1])
    return est


async def run(args):
    install_memcache()
    if getattr(args, "new_quick", False):
        opt.quick_calls_formula = _new_quick_calls_formula   # patch the full-N quick call count
        print("[using alternative quick_calls_formula: 2n + 2k*ln(n/k) + 2k + k*log2(k)]")
    client = NoNetworkClient()

    for disp, key in MODELS:
        actual = _actual_costs(args.dataset, key)
        if not actual:
            print(f"[skip] {disp}: no results_{key}.json"); continue
        prepared = _prepared(args.dataset, args)
        # accumulate abs pct error per est-alg
        errs = {}
        for qid, query, top in prepared:
            est = await estimate_query(args.dataset, query, top, client, key)
            if args.debug and qid == prepared[0][0]:
                print(f"\n[{disp}] est keys for qid={qid}: {sorted(est)}")
            for est_alg, pred in est.items():
                act_alg = EST_TO_ACTUAL.get(est_alg)
                if act_alg is None or act_alg not in actual:
                    continue
                a = actual[act_alg].get(qid)
                if a is None or a <= 0:
                    continue
                errs.setdefault(est_alg, []).append((pred - a) / a * 100)  # signed % error
                if args.debug and qid == prepared[0][0]:
                    print(f"    {est_alg:<12s} est=${pred:.4f}  actual=${a:.4f}  "
                          f"err={abs(pred-a)/a*100:.1f}%")
        print(f"\n== {disp} ({args.dataset}, {len(prepared)} queries) ==")
        for est_alg in ["ext_point_4", "point", "ext_merge_4", "ext_bubble_4", "quick", "quick_3"]:
            if est_alg in errs:
                vals = errs[est_alg]
                mabs = statistics.mean(abs(v) for v in vals)
                bias = statistics.mean(vals)                    # signed: + = over-estimate
                print(f"  {est_alg:<14s} mean|err|% = {mabs:5.1f}%   bias% = {bias:+6.1f}%  "
                      f"(n={len(vals)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["dl20", "hellaswag", "nfcorpus"], default="dl20")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--new-quick", action="store_true",
                    help="Use the alternative partial-quicksort call formula for quick/quick_3.")
    ap.add_argument("--dl20-run-file", default="data/run.msmarco-v1-passage.bm25-default.dl20.txt")
    ap.add_argument("--hellaswag-dir", default="data/hellaswag")
    ap.add_argument("--nfcorpus-dir", default="data/nfcorpus")
    ap.add_argument("--nfcorpus-limit", type=int, default=3,
                    help="Number of NFCorpus queries (matches run_optimizer_nfcorpus default 3).")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
