#!/usr/bin/env python3
"""Ablation on the OPTIMIZER SAMPLE SIZE for DL20.

The optimizer probes a small sample of each query's pool to estimate every access
path's quality and cost before committing to one (or, for RRF, a few). This study
fixes the base ranker (Claude Haiku-4.5 by default) and a single budget, and sweeps
the sample size -- how many items are probed during optimization. A larger sample
should sharpen the quality/cost estimates (better selection) but costs more to probe.

For each sample size we run the optimizer over all DL20 queries and report ndcg@10,
the optimization (probing) cost, and the total price. Uses the SAME seed as
test/run_optimizer.py (random.Random(0)).

The committed samplesize_<model>.json files were produced with budget 28 (Haiku)
and 9 (GPT-5 mini) over sample sizes 16,20,24,28; every response they need is in the
response cache, so they can be regenerated with --provider cache.

Usage:
    python test/dl20_ablation/run_samplesize_ablation.py --model claude-haiku-4-5 --provider cache --budget 28
    python test/dl20_ablation/run_samplesize_ablation.py --model openai-gpt-5-mini --provider cache --budget 9
    python test/dl20_ablation/run_samplesize_ablation.py --sample-sizes 12,16,20,24,28 --budget 56   # live, uncached sizes
    python test/dl20_ablation/run_samplesize_ablation.py --policies llm_judge --query-limit 5          # smoke test
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "test"))

import run_optimizer as RO  # noqa: E402
from order_by.clients import PROVIDERS, build_client  # noqa: E402
from order_by.utils import load_env_file  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def _make_args(sample_size, policy, base):
    """SimpleNamespace of args that run_optimizer_dl20 reads (judge = base model)."""
    return SimpleNamespace(
        model=base.ranking_model,
        judge_model=base.ranking_model,
        judge_client=None,
        proxy_policy=policy,
        total_ranking_budget=base.budget,
        sample_size=sample_size,
        ext_point_batch=base.ext_point_batch,
        rrf_k=base.rrf_k,
        ensemble_max_lists=base.ensemble_max_lists,
        dl20_run_file=base.dl20_run_file,
        hit_depth=base.hit_depth,
        query_limit=base.query_limit,
        seed=0,
    )


async def _run(base):
    load_env_file(PROJECT_ROOT / ".env")

    ranking_client = build_client(base.provider)

    # Oracle reference (best standalone algorithm per query) for the accuracy metric:
    # fraction of queries where the policy's chosen ranking matches/beats the oracle-best.
    oracle_best, oracle_scores, oracle_costs = RO._load_oracle_data(base.ranking_model, "dl20")

    def _oracle_accuracy(per_query_scores):
        if not oracle_best or not oracle_scores:
            return float("nan")
        n = correct = 0
        for qid, opt_ndcg in per_query_scores.items():
            best_ndcg = RO._oracle_summary(str(qid), oracle_best, oracle_scores, oracle_costs)[1]
            if best_ndcg == best_ndcg:  # skip NaN
                n += 1
                correct += (float(opt_ndcg) >= best_ndcg)
        return correct / n if n else float("nan")

    results = {p: {} for p in base.policies}
    for policy in base.policies:
        for ss in base.sample_sizes:
            print(f"\n=== policy={policy}  sample_size={ss}  budget=${base.budget} ===", flush=True)
            args = _make_args(ss, policy, base)
            res = await RO.run_optimizer_dl20(args, ranking_client)
            res["oracle_accuracy"] = _oracle_accuracy(res.get("per_query_scores", {}))
            results[policy][str(ss)] = res
            price = res["total_ranking_cost"] + res["total_optimization_cost"]
            print(f"    -> ndcg@10={res['score_mean']:.3f}  acc={res['oracle_accuracy']*100:.1f}%  "
                  f"opt_cost=${res['total_optimization_cost']:.4f}  "
                  f"rank_cost=${res['total_ranking_cost']:.4f}  total=${price:.4f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"samplesize_{base.ranking_model}.json"
    out_path.write_text(json.dumps({
        "dataset": "dl20", "generated_at": RO._now_iso(),
        "ranking_model": base.ranking_model, "policies": base.policies,
        "budget": base.budget, "seed": 0, "sample_sizes": base.sample_sizes,
        "query_limit": base.query_limit, "results": results,
    }, indent=2))

    # ── table: one block per policy (ndcg, opt cost, total) ───────────────────
    print("\n" + "=" * 70)
    print(f"DL20 SAMPLE-SIZE ABLATION  (ranker={base.ranking_model}, budget=${base.budget})")
    print("=" * 70)
    for policy in base.policies:
        print(f"\n[{policy}]")
        print(f"  {'sample_size':<12s}  {'ndcg@10':>8s}  {'acc%':>6s}  {'opt_cost$':>10s}  {'total$':>9s}")
        print("  " + "-" * 53)
        for ss in base.sample_sizes:
            r = results[policy][str(ss)]
            total = r["total_ranking_cost"] + r["total_optimization_cost"]
            acc = r.get("oracle_accuracy", float("nan")) * 100
            print(f"  {ss:<12d}  {r['score_mean']:>8.3f}  {acc:>6.1f}  "
                  f"{r['total_optimization_cost']:>10.3f}  {total:>9.3f}")
    print(f"\nsaved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="claude-haiku-4-5", help="Base ranking model.")
    ap.add_argument("--provider", default="anthropic", choices=PROVIDERS)
    ap.add_argument("--sample-sizes", default="16,20,24,28",
                    help="Comma-separated optimizer sample sizes to sweep (default 16,20,24,28).")
    ap.add_argument("--budget", type=float, default=42.0,
                    help="Total DL20 budget (default 42, from run_all.py DL20/Haiku sweep).")
    ap.add_argument("--policies", default="rrf_ensemble,llm_judge",
                    help="Comma-separated optimizer policies to sweep (default rrf_ensemble,llm_judge).")
    ap.add_argument("--ext-point-batch", type=int, default=8)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--ensemble-max-lists", type=int, default=0)
    ap.add_argument("--hit-depth", type=int, default=100)
    ap.add_argument("--dl20-run-file", default="data/run.msmarco-v1-passage.bm25-default.dl20.txt")
    ap.add_argument("--query-limit", type=int, default=None, help="Cap queries (smoke test).")
    a = ap.parse_args()
    base = SimpleNamespace(
        ranking_model=a.model, provider=a.provider,
        sample_sizes=[int(s) for s in a.sample_sizes.split(",") if s.strip()],
        budget=a.budget,
        policies=[p.strip() for p in a.policies.split(",") if p.strip()],
        ext_point_batch=a.ext_point_batch, rrf_k=a.rrf_k,
        ensemble_max_lists=a.ensemble_max_lists, hit_depth=a.hit_depth,
        dl20_run_file=a.dl20_run_file, query_limit=a.query_limit,
    )
    asyncio.run(_run(base))


if __name__ == "__main__":
    main()
