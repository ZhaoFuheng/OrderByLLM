#!/usr/bin/env python3
"""Judge-model ablation on DL20.

A single base model does ALL the ranking; we compare optimizer *selection*
strategies on top of that identical ranking, using the SAME DL20 seed/shuffle as
test/run_optimizer.py (random.Random(0), hit_depth=100, per-query shuffle). Only the
selection policy / judge model differs, so every configuration ranks identical
passages and the comparison isolates the judge.

Configurable via flags (see --ranking-model / --judges). Two example setups:

  Haiku base (default):
    ranking = claude-haiku-4-5 (anthropic)
    judges  = haiku (anthropic), sonnet-4-5 (anthropic), gpt-5-nano (openai) + RRF

  Llama base:
    ranking = llama3.1-70b (fireworks)
    judges  = llama (fireworks), haiku-4-5 (anthropic), sonnet-4-5 (anthropic) + RRF

Each judge runs through its own provider client (built lazily and cached), so a
judge on a different provider than the ranking model works (e.g. llama ranking +
haiku judge). Judges on the ranking model's provider reuse the ranking client.

Budgets are the TOTAL dollar budget across all queries (run_optimizer_dl20 divides
by the query count internally), matching run_all.py's semantics. Pass the DL20 sweep
for whichever base model you use, e.g. run_all.py has DL20 Haiku "14,28,42,56" and
DL20 Llama "1,2,4,6". A judge can only choose among algorithms that fit the budget,
so a too-small budget forces every policy onto the cheapest algorithm and they
cannot diverge -- use the model's real DL20 sweep.

NOTE on cost: run_optimizer_dl20 prices ALL tokens (ranking + judge) at the ranking
model's rate, so reported cost is exact for the same-provider judge and RRF, but
approximate for cross-provider judges (their few selection calls are mispriced). The
ndcg@10 quality numbers are exact for every configuration.

Every response behind the committed results_<model>.json files is in the response
cache, so `cache` works as the provider of the ranker and of every judge.

Usage:
    # Llama base, replayed from the cache (reproduces results_llama3.1-70b.json)
    python test/dl20_ablation/run_ablation.py \
        --ranking-model llama3.1-70b --ranking-provider cache \
        --judges llama3.1-70b:cache,claude-haiku-4-5:cache,claude-sonnet-4-5:cache \
        --budgets 1,2,4,6

    # Llama base, live
    python test/dl20_ablation/run_ablation.py \
        --ranking-model llama3.1-70b --ranking-provider fireworks \
        --judges llama3.1-70b:fireworks,claude-haiku-4-5:anthropic,claude-sonnet-4-5:anthropic \
        --budgets 1,2,4,6

    # Haiku base (original), 3-query smoke test
    python test/dl20_ablation/run_ablation.py --query-limit 3 --budgets 42
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

import run_optimizer as RO  # noqa: E402  (test/run_optimizer.py)
from order_by.clients import PROVIDERS, build_client  # noqa: E402
from order_by.utils import load_env_file  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# Pretty short names for the comparison table / config keys.
_NAME_MAP = {
    "llama3.1-70b": "llama",
    "claude-haiku-4-5": "haiku",
    "claude-sonnet-4-5": "sonnet",
    "openai-gpt-5-nano": "gpt5nano",
    "openai-gpt-5-mini": "gpt5mini",
}


def _judge_name(model: str) -> str:
    return _NAME_MAP.get(model, model.replace(".", "").replace("-", "")) + "_judge"


def _make_args(budget, policy, judge_model, judge_client, base):
    """Build the SimpleNamespace of args that run_optimizer_dl20 reads."""
    return SimpleNamespace(
        model=base.ranking_model,
        judge_model=judge_model,
        judge_client=judge_client,
        proxy_policy=policy,
        total_ranking_budget=budget,
        sample_size=base.sample_size,
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

    # Lazily build and cache one client per provider.
    _clients: dict = {}

    def get_client(provider):
        if provider not in _clients:
            _clients[provider] = build_client(provider)
        return _clients[provider]

    ranking_client = get_client(base.ranking_provider)   # base model does all ranking

    # Build the config list: one llm_judge config per --judges entry, then RRF.
    configs = []
    for jm, jp in base.judges:
        configs.append((_judge_name(jm), "llm_judge", jm, jp))
    if base.include_rrf:
        configs.append(("rrf", "rrf_ensemble", None, None))

    results: dict = {}
    for name, policy, judge_model, provider in configs:
        judge_client = get_client(provider) if provider else None
        results[name] = {}
        for budget in base.budgets:
            budget_str = f"{budget:.4f}".rstrip("0").rstrip(".")
            print(f"\n=== config={name}  policy={policy}  judge={judge_model or '-'}"
                  f" ({provider or '-'})  budget={budget_str} ===", flush=True)
            args = _make_args(budget, policy, judge_model, judge_client, base)
            res = await RO.run_optimizer_dl20(args, ranking_client)
            results[name][budget_str] = res
            print(f"    -> score_mean={res['score_mean']:.3f}  "
                  f"ranking_cost=${res['total_ranking_cost']:.4f}  "
                  f"optimization_cost=${res['total_optimization_cost']:.4f}  "
                  f"chosen={res['chosen_alg_counts']}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"results_{base.ranking_model}.json"
    budget_strs = [f"{b:.4f}".rstrip("0").rstrip(".") for b in base.budgets]
    payload = {
        "dataset": "dl20",
        "generated_at": RO._now_iso(),
        "ranking_model": base.ranking_model,
        "ranking_provider": base.ranking_provider,
        "seed": 0,
        "hit_depth": base.hit_depth,
        "sample_size": base.sample_size,
        "budgets": budget_strs,
        "query_limit": base.query_limit,
        "configs": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    # ── comparison table ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"DL20 JUDGE ABLATION  (ranking={base.ranking_model}, seed=0"
          + (f", {base.query_limit} queries" if base.query_limit else "") + ")")
    print("=" * 72)
    print("Each cell: ndcg@10 (total price = ranking + optimization cost)")
    header = f"{'config':<16s}" + "".join(f"  budget=${b:<14s}" for b in budget_strs)
    print(header)
    print("-" * len(header))
    for name, *_ in configs:
        row = f"{name:<16s}"
        for b in budget_strs:
            r = results[name][b]
            price = r["total_ranking_cost"] + r["total_optimization_cost"]
            row += f"  {r['score_mean']:>6.3f} (${price:>7.2f}) "
        print(row)
    print(f"\nsaved -> {out_path}")


def _parse_judges(spec: str):
    """Parse 'model:provider,model:provider' -> [(model, provider), ...]."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"--judges entry '{tok}' must be 'model:provider'")
        m, p = tok.split(":", 1)
        out.append((m.strip(), p.strip()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranking-model", default="claude-haiku-4-5",
                    help="Base model that performs all ranking (default: claude-haiku-4-5).")
    ap.add_argument("--ranking-provider", default="anthropic",
                    choices=PROVIDERS,
                    help="Provider/client for the ranking model (default: anthropic).")
    ap.add_argument("--judges",
                    default="claude-haiku-4-5:anthropic,claude-sonnet-4-5:anthropic,openai-gpt-5-nano:openai",
                    help="Comma-separated 'model:provider' judge configs (each run as llm_judge). "
                         "Default reproduces the Haiku-base experiment.")
    ap.add_argument("--no-rrf", action="store_true",
                    help="Skip the rrf_ensemble (self-consistency) baseline config.")
    ap.add_argument("--budgets", default="14,28,42,56",
                    help="Comma-separated TOTAL dollar budgets, split per-query internally "
                         "(default: 14,28,42,56 = DL20/Haiku sweep from run_all.py; "
                         "use 1,2,4,6 for DL20/Llama).")
    ap.add_argument("--sample-size", type=int, default=20)
    ap.add_argument("--ext-point-batch", type=int, default=8)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--ensemble-max-lists", type=int, default=0)
    ap.add_argument("--hit-depth", type=int, default=100)
    ap.add_argument("--dl20-run-file",
                    default="data/run.msmarco-v1-passage.bm25-default.dl20.txt")
    ap.add_argument("--query-limit", type=int, default=None,
                    help="Cap number of DL20 queries (smoke test); default: all.")
    a = ap.parse_args()
    base = SimpleNamespace(
        ranking_model=a.ranking_model,
        ranking_provider=a.ranking_provider,
        judges=_parse_judges(a.judges),
        include_rrf=not a.no_rrf,
        budgets=[float(b.strip()) for b in a.budgets.split(",") if b.strip()],
        sample_size=a.sample_size,
        ext_point_batch=a.ext_point_batch,
        rrf_k=a.rrf_k,
        ensemble_max_lists=a.ensemble_max_lists,
        hit_depth=a.hit_depth,
        dl20_run_file=a.dl20_run_file,
        query_limit=a.query_limit,
    )
    asyncio.run(_run(base))


if __name__ == "__main__":
    main()
