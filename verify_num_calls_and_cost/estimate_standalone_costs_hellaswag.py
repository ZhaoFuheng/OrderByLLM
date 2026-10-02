#!/usr/bin/env python3
"""Estimate each standalone algorithm's cost on HellaSwag for the three models.

Reads the measured per-query costs from test/hellaswag/results_<model>.json (no API
calls) and, for every standalone access path, reports:
  * per-query mean cost +/- standard error (SE = stdev / sqrt(n_queries)), and
  * the total cost over all queries (+/- SE of the total = stdev * sqrt(n)).

The standard error captures query-to-query variability in cost (some queries are
longer / harder and cost more), so it quantifies the uncertainty of the cost
estimate rather than assuming a single deterministic number.

Prints a table (models as columns) and writes a CSV next to this script.

Usage:
    python verify_num_calls_and_cost/estimate_standalone_costs_hellaswag.py
    python verify_num_calls_and_cost/estimate_standalone_costs_hellaswag.py --per-query
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HS_DIR = PROJECT_ROOT / "test" / "hellaswag"

MODELS = [
    ("Llama-3.1 70B", "llama3.1-70b"),
    ("GPT-5 mini", "openai-gpt-5-mini"),
    ("Haiku-4.5", "claude-haiku-4-5"),
]
# Standalone access paths, cheapest-first display order.
ALGOS = [
    ("external_pointwise_4", "ext_point_4"),
    ("pointwise", "point"),
    ("external_merge_sort_4", "ext_merge_4"),
    ("quick_sort", "quick"),
    ("external_bubble_sort_4", "ext_bubble_4"),
    ("quick_sort3", "quick_3"),
]


def _per_query_costs(metric) -> list[float]:
    """Flatten per_query_costs {seed: {qid: cost}} -> one cost per query (avg over seeds)."""
    pq = metric.get("per_query_costs", {})
    by_q: dict[str, list[float]] = {}
    for seed_map in pq.values():
        for qid, cost in seed_map.items():
            by_q.setdefault(qid, []).append(float(cost))
    return [sum(v) / len(v) for v in by_q.values()]


def _stats(costs: list[float]) -> dict:
    n = len(costs)
    if n == 0:
        return {"n": 0}
    mean = statistics.mean(costs)
    sd = statistics.stdev(costs) if n > 1 else 0.0
    se_mean = sd / math.sqrt(n)                 # SE of the per-query mean
    total = sum(costs)
    se_total = sd * math.sqrt(n)                # SE of the total (= n * se_mean)
    return {"n": n, "mean": mean, "sd": sd, "se_mean": se_mean,
            "total": total, "se_total": se_total}


def load(model_key: str) -> dict:
    p = HS_DIR / f"results_{model_key}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    out = {}
    for m in d.get("metrics", []):
        out[m["algorithm"]] = _stats(_per_query_costs(m))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-query", action="store_true",
                    help="Show per-query mean +/- SE instead of total cost +/- SE.")
    args = ap.parse_args()

    data = {key: load(key) for _, key in MODELS}
    n_by_model = {key: (next(iter(data[key].values()), {}).get("n", 0)) for _, key in MODELS}

    mode = "per-query mean $/query +/- SE" if args.per_query else "total cost $ +/- SE"
    print(f"HellaSwag standalone cost estimate  ({mode})")
    print("n queries: " + ", ".join(f"{disp}={n_by_model[key]}" for disp, key in MODELS))
    header = f"{'algorithm':<14s}" + "".join(f"  {disp:>26s}" for disp, _ in MODELS)
    print("-" * len(header)); print(header); print("-" * len(header))

    rows = []
    for alg, short in ALGOS:
        cells = []
        row = {"algorithm": short}
        for disp, key in MODELS:
            s = data.get(key, {}).get(alg)
            if not s or s.get("n", 0) == 0:
                cells.append(f"{'--':>26s}"); row[key] = ""
                continue
            if args.per_query:
                val, se = s["mean"], s["se_mean"]
            else:
                val, se = s["total"], s["se_total"]
            cells.append(f"{('$%.4f +/- %.4f' % (val, se)):>26s}")
            row[key] = f"{val:.4f}"; row[f"{key}_se"] = f"{se:.4f}"
        print(f"{short:<14s}" + "".join("  " + c for c in cells))
        rows.append(row)

    out_csv = PROJECT_ROOT / "verify_num_calls_and_cost" / "standalone_costs_hellaswag.csv"
    fields = ["algorithm"]
    for _, key in MODELS:
        fields += [key, f"{key}_se"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nsaved -> {out_csv}   (values are {'per-query mean' if args.per_query else 'total'} + SE columns)")


if __name__ == "__main__":
    main()
