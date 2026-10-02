#!/usr/bin/env python3
"""Option-A sample-size ablation figure: quality is flat within CI, cost rises.

x-axis = optimizer sample size. Left y-axis = ndcg@10 with 80% confidence intervals
(one line per policy) -- the intervals overlap across sample sizes, i.e. quality does
not change significantly. Right y-axis = optimization (probing) cost, which grows
~linearly with sample size. Message: use the smallest sample; extra probing is
overhead. Reads test/dl20_ablation/samplesize_<model>.json.

Usage:
    python test/dl20_ablation/plot_samplesize.py --model claude-haiku-4-5
"""
import argparse
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
_Z = 1.2816  # 80% CI

# policy -> (legend label, color, marker) -- matches the house Self-Cons / Judge style
_STYLE = {
    "rrf_ensemble": ("Self-Cons", "tab:orange", "D"),
    "llm_judge":    ("Judge",     "tab:cyan",   "P"),
}


def _ci(per_query_scores):
    vals = [float(v) for v in per_query_scores.values()]
    n = len(vals)
    if n < 2:
        return (statistics.mean(vals) if vals else float("nan")), 0.0
    return statistics.mean(vals), _Z * statistics.pstdev(vals) / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--ylim", default="0.6,0.7", help="ndcg y-axis limits, e.g. 0.6,0.7")
    args = ap.parse_args()

    d = json.loads((HERE / f"samplesize_{args.model}.json").read_text())
    sizes = d["sample_sizes"]
    budget = d["budget"]
    results = d["results"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(8, 5.5))

    handles = []
    # ── ndcg with CI, per policy ─────────────────────────────────────────────
    for policy in d.get("policies", list(results)):
        if policy not in results:
            continue
        label, color, marker = _STYLE.get(policy, (policy, "gray", "o"))
        means, errs = [], []
        for s in sizes:
            m, c = _ci(results[policy][str(s)].get("per_query_scores", {}))
            means.append(m); errs.append(c)
        ax.errorbar(sizes, means, yerr=errs, color=color, marker=marker, markersize=9,
                    linewidth=2, capsize=4, elinewidth=1.3, markeredgecolor="black",
                    markeredgewidth=0.6, zorder=3)
        handles.append(Line2D([0], [0], color=color, marker=marker, markersize=9,
                              markeredgecolor="black", markeredgewidth=0.6,
                              linewidth=2, label=label))

    ax.set_xlabel("optimizer sample size", fontsize=14)
    ax.set_ylabel("ndcg@10", fontsize=14)
    ax.set_xticks(sizes)
    lo, hi = [float(x) for x in args.ylim.split(",")]
    ax.set_ylim(lo, hi)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              bbox_transform=ax.transAxes, ncol=len(handles), fontsize=12, framealpha=0.9)

    out = HERE / (f"dl20samplesize{args.model}".replace("_", "").replace("-", "") + ".png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"saved -> {out}   (budget=${budget}, 80% CI)")


if __name__ == "__main__":
    main()
