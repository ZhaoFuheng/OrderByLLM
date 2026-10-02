#!/usr/bin/env python3
"""Plot the DL20 judge-model ablation as a price-vs-quality figure.

x-axis = total price ($ = ranking + optimization cost); y-axis = ndcg@10. Each
selector (per-model judge, or Self-Consistency/RRF) is a curve over the budget
sweep. Reads test/dl20_ablation/results_<model>.json and writes a LaTeX-safe
PNG next to it. House style matches the other figures (fonts, grid, legend on top).

Usage:
    python test/dl20_ablation/plot_ablation.py --model llama3.1-70b
"""
import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

# config key in the JSON -> (legend label, color, marker)
_STYLE = {
    "llama_judge":  ("Llama-3.1 70B judge", "tab:blue",   "o"),
    "haiku_judge":  ("Haiku-4.5 judge",     "tab:green",  "s"),
    "sonnet_judge": ("Sonnet-4.5 judge",    "tab:red",    "^"),
    "gpt5nano_judge": ("GPT-5-nano judge",  "tab:purple", "v"),
    "rrf":          ("Self-Cons (RRF)",     "tab:orange", "D"),
}
_ORDER = ["llama_judge", "haiku_judge", "sonnet_judge", "gpt5nano_judge", "rrf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1-70b")
    ap.add_argument("--annotate-budget", action="store_true",
                    help="Label each point with its dollar budget.")
    args = ap.parse_args()

    data = json.loads((HERE / f"results_{args.model}.json").read_text())
    configs = data["configs"]
    budgets = data["budgets"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(9, 6))

    handles = []
    for name in _ORDER:
        if name not in configs:
            continue
        label, color, marker = _STYLE[name]
        pts = []
        for b in budgets:
            r = configs[name].get(b)
            if not r:
                continue
            price = r["total_ranking_cost"]   # ranking cost only (exclude optimization cost)
            pts.append((price, r["score_mean"], b))
        pts.sort()  # by price
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, linestyle="--", linewidth=2, marker=marker, markersize=11,
                color=color, markeredgecolor="black", markeredgewidth=0.8, zorder=3)
        handles.append(Line2D([0], [0], color=color, marker=marker, linestyle="--",
                              markeredgecolor="black", markeredgewidth=0.8,
                              markersize=10, label=label))
        if args.annotate_budget:
            for x, y, b in pts:
                ax.annotate(f"${b}", (x, y), textcoords="offset points",
                            xytext=(5, -12), fontsize=8, color=color)

    ax.set_xlabel("Price (\\$)", fontsize=14)
    ax.set_ylabel("ndcg@10", fontsize=14)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              bbox_transform=ax.transAxes, ncol=math.ceil(len(handles) / 2),
              fontsize=12, framealpha=0.9)

    out = HERE / (f"dl20judgeablation{args.model}".replace("_", "").replace("-", "") + ".png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
