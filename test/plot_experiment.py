"""
Plot experiment results following the NBA.ipynb style:
  - Marker shape  encodes algorithm family (P=bm25, o=pointwise, ^=ext_pointwise,
                                            s=quick, *=bubble, D=merge)
  - Marker colour encodes variant (blue=plain, orange=+wiki search)
  - Error bars     show score_std across seeds
  - Log curve fit  over all data points (dashed red, with R²)

Supported datasets (add new entries to _YLIM to extend):
  - population  (kendall tau)
  - dl20        (ndcg@10)

Usage:
    # single file
    python test/plot_experiment.py --input test/dl20/results_openai-gpt-4.1.json

    # all JSONs in a directory  →  figures written alongside the JSONs
    python test/plot_experiment.py --input-dir test/dl20
    python test/plot_experiment.py --input-dir test/population

    # custom output directory
    python test/plot_experiment.py --input-dir test/dl20 --output-dir figures/test
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from scipy.optimize import curve_fit
import math


# ── Per-dataset y-axis limits (add new datasets here) ────────────────────────

_YLIM = {
    "dl19":           (0.40, 0.90),
    "dl20":           (0.40, 0.80),
    "population":     (0.95, 1.00),
    "sembench_movie": (0.50, 1.00),
}
_YLIM_DEFAULT = (0.10, 1.00)


# ── Marker / colour helpers ───────────────────────────────────────────────────

_FAMILY_COLOR = {
    "bm25":         "tab:gray",
    "pointwise":    "tab:blue",
    "ext_pointwise":"tab:green",
    "quick":        "tab:purple",
    "quick3":       "tab:purple",
    "bubble":       "tab:red",
    "merge":        "tab:brown",
}

_FAMILY_MARKER = {
    "bm25":         "P",
    "pointwise":    "o",
    "ext_pointwise":"^",
    "quick":        "s",
    "quick3":       "h",
    "bubble":       "*",
    "merge":        "D",
}


def _family(alg_name: str) -> str:
    n = alg_name.lower()
    if "bm25"   in n: return "bm25"
    if "quick_sort3" in n: return "quick3"
    if "quick"  in n: return "quick"
    if "bubble" in n: return "bubble"
    if "merge"  in n: return "merge"
    if "ext"    in n or "external" in n: return "ext_pointwise"
    return "pointwise"


def _short_label(alg_name: str) -> str:
    return (
        alg_name
        .replace("external_pointwise_4", "ext_point_4")
        .replace("external_pointwise", "ext_point")
        .replace("external_bubble_sort_4", "ext_bubble_4")
        .replace("external_bubble_sort", "ext_bubble")
        .replace("external_merge_sort_4",  "ext_merge_4")
        .replace("external_merge_sort",  "ext_merge")
        .replace("quick_sort3", "quick_3")
        .replace("quick_sort",  "quick")
        .replace("pointwise_with_search", "point_search")
        .replace("pointwise", "point")
        .replace("_with_search", "_search")
    )


def _style(alg_name: str):
    """Return (marker, facecolor, edgecolor, edgewidth, short_label)."""
    fam         = _family(alg_name)
    marker      = _FAMILY_MARKER[fam]
    color       = _FAMILY_COLOR[fam]
    with_search = "with_search" in alg_name.lower()
    # wiki-search variants: thick black edge to distinguish from plain
    edgecolor   = "black" if with_search else color
    edgewidth   = 2.0     if with_search else 0.8
    return marker, color, edgecolor, edgewidth, _short_label(alg_name)


# ── Curve fit ────────────────────────────────────────────────────────────────

def _log_fit(xs: np.ndarray, ys: np.ndarray):
    """Fit y = A * log(x + x0) + B.  Returns (popt, r2) or None on failure."""
    if len(xs) < 3:
        return None
    try:
        p0 = [np.ptp(ys), ys.min(), 0.1]
        bounds = ([-np.inf, -np.inf, 1e-9], [np.inf, np.inf, np.inf])
        popt, _ = curve_fit(
            lambda x, A, B, x0: A * np.log(x + x0) + B,
            xs, ys, p0=p0, bounds=bounds, maxfev=80_000,
        )
        y_hat = popt[0] * np.log(xs + popt[2]) + popt[1]
        ss_res = np.sum((ys - y_hat) ** 2)
        ss_tot = np.sum((ys - ys.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return popt, r2
    except Exception:
        return None


# ── Main plot function ────────────────────────────────────────────────────────

_OPTIMIZER_MARKER = {
    "borda":     "X",
    "llm_judge": "P",
    "ideal":     "H",
}

_OPTIMIZER_COLOR = {
    "borda":     "tab:orange",
    "llm_judge": "tab:cyan",
    "ideal":     "gold",
}


def _load_optimizer_data(results_dir: Path, model: str) -> list[dict]:
    """Load optimizer JSON and return a flat list of
    {policy, budget, score, cost} dicts for the given model."""
    pattern = f"optimizer_{model.replace('/', '-')}*.json"
    dots = []
    for p in sorted(results_dir.glob(pattern)):
        data = json.loads(p.read_text(encoding="utf-8"))
        by_model = data.get("results_by_model", {}).get(model, {})
        for policy, budgets in by_model.items():
            if policy == "ideal":
                continue
            for budget_str, rec in budgets.items():
                score = rec.get("score_mean", None)
                cost = rec.get("total_ranking_cost", rec.get("ranking_cost", None))
                # Backwards compat: old format stored cost inside seed_results
                if cost is None:
                    for sr in rec.get("seed_results", []):
                        c = sr.get("total_ranking_cost", sr.get("ranking_cost", None))
                        if c is not None:
                            cost = c
                            break
                if score is not None and cost is not None:
                    dots.append({
                        "policy": policy,
                        "budget": budget_str,
                        "score": score,
                        "cost": cost,
                    })
    return dots


def plot_payload(payload: dict, output_dir: Path, results_dir: Path | None = None) -> Path:
    dataset     = payload.get("dataset", "unknown")
    metric_name = payload.get("metric_name", "kendall_tau")
    model       = payload.get("settings", {}).get("model", "")
    points      = payload.get("metrics", [])
    if not points:
        raise ValueError("No metrics in payload.")

    xs    = np.array([float(p.get("price",      0.0)) for p in points])
    ys    = np.array([float(p.get("score_mean", p.get("score", 0.0))) for p in points])
    yerrs = np.array([float(p.get("score_std",  0.0)) for p in points])
    algs  = [str(p.get("algorithm", f"alg_{i}")) for i, p in enumerate(points)]

    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(9, 6))

    for x, y, yerr, alg in zip(xs, ys, yerrs, algs):
        marker, color, edgecolor, edgewidth, short_label = _style(alg)
        ax.errorbar(
            x, y,
            yerr=yerr if yerr > 0 else None,
            fmt=marker,
            color=color,
            markeredgecolor=edgecolor,
            markeredgewidth=edgewidth,
            markersize=10,
            capsize=4,
            elinewidth=1.2,
            linewidth=0,
            zorder=3,
        )

    # (text labels next to algorithm dots removed for cleaner plots)


    # ── Optimizer dots (borda, llm_judge, ideal) ────────────────────────────
    opt_dots = []
    if results_dir and model:
        opt_dots = _load_optimizer_data(results_dir, model)

    opt_texts = []
    present_policies = set()
    for dot in opt_dots:
        policy = dot["policy"]
        present_policies.add(policy)
        marker = _OPTIMIZER_MARKER.get(policy, "X")
        color  = _OPTIMIZER_COLOR.get(policy, "tab:olive")
        ax.plot(
            dot["cost"], dot["score"],
            marker=marker, color=color,
            markeredgecolor="black", markeredgewidth=1.2,
            markersize=13, zorder=5, linestyle="None",
        )
        if policy not in ("borda", "llm_judge", "ideal"):
            label = f'{policy} ${dot["budget"]}'
            opt_texts.append(ax.text(dot["cost"], dot["score"], label, fontsize=10, fontstyle="italic"))

    # Connect optimizer dots with curves, starting from bm25
    for curve_policy, curve_style in [("borda", "--"), ("llm_judge", "--"), ("ideal", "--")]:
        curve_dots = sorted([d for d in opt_dots if d["policy"] == curve_policy], key=lambda d: d["cost"])
        if curve_dots:
            cx = [d["cost"] for d in curve_dots]
            cy = [d["score"] for d in curve_dots]
            for x, y, alg in zip(xs, ys, algs):
                if "bm25" in alg.lower():
                    cx.insert(0, float(x))
                    cy.insert(0, float(y))
                    break
            if len(cx) >= 2:
                ax.plot(cx, cy, color=_OPTIMIZER_COLOR.get(curve_policy, "tab:olive"),
                        linestyle=curve_style, linewidth=2.5, zorder=6)

    if opt_texts:
        all_xs = list(xs) + [d["cost"] for d in opt_dots]
        all_ys = list(ys) + [d["score"] for d in opt_dots]
        adjust_text(
            opt_texts,
            x=all_xs, y=all_ys,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.6, shrinkA=10, shrinkB=4),
            expand=(1.5, 2.0),
            iter_lim=300,
        )

    # ── Legend: one entry per family (colour + shape) + wiki indicator ────────
    from matplotlib.lines import Line2D

    _FAMILY_LEGEND = {
        "bm25": "bm25",
        "pointwise": "point",
        "ext_pointwise": "ext_point_4",
        "quick": "quick",
        "quick3": "quick_3",
        "bubble": "ext_bubble_4",
        "merge": "ext_merge_4",
    }
    _OPTIMIZER_LEGEND = {
        "borda": "Opt(borda)",
        "llm_judge": "Opt(judge)",
        "ideal": "Opt(ideal)",
    }
    present_families = {_family(a) for a in algs}
    family_entries = [
        Line2D([0], [0], marker=_FAMILY_MARKER[f], color=_FAMILY_COLOR[f],
               linestyle="None", markersize=9, label=_FAMILY_LEGEND.get(f, f))
        for f in _FAMILY_MARKER
        if f in present_families
    ]
    optimizer_entries = [
        Line2D([0], [0], marker=_OPTIMIZER_MARKER[p], color=_OPTIMIZER_COLOR[p],
               markeredgecolor="black", markeredgewidth=1.2,
               linestyle="--", linewidth=2, markersize=10,
               label=_OPTIMIZER_LEGEND.get(p, p))
        for p in _OPTIMIZER_MARKER
        if p in present_policies
    ]
    has_search = any("with_search" in a.lower() for a in algs)
    wiki_entries = (
        [Line2D([0], [0], marker="o", color="grey", markeredgecolor="black",
                markeredgewidth=2, linestyle="None", markersize=9, label="w/ search")]
        if has_search else []
    )

    ax.set_xlabel("Price ($)", fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)

    all_handles = family_entries + wiki_entries + optimizer_entries
    ncol = math.ceil(len(all_handles) / 2)
    ax.legend(
        handles=all_handles,
        title="Algorithm",
        title_fontsize=13,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        bbox_transform=ax.transAxes,
        ncol=ncol,
        fontsize=13,
        framealpha=0.9,
        borderpad=0.5,
    )

    all_x = np.concatenate([xs, np.array([d["cost"] for d in opt_dots])]) if opt_dots else xs
    all_y = np.concatenate([ys, np.array([d["score"] for d in opt_dots])]) if opt_dots else ys
    x_pad = max(all_x) * 0.15
    ax.set_xlim(min(-0.05, -x_pad), max(all_x) + x_pad)
    y_min_preset, y_max_preset = _YLIM.get(dataset, _YLIM_DEFAULT)
    y_pad = 0.03
    y_min = min(y_min_preset, all_y.min() - y_pad)
    y_max_cap = 1.005 if dataset == "population" and model == "openai-gpt-4.1" else 1.05
    y_max = min(y_max_cap, max(y_max_preset, all_y.max() + y_pad))
    ax.set_ylim(y_min, y_max)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = f"_{model.replace('/', '-')}" if model else ""
    out_path = output_dir / f"{dataset}{model_tag}_{metric_name}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).parent


def _resolve(path_str: str) -> Path:
    """Resolve a path against cwd first, then against the script's directory."""
    p = Path(path_str)
    if p.is_absolute() or p.exists():
        return p
    candidate = _SCRIPT_DIR / p
    if candidate.exists():
        return candidate
    return p  # let the caller surface a meaningful error


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description="Read experiment JSON(s) and write plot(s) into an output directory."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Single experiment JSON file path.")
    group.add_argument(
        "--input-dir",
        help="Directory containing experiment JSON files; one figure per file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for figure outputs. "
            "Defaults to 'figures' for --input, or to --input-dir for --input-dir."
        ),
    )
    args = parser.parse_args()

    if args.input:
        input_path = _resolve(args.input)
        output_dir = _resolve(args.output_dir) if args.output_dir else Path("figures")
        payload    = load_payload(input_path)
        out_path   = plot_payload(payload, output_dir, results_dir=input_path.parent)
        print(f"Wrote figure to {out_path}")
    else:
        input_dir  = _resolve(args.input_dir)
        output_dir = _resolve(args.output_dir) if args.output_dir else input_dir
        json_files = sorted(f for f in input_dir.glob("*.json") if not f.name.startswith("optimizer_"))
        if not json_files:
            print(f"No JSON files found in {input_dir}")
            return
        for json_path in json_files:
            try:
                payload  = load_payload(json_path)
                out_path = plot_payload(payload, output_dir, results_dir=input_dir)
                print(f"Wrote figure to {out_path}")
            except Exception as exc:
                print(f"Skipped {json_path.name}: {exc}")


if __name__ == "__main__":
    main()
