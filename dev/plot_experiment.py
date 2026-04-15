"""
Plot experiment results following the NBA.ipynb style:
  - Marker shape  encodes algorithm family (P=bm25, o=pointwise, ^=ext_pointwise,
                                            s=quick, *=bubble, D=merge)
  - Marker colour encodes variant (blue=plain, orange=+wiki search)
  - Error bars     show score_std across seeds
  - Log curve fit  over all data points (dashed red, with R²)

Usage:
    python dev/plot_experiment.py --input dev/results_nba.json
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit


# ── Marker / colour helpers ───────────────────────────────────────────────────

# One colour per algorithm family.
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

def plot_payload(payload: dict, output_dir: Path, include_dl19_bar: bool = False) -> Path:
    dataset     = payload.get("dataset", "unknown")
    metric_name = payload.get("metric_name", "kendall_tau")
    model       = payload.get("settings", {}).get("model", "")
    points     = payload.get("metrics", [])
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

    # Log curve fit — only drawn when R² ≥ 0.3 and slope is positive
    fit = _log_fit(xs, ys)
    if fit is not None:
        popt, r2 = fit
        A, B, x0 = popt
        if A > 0 and r2 >= 0.3:
            x_fit = np.linspace(xs.min(), xs.max(), 500)
            y_fit = A * np.log(x_fit + x0) + B
            ax.plot(x_fit, y_fit, "r--", linewidth=1.8, label=f"log fit (R²={r2:.3f})", zorder=2)

    # ── Legend: one entry per family (colour + shape) + wiki indicator ────────
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    _FAMILY_LEGEND = {
        "bm25": "bm25",
        "pointwise": "point",
        "ext_pointwise": "ext_point_4",
        "quick": "quick",
        "quick3": "quick_3",
        "bubble": "ext_bubble_4",
        "merge": "ext_merge_4",
    }
    present_families = {_family(a) for a in algs}
    family_entries = [
        Line2D([0], [0], marker=_FAMILY_MARKER[f], color=_FAMILY_COLOR[f],
               linestyle="None", markersize=9, label=_FAMILY_LEGEND.get(f, f))
        for f in _FAMILY_MARKER
        if f in present_families
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
    # ax.set_title(dataset.upper(), pad=14)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Legend placed outside the axes, just below the title (above the plot area).
    all_handles = family_entries + wiki_entries
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

    x_pad = max(xs) * 0.15
    ax.set_xlim(-x_pad, max(xs) + x_pad)
    y_min = 0.4 if dataset == "dl19" else 0.1
    y_max = 0.9 if dataset == "dl19" else 1.0
    ax.set_ylim(y_min, y_max)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = f"_{model.replace('/', '-')}" if model else ""
    out_path = output_dir / f"{dataset}{model_tag}_{metric_name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    # Optionally emit the per-query violin/bar chart alongside the scatter plot.
    if include_dl19_bar and dataset == "dl19":
        bar_path = plot_dl19_bar(payload, output_dir)
        print(f"  + bar chart → {bar_path}")

    return out_path


# ── DL19 bar chart ───────────────────────────────────────────────────────────

_ALG_ORDER = [
    "bm25",
    "external_pointwise_4",
    "pointwise",
    "quick_sort",
    "quick_sort3",
    "external_merge_sort_4",
    "external_bubble_sort_4",
]

# Colour for the highlighted-query trace line
_TRACE_COLOR = "#e63946"
_TRACE_QID = "1037798"
_DL19_BAR_COLORS = {
    "bm25": "tab:gray",
    "point": "tab:blue",
    "ext_point_4": "tab:green",
    "quick": "#b39ddb",
    "quick_3": "tab:purple",
    "ext_merge_4": "tab:brown",
    "ext_bubble_4": "tab:red",
    "optimal": "#66c2a5",
}


def plot_dl19_bar(payload: dict, output_dir: Path) -> Path:
    """Dot-cloud + mean-star chart for DL19 per-query ndcg@10.

    Algorithms are shown in _ALG_ORDER, plus an oracle-style `optimal` column.
    One highlighted query gets a dashed trace line so you can see how a
    representative query behaves across algorithms.
    """
    dataset     = payload.get("dataset", "dl19")
    metric_name = payload.get("metric_name", "ndcg@10")
    model       = payload.get("settings", {}).get("model", "")
    points      = payload.get("metrics", [])

    # ── Build ordered algorithm list ──────────────────────────────────────────
    by_alg       = {p.get("algorithm", ""): p for p in points}
    ordered_algs = [a for a in _ALG_ORDER if a in by_alg]
    ordered_algs += [a for a in by_alg if a not in ordered_algs]

    # ── Collect per-query scores averaged across seeds ─────────────────────────
    def _mean_per_qid(p: dict) -> dict[str, float]:
        pqs = p.get("per_query_scores", {})
        if not pqs:
            return {}
        totals: dict[str, list[float]] = {}
        for seed_dict in pqs.values():
            for qid, score in seed_dict.items():
                totals.setdefault(qid, []).append(float(score))
        return {qid: float(np.mean(vs)) for qid, vs in totals.items()}

    per_qid_by_alg = [_mean_per_qid(by_alg[a]) for a in ordered_algs]
    all_qids = sorted({qid for qid_dict in per_qid_by_alg for qid in qid_dict})
    optimal_by_qid = {
        qid: max(qid_dict.get(qid, float("-inf")) for qid_dict in per_qid_by_alg)
        for qid in all_qids
    }
    ordered_algs.append("optimal")
    per_qid_by_alg.append(optimal_by_qid)
    alg_labels = [_short_label(a) if a != "optimal" else "optimal" for a in ordered_algs]
    n_algs = len(ordered_algs)

    # Use family-consistent colors; keep quick and quick_3 in the purple family.
    alg_colors = [_DL19_BAR_COLORS.get(label, "tab:gray") for label in alg_labels]
    palette = dict(zip(alg_labels, alg_colors))

    # Long-form DataFrame for seaborn
    rows = [
        {"algorithm": alg_labels[i], "score": score, "qid": qid}
        for i, qid_dict in enumerate(per_qid_by_alg)
        for qid, score in qid_dict.items()
    ]
    if not rows:
        # Fallback: use aggregate means only
        rows = [{"algorithm": alg_labels[i], "score": float(by_alg[a].get("score_mean", 0)),
                 "qid": ""}
                for i, a in enumerate(ordered_algs)]
    df = pd.DataFrame(rows)

    means = np.array([
        float(df[df["algorithm"] == lbl]["score"].mean()) if lbl in df["algorithm"].values
        else float(by_alg[a].get("score_mean", 0.0))
        for lbl, a in zip(alg_labels, ordered_algs)
    ])
    # Use a fixed highlighted query when present.
    trace_qid = None
    if _TRACE_QID in all_qids:
        trace_qid = _TRACE_QID

    # ── Plot ──────────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(max(10, n_algs * 1.7), 6))

    # Individual query dots with jitter to create the cloud look.
    sns.stripplot(
        data=df, x="algorithm", y="score",
        order=alg_labels, hue="algorithm", hue_order=alg_labels,
        palette=palette, legend=False,
        size=6, alpha=0.35, jitter=0.18,
        linewidth=0.3, edgecolor="white",
        ax=ax, zorder=3,
    )

    # Mean stars
    for xi, (mean_val, color) in enumerate(zip(means, alg_colors)):
        ax.scatter(xi, mean_val, marker="*", color=color,
                   s=280, zorder=5, edgecolors="black", linewidths=0.6)
        ax.text(
            xi,
            min(mean_val + 0.035, 1.06),
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            zorder=6,
        )

    # Dashed trace line for one highlighted query.
    if trace_qid is not None:
        trace_x, trace_y = [], []
        for xi, qid_dict in enumerate(per_qid_by_alg):
            if trace_qid in qid_dict:
                trace_x.append(xi)
                trace_y.append(qid_dict[trace_qid])
        if len(trace_x) > 1:
            ax.plot(
                trace_x, trace_y,
                linestyle="--", color=_TRACE_COLOR,
                linewidth=1.8, marker="o", markersize=6, zorder=4,
            )

    ax.set_xticks(range(n_algs))
    ax.set_xticklabels(alg_labels, rotation=35, ha="right", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=13)
    ax.set_ylim(0.0, 1.08)
    ax.grid(True, axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.legend_.remove() if ax.legend_ else None

    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = f"_{model.replace('/', '-')}" if model else ""
    out_path = output_dir / f"{dataset}{model_tag}_{metric_name}_bar.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

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
    parser.add_argument(
        "--include-dl19-bar",
        action="store_true",
        help="Also generate the DL19 violin/bar companion plot.",
    )
    args = parser.parse_args()

    if args.input:
        output_dir = Path(args.output_dir) if args.output_dir else Path("figures")
        payload  = load_payload(Path(args.input))
        out_path = plot_payload(payload, output_dir, include_dl19_bar=args.include_dl19_bar)
        print(f"Wrote figure to {out_path}")
    else:
        input_dir  = Path(args.input_dir)
        output_dir = Path(args.output_dir) if args.output_dir else input_dir
        json_files = sorted(input_dir.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {input_dir}")
            return
        for json_path in json_files:
            try:
                payload  = load_payload(json_path)
                out_path = plot_payload(payload, output_dir, include_dl19_bar=args.include_dl19_bar)
                print(f"Wrote figure to {out_path}")
            except Exception as exc:
                print(f"Skipped {json_path.name}: {exc}")


if __name__ == "__main__":
    main()
