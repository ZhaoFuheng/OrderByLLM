"""
Cost-quality scatter plot WITH per-query confidence intervals.

Same styling as test/plot_experiment.py (marker = algorithm family, optimizer
overlay: Self-Cons=diamond, Judge; ext_merge=X; _6/_8 batch sizes dropped), but the
y error bars are a 80% confidence interval computed ACROSS QUERIES (from
per_query_scores / per_movie_scores) rather than the across-seed std. Benchmarks
with many queries (dl19, dl20, hellaswag, nfcorpus, sembench) get a meaningful
interval; single-item datasets fall back to score_std.

Works for both dev and test datasets (optimizer dots only appear where an
optimizer_<model>.json exists in the same dir).

Usage:
    python plot_experiment_ci.py --input test/dl20/results_llama3.1-70b.json
    python plot_experiment_ci.py --input-dir test/dl20 --output-dir figures_with_confidence_interval/testFigures
"""
import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from scipy.optimize import curve_fit


# ── Per-dataset y-axis limits ────────────────────────────────────────────────

_YLIM = {
    "nba":            (0.40, 1.00),
    "dl19":           (0.40, 0.90),
    "dl20":           (0.40, 0.80),
    "population":     (0.95, 1.00),
    "sembench_movie": (0.60, 1.00),
}
_YLIM_DEFAULT = (0.10, 1.00)

# Batch sizes 6/8 are no longer studied; old results_*.json may still carry them.
_EXCLUDED_ALGS = {
    "external_merge_sort_6", "external_merge_sort_8",
    "external_bubble_sort_6", "external_bubble_sort_8",
}

_CI_Z = 1.2816  # 80% normal-approx confidence interval

# Datasets that get a dashed log-fit line across all algorithm points (dev plots).
_FIT_DATASETS = {"nba", "dl19"}


def _log_fit(xs: np.ndarray, ys: np.ndarray):
    """Fit y = A*log(x + x0) + B. Returns (popt, r2) or None on failure."""
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
    "merge":        "X",
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
    fam         = _family(alg_name)
    marker      = _FAMILY_MARKER[fam]
    color       = _FAMILY_COLOR[fam]
    with_search = "with_search" in alg_name.lower()
    edgecolor   = "black" if with_search else color
    edgewidth   = 2.0     if with_search else 0.8
    return marker, color, edgecolor, edgewidth, _short_label(alg_name)


# ── Per-query mean + confidence interval ─────────────────────────────────────

def _per_query_vals(point: dict) -> list[float]:
    """Per-query scores for one algorithm (averaged across seeds). Handles both the
    experiment shape {seed: {qid: score}} and the flat optimizer shape {qid: score}.
    Returns [] when there is no per-query data."""
    raw = None
    for k in ("per_query_scores", "per_movie_scores"):
        v = point.get(k)
        if isinstance(v, dict) and v:
            raw = v
            break
    if raw is None:
        return []
    by_item: dict[str, list[float]] = {}
    if all(isinstance(v, dict) for v in raw.values()):
        for seed_scores in raw.values():
            for qid, s in seed_scores.items():
                by_item.setdefault(qid, []).append(float(s))
    else:
        for qid, s in raw.items():
            try:
                by_item.setdefault(qid, []).append(float(s))
            except (TypeError, ValueError):
                pass
    return [sum(v) / len(v) for v in by_item.values() if v]


def _mean_ci(point: dict, z: float = _CI_Z) -> tuple[float, float]:
    """Return (mean, half_width) — a z*sem confidence interval across queries.
    Falls back to (score_mean, score_std) when there is no per-query data."""
    vals = _per_query_vals(point)
    if len(vals) >= 2:
        mean = statistics.mean(vals)
        half = z * statistics.stdev(vals) / math.sqrt(len(vals))
        return mean, half
    if len(vals) == 1:
        return vals[0], 0.0
    return float(point.get("score_mean", point.get("score", 0.0))), float(point.get("score_std", 0.0))


# ── Optimizer overlay ─────────────────────────────────────────────────────────

_OPTIMIZER_MARKER = {
    "rrf_ensemble": "D",
    "llm_judge":    "P",
}

_OPTIMIZER_COLOR = {
    "rrf_ensemble": "tab:orange",
    "llm_judge":    "tab:cyan",
}


def _load_optimizer_data(results_dir: Path, model: str) -> list[dict]:
    dots = []
    p = results_dir / f"optimizer_{model.replace('/', '-')}.json"
    if not p.exists():
        return dots
    data = json.loads(p.read_text(encoding="utf-8"))
    by_model = data.get("results_by_model", {}).get(model, {})
    for policy, budgets in by_model.items():
        if policy == "ideal":
            continue
        for budget_str, rec in budgets.items():
            score = rec.get("score_mean", None)
            cost = rec.get("total_ranking_cost", rec.get("ranking_cost", None))
            if cost is None:
                for sr in rec.get("seed_results", []):
                    c = sr.get("total_ranking_cost", sr.get("ranking_cost", None))
                    if c is not None:
                        cost = c
                        break
            # Optimizer's TOTAL cost = ranking cost + the optimizer's own sampling
            # (optimization) overhead, so it's comparable to a standalone algorithm's
            # full-run price.
            if cost is not None:
                cost += rec.get("total_optimization_cost", rec.get("optimization_cost", 0.0)) or 0.0
            if score is not None and cost is not None:
                # 80% CI across queries for this optimizer point (same as algorithms).
                _mean, ci = _mean_ci(rec)
                dots.append({"policy": policy, "budget": budget_str,
                             "score": score, "cost": cost, "ci": ci,
                             "vals": _per_query_vals(rec)})
    return dots


def _draw_box(ax, x, vals, color, width):
    """Draw a translucent, family-colored box plot of `vals` at position x."""
    bp = ax.boxplot(
        [vals], positions=[x], widths=[width], patch_artist=True,
        manage_ticks=False, zorder=2,
        flierprops=dict(marker=".", markersize=3, markerfacecolor=color,
                        markeredgecolor=color, alpha=0.35),
    )
    for b in bp["boxes"]:
        b.set(facecolor=color, alpha=0.22, edgecolor=color, linewidth=1.0)
    for w in bp["whiskers"]:
        w.set(color=color, alpha=0.7, linewidth=1.0)
    for c in bp["caps"]:
        c.set(color=color, alpha=0.7, linewidth=1.0)
    for m in bp["medians"]:
        m.set(color="black", linewidth=1.2)
    return bp


# ── Main plot function ────────────────────────────────────────────────────────

def plot_payload(payload: dict, output_dir: Path, results_dir: Path | None = None,
                 spread: str = "ci") -> Path:
    dataset     = payload.get("dataset", "unknown")
    metric_name = payload.get("metric_name", "kendall_tau")
    model       = payload.get("settings", {}).get("model", "")
    points      = payload.get("metrics", [])
    points      = [p for p in points if str(p.get("algorithm", "")) not in _EXCLUDED_ALGS]
    if not points:
        raise ValueError("No metrics in payload.")

    xs   = np.array([float(p.get("price", 0.0)) for p in points])
    mc   = [_mean_ci(p) for p in points]
    ys   = np.array([m for m, _ in mc])
    yerrs = np.array([h for _, h in mc])
    algs = [str(p.get("algorithm", f"alg_{i}")) for i, p in enumerate(points)]

    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(9, 6))

    # Box width in x (data) units — a small fraction of the price range.
    x_span = float(xs.max() - xs.min()) or float(xs.max()) or 1.0
    box_w  = max(x_span * 0.025, 1e-3)

    for x, y, yerr, alg, point in zip(xs, ys, yerrs, algs, points):
        marker, color, edgecolor, edgewidth, _ = _style(alg)
        if spread == "box":
            vals = _per_query_vals(point)
            if len(vals) >= 2:
                _draw_box(ax, x, vals, color, box_w)
            # keep the family marker (shape/colour identity) at the mean
            ax.plot(x, y, marker=marker, color=color,
                    markeredgecolor=edgecolor, markeredgewidth=edgewidth,
                    markersize=10, linestyle="None", zorder=4)
        else:
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

    # ── Dashed log-fit across all algorithm points (dev datasets) ────────────
    fit_entries = []
    if dataset in _FIT_DATASETS:
        fit = _log_fit(xs, ys)
        if fit is not None:
            popt, r2 = fit
            A, B, x0 = popt
            if A > 0 and r2 >= 0.3:
                x_fit = np.linspace(xs.min(), xs.max(), 500)
                y_fit = A * np.log(x_fit + x0) + B
                ax.plot(x_fit, y_fit, "r--", linewidth=1.8, zorder=2)
                from matplotlib.lines import Line2D as _L
                fit_entries = [_L([0], [0], color="r", linestyle="--", linewidth=1.8,
                                  label=f"log fit (R²={r2:.3f})")]

    # ── Optimizer dots (rrf_ensemble→Self-Cons, llm_judge→Judge) ─────────────
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
        if spread == "box" and len(dot.get("vals", [])) >= 2:
            _draw_box(ax, dot["cost"], dot["vals"], color, box_w)
            ax.plot(dot["cost"], dot["score"], marker=marker, color=color,
                    markeredgecolor="black", markeredgewidth=1.2,
                    markersize=13, linestyle="None", zorder=5)
        else:
            ax.errorbar(
                dot["cost"], dot["score"],
                yerr=dot.get("ci") or None,
                fmt=marker, color=color,
                markeredgecolor="black", markeredgewidth=1.2,
                markersize=13, capsize=4, elinewidth=1.2, ecolor=color,
                zorder=5, linestyle="None",
            )
        if policy not in ("rrf_ensemble", "llm_judge"):
            label = f'{policy} ${dot["budget"]}'
            opt_texts.append(ax.text(dot["cost"], dot["score"], label, fontsize=10, fontstyle="italic"))

    for curve_policy, curve_style in [("rrf_ensemble", "--"), ("llm_judge", "--")]:
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
            opt_texts, x=all_xs, y=all_ys, ax=ax,
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.6, shrinkA=10, shrinkB=4),
            expand=(1.5, 2.0), iter_lim=300,
        )

    # ── Legend ────────────────────────────────────────────────────────────────
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
        "rrf_ensemble": "Self-Cons",
        "llm_judge": "Judge",
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
    _spread_label = "per-query box" if spread == "box" else "80% CI across queries"
    ax.set_ylabel(f"{metric_name} ({_spread_label})", fontsize=14)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)

    all_handles = family_entries + wiki_entries + optimizer_entries + fit_entries
    # Cap columns so the legend never overflows the fixed canvas width (long labels
    # like ext_bubble_4 at >4 columns run off the right edge); extra entries wrap to
    # a third row, which the reserved top margin accommodates.
    ncol = min(4, math.ceil(len(all_handles) / 2))
    ax.legend(
        handles=all_handles, title="Algorithm", title_fontsize=12,
        loc="lower center", bbox_to_anchor=(0.5, 1.02), bbox_transform=ax.transAxes,
        ncol=ncol, fontsize=12, framealpha=0.9, borderpad=0.4,
        columnspacing=1.2, handletextpad=0.5,
    )

    opt_y   = np.array([d["score"] for d in opt_dots]) if opt_dots else np.array([])
    opt_ci  = np.array([d.get("ci", 0.0) or 0.0 for d in opt_dots]) if opt_dots else np.array([])
    all_x = np.concatenate([xs, np.array([d["cost"] for d in opt_dots])]) if opt_dots else xs
    if spread == "box":
        # Include the full per-query spread so boxes/whiskers/fliers aren't clipped.
        pq = [v for p in points for v in _per_query_vals(p)]
        pq += [v for d in opt_dots for v in d.get("vals", [])]
        lo0 = min(pq) if pq else float(ys.min())
        hi0 = max(pq) if pq else float(ys.max())
        all_y_lo = np.array([lo0, float((ys).min()), float(opt_y.min()) if opt_dots else lo0])
        all_y_hi = np.array([hi0, float((ys).max()), float(opt_y.max()) if opt_dots else hi0])
    else:
        # Include the error-bar extent (algorithms AND optimizer) so CIs aren't clipped.
        ys_lo = ys - yerrs
        ys_hi = ys + yerrs
        all_y_lo = np.concatenate([ys_lo, opt_y - opt_ci]) if opt_dots else ys_lo
        all_y_hi = np.concatenate([ys_hi, opt_y + opt_ci]) if opt_dots else ys_hi

    x_pad = max(all_x) * 0.15
    ax.set_xlim(min(-0.05, -x_pad), max(all_x) + x_pad)
    y_min_preset, y_max_preset = _YLIM.get(dataset, _YLIM_DEFAULT)
    y_pad = 0.03
    y_min = min(y_min_preset, float(all_y_lo.min()) - y_pad)
    y_max = min(1.05, max(y_max_preset, float(all_y_hi.max()) + y_pad))
    ax.set_ylim(y_min, y_max)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = f"_{model.replace('/', '-')}" if model else ""
    suffix = "box" if spread == "box" else "ci"
    # Figure filenames must contain no '_' or '-' (LaTeX-safe): strip both.
    stem = f"{dataset}{model_tag}_{metric_name}_{suffix}".replace("_", "").replace("-", "")
    out_path = output_dir / f"{stem}.png"
    # Fixed margins + fixed 9x6 canvas so EVERY figure has identical output
    # dimensions (do NOT use bbox_inches="tight", which crops to per-figure content
    # and makes datasets with wider labels/legends produce larger images).
    fig.subplots_adjust(left=0.12, right=0.97, top=0.77, bottom=0.11)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).parent


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute() or p.exists():
        return p
    candidate = _SCRIPT_DIR / p
    return candidate if candidate.exists() else p


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description="Cost-quality plots with per-query 80% confidence intervals."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Single experiment JSON file path.")
    group.add_argument("--input-dir", help="Directory of experiment JSONs; one figure per file.")
    parser.add_argument("--output-dir", default=None, help="Directory for figure outputs.")
    parser.add_argument("--spread", choices=["ci", "box"], default="ci",
                        help="Show the per-query spread as a 80%% CI error bar (ci) "
                             "or a box plot at each point (box). Default: ci.")
    args = parser.parse_args()

    if args.input:
        input_path = _resolve(args.input)
        output_dir = _resolve(args.output_dir) if args.output_dir else Path("figures_with_confidence_interval")
        payload    = load_payload(input_path)
        out_path   = plot_payload(payload, output_dir, results_dir=input_path.parent, spread=args.spread)
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
                out_path = plot_payload(payload, output_dir, results_dir=input_dir, spread=args.spread)
                print(f"Wrote figure to {out_path}")
            except Exception as exc:
                print(f"Skipped {json_path.name}: {exc}")


if __name__ == "__main__":
    main()
