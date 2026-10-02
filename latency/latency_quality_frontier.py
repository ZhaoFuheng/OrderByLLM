#!/usr/bin/env python3
"""Latency vs quality frontier for DL20 on Haiku 4.5.

x-axis = estimated wall-clock latency per query (cache-only replay + calibrated
formula, concurrency 100); y-axis = quality (ndcg@10). Plots each standalone
algorithm as a labeled point and the optimizer (rrf_ensemble self-consistency and
llm_judge) as a curve over the budget sweep {14,28,42,56} = run_all.py DL20/Haiku.

Latency is MEASURED here (subset of queries; per-query latency is low-variance so a
subset avg tracks the full 54-query avg). Quality is read from the OFFICIAL 54-query
results so the y-values match the paper:
  standalone -> test/dl20/results_claude-haiku-4-5.json  (score_mean)
  optimizer  -> test/dl20/optimizer_claude-haiku-4-5.json (per policy/budget score_mean)

The optimizer's per-query budget is total/54 (as in the official run), so algorithm
choices -- and thus cache hits and latency -- reproduce the official run exactly.

Usage:
    python latency/latency_quality_frontier.py --queries 10 --speedup 30
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import estimate_dl20_latency as E  # noqa: E402  (reuse Sim, patches, dl20_prepared, etc.)
from order_by.replay import NoNetworkClient, install_memcache  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
STANDALONE = E.STANDALONE
_SHORT = {"pointwise": "point", "external_pointwise_4": "ext_point_4",
          "external_merge_sort_4": "ext_merge_4", "external_bubble_sort_4": "ext_bubble_4",
          "quick_sort": "quick", "quick_sort3": "quick_3"}


def _load_quality(model):
    r = json.loads((PROJECT_ROOT / "test" / "dl20" / f"results_{model}.json").read_text())
    standalone_q = {m["algorithm"]: m["score_mean"] for m in r["metrics"]}
    o = json.loads((PROJECT_ROOT / "test" / "dl20" / f"optimizer_{model}.json").read_text())
    by = o["results_by_model"][model]
    opt_q = {pol: {b: rec.get("score_mean") for b, rec in budgets.items()}
             for pol, budgets in by.items()}
    return standalone_q, opt_q


async def measure(args, client):
    prepared = E.dl20_prepared(args)
    n_full = len(prepared)                 # 54, for the official per-query budget
    prepared = prepared[: args.queries]

    # ---- standalone latency ----
    standalone_lat = {}
    print("Standalone latency (per query):")
    for name in STANDALONE:
        E.SIM.reset()
        lats = []
        for qid, query, top in prepared:
            t0 = time.perf_counter()
            await E.RE.passage_algorithm(name, top, query, client, args.model)
            lats.append((time.perf_counter() - t0) * E.SIM.speedup)
        standalone_lat[name] = round(statistics.mean(lats), 2)
        print(f"  {name:<24s} {standalone_lat[name]:>8.2f}s")

    # ---- optimizer latency across budgets x policies ----
    budgets = [float(b) for b in args.budgets.split(",")]
    policies = args.policies.split(",")
    opt_lat = {p: {} for p in policies}
    for policy in policies:
        print(f"\nOptimizer latency [{policy}]:")
        for budget in budgets:
            per_query_budget = budget / n_full   # official per-query budget (total/54)
            oargs = SimpleNamespace(
                model=args.model, judge_model=args.model, judge_client=None,
                proxy_policy=policy, total_ranking_budget=budget,
                sample_size=20, ext_point_batch=8, rrf_k=60, ensemble_max_lists=0,
                dl20_run_file=args.dl20_run_file, hit_depth=args.hit_depth)
            E.SIM.reset()
            lats = []
            for qid, query, top in prepared:
                o = E.RO._build_passage_optimizer(query, top[:], client, oargs, per_query_budget)
                t0 = time.perf_counter()
                await o.physical_order_by_impl()
                lats.append((time.perf_counter() - t0) * E.SIM.speedup)
            bkey = f"{budget:.4f}".rstrip("0").rstrip(".")
            opt_lat[policy][bkey] = round(statistics.mean(lats), 2)
            print(f"  budget={bkey:<4s} {opt_lat[policy][bkey]:>8.2f}s/query")
    return standalone_lat, opt_lat


# ── House style (matches test/plot_experiment.py) ────────────────────────────
_FAMILY_COLOR = {"pointwise": "tab:blue", "ext_pointwise": "tab:green",
                 "quick": "tab:purple", "quick3": "tab:purple",
                 "bubble": "tab:red", "merge": "tab:brown"}
_FAMILY_MARKER = {"pointwise": "o", "ext_pointwise": "^", "quick": "s",
                  "quick3": "h", "bubble": "*", "merge": "X"}
_FAMILY_LEGEND = {"pointwise": "point", "ext_pointwise": "ext_point_4",
                  "quick": "quick", "quick3": "quick_3",
                  "bubble": "ext_bubble_4", "merge": "ext_merge_4"}
_ALG_FAMILY = {"pointwise": "pointwise", "external_pointwise_4": "ext_pointwise",
               "external_merge_sort_4": "merge", "external_bubble_sort_4": "bubble",
               "quick_sort": "quick", "quick_sort3": "quick3"}
_OPTIMIZER_MARKER = {"rrf_ensemble": "D", "llm_judge": "P"}
_OPTIMIZER_COLOR = {"rrf_ensemble": "tab:orange", "llm_judge": "tab:cyan"}
_OPTIMIZER_LEGEND = {"rrf_ensemble": "Self-Cons", "llm_judge": "Judge"}


def plot(standalone_lat, standalone_q, opt_lat, opt_q, model, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import math
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(9, 6))

    # ── standalone algorithm dots ────────────────────────────────────────────
    present_families = []
    for name in STANDALONE:
        if name not in standalone_lat or standalone_q.get(name) is None:
            continue
        fam = _ALG_FAMILY[name]
        present_families.append(fam)
        ax.plot(standalone_lat[name], standalone_q[name],
                marker=_FAMILY_MARKER[fam], color=_FAMILY_COLOR[fam],
                markeredgecolor=_FAMILY_COLOR[fam], markeredgewidth=0.8,
                markersize=10, linestyle="None", zorder=3)

    # ── optimizer dots + dashed budget curve ─────────────────────────────────
    present_policies = []
    for policy, lat_by_b in opt_lat.items():
        pts = sorted((lat, opt_q.get(policy, {}).get(b))
                     for b, lat in lat_by_b.items() if opt_q.get(policy, {}).get(b) is not None)
        if not pts:
            continue
        present_policies.append(policy)
        cx = [p[0] for p in pts]; cy = [p[1] for p in pts]
        ax.plot(cx, cy, color=_OPTIMIZER_COLOR[policy], linestyle="--",
                linewidth=2.5, zorder=6)
        for x, y in pts:
            ax.plot(x, y, marker=_OPTIMIZER_MARKER[policy], color=_OPTIMIZER_COLOR[policy],
                    markeredgecolor="black", markeredgewidth=1.2, markersize=13,
                    linestyle="None", zorder=7)

    # ── legend (family shapes + optimizer), title on top like the other figs ──
    seen = set()
    family_entries = [
        Line2D([0], [0], marker=_FAMILY_MARKER[f], color=_FAMILY_COLOR[f],
               linestyle="None", markersize=9, label=_FAMILY_LEGEND[f])
        for f in ["pointwise", "ext_pointwise", "quick", "quick3", "bubble", "merge"]
        if f in present_families and not (f in seen or seen.add(f))
    ]
    optimizer_entries = [
        Line2D([0], [0], marker=_OPTIMIZER_MARKER[p], color=_OPTIMIZER_COLOR[p],
               markeredgecolor="black", markeredgewidth=1.2, linestyle="--",
               linewidth=2, markersize=10, label=_OPTIMIZER_LEGEND[p])
        for p in ["rrf_ensemble", "llm_judge"] if p in present_policies
    ]
    all_handles = family_entries + optimizer_entries

    ax.set_xscale("log")   # latency spans ~230x (5s -> 1130s); log keeps points legible
    ax.set_xlabel("average query latency (s)", fontsize=14)
    ax.set_ylabel("ndcg@10", fontsize=14)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(handles=all_handles, title="Algorithm", title_fontsize=13,
              loc="lower center", bbox_to_anchor=(0.5, 1.02), bbox_transform=ax.transAxes,
              ncol=math.ceil(len(all_handles) / 2), fontsize=13,
              framealpha=0.9, borderpad=0.5)

    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved plot -> {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--speedup", type=float, default=30.0)
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--budgets", default="14,28,42,56")
    ap.add_argument("--policies", default="rrf_ensemble,llm_judge")
    ap.add_argument("--latency-model", default=None)
    ap.add_argument("--hit-depth", type=int, default=100)
    ap.add_argument("--dl20-run-file", default="data/run.msmarco-v1-passage.bm25-default.dl20.txt")
    ap.add_argument("--replot", action="store_true",
                    help="Skip measurement; re-render the plot from the saved JSON.")
    args = ap.parse_args()

    data_path = OUT_DIR / f"latency_quality_{args.model}.json"
    # LaTeX-safe figure name (no '_' or '-'), matching the other figures.
    png_path = OUT_DIR / (f"dl20latencyquality{args.model}".replace("_", "").replace("-", "") + ".png")

    if args.replot:
        payload = json.loads(data_path.read_text())
        standalone_lat = {n: v["latency_s"] for n, v in payload["standalone"].items() if v["latency_s"] is not None}
        standalone_q = {n: v["ndcg10"] for n, v in payload["standalone"].items()}
        opt_lat = {p: {b: v["latency_s"] for b, v in bs.items()} for p, bs in payload["optimizer"].items()}
        opt_q = {p: {b: v["ndcg10"] for b, v in bs.items()} for p, bs in payload["optimizer"].items()}
        plot(standalone_lat, standalone_q, opt_lat, opt_q, args.model, png_path)
        return

    mpath = Path(args.latency_model) if args.latency_model else OUT_DIR / f"latency_model_{args.model}.json"
    fit = json.loads(mpath.read_text())["fit"]
    eff = fit.get("effective", fit["split"])
    E.SIM = E.Sim(E.LatencyModel(eff), args.concurrency, args.speedup)

    install_memcache()
    client = NoNetworkClient()
    E.install_patches()

    print(f"Latency model: {eff.get('chosen','?')} (R²={eff.get('r2','?')}), "
          f"concurrency={args.concurrency}, speedup={args.speedup}, queries={args.queries}\n")

    standalone_lat, opt_lat = asyncio.run(measure(args, client))
    standalone_q, opt_q = _load_quality(args.model)

    payload = {"model": args.model, "queries": args.queries, "concurrency": args.concurrency,
               "standalone": {n: {"latency_s": standalone_lat.get(n), "ndcg10": standalone_q.get(n)}
                              for n in STANDALONE},
               "optimizer": {p: {b: {"latency_s": opt_lat[p].get(b), "ndcg10": opt_q.get(p, {}).get(b)}
                                 for b in opt_lat[p]} for p in opt_lat}}
    data_path.write_text(json.dumps(payload, indent=2))
    plot(standalone_lat, standalone_q, opt_lat, opt_q, args.model, png_path)
    print(f"saved data -> {data_path}")


if __name__ == "__main__":
    main()
