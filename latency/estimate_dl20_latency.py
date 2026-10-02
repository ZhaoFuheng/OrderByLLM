#!/usr/bin/env python3
"""Steps 2 & 3 of the latency study: estimate DL20 wall-clock latency on Haiku
under a fixed concurrency budget, using the calibrated per-request latency model.

Everything is cached, so a real run has ~zero latency. We REPLAY the cached run
but inject each request's *estimated* latency (from latency/latency_model_<model>.json)
as an asyncio.sleep held under a GLOBAL Semaphore(concurrency). Because the sorting
algorithms are genuinely async (asyncio.gather for independent work, sequential
await for dependent work), the real event loop reproduces the exact parallelism and
dependency structure -- the measured wall-clock IS the estimated latency.

Injection is at the atomic LLM functions (each returns (..., in_tok, out_tok)):
  pair_comparison.Pair_Comparison_Key.compare   (quick sort)
  pointwise.Pointwise_Key.value / PointwiseRelevanceKey.value  (pointwise)
  pair_comparison.external_comparisons          (merge / bubble sort)
  pointwise.external_values                     (external pointwise)
Cache-hit returns carry the real token counts, so no API calls are made.

  Part 2 (--part standalone): for each standalone algorithm, run all DL20 queries
  concurrently under Semaphore(concurrency) and report the total wall-clock to rank
  all queries plus the average per query.

  Part 3 (--part optimizer): run the optimizer per query (policy rrf_ensemble by
  default, so all work flows through the patched sorting/pointwise functions and is
  fully timed -- an llm_judge policy's single selection call is NOT timed) and report
  the average latency per query.

SPEEDUP scales every injected sleep by 1/SPEEDUP and the reported latency back up, to
shorten real runtime. Keep it modest (default 1 = real-magnitude sleeps, most
accurate); large SPEEDUP inflates estimates because fixed per-request overhead
(cache reads, scheduling) stops being negligible.

Usage:
  python latency/estimate_dl20_latency.py --part standalone
  python latency/estimate_dl20_latency.py --part optimizer --optimizer-queries 10 --budget 56
  python latency/estimate_dl20_latency.py --part both --concurrency 100
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
sys.path.insert(0, str(PROJECT_ROOT / "test"))

import run_experiment as RE   # noqa: E402
import run_optimizer as RO    # noqa: E402
import order_by.pair_comparison as pc  # noqa: E402
import order_by.pointwise as pw        # noqa: E402
import order_by.sorting as srt         # noqa: E402
import order_by.optimizer as opt       # noqa: E402
from benchmarks import load_dl20  # noqa: E402
from order_by.clients import PROVIDERS, build_client  # noqa: E402
from order_by.replay import NoNetworkClient, install_memcache  # noqa: E402
from order_by.utils import load_env_file  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
STANDALONE = ["pointwise", "external_pointwise_4", "external_merge_sort_4",
              "external_bubble_sort_4", "quick_sort", "quick_sort3"]


class LatencyModel:
    def __init__(self, split, min_latency=0.05):
        self.a = split["intercept"]
        self.b_out = split["per_output_token"]
        self.b_in = split["per_input_token"]
        self.min = min_latency

    def __call__(self, in_tok, out_tok):
        return max(self.min, self.a + self.b_out * out_tok + self.b_in * in_tok)


class Sim:
    """Global concurrency gate + per-request recorder + sleep injector."""
    def __init__(self, latency_fn, concurrency, speedup):
        self.latency_fn = latency_fn
        self.sem = asyncio.Semaphore(concurrency)
        self.speedup = speedup
        self.reset()

    def reset(self):
        self.n = 0
        self.sum_lat = 0.0
        self.sum_in = 0
        self.sum_out = 0
        self.max_lat = 0.0

    async def gate(self, in_tok, out_tok):
        lat = self.latency_fn(in_tok, out_tok)
        self.n += 1
        self.sum_lat += lat
        self.sum_in += in_tok
        self.sum_out += out_tok
        self.max_lat = max(self.max_lat, lat)
        async with self.sem:
            await asyncio.sleep(lat / self.speedup)


SIM: Sim = None  # set in main


def _timed(orig):
    """Wrap an atomic LLM async fn so it sleeps by estimated latency after returning
    its (cached) result. Token counts are the last two elements of the return tuple."""
    async def wrapper(*a, **k):
        res = await orig(*a, **k)
        try:
            in_tok, out_tok = int(res[-2]), int(res[-1])
        except (TypeError, ValueError, IndexError):
            return res
        if in_tok or out_tok:
            await SIM.gate(in_tok, out_tok)
        return res
    return wrapper


def install_patches():
    """Monkeypatch the atomic LLM functions in every namespace that references them,
    so every algorithm launched through RE / RO / the optimizer is timed."""
    pc.Pair_Comparison_Key.compare = _timed(pc.Pair_Comparison_Key.compare)
    pw.Pointwise_Key.value = _timed(pw.Pointwise_Key.value)
    pw.PointwiseRelevanceKey.value = _timed(pw.PointwiseRelevanceKey.value)
    # LLM-judge selection call goes through _call_llm (not the sorting atoms);
    # process_llm_judge_item returns (id, in_tok, out_tok) -> time it too.
    opt.OrderByOptimizer.process_llm_judge_item = _timed(opt.OrderByOptimizer.process_llm_judge_item)

    timed_ext_cmp = _timed(pc.external_comparisons)
    timed_ext_val = _timed(pw.external_values)
    for mod in (pc, pw, srt, opt, RE, RO):
        if getattr(mod, "external_comparisons", None) is not None:
            mod.external_comparisons = timed_ext_cmp
        if getattr(mod, "external_values", None) is not None:
            mod.external_values = timed_ext_val


def dl20_prepared(args):
    """DL20 queries with the canonical seed-0 per-query shuffle (matches cache keys)."""
    return load_dl20(RE._resolve(args.dl20_run_file), args.hit_depth).shuffled(0)


# ── Part 2: standalone algorithms ────────────────────────────────────────────
async def run_standalone(args, client):
    """Run each query ONE AT A TIME (no cross-query concurrency); the concurrency
    budget applies to the parallel requests WITHIN a single query. Per-query latency
    = wall-clock to rank that query's passages; total = sum over queries."""
    prepared = dl20_prepared(args)
    if args.max_queries:
        prepared = prepared[: args.max_queries]
    algos = args.algorithms.split(",") if args.algorithms else STANDALONE
    results = {}
    for name in algos:
        SIM.reset()
        per_query_lat = []
        for qid, query, top in prepared:
            t0 = time.perf_counter()
            await RE.passage_algorithm(name, top, query, client, args.model)
            per_query_lat.append((time.perf_counter() - t0) * SIM.speedup)
        nq = len(prepared)
        total = sum(per_query_lat)
        results[name] = {
            "total_latency_s": round(total, 2),                 # sum over all queries, one at a time
            "avg_latency_per_query_s": round(total / nq, 2),
            "median_latency_per_query_s": round(statistics.median(per_query_lat), 2),
            "min_query_s": round(min(per_query_lat), 2),
            "max_query_s": round(max(per_query_lat), 2),
            "n_requests": SIM.n,
            "avg_requests_per_query": round(SIM.n / nq, 1),
            "avg_request_latency_s": round(SIM.sum_lat / SIM.n, 3) if SIM.n else 0.0,
        }
        r = results[name]
        print(f"  {name:<24s} total={r['total_latency_s']:>8.1f}s  "
              f"per_query avg={r['avg_latency_per_query_s']:>6.2f}s "
              f"med={r['median_latency_per_query_s']:>6.2f}s  "
              f"reqs/q={r['avg_requests_per_query']}", flush=True)
    return {"n_queries": len(prepared), "concurrency": args.concurrency,
            "note": "queries run sequentially; concurrency applies within a query",
            "algorithms": results}


# ── Part 3: optimizer ─────────────────────────────────────────────────────────
async def run_optimizer_latency(args, client):
    all_queries = dl20_prepared(args)
    prepared = all_queries[: args.max_queries] if args.max_queries else all_queries
    if args.optimizer_queries:
        prepared = prepared[: args.optimizer_queries]
    oargs = SimpleNamespace(
        model=args.model, judge_model=args.model, judge_client=None,
        proxy_policy=args.proxy_policy, total_ranking_budget=args.budget,
        sample_size=20, ext_point_batch=8, rrf_k=60, ensemble_max_lists=0,
        dl20_run_file=args.dl20_run_file, hit_depth=args.hit_depth,
    )
    per_query_budget = args.budget / len(all_queries)   # as in the official run: total / all queries
    per_query = []
    for qid, query, top in prepared:
        SIM.reset()
        o = RO._build_passage_optimizer(query, top[:], client, oargs, per_query_budget)
        t0 = time.perf_counter()
        (sorted_data, num_calls, in_t, out_t), chosen_alg, _, opt_cost = await o.physical_order_by_impl()
        wall = (time.perf_counter() - t0) * SIM.speedup
        per_query.append({"qid": qid, "latency_s": round(wall, 2), "chosen_alg": chosen_alg,
                          "n_requests": SIM.n})
        print(f"  qid={qid:<10s} latency={wall:>6.2f}s  chosen={chosen_alg:<24s} "
              f"reqs={SIM.n}", flush=True)
    lats = [q["latency_s"] for q in per_query]
    return {
        "n_queries": len(per_query), "concurrency": args.concurrency,
        "proxy_policy": args.proxy_policy, "budget": args.budget,
        "avg_latency_per_query_s": round(statistics.mean(lats), 2) if lats else 0.0,
        "median_latency_per_query_s": round(statistics.median(lats), 2) if lats else 0.0,
        "min_s": round(min(lats), 2) if lats else 0.0,
        "max_s": round(max(lats), 2) if lats else 0.0,
        "per_query": per_query,
    }


def main():
    global SIM
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["standalone", "optimizer", "both"], default="both")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--provider", default="anthropic", choices=PROVIDERS,
                    help="Provider used only with --allow-network (default: anthropic).")
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--speedup", type=float, default=1.0,
                    help="Scale injected sleeps by 1/speedup (default 1 = real magnitude, most accurate).")
    ap.add_argument("--latency-model", default=None,
                    help="Path to latency_model_<model>.json (default: latency/latency_model_<model>.json).")
    ap.add_argument("--algorithms", default=None,
                    help="Comma-separated standalone algorithms (default: all six).")
    ap.add_argument("--max-queries", type=int, default=None,
                    help="Cap DL20 queries used for the standalone part (default: all 54).")
    ap.add_argument("--optimizer-queries", type=int, default=10,
                    help="How many queries to time the optimizer on (default: 10).")
    ap.add_argument("--proxy-policy", default="rrf_ensemble")
    ap.add_argument("--budget", type=float, default=56.0,
                    help="Total DL20 optimizer budget (default 56 = run_all.py DL20/Haiku max).")
    ap.add_argument("--hit-depth", type=int, default=100)
    ap.add_argument("--dl20-run-file", default="data/run.msmarco-v1-passage.bm25-default.dl20.txt")
    ap.add_argument("--allow-network", action="store_true",
                    help="Permit real API calls (default: cache-only; misses abort loudly).")
    args = ap.parse_args()

    mpath = Path(args.latency_model) if args.latency_model else OUT_DIR / f"latency_model_{args.model}.json"
    if not mpath.exists():
        print(f"ERROR: latency model {mpath} not found. Run calibrate_latency.py first.")
        sys.exit(1)
    model_json = json.loads(mpath.read_text())
    # Prefer the calibration's chosen "effective" model; fall back to "split".
    eff = model_json["fit"].get("effective", model_json["fit"]["split"])
    latency_fn = LatencyModel(eff)
    SIM = Sim(latency_fn, args.concurrency, args.speedup)

    # Cache-only guarantee: route reads through RAM (no LRU write churn) and refuse
    # all network calls. Coverage was verified with check_cache_coverage.py; a genuine
    # miss here raises RealCall and aborts loudly rather than silently spending.
    if args.allow_network:
        load_env_file(PROJECT_ROOT / ".env")
        client = build_client(args.provider)
    else:
        install_memcache()
        client = NoNetworkClient()
    install_patches()

    print(f"Latency model: effective='{eff.get('chosen','split')}' R²={eff.get('r2','?')}")
    print(f"  latency ≈ {latency_fn.a:.3f}s + {latency_fn.b_out*1000:.4f}ms*out "
          f"+ {latency_fn.b_in*1000:.4f}ms*in  | concurrency={args.concurrency} speedup={args.speedup}\n")

    out = {"model": args.model, "concurrency": args.concurrency, "speedup": args.speedup,
           "latency_model": eff}

    if args.part in ("standalone", "both"):
        print("=== PART 2: standalone algorithms (all queries under concurrency cap) ===")
        out["standalone"] = asyncio.run(run_standalone(args, client))
    if args.part in ("optimizer", "both"):
        print(f"\n=== PART 3: optimizer per-query latency (policy={args.proxy_policy}, budget=${args.budget}) ===")
        out["optimizer"] = asyncio.run(run_optimizer_latency(args, client))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"dl20_latency_{args.model}.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 68)
    print(f"DL20 LATENCY ESTIMATE — {args.model}  (concurrency={args.concurrency})")
    print("=" * 68)
    if "standalone" in out:
        print("Standalone (total latency to rank all DL20 queries | per-query):")
        for name, r in out["standalone"]["algorithms"].items():
            print(f"  {name:<24s} {r['total_latency_s']:>7.1f}s total   "
                  f"{r['avg_latency_per_query_s']:>6.2f}s/query")
    if "optimizer" in out:
        o = out["optimizer"]
        print(f"\nOptimizer ({o['proxy_policy']}, {o['n_queries']} queries): "
              f"avg={o['avg_latency_per_query_s']}s/query  median={o['median_latency_per_query_s']}s  "
              f"range=[{o['min_s']}, {o['max_s']}]s")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
