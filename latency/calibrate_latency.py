#!/usr/bin/env python3
"""Step 1 of the latency study: calibrate per-request latency vs. tokens.

Everything in the experiment is cached, so a cached run has ~zero latency. To
estimate how long the workload WOULD take, we first learn how request latency
depends on token counts for the target model. We do this by issuing *fresh*
(cache-bypassed) requests of the SAME kinds the algorithms make -- batched
external comparisons (merge/bubble), single pairwise comparisons (quick sort),
and pointwise scoring -- over real DL20 passages of varying length, timing each
in isolation (concurrency 1, so no contention pollutes the per-request latency).

We then fit two models by least squares:
    simple : latency ≈ a + b * total_tokens
    split  : latency ≈ a + b_out * output_tokens + b_in * input_tokens
Output tokens usually dominate (autoregressive decode), so `split` is the model
the simulator uses; `simple` is reported for reference / the paper's intuition.

Writes latency/latency_model_<model>.json (coefficients + R^2 + raw samples) and
latency/latency_vs_tokens_<model>.png (scatter + fit).

Usage:
    python latency/calibrate_latency.py --model claude-haiku-4-5 --provider anthropic
    python latency/calibrate_latency.py --n-queries 6 --batch-sizes 2,3,4,6,8
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "test"))

import run_experiment as RE  # noqa: E402  (test/run_experiment.py: _resolve, _safe_prompt)
from benchmarks import load_dl20  # noqa: E402
from order_by.clients import PROVIDERS, build_client  # noqa: E402
from order_by.pair_comparison import external_comparisons, Pair_Comparison_Key, PassageComparisonReasoning  # noqa: E402
from order_by.pointwise import PointwiseRelevanceKey  # noqa: E402
from order_by.utils import load_env_file  # noqa: E402
from prompts.all_prompts import (  # noqa: E402
    passage_external_comparison_prompt_template,
    passage_pairwise_comparison_prompt_template,
    passage_pointwise_prompt_template,
)

OUT_DIR = Path(__file__).resolve().parent


async def _timed(coro):
    t0 = time.perf_counter()
    res = await coro
    dt = time.perf_counter() - t0
    return dt, res


async def collect(args, client):
    bench = load_dl20(RE._resolve(args.dl20_run_file), args.hit_depth)
    rng = random.Random(0)
    # A handful of queries; shuffle each ranking (same style as the canonical run).
    queries = bench.first_stage[: args.n_queries]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]

    # Warmup: the first few requests pay cold-start (deployment spin-up, TLS, client
    # init) and are wild outliers; issue and discard a few before timing anything.
    if args.warmup > 0 and queries:
        q0 = queries[0]
        wtop = q0[2][:]
        rng.shuffle(wtop)
        wex = RE._safe_prompt(passage_external_comparison_prompt_template, question=q0[1])
        print(f"  [warmup {args.warmup} discarded]", flush=True)
        for w in range(args.warmup):
            try:
                await external_comparisons(wtop[: 2 + (w % 3)], client, wex + f"\n<!--warm{w}-->",
                                           args.model, isPassage=True, useCache=False)
            except Exception:
                pass

    samples = []  # each: {kind, m, in_tok, out_tok, total, latency}

    for qi, (qid, query, ranking) in enumerate(queries):
        top = ranking[:]
        rng.shuffle(top)
        ex_prompt = RE._safe_prompt(passage_external_comparison_prompt_template, question=query)
        pw_prompt = RE._safe_prompt(passage_pairwise_comparison_prompt_template, question=query)
        pt_prompt = RE._safe_prompt(passage_pointwise_prompt_template, question=query)

        # (a) external batched comparisons of varying batch size m -> spans in/out tokens
        for m in batch_sizes:
            batch = top[:m]
            if len(batch) < 2:
                continue
            dt, res = await _timed(external_comparisons(
                batch, client, ex_prompt, args.model, isPassage=True, useCache=False))
            _sorted, _calls, in_tok, out_tok = res
            samples.append({"kind": f"ext_cmp_m{m}", "m": m,
                            "in_tok": in_tok, "out_tok": out_tok,
                            "total": in_tok + out_tok, "latency": dt})
            print(f"  q{qi} ext_cmp m={m:<2d} in={in_tok:<6d} out={out_tok:<5d} "
                  f"lat={dt:.2f}s", flush=True)

        # (b) single pairwise comparison (quick sort's atomic op). Bypass cache by
        #     using a throwaway Pair_Comparison_Key on a unique passage pair each time.
        a = Pair_Comparison_Key((top[0][0], top[0][1]), PassageComparisonReasoning)
        b = Pair_Comparison_Key((top[1][0], top[1][1]), PassageComparisonReasoning)
        # Force a fresh call: prime cache-miss by appending a nonce to the prompt template.
        nonce_prompt = pw_prompt + f"\n<!--calib q{qi}-->"
        dt, res = await _timed(a.compare(b, client, nonce_prompt, args.model))
        _cmp, _n, in_tok, out_tok = res
        if in_tok or out_tok:
            samples.append({"kind": "pairwise", "m": 2, "in_tok": in_tok, "out_tok": out_tok,
                            "total": in_tok + out_tok, "latency": dt})
            print(f"  q{qi} pairwise    in={in_tok:<6d} out={out_tok:<5d} lat={dt:.2f}s", flush=True)

        # (c) pointwise scoring (a single passage). Bypass cache via env flag.
        os.environ["POINTWISE_NO_CACHE"] = "1"
        key = PointwiseRelevanceKey(top[0][1], pt_prompt, True, False)
        dt, res = await _timed(key.value(client, args.model, float))
        _val, _n, in_tok, out_tok = res
        samples.append({"kind": "pointwise", "m": 1, "in_tok": in_tok, "out_tok": out_tok,
                        "total": in_tok + out_tok, "latency": dt})
        print(f"  q{qi} pointwise    in={in_tok:<6d} out={out_tok:<5d} lat={dt:.2f}s", flush=True)
        os.environ.pop("POINTWISE_NO_CACHE", None)

    return samples


# ── tiny dependency-free least-squares (normal equations) ────────────────────
def _lstsq(X, y):
    """Solve min ||X b - y||^2 via normal equations (X^T X) b = X^T y. Pure Python."""
    n, k = len(X), len(X[0])
    XtX = [[sum(X[r][i] * X[r][j] for r in range(n)) for j in range(k)] for i in range(k)]
    Xty = [sum(X[r][i] * y[r] for r in range(n)) for i in range(k)]
    # Gaussian elimination
    A = [row[:] + [Xty[i]] for i, row in enumerate(XtX)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        if abs(A[col][col]) < 1e-12:
            A[col][col] = 1e-12
        pivval = A[col][col]
        A[col] = [v / pivval for v in A[col]]
        for r in range(k):
            if r != col:
                f = A[r][col]
                A[r] = [A[r][c] - f * A[col][c] for c in range(k + 1)]
    return [A[i][k] for i in range(k)]


def _r2(y, yhat):
    ybar = sum(y) / len(y)
    ss_tot = sum((v - ybar) ** 2 for v in y)
    ss_res = sum((y[i] - yhat[i]) ** 2 for i in range(len(y)))
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit(samples):
    y = [s["latency"] for s in samples]
    med = statistics.median(y)
    mean = statistics.mean(y)
    # simple: a + b*total
    Xs = [[1.0, s["total"]] for s in samples]
    a, b = _lstsq(Xs, y)
    r2_s = _r2(y, [a + b * s["total"] for s in samples])
    # split: a + b_out*out + b_in*in
    Xp = [[1.0, s["out_tok"], s["in_tok"]] for s in samples]
    a2, b_out, b_in = _lstsq(Xp, y)
    r2_p = _r2(y, [a2 + b_out * s["out_tok"] + b_in * s["in_tok"] for s in samples])
    # out_only: a + b*out  (the physically-motivated LLM decode model)
    Xo = [[1.0, s["out_tok"]] for s in samples]
    ao, bo = _lstsq(Xo, y)
    r2_o = _r2(y, [ao + bo * s["out_tok"] for s in samples])

    # Choose a DEFENSIBLE effective model for the simulator. LLM latency should be
    # non-decreasing in output tokens; if the best-fitting model has a non-negative
    # output slope AND explains a meaningful fraction of variance, use it. Otherwise,
    # latency in this token range is dominated by a ~fixed per-request cost + server
    # noise, so fall back to a constant = median (robust to cold-start outliers).
    candidates = []
    if bo >= 0:
        candidates.append(("out_only", r2_o, {"intercept": ao, "per_output_token": bo, "per_input_token": 0.0}))
    if b >= 0:
        candidates.append(("simple", r2_s, {"intercept": a, "per_output_token": b, "per_input_token": b}))
    best = max(candidates, key=lambda c: c[1]) if candidates else None
    if best and best[1] >= 0.15:
        eff_name, eff_r2, eff = best[0], best[1], best[2]
    else:
        eff_name, eff_r2 = "constant_median", 0.0
        eff = {"intercept": med, "per_output_token": 0.0, "per_input_token": 0.0}

    return {
        "median_latency_s": med, "mean_latency_s": mean,
        "simple": {"intercept": a, "per_total_token": b, "r2": r2_s,
                   "formula": "latency_s = intercept + per_total_token * total_tokens"},
        "split": {"intercept": a2, "per_output_token": b_out, "per_input_token": b_in, "r2": r2_p,
                  "formula": "latency_s = intercept + per_output_token*out + per_input_token*in"},
        "out_only": {"intercept": ao, "per_output_token": bo, "r2": r2_o,
                     "formula": "latency_s = intercept + per_output_token * out_tokens"},
        "effective": {**eff, "chosen": eff_name, "r2": eff_r2,
                      "formula": "latency_s = intercept + per_output_token*out + per_input_token*in"},
    }


def plot(samples, fitres, model, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [plot skipped: {e}]")
        return
    kinds = sorted(set(s["kind"] for s in samples))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, kd in enumerate(kinds):
        pts = [s for s in samples if s["kind"] == kd]
        ax.scatter([p["total"] for p in pts], [p["latency"] for p in pts],
                   s=28, color=cmap(i % 10), label=kd, alpha=0.8)
    xs = sorted(s["total"] for s in samples)
    a, b = fitres["simple"]["intercept"], fitres["simple"]["per_total_token"]
    ax.plot(xs, [a + b * x for x in xs], "k--",
            label=f"simple fit (R²={fitres['simple']['r2']:.2f})")
    ax.set_xlabel("total tokens (input + output)")
    ax.set_ylabel("request latency (s)")
    ax.set_title(f"Per-request latency vs tokens — {model}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"  saved plot -> {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--provider", default="anthropic", choices=PROVIDERS)
    ap.add_argument("--n-queries", type=int, default=10)
    ap.add_argument("--batch-sizes", default="2,3,4,6,8")
    ap.add_argument("--warmup", type=int, default=3,
                    help="Discard this many warmup requests (cold-start outliers) before timing.")
    ap.add_argument("--hit-depth", type=int, default=100)
    ap.add_argument("--dl20-run-file",
                    default="data/run.msmarco-v1-passage.bm25-default.dl20.txt")
    a = ap.parse_args()
    load_env_file(PROJECT_ROOT / ".env")
    client = build_client(a.provider)

    samples = asyncio.run(collect(a, client))
    if len(samples) < 3:
        print("Not enough samples to fit."); return
    fitres = fit(samples)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"model": a.model, "provider": a.provider, "n_samples": len(samples),
               "fit": fitres, "samples": samples}
    out_json = OUT_DIR / f"latency_model_{a.model}.json"
    out_json.write_text(json.dumps(payload, indent=2))
    plot(samples, fitres, a.model, OUT_DIR / f"latency_vs_tokens_{a.model}.png")

    print("\n" + "=" * 66)
    print(f"LATENCY CALIBRATION — {a.model}  ({len(samples)} requests after warmup)")
    print("=" * 66)
    s, o, eff = fitres["simple"], fitres["out_only"], fitres["effective"]
    print(f"median per-request latency: {fitres['median_latency_s']:.2f}s   "
          f"(mean {fitres['mean_latency_s']:.2f}s)")
    print(f"simple  : {s['intercept']:.2f} + {s['per_total_token']*1000:.4f} ms/tok*total  (R²={s['r2']:.3f})")
    print(f"out_only: {o['intercept']:.2f} + {o['per_output_token']*1000:.4f} ms/tok*out    (R²={o['r2']:.3f})")
    print(f"EFFECTIVE (simulator uses this): chosen='{eff['chosen']}'  R²={eff['r2']:.3f}")
    print(f"  latency ≈ {eff['intercept']:.3f} + {eff['per_output_token']*1000:.4f} ms/tok*out "
          f"+ {eff['per_input_token']*1000:.4f} ms/tok*in")
    print(f"\nsaved -> {out_json}")


if __name__ == "__main__":
    main()
