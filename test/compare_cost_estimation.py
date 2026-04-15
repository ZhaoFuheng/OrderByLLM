import argparse
import asyncio
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import ir_datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from order_by.pair_comparison import external_comparisons
from order_by.pointwise import PointwiseRelevanceKey, external_values
from order_by.sorting import (
    external_bubble_sort,
    external_merge_sort,
    external_pointwise_sort,
    pointwise_sort,
    quick_sort,
)
from order_by.utils import (
    build_client, tokens2price,
    bubble_sort_calls_sim, merge_sort_calls_sim,
    quick_sort_calls_balanced, quick_sort_calls_sim,
    quick_calls_formula, bubble_calls_formula, merge_calls_formula,
)
from prompts.all_prompts import (
    passage_external_comparison_prompt_template,
    passage_external_pointwise_prompt_template,
    passage_pairwise_comparison_prompt_template,
    passage_pointwise_prompt_template,
)

LIMIT_K = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _safe_prompt(template: str, **kwargs) -> str:
    class _P(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"
    return template.format_map(_P(**kwargs))


# ── DL20 data loading ─────────────────────────────────────────────────────────

def load_dl20_queries(run_path: Path, hit_depth: int, n: int | None = None):
    """Load DL20 queries from a BM25 run file.  Returns a list of
    (qid, query_text, [(docid, text), ...]) tuples, sorted by qid.
    Only includes queries that have ground-truth relevance judgments (qrels).
    """
    ds = ir_datasets.load("msmarco-passage/trec-dl-2020")
    docstore = ds.docs_store()
    query_map = {str(q.query_id): q.text for q in ds.queries_iter()}

    qrels_qids = set()
    for q in ds.qrels_iter():
        qrels_qids.add(str(q.query_id))

    by_qid: dict[str, list] = defaultdict(list)
    with run_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            qid, _, docid, rank, _, _ = parts
            if int(rank) > hit_depth or qid not in query_map:
                continue
            doc = docstore.get(docid)
            if doc is None:
                continue
            text = (doc.title + " " if getattr(doc, "title", None) else "") + doc.text
            by_qid[qid].append((int(rank), docid, text))

    queries = []
    for qid, entries in sorted(by_qid.items()):
        if qid not in qrels_qids:
            continue
        entries.sort(key=lambda x: x[0])
        ranking = [(docid, text) for _, docid, text in entries]
        queries.append((qid, query_map[qid], ranking))
        if n and len(queries) >= n:
            break
    return queries


# ── Algorithm runners ─────────────────────────────────────────────────────────

async def run_alg(
    alg_name: str,
    data: list,
    client,
    pw_prompt: str,
    ex_prompt: str,
    p_prompt: str,
    ep_prompt: str,
    model: str,
) -> tuple[int, int, int]:
    """Run *alg_name* on *data*; return (in_tokens, out_tokens, num_api_calls)."""
    if alg_name == "quick":
        _, n_calls, in_tok, out_tok = await quick_sort(
            data[:], client, pw_prompt, model, isPassage=True, vote=1, limit_k=LIMIT_K
        )
    elif alg_name == "quick_3":
        _, n_calls, in_tok, out_tok = await quick_sort(
            data[:], client, pw_prompt, model, isPassage=True, vote=3, limit_k=LIMIT_K
        )
    elif alg_name.startswith("ext_bubble_"):
        b = int(alg_name.split("_")[-1])
        _, n_calls, in_tok, out_tok = await external_bubble_sort(
            data[:], external_comparisons, b, client, ex_prompt, model,
            isPassage=True, limit_k=LIMIT_K,
        )
    elif alg_name.startswith("ext_merge_"):
        b = int(alg_name.split("_")[-1])
        _, n_calls, in_tok, out_tok = await external_merge_sort(
            data[:], external_comparisons, b, client, ex_prompt, model,
            isPassage=True, limit_k=LIMIT_K,
        )
    elif alg_name == "pointwise":
        _, _, n_calls, in_tok, out_tok = await pointwise_sort(
            data[:], client, p_prompt, model, float,
            key_class=PointwiseRelevanceKey, isPassage=True,
        )
    elif alg_name.startswith("ext_point_"):
        b = int(alg_name.split("_")[-1])
        _, _, n_calls, in_tok, out_tok, _ = await external_pointwise_sort(
            data[:], external_values, client, ep_prompt, model, float,
            isPassage=True, memory_size=b,
        )
    else:
        raise ValueError(f"Unknown algorithm: {alg_name!r}")
    return in_tok, out_tok, n_calls



# ── Cost estimation (mirrors OrderByOptimizer.estimated_total_price) ──────────

def _estimate_price_common(alg_name, sample_run_price, sample_size, total_size, k,
                     quick_fn, bubble_fn, merge_fn,
                     actual_sample_api_calls=None,
                     quick_sample_run_price=None,
                     quick_sample_api_calls=None):
    """Core estimation logic parameterised by the call-counting functions."""
    if alg_name == "pointwise" or alg_name.startswith("ext_point_"):
        if actual_sample_api_calls:
            return sample_run_price * total_size / sample_size
        return sample_run_price * total_size / sample_size
    if alg_name in ("quick", "quick_3"):
        if alg_name == "quick_3":
            # Estimate quick_3 as 3x the quick(vote=1) price estimate.
            v = 3
            base_sample_price = quick_sample_run_price if quick_sample_run_price is not None else sample_run_price / v
            base_sample_calls = quick_sample_api_calls if quick_sample_api_calls is not None else (
                actual_sample_api_calls / v if actual_sample_api_calls is not None else None
            )
            quick_1_est = _estimate_price_common(
                "quick",
                base_sample_price,
                sample_size,
                total_size,
                k,
                quick_fn=quick_fn,
                bubble_fn=bubble_fn,
                merge_fn=merge_fn,
                actual_sample_api_calls=base_sample_calls,
            )
            return v * quick_1_est

        s_calls = actual_sample_api_calls
        total_calls = quick_fn(total_size, 1, k)
        correction_factor = quick_fn(sample_size, 1, min(k, sample_size)) / s_calls
        return sample_run_price * total_calls * correction_factor / max(s_calls, 1)
    if "ext_bubble" in alg_name:
        batch_size  = int(alg_name.split("_")[-1])
        s_calls = actual_sample_api_calls
        total_calls = bubble_fn(total_size, batch_size, k)        
        return sample_run_price * total_calls / max(s_calls, 1)
    if "ext_merge" in alg_name:
        batch_size  = int(alg_name.split("_")[-1])
        s_calls = actual_sample_api_calls
        total_calls = merge_fn(total_size, batch_size, k)
        return sample_run_price * total_calls / max(s_calls, 1)
    raise ValueError(f"Unknown algorithm for estimation: {alg_name!r}")


def estimated_price_sim(alg_name, sample_run_price, sample_size, total_size, k,
                        actual_sample_api_calls=None,
                        quick_sample_run_price=None,
                        quick_sample_api_calls=None):
    """Estimate using simulation-based call counting.

    For `quick` / `quick_3`, this path explicitly uses
    `quick_sort_calls_sim(...)`.
    """
    assert actual_sample_api_calls is not None, print(f'actual_sample_api_calls is not provided for {alg_name}')
    quick_call_counter = quick_sort_calls_sim
    bubble_call_counter = bubble_sort_calls_sim
    merge_call_counter = merge_sort_calls_sim
    return _estimate_price_common(
        alg_name, sample_run_price, sample_size, total_size, k,
        quick_fn=quick_call_counter,
        bubble_fn=bubble_call_counter,
        merge_fn=merge_call_counter,
        actual_sample_api_calls=actual_sample_api_calls,
        quick_sample_run_price=quick_sample_run_price,
        quick_sample_api_calls=quick_sample_api_calls,
    )


def estimated_price_formula(alg_name, sample_run_price, sample_size, total_size, k,
                            actual_sample_api_calls=None,
                            quick_sample_run_price=None,
                            quick_sample_api_calls=None):
    """Estimate using closed-form complexity formulas.

    For `quick` / `quick_3`, this path explicitly uses
    `quick_calls_formula(...)`.
    """
    assert actual_sample_api_calls is not None, print(f'actual_sample_api_calls is not provided for {alg_name}')
    # quick_call_counter = quick_calls_formula
    quick_call_counter = quick_sort_calls_sim
    bubble_call_counter = bubble_calls_formula
    merge_call_counter = merge_calls_formula
    return _estimate_price_common(
        alg_name, sample_run_price, sample_size, total_size, k,
        quick_fn=quick_call_counter,
        bubble_fn=bubble_call_counter,
        merge_fn=merge_call_counter,
        actual_sample_api_calls=actual_sample_api_calls,
        quick_sample_run_price=quick_sample_run_price,
        quick_sample_api_calls=quick_sample_api_calls,
    )


def _estimate_calls_common(alg_name, sample_size, total_size, k,
                           quick_fn, bubble_fn, merge_fn,
                           actual_sample_api_calls=None):
    """Estimate total API calls using the same logic as the cost estimator."""
    if alg_name == "pointwise" or alg_name.startswith("ext_point_"):
        if actual_sample_api_calls is None:
            return float(total_size)
        return actual_sample_api_calls * total_size / sample_size
    if alg_name in ("quick", "quick_3"):
        v = 3 if alg_name == "quick_3" else 1
        assert actual_sample_api_calls is not None, print(f'actual_sample_api_calls is not provided for {alg_name}')
        total_calls = v*quick_fn(total_size, 1.0, k)
        return total_calls
    if "ext_bubble" in alg_name:
        batch_size = int(alg_name.split("_")[-1])
        return bubble_fn(total_size, batch_size, k)
    if "ext_merge" in alg_name:
        batch_size = int(alg_name.split("_")[-1])
        return merge_fn(total_size, batch_size, k)
    raise ValueError(f"Unknown algorithm for estimation: {alg_name!r}")


def estimated_calls_sim(alg_name, sample_size, total_size, k,
                        actual_sample_api_calls=None):
    """Estimate total API calls using simulation-based call counting.

    For `quick` / `quick_3`, this path explicitly uses
    `quick_sort_calls_sim(...)`.
    """
    quick_call_counter = quick_sort_calls_sim
    bubble_call_counter = bubble_sort_calls_sim
    merge_call_counter = merge_sort_calls_sim
    return _estimate_calls_common(
        alg_name, sample_size, total_size, k,
        quick_fn=quick_call_counter,
        bubble_fn=bubble_call_counter,
        merge_fn=merge_call_counter,
        actual_sample_api_calls=actual_sample_api_calls,
    )


def estimated_calls_formula(alg_name, sample_size, total_size, k,
                            actual_sample_api_calls=None):
    """Estimate total API calls using closed-form call counting.

    For `quick` / `quick_3`, this path explicitly uses
    `quick_calls_formula(...)`.
    """
    quick_call_counter = quick_calls_formula
    bubble_call_counter = bubble_calls_formula
    merge_call_counter = merge_calls_formula
    return _estimate_calls_common(
        alg_name, sample_size, total_size, k,
        quick_fn=quick_call_counter,
        bubble_fn=bubble_call_counter,
        merge_fn=merge_call_counter,
        actual_sample_api_calls=actual_sample_api_calls,
    )

# ── Main ──────────────────────────────────────────────────────────────────────

async def _run(args) -> None:
    _load_env(PROJECT_ROOT / ".env")
    client = build_client()

    print(f"Loading DL20 queries (n={args.n_queries or 'all'})…")
    queries = load_dl20_queries(
        PROJECT_ROOT / args.run_file, args.hit_depth, args.n_queries
    )
    print(f"  Loaded {len(queries)} queries\n")

    records: list[dict] = []
    rng = random.Random(0)

    for qi, (qid, query, ranking) in enumerate(queries, 1):
        shuffled = ranking[:]
        rng.shuffle(shuffled)
        total_size = len(ranking)
        pw_prompt  = _safe_prompt(passage_pairwise_comparison_prompt_template, question=query)
        ex_prompt  = _safe_prompt(passage_external_comparison_prompt_template, question=query)
        p_prompt   = _safe_prompt(passage_pointwise_prompt_template,           question=query)
        ep_prompt  = _safe_prompt(passage_external_pointwise_prompt_template,  question=query)

        print(f"[{qi}/{len(queries)}] qid={qid}  total_docs={total_size}")
        print(f"  {'algorithm':<16}  {'sample':>6}  {'sample_price':>13}  "
              f"{'est_form':>10}  {'r_form':>6}  {'actual':>10}  "
              f"{'c_form':>7}  {'c_act':>7}")
        print(f"  {'-'*16}  {'-'*6}  {'-'*13}  "
              f"{'-'*10}  {'-'*6}  {'-'*10}  {'-'*7}  {'-'*7}")

        quick_sample_run_price = None
        quick_sample_calls = None
        for alg in args.algs:
            eff_sample  = args.sample_size
            sample_data = shuffled[:eff_sample]

            # Sample run — mirrors how the optimizer probes an algorithm
            in_s, out_s, sample_calls = await run_alg(
                alg, sample_data, client, pw_prompt, ex_prompt, p_prompt, ep_prompt, args.model
            )
            sample_run_price = tokens2price(args.model, in_s, out_s)
            if alg == "quick":
                quick_sample_run_price = sample_run_price
                quick_sample_calls = sample_calls
            elif alg == "quick_3" and (quick_sample_run_price is None or quick_sample_calls is None):
                quick_in_s, quick_out_s, quick_calls = await run_alg(
                    "quick", sample_data, client, pw_prompt, ex_prompt, p_prompt, ep_prompt, args.model
                )
                quick_sample_run_price = tokens2price(args.model, quick_in_s, quick_out_s)
                quick_sample_calls = quick_calls
            est_sim    = estimated_price_sim(alg, sample_run_price, eff_sample, total_size, LIMIT_K,
                                             actual_sample_api_calls=sample_calls,
                                             quick_sample_run_price=quick_sample_run_price,
                                             quick_sample_api_calls=quick_sample_calls)
            est_form   = estimated_price_formula(alg, sample_run_price, eff_sample, total_size, LIMIT_K,
                                                  actual_sample_api_calls=sample_calls,
                                                  quick_sample_run_price=quick_sample_run_price,
                                                  quick_sample_api_calls=quick_sample_calls)
            est_calls_sim = estimated_calls_sim(alg, eff_sample, total_size, LIMIT_K,
                                                actual_sample_api_calls=sample_calls)
            est_calls_form = estimated_calls_formula(alg, eff_sample, total_size, LIMIT_K,
                                                     actual_sample_api_calls=sample_calls)

            # Full run — ground-truth cost
            in_f, out_f, actual_calls = await run_alg(
                alg, shuffled, client, pw_prompt, ex_prompt, p_prompt, ep_prompt, args.model
            )
            actual = tokens2price(args.model, in_f, out_f)

            r_sim  = est_sim  / actual if actual > 0 else float("nan")
            r_form = est_form / actual if actual > 0 else float("nan")
            print(
                f"  {alg:<16}  {eff_sample:>6}  ${sample_run_price:>12.5f}  "
                f"${est_form:>9.4f}  {r_form:>5.2f}x  "
                f"${actual:>9.4f}  {est_calls_form:>7.1f}  {actual_calls:>7}"
            )
            records.append({
                "qid":              qid,
                "alg":              alg,
                "sample_size":      eff_sample,
                "total_size":       total_size,
                "sample_api_calls": sample_calls,
                "sample_run_price": sample_run_price,
                "est_sim":          est_sim,
                "est_formula":      est_form,
                "est_calls_sim":    est_calls_sim,
                "est_calls_formula": est_calls_form,
                "actual":           actual,
                "actual_api_calls": actual_calls,
                "ratio_sim":        r_sim,
                "ratio_formula":    r_form,
            })
        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    def _mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    def _std(vals, mu):
        return (sum((x - mu) ** 2 for x in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0

    W = 106
    print("=" * W)
    print(f"  {'Algorithm':<16}  {'Actual':>10}  │  "
          f"{'Form Est':>10}  {'Mean':>6}  {'Std':>5}  │  "
          f"{'Act Calls':>9}  {'Form Calls':>10}")
    print("-" * W)
    algs = sorted({r["alg"] for r in records})
    for alg in algs:
        pts          = [r for r in records if r["alg"] == alg]
        mean_actual  = _mean([r["actual"]       for r in pts])
        mean_form    = _mean([r["est_formula"]  for r in pts])
        mean_actual_calls = _mean([r["actual_api_calls"] for r in pts])
        mean_form_calls = _mean([r["est_calls_formula"] for r in pts])
        ratios_form  = [r["ratio_formula"] for r in pts if r["actual"] > 0]
        mu_form      = _mean(ratios_form)
        std_form     = _std(ratios_form, mu_form)
        print(
            f"  {alg:<16}  ${mean_actual:>9.4f}  │  "
            f"${mean_form:>9.4f}  {mu_form:>5.2f}x  {std_form:>5.2f}  │  "
            f"{mean_actual_calls:>9.1f}  {mean_form_calls:>10.1f}"
        )
    print("=" * W)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    safe_model = args.model.replace("/", "-")
    out_dir    = PROJECT_ROOT / "test" / "price_estimation"
    out_json   = out_dir / f"cost_comparison_{safe_model}.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model":     args.model,
                "n_queries": len(queries),
                "algs":      args.algs,
                "records":   records,
            },
            f, indent=2,
        )
    print(f"\nResults saved → {out_json}")


def main() -> None:
    _load_env(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Compare optimizer cost estimates vs actual costs on DL20.",
    )
    parser.add_argument("--model",       default="llama3.1-70b")
    parser.add_argument("--n-queries",   type=int, default=None,
                        help="Number of DL20 queries to process (default: all 54).")
    parser.add_argument("--sample-size", type=int, default=20,
                        help="Sample size for quick/quick_3/pointwise/ext_point estimation (default: 20).")
    parser.add_argument(
        "--algs",
        default="quick,quick_3,ext_bubble_4,ext_merge_4,pointwise,ext_point_4",
        help="Comma-separated list of algorithms to evaluate.",
    )
    parser.add_argument("--run-file",  default="data/run.msmarco-v1-passage.bm25-default.dl20.txt")
    parser.add_argument("--hit-depth", type=int, default=100)

    args = parser.parse_args()
    args.algs = [a.strip() for a in args.algs.split(",") if a.strip()]

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
