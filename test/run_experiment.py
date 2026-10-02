import argparse
import asyncio
import json
import logging
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytrec_eval
from openai import AsyncOpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks import PassageBenchmark, load_dl20, load_hellaswag, load_nfcorpus
from order_by.clients import PROVIDER_HELP, PROVIDERS, build_client, run_async
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
    PhaseTracker,
    gather_bounded,
    kendalltau_distance,
    load_env_file,
    load_movie_reviews,
    query_concurrency,
    tokens2price,
)
from prompts.all_prompts import (
    movie_external_comparison_prompt_template,
    movie_external_pointwise_prompt_template,
    movie_pairwise_comparison_prompt_template,
    movie_pointwise_prompt_template,
    passage_external_comparison_prompt_template,
    passage_external_pointwise_prompt_template,
    passage_pairwise_comparison_prompt_template,
    passage_pointwise_prompt_template,
    population_external_comparison_prompt_template,
    population_external_pointwise_prompt_template,
    population_pairwise_comparison_prompt_template,
    population_pointwise_prompt_template,
)


# Population algorithms — includes wiki-search variants (same structure as NBA in dev)
POPULATION_ALGORITHMS = [
    "pointwise",
    "pointwise_with_search",
    "external_pointwise_4",
    "external_pointwise_4_with_search",
    "quick_sort",
    "quick_sort3",
    "external_bubble_sort_4",
    "external_merge_sort_4",
]

# Algorithms run on every query of a passage benchmark (DL20, HellaSwag,
# NFCorpus) and on every SembenchMovie movie; batch size 4, no search variants.
PASSAGE_ALGORITHMS = [
    "pointwise",
    "quick_sort",
    "quick_sort3",
    "external_pointwise_4",
    "external_merge_sort_4",
    "external_bubble_sort_4",
]

EXTERNAL_POINTWISE_MEMORY_SIZES = (4,)

# Wikipedia infobox field used for country population lookups.
POPULATION_WIKI_FIELD = "population_estimate"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_prompt(template: str, **kwargs) -> str:
    """Fill only the specified placeholders; leave any others intact."""
    class _PartialFmt(dict):
        def __missing__(self, key):
            return "{" + key + "}"
    return template.format_map(_PartialFmt(**kwargs))


def _summarize(seed_to_score: dict[int, float]) -> tuple[float, float]:
    vals = list(seed_to_score.values())
    mean_v = float(statistics.mean(vals)) if vals else 0.0
    std_v = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    return mean_v, std_v


def _price(model: str, in_toks: int, out_toks: int) -> float:
    if in_toks <= 0 and out_toks <= 0:
        return 0.0
    return tokens2price(model, in_toks, out_toks)


def _normalize_docids(sorted_data):
    out = []
    for x in sorted_data:
        if isinstance(x, tuple):
            out.append(str(x[0]))
        else:
            out.append(str(x))
    return out


def _to_run_scores(docids: list[str]) -> dict[str, float]:
    n = len(docids)
    return {docid: float(n - i) for i, docid in enumerate(docids)}


def _empty_acc(algs: list[str]) -> dict:
    return {
        alg: {"in_tokens": 0, "out_tokens": 0, "seed_scores": {}, "per_item_scores": {}, "per_item_costs": {}}
        for alg in algs
    }


# ── Population (kendalltau, web-search variants) ─────────────────────────────

async def _run_population_algorithms_once(
    country_names: list[str],
    client: AsyncOpenAI,
    model: str,
    seed: int = 0,
):
    outputs = {}
    pbar = tqdm(
        total=len(POPULATION_ALGORITHMS),
        desc=f"  seed={seed}",
        unit="alg",
        leave=False,
    )

    async def _run(name, coro):
        pbar.set_postfix_str(name)
        result = await coro
        outputs[name] = result
        pbar.update(1)

    await _run("pointwise", _wrap(pointwise_sort(
        country_names[:], client, population_pointwise_prompt_template, model, float
    )))

    await _run("pointwise_with_search", _wrap(pointwise_sort(
        country_names[:], client, population_pointwise_prompt_template, model, float,
        use_wiki=True,
        wiki_field=POPULATION_WIKI_FIELD,
    )))

    for m in EXTERNAL_POINTWISE_MEMORY_SIZES:
        await _run(f"external_pointwise_{m}", _wrap(external_pointwise_sort(
            country_names[:], external_values, client,
            population_external_pointwise_prompt_template, model, float,
            isPassage=False, memory_size=m,
        )))
        await _run(f"external_pointwise_{m}_with_search", _wrap(external_pointwise_sort(
            country_names[:], external_values, client,
            population_external_pointwise_prompt_template, model, float,
            isPassage=False, memory_size=m,
            wiki_field=POPULATION_WIKI_FIELD,
        )))

    await _run("quick_sort", _wrap(quick_sort(
        country_names[:], client, population_pairwise_comparison_prompt_template,
        model, isPassage=False, vote=1,
    )))

    await _run("quick_sort3", _wrap(quick_sort(
        country_names[:], client, population_pairwise_comparison_prompt_template,
        model, isPassage=False, vote=3,
    )))

    for m in EXTERNAL_POINTWISE_MEMORY_SIZES:
        pbar.set_postfix_str(f"ext_bubble_{m} | ext_merge_{m}  [parallel]")
        (bubble_result, merge_result) = await asyncio.gather(
            _wrap(external_bubble_sort(
                country_names[:], external_comparisons, m, client,
                population_external_comparison_prompt_template, model, isPassage=False,
            )),
            _wrap(external_merge_sort(
                country_names[:], external_comparisons, m, client,
                population_external_comparison_prompt_template, model, isPassage=False,
            )),
        )
        outputs[f"external_bubble_sort_{m}"] = bubble_result
        outputs[f"external_merge_sort_{m}"] = merge_result
        pbar.update(2)

    pbar.close()
    return outputs


def _wrap(coro):
    """Normalise sort results to a uniform (sorted_data, in_tokens, out_tokens) tuple."""
    async def _inner():
        result = await coro
        return (result[0], result[2], result[3])
    return _inner()


def _resolve(p: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT when it is not absolute."""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


async def run_population(args, client: AsyncOpenAI, pbar: tqdm | None = None, alg_pbar: tqdm | None = None) -> dict:
    df = pd.read_csv(_resolve(args.population_csv))
    if "Population (2020)" not in df.columns or "Country" not in df.columns:
        raise ValueError("Population CSV must contain 'Population (2020)' and 'Country'.")
    if args.population_limit is not None:
        if args.population_limit <= 1:
            raise ValueError("--population-limit must be greater than 1.")
        df = df.head(args.population_limit).copy()

    country_names = df["Country"].astype(str).tolist()
    gold = (
        df.sort_values(by=["Population (2020)", "Country"], ascending=[True, True])["Country"]
        .astype(str)
        .tolist()
    )
    acc = _empty_acc(POPULATION_ALGORITHMS)

    for i, seed in enumerate(args.seeds, 1):
        if pbar is not None:
            pbar.set_description(f"[{args.model}] seed {i}/{len(args.seeds)}")
        shuffled = country_names[:]
        random.Random(seed).shuffle(shuffled)
        outputs = await _run_population_algorithms_once(shuffled[:], client, args.model, seed=seed)

        for alg, (sorted_data, in_t, out_t) in outputs.items():
            acc[alg]["in_tokens"] += in_t
            acc[alg]["out_tokens"] += out_t
            score = float(kendalltau_distance(gold[:], [str(x) for x in sorted_data]))
            acc[alg]["seed_scores"][seed] = score
        if pbar is not None:
            pbar.update(1)

    metrics = []
    for alg, info in acc.items():
        mean_v, std_v = _summarize(info["seed_scores"])
        metrics.append(
            {
                "algorithm": alg,
                "seed_scores": info["seed_scores"],
                "score_mean": mean_v,
                "score_std": std_v,
                "price": _price(args.model, info["in_tokens"], info["out_tokens"]),
                "tokens": info["in_tokens"] + info["out_tokens"],
            }
        )

    return {
        "dataset": "population",
        "generated_at": _now_iso(),
        "settings": {
            "csv": args.population_csv,
            "model": args.model,
            "seeds": args.seeds,
            "algorithms": [m["algorithm"] for m in metrics],
        },
        "metrics": metrics,
        "metric_name": "kendalltau",
    }


# ── Passage benchmarks (ndcg@10): DL20, HellaSwag, NFCorpus ──────────────────

def passage_algorithm(name: str, ranking: list[tuple[str, str]], query: str, client: AsyncOpenAI, model: str):
    """Coroutine that ranks one query's candidates with the named algorithm (one
    of PASSAGE_ALGORITHMS; batch size 4, LIMIT 10). It resolves to whatever the
    underlying sort function returns."""
    def prompt(template):
        return _safe_prompt(template, question=query)

    if name == "pointwise":
        return pointwise_sort(
            ranking[:], client, prompt(passage_pointwise_prompt_template), model, float,
            key_class=PointwiseRelevanceKey, isPassage=True,
        )
    if name == "external_pointwise_4":
        return external_pointwise_sort(
            ranking[:], external_values, client, prompt(passage_external_pointwise_prompt_template),
            model, float, isPassage=True, memory_size=4,
        )
    if name in ("quick_sort", "quick_sort3"):
        return quick_sort(
            ranking[:], client, prompt(passage_pairwise_comparison_prompt_template), model,
            isPassage=True, vote=3 if name == "quick_sort3" else 1, limit_k=10,
        )
    if name in ("external_bubble_sort_4", "external_merge_sort_4"):
        sort = external_bubble_sort if name == "external_bubble_sort_4" else external_merge_sort
        return sort(
            ranking[:], external_comparisons, 4, client, prompt(passage_external_comparison_prompt_template),
            model, isPassage=True, limit_k=10,
        )
    raise ValueError(f"Unknown passage algorithm: {name}")


async def _run_passage_algorithms_once(
    ranking: list[tuple[str, str]],
    query: str,
    client: AsyncOpenAI,
    model: str,
    tracker: "PhaseTracker | None" = None,
):
    """Rank one query's candidates with every algorithm in PASSAGE_ALGORITHMS."""
    tracker = tracker or PhaseTracker(None, 0)  # no-op when disabled

    def run(name):
        return tracker.run(name, passage_algorithm(name, ranking, query, client, model))

    outputs = {}

    # Pointwise variants return LLM relevance scores directly.
    p_ids, p_scores, _, p_in, p_out = await run("pointwise")
    outputs["pointwise"] = (p_ids, p_scores, p_in, p_out)

    ep_ids, ep_scores, _, ep_in, ep_out, _ = await run("external_pointwise_4")
    outputs["external_pointwise_4"] = (ep_ids, ep_scores, ep_in, ep_out)

    # Comparison-based algorithms return worst-to-best order (no direct scores).
    q1_sorted, _, q1_in, q1_out = await run("quick_sort")
    outputs["quick_sort"] = (_normalize_docids(q1_sorted), None, q1_in, q1_out)

    # The remaining three run concurrently to reduce total wall-clock time.
    (
        (q3_sorted, _, q3_in, q3_out),
        (eb4_sorted, _, eb4_in, eb4_out),
        (em4_sorted, _, em4_in, em4_out),
    ) = await asyncio.gather(
        run("quick_sort3"), run("external_bubble_sort_4"), run("external_merge_sort_4"),
    )
    outputs["quick_sort3"]            = (_normalize_docids(q3_sorted), None, q3_in, q3_out)
    outputs["external_bubble_sort_4"] = (_normalize_docids(eb4_sorted), None, eb4_in, eb4_out)
    outputs["external_merge_sort_4"]  = (_normalize_docids(em4_sorted), None, em4_in, em4_out)

    return outputs


async def _run_passage_benchmark(
    dataset: str,
    bench: PassageBenchmark,
    settings: dict,
    args,
    client: AsyncOpenAI,
    pbar: tqdm | None = None,
    alg_pbar: tqdm | None = None,
) -> dict:
    """Run every algorithm on every query of `bench` and score it with ndcg@10.
    When the benchmark has a BM25 first stage, it is reported as a zero-cost
    baseline. `settings` is the dataset-specific part of the payload's settings."""
    first_stage = bench.first_stage
    algorithms = (["bm25"] if bench.bm25 is not None else []) + PASSAGE_ALGORITHMS
    acc = _empty_acc(algorithms)

    for seed in args.seeds:
        run_by_alg = {alg: {} for alg in algorithms}

        if bench.bm25 is not None:
            # BM25 is the original run-file order — same for every seed, zero LLM cost.
            for qid, _, _ in first_stage:
                run_by_alg["bm25"][str(qid)] = _to_run_scores(bench.bm25[qid])

        if pbar is not None:
            sidx = args.seeds.index(seed) + 1
            pbar.reset(total=len(first_stage))
            pbar.set_description(f"[{args.model}] seed {sidx}/{len(args.seeds)}")

        tracker = PhaseTracker(alg_pbar, len(first_stage))

        async def _rank_query(qid, query, top_ranking):
            outputs = await _run_passage_algorithms_once(
                top_ranking, query, client, args.model, tracker=tracker,
            )
            if pbar is not None:
                pbar.update(1)
            tracker.query_done()
            return qid, outputs

        # Candidates are shuffled up front (bench.shuffled), so the input order is
        # identical regardless of how many queries run concurrently.
        results = await gather_bounded(
            [_rank_query(*q) for q in bench.shuffled(seed)],
            query_concurrency(args.model, args.provider),
        )
        for qid, outputs in results:
            for alg, (docids, scores, in_t, out_t) in outputs.items():
                acc[alg]["in_tokens"] += in_t
                acc[alg]["out_tokens"] += out_t
                acc[alg]["per_item_costs"].setdefault(seed, {})[str(qid)] = _price(args.model, in_t, out_t)
                if scores is not None:
                    # Pointwise / external_pointwise: use LLM relevance scores directly.
                    run_by_alg[alg][str(qid)] = {str(d): float(s) for d, s in zip(docids, scores)}
                else:
                    # Comparison-based (worst-to-best order): assign rank score i+1.
                    run_by_alg[alg][str(qid)] = {str(d): float(i + 1) for i, d in enumerate(docids)}

        for alg in algorithms:
            alg_metrics = bench.evaluator.evaluate(run_by_alg[alg])
            per_query = {qid: float(m["ndcg_cut_10"]) for qid, m in alg_metrics.items()}
            score = sum(per_query.values()) / len(per_query) if per_query else 0.0
            acc[alg]["seed_scores"][seed] = float(score)
            acc[alg]["per_item_scores"][seed] = per_query

    metrics = []
    for alg, info in acc.items():
        mean_v, std_v = _summarize(info["seed_scores"])
        in_t = info["in_tokens"]
        out_t = info["out_tokens"]
        metrics.append(
            {
                "algorithm": alg,
                "seed_scores": info["seed_scores"],
                "per_query_scores": info["per_item_scores"],
                "per_query_costs": info["per_item_costs"],
                "score_mean": mean_v,
                "score_std": std_v,
                "price": _price(args.model, in_t, out_t),
                "tokens": in_t + out_t,
            }
        )

    return {
        "dataset": dataset,
        "generated_at": _now_iso(),
        "settings": {
            **settings,
            "model": args.model,
            "seeds": args.seeds,
            "algorithms": [m["algorithm"] for m in metrics],
        },
        "metrics": metrics,
        "metric_name": "ndcg@10",
    }


async def run_dl20(args, client: AsyncOpenAI, pbar: tqdm | None = None, alg_pbar: tqdm | None = None) -> dict:
    bench = load_dl20(_resolve(args.dl20_run_file), args.hit_depth)
    for _, _, ranking in bench.first_stage:
        assert len(ranking) == 100, f"Expected 100 docs per query, got {len(ranking)}"
    settings = {"run_file": args.dl20_run_file, "hit_depth": args.hit_depth}
    return await _run_passage_benchmark("dl20", bench, settings, args, client, pbar, alg_pbar)


async def run_hellaswag(args, client: AsyncOpenAI, pbar: tqdm | None = None, alg_pbar: tqdm | None = None) -> dict:
    bench = load_hellaswag(_resolve(args.hellaswag_dir), args.hellaswag_num_queries)
    bench.limit(args.hellaswag_limit)
    settings = {"data_dir": str(args.hellaswag_dir)}
    return await _run_passage_benchmark("hellaswag", bench, settings, args, client, pbar, alg_pbar)


async def run_nfcorpus(args, client: AsyncOpenAI, pbar: tqdm | None = None, alg_pbar: tqdm | None = None) -> dict:
    bench = load_nfcorpus(_resolve(args.nfcorpus_dir))
    bench.limit(args.nfcorpus_limit)
    settings = {"data_dir": str(args.nfcorpus_dir)}
    return await _run_passage_benchmark("nfcorpus", bench, settings, args, client, pbar, alg_pbar)


# ── SembenchMovie (kendalltau, rank reviews of top-K movies by positivity) ────

def _load_movie_bm25_run(run_path: Path) -> dict[str, list[str]]:
    """Load a TREC-format BM25 run file and return {movie_id: [reviewId, ...]}
    ordered by BM25 rank (best first)."""
    bm25_by_movie: dict[str, list[tuple[int, str]]] = {}
    with run_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            movie_id, _, review_id, rank = parts[0], parts[1], parts[2], int(parts[3])
            bm25_by_movie.setdefault(movie_id, []).append((rank, review_id))
    return {
        mid: [rid for _, rid in sorted(entries)]
        for mid, entries in bm25_by_movie.items()
    }


def _build_sembench_movie_data(
    csv_path: Path,
    top_k_reviewed_movies: int = 5,
    review_limit: int | None = None,
):
    return load_movie_reviews(csv_path, top_k_reviewed_movies, review_limit)


async def _run_sembench_movie_algorithms_once(
    ranking: list[tuple[str, str]],
    client: AsyncOpenAI,
    model: str,
    tracker: "PhaseTracker | None" = None,
):
    tracker = tracker or PhaseTracker(None, 0)  # no-op when disabled

    outputs = {}

    p_ids, p_scores, _, p_in, p_out = await tracker.run("pointwise", pointwise_sort(
        ranking[:], client, movie_pointwise_prompt_template, model, float,
        key_class=PointwiseRelevanceKey, isPassage=False, isReview=True,
    ))
    outputs["pointwise"] = (p_ids, p_scores, p_in, p_out)

    ep_ids, ep_scores, _, ep_in, ep_out, _ = await tracker.run("external_pointwise_4", external_pointwise_sort(
        ranking[:], external_values, client, movie_external_pointwise_prompt_template,
        model, float, isPassage=False, isReview=True, memory_size=4,
    ))
    outputs["external_pointwise_4"] = (ep_ids, ep_scores, ep_in, ep_out)

    q1_sorted, _, q1_in, q1_out = await tracker.run("quick_sort", quick_sort(
        ranking[:], client, movie_pairwise_comparison_prompt_template,
        model, isPassage=False, vote=1, isReview=True, limit_k=10,
    ))
    outputs["quick_sort"] = (_normalize_docids(q1_sorted), None, q1_in, q1_out)

    def _ext(kind, m):
        fn = external_merge_sort if kind == "merge" else external_bubble_sort
        return tracker.run(f"external_{kind}_sort_{m}", fn(
            ranking[:], external_comparisons, m, client,
            movie_external_comparison_prompt_template, model,
            isPassage=False, isReview=True, limit_k=10))

    (
        (q3_sorted, _, q3_in, q3_out),
        (em4_sorted, _, em4_in, em4_out),
        (eb4_sorted, _, eb4_in, eb4_out),
    ) = await asyncio.gather(
        tracker.run("quick_sort3", quick_sort(ranking[:], client, movie_pairwise_comparison_prompt_template,
                   model, isPassage=False, vote=3, isReview=True, limit_k=10)),
        _ext("merge", 4),
        _ext("bubble", 4),
    )
    outputs["quick_sort3"]            = (_normalize_docids(q3_sorted), None, q3_in, q3_out)
    outputs["external_merge_sort_4"]  = (_normalize_docids(em4_sorted), None, em4_in, em4_out)
    outputs["external_bubble_sort_4"] = (_normalize_docids(eb4_sorted), None, eb4_in, eb4_out)

    return outputs


async def run_sembench_movie(
    args, client: AsyncOpenAI, pbar: tqdm | None = None, alg_pbar: tqdm | None = None
) -> dict:
    first_stage, gold_by_movie, qrels_by_movie = _build_sembench_movie_data(
        _resolve(args.movie_csv),
        top_k_reviewed_movies=args.movie_top_k,
        review_limit=args.movie_review_limit,
    )

    bm25_run_path = _resolve(args.movie_bm25_run)
    bm25_by_movie: dict[str, list[str]] = {}
    if bm25_run_path.exists():
        bm25_by_movie = _load_movie_bm25_run(bm25_run_path)
        tqdm.write(f"  [bm25] loaded BM25 rankings for {len(bm25_by_movie)} movies from {bm25_run_path}")
    else:
        tqdm.write(f"  [bm25] run file not found: {bm25_run_path} — skipping BM25")

    algorithms = ["bm25"] + PASSAGE_ALGORITHMS
    acc = _empty_acc(algorithms)

    for seed in args.seeds:
        rng = random.Random(seed)
        movie_scores: dict[str, list[tuple[str, float]]] = {alg: [] for alg in algorithms}

        if pbar is not None:
            sidx = args.seeds.index(seed) + 1
            pbar.reset(total=len(first_stage))
            pbar.set_description(f"[{args.model}] seed {sidx}/{len(args.seeds)}")

        # Always create a tracker (no-op display when alg_pbar is None).
        tracker = PhaseTracker(alg_pbar, len(first_stage))

        # Each movie runs the full algorithm set — pointwise, ext_pointwise, quick,
        # quick_3, and ext_merge/ext_bubble 4 — inside _run_sembench_movie_algorithms_once.
        # Pre-shuffle sequentially so the shared rng stays deterministic regardless
        # of query concurrency (matches the passage benchmarks).
        all_shuffled: dict[str, list] = {}
        for movie_id, ranking in first_stage:
            shuffled = ranking[:]
            rng.shuffle(shuffled)
            all_shuffled[movie_id] = shuffled

        async def _rank_movie(movie_id):
            out = await _run_sembench_movie_algorithms_once(
                all_shuffled[movie_id][:], client, args.model, tracker=tracker,
            )
            if pbar is not None:
                pbar.update(1)
            tracker.query_done()
            return movie_id, out

        all_outputs: dict[str, dict] = dict(await gather_bounded(
            [_rank_movie(mid) for mid, _ in first_stage],
            query_concurrency(args.model, args.provider),
        ))

        # Inject BM25 rankings (zero-cost baseline)
        for movie_id, _ in first_stage:
            if movie_id in bm25_by_movie:
                all_outputs[movie_id]["bm25"] = (bm25_by_movie[movie_id], None, 0, 0)

        # Evaluate all algorithms across all movies
        for movie_id, _ in first_stage:
            outputs = all_outputs[movie_id]
            movie_evaluator = pytrec_eval.RelevanceEvaluator(
                {movie_id: qrels_by_movie[movie_id]}, {"ndcg_cut.10"}
            )
            for alg, (docids, scores, in_t, out_t) in outputs.items():
                acc[alg]["in_tokens"] += in_t
                acc[alg]["out_tokens"] += out_t
                acc[alg]["per_item_costs"].setdefault(seed, {})[movie_id] = _price(args.model, in_t, out_t)
                if scores is not None:
                    run = {movie_id: {str(d): float(s) for d, s in zip(docids, scores)}}
                else:
                    run = {movie_id: {str(d): float(i + 1) for i, d in enumerate(docids)}}
                result = movie_evaluator.evaluate(run)
                ndcg_val = float(result[movie_id]["ndcg_cut_10"])
                movie_scores[alg].append((movie_id, ndcg_val))

        for alg in algorithms:
            per_movie = {movie_id: ndcg for movie_id, ndcg in movie_scores[alg]}
            acc[alg]["seed_scores"][seed] = (
                sum(per_movie.values()) / len(per_movie) if per_movie else 0.0
            )
            acc[alg]["per_item_scores"][seed] = per_movie

    metrics = []
    for alg, info in acc.items():
        mean_v, std_v = _summarize(info["seed_scores"])
        metrics.append({
            "algorithm": alg,
            "seed_scores": info["seed_scores"],
            "per_movie_scores": info["per_item_scores"],
            "per_movie_costs": info["per_item_costs"],
            "score_mean": mean_v,
            "score_std": std_v,
            "price": _price(args.model, info["in_tokens"], info["out_tokens"]),
            "tokens": info["in_tokens"] + info["out_tokens"],
        })

    return {
        "dataset": "sembench_movie",
        "generated_at": _now_iso(),
        "settings": {
            "csv": args.movie_csv,
            "top_k_movies": args.movie_top_k,
            "review_limit": args.movie_review_limit,
            "model": args.model,
            "seeds": args.seeds,
            "algorithms": [m["algorithm"] for m in metrics],
        },
        "metrics": metrics,
        "metric_name": "ndcg@10",
    }


def _output_path(dataset: str, model: str) -> Path:
    """Auto-derive output path: test/<dataset>/results_<model>.json"""
    safe_model = model.replace("/", "-")
    return PROJECT_ROOT / "test" / dataset / f"results_{safe_model}.json"


_RUNNERS = {
    "dl20": run_dl20,
    "population": run_population,
    "sembench_movie": run_sembench_movie,
    "nfcorpus": run_nfcorpus,
    "hellaswag": run_hellaswag,
}


class _TqdmHandler(logging.Handler):
    """Route all log records through tqdm.write so they don't corrupt progress bars."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def main():
    handler = _TqdmHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logging.root.setLevel(logging.WARNING)
    logging.root.handlers = [handler]
    load_env_file(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run test-set experiments.")
    parser.add_argument(
        "--dataset", required=True,
        help=f"Comma-separated list of datasets to run. Valid: {', '.join(_RUNNERS)}",
    )
    _DEFAULT_MODELS = "llama3.1-70b,claude-haiku-4-5,openai-gpt-5-mini"
    parser.add_argument(
        "--models",
        default=_DEFAULT_MODELS,
        help="Comma-separated list of model names to run (default: all three).",
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default="cortex",
        help=PROVIDER_HELP,
    )
    parser.add_argument("--dl20-run-file", default="data/run.msmarco-v1-passage.bm25-default.dl20.txt")
    parser.add_argument("--hit-depth", type=int, default=100)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--population-csv", default="data/population_by_country_2020.csv")
    parser.add_argument(
        "--population-limit",
        type=int,
        default=None,
        help="Optional limit for population rows (e.g., 10 for quick sanity checks).",
    )
    parser.add_argument(
        "--movie-csv",
        default="data/movie/rotten_tomatoes_movie_reviews.csv",
        help="Path to the Rotten Tomatoes reviews CSV.",
    )
    parser.add_argument(
        "--movie-bm25-run",
        default="data/run.sembench_movie.bm25-sentiment.txt",
        help="BM25 TREC-format run file for movie reviews.",
    )
    parser.add_argument(
        "--movie-top-k",
        type=int,
        default=5,
        help="Number of top-reviewed movies to include (default: 5).",
    )
    parser.add_argument(
        "--movie-review-limit",
        type=int,
        default=None,
        help="Max reviews per movie (default: all reviews).",
    )
    parser.add_argument(
        "--nfcorpus-dir",
        default="data/nfcorpus",
        help="Path to the NFCorpus data directory (default: data/nfcorpus).",
    )
    parser.add_argument(
        "--nfcorpus-limit",
        type=int,
        default=3,
        help="Query limit for NFCorpus (default: 3).",
    )
    parser.add_argument(
        "--hellaswag-dir",
        default="data/hellaswag",
        help="Path to the HellaSwag data directory (default: data/hellaswag).",
    )
    parser.add_argument(
        "--hellaswag-num-queries",
        type=int,
        default=100,
        help="Number of HellaSwag questions to pool (default 100). The candidate "
             "corpus is the endings of these questions (4 each), so N -> 4N docs "
             "(100 -> 400). Larger N = harder/larger-cardinality task. Use 200 "
             "for the full 800-doc pool.",
    )
    parser.add_argument(
        "--hellaswag-limit",
        type=int,
        default=None,
        help="Optional further cap on how many of the pooled queries to actually "
             "rank (for smoke tests). The pool/corpus size is set by "
             "--hellaswag-num-queries, not this.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output path (only valid with a single --models and single "
             "--dataset). Default: test/<dataset>/results_<model>.json.",
    )
    args = parser.parse_args()
    args.seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    if not args.seeds:
        raise ValueError("At least one seed is required.")

    datasets = [d.strip() for d in args.dataset.split(",") if d.strip()]
    for d in datasets:
        if d not in _RUNNERS:
            parser.error(f"Invalid dataset '{d}'. Valid: {', '.join(_RUNNERS)}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    client = build_client(args.provider)

    async def _run_one(model: str, dataset: str, pos: int):
        import copy
        model_args = copy.copy(args)
        model_args.model = model
        model_args.dataset = dataset

        n_seeds = len(model_args.seeds)
        unit = {"population": "seed", "sembench_movie": "movie"}.get(dataset, "query")
        pbar = tqdm(
            total=n_seeds,
            desc=f"[{model}/{dataset}]",
            position=pos * 2,
            leave=True,
            unit=unit,
        )
        alg_pbar = tqdm(
            total=0,
            desc=f"  alg: {'':35s}",
            bar_format="{desc}",
            position=pos * 2 + 1,
            leave=True,
        )

        payload = await _RUNNERS[dataset](model_args, client, pbar=pbar, alg_pbar=alg_pbar)

        pbar.set_description(f"[{model}/{dataset}] done")
        alg_pbar.set_description(f"  alg: {'':35s}")
        pbar.close()
        alg_pbar.close()

        if args.output and len(models) == 1 and len(datasets) == 1:
            output = Path(args.output)
            if not output.is_absolute():
                output = PROJECT_ROOT / output
        else:
            if args.output:
                tqdm.write("  [output] --output ignored: only valid with a single model and dataset; "
                           "falling back to the auto-derived path.")
            output = _output_path(dataset, model)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tqdm.write(f"Wrote results to {output}")

    async def _run_all():
        for dataset in datasets:
            tqdm.write(f"\n{'='*60}\nDataset: {dataset}\n{'='*60}")
            await asyncio.gather(*[_run_one(m, dataset, i) for i, m in enumerate(models)])

    run_async(_run_all(), client)


if __name__ == "__main__":
    main()
