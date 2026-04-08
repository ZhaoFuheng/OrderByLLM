import argparse
import asyncio
import json
import logging
import os
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import ir_datasets
import pandas as pd
import pytrec_eval
from openai import AsyncOpenAI
from tqdm import tqdm

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
from order_by.utils import build_client, kendalltau_distance, load_movie_reviews, tokens2price
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

# DL20 algorithms — no search variants, batch size 4 only (same as DL19)
DL20_ALGORITHMS = [
    "bm25",
    "pointwise",
    "quick_sort",
    "quick_sort3",
    "external_pointwise_4",
    "external_merge_sort_4",
    "external_bubble_sort_4",
]

# SembenchMovie algorithms — rank movie reviews by positivity
SEMBENCH_MOVIE_ALGORITHMS = [
    "bm25",
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


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


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


async def run_population(args, client: AsyncOpenAI, pbar: tqdm | None = None) -> dict:
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


# ── DL20 (ndcg@10, same structure as DL19) ───────────────────────────────────

def _build_dl20_data(run_path: Path, hit_depth: int):
    ds = ir_datasets.load("msmarco-passage/trec-dl-2020")
    docstore = ds.docs_store()
    query_map = {str(q.query_id): q.text for q in ds.queries_iter()}

    by_qid = defaultdict(list)
    with run_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            qid, _, docid, rank, _, _ = parts
            if int(rank) > hit_depth:
                continue
            if qid not in query_map:
                continue
            doc = docstore.get(docid)
            if doc is None:
                continue
            text = (doc.title + " " if getattr(doc, "title", None) else "") + doc.text
            by_qid[qid].append((int(rank), docid, text))

    first_stage = []
    bm25_by_qid = {}
    for qid, entries in sorted(by_qid.items(), key=lambda x: x[0]):
        entries.sort(key=lambda x: x[0])
        ranking = [(docid, text) for _, docid, text in entries]
        first_stage.append((qid, query_map[qid], ranking))
        bm25_by_qid[qid] = [docid for _, docid, _ in entries]

    qrels_by_qid = defaultdict(dict)
    for q in ds.qrels_iter():
        qrels_by_qid[str(q.query_id)][str(q.doc_id)] = int(q.relevance)
    qrels_by_qid = dict(qrels_by_qid)

    # DL20 only has qrels for 54 of the 200 queries — filter to those only.
    first_stage = [(qid, q, r) for qid, q, r in first_stage if qid in qrels_by_qid]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_by_qid, {"ndcg_cut.10"})

    return first_stage, evaluator, bm25_by_qid


async def _run_dl20_algorithms_once(
    ranking: list[tuple[str, str]],
    query: str,
    client: AsyncOpenAI,
    model: str,
    alg_pbar: tqdm | None = None,
):
    def _set_alg(name: str):
        if alg_pbar is not None:
            alg_pbar.set_description(f"  alg: {name:<35s}")

    assert len(ranking) == 100, f"Expected 100 docs per query, got {len(ranking)}"
    outputs = {}
    p_prompt  = _safe_prompt(passage_pointwise_prompt_template,           question=query)
    ep_prompt = _safe_prompt(passage_external_pointwise_prompt_template,  question=query)
    pw_prompt = _safe_prompt(passage_pairwise_comparison_prompt_template, question=query)
    ex_prompt = _safe_prompt(passage_external_comparison_prompt_template, question=query)

    _set_alg("pointwise")
    p_ids, p_scores, _, p_in, p_out = await pointwise_sort(
        ranking[:], client, p_prompt, model, float,
        key_class=PointwiseRelevanceKey, isPassage=True,
    )
    outputs["pointwise"] = (p_ids, p_scores, p_in, p_out)

    _set_alg("external_pointwise_4")
    ep_ids, ep_scores, _, ep_in, ep_out, _ = await external_pointwise_sort(
        ranking[:], external_values, client, ep_prompt, model, float,
        isPassage=True, memory_size=4,
    )
    outputs["external_pointwise_4"] = (ep_ids, ep_scores, ep_in, ep_out)

    _set_alg("quick_sort")
    q1_sorted, _, q1_in, q1_out = await quick_sort(
        ranking[:], client, pw_prompt, model, isPassage=True, vote=1, limit_k=10,
    )
    outputs["quick_sort"] = (_normalize_docids(q1_sorted), None, q1_in, q1_out)

    _set_alg("quick_sort3 | ext_bubble_4 | ext_merge_4  [parallel]")
    (
        (q3_sorted, _, q3_in, q3_out),
        (eb_sorted, _, eb_in, eb_out),
        (em_sorted, _, em_in, em_out),
    ) = await asyncio.gather(
        quick_sort(ranking[:], client, pw_prompt, model, isPassage=True, vote=3, limit_k=10),
        external_bubble_sort(ranking[:], external_comparisons, 4, client, ex_prompt, model, isPassage=True, limit_k=10),
        external_merge_sort(ranking[:], external_comparisons, 4, client, ex_prompt, model, isPassage=True, limit_k=10),
    )
    outputs["quick_sort3"]            = (_normalize_docids(q3_sorted), None, q3_in, q3_out)
    outputs["external_bubble_sort_4"] = (_normalize_docids(eb_sorted), None, eb_in, eb_out)
    outputs["external_merge_sort_4"]  = (_normalize_docids(em_sorted), None, em_in, em_out)

    if alg_pbar is not None:
        alg_pbar.set_description(f"  alg: {'done':<35s}")
    return outputs


async def run_dl20(args, client: AsyncOpenAI, pbar: tqdm | None = None, alg_pbar: tqdm | None = None) -> dict:
    first_stage, evaluator, bm25_by_qid = _build_dl20_data(_resolve(args.dl20_run_file), args.hit_depth)
    acc = _empty_acc(DL20_ALGORITHMS)

    for seed in args.seeds:
        rng = random.Random(seed)
        run_by_alg = {alg: {} for alg in DL20_ALGORITHMS}

        for qid, _, _ in first_stage:
            run_by_alg["bm25"][str(qid)] = _to_run_scores(bm25_by_qid[qid])

        if pbar is not None:
            sidx = args.seeds.index(seed) + 1
            pbar.reset(total=len(first_stage))
            pbar.set_description(f"[{args.model}] seed {sidx}/{len(args.seeds)}")

        for qid, query, ranking in first_stage:
            top_ranking = ranking[:]
            rng.shuffle(top_ranking)
            outputs = await _run_dl20_algorithms_once(
                top_ranking,
                query,
                client,
                args.model,
                alg_pbar=alg_pbar,
            )
            if pbar is not None:
                pbar.update(1)
            for alg, (docids, scores, in_t, out_t) in outputs.items():
                acc[alg]["in_tokens"] += in_t
                acc[alg]["out_tokens"] += out_t
                acc[alg]["per_item_costs"].setdefault(seed, {})[str(qid)] = _price(args.model, in_t, out_t)
                if scores is not None:
                    run_by_alg[alg][str(qid)] = {str(d): float(s) for d, s in zip(docids, scores)}
                else:
                    run_by_alg[alg][str(qid)] = {str(d): float(i + 1) for i, d in enumerate(docids)}

        for alg in DL20_ALGORITHMS:
            alg_metrics = evaluator.evaluate(run_by_alg[alg])
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
        "dataset": "dl20",
        "generated_at": _now_iso(),
        "settings": {
            "run_file": args.dl20_run_file,
            "hit_depth": args.hit_depth,
            "model": args.model,
            "seeds": args.seeds,
            "algorithms": [m["algorithm"] for m in metrics],
        },
        "metrics": metrics,
        "metric_name": "ndcg@10",
    }


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
    alg_pbar: tqdm | None = None,
):
    def _set_alg(name: str):
        if alg_pbar is not None:
            alg_pbar.set_description(f"  alg: {name:<35s}")

    outputs = {}

    _set_alg("pointwise")
    p_ids, p_scores, _, p_in, p_out = await pointwise_sort(
        ranking[:], client, movie_pointwise_prompt_template, model, float,
        key_class=PointwiseRelevanceKey, isPassage=False, isReview=True,
    )
    outputs["pointwise"] = (p_ids, p_scores, p_in, p_out)

    _set_alg("external_pointwise_4")
    ep_ids, ep_scores, _, ep_in, ep_out, _ = await external_pointwise_sort(
        ranking[:], external_values, client, movie_external_pointwise_prompt_template,
        model, float, isPassage=False, isReview=True, memory_size=4,
    )
    outputs["external_pointwise_4"] = (ep_ids, ep_scores, ep_in, ep_out)

    _set_alg("quick_sort")
    q1_sorted, _, q1_in, q1_out = await quick_sort(
        ranking[:], client, movie_pairwise_comparison_prompt_template,
        model, isPassage=False, vote=1, isReview=True, limit_k=10,
    )
    outputs["quick_sort"] = (_normalize_docids(q1_sorted), None, q1_in, q1_out)

    _set_alg("quick_sort3 | ext_merge_4  [parallel]")
    (
        (q3_sorted, _, q3_in, q3_out),
        (em_sorted, _, em_in, em_out),
    ) = await asyncio.gather(
        quick_sort(ranking[:], client, movie_pairwise_comparison_prompt_template,
                   model, isPassage=False, vote=3, isReview=True, limit_k=10),
        external_merge_sort(ranking[:], external_comparisons, 4, client,
                            movie_external_comparison_prompt_template, model, isPassage=False, isReview=True, limit_k=10),
    )
    outputs["quick_sort3"]           = (_normalize_docids(q3_sorted), None, q3_in, q3_out)
    outputs["external_merge_sort_4"] = (_normalize_docids(em_sorted), None, em_in, em_out)

    if alg_pbar is not None:
        alg_pbar.set_description(f"  alg: {'done':<35s}")
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

    acc = _empty_acc(SEMBENCH_MOVIE_ALGORITHMS)

    for seed in args.seeds:
        rng = random.Random(seed)
        movie_scores: dict[str, list[tuple[str, float]]] = {alg: [] for alg in SEMBENCH_MOVIE_ALGORITHMS}

        if pbar is not None:
            sidx = args.seeds.index(seed) + 1
            pbar.reset(total=len(first_stage) + 1)  # +1 for the bubble phase
            pbar.set_description(f"[{args.model}] seed {sidx}/{len(args.seeds)}")

        # Phase 1: fast algorithms — one movie at a time (pointwise, ext_pointwise,
        #   quick_sort, quick_sort3, ext_merge).  ext_bubble_sort is deferred.
        all_outputs:   dict[str, dict] = {}
        all_shuffled:  dict[str, list] = {}
        for movie_id, ranking in first_stage:
            shuffled = ranking[:]
            rng.shuffle(shuffled)
            all_shuffled[movie_id] = shuffled
            all_outputs[movie_id] = await _run_sembench_movie_algorithms_once(
                shuffled, client, args.model, alg_pbar=alg_pbar,
            )
            if pbar is not None:
                pbar.update(1)

        # Phase 2: ext_bubble_sort — run ALL movies in parallel so the slow
        #   algorithm's latency is paid once rather than once-per-movie.
        if alg_pbar is not None:
            alg_pbar.set_description(f"  alg: {'ext_bubble_sort_4 [all movies]':<35s}")
        tqdm.write(f"  [bubble] launching ext_bubble_sort_4 for all {len(first_stage)} movies in parallel")
        t0 = asyncio.get_event_loop().time()
        bubble_results = await asyncio.gather(*[
            external_bubble_sort(
                all_shuffled[movie_id][:], external_comparisons, 4, client,
                movie_external_comparison_prompt_template, args.model,
                isPassage=False, isReview=True, limit_k=10,
            )
            for movie_id, _ in first_stage
        ])
        elapsed = asyncio.get_event_loop().time() - t0
        tqdm.write(f"  [bubble] all done in {elapsed:.1f}s")
        if pbar is not None:
            pbar.update(1)

        for (movie_id, _), (eb_sorted, _, eb_in, eb_out) in zip(first_stage, bubble_results):
            all_outputs[movie_id]["external_bubble_sort_4"] = (
                _normalize_docids(eb_sorted), None, eb_in, eb_out
            )

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

        for alg in SEMBENCH_MOVIE_ALGORITHMS:
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



_DEFAULT_MODELS = "llama3.1-70b,openai-gpt-4.1"


def _output_path(dataset: str, model: str) -> Path:
    """Auto-derive output path: test/<dataset>/results_<model>.json"""
    safe_model = model.replace("/", "-")
    return PROJECT_ROOT / "test" / dataset / f"results_{safe_model}.json"


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
    _load_env_file(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run test-set experiments.")
    parser.add_argument("--dataset", choices=["dl20", "population", "sembench_movie"], required=True)
    parser.add_argument(
        "--models",
        default=_DEFAULT_MODELS,
        help="Comma-separated list of model names to run (default: all three).",
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
    args = parser.parse_args()
    args.seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    if not args.seeds:
        raise ValueError("At least one seed is required.")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    client = build_client()

    async def _run_one(model: str, pos: int):
        import copy
        model_args = copy.copy(args)
        model_args.model = model

        n_seeds = len(model_args.seeds)
        unit = "seed" if model_args.dataset == "population" else "movie" if model_args.dataset == "sembench_movie" else "query"
        pbar = tqdm(
            total=n_seeds,
            desc=f"[{model}]",
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

        if model_args.dataset == "dl20":
            payload = await run_dl20(model_args, client, pbar=pbar, alg_pbar=alg_pbar)
        elif model_args.dataset == "sembench_movie":
            payload = await run_sembench_movie(model_args, client, pbar=pbar, alg_pbar=alg_pbar)
        else:
            payload = await run_population(model_args, client, pbar=pbar)

        pbar.set_description(f"[{model}] done")
        alg_pbar.set_description(f"  alg: {'':35s}")
        pbar.close()
        alg_pbar.close()

        output = _output_path(model_args.dataset, model)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tqdm.write(f"Wrote results to {output}")

    async def _run_all():
        await asyncio.gather(*[_run_one(m, i) for i, m in enumerate(models)])

    asyncio.run(_run_all())


if __name__ == "__main__":
    main()
