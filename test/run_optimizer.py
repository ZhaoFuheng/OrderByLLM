import argparse
import asyncio
import json
import logging
import os
import random

import sys
from collections import Counter, defaultdict
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

from order_by.optimizer import OrderByOptimizer
from order_by.utils import build_client, kendalltau_distance, load_movie_reviews, tokens2price
from prompts.all_prompts import (
    direct_inquiry_factual_knowledge_prompt,
    llm_judge_prompt,
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

POPULATION_WIKI_FIELD = "population_estimate"

_POPULATION_FACTUAL_PROMPT = direct_inquiry_factual_knowledge_prompt.format_map(
    {"description": "Rank countries by their population in 2020", "query": "population of the country", "example": "```{example}```"}
)


# ── Helpers ───────────────────────────────────────────────────────────────────

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
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _output_path(dataset: str, model: str) -> Path:
    safe_model = model.replace("/", "-")
    return PROJECT_ROOT / "test" / dataset / f"optimizer_{safe_model}.json"


def _normalize_docids(sorted_data) -> list[str]:
    return [str(x[0]) if isinstance(x, tuple) else str(x) for x in sorted_data]


def _safe_prompt(template: str, **kwargs) -> str:
    class _Partial(dict):
        def __missing__(self, key):
            return "{" + key + "}"
    return template.format_map(_Partial(**kwargs))


# ── Population ────────────────────────────────────────────────────────────────

async def _run_optimizer_population(
    country_names: list[str],
    gold: list[str],
    client: AsyncOpenAI,
    args,
) -> dict:
    """Run optimizer on a single shuffled country list; return per-seed record."""
    opt = OrderByOptimizer(
        client=client,
        data=country_names[:],
        factual_knowledge_prompt_template=_POPULATION_FACTUAL_PROMPT,
        pointwise_prompt_template=population_pointwise_prompt_template,
        external_pointwise_prompt_template=population_external_pointwise_prompt_template,
        pairwise_comparison_prompt_template=population_pairwise_comparison_prompt_template,
        external_pairwise_prompt_template=population_external_comparison_prompt_template,
        dollar_budget_constraint=args.total_ranking_budget,
        model_name=args.model,
        isPassage=False,
        llm_judge_prompt_template=llm_judge_prompt.replace("{criteria}", "Rank countries by population from smallest to largest"),
        judge_model=args.judge_model,
        sample_size=args.sample_size,
        proxy_ground_truth_policy=args.proxy_policy,
        enable_factual_web_search=True,
        external_pointwise_memory_size=args.ext_point_batch,
        wiki_field=POPULATION_WIKI_FIELD,
    )
    (sorted_data, num_calls, in_tok, out_tok), chosen_alg, ranking_cost, opt_cost = \
        await opt.physical_order_by_impl()

    predicted = [str(x) for x in _normalize_docids(sorted_data)]
    score = float(kendalltau_distance(gold[:], predicted))

    total_cost = tokens2price(args.model, in_tok, out_tok)
    return {
        "chosen_alg": chosen_alg,
        "score": score,
        "ranking_budget": args.total_ranking_budget,
        "ranking_cost": total_cost - opt_cost,
        "optimization_cost": opt_cost,
        "in_tokens": in_tok,
        "out_tokens": out_tok,
    }


async def run_optimizer_population(args, client: AsyncOpenAI, pbar: tqdm | None = None) -> dict:
    df = pd.read_csv(_resolve(args.population_csv))
    if "Population (2020)" not in df.columns or "Country" not in df.columns:
        raise ValueError("Population CSV must have 'Population (2020)' and 'Country' columns.")
    if args.population_limit is not None:
        df = df.head(args.population_limit).copy()

    country_names = df["Country"].astype(str).tolist()
    gold = (
        df.sort_values(["Population (2020)", "Country"], ascending=[True, True])["Country"]
        .astype(str)
        .tolist()
    )

    shuffled = country_names[:]
    random.Random(0).shuffle(shuffled)
    rec = await _run_optimizer_population(shuffled, gold, client, args)
    if pbar is not None:
        pbar.update(1)

    return {
        "dataset": "population",
        "generated_at": _now_iso(),
        "settings": {
            "csv": args.population_csv,
            "model": args.model,
            "total_ranking_budget": args.total_ranking_budget,
            "sample_size": args.sample_size,
            "proxy_policy": args.proxy_policy,
        },
        "score_mean": round(float(rec["score"]), 3),
        "score_std": 0.0,
        "ranking_cost": rec["ranking_cost"],
        "optimization_cost": rec["optimization_cost"],
        "chosen_alg": rec["chosen_alg"],
        "in_tokens": rec["in_tokens"],
        "out_tokens": rec["out_tokens"],
        "metric_name": "kendalltau",
    }


# ── DL20 ──────────────────────────────────────────────────────────────────────

def _build_dl20_data(run_path: Path, hit_depth: int):
    ds = ir_datasets.load("msmarco-passage/trec-dl-2020")
    docstore = ds.docs_store()
    query_map = {str(q.query_id): q.text for q in ds.queries_iter()}

    by_qid = defaultdict(list)
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

    first_stage, bm25_by_qid = [], {}
    for qid, entries in sorted(by_qid.items()):
        entries.sort(key=lambda x: x[0])
        ranking = [(docid, text) for _, docid, text in entries]
        first_stage.append((qid, query_map[qid], ranking))
        bm25_by_qid[qid] = [docid for _, docid, _ in entries]

    qrels_by_qid = defaultdict(dict)
    for q in ds.qrels_iter():
        qrels_by_qid[str(q.query_id)][str(q.doc_id)] = int(q.relevance)
    qrels_by_qid = dict(qrels_by_qid)

    first_stage = [(qid, q, r) for qid, q, r in first_stage if qid in qrels_by_qid]
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_by_qid, {"ndcg_cut.10"})
    return first_stage, evaluator, qrels_by_qid


async def _run_optimizer_dl20_query(
    qid: str,
    query: str,
    ranking: list[tuple[str, str]],
    client: AsyncOpenAI,
    args,
    budget: float = 0.0,
    ideal_oracle: dict | None = None,
) -> dict:
    """Run optimizer on a single DL20 query with a pre-allocated budget slice."""
    p_prompt  = _safe_prompt(passage_pointwise_prompt_template,           question=query)
    ep_prompt = _safe_prompt(passage_external_pointwise_prompt_template,  question=query)
    pw_prompt = _safe_prompt(passage_pairwise_comparison_prompt_template, question=query)
    ex_prompt = _safe_prompt(passage_external_comparison_prompt_template, question=query)

    opt = OrderByOptimizer(
        client=client,
        data=ranking[:],
        factual_knowledge_prompt_template=direct_inquiry_factual_knowledge_prompt.format_map(
            {"description": "Rank passages by relevance to a query", "query": query, "example": "```{example}```"}
        ),
        pointwise_prompt_template=p_prompt,
        external_pointwise_prompt_template=ep_prompt,
        pairwise_comparison_prompt_template=pw_prompt,
        external_pairwise_prompt_template=ex_prompt,
        dollar_budget_constraint=budget,
        model_name=args.model,
        isPassage=True,
        llm_judge_prompt_template=llm_judge_prompt.replace("{criteria}", f"Rank passages by relevance to the query: {query}"),
        judge_model=args.judge_model,
        sample_size=min(args.sample_size, len(ranking)),
        proxy_ground_truth_policy=args.proxy_policy,
        k=10,
        enable_factual_web_search=False,
        external_pointwise_memory_size=args.ext_point_batch,
        has_id_and_row=True,
        ideal_oracle=ideal_oracle,
    )
    (sorted_data, num_calls, in_tok, out_tok), chosen_alg, _, opt_cost = \
        await opt.physical_order_by_impl()

    total_cost = tokens2price(args.model, in_tok, out_tok)
    doc_ids = _normalize_docids(sorted_data)
    return {
        "qid": qid,
        "chosen_alg": chosen_alg,
        "doc_ids": doc_ids,
        "ranking_budget": budget,
        "ranking_cost": total_cost - opt_cost,
        "optimization_cost": opt_cost,
        "in_tokens": in_tok,
        "out_tokens": out_tok,
    }


def _load_oracle_data(model: str, dataset: str):
    """Load per-query best algorithm, per-algorithm ndcg scores, and per-query costs
    from a pre-computed results JSON.  Returns (oracle_best, alg_scores, alg_costs) where:
      oracle_best  : {qid: best_alg_name}
      alg_scores   : {qid: {alg: ndcg}}
      alg_costs    : {qid: {alg: cost}}
    Returns (None, None, None) if the file is missing or unreadable.
    """
    results_path = PROJECT_ROOT / "test" / dataset / f"results_{model}.json"
    if not results_path.exists():
        tqdm.write(f"  [oracle] results file not found: {results_path}")
        return None, None, None
    try:
        data = json.load(results_path.open())
        alg_scores: dict[str, dict[str, float]] = {}
        alg_costs: dict[str, dict[str, float]] = {}
        skip_algs = {"bm25"}
        for m in data.get("metrics", []):
            alg = m["algorithm"]
            if alg in skip_algs:
                continue
            score_key = "per_query_scores" if "per_query_scores" in m else "per_movie_scores"
            for seed_scores in m.get(score_key, {}).values():
                for qid, score in seed_scores.items():
                    alg_scores.setdefault(qid, {})
                    if alg not in alg_scores[qid] or score > alg_scores[qid][alg]:
                        alg_scores[qid][alg] = score
            cost_key = "per_query_costs" if "per_query_costs" in m else "per_movie_costs"
            for seed_costs in m.get(cost_key, {}).values():
                for qid, cost in seed_costs.items():
                    alg_costs.setdefault(qid, {})
                    alg_costs[qid][alg] = cost
        oracle_best = {qid: max(scores, key=scores.get) for qid, scores in alg_scores.items()}
        return oracle_best, alg_scores, alg_costs
    except Exception as e:
        tqdm.write(f"  [oracle] failed to load results file: {e}")
        return None, None, None


async def run_optimizer_dl20(args, client: AsyncOpenAI, pbar: tqdm | None = None) -> dict:
    first_stage, evaluator, qrels_by_qid = _build_dl20_data(_resolve(args.dl20_run_file), args.hit_depth)

    query_items = [(qid, query, ranking, len(ranking)) for qid, query, ranking in first_stage]

    # Load oracle best-per-query from the pre-computed results file (optional).
    oracle_best, oracle_alg_scores, oracle_alg_costs = _load_oracle_data(args.model, "dl20")
    if oracle_best:
        tqdm.write(f"  [oracle] loaded best-alg reference for {len(oracle_best)} queries from results_{args.model}.json")

    rng = random.Random(0)
    run_dict: dict[str, dict[str, float]] = {}
    per_query_records: list[dict] = []
    total_in, total_out, total_opt_budget = 0, 0, 0.0

    # Fixed budget per query.
    per_query_budget = args.total_ranking_budget / len(query_items) if query_items else 0.0

    if pbar is not None:
        pbar.reset(total=len(query_items))
        pbar.set_description(f"[{args.model}]")

    for qid, query, ranking, n_docs in query_items:
        ranking_budget = per_query_budget

        shuffled = ranking[:]
        rng.shuffle(shuffled)
        ideal = qrels_by_qid.get(str(qid)) if args.proxy_policy == "ideal" else None
        rec = await _run_optimizer_dl20_query(qid, query, shuffled, client, args, budget=ranking_budget, ideal_oracle=ideal)

        chosen = rec["chosen_alg"]

        doc_ids = rec["doc_ids"]
        run_dict[str(qid)] = {str(d): float(i + 1) for i, d in enumerate(doc_ids)}

        chosen_ndcg = float(
            evaluator.evaluate({str(qid): run_dict[str(qid)]})
            .get(str(qid), {})
            .get("ndcg_cut_10", float("nan"))
        )

        if oracle_best and oracle_alg_scores:
            best      = oracle_best.get(str(qid), "?")
            best_ndcg = oracle_alg_scores.get(str(qid), {}).get(best, float("nan"))
            best_cost = oracle_alg_costs.get(str(qid), {}).get(best, float("nan")) if oracle_alg_costs else float("nan")
            is_match  = chosen_ndcg >= best_ndcg
            color = "\033[32m" if is_match else "\033[34m"
            mark = "✓" if is_match else "✗"
            cost_str = f"cost=${best_cost:.4f}  " if best_cost == best_cost else ""
            tqdm.write(
                f"{color}  qid={qid:<12s}  "
                f"optimizer={chosen:<26s}ndcg={chosen_ndcg:.3f}  "
                f"oracle_best={best:<26s}ndcg={best_ndcg:.3f}  {cost_str}{mark}\033[0m"
            )

        else:
            tqdm.write(f"  qid={qid:<12s}  optimizer={chosen:<26s}ndcg={chosen_ndcg:.3f}")

        total_in  += rec["in_tokens"]
        total_out += rec["out_tokens"]
        total_opt_budget += rec["optimization_cost"]
        per_query_records.append({
            "qid": qid,
            "chosen_alg": chosen,
            "ranking_budget": ranking_budget,
            "ranking_cost": rec["ranking_cost"],
            "optimization_cost": rec["optimization_cost"],
        })
        if pbar is not None:
            pbar.update(1)

    if oracle_best and oracle_alg_scores:
        correct = sum(
            1 for r in per_query_records
            if float(evaluator.evaluate({str(r["qid"]): run_dict[str(r["qid"])]})
                     .get(str(r["qid"]), {}).get("ndcg_cut_10", -1))
               >= oracle_alg_scores.get(str(r["qid"]), {}).get(oracle_best.get(str(r["qid"]), ""), float("inf"))
        )
        total = len(per_query_records)
        tqdm.write(f"  [oracle accuracy] {correct}/{total} queries matched oracle-best ({correct/total*100:.1f}%)")

    alg_metrics = evaluator.evaluate(run_dict)
    per_query_scores = {qid: float(m["ndcg_cut_10"]) for qid, m in alg_metrics.items()}
    score = sum(per_query_scores.values()) / len(per_query_scores) if per_query_scores else 0.0

    alg_counts = dict(Counter(r["chosen_alg"] for r in per_query_records))
    total_ranking_cost = tokens2price(args.model, total_in, total_out) - total_opt_budget
    return {
        "dataset": "dl20",
        "generated_at": _now_iso(),
        "settings": {
            "run_file": args.dl20_run_file,
            "hit_depth": args.hit_depth,
            "model": args.model,
            "total_ranking_budget": args.total_ranking_budget,
            "sample_size": args.sample_size,
            "proxy_policy": args.proxy_policy,
        },
        "score_mean": round(float(score), 3),
        "score_std": 0.0,
        "per_query_scores": per_query_scores,
        "per_query_chosen_alg": {r["qid"]: r["chosen_alg"] for r in per_query_records},
        "chosen_alg_counts": alg_counts,
        "total_ranking_budget": args.total_ranking_budget,
        "per_query_budget": per_query_budget,
        "total_optimization_cost": total_opt_budget,
        "total_ranking_cost": total_ranking_cost,
        "in_tokens": total_in,
        "out_tokens": total_out,
        "metric_name": "ndcg@10",
    }


# ── SembenchMovie ─────────────────────────────────────────────────────────────

def _build_sembench_movie_data(
    csv_path: Path,
    top_k_reviewed_movies: int = 5,
    review_limit: int | None = None,
):
    return load_movie_reviews(csv_path, top_k_reviewed_movies, review_limit)


async def _run_optimizer_movie(
    movie_id: str,
    ranking: list[tuple[str, str]],
    client: AsyncOpenAI,
    args,
    budget: float = 0.0,
    ideal_oracle: dict | None = None,
) -> dict:
    """Run optimizer on a single movie's reviews with a pre-allocated budget slice."""
    opt = OrderByOptimizer(
        client=client,
        data=ranking[:],
        factual_knowledge_prompt_template=direct_inquiry_factual_knowledge_prompt.format_map(
            {"description": "Rank movie reviews by positivity", "query": "positivity of the review", "example": "```{example}```"}
        ),
        pointwise_prompt_template=movie_pointwise_prompt_template,
        external_pointwise_prompt_template=movie_external_pointwise_prompt_template,
        pairwise_comparison_prompt_template=movie_pairwise_comparison_prompt_template,
        external_pairwise_prompt_template=movie_external_comparison_prompt_template,
        dollar_budget_constraint=budget,
        model_name=args.model,
        isPassage=False,
        isReview=True,
        llm_judge_prompt_template=llm_judge_prompt.replace("{criteria}", "Rank movie reviews from most negative to most positive"),
        judge_model=args.judge_model,
        sample_size=min(args.sample_size, len(ranking)),
        proxy_ground_truth_policy=args.proxy_policy,
        k=10,
        enable_factual_web_search=False,
        external_pointwise_memory_size=args.ext_point_batch,
        has_id_and_row=True,
        ideal_oracle=ideal_oracle,
    )
    (sorted_data, num_calls, in_tok, out_tok), chosen_alg, ranking_budget, opt_budget = \
        await opt.physical_order_by_impl()

    total_cost = tokens2price(args.model, in_tok, out_tok)
    return {
        "movie_id": movie_id,
        "chosen_alg": chosen_alg,
        "doc_ids": _normalize_docids(sorted_data),
        "ranking_budget": budget,
        "ranking_cost": total_cost - opt_budget,
        "optimization_cost": opt_budget,
        "in_tokens": in_tok,
        "out_tokens": out_tok,
    }


async def run_optimizer_sembench_movie(
    args, client: AsyncOpenAI, pbar: tqdm | None = None
) -> dict:
    first_stage, gold_by_movie, qrels_by_movie = _build_sembench_movie_data(
        _resolve(args.movie_csv),
        top_k_reviewed_movies=args.movie_top_k,
        review_limit=args.movie_review_limit,
    )

    movie_items = [(movie_id, ranking, len(ranking)) for movie_id, ranking in first_stage]

    # Load oracle best-per-movie from the pre-computed results file (optional).
    oracle_best, oracle_alg_scores, oracle_alg_costs = _load_oracle_data(args.model, "sembench_movie")
    if oracle_best:
        tqdm.write(f"  [oracle] loaded best-alg reference for {len(oracle_best)} movies from results_{args.model}.json")

    rng = random.Random(0)
    per_movie_scores: dict[str, float] = {}
    per_movie_records: list[dict] = []
    total_in, total_out, total_opt_budget = 0, 0, 0.0

    per_movie_budget = {
        mid: args.total_ranking_budget / len(movie_items)
        for mid, _, n in movie_items
    }

    if pbar is not None:
        pbar.reset(total=len(movie_items))
        pbar.set_description(f"[{args.model}]")

    for movie_id, ranking, n_reviews in movie_items:
        ranking_budget = per_movie_budget[movie_id]

        shuffled = ranking[:]
        rng.shuffle(shuffled)
        ideal = qrels_by_movie.get(movie_id) if args.proxy_policy == "ideal" else None
        rec = await _run_optimizer_movie(movie_id, shuffled, client, args, budget=ranking_budget, ideal_oracle=ideal)

        n = len(rec["doc_ids"])
        run = {movie_id: {rid: float(i + 1) for i, rid in enumerate(rec["doc_ids"])}}
        movie_evaluator = pytrec_eval.RelevanceEvaluator(
            {movie_id: qrels_by_movie[movie_id]}, {"ndcg_cut.10"}
        )
        result = movie_evaluator.evaluate(run)
        ndcg_val = float(result[movie_id]["ndcg_cut_10"])
        per_movie_scores[movie_id] = ndcg_val

        chosen = rec["chosen_alg"]
        if oracle_best and oracle_alg_scores:
            best      = oracle_best.get(movie_id, "?")
            best_ndcg = oracle_alg_scores.get(movie_id, {}).get(best, float("nan"))
            best_cost = oracle_alg_costs.get(movie_id, {}).get(best, float("nan")) if oracle_alg_costs else float("nan")
            is_match  = ndcg_val >= best_ndcg
            color = "\033[32m" if is_match else "\033[34m"
            mark = "✓" if is_match else "✗"
            cost_str = f"cost=${best_cost:.4f}  " if best_cost == best_cost else ""
            tqdm.write(
                f"{color}  movie={movie_id:<20s}  "
                f"optimizer={chosen:<26s}ndcg={ndcg_val:.3f}  "
                f"oracle_best={best:<26s}ndcg={best_ndcg:.3f}  {cost_str}{mark}\033[0m"
            )
        else:
            tqdm.write(f"  movie={movie_id:<20s}  optimizer={chosen:<26s}ndcg={ndcg_val:.3f}")

        total_in  += rec["in_tokens"]
        total_out += rec["out_tokens"]
        total_opt_budget += rec["optimization_cost"]
        per_movie_records.append({
            "movie_id": movie_id,
            "chosen_alg": rec["chosen_alg"],
            "score": ndcg_val,
            "ranking_budget": ranking_budget,
            "ranking_cost": rec["ranking_cost"],
            "optimization_cost": rec["optimization_cost"],
        })
        if pbar is not None:
            pbar.update(1)

    if oracle_best and oracle_alg_scores:
        correct = sum(
            1 for r in per_movie_records
            if r["score"] >= oracle_alg_scores.get(r["movie_id"], {}).get(oracle_best.get(r["movie_id"], ""), float("inf"))
        )
        total = len(per_movie_records)
        tqdm.write(f"  [oracle accuracy] {correct}/{total} movies matched oracle-best ({correct/total*100:.1f}%)")

    score = sum(per_movie_scores.values()) / len(per_movie_scores) if per_movie_scores else 0.0
    alg_counts = dict(Counter(r["chosen_alg"] for r in per_movie_records))
    total_ranking_cost = tokens2price(args.model, total_in, total_out) - total_opt_budget
    return {
        "dataset": "sembench_movie",
        "generated_at": _now_iso(),
        "settings": {
            "csv": args.movie_csv,
            "top_k_movies": args.movie_top_k,
            "review_limit": args.movie_review_limit,
            "model": args.model,
            "total_ranking_budget": args.total_ranking_budget,
            "sample_size": args.sample_size,
            "proxy_policy": args.proxy_policy,
        },
        "score_mean": round(float(score), 3),
        "score_std": 0.0,
        "per_movie_scores": per_movie_scores,
        "per_movie_chosen_alg": {r["movie_id"]: r["chosen_alg"] for r in per_movie_records},
        "chosen_alg_counts": alg_counts,
        "total_ranking_budget": args.total_ranking_budget,
        "per_movie_budget": per_movie_budget,
        "total_optimization_cost": total_opt_budget,
        "total_ranking_cost": total_ranking_cost,
        "in_tokens": total_in,
        "out_tokens": total_out,
        "metric_name": "ndcg@10",
    }


# ── Entry point ───────────────────────────────────────────────────────────────


class _TqdmHandler(logging.Handler):
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

    parser = argparse.ArgumentParser(description="Run OrderByOptimizer on benchmark datasets.")
    parser.add_argument("--dataset", choices=["population", "dl20", "sembench_movie"], required=True)
    parser.add_argument("--models", required=True,
                        help="Comma-separated list of model names. Example: --models llama3.1-70b,openai-gpt-4.1")
    parser.add_argument("--judge-model", default="openai-gpt-4.1",
                        help="Model used as the LLM judge for proxy ground-truth (default: openai-gpt-4.1).")
    parser.add_argument("--budgets", default="0.10",
                        help="Comma-separated list of total dollar budgets to sweep over (default: 0.10). "
                             "Example: --budgets 0.01,0.02,0.05,0.10,0.20")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0).")
    parser.add_argument("--sample-size", type=int, default=20,
                        help="Number of items to probe during optimization (default: 20).")
    parser.add_argument("--proxy-policies", default="borda",
                        help="Comma-separated list of proxy ground-truth policies to sweep over. "
                             "Choices: borda, llm_judge, ideal (default: borda). "
                             "Example: --proxy-policies borda,llm_judge")
    parser.add_argument("--ext-point-batch", type=int, default=8,
                        help="Batch size for external-pointwise during optimization (default: 8).")

    # Population-specific
    parser.add_argument("--population-csv", default="data/population_by_country_2020.csv")
    parser.add_argument("--population-limit", type=int, default=None,
                        help="Optional row limit for quick sanity checks.")

    # DL20-specific
    parser.add_argument("--dl20-run-file",
                        default="data/run.msmarco-v1-passage.bm25-default.dl20.txt")
    parser.add_argument("--hit-depth", type=int, default=100)

    # SembenchMovie-specific
    parser.add_argument("--movie-csv", default="data/movie/rotten_tomatoes_movie_reviews.csv")
    parser.add_argument("--movie-top-k", type=int, default=5)
    parser.add_argument("--movie-review-limit", type=int, default=None)

    # Output
    parser.add_argument("--output", default=None,
                        help="Override output path (default: test/<dataset>/optimizer_<model>.json).")

    args = parser.parse_args()
    args.models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    if not args.models:
        raise ValueError("At least one model is required.")
    args.seeds = [args.seed]
    args.budgets = [float(b.strip()) for b in str(args.budgets).split(",") if b.strip()]
    if not args.budgets:
        raise ValueError("At least one budget is required.")
    valid_policies = {"borda", "llm_judge", "ideal"}
    args.proxy_policies = [p.strip() for p in str(args.proxy_policies).split(",") if p.strip()]
    invalid = set(args.proxy_policies) - valid_policies
    if invalid:
        raise ValueError(f"Invalid proxy policies: {invalid}. Must be one of {valid_policies}.")
    if not args.proxy_policies:
        raise ValueError("At least one proxy policy is required.")

    client = build_client()

    unit_map = {"population": "seed", "dl20": "query", "sembench_movie": "movie"}

    async def _run():
        model_results = {}
        for model in args.models:
            args.model = model
            policy_results = {}
            for policy in args.proxy_policies:
                args.proxy_policy = policy
                budget_results = {}
                for budget in args.budgets:
                    args.total_ranking_budget = budget
                    budget_str = f"{budget:.4f}".rstrip("0").rstrip(".")
                    tqdm.write(
                        f"\n[model={model}] [policy={policy}] [budget={budget_str}] "
                        f"[sample_size={args.sample_size}] starting..."
                    )
                    pbar = tqdm(
                        total=1,
                        desc=f"[{model}] policy={policy} budget={budget_str} sample_size={args.sample_size}",
                        unit=unit_map[args.dataset],
                        leave=True,
                    )
                    if args.dataset == "population":
                        result = await run_optimizer_population(args, client, pbar=pbar)
                    elif args.dataset == "dl20":
                        result = await run_optimizer_dl20(args, client, pbar=pbar)
                    else:
                        result = await run_optimizer_sembench_movie(args, client, pbar=pbar)
                    pbar.close()
                    budget_results[budget_str] = result
                    score_mean = float(result.get("score_mean", float("nan")))
                    total_ranking_cost = float(
                        result.get("total_ranking_cost", result.get("ranking_cost", float("nan")))
                    )
                    tqdm.write(
                        f"[model={model}] [policy={policy}] [budget={budget_str}] "
                        f"[sample_size={args.sample_size}] [score_mean={score_mean:.3f}] "
                        f"[total_ranking_cost={total_ranking_cost:.4f}] done."
                    )
                policy_results[policy] = budget_results
            model_results[model] = policy_results
        return model_results

    all_results = asyncio.run(_run())

    payload = {
        "dataset": args.dataset,
        "models": args.models,
        "budgets": args.budgets,
        "proxy_policies": args.proxy_policies,
        "results_by_model": all_results,
    }

    out = Path(args.output) if args.output else _output_path(args.dataset, "_".join(args.models))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tqdm.write(f"Wrote results to {out}")


if __name__ == "__main__":
    main()
