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
from order_by.utils import build_client, kendalltau_distance, tokens2price
from prompts.all_prompts import (
    nba_external_comparison_prompt_template,
    nba_external_pointwise_prompt_template,
    nba_pairwise_comparison_prompt_template,
    nba_pointwise_prompt_template,
    passage_external_comparison_prompt_template,
    passage_external_pointwise_prompt_template,
    passage_pairwise_comparison_prompt_template,
    passage_pointwise_prompt_template,
)


# NBA algorithms (includes wiki-search variants)
ALGORITHMS = [
    "pointwise",
    "pointwise_with_search",
    "external_pointwise_4",
    "external_pointwise_4_with_search",
    "quick_sort",
    "quick_sort3",
    "external_bubble_sort_4",
    "external_merge_sort_4",
]

# DL19 algorithms — no search variants, batch size 4 only
DL19_ALGORITHMS = [
    "bm25",
    "pointwise",
    "quick_sort",
    "quick_sort3",
    "external_pointwise_4",
    "external_merge_sort_4",
    "external_bubble_sort_4",
]

EXTERNAL_POINTWISE_MEMORY_SIZES = (4,)

# Wikipedia infobox field label used for NBA height lookups.
NBA_WIKI_FIELD = "Listed height"


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


def _resolve(p: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT when it is not absolute."""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def _empty_acc(algs: list[str] = None) -> dict:
    return {
        alg: {"in_tokens": 0, "out_tokens": 0, "seed_scores": {}, "per_item_scores": {}}
        for alg in (algs or ALGORITHMS)
    }


async def _run_nba_algorithms_once(
    data_names: list[str],
    client: AsyncOpenAI,
    model: str,
    seed: int = 0,
):
    outputs = {}
    pbar = tqdm(total=len(ALGORITHMS), desc=f"  seed={seed}", unit="alg", leave=False)

    async def _run(name, coro):
        pbar.set_postfix_str(name)
        result = await coro
        outputs[name] = result
        pbar.update(1)

    await _run("pointwise", _wrap(pointwise_sort(
        data_names[:], client, nba_pointwise_prompt_template, model, float
    )))

    await _run("pointwise_with_search", _wrap(pointwise_sort(
        data_names[:], client, nba_pointwise_prompt_template, model, float,
        use_wiki=True,
        wiki_field=NBA_WIKI_FIELD,
    )))

    for m in EXTERNAL_POINTWISE_MEMORY_SIZES:
        await _run(f"external_pointwise_{m}", _wrap(external_pointwise_sort(
            data_names[:], external_values, client,
            nba_external_pointwise_prompt_template, model, float,
            isPassage=False, memory_size=m,
        )))
        await _run(f"external_pointwise_{m}_with_search", _wrap(external_pointwise_sort(
            data_names[:], external_values, client,
            nba_external_pointwise_prompt_template, model, float,
            isPassage=False, memory_size=m,
            wiki_field=NBA_WIKI_FIELD,
        )))

    await _run("quick_sort", _wrap(quick_sort(
        data_names[:], client, nba_pairwise_comparison_prompt_template,
        model, isPassage=False, vote=1,
    )))

    await _run("quick_sort3", _wrap(quick_sort(
        data_names[:], client, nba_pairwise_comparison_prompt_template,
        model, isPassage=False, vote=3,
    )))

    for m in EXTERNAL_POINTWISE_MEMORY_SIZES:
        pbar.set_postfix_str(f"ext_bubble_{m} | ext_merge_{m}  [parallel]")
        (bubble_result, merge_result) = await asyncio.gather(
            _wrap(external_bubble_sort(
                data_names[:], external_comparisons, m, client,
                nba_external_comparison_prompt_template, model, isPassage=False,
            )),
            _wrap(external_merge_sort(
                data_names[:], external_comparisons, m, client,
                nba_external_comparison_prompt_template, model, isPassage=False,
            )),
        )
        outputs[f"external_bubble_sort_{m}"] = bubble_result
        outputs[f"external_merge_sort_{m}"]  = merge_result
        pbar.update(2)

    pbar.close()
    return outputs


def _wrap(coro):
    """Normalise sort results to a uniform (sorted_data, in_tokens, out_tokens) tuple."""
    async def _inner():
        result = await coro
        # pointwise_sort non-passage returns (sorted, api_calls, in, out)
        # external_pointwise_sort non-passage returns (sorted, api_calls, in, out)
        # quick_sort / bubble / merge return (sorted, api_calls, in, out)
        return (result[0], result[2], result[3])
    return _inner()



async def run_nba(args, client: AsyncOpenAI, pbar: tqdm | None = None) -> dict:
    df = pd.read_csv(_resolve(args.nba_csv))
    if "h_meters" not in df.columns or "full_name" not in df.columns:
        raise ValueError("NBA CSV must contain 'h_meters' and 'full_name'.")
    if args.nba_limit is not None:
        if args.nba_limit <= 1:
            raise ValueError("--nba-limit must be greater than 1.")
        df = df.head(args.nba_limit).copy()

    names = df["full_name"].astype(str).tolist()
    gold = (
        df.sort_values(by=["h_meters", "full_name"], ascending=[True, True])["full_name"]
        .astype(str)
        .tolist()
    )
    acc = _empty_acc(ALGORITHMS)

    for i, seed in enumerate(args.seeds, 1):
        if pbar is not None:
            pbar.set_description(f"[{args.model}] seed {i}/{len(args.seeds)}")
        shuffled = names[:]
        random.Random(seed).shuffle(shuffled)
        outputs = await _run_nba_algorithms_once(shuffled[:], client, args.model, seed=seed)

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
        "dataset": "nba",
        "generated_at": _now_iso(),
        "settings": {
            "csv": args.nba_csv,
            "model": args.model,
            "seeds": args.seeds,
            "algorithms": [m["algorithm"] for m in metrics],
        },
        "metrics": metrics,
        "metric_name": "kendalltau",
    }


def _build_dl19_data(run_path: Path, hit_depth: int):
    ds = ir_datasets.load("msmarco-passage/trec-dl-2019")
    docstore = ds.docs_store()
    query_map = {str(q.query_id): q.text for q in ds.queries_iter()}

    # Store (rank, docid, text) so we can recover the original BM25 order.
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
    bm25_by_qid = {}   # qid -> [docid, ...] in BM25 rank order (rank 1 = best = index 0)
    for qid, entries in sorted(by_qid.items(), key=lambda x: x[0]):
        entries.sort(key=lambda x: x[0])          # sort by rank ascending
        ranking = [(docid, text) for _, docid, text in entries]
        first_stage.append((qid, query_map[qid], ranking))
        bm25_by_qid[qid] = [docid for _, docid, _ in entries]

    # qrels = defaultdict(dict)
    # for q in ds.qrels_iter():
    #     qrels[str(q.query_id)][str(q.doc_id)] = int(q.relevance)
    # evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})

    # Build qrels dict
    qrels_by_qid = defaultdict(dict)
    for q in ds.qrels_iter():                      # has query_id, doc_id, relevance
        qrels_by_qid[str(q.query_id)][str(q.doc_id)] = int(q.relevance)
    qrels_by_qid = dict(qrels_by_qid)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_by_qid, {'ndcg_cut.10'})

    return first_stage, evaluator, bm25_by_qid


async def _run_dl19_algorithms_once(
    ranking: list[tuple[str, str]],
    query: str,
    client: AsyncOpenAI,
    model: str,
    alg_pbar: tqdm | None = None,
):
    def _set_alg(name: str):
        if alg_pbar is not None:
            alg_pbar.set_description(f"  alg: {name:<35s}")

    assert len(ranking) == 100, print(len(ranking), 'not 100 for DL19')
    outputs = {}
    p_prompt  = _safe_prompt(passage_pointwise_prompt_template,           question=query)
    ep_prompt = _safe_prompt(passage_external_pointwise_prompt_template,  question=query)
    pw_prompt = _safe_prompt(passage_pairwise_comparison_prompt_template, question=query)
    ex_prompt = _safe_prompt(passage_external_comparison_prompt_template, question=query)

    # pointwise — capture scores directly for use in run scoring
    _set_alg("pointwise")
    p_ids, p_scores, _, p_in, p_out = await pointwise_sort(
        ranking[:], client, p_prompt, model, float,
        key_class=PointwiseRelevanceKey, isPassage=True,
    )
    outputs["pointwise"] = (p_ids, p_scores, p_in, p_out)

    # external_pointwise_4 — also has direct scores
    _set_alg("external_pointwise_4")
    ep_ids, ep_scores, _, ep_in, ep_out, _ = await external_pointwise_sort(
        ranking[:], external_values, client, ep_prompt, model, float,
        isPassage=True, memory_size=4,
    )
    outputs["external_pointwise_4"] = (ep_ids, ep_scores, ep_in, ep_out)

    # Comparison-based algorithms return worst-to-best order (no direct scores).
    _set_alg("quick_sort")
    q1_sorted, _, q1_in, q1_out = await quick_sort(
        ranking[:], client, pw_prompt, model, isPassage=True, vote=1,
    )
    outputs["quick_sort"] = (_normalize_docids(q1_sorted), None, q1_in, q1_out)

    # Run quick_sort3, external_bubble_sort_4, and external_merge_sort_4 concurrently
    # to saturate the rate limit and reduce total wall-clock time.
    _set_alg("quick_sort3 | ext_bubble_4 | ext_merge_4  [parallel]")
    (
        (q3_sorted, _, q3_in, q3_out),
        (eb_sorted, _, eb_in, eb_out),
        (em_sorted, _, em_in, em_out),
    ) = await asyncio.gather(
        quick_sort(ranking[:], client, pw_prompt, model, isPassage=True, vote=3),
        external_bubble_sort(ranking[:], external_comparisons, 4, client, ex_prompt, model, isPassage=True),
        external_merge_sort(ranking[:], external_comparisons, 4, client, ex_prompt, model, isPassage=True),
    )
    outputs["quick_sort3"]            = (_normalize_docids(q3_sorted), None, q3_in, q3_out)
    outputs["external_bubble_sort_4"] = (_normalize_docids(eb_sorted), None, eb_in, eb_out)
    outputs["external_merge_sort_4"]  = (_normalize_docids(em_sorted), None, em_in, em_out)

    if alg_pbar is not None:
        alg_pbar.set_description(f"  alg: {'done':<35s}")
    return outputs


async def run_dl19(args, client: AsyncOpenAI, pbar: tqdm | None = None, alg_pbar: tqdm | None = None) -> dict:
    first_stage, evaluator, bm25_by_qid = _build_dl19_data(_resolve(args.dl19_run_file), args.hit_depth)
    acc = _empty_acc(DL19_ALGORITHMS)

    for seed in args.seeds:
        rng = random.Random(seed)
        run_by_alg = {alg: {} for alg in DL19_ALGORITHMS}

        # BM25 is the original run-file order — same for every seed, zero LLM cost.
        for qid, _, _ in first_stage:
            run_by_alg["bm25"][str(qid)] = _to_run_scores(bm25_by_qid[qid])

        if pbar is not None:
            sidx = args.seeds.index(seed) + 1
            pbar.reset(total=len(first_stage))
            pbar.set_description(f"[{args.model}] seed {sidx}/{len(args.seeds)}")
        for qid, query, ranking in first_stage:
            # Shuffle each query's document list individually (matches DL19.ipynb)
            top_ranking = ranking[:]
            rng.shuffle(top_ranking)
            outputs = await _run_dl19_algorithms_once(
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
                if scores is not None:
                    # Pointwise / external_pointwise: use LLM relevance scores directly.
                    run_by_alg[alg][str(qid)] = {str(d): float(s) for d, s in zip(docids, scores)}
                else:
                    # Comparison-based (worst-to-best order): assign rank score i+1.
                    run_by_alg[alg][str(qid)] = {str(d): float(i + 1) for i, d in enumerate(docids)}
        pass  # bars are owned and closed by the caller

        for alg in DL19_ALGORITHMS:
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
                "score_mean": mean_v,
                "score_std": std_v,
                "price": _price(args.model, in_t, out_t),
                "tokens": in_t + out_t,
            }
        )

    return {
        "dataset": "dl19",
        "generated_at": _now_iso(),
        "settings": {
            "run_file": args.dl19_run_file,
            "hit_depth": args.hit_depth,
            "model": args.model,
            "seeds": args.seeds,
            "algorithms": [m["algorithm"] for m in metrics],
        },
        "metrics": metrics,
        "metric_name": "ndcg@10",
    }




_DEFAULT_MODELS = "llama3.1-70b,llama3.1-405b,openai-gpt-4.1"


def _output_path(dataset: str, model: str) -> Path:
    """Auto-derive output path: dev/<dataset>/results_<model>.json"""
    safe_model = model.replace("/", "-")   # guard against any path separators
    return PROJECT_ROOT / "dev" / dataset / f"results_{safe_model}.json"


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

    parser = argparse.ArgumentParser(description="Run script-based experiments.")
    parser.add_argument("--dataset", choices=["dl19", "nba"], required=True)
    parser.add_argument(
        "--models",
        default=_DEFAULT_MODELS,
        help="Comma-separated list of model names to run (default: all three).",
    )
    parser.add_argument("--dl19-run-file", default="data/run.msmarco-v1-passage.bm25-default.dl19.txt")
    parser.add_argument("--hit-depth", type=int, default=100)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--nba-csv", default="data/nba_heights_200.csv")
    parser.add_argument(
        "--nba-limit",
        type=int,
        default=None,
        help="Optional limit for NBA rows (e.g., 10 for quick sanity checks).",
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
        unit = "seed" if model_args.dataset == "nba" else "query"
        # Two tqdm rows per model: progress bar + current-alg label.
        pbar = tqdm(
            total=n_seeds,          # NBA: one tick per seed; DL19: reset per seed to n_queries
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

        if model_args.dataset == "dl19":
            payload = await run_dl19(model_args, client, pbar=pbar, alg_pbar=alg_pbar)
        else:
            payload = await run_nba(model_args, client, pbar=pbar)

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

