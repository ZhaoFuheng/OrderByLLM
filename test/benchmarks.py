"""Loaders for the passage-ranking test benchmarks (DL20, HellaSwag, NFCorpus).

Shared by run_experiment.py and run_optimizer.py so both rank exactly the same
candidates. Shuffling candidates the same way in both is what lets the
optimizer's final ranking reuse the LLM responses cached by the standalone runs.
"""
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import ir_datasets
import pytrec_eval


@dataclass
class PassageBenchmark:
    """A query -> candidate-passages ranking task scored with nDCG@10.

    first_stage : [(query_id, query_text, [(doc_id, doc_text), ...]), ...]
    qrels       : {query_id: {doc_id: relevance}}
    bm25        : {query_id: [doc_id, ...]} best-first, when the candidates come
                  from a BM25 first stage; None otherwise.
    shared_pool : True when every query ranks the same candidate pool.
    """
    first_stage: list[tuple[str, str, list[tuple[str, str]]]]
    qrels: dict[str, dict[str, int]]
    bm25: dict[str, list[str]] | None = None
    shared_pool: bool = False

    def __post_init__(self):
        self.evaluator = pytrec_eval.RelevanceEvaluator(self.qrels, {"ndcg_cut.10"})
        # The full query list: `shuffled` always draws its shuffles over this, in
        # order, so a query's candidate order does not depend on which other
        # queries `limit` / `select` kept (and its cached responses stay valid).
        self._all_queries = self.first_stage

    def limit(self, num_queries: int | None) -> None:
        """Keep only the first `num_queries` queries (candidates are unchanged)."""
        if num_queries is not None:
            self.first_stage = self.first_stage[:num_queries]

    def select(self, query_ids: list[str]) -> None:
        """Keep only these queries, in benchmark order."""
        missing = set(query_ids) - {qid for qid, _, _ in self.first_stage}
        if missing:
            raise ValueError(f"unknown query ids: {sorted(missing)}")
        keep = set(query_ids)
        self.first_stage = [q for q in self.first_stage if q[0] in keep]

    def shuffled(self, seed: int) -> list[tuple[str, str, list[tuple[str, str]]]]:
        """Each kept query with its candidates in the seeded random order the
        algorithms are given. A shared pool is shuffled once, so every query
        ranks the same input order; otherwise one rng shuffles each query's
        candidates in turn (over the full benchmark, see __post_init__)."""
        rng = random.Random(seed)
        kept = {qid for qid, _, _ in self.first_stage}
        if self.shared_pool:
            pool = self._all_queries[0][2][:] if self._all_queries else []
            rng.shuffle(pool)
            return [(qid, query, pool[:]) for qid, query, _ in self.first_stage]
        prepared = []
        for qid, query, ranking in self._all_queries:
            candidates = ranking[:]
            rng.shuffle(candidates)
            if qid in kept:
                prepared.append((qid, query, candidates))
        return prepared


def _read_beir(data_dir: Path, query_field_fallback: str | None = None):
    """Read a BEIR-format directory: queries.jsonl, corpus.jsonl, test.tsv.
    Returns (queries, corpus, qrels) with corpus as [(doc_id, "title text")]."""
    queries: dict[str, str] = {}
    with (data_dir / "queries.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if query_field_fallback is None:
                queries[str(obj["_id"])] = obj["text"]
            else:
                queries[str(obj["_id"])] = obj.get("text", obj.get(query_field_fallback, ""))

    corpus: list[tuple[str, str]] = []
    with (data_dir / "corpus.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            title = obj.get("title", "")
            text = obj.get("text", "")
            corpus.append((str(obj["_id"]), (title + " " + text).strip() if title else text))

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with (data_dir / "test.tsv").open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("query-id"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, docid, score = parts[0], parts[1], int(parts[2])
            qrels[str(qid)][str(docid)] = score
    return queries, corpus, dict(qrels)


def load_dl20(run_path: Path, hit_depth: int) -> PassageBenchmark:
    """TREC DL 2020: rerank the top `hit_depth` BM25 passages of each query."""
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
            if int(rank) > hit_depth or qid not in query_map:
                continue
            doc = docstore.get(docid)
            if doc is None:
                continue
            text = (doc.title + " " if getattr(doc, "title", None) else "") + doc.text
            by_qid[qid].append((int(rank), docid, text))

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for q in ds.qrels_iter():
        qrels[str(q.query_id)][str(q.doc_id)] = int(q.relevance)
    qrels = dict(qrels)

    first_stage, bm25 = [], {}
    for qid, entries in sorted(by_qid.items(), key=lambda x: x[0]):
        entries.sort(key=lambda x: x[0])
        bm25[qid] = [docid for _, docid, _ in entries]
        # DL20 only has qrels for 54 of the 200 queries — keep those only.
        if qid in qrels:
            first_stage.append((qid, query_map[qid], [(docid, text) for _, docid, text in entries]))

    return PassageBenchmark(first_stage, qrels, bm25=bm25)


def load_hellaswag(data_dir: Path, num_queries: int | None = 100) -> PassageBenchmark:
    """HellaSwag as a POOLED retrieval task over the first `num_queries` questions.

    The candidate pool is the set of endings belonging to those questions (4
    each, so N questions -> 4N docs; the default 100 -> 400 docs). Every query
    ranks that shared pool and must surface its own correct ending — no
    first-stage filtering. `num_queries=None` uses all questions (200 -> 800 docs).
    """
    queries, corpus, qrels_all = _read_beir(data_dir)

    docs_by_query: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for doc_id, text in corpus:
        docs_by_query[doc_id.rsplit("-", 1)[0]].append((doc_id, text))

    # The first `num_queries` questions that have a query, qrels, and endings.
    selected = [
        qid for qid in sorted(queries.keys())
        if qid in qrels_all and docs_by_query.get(qid)
    ]
    if num_queries is not None:
        selected = selected[:num_queries]

    pool = [doc for qid in selected for doc in docs_by_query[qid]]
    first_stage = [(qid, queries[qid], pool[:]) for qid in selected]
    return PassageBenchmark(first_stage, {qid: qrels_all[qid] for qid in selected}, shared_pool=True)


def load_nfcorpus(data_dir: Path) -> PassageBenchmark:
    """NFCorpus (nutrition / medical IR): every test query ranks the entire
    corpus, with no first-stage retrieval."""
    queries, corpus, qrels = _read_beir(data_dir, query_field_fallback="title")
    corpus = list(dict(corpus).items())  # one entry per doc id
    first_stage = [(qid, queries[qid], corpus[:]) for qid in sorted(qrels.keys()) if qid in queries]
    return PassageBenchmark(first_stage, qrels)
