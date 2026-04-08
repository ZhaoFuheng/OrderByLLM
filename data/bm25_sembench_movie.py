"""
Generate BM25 sentiment rankings for SembenchMovie reviews.

For each of the top-K most-reviewed movies, rank all reviews by BM25
relevance to a positive-sentiment query.  Output is a TREC-style run file:

    movie_id Q0 reviewId rank score BM25

Usage:
    python data/bm25_sembench_movie.py
    python data/bm25_sembench_movie.py --top-k 5 --output data/run.sembench_movie.bm25-sentiment.txt
"""

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0].parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from order_by.utils import load_movie_reviews


POSITIVE_QUERY = (
    "Very positive. Strong positive sentiment, indicating high satisfaction."
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def bm25_rank(
    docs: list[tuple[str, str]],
    query: str,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[tuple[str, float]]:
    """Return (doc_id, score) pairs sorted by BM25 score descending."""
    query_tokens = _tokenize(query)
    doc_tokens = [(_id, _tokenize(text)) for _id, text in docs]

    N = len(doc_tokens)
    avgdl = sum(len(toks) for _, toks in doc_tokens) / N if N else 1.0

    df: Counter[str] = Counter()
    for _, toks in doc_tokens:
        df.update(set(toks))

    scored: list[tuple[str, float]] = []
    for doc_id, toks in doc_tokens:
        tf = Counter(toks)
        dl = len(toks)
        score = 0.0
        for qt in query_tokens:
            n_q = df.get(qt, 0)
            if n_q == 0:
                continue
            idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
            f_qt = tf.get(qt, 0)
            numerator = f_qt * (k1 + 1.0)
            denominator = f_qt + k1 * (1.0 - b + b * dl / avgdl)
            score += idf * numerator / denominator
        scored.append((doc_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BM25 sentiment rankings for SembenchMovie."
    )
    parser.add_argument(
        "--csv",
        default="data/movie/rotten_tomatoes_movie_reviews.csv",
        help="Path to the Rotten Tomatoes reviews CSV.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top-reviewed movies to include (default: 5).",
    )
    parser.add_argument(
        "--review-limit",
        type=int,
        default=None,
        help="Max reviews per movie (default: no limit).",
    )
    parser.add_argument(
        "--query",
        default=POSITIVE_QUERY,
        help="Positive-sentiment query for BM25 ranking.",
    )
    parser.add_argument(
        "--output",
        default="data/run.sembench_movie.bm25-sentiment.txt",
        help="Output TREC-format run file.",
    )
    args = parser.parse_args()

    first_stage, _, _ = load_movie_reviews(
        args.csv, top_k_reviewed_movies=args.top_k, review_limit=args.review_limit,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for movie_id, ranking in first_stage:
            ranked = bm25_rank(ranking, args.query)

            for rank, (review_id, bm25_score) in enumerate(ranked, start=1):
                out.write(f"{movie_id} Q0 {review_id} {rank} {bm25_score:.6f} BM25\n")

            print(f"  {movie_id}: {len(ranked)} reviews ranked")

    print(f"\nWrote BM25 run file ({len(first_stage)} movies) to: {output_path}")


if __name__ == "__main__":
    main()
