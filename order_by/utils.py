import tiktoken
import hashlib
from pathlib import Path
from scipy.stats import kendalltau
import random
from openai import OpenAI, AsyncOpenAI
import inspect
from typing import Awaitable, TypeVar, Union, Any, Dict
import json
import httpx
import asyncio
from collections import defaultdict
import math, statistics

import pandas as pd


def load_movie_reviews(
    csv_path: str | Path,
    top_k_reviewed_movies: int = 5,
    review_limit: int | None = None,
) -> tuple[list[tuple[str, list]], dict[str, list[str]], dict[str, dict[str, int]]]:
    """Load Rotten Tomatoes reviews CSV, discard rows with no valid x/y score
    or missing text, and return data for the top-K most-reviewed movies.

    Returns (first_stage, gold_by_movie, qrels_by_movie):
        first_stage   : [(movie_id, [(reviewId, reviewText), ...])]
        gold_by_movie : {movie_id: [reviewId, ...]} in gold order (score desc)
        qrels_by_movie: {movie_id: {reviewId: int_grade}}
    """
    reviews = pd.read_csv(
        csv_path,
        usecols=["id", "reviewId", "reviewText", "originalScore"],
    )
    split = reviews["originalScore"].str.split("/", expand=True)
    reviews["score"] = (
        pd.to_numeric(split[0], errors="coerce")
        / pd.to_numeric(split[1], errors="coerce")
    )
    reviews = reviews.dropna(subset=["score", "reviewText"]).copy()

    top_movies = (
        reviews.groupby("id")
        .size()
        .sort_values(ascending=False)
        .head(top_k_reviewed_movies)
        .index.tolist()
    )

    first_stage: list[tuple[str, list]] = []
    gold_by_movie: dict[str, list[str]] = {}
    qrels_by_movie: dict[str, dict[str, int]] = {}

    for movie_id in top_movies:
        sub = reviews[reviews["id"] == movie_id].drop_duplicates("reviewId")
        if review_limit is not None:
            sub = sub.head(review_limit)
        gold_sub = sub.sort_values("score", ascending=False)
        gold_by_movie[movie_id] = [str(r["reviewId"]) for _, r in gold_sub.iterrows()]
        qrels_by_movie[movie_id] = {
            str(r["reviewId"]): int(float(r["score"]) * 10)
            for _, r in sub.iterrows()
        }
        ranking = [(str(r["reviewId"]), str(r["reviewText"])) for _, r in sub.iterrows()]
        first_stage.append((movie_id, ranking))

    return first_stage, gold_by_movie, qrels_by_movie


# ── Sorting call-count estimators ─────────────────────────────────────────

def bubble_sort_calls_sim(n: int, k: int, limit_k: int) -> int:
    """Count the exact number of sortfunc calls in external_bubble_sort."""
    stride = max(1, k // 2)
    limit_k = min(limit_k, n)
    calls = 0
    for pass_end in range(n, n - limit_k, -stride):
        if pass_end < stride:
            calls += 1
            break
        start = 0
        while start <= pass_end - stride:
            calls += 1
            start += stride
    return max(calls, 1)


def merge_sort_calls_sim(n: int, k: int, limit_k: int) -> int:
    """Approximate the number of sortfunc calls in external_merge_sort."""
    limit_k = min(limit_k, n)
    n_chunks = math.ceil(n / k)
    calls = n_chunks
    items_per_call = max(1, k / 2)
    chunk_sizes = [min(k, n - i * k) for i in range(n_chunks)]
    while len(chunk_sizes) > 1:
        new_sizes = []
        for i in range(0, len(chunk_sizes), 2):
            if i + 1 < len(chunk_sizes):
                combined = chunk_sizes[i] + chunk_sizes[i + 1]
                produced = min(combined, limit_k)
                calls += math.ceil(produced / items_per_call)
                new_sizes.append(produced)
            else:
                new_sizes.append(chunk_sizes[i])
        chunk_sizes = new_sizes
    return max(calls, 1)


def quick_sort_calls_balanced(n: int, vote: int, limit_k: int) -> int:
    """Quicksort API calls with ideal balanced partitions (lower bound)."""
    if n <= 1:
        return 0
    limit_k = min(limit_k, n)
    n_greater = (n - 1) // 2
    n_less = n - 1 - n_greater

    calls = n - 1
    if vote > 1:
        if n_greater > 1:
            calls += n_less * min(vote - 1, n_greater)
        if n_less > 1:
            calls += n_greater * min(vote - 1, n_less)

    if limit_k >= n:
        calls += quick_sort_calls_balanced(n_less, vote, n_less)
        calls += quick_sort_calls_balanced(n_greater, vote, n_greater)
    elif limit_k <= n_greater:
        calls += quick_sort_calls_balanced(n_greater, vote, limit_k)
    elif limit_k == n_greater + 1:
        calls += quick_sort_calls_balanced(n_greater, vote, n_greater)
    else:
        left_k = limit_k - n_greater - 1
        calls += quick_sort_calls_balanced(n_less, vote, left_k)
        calls += quick_sort_calls_balanced(n_greater, vote, n_greater)
    return calls


def quick_sort_calls_sim(n: int, vote: int, limit_k: int) -> int:
    """Quicksort API calls with random-pivot correction (x1.5)."""
    RANDOM_PIVOT_FACTOR = 1.5
    return int(quick_sort_calls_balanced(n, vote, limit_k) * RANDOM_PIVOT_FACTOR)


def quick_calls_formula(n: int, vote: int, limit_k: int) -> float:
    """O(v*(2n + K*log2(K))) for partial sort, O(v*n*log2(n)) for full sort."""
    if n <= 1:
        return 0.0
    K = min(limit_k, n)
    if K >= n:
        return vote * n * math.log2(max(n, 2))
    return vote * 2 * (n + K * math.log2(max(K, 2)))


def bubble_calls_formula(n: int, m: int, limit_k: int) -> float:
    """O(4*K*N/m^2): passes ~ 2K/m, windows/pass ~ 2N/m."""
    K = min(limit_k, n)
    stride = max(1, m // 2)
    n_passes = math.ceil(K / stride)
    windows_per_pass = math.ceil(n / stride)
    return max(n_passes * windows_per_pass, 1.0)


def merge_calls_formula(n: int, m: int, limit_k: int) -> float:
    """O(N/m * (2 + 2*log2(K/m))): initial sorts + merge tree with limit_k early exit."""
    K = min(limit_k, n)
    if m <= 0:
        return 1.0
    n_chunks = int(n / m)
    if K <= m:
        return float(n_chunks)
    log_term = math.log2(max(K / m, 1))
    return max(n_chunks * (2 + 2 * log_term), 1.0)


def is_async_client(client) -> bool:
    return isinstance(client, AsyncOpenAI)


def build_client() -> AsyncOpenAI:
    """Build an AsyncOpenAI client from environment variables.

    Required env var : OPENAI_API_KEY
    Optional env var : OPENAI_BASE_URL  (set to point at Snowflake Cortex, etc.)
    """
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required in environment or .env")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


T = TypeVar("T")
async def resolve(v: Union[T, Awaitable[T]]) -> T:
    return await v if inspect.isawaitable(v) else v

def hash_prompt(prompt: str, modelname: str) -> str:
    return hashlib.sha256(f"{modelname}:{prompt}".encode()).hexdigest()

def count_tokens(text, model="gpt-3.5-turbo"):
    try:
        # Load the appropriate tokenizer for the model
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error: {e}")
        return None
def tokens2price(model, in_tokens, out_tokens):
    """
    Calculate the API cost given the model and number of input/output tokens.

    Pricing (as of Sep 2025):
    - gpt-4o:          $2.50 per 1M input tokens, $10.00 per 1M output tokens
    - gpt-4o-mini:     $0.150 per 1M input tokens, $0.600 per 1M output tokens
    - llama3.1-70b:    $0.40 per 1M input tokens, $0.40 per 1M output tokens
    - llama3.1-405b:   $4.00 per 1M input tokens, $4.00 per 1M output tokens
    - openai-gpt-5:    $1.25 per 1M input tokens, $10.00 per 1M output tokens
    - openai-gpt-5-mini:$0.25 per 1M input tokens, $2.00 per 1M output tokens
    """
    # Define a dictionary for easy rate lookup
    pricing = {
        'gpt-4o': (2.50, 10.00),
        'gpt-4o-mini': (0.150, 0.600),
        'llama3.1-70b': (0.40, 0.40),
        'llama3.1-405b': (4.00, 4.00),
        'openai-gpt-5': (1.25, 10.00),
        'openai-gpt-5-mini': (0.25, 2.00),
        'openai-gpt-4.1': (2.00, 8.00),
        'mistral-7b': (0.25, 0.25)
    }

    if model in pricing:
        input_price_per_1M, output_price_per_1M = pricing[model]
        input_rate = input_price_per_1M / 1_000_000
        output_rate = output_price_per_1M / 1_000_000
    else:
        raise ValueError(f"Unknown model: {model}")

    total_cost = in_tokens * input_rate + out_tokens * output_rate
    return round(total_cost, 6)
        


def num_inversions(gold, predict):
    gold_positions = {value: idx for idx, value in enumerate(gold)}
    mapped_predict = [gold_positions[item] for item in predict if item in gold_positions]

    def merge_and_count(arr, temp_arr, left, mid, right):
        i, j, k = left, mid + 1, left
        inv_count = 0
        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp_arr[k] = arr[i]
                i += 1
            else:
                temp_arr[k] = arr[j]
                inv_count += (mid - i + 1)
                j += 1
            k += 1
        while i <= mid:
            temp_arr[k] = arr[i]
            i += 1
            k += 1
        while j <= right:
            temp_arr[k] = arr[j]
            j += 1
            k += 1
        for i in range(left, right + 1):
            arr[i] = temp_arr[i]
        return inv_count

    def merge_sort_and_count(arr, temp_arr, left, right):
        inv_count = 0
        if left < right:
            mid = (left + right) // 2
            inv_count += merge_sort_and_count(arr, temp_arr, left, mid)
            inv_count += merge_sort_and_count(arr, temp_arr, mid + 1, right)
            inv_count += merge_and_count(arr, temp_arr, left, mid, right)
        return inv_count
    temp_arr = mapped_predict[:]
    return merge_sort_and_count(mapped_predict, temp_arr, 0, len(mapped_predict) - 1)

def num_out_of_place(gold: list, predict: list) -> int:
    if len(gold) != len(predict):
        print('gold and predict list should be the same length')
    out_of_place_count = 0
    for k1, k2 in zip(gold, predict):
        if k1 != k2:
            out_of_place_count += 1
    return out_of_place_count

def kendalltau_distance(gold: list, predict: list) -> float:
    """Return Kendall tau distance (# of discordant pairs) between two rankings."""
    gold_pos = {v: i for i, v in enumerate(gold)}
    # Map predict into the rank positions of gold
    gold_ranks = [gold_pos[v] for v in gold if v in gold_pos]
    gold_set = set(gold)

    seen = set()
    need_to_fix_idxs = []
    for i, v in enumerate(predict):
        if v in seen or v not in gold_set:
            need_to_fix_idxs.append(i)
        else:
            seen.add(v)
    missing = [v for v in gold if v not in set(predict)]

    for i, idx in enumerate(need_to_fix_idxs):
        predict[idx] = missing[i]

    pred_ranks = [gold_pos[v] for v in predict if v in gold_pos]


    if len(gold_ranks) != len(pred_ranks):
        print("length of gold: ", len(gold_ranks))
        print("length of prediction: ", len(pred_ranks))
        raise ValueError("gold and predict must have the same items for Kendall tau distance")

    tau, p_value = kendalltau(gold_ranks, pred_ranks)
    return tau

def borda(rankings):
    scores = defaultdict(int)
    for ranking in rankings:
        n = len(ranking)
        for position, item in enumerate(ranking):
            # worst item first → lowest score
            score = position
            scores[item] += score
    ranked_items = sorted(scores.items(), key=lambda x: (x[1], x[0]))
    # print('ranked_items', ranked_items)
    # ranked_items = [item for item, score in ranked_items]
    return ranked_items

def create_numbered_passages(passages, usePID = False):
    if usePID:
        return "\n".join([f"passage_id:{pid}\n{p}\n\n" for i, (pid, p) in enumerate(passages)]) 
    return "\n".join([f"passage_id:{i+1}\n{p}\n\n" for i, p in enumerate(passages)])

def create_numbered_SQLs(sqls, usePID = False):
    if usePID:
        return "\n".join([f"sql_id:{pid}\n{p}\n\n" for i, (pid, p) in enumerate(sqls)]) 
    return "\n".join([f"sql_id:{i+1}\n{p}\n\n" for i, p in enumerate(sqls)])

def create_numbered_reviews(reviews, usePID = False):
    if usePID:
        return "\n".join([f"review_id:{pid}\n{r}\n\n" for i, (pid, r) in enumerate(reviews)]) 
    return "\n".join([f"review_id:{i+1}\n{r}\n\n" for i, r in enumerate(reviews)])

def create_numbered_rankings(ranks):
    return "\n".join([f"id:{i+1}\n{p}\n\n" for i, p in enumerate(ranks)])


# not used in project; use OpenAI SDK instead
# https://docs.snowflake.com/en/user-guide/snowflake-cortex/open_ai_sdk
class SnowflakeClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def responses(self, model, prompt, schema):
        return self.client.responses(model, prompt, schema)