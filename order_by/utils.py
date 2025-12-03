import tiktoken
import hashlib
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


def is_async_client(client) -> bool:
    return isinstance(client, AsyncOpenAI)


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

def borda(rankings, k):
    # print(f'rankings: {rankings}')
    scores = defaultdict(int)
    for ranking in rankings:
        n = len(ranking)
        for position, item in enumerate(ranking):
            # worst item first → lowest score
            # if k:
            #     assert n >= k, print(f'ranking length {n} is less than k {k}')
            #     cutoff = n - k
            #     score = (position - cutoff + 1) if position >= cutoff else 0
            # else:
            score = position
            scores[item] += score
    ranked_items = sorted(scores.items(), key=lambda x: (x[1], x[0]))
    if k:
        ranked_items = [item for item, score in ranked_items[-k:]]
    else:
        ranked_items = [item for item, score in ranked_items]
    return ranked_items

def bradley_terry(rankings, k, max_iter=200, tol=1e-6):

    # Collect all unique items
    items = list({item for rank in rankings for item in rank})

    # Initialize strengths for each item
    strength = {item: 1.0 for item in items}

    # Pairwise win counts: wins[a][b] = number of times a beats b
    wins = {a: defaultdict(int) for a in items}

    # Rankings are ascending order (worst → best)
    # So later items beat earlier items
    for rank in rankings:
        for i in range(len(rank)):
            for j in range(i + 1, len(rank)):
                winner = rank[j]
                loser = rank[i]
                wins[winner][loser] += 1

    # MM iterations for Bradley–Terry
    for _ in range(max_iter):
        new_strength = {}
        max_change = 0

        for a in items:
            numerator = 0.0
            denom = 0.0

            for b in items:
                if a == b:
                    continue

                w_ab = wins[a][b]          # a beats b
                w_ba = wins[b][a]          # b beats a

                numerator += w_ab

                # Avoid division by zero
                denom += (w_ab + w_ba) / (strength[a] + strength[b])

            new_val = numerator / denom if denom > 0 else strength[a]
            new_strength[a] = new_val
            max_change = max(max_change, abs(new_val - strength[a]))

        strength = new_strength

        if max_change < tol:
            break

    # Sort items by descending strength (best first)
    ranked_items = sorted(items, key=lambda x: (-strength[x], x))

    # Apply top-k cutoff if provided
    if k:
        ranked_items = ranked_items[:k]

    return ranked_items[::-1]


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
