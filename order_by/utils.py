import tiktoken
import hashlib
from scipy.stats import kendalltau
import random
from openai import OpenAI, AsyncOpenAI
import inspect
from typing import Awaitable, TypeVar, Union

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
    
def create_numbered_passages(passages):
    return "\n".join([f"passage_id:{i+1}\n{p}\n\n" for i, p in enumerate(passages)])

if __name__ == "__main__":
    sample_text = "This is a sample text to calculate token count."
    token_count = count_tokens(sample_text)
    print(f"Number of tokens: {token_count}")