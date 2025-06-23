from .pointwise import Pointwise_Key, external_values
from .pair_comparison import Pair_Comparison_Key, external_comparisons
import collections
import asyncio

async def pointwise_sort(data, client, prompt_template, modelname, output_type):
    total_api_calls = 0
    total_tokens = 0
    data_confidence = []

    async def compute_sort_key(item):
        key_obj = Pointwise_Key(item)
        prompt = prompt_template.format(key=str(key_obj.key))
        value, api_calls, tokens, confidence = await key_obj.value(client, prompt, modelname, output_type)
        return (value, item, api_calls, tokens, confidence)

    tasks = [compute_sort_key(item) for item in data]
    results = await asyncio.gather(*tasks)

    sorted_results = sorted(results, key=lambda x: (x[0], x[1]))
    sorted_data = [item for _, item, _, _, _ in sorted_results]
    data_confidence = [confidence for _, _, _, _, confidence in sorted_results]

    total_api_calls = sum(api_calls for _, _, api_calls, _, _ in results)
    total_tokens = sum(tokens for _, _, _, tokens, _ in results)

    return sorted_data, total_api_calls, total_tokens, data_confidence
    

async def bubble_sort(data, client, prompt_template, modelname):
    n = len(data)
    total_api_calls = 0
    total_tokens = 0
    wrapped_data = [Pair_Comparison_Key(item) for item in data]

    for i in range(n):
        for j in range(0, n - i - 1):
            comparison_result, api_calls, tokens = await wrapped_data[j].compare(
                wrapped_data[j + 1], client, prompt_template, modelname
            )
            total_api_calls += api_calls
            total_tokens += tokens
            if comparison_result == 1:
                wrapped_data[j], wrapped_data[j + 1] = wrapped_data[j + 1], wrapped_data[j]

    sorted_data = [item.key for item in wrapped_data]
    return sorted_data, total_api_calls, total_tokens


async def quick_sort(data, client, prompt_template, modelname):
    if len(data) <= 1:
        return data, 0, 0

    total_api_calls = 0
    total_tokens = 0

    pivot = Pair_Comparison_Key(data[0])
    less = []
    greater = []

    for item in data[1:]:
        wrapped_item = Pair_Comparison_Key(item)
        comparison_result, api_calls, tokens = await wrapped_item.compare(
            pivot, client, prompt_template, modelname
        )
        total_api_calls += api_calls
        total_tokens += tokens

        if comparison_result == -1:
            less.append(item)
        else:
            greater.append(item)

    # Parallel recursive sort
    left_task = asyncio.create_task(quick_sort(less, client, prompt_template, modelname))
    right_task = asyncio.create_task(quick_sort(greater, client, prompt_template, modelname))

    sorted_less, left_api_calls, left_tokens = await left_task
    sorted_greater, right_api_calls, right_tokens = await right_task

    total_api_calls += left_api_calls + right_api_calls
    total_tokens += left_tokens + right_tokens

    return sorted_less + [pivot.key] + sorted_greater, total_api_calls, total_tokens

async def heap_sort(data, client, prompt_template, modelname):
    async def heapify(data, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        api_calls = 0
        tokens = 0

        if left < n:
            comparison_result, calls, toks = await data[left].compare(data[largest], client, prompt_template, modelname)
            api_calls += calls
            tokens += toks
            if comparison_result == 1:
                largest = left

        if right < n:
            comparison_result, calls, toks = await data[right].compare(data[largest], client, prompt_template, modelname)
            api_calls += calls
            tokens += toks
            if comparison_result == 1:
                largest = right

        if largest != i:
            data[i], data[largest] = data[largest], data[i]
            recursive_calls, recursive_toks = await heapify(data, n, largest)
            api_calls += recursive_calls
            tokens += recursive_toks

        return api_calls, tokens

    n = len(data)
    total_api_calls = 0
    total_tokens = 0

    wrapped_data = [Pair_Comparison_Key(item) for item in data]

    # Build heap (heapify each node from bottom up)
    for i in range(n // 2 - 1, -1, -1):
        api_calls, tokens = await heapify(wrapped_data, n, i)
        total_api_calls += api_calls
        total_tokens += tokens

    # Extract elements from heap
    for i in range(n - 1, 0, -1):
        wrapped_data[0], wrapped_data[i] = wrapped_data[i], wrapped_data[0]
        api_calls, tokens = await heapify(wrapped_data, i, 0)
        total_api_calls += api_calls
        total_tokens += tokens

    sorted_data = [item.key for item in wrapped_data]
    return sorted_data, total_api_calls, total_tokens


async def external_bubble_sort(data, sortfunc, k, client, prompt_template, modelname):
    total_api_calls = 0
    total_tokens = 0
    n = len(data)

    for pass_end in range(n, 0, -k // 2):  # Shrink the range in each pass
        if pass_end < k // 2:
            sorted_chunk, num, tokens = await sortfunc(data[:pass_end], client, prompt_template, modelname)
            total_api_calls += num
            total_tokens += tokens
            data[:pass_end] = sorted_chunk
            break

        start = 0
        while start <= pass_end - k // 2:
            end = min(start + k, n)
            chunk = data[start:end]
            sorted_chunk, num, tokens = await sortfunc(chunk, client, prompt_template, modelname)
            total_api_calls += num
            total_tokens += tokens
            data[start:end] = sorted_chunk
            start += k // 2

    return data, total_api_calls, total_tokens



async def external_merge_sort(data, sortfunc, k, client, prompt_template, modelname):
    async def merge_sorted_chunks(l1, l2):
        def fix_duplicates(buffer, original):
            """Ensure buffer has same multiset as original by correcting LLM errors."""
            if set(buffer) == set(original) and all(buffer.count(x) == original.count(x) for x in set(buffer)):
                return buffer

            target_counts = collections.Counter(original)
            result = []
            used = collections.Counter()

            for item in buffer:
                if used[item] < target_counts[item]:
                    result.append(item)
                    used[item] += 1

            # Fill in missing items
            for item, count in target_counts.items():
                while used[item] < count:
                    result.append(item)
                    used[item] += 1

            return result

        def fill_buffer(i, j, buffer, membership, buffer_metadata, half):
            """Fill buffer with up to `half` items from each list."""
            while buffer_metadata['l1'] < half and i[0] < len(l1):
                item = l1[i[0]]
                buffer.append(item)
                membership[item].append('l1')
                buffer_metadata['l1'] += 1
                i[0] += 1
            while buffer_metadata['l2'] < half and j[0] < len(l2):
                item = l2[j[0]]
                buffer.append(item)
                membership[item].append('l2')
                buffer_metadata['l2'] += 1
                j[0] += 1

        i, j = [0], [0]  # Use lists for mutable indices in helpers
        merged = []
        buffer = []
        membership = collections.defaultdict(list)
        buffer_metadata = {'l1': 0, 'l2': 0}
        total_api_calls = 0
        total_tokens = 0
        half = max(k // 2, 1)

        while i[0] < len(l1) or j[0] < len(l2):
            if i[0] >= len(l1) and buffer_metadata['l1'] == 0:
                merged.extend(buffer + l2[j[0]:])
                break
            if j[0] >= len(l2) and buffer_metadata['l2'] == 0:
                merged.extend(buffer + l1[i[0]:])
                break

            fill_buffer(i, j, buffer, membership, buffer_metadata, half)

            original_buffer = buffer[:]
            buffer, num, tokens = await sortfunc(buffer, client, prompt_template, modelname)
            total_api_calls += num
            total_tokens += tokens

            buffer = fix_duplicates(buffer, original_buffer)

            items_taken = 0
            for key in buffer:
                if not membership[key]:
                    print("⚠️ LLM returned extra or duplicate key:", key)
                    continue
                source = membership[key].pop()
                buffer_metadata[source] -= 1
                merged.append(key)
                items_taken += 1
                if buffer_metadata[source] == 0:
                    break

            buffer = buffer[items_taken:]

        if buffer:
            merged.extend(buffer)

        return merged, total_api_calls, total_tokens

    # Step 1: Chunk sorting in parallel
    chunks = [data[i:i + k] for i in range(0, len(data), k)]
    sort_tasks = [sortfunc(chunk, client, prompt_template, modelname) for chunk in chunks]
    sort_results = await asyncio.gather(*sort_tasks)

    total_api_calls = sum(r[1] for r in sort_results)
    total_tokens = sum(r[2] for r in sort_results)
    sorted_chunks = [r[0] for r in sort_results]

    # Step 2: Merging sorted chunks in parallel (repeated passes)
    while len(sorted_chunks) > 1:
        merge_tasks = []
        for i in range(0, len(sorted_chunks), 2):
            if i + 1 < len(sorted_chunks):
                merge_tasks.append(merge_sorted_chunks(sorted_chunks[i], sorted_chunks[i + 1]))
            else:
                merge_tasks.append(asyncio.sleep(0, result=(sorted_chunks[i], 0, 0)))  # no-op async return

        merge_results = await asyncio.gather(*merge_tasks)
        total_api_calls += sum(r[1] for r in merge_results)
        total_tokens += sum(r[2] for r in merge_results)
        sorted_chunks = [r[0] for r in merge_results]

    return sorted_chunks[0], total_api_calls, total_tokens

async def external_pointwise_sort(data, sortfunc, m, client, prompt_template, modelname, output_type):
    total_api_calls = 0
    total_tokens = 0
    key_and_value = {}
    data_confidence = {}

    # Split data into chunks of size m
    chunks = [data[i:i + m] for i in range(0, len(data), m)]

    # Launch all chunk sorting tasks in parallel
    tasks = [sortfunc(chunk, client, prompt_template, modelname, output_type) for chunk in chunks]
    results = await asyncio.gather(*tasks)

    # Aggregate results
    for chunk, (chunk_vals, api_calls, tokens, confidences) in zip(chunks, results):
        total_api_calls += api_calls
        total_tokens += tokens
        for k, v, conf in zip(chunk, chunk_vals, confidences):
            try:
                key_and_value[k] = output_type(v)
                data_confidence[k] = conf
            except Exception as e:
                print(e)
                key_and_value[k] = output_type(0.0)
                data_confidence[k] = 0

    # Sort by computed value then original key
    sorted_data = sorted(key_and_value, key=lambda k: (key_and_value[k], k))
    sorted_confidence = [data_confidence[k] for k in sorted_data]

    return sorted_data, total_api_calls, total_tokens, sorted_confidence

async def insertion_sort(sorted_list, data, client, prompt_template, modelname):
    """Inserts each item from `data` into the already sorted `sorted_list` using LLM-based comparison."""
    total_api_calls = 0
    total_tokens = 0

    for item in data:
        key_item = Pair_Comparison_Key(item)
        left, right = 0, len(sorted_list)

        # Binary search to find the correct insert position
        while left < right:
            mid = (left + right) // 2
            key_mid = Pair_Comparison_Key(sorted_list[mid])
            comparison_result, api_calls, tokens = await key_item.compare(key_mid, client, prompt_template, modelname)

            total_api_calls += api_calls
            total_tokens += tokens

            if comparison_result == -1:  # item < mid
                right = mid
            else:  # item >= mid
                left = mid + 1

        sorted_list.insert(left, item)

    return sorted_list, total_api_calls, total_tokens

from openai import OpenAI
import os

async def main():
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    modelname = 'gpt-4o'

    prompt_template = "In scale 1-100, how friendly is key: {key}? Rate the confidence in the proposed answer on a scale of 0-10.\n Output an int.\n"
    sorted_data, num, tokens, conf = await pointwise_sort(['cat', 'tiger', 'dolphin'], client, prompt_template, modelname, int)
    print(sorted_data, num, tokens, conf)

    prompt_template = "In scale 1-100, how friendly are keys: {keys}? For each answer, also provide a confidence rating on a scale of 0-10.\n"
    sorted_data, num, tokens, conf = await external_pointwise_sort(['cat', 'tiger', 'dolphin'], external_values, 4, client, prompt_template, modelname, int)
    print(sorted_data, num, tokens, conf)

    prompt_template = "Which is greater {key1} or {key2}? Output the greater key.\n"
    # ignore bubble sort as it requries too many api calls when list is long.
    sorted_data, num, tokens = await bubble_sort([34, 87, 12, 59, 3, 71, 45, 90], client, prompt_template, modelname)
    print("bubble sort: ", sorted_data, num, tokens)
    sorted_data, num, tokens = await quick_sort([34, 87, 12, 59, 3, 71, 45, 90], client, prompt_template, modelname)
    print("quick sort: ", sorted_data, num, tokens)
    sorted_data, num, tokens = await heap_sort([34, 87, 12, 59, 3, 71, 45, 90], client, prompt_template, modelname)
    print("heap sort: ", sorted_data, num, tokens)
    sorted_data, num, tokens = await insertion_sort([1, 3, 5, 7, 9], [2, 15], client, prompt_template, modelname)
    print("heap sort: ", sorted_data, num, tokens)


    prompt_template = "Given a list of keys: {keys}\nSort the keys in ascending order.\n"
    sorted_data, num, tokens = await external_bubble_sort([34, 87, 12, 59, 3, 71, 45, 90], external_comparisons, 4, client, prompt_template, modelname)
    print("external bubble sort: ", sorted_data, num, tokens)
    sorted_data, num, tokens = await external_merge_sort([34, 87, 12, 59, 3, 71, 45, 90], external_comparisons, 4, client, prompt_template, modelname)
    print("external merge sort ", sorted_data, num, tokens)

if __name__ == "__main__":
    asyncio.run(main())