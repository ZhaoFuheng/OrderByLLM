from .pointwise import Pointwise_Key, external_values, PassageExternalPointwiseReasoning, ExternalPointwiseReasoning
from .pair_comparison import *
import collections
import asyncio
import statistics
from typing import List, Callable, Tuple
import hashlib
import random
import heapq

random.seed(0)

async def pointwise_sort(data, client, prompt_template, modelname, output_type, key_class = Pointwise_Key):
    total_api_calls = 0
    total_tokens = 0
    data_confidence = []

    async def compute_sort_key(item, key_class, prompt_template):
        key_obj = key_class(item, prompt_template)
        value, api_calls, tokens, confidence = await key_obj.value(client, modelname, output_type)
        return (value, item, api_calls, tokens, confidence)

    if key_class == Pointwise_Key:
        tasks = [compute_sort_key(item, key_class, prompt_template) for item in data]
        results = await asyncio.gather(*tasks)
        sorted_results = sorted(results, key=lambda x: (x[0], x[1]))
    else:
        tasks = [ compute_sort_key(text, key_class, prompt_template) for doc_id, text in data]
        results = await asyncio.gather(*tasks)
        sorted_results = [(score, doc_id, api_calls, tokens, confidence, text) for (doc_id, _), (score, text, api_calls, tokens, confidence) in zip(data, results)]
        sorted_results.sort(key=lambda r: (r[1], len(r[-1])))
        sorted_results = [r[:-1] for r in sorted_results]

    sorted_data = [item for _, item, _, _, _ in sorted_results]
    data_confidence = [confidence for _, _, _, _, confidence in sorted_results]
    scores = [score for score, _, _, _, _ in sorted_results]

    total_api_calls = sum(api_calls for _, _, api_calls, _, _ in results)
    total_tokens = sum(tokens for _, _, _, tokens, _ in results)

    if key_class == Pointwise_Key:
        return sorted_data, total_api_calls, total_tokens, data_confidence
    return sorted_data, scores, total_api_calls, total_tokens, data_confidence


# async def bubble_sort(data, client, prompt_template, modelname):
#     n = len(data)
#     total_api_calls = 0
#     total_tokens = 0
#     wrapped_data = [Pair_Comparison_Key(item) for item in data]

#     for i in range(n):
#         for j in range(0, n - i - 1):
#             comparison_result, api_calls, tokens = await wrapped_data[j].compare(
#                 wrapped_data[j + 1], client, prompt_template, modelname
#             )
#             total_api_calls += api_calls
#             total_tokens += tokens
#             if comparison_result == 1:
#                 wrapped_data[j], wrapped_data[j + 1] = wrapped_data[j + 1], wrapped_data[j]

#     sorted_data = [item.key for item in wrapped_data]
#     return sorted_data, total_api_calls, total_tokens



async def quick_sort(data, client, prompt_template, modelname, isPassage, vote = 1):
    random.seed(0)
    if len(data) <= 1:
        return data, 0, 0

    total_api_calls = 0
    total_tokens = 0
    schema = ComparisonReasoning
    if isPassage:
        schema = PassageComparisonReasoning

    pivot = Pair_Comparison_Key(data[0], schema)
    pivot_item = data[0]
    less = []
    greater = []

    for item in data[1:]:
        wrapped_item = Pair_Comparison_Key(item, schema)
        comparison_result, api_calls, tokens = await wrapped_item.compare(
            pivot, client, prompt_template, modelname
        )
        total_api_calls += api_calls
        total_tokens += tokens
        
        if vote == 1:
            if comparison_result == -1:
                less.append(item)
            else:
                greater.append(item)
        else:
            put_in_less = (comparison_result == -1)
            validation_votes = 1  # already have one vote
            total_votes = 1
            sample_pool = greater if put_in_less else less
            sample_size = min(vote - 1, len(sample_pool))
            sampled_items = random.sample(sample_pool, sample_size)
            for peer_item in sampled_items:
                total_votes += 1
                peer_key = Pair_Comparison_Key(peer_item, schema)
                additional_result, additional_calls, additional_tokens = await wrapped_item.compare(
                    peer_key, client, prompt_template, modelname
                )
                total_api_calls += additional_calls
                total_tokens += additional_tokens

                if put_in_less and additional_result == -1:
                    validation_votes += 1
                elif not put_in_less and additional_result != -1:
                    validation_votes += 1
            if validation_votes < (total_votes + 1) // 2:
                put_in_less = not put_in_less  # Flip decision
            if put_in_less:
                less.append(item)
            else:
                greater.append(item)

    # Parallel recursive sort
    left_task = asyncio.create_task(quick_sort(less, client, prompt_template, modelname, isPassage, vote))
    right_task = asyncio.create_task(quick_sort(greater, client, prompt_template, modelname, isPassage, vote))

    sorted_less, left_api_calls, left_tokens = await left_task
    sorted_greater, right_api_calls, right_tokens = await right_task

    total_api_calls += left_api_calls + right_api_calls
    total_tokens += left_tokens + right_tokens

    return sorted_less + [pivot_item] + sorted_greater, total_api_calls, total_tokens

# async def heap_sort(data, client, prompt_template, modelname):
#     async def heapify(data, n, i):
#         largest = i
#         left = 2 * i + 1
#         right = 2 * i + 2
#         api_calls = 0
#         tokens = 0

#         if left < n:
#             comparison_result, calls, toks = await data[left].compare(data[largest], client, prompt_template, modelname)
#             api_calls += calls
#             tokens += toks
#             if comparison_result == 1:
#                 largest = left

#         if right < n:
#             comparison_result, calls, toks = await data[right].compare(data[largest], client, prompt_template, modelname)
#             api_calls += calls
#             tokens += toks
#             if comparison_result == 1:
#                 largest = right

#         if largest != i:
#             data[i], data[largest] = data[largest], data[i]
#             recursive_calls, recursive_toks = await heapify(data, n, largest)
#             api_calls += recursive_calls
#             tokens += recursive_toks

#         return api_calls, tokens

#     n = len(data)
#     total_api_calls = 0
#     total_tokens = 0

#     wrapped_data = [Pair_Comparison_Key(item) for item in data]

#     # Build heap (heapify each node from bottom up)
#     for i in range(n // 2 - 1, -1, -1):
#         api_calls, tokens = await heapify(wrapped_data, n, i)
#         total_api_calls += api_calls
#         total_tokens += tokens

#     # Extract elements from heap
#     for i in range(n - 1, 0, -1):
#         wrapped_data[0], wrapped_data[i] = wrapped_data[i], wrapped_data[0]
#         api_calls, tokens = await heapify(wrapped_data, i, 0)
#         total_api_calls += api_calls
#         total_tokens += tokens

#     sorted_data = [item.key for item in wrapped_data]
#     return sorted_data, total_api_calls, total_tokens


async def external_bubble_sort(data, sortfunc, k, client, prompt_template, modelname, isPassage=False):
    total_api_calls = 0
    total_tokens = 0
    n = len(data)
    for pass_end in range(n, 0, -k // 2):  # Shrink the range in each pass
        if pass_end < k // 2:
            sorted_chunk, num, tokens = await sortfunc(data[:pass_end], client, prompt_template, modelname, isPassage)
            assert len(sorted_chunk) == len(data[:pass_end])
            total_api_calls += num
            total_tokens += tokens
            data[:pass_end] = sorted_chunk
            break

        start = 0
        while start <= pass_end - k // 2:
            end = min(start + k, n)
            chunk = data[start:end]
            sorted_chunk, num, tokens = await sortfunc(chunk, client, prompt_template, modelname, isPassage)
            assert len(sorted_chunk) == len(chunk)
            total_api_calls += num
            total_tokens += tokens
            data[start:end] = sorted_chunk
            start += k // 2
    return data, total_api_calls, total_tokens



async def external_merge_sort(data, sortfunc, k, client, prompt_template, modelname, isPassage = False):
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

        result_length = len(l1) + len(l2)
        i, j = [0], [0]  # Use lists for mutable indices in helpers
        merged = []
        buffer = []
        membership = collections.defaultdict(list)
        buffer_metadata = {'l1': 0, 'l2': 0}
        total_api_calls = 0
        total_tokens = 0
        half = max(k // 2, 1)

        done_early = False

        while i[0] < len(l1) or j[0] < len(l2):
            if i[0] >= len(l1) and buffer_metadata['l1'] == 0:
                merged.extend(buffer + l2[j[0]:])
                done_early = True
                break
            if j[0] >= len(l2) and buffer_metadata['l2'] == 0:
                merged.extend(buffer + l1[i[0]:])
                done_early = True
                break

            fill_buffer(i, j, buffer, membership, buffer_metadata, half)

            original_buffer = buffer[:]
            buffer, num, tokens = await sortfunc(buffer, client, prompt_template, modelname, isPassage)
            total_api_calls += num
            total_tokens += tokens

            assert len(original_buffer) == len(buffer)

            buffer = fix_duplicates(buffer, original_buffer)

            assert len(original_buffer) == len(buffer), print('after fix duplicate')

            items_taken = 0
            for key in buffer:
                if not membership[key]:
                    print("LLM returned extra or duplicate key:", key)
                    continue
                source = membership[key].pop()
                buffer_metadata[source] -= 1
                merged.append(key)
                items_taken += 1
                if buffer_metadata[source] == 0:
                    break

            buffer = buffer[items_taken:]

        if buffer and not done_early:
            merged.extend(buffer)
        assert len(merged) == result_length, print(len(merged), result_length)
        return merged, total_api_calls, total_tokens

    # Step 1: Chunk sorting in parallel
    chunks = [data[i:i + k] for i in range(0, len(data), k)]
    sort_tasks = [sortfunc(chunk, client, prompt_template, modelname, isPassage) for chunk in chunks]
    sort_results = await asyncio.gather(*sort_tasks)

    total_api_calls = sum(r[1] for r in sort_results)
    total_tokens = sum(r[2] for r in sort_results)
    sorted_chunks = [r[0] for r in sort_results]

    for i, arr in enumerate(sorted_chunks):
        assert len(arr) == len(chunks[i]), print("check sorted chunks")

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


async def determine_external_pointwise_memory_size(data, sortfunc, client, prompt_template, modelname, output_type, schema, diff_tolerance, threshold=0.5, max_m=32):
    m = 2
    max_len = len(data)
    if "mini" in modelname:
        max_m = 8

    while m * 2 <= max_len and m <= max_m:
        batch1 = data[:m]
        batch2 = data[m:2*m]
        batch3 = batch1 + batch2  # full 2m batch

        results = await asyncio.gather(
            sortfunc(batch1, client, prompt_template, modelname, output_type, schema),
            sortfunc(batch2, client, prompt_template, modelname, output_type, schema),
            sortfunc(batch3, client, prompt_template, modelname, output_type, schema),
        )

        vals1, _, _, _ = results[0]
        vals2, _, _, _ = results[1]
        vals3, _, _, _ = results[2]

        if len(vals3) != 2*m or len(vals2) != m or len(vals1) != m:
            break

        combined_vals = vals1 + vals2
        # Convert values to output_type
        combined_vals = [output_type(val) for val in combined_vals]
        assert len(combined_vals)==2*m, f"should be 2*m, m={m}"

        batch3_vals = [output_type(val) for val in vals3]

        agree_count = 0
        for key, v1, v2 in zip(batch3, combined_vals, batch3_vals):
            if output_type in (float, int):
                if abs(v1-v2) <= diff_tolerance:
                    agree_count += 1
                else:
                    print(f"from window m={m}: {v1}", f"from window 2*m: {v2}")
            else:
                if v1 == v2:
                    agree_count += 1
                else:
                    print(f"from window m={m}: {v1}", f"from window 2*m: {v2}")
        agreement_ratio = agree_count / (2 * m)

        if agreement_ratio >= threshold:
            m *= 2
        else:
            print(f"Final memory size for external pointwise: m = {m} (agreement = {agreement_ratio:.2f})")
            return m
    print(f"Final memory size for external pointwise: m = {m} (agreement = {agreement_ratio:.2f})")
    return m

async def external_pointwise_sort(data, sortfunc, client, prompt_template, modelname, output_type, diff_tolerance = 0.01, isPassage=False):
    total_api_calls = 0
    total_tokens = 0
    key_and_value = {}
    data_confidence = {}
    key_and_text = {}


    if not isPassage:
        # define memory size
        m =  await determine_external_pointwise_memory_size(data, sortfunc, client, prompt_template, modelname, output_type, ExternalPointwiseReasoning, diff_tolerance)

        chunks = [data[i:i + m] for i in range(0, len(data), m)]
        tasks = [sortfunc(chunk, client, prompt_template, modelname, output_type, ExternalPointwiseReasoning) for chunk in chunks]
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
        confidence = [data_confidence[k] for k in sorted_data]

        return sorted_data, total_api_calls, total_tokens, confidence
    else:
        passage_texts = [text for id, text in data]
        passage_ids = [id for id, text in data]

        # define memory size
        m =  await determine_external_pointwise_memory_size(passage_texts, sortfunc, client, prompt_template, modelname, output_type, PassageExternalPointwiseReasoning, diff_tolerance)


        chunks = [passage_texts[i:i + m] for i in range(0, len(passage_texts), m)]
        id_chunks = [passage_ids[i:i + m] for i in range(0, len(passage_texts), m)]
        tasks = [sortfunc(chunk, client, prompt_template, modelname, output_type, PassageExternalPointwiseReasoning) for chunk in chunks]
        results = await asyncio.gather(*tasks)

        for chunk, text_chunk, (chunk_vals, api_calls, tokens, confidences) in zip(id_chunks, chunks, results):
            total_api_calls += api_calls
            total_tokens += tokens
            for k, v, text, conf in zip(chunk, chunk_vals, text_chunk, confidences):
                try:
                    key_and_value[k] = output_type(v)
                    data_confidence[k] = conf
                    key_and_text[k] = text
                except Exception as e:
                    print(e)
                    key_and_value[k] = output_type(0.0)
                    data_confidence[k] = 0
        sorted_data = sorted(key_and_value, key=lambda k: (key_and_value[k], k))
        confidence = [data_confidence[k] for k in sorted_data]
        scores = [key_and_value[k] for k in sorted_data]
        sorted_data_text = [key_and_text[k] for k in sorted_data]
        
        return sorted_data, scores, total_api_calls, total_tokens, confidence, sorted_data_text


async def hybrid_sort(data, sortfunc, vote, client, pointwise_prompt_template, comparison_prompt_template, modelname, output_type, diff=0.01, isPassage = False):
    total_api_calls = 0
    total_tokens = 0

    if not isPassage:
        sorted_list, api_calls, tokens, confidence = await external_pointwise_sort(data, sortfunc, client, pointwise_prompt_template, modelname, output_type, diff, isPassage)
    else:
        sorted_list, scores, api_calls, tokens, confidence, texts = await external_pointwise_sort(data, sortfunc, client, pointwise_prompt_template, modelname, output_type, diff, isPassage)
        sorted_list = [(docid, text) for docid, text in zip(sorted_list, texts)]
    total_api_calls  += api_calls
    total_tokens += tokens

    max_confidence = max(confidence)
    median_confidence = statistics.median(confidence)

    if isPassage:
        schema = PassageComparisonReasoning
    else:
        schema = ComparisonReasoning

    # Heap of (confidence, index)
    heap = [(conf, idx) for idx, conf in enumerate(confidence)]
    heapq.heapify(heap)  # Min-heap based on confidence

    sort_low_conf = []
    while heap[0][0] < 7 or len(heap) > 0.95 * len(data):
        conf, idx = heapq.heappop(heap)
        item = sorted_list[idx]
        if conf >= median_confidence or conf == max_confidence:
            break
        sort_low_conf.append(item)

    # Remove the selected low-confidence items from sorted_list
    sorted_list = [item for item in sorted_list if item not in sort_low_conf]

    for item in sort_low_conf[::-1]:
        key_item = Pair_Comparison_Key(item, schema)
        left, right = 0, len(sorted_list)

        while left < right:
            mid = (left + right) // 2

            # Step 1: Compare with mid
            key_mid = Pair_Comparison_Key(sorted_list[mid], schema)
            cmp_mid, api_calls, tokens = await key_item.compare(
                key_mid, client, comparison_prompt_template, modelname
            )
            total_api_calls += api_calls
            total_tokens += tokens

            if cmp_mid == -1:
                
                sampled_indices = [i for i in range(mid + 1, min(mid + vote + 1, len(sorted_list)))]

                # item < mid → should also be < mid+1 and mid+2 if mid is in correct order
                disagree = 0
                total_votes = 1
                for offset in sampled_indices:
                    if offset < right:
                        key_check = Pair_Comparison_Key(sorted_list[offset], schema)
                        cmp_check, api_calls, tokens = await key_item.compare(
                            key_check, client, comparison_prompt_template, modelname
                        )
                        total_api_calls += api_calls
                        total_tokens += tokens
                        total_votes += 1
                        if cmp_check == 1:  # item > mid+1 or mid+2
                            disagree += 1

                if disagree > total_votes//2:
                    left = mid + 1  # contradicts original direction
                else:
                    right = mid
            else:
                sampled_indices = [ i for i in range(max(mid - vote, left), mid)]
                disagree = 0
                total_votes = 1
                for offset in sampled_indices:
                    if offset >= left:
                        key_check = Pair_Comparison_Key(sorted_list[offset], schema)
                        cmp_check, api_calls, tokens = await key_item.compare(
                            key_check, client, comparison_prompt_template, modelname
                        )
                        total_api_calls += api_calls
                        total_tokens += tokens
                        total_votes += 1
                        if cmp_check == -1:  # item < mid-1 or mid-2
                            disagree += 1

                if disagree > total_votes // 2:
                    right = mid  # contradicts original direction
                else:
                    left = mid + 1
        # Final insertion
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