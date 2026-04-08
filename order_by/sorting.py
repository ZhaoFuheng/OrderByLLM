from .pointwise import Pointwise_Key, external_values, PassageExternalPointwiseReasoning, ExternalPointwiseReasoning, SQLExternalPointwiseReasoning, ReviewExternalPointwiseReasoning
from .pair_comparison import *
import collections
import asyncio
import statistics
from typing import Any, List, Callable, Tuple
import hashlib
import random
import heapq
import collections
import logging
log = logging.getLogger(__name__)

random.seed(0)

async def pointwise_sort(data, client, prompt_template, modelname, output_type, key_class = Pointwise_Key, isPassage=True, isReview=False, use_wiki=False, wiki_field=None):
    total_api_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0

    async def compute_sort_key(item, key_class, prompt_template, isPassage=None, isReview=False):
        if use_wiki and key_class == Pointwise_Key:
            from .tools import web_search_pointwise_value
            prompt = prompt_template.format(key=str(item))
            value, api_calls, input_tokens, output_tokens = await web_search_pointwise_value(
                client, modelname, prompt,
                wiki_entity=str(item),
                wiki_field=wiki_field,
            )
            return (value, item, api_calls, input_tokens, output_tokens)

        if isPassage == None:
            assert key_class == Pointwise_Key, print('key_class', key_class)
            key_obj = key_class(item, prompt_template)
        else:
            assert key_class != Pointwise_Key, print('key_class', key_class)
            key_obj = key_class(item, prompt_template, isPassage, isReview)
        value, api_calls, input_tokens, output_tokens = await key_obj.value(client, modelname, output_type)
        return (value, item, api_calls, input_tokens, output_tokens)

    if key_class == Pointwise_Key:
        results = []
        req_batch_size = 20
        for i in range(0, len(data), req_batch_size):
            batch = data[i:i+req_batch_size]
            tasks = [compute_sort_key(item, key_class, prompt_template) for item in batch]
            results_batch = await asyncio.gather(*tasks)
            results.extend(results_batch)   
        sorted_results = sorted(results, key=lambda x: (x[0], x[1]))
    else:
        results = []
        req_batch_size = 20
        for i in range(0, len(data), req_batch_size):
            batch = data[i:i+req_batch_size]
            tasks = [compute_sort_key(text, key_class, prompt_template, isPassage, isReview) for doc_id, text in batch]
            results_batch = await asyncio.gather(*tasks)
            results.extend(results_batch)
        sorted_results = [(score, doc_id, api_calls, in_tokens, out_tokens, text) for (doc_id, _), (score, text, api_calls, in_tokens, out_tokens) in zip(data, results)]
        sorted_results.sort(key=lambda r: (r[1], -len(r[-1])))
        sorted_results = [r[:-1] for r in sorted_results]

    sorted_data = [item for _, item, _, _, _ in sorted_results]
    scores = [score for score, _, _, _, _, in sorted_results]

    total_api_calls = sum(api_calls for _, _, api_calls, _, _, in results)
    total_input_tokens = sum(input_tokens for _, _, _, input_tokens, _ in results)
    total_output_tokens = sum(output_tokens for _, _, _, _, output_tokens in results)

    if key_class == Pointwise_Key:
        return sorted_data, total_api_calls, total_input_tokens, total_output_tokens
    return sorted_data, scores, total_api_calls, total_input_tokens, total_output_tokens


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



async def quick_sort(data, client, prompt_template, modelname, isPassage, vote = 1, isSQL=False, isReview=False, limit_k=None, base_seed: int = 0):
    if len(data) <= 1:
        return data, 0, 0, 0
    # print('quick_sort', len(data))
    total_api_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    s = ComparisonReasoning
    if isPassage:
        s = PassageComparisonReasoning
    if isSQL:
        s = SQLComparisonReasoning
    if isReview:
        s = ReviewComparisonReasoning

    pivot = Pair_Comparison_Key(data[0], s)
    pivot_item = data[0]
    less = []
    greater = []

    if limit_k is None:
        limit_k = len(data)
    else:
        limit_k = min(limit_k, len(data))

    # Deterministic RNG seeded from call context to avoid concurrency nondeterminism
    seed_material = f"{base_seed}|{pivot_item}|{len(data)}"
    import hashlib as _qs_hashlib
    local_seed = int(_qs_hashlib.sha256(seed_material.encode()).hexdigest(), 16) % (2**32)
    local_rng = random.Random(local_seed)

    # First, compare all items to the pivot concurrently
    rest_items = data[1:]
    compare_results = []
    req_batch_size = 25
    wrapped_items = [Pair_Comparison_Key(item, s) for item in rest_items]
    for i in range(0, len(wrapped_items), req_batch_size):
        batch_wrapped_items = wrapped_items[i:i+req_batch_size]
        compare_tasks = [wrapped.compare(pivot, client, prompt_template, modelname) for wrapped in batch_wrapped_items]
        compare_results_batch = await asyncio.gather(*compare_tasks)
        compare_results.extend(compare_results_batch)


    initial_less = []
    initial_greater = []
    initial_decisions = []  # (item, put_in_less)
    for item, (comparison_result, api_calls, input_tokens, output_tokens) in zip(rest_items, compare_results):
        total_api_calls += api_calls
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        put_in_less = (comparison_result == -1)
        initial_decisions.append((item, put_in_less))
        if put_in_less:
            initial_less.append(item)
        else:
            initial_greater.append(item)

    assert len(initial_decisions) == len(data)-1, print('initial_decisions', len(initial_decisions), 'data', len(data))

    if vote == 1:
        less.extend(initial_less)
        greater.extend(initial_greater)
    else:
        items_comparison_results = collections.defaultdict(dict)
        peer_tasks = []
        # Validate each item's placement using peer comparisons; run peers concurrently per item
        for (item, put_in_less) in initial_decisions:
            if put_in_less:
                initial_decision_value = -1
            else:
                initial_decision_value = 1
            items_comparison_results[item][pivot_item] = initial_decision_value 

            wrapped_item = Pair_Comparison_Key(item, s)
            sample_pool = initial_greater if put_in_less else initial_less
            sample_size = min(vote - 1, len(sample_pool))
            sampled_items = local_rng.sample(sample_pool, sample_size)

            if len(sampled_items) <= 1: # only one peer, no need to compare
                if put_in_less:
                    less.append(item)
                else:
                    greater.append(item)
                continue

            for peer_item in sampled_items:
                peer_key = Pair_Comparison_Key(peer_item, s)
                peer_tasks.append([item, peer_item, wrapped_item.compare(peer_key, client, prompt_template, modelname)])

        if peer_tasks:
            coros = [peer_task[2] for peer_task in peer_tasks]
            peer_results = []
            req_batch_size = 25
            for i in range(0, len(coros), req_batch_size):
                batch_coros = coros[i:i+req_batch_size]
                batch_peer_results = await asyncio.gather(*batch_coros)
                peer_results.extend(batch_peer_results)

            for result, [item, peer_item, _] in zip(peer_results, peer_tasks):
                additional_result, additional_calls, additional_input_tokens, additional_output_tokens = result
                total_api_calls += additional_calls
                total_input_tokens += additional_input_tokens
                total_output_tokens += additional_output_tokens
                
                items_comparison_results[item][peer_item] = additional_result
                items_comparison_results[peer_item][item] = -1 * additional_result
                assert additional_result == -1 or additional_result == 1, print('additional_result', additional_result)
        # print('items_comparison_results', items_comparison_results)
        # first pick out the items that are in the correct position
        for (item, put_in_less) in initial_decisions:
            if item in less or item in greater:
                continue
            if put_in_less: # item is in less
                if sum(items_comparison_results[item].values()) == -1 * len(items_comparison_results[item].values()):
                    less.append(item)
            else:
                if sum(items_comparison_results[item].values()) == len(items_comparison_results[item].values()):
                    greater.append(item)

        while_loop_count = 0
        while len(less) + len(greater) < len(initial_decisions):
            while_loop_count += 1
            deadlock = True

            for (item, put_in_less) in initial_decisions:
                if item in less or item in greater:
                    continue
                L_count = 0
                G_count = 0
                if put_in_less:
                    L_count = 1.5
                else:
                    G_count = 1.5
                all_contained = all( (peer_item in less or peer_item in greater) for peer_item in items_comparison_results[item].keys())
                if not all_contained:
                    continue
                deadlock = False
                for peer_item, peer_result in items_comparison_results[item].items():
                    if peer_item in less and peer_result == -1:
                        L_count += 1
                    elif peer_item in less and peer_result == 1:
                        pass
                    if peer_item in greater and peer_result == 1:
                        G_count += 1
                    elif peer_item in greater and peer_result == -1:
                        pass
                # print(f'L_count: {L_count}, G_count: {G_count}, already filled: {len(less)} {len(greater)}')
                if L_count > G_count:
                    less.append(item)
                else:
                    greater.append(item)
            if while_loop_count > 200:
                assert False, print('while loop count too large', while_loop_count)
            if deadlock:
                for (item, put_in_less) in initial_decisions:
                    L_count = 0
                    G_count = 0
                    if put_in_less:
                        L_count = 1.5
                    else:
                        G_count = 1.5
                    if item in less or item in greater:
                        continue   
                    for peer_item, peer_result in items_comparison_results[item].items():
                        if peer_item in greater and peer_result == 1:
                            G_count += 1
                        if peer_item in less and peer_result == -1:
                            L_count += 1
                    if L_count > G_count:
                        less.append(item)
                    else:
                        greater.append(item)
                assert len(less) + len(greater) == len(data)-1, print('less', len(less), 'greater', len(greater), 'initial_decisions', len(initial_decisions), 'data', len(data))
                break
    assert len(less) + len(greater) == len(data)-1, print('less', len(less), 'greater', len(greater), 'initial_decisions', len(initial_decisions), 'data', len(data))

    # Decide how much to sort based on limit_k (sort only the last k elements)
    if limit_k >= len(data):
        # print('full sort', len(data))
        # Full sort
        left_seed = int(_qs_hashlib.sha256(f"{base_seed}|left|{pivot_item}|{len(less)}".encode()).hexdigest(), 16) % (2**32)
        right_seed = int(_qs_hashlib.sha256(f"{base_seed}|right|{pivot_item}|{len(greater)}".encode()).hexdigest(), 16) % (2**32)
        left_task = asyncio.create_task(quick_sort(less[:], client, prompt_template, modelname, isPassage, vote, isSQL, isReview, base_seed=left_seed))
        right_task = asyncio.create_task(quick_sort(greater[:], client, prompt_template, modelname, isPassage, vote, isSQL, isReview, base_seed=right_seed))

        sorted_less, left_api_calls, left_input_tokens, left_output_tokens = await left_task
        sorted_greater, right_api_calls, right_input_tokens, right_output_tokens = await right_task

        total_api_calls += left_api_calls + right_api_calls
        total_input_tokens += left_input_tokens + right_input_tokens
        total_output_tokens += left_output_tokens + right_output_tokens

        return sorted_less + [pivot_item] + sorted_greater, total_api_calls, total_input_tokens, total_output_tokens

    # Partial tail sort: ensure the last k elements are sorted
    k = limit_k
    n_greater = len(greater)

    if k <= n_greater:
        # Tail entirely within greater
        right_seed = int(_qs_hashlib.sha256(f"{base_seed}|right|{pivot_item}|{len(greater)}".encode()).hexdigest(), 16) % (2**32)
        sorted_greater, r_api, r_in, r_out = await quick_sort(
            greater[:], client, prompt_template, modelname, isPassage, vote, isSQL, isReview, limit_k=k, base_seed=right_seed
        )
        total_api_calls += r_api
        total_input_tokens += r_in
        total_output_tokens += r_out
        assert len(sorted_greater) == k, print('sorted_greater', len(sorted_greater), 'k', k)
        return sorted_greater, total_api_calls, total_input_tokens, total_output_tokens

    if k == n_greater + 1:
        # Tail is pivot + all of greater; fully sort greater and return exactly k items
        right_seed = int(_qs_hashlib.sha256(f"{base_seed}|right|{pivot_item}|{len(greater)}".encode()).hexdigest(), 16) % (2**32)
        sorted_greater, r_api, r_in, r_out = await quick_sort(
            greater[:], client, prompt_template, modelname, isPassage, vote, isSQL, isReview, limit_k=n_greater, base_seed=right_seed
        )
        total_api_calls += r_api
        total_input_tokens += r_in
        total_output_tokens += r_out
        return [pivot_item] + sorted_greater, total_api_calls, total_input_tokens, total_output_tokens

    # Tail spans some of less, plus pivot, plus all of greater
    left_k = k - (n_greater + 1)
    left_seed = int(_qs_hashlib.sha256(f"{base_seed}|left|{pivot_item}|{len(less)}".encode()).hexdigest(), 16) % (2**32)
    right_seed = int(_qs_hashlib.sha256(f"{base_seed}|right|{pivot_item}|{len(greater)}".encode()).hexdigest(), 16) % (2**32)
    left_task = asyncio.create_task(
        quick_sort(less[:], client, prompt_template, modelname, isPassage, vote, isSQL, isReview, limit_k=left_k, base_seed=left_seed)
    )
    right_task = asyncio.create_task(
        quick_sort(greater[:], client, prompt_template, modelname, isPassage, vote, isSQL, isReview, limit_k=n_greater, base_seed=right_seed)
    )
    left_sorted, l_api, l_in, l_out = await left_task
    right_sorted, r_api, r_in, r_out = await right_task

    total_api_calls += l_api + r_api
    total_input_tokens += l_in + r_in
    total_output_tokens += l_out + r_out

    # Return exactly k items
    return left_sorted + [pivot_item] + right_sorted, total_api_calls, total_input_tokens, total_output_tokens


async def external_bubble_sort(data, sortfunc, k, client, prompt_template, modelname, isPassage=False, isSQL=False, isReview=False, limit_k=None):
    total_api_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    
    if limit_k is None:
        limit_k = len(data)
    else:
        limit_k = min(limit_k, len(data))

    n = len(data)
    for pass_end in range(n, n-limit_k, -k // 2):  # Shrink the range in each pass
        if pass_end < k // 2:
            sorted_chunk, num, in_tokens, out_tokens = await sortfunc(data[:pass_end], client, prompt_template, modelname, isPassage, isSQL, isReview)
            assert len(sorted_chunk) == len(data[:pass_end])
            total_api_calls += num
            total_input_tokens += in_tokens
            total_output_tokens += out_tokens
            data[:pass_end] = sorted_chunk
            break

        start = 0
        while start <= pass_end - k // 2:
            end = min(start + k, n)
            chunk = data[start:end]
            sorted_chunk, num, in_tokens, out_tokens = await sortfunc(chunk, client, prompt_template, modelname, isPassage, isSQL, isReview)
            assert len(sorted_chunk) == len(chunk)
            total_api_calls += num
            total_input_tokens += in_tokens
            total_output_tokens += out_tokens
            data[start:end] = sorted_chunk
            start += k // 2
    return data[n-limit_k:], total_api_calls,  total_input_tokens, total_output_tokens



async def external_merge_sort(data, sortfunc, k, client, prompt_template, modelname, isPassage = False, isSQL=False, isReview=False, limit_k=None, useCache=True):
    if limit_k is None:
        limit_k = len(data)
    else:
        limit_k = min(limit_k, len(data))

    async def merge_sorted_chunks(l1, l2, limit_k):
        # make l1 and l2 descending
        l1 = l1[::-1]
        l2 = l2[::-1] 

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

        def fill_buffer(list_i, list_j, buffer, membership, buffer_metadata, half):
            """Fill buffer with up to `half` items from each list."""
            """The buffer is more likely to be descending"""
            while buffer_metadata['l1'] < half and list_i[0] < len(l1):
                item = l1[list_i[0]]
                buffer.append(item)
                membership[item].append('l1')
                buffer_metadata['l1'] += 1
                list_i[0] += 1
            while buffer_metadata['l2'] < half and list_j[0] < len(l2):
                item = l2[list_j[0]]
                buffer.append(item)
                membership[item].append('l2')
                buffer_metadata['l2'] += 1
                list_j[0] += 1

        result_length = len(l1) + len(l2)
        list_i, list_j = [0], [0]  # Use lists for mutable indices in helpers
        # merged and buffer will both be descending
        merged = []
        buffer = []
        membership = collections.defaultdict(list)
        buffer_metadata = {'l1': 0, 'l2': 0}
        half = max(k // 2, 1)

        done_early = False

        total_api_calls = 0
        total_input_tokens = 0
        total_output_tokens = 0

        while list_i[0] < len(l1) or list_j[0] < len(l2):
            if len(merged) >= limit_k:
                # print(f'limit_k {limit_k} reached, {l1} {l2} {merged}, total_api_calls {total_api_calls}')
                final_merged = merged[::-1]
                return final_merged[-limit_k:], total_api_calls, total_input_tokens, total_output_tokens
            
            if list_i[0] >= len(l1) and buffer_metadata['l1'] == 0:
                merged.extend(buffer + l2[list_j[0]:])
                done_early = True
                break
            if list_j[0] >= len(l2) and buffer_metadata['l2'] == 0:
                merged.extend(buffer + l1[list_i[0]:])
                done_early = True
                break

            fill_buffer(list_i, list_j, buffer, membership, buffer_metadata, half)

            original_buffer = buffer[:]
            ascending_buffer = buffer[::-1]
            ascending_buffer, num, in_tokens, out_tokens = await sortfunc(ascending_buffer, client, prompt_template, modelname, isPassage, isSQL, isReview, useCache=useCache)
            total_api_calls += num
            total_input_tokens += in_tokens
            total_output_tokens += out_tokens

            assert len(original_buffer) == len(buffer)

            ascending_buffer = fix_duplicates(ascending_buffer, original_buffer)

            assert len(original_buffer) == len(ascending_buffer), print('after fix duplicate')

            items_taken = 0
            # buffer is descending
            buffer = ascending_buffer[::-1]
            for key in buffer:
                if not membership[key]:
                    log.error("LLM returned extra or duplicate key: %s", key)
                    continue
                source = membership[key].pop()
                buffer_metadata[source] -= 1
                merged.append(key)
                items_taken += 1
                if buffer_metadata[source] == 0:
                    break
            # print('before', buffer)
            buffer = buffer[items_taken:]
            # print('after', buffer)

        if buffer and not done_early:
            merged.extend(buffer)
        assert len(merged) == result_length
        # swtich back to ascending
        final_merged = merged[::-1]

        # print(len(l1), len(l2), len(merged), 'total_api_calls', total_api_calls)

        return final_merged[-limit_k:], total_api_calls, total_input_tokens, total_output_tokens

    # Step 1: Chunk sorting in parallel
    assert type(k) == int, print('k', k)
    chunks = [data[i:i + k] for i in range(0, len(data), k)]
    sort_tasks = [sortfunc(chunk, client, prompt_template, modelname, isPassage, isSQL, isReview, useCache=useCache) for chunk in chunks]
    sort_results = await asyncio.gather(*sort_tasks)

    total_api_calls = sum(r[1] for r in sort_results)
    total_input_tokens = sum(r[2] for r in sort_results)
    total_output_tokens = sum(r[3] for r in sort_results)
    sorted_chunks = [r[0] for r in sort_results]

    for i, arr in enumerate(sorted_chunks):
        assert len(arr) == len(chunks[i]), print("check sorted chunks")

    # Step 2: Merging sorted chunks in parallel (repeated passes)
    while len(sorted_chunks) > 1:
        merge_tasks = []
        for i in range(0, len(sorted_chunks), 2):
            if i + 1 < len(sorted_chunks):
                merge_tasks.append(merge_sorted_chunks(sorted_chunks[i], sorted_chunks[i + 1], limit_k))
            else:
                merge_tasks.append(asyncio.sleep(0, result=(sorted_chunks[i], 0, 0, 0)))  # no-op async return

        merge_results = await asyncio.gather(*merge_tasks)
        total_api_calls += sum(r[1] for r in merge_results)
        total_input_tokens  += sum(r[2] for r in merge_results)
        total_output_tokens  += sum(r[3] for r in merge_results)
        sorted_chunks = [r[0] for r in merge_results]

    return sorted_chunks[0], total_api_calls, total_input_tokens, total_output_tokens


async def determine_external_pointwise_memory_size(data, sortfunc, client, prompt_template, modelname, output_type, schema, diff_tolerance, threshold=0.7, max_m=16):
    m = 2
    total_input_tokens, total_output_tokens = 0, 0
    max_len = len(data)
    if "mini" in modelname:
        max_m = 8

    while m * 2 <= max_len and m * 2 <= max_m:
        batch1 = data[:m]
        batch2 = data[m:2*m]
        batch3 = batch1 + batch2  # full 2m batch

        results = await asyncio.gather(
            sortfunc(batch1, client, prompt_template, modelname, output_type, schema),
            sortfunc(batch2, client, prompt_template, modelname, output_type, schema),
            sortfunc(batch3, client, prompt_template, modelname, output_type, schema),
        )

        vals1, _, input_tokens, output_tokens = results[0] 
        if m == 2: # batch 1 will be cached besides the first run
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

        vals2, _, input_tokens, output_tokens = results[1]
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        vals3, _, input_tokens, output_tokens = results[2]
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

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
                # else:
                #     print(f"from window m={m}: {v1}", f"from window 2m: {v2}")
            else:
                if v1 == v2:
                    agree_count += 1
                # else:
                #     print(f"from window m={m}: {v1}", f"from window 2m: {v2}")
        agreement_ratio = agree_count / (2 * m)

        if agreement_ratio >= threshold:
            m *= 2
        else:
            print(f"Final memory size for external pointwise: m = {m} (agreement = {agreement_ratio:.2f}), modelname: {modelname}")
            return m , total_input_tokens, total_output_tokens 
    print(f"Final memory size for external pointwise: m = {m}, modelname: {modelname}")
    return m, total_input_tokens, total_output_tokens 

async def external_pointwise_sort(
    data,
    sortfunc,
    client,
    prompt_template,
    modelname,
    output_type,
    diff_tolerance=0.01,
    isPassage=False,
    isSQL=False,
    isReview=False,
    max_m=16,
    batch_size_data=None,
    memory_size=None,
    use_web_search=False,
    wiki_field=None,
):
    import functools

    total_api_calls = 0
    key_and_value = {} 
    key_and_text = {}

    total_input_tokens = 0
    total_output_tokens = 0

    if wiki_field:
        from .tools import wiki_search_external_values
        sortfunc = functools.partial(wiki_search_external_values, wiki_field=wiki_field)
    elif use_web_search:
        from .tools import web_search_external_values
        sortfunc = web_search_external_values

    if memory_size is None:
        raise ValueError("memory_size must be provided for external_pointwise_sort.")
    m = int(memory_size)
    if m <= 0:
        raise ValueError(f"memory_size must be > 0, got {memory_size}")

    if not isPassage and not isSQL and not isReview:

        chunks = [data[i:i + m] for i in range(0, len(data), m)]
        results = []
        req_batch_size = 20
        for i in range(0, len(chunks), req_batch_size):
            batch_chunks = chunks[i:i+req_batch_size]
            tasks = [
                sortfunc(chunk, client, prompt_template, modelname, output_type, ExternalPointwiseReasoning)
                for chunk in batch_chunks
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        # Aggregate results
        for chunk, (chunk_vals, api_calls, in_tokens, out_tokens) in zip(chunks, results):
            total_api_calls += api_calls
            total_input_tokens += in_tokens
            total_output_tokens += out_tokens

            for k, v in zip(chunk, chunk_vals):
                try:
                    key_and_value[k] = output_type(v)
                except Exception as e:
                    print(e)
                    print("??????????")
                    key_and_value[k] = output_type(0.0)

        # Sort by computed value then original key
        sorted_data = sorted(key_and_value, key=lambda k: (key_and_value[k], k))

        return sorted_data, total_api_calls, total_input_tokens, total_output_tokens
    else:
        if isPassage:
            s = PassageExternalPointwiseReasoning
        elif isReview:
            s = ReviewExternalPointwiseReasoning
        else:
            s = SQLExternalPointwiseReasoning
        texts = [text for id, text in data]
        ids = [id for id, text in data]

        text_chunks = [texts[i:i + m] for i in range(0, len(texts), m)]
        id_chunks = [ids[i:i + m] for i in range(0, len(texts), m)]
        tasks = [sortfunc(text_chunk, client, prompt_template, modelname, output_type, s) for text_chunk in text_chunks]
        results = await asyncio.gather(*tasks)

        for id_chunk, text_chunk, (chunk_vals, api_calls, in_tokens, out_tokens) in zip(id_chunks, text_chunks, results):
            total_api_calls += api_calls
            total_input_tokens += in_tokens
            total_output_tokens += out_tokens
            for k, v, text in zip(id_chunk, chunk_vals, text_chunk):
                try:
                    key_and_value[k] = output_type(v)
                    key_and_text[k] = text
                except Exception as e:
                    print(e)
                    key_and_value[k] = output_type(0.0)
        sorted_data = sorted(key_and_value, key=lambda k: (key_and_value[k], -len(key_and_text[k])))
        scores = [key_and_value[k] for k in sorted_data]
        sorted_data_text = [key_and_text[k] for k in sorted_data]
        
        return sorted_data, scores, total_api_calls, total_input_tokens, total_output_tokens, sorted_data_text
