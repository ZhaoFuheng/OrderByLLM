from .pointwise import Pointwise_Key, external_values
from .pair_comparison import Pair_Comparison_Key, external_comparisons
import collections

def pointwise_sort(data, client, prompt_template, modelname, output_type):
    total_api_calls = 0
    total_tokens = 0
    def sort_key(pointwise_key):
        nonlocal total_api_calls
        nonlocal total_tokens
        prompt = prompt_template.format(key=str(pointwise_key.key))
        value, api_calls, tokens = pointwise_key.value(client, prompt, modelname, output_type)
        total_api_calls += api_calls
        total_tokens += tokens
        return value
    
    sorted_data = sorted(
        data,
        key=lambda item: (sort_key(Pointwise_Key(item)), item)
        )
    return sorted_data, total_api_calls, total_tokens
    

def bubble_sort(data, client, prompt_template, modelname):
    n = len(data)
    total_api_calls = 0
    total_tokens = 0

    wrapped_data = [Pair_Comparison_Key(item) for item in data]

    for i in range(n):
        for j in range(0, n - i - 1):
            comparison_result, api_calls, tokens = wrapped_data[j].compare(
                wrapped_data[j + 1], client, prompt_template, modelname
            )
            total_api_calls += api_calls
            total_tokens += tokens
            if comparison_result == 1:  # wrapped_data[j] > wrapped_data[j + 1]
                wrapped_data[j], wrapped_data[j + 1] = wrapped_data[j + 1], wrapped_data[j]

    sorted_data = [item.key for item in wrapped_data]
    return sorted_data, total_api_calls, total_tokens


def quick_sort(data, client, prompt_template, modelname):
    if len(data) <= 1:
        return data, 0, 0
    total_api_calls = 0
    total_tokens = 0

    # Wrap pivot and data items in Pair_Comparison_Key
    # print('pivot: ', data[0])
    pivot = Pair_Comparison_Key(data[0])
    less = []
    greater = []

    for item in data[1:]:
        wrapped_item = Pair_Comparison_Key(item)

        comparison_result, api_calls, tokens = wrapped_item.compare(
            pivot, client, prompt_template, modelname
        )
        total_api_calls += api_calls
        total_tokens += tokens

        if comparison_result == -1:  # wrapped_item < pivot
            less.append(item)
        else:
            greater.append(item)

    sorted_less, left_api_calls, left_tokens = quick_sort(less, client, prompt_template, modelname)
    sorted_greater, right_api_calls, right_tokens = quick_sort(greater, client, prompt_template, modelname)

    total_api_calls += left_api_calls + right_api_calls
    total_tokens += left_tokens  + right_tokens

    return sorted_less + [pivot.key] + sorted_greater, total_api_calls, total_tokens


def heap_sort(data, client, prompt_template, modelname):
    def heapify(data, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        api_calls = 0
        tokens = 0
        if left < n:
            comparison_result, calls, toks = data[left].compare(data[largest], client, prompt_template, modelname)
            api_calls += calls
            tokens += toks
            if comparison_result == 1:  # data[left] > data[largest]
                largest = left

        if right < n:
            comparison_result, calls, toks = data[right].compare(data[largest], client, prompt_template, modelname)
            api_calls += calls
            tokens += toks
            if comparison_result == 1:
                largest = right

        # Swap and continue heapifying if needed
        if largest != i:
            data[i], data[largest] = data[largest], data[i]
            additional_calls, additional_toks = heapify(data, n, largest)
            api_calls += additional_calls
            tokens += additional_toks

        return api_calls, tokens

    n = len(data)
    total_api_calls = 0
    total_tokens = 0

    wrapped_data = [Pair_Comparison_Key(item) for item in data]
    for i in range(n // 2 - 1, -1, -1):
        api_calls, tokens = heapify(wrapped_data, n, i)
        total_api_calls += api_calls
        total_tokens += tokens
    for i in range(n - 1, 0, -1):
        wrapped_data[i], wrapped_data[0] = wrapped_data[0], wrapped_data[i]
        api_calls, tokens = heapify(wrapped_data, i, 0)
        total_api_calls += api_calls
        total_tokens += tokens

    sorted_data = [item.key for item in wrapped_data]
    return sorted_data, total_api_calls, total_tokens

def external_bubble_sort(data, sortfunc, k, client, prompt_template, modelname):
    total_api_calls = 0
    total_tokens = 0
    n = len(data)
    for pass_end in range(n, 0, -k // 2):  # Shrink the range in each pass
        if pass_end < k // 2:
            sorted_chunk, num, tokens = sortfunc(data[:pass_end], client, prompt_template, modelname)
            total_api_calls += num
            total_tokens += tokens
            data[:pass_end] = sorted_chunk
            break
        start = 0
        while start <= pass_end - k // 2:
            end = min(start + k, n)
            chunk = data[start:end]
            sorted_chunk, num, tokens = sortfunc(chunk, client, prompt_template, modelname)
            total_api_calls += num
            total_tokens += tokens
            data[start:end] = sorted_chunk
            start += k // 2

    return data, total_api_calls, total_tokens


def external_merge_sort(data, sortfunc, k, client, prompt_template, modelname):
    def merge_sorted_chunks(l1, l2, sortfunc, k, client, prompt_template, modelname):
        i, j = 0, 0
        merged = []
        buffer = []
        membership = collections.defaultdict(list)
        buffer_metadata = {'l1':0, 'l2':0}
        total_api_calls = 0
        total_tokens = 0
        half = max((k - len(buffer)) // 2, 1)

        while i < len(l1) or j < len(l2):
            if i >= len(l1):
                half = k
                if buffer_metadata['l1'] == 0:
                    merged.extend(buffer)
                    buffer = []
                    merged.extend(l2[j:])
                    break

            if j >= len(l2):
                half = k
                if buffer_metadata['l2'] == 0:
                    merged.extend(buffer)
                    buffer = []
                    merged.extend(l1[i:])
                    break

            count = 0
            for _ in range(half - buffer_metadata['l1']):
                if i < len(l1):
                    buffer.append(l1[i])
                    membership[l1[i]].append('l1')
                    i += 1
                    count += 1
                else:
                    break
            buffer_metadata['l1'] += count


            count = 0
            for _ in range(half - buffer_metadata['l2']):
                if j < len(l2):
                    buffer.append(l2[j])
                    membership[l2[j]].append('l2')
                    j += 1
                    count += 1
                else:
                    break
            buffer_metadata['l2'] += count

            assert len(buffer) == sum(buffer_metadata.values()), print(buffer, buffer_metadata, membership)

            buffer, num, tokens =  sortfunc(buffer, client, prompt_template, modelname)
            # print(buffer, buffer_metadata, membership)
            total_api_calls += num
            total_tokens += tokens
            items_taken = 0
            for key in buffer:
                merged.append(key)
                items_taken += 1
                key_from_list = membership[key]
                source = key_from_list.pop(-1)
                buffer_metadata[source] -= 1

                if len(key_from_list) == 0:
                    del membership[key]
                else:
                    membership[key] = key_from_list
                if buffer_metadata[source] == 0:
                    break
            assert items_taken > 0
            buffer = buffer[items_taken:]
        if buffer:
            merged.extend(buffer)
        return merged, total_api_calls, total_tokens
 
    # Divide data into chunks of size k and sort each chunk
    chunks = []
    total_api_calls = 0
    total_tokens = 0
    for i in range(0, len(data), k):
        chunk = data[i:i + k]  # Take a chunk of size k or less
        sorted_chunk, num, tokens =  sortfunc(chunk, client, prompt_template, modelname)
        total_api_calls += num
        total_tokens += tokens
        chunks.append(sorted_chunk)
    
    while len(chunks) > 1:
        # print(chunks)
        merged_chunks = []
        # 2-way merge
        for i in range(0, len(chunks), 2):
            if i + 1 < len(chunks):
                merged_chunk, num, tokens = merge_sorted_chunks(chunks[i], chunks[i + 1], sortfunc, k, client, prompt_template, modelname)
                total_api_calls += num
                total_tokens += tokens
            else:
                merged_chunk = chunks[i]
            merged_chunks.append(merged_chunk)
        
        chunks = merged_chunks

    return chunks[0], total_api_calls, total_tokens





def external_pointwise_sort(data, sortfunc, m, client, prompt_template, modelname, output_type):
    total_api_calls = 0
    total_tokens = 0
    n = len(data)
    key_and_value = {}
    start = 0
    while start < len(data):
        chunk = data[start : start+m]
        start += m
        chunk_vals, num, tokens = sortfunc(chunk, client, prompt_template, modelname, output_type)
        total_api_calls += num
        total_tokens += tokens
        for k, v in zip(chunk, chunk_vals):
            try:
                key_and_value[k] = output_type(v)
            except Exception as e:
                 key_and_value[k] = v
    sorted_data = sorted(
        key_and_value,
        key=lambda k: (key_and_value[k], k)
        )
    return sorted_data, total_api_calls, total_tokens


from openai import OpenAI
import os
if __name__ == "__main__":
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    modelname = 'gpt-4o'

    prompt_template = "In scale 1-100, how friendly is {key}?\n Output an int.\n"
    sorted_data, num, tokens = pointwise_sort(['cat', 'tiger', 'dolphin'], client, prompt_template, modelname, int)
    print(sorted_data, num, tokens)

    prompt_template = "In scale 1-100, how friendly are {keys}?\n Output a json array of integers, where each index corresponds to the respective key in the provided list.\n"
    sorted_data, num, tokens = external_pointwise_sort(['cat', 'tiger', 'dolphin'], external_values, 4, client, prompt_template, modelname, int)
    print(sorted_data, num, tokens)

    prompt_template = "Which is greater {key1} or {key2}? Output the greater key.\n"
    # ignore bubble sort as it requries too many api calls when list is long.
    # sorted_data, num, tokens = bubble_sort([34, 87, 12, 59, 3, 71, 45, 90], client, prompt_template, modelname)
    # print("bubble sort: ", sorted_data, num, tokens)
    sorted_data, num, tokens = quick_sort([34, 87, 12, 59, 3, 71, 45, 90], client, prompt_template, modelname)
    print("quick sort: ", sorted_data, num, tokens)
    sorted_data, num, tokens = heap_sort([34, 87, 12, 59, 3, 71, 45, 90], client, prompt_template, modelname)
    print("heap sort: ", sorted_data, num, tokens)


    prompt_template = "Given a list of keys: {keys}\nSort the keys in ascending order.\n"
    sorted_data, num, tokens = external_bubble_sort([34, 87, 12, 59, 3, 71, 45, 90], external_comparisons, 4, client, prompt_template, modelname)
    print("external bubble sort: ", sorted_data, num, tokens)
    sorted_data, num, tokens = external_merge_sort([34, 87, 12, 59, 3, 71, 45, 90], external_comparisons, 4, client, prompt_template, modelname)
    print("external merge sort ", sorted_data, num, tokens)