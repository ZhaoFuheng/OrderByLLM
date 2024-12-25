from pointwise import Pointwise_Key
from point_comparison import Point_Comparison_Key, bulk_sort

def pointwise_sort(data, client, prompt_template, modelname, output_type):
    total_api_calls = 0
    def sort_key(pointwise_key):
        nonlocal total_api_calls
        prompt = prompt_template.format(key=str(pointwise_key.key))
        value, api_calls = pointwise_key.value(client, prompt, modelname, output_type)
        total_api_calls += api_calls
        return value
    
    sorted_data = sorted(data, key=lambda item: sort_key(Pointwise_Key(item)))
    return sorted_data, total_api_calls
    

def bubble_sort(data, client, prompt_template, modelname):
    n = len(data)
    total_api_calls = 0

    wrapped_data = [Point_Comparison_Key(item) for item in data]

    for i in range(n):
        for j in range(0, n - i - 1):
            comparison_result, api_calls = wrapped_data[j].compare(
                wrapped_data[j + 1], client, prompt_template, modelname
            )
            total_api_calls += api_calls
            if comparison_result == 1:  # wrapped_data[j] > wrapped_data[j + 1]
                wrapped_data[j], wrapped_data[j + 1] = wrapped_data[j + 1], wrapped_data[j]

    sorted_data = [item.key for item in wrapped_data]
    return sorted_data, total_api_calls


def quick_sort(data, client, prompt_template, modelname):
    if len(data) <= 1:
        return data, 0
    total_api_calls = 0

    # Wrap pivot and data items in Point_Comparison_Key
    pivot = Point_Comparison_Key(data[0])
    less = []
    greater = []

    for item in data[1:]:
        wrapped_item = Point_Comparison_Key(item)

        comparison_result, api_calls = wrapped_item.compare(
            pivot, client, prompt_template, modelname
        )
        total_api_calls += api_calls

        if comparison_result == -1:  # wrapped_item < pivot
            less.append(item)
        else:
            greater.append(item)

    sorted_less, left_api_calls = quick_sort(less, client, prompt_template, modelname)
    sorted_greater, right_api_calls = quick_sort(greater, client, prompt_template, modelname)

    total_api_calls += left_api_calls + right_api_calls

    return sorted_less + [pivot.key] + sorted_greater, total_api_calls


def heap_sort(data, client, prompt_template, modelname):
    def heapify(data, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        api_calls = 0
        if left < n:
            comparison_result, calls = data[left].compare(data[largest], client, prompt_template, modelname)
            api_calls += calls
            if comparison_result == 1:  # data[left] > data[largest]
                largest = left

        if right < n:
            comparison_result, calls = data[right].compare(data[largest], client, prompt_template, modelname)
            api_calls += calls
            if comparison_result == 1:
                largest = right

        # Swap and continue heapifying if needed
        if largest != i:
            data[i], data[largest] = data[largest], data[i]
            additional_calls = heapify(data, n, largest)
            api_calls += additional_calls

        return api_calls

    n = len(data)
    total_api_calls = 0

    wrapped_data = [Point_Comparison_Key(item) for item in data]
    for i in range(n // 2 - 1, -1, -1):
        total_api_calls += heapify(wrapped_data, n, i)

    for i in range(n - 1, 0, -1):
        wrapped_data[i], wrapped_data[0] = wrapped_data[0], wrapped_data[i]
        total_api_calls += heapify(wrapped_data, i, 0)

    sorted_data = [item.key for item in wrapped_data]
    return sorted_data, total_api_calls



def external_bubble_sort(data, sortfunc, k, client, prompt_template, modelname):
    total_api_calls = 0
    n = len(data)
    for pass_end in range(n, 0, -k // 2):  # Shrink the range in each pass
        if pass_end < k // 2:
            sorted_chunk, num = sortfunc(data[:pass_end], client, prompt_template, modelname)
            total_api_calls += num
            data[:pass_end] = sorted_chunk
            break
        for start in range(0, pass_end - k // 2, k // 2):
            end = min(start + k, n)
            chunk = data[start:end]
            sorted_chunk, num = sortfunc(chunk, client, prompt_template, modelname)
            total_api_calls += num
            data[start:end] = sorted_chunk

    return data, total_api_calls


def external_merge_sort(data, sortfunc, k, client, prompt_template, modelname):
    def merge_sorted_chunks(l1, l2, sortfunc, k, client, prompt_template, modelname):
        i, j = 0, 0
        merged = []
        buffer = []
        total_api_calls = 0
        while i < len(l1) or j < len(l2):
            last_items = set()
            if j >= len(l2):
                half = k - len(buffer)
            else:
                half = (k - len(buffer)) // 2

            last_item = None
            for _ in range(half):
                if i < len(l1):
                    last_item = l1[i]
                    buffer.append(l1[i])
                    i += 1
            if last_item is not None:
                last_items.add(last_item)
            
            last_item = None
            for _ in range(k - len(buffer)):
                if j < len(l2):
                    last_item = l2[j]
                    buffer.append(l2[j])
                    j += 1
            if last_item is not None:
                last_items.add(last_item)

            buffer, num =  sortfunc(buffer, client, prompt_template, modelname)
            total_api_calls += num
            items_taken = 0
            for key in buffer:
                merged.append(key)
                items_taken += 1
                if key in last_items:
                    break
            assert items_taken > 0
            buffer = buffer[items_taken:]
        if buffer:
            merged.extend(buffer)
        return merged, total_api_calls
 
    # Divide data into chunks of size k and sort each chunk
    chunks = []
    total_api_calls = 0
    for i in range(0, len(data), k):
        chunk = data[i:i + k]  # Take a chunk of size k or less
        sorted_chunk, num =  sortfunc(chunk, client, prompt_template, modelname)
        total_api_calls += num
        chunks.append(sorted_chunk)
    
    while len(chunks) > 1:
        print(chunks)
        merged_chunks = []
        # 2-way merge
        for i in range(0, len(chunks), 2):
            if i + 1 < len(chunks):
                merged_chunk, num = merge_sorted_chunks(chunks[i], chunks[i + 1], sortfunc, k, client, prompt_template, modelname)
                total_api_calls += num
            else:
                merged_chunk = chunks[i]
            merged_chunks.append(merged_chunk)
        
        chunks = merged_chunks

    return chunks[0], total_api_calls


from openai import OpenAI
import os
if __name__ == "__main__":
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    modelname = 'gpt-4o'

    # prompt_template = "In scale 1-100, how friendly is {key}? Output an int without explanation.\n"
    # sorted_data, num = pointwise_sort(['cat', 'tiger', 'dolphin'], client, prompt_template, modelname, int)
    # print(sorted_data, num)

    prompt_template = "Which is greater {key1} or {key2}? Output the greater one with no explanation.\n"
    sorted_data, num = bubble_sort([34, 87, 12, 59, 3, 71, 45, 90, 28, 64], client, prompt_template, modelname)
    print("bubble sort: ", sorted_data, num)
    sorted_data, num = quick_sort([34, 87, 12, 59, 3, 71, 45, 90, 28, 64], client, prompt_template, modelname)
    print("quick sort: ", sorted_data, num)
    sorted_data, num = heap_sort([34, 87, 12, 59, 3, 71, 45, 90, 28, 64], client, prompt_template, modelname)
    print("heap sort: ", sorted_data, num)


    prompt_template = "Given a list of keys: {keys}\nOutput a sorted json array in ascending order with no explanation.\n"
    sorted_data, num = external_bubble_sort([34, 87, 12, 59, 3, 71, 45, 90, 28, 64], bulk_sort, 4, client, prompt_template, modelname)
    print("external bubble sort: ", sorted_data, num)
    sorted_data, num = external_merge_sort([34, 87, 12, 59, 3, 71, 45, 90, 28, 64], bulk_sort, 4, client, prompt_template, modelname)
    print("external merge sort ", sorted_data, num)