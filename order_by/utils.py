import tiktoken

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
    out_of_place_count = 0
    for i, item in enumerate(predict):
        if i < len(gold):
            if gold[i] != predict[i]:
                out_of_place_count += 1
        else:
            out_of_place_count += 1
    return out_of_place_count
    
if __name__ == "__main__":
    sample_text = "This is a sample text to calculate token count."
    token_count = count_tokens(sample_text)
    print(f"Number of tokens: {token_count}")