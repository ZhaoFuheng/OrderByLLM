class Point_Comparison_Key:
    def __init__(self, key):
        self.key = key
    
    def get_greater(self, client, prompt, modelname, possibles):
        api_call = 0 
        while api_call < 3:
            api_call += 1
            response = client.chat.completions.create(
                model = modelname,
                messages=[{"role": "user", "content": prompt}],
                n = 1,
                temperature=0.0,
                max_tokens=4096,
            )
            response = [choice.message.content.strip() for choice in response.choices][0]
            if response in possibles:
                return response, api_call
        # return a random item
        return possibles.pop(), api_call

    def compare(self, other, client, prompt_template, modelname):
        if self.key == other.key:
            return 0, 0
        possibles = {str(self.key), str(other.key)}
        prompt = prompt_template.format(key1=str(self.key), key2=str(other.key))
        greater_key, num = self.get_greater(client, prompt, modelname, possibles)

        if greater_key == str(self.key):
            return 1, num
        return -1, num

    
    def __repr__(self):
        return str(self.key)


def create_comparator(client, prompt_template, modelname):
    def comparator(key1, key2):
        return key1.compare(key2, client, prompt_template, modelname)[0]
    return comparator
        

import json
def bulk_sort(data, client, prompt_template, modelname):
    api_call = 0 
    prompt = prompt_template.format(keys = str(data))
    while api_call < 3:
        api_call += 1
        try:
            # Make the API call with "type": "json_object"
            response = client.chat.completions.create(
                model=modelname,
                messages=[{"role": "user", "content": prompt}],
                n=1,
                temperature=0.0,
                max_tokens=4096,
                response_format={ "type": "json_object" }
            )
            response = [choice.message.content.strip() for choice in response.choices][0]
            json_data = json.loads(response)  # Safely decode JSON response
            assert len(json_data.keys()) == 1
            for key, val in json_data.items():
                return val, api_call
        except json.JSONDecodeError as jde:
            print(f"[ERROR] Attempt {api_call}: Failed to decode JSON: {jde}")
        except Exception as e:
            print(f"[ERROR] Attempt {api_call}: {e}")
    return data, api_call

from openai import OpenAI
import os
from functools import cmp_to_key

if __name__ == "__main__":
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt_template = "Which number is greater: {key1} or {key2}?"
    modelname = 'gpt-4o'
    comparator = create_comparator(client, prompt_template, modelname)

    keys = [Point_Comparison_Key(10), Point_Comparison_Key(15), Point_Comparison_Key(1)]
    sorted_keys = sorted(keys, key=cmp_to_key(comparator))

    print("Sorted Keys:", sorted_keys)
