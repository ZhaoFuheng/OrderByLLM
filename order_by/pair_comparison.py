import json
from pydantic import BaseModel

class Step(BaseModel):
    explanation: str

class ComparisonReasoning(BaseModel):
    steps: list[Step]
    key: str

class ExternalComparisonReasoning(BaseModel):
    steps: list[Step]
    sorted_list: list[str]

class Pair_Comparison_Key:
    def __init__(self, key):
        self.key = key
        self.datatype = type(key)
    
    def get_greater(self, client, prompt, modelname, possibles):
        api_call = 0 
        while api_call < 3:
            api_call += 1
            # response = client.chat.completions.create(
            response = client.beta.chat.completions.parse(
                model=modelname,
                messages=[
                    {"role": "system", "content": "You are a helpful agent. Help the user compare the given two keys step by step."},
                    {"role": "user", "content": prompt}],
                n = 1,
                temperature=0.0,
                max_tokens=4096,
                response_format=ComparisonReasoning
            )
            try:
                response = [choice.message.content.strip() for choice in response.choices][0]
                # print(response)
                json_data = json.loads(response)
                if json_data['key'] in possibles or  self.datatype(json_data['key']) in possibles:
                    return  self.datatype(json_data['key']), api_call
                else:
                    print("output is not contained in [key1, key2]; try again\n")
            except Exception as e:
                print(f"[ERROR] Attempt {api_call}: {e}")
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
        

def external_comparisons(data, client, prompt_template, modelname):
    if len(data) == 0:
        return
    datatype = type(data[0])
    api_call = 0 
    prompt = prompt_template.format(keys = str(data))
    best_effort = None
    
    while api_call < 3:
        api_call += 1
        try:
            # Make the API call with "type": "json_object"
            # response = client.chat.completions.create(
            response = client.beta.chat.completions.parse(
                model=modelname,
                messages=[
                    {"role": "system", "content": "You are a helpful sorting agent. Help the user sort the input list step by step."},
                    {"role": "user", "content": prompt}],
                n=1,
                temperature=0.0,
                max_tokens=4096,
                response_format=ExternalComparisonReasoning
            )
            response = [choice.message.content.strip() for choice in response.choices][0]
            # print(response)
            json_data = json.loads(response)  # Safely decode JSON response
            assert len(json_data.keys()) == 2, print(json_data)
            val = json_data['sorted_list']
            if len(val) == len(data):
                return [datatype(v) for v in val], api_call
            else:
                print(f'ISSUE: not the same length as input; try again\n')
                continue
        except json.JSONDecodeError as jde:
            print(f"[ERROR] Attempt {api_call}: Failed to decode JSON: {jde}")
        except Exception as e:
            print(f"[ERROR] Attempt {api_call}: {e}")
    if best_effort and len(best_effort) == len(data):
        return best_effort, api_call
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

    keys = [Pair_Comparison_Key(10), Pair_Comparison_Key(15), Pair_Comparison_Key(1)]
    sorted_keys = sorted(keys, key=cmp_to_key(comparator))

    print("Sorted Keys:", sorted_keys)
