import json
from pydantic import BaseModel
from .utils import count_tokens, hash_prompt
from diskcache import Cache
import json

cache = Cache('./sort_cache')

class Step(BaseModel):
    explanation: str

class ComparisonReasoning(BaseModel):
    steps: list[Step]
    key: str

class ExternalComparisonReasoning(BaseModel):
    keys: list[str]
    steps: list[Step]
    sorted_list: list[str]

class Pair_Comparison_Key:
    def __init__(self, key):
        self.key = key
        self.datatype = type(key)
    
    async def get_greater(self, client, prompt, modelname, possibles):
        key_hash = hash_prompt(prompt, modelname)

        if key_hash in cache:
            cached = cache[key_hash]
            parsed = ComparisonReasoning(**cached['parsed'])
            if parsed.key in possibles:
                return self.datatype(parsed.key), 0, cached['tokens']
            
        api_call = 0 
        total_tokens = 0
        while api_call < 3:
            api_call += 1
            # response = client.chat.completions.create(
            try:
                response = client.responses.parse(
                    model=modelname,
                    input=[
                        {"role": "system", "content": "You are a helpful agent. Think step by step. Output a JSON object."},
                        {"role": "user", "content": prompt}],
                    temperature=0.0,
                    text_format=ComparisonReasoning
                )
                parsed = response.output[0].content[0].parsed
                total_tokens += response.usage.total_tokens

                cache[key_hash] = {
                    'parsed': parsed.dict(),
                    'tokens': total_tokens}

                if parsed.key in possibles:
                    return  self.datatype(parsed.key), api_call, total_tokens
                else:
                    print(f"output:{parsed} is not contained in [key1, key2]; try again\n")
            except Exception as e:
                print(f"[ERROR] Attempt {api_call}: {e}")
        # return a random item
        return possibles.pop(), api_call, total_tokens

    async def compare(self, other, client, prompt_template, modelname):
        if self.key == other.key:
            return 0, 0, 0
        possibles = {str(self.key), str(other.key)}
        prompt = prompt_template.format(key1=str(self.key), key2=str(other.key))
        greater_key, num, tokens = await self.get_greater(client, prompt, modelname, possibles)

        if greater_key == self.key or greater_key == str(self.key):
            return 1, num, tokens
        return -1, num, tokens

    
    def __repr__(self):
        return str(self.key)


def create_comparator(client, prompt_template, modelname):
    def comparator(key1, key2):
        return key1.compare(key2, client, prompt_template, modelname)[0]
    return comparator
        

async def external_comparisons(data, client, prompt_template, modelname):
    if len(data) == 0:
        return data, 0, 0
    
    datatype = type(data[0])
    api_call = 0 
    total_tokens = 0
    prompt = prompt_template.format(keys = str(data))
    best_effort = None

    key_hash = hash_prompt(prompt, modelname)
    if key_hash in cache:
        cached = cache[key_hash]
        parsed = ExternalComparisonReasoning(**cached['parsed'])
        vals = parsed.sorted_list
        if len(vals) == len(data):
            return [datatype(v) for v in vals], 0, cached['tokens']
    
    while api_call < 3:
        api_call += 1
        try:
            # Make the API call with "type": "json_object"
            # response = client.chat.completions.create(
            response = client.responses.parse(
                model=modelname,
                input=[
                    {"role": "system", "content": "You are a helpful agent. Think step by step. Output a JSON object."},
                    {"role": "user", "content": prompt}],
                temperature=0.0,
                text_format=ExternalComparisonReasoning
            )
            parsed = response.output[0].content[0].parsed
            total_tokens += response.usage.total_tokens
            vals = parsed.sorted_list
            cache[key_hash] = {
                'parsed': parsed.dict(),
                'tokens': total_tokens}
            if len(vals) == len(data):
                return [datatype(v) for v in vals], api_call, total_tokens
            else:
                print(f'ISSUE: not the same length as input; try again\n')
                continue
        except json.JSONDecodeError as jde:
            print(f"[ERROR] Attempt {api_call}: Failed to decode JSON: {jde}")
        except Exception as e:
            print(f"[ERROR] Attempt {api_call}: {e}")
    return data, api_call, total_tokens
