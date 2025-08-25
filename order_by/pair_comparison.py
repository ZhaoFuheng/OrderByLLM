import json
from pydantic import BaseModel
from .utils import count_tokens, hash_prompt, resolve, create_numbered_passages
from diskcache import Cache
import json
from typing import List, Callable, Dict, Tuple
import asyncio
import numpy as np

np.random.seed(0)

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


class PassageComparisonReasoning(BaseModel):
    steps: list[Step]
    BetterPassageKey: str

class PassageExternalComparisonReasoning(BaseModel):
    steps: list[Step]
    WorstToBestPassageKeys: list[int]


class Pair_Comparison_Key:
    def __init__(self, key, schema):
        if schema == ComparisonReasoning:
            self.key = key
            self.datatype = type(key)
        elif schema == PassageComparisonReasoning:
            self.key = key[1]
            self.datatype = str
            self.docid = key[0]
        self.schema=schema
    
    async def get_greater(self, client, prompt, modelname, possibles):
        key_hash = hash_prompt(prompt, modelname)

        if key_hash in cache:
            cached = cache[key_hash]
            try:
                parsed = self.schema(**cached['parsed'])
                if self.schema == ComparisonReasoning and parsed.key in possibles:
                    return self.datatype(parsed.key), 0, cached['tokens']
                elif self.schema == PassageComparisonReasoning and parsed.BetterPassageKey in ['A', 'B']:
                    return parsed.BetterPassageKey, 0, cached['tokens']
            except Exception:
                print("check cache")
                pass 
        
        api_call = 0 
        total_tokens = 0
        while api_call < 3:
            api_call += 1
            # response = client.chat.completions.create(
            try:
                if 'gpt-5' in modelname:
                    response = await resolve( client.responses.parse(
                        model=modelname,
                        input=[
                            {"role": "system", "content": "You are a helpful agent. Think step by step. Output a JSON object."},
                            {"role": "user", "content": prompt}],
                        text={
                            "verbosity": "low"
                        },
                        text_format=self.schema,
                    ))
                    parsed = response.output_parsed
                else:
                    response = await resolve( client.responses.parse(
                        model=modelname,
                        input=[
                            {"role": "system", "content": "You are a helpful agent. Think step by step. Output a JSON object."},
                            {"role": "user", "content": prompt}],
                        temperature=0.0,
                        text_format=self.schema,
                    ))
                    parsed = response.output[0].content[0].parsed
                total_tokens += response.usage.total_tokens

                cache[key_hash] = {
                    'parsed': parsed.dict(),
                    'tokens': total_tokens}
                
                if self.schema == ComparisonReasoning and parsed.key in possibles:
                    return  self.datatype(parsed.key), api_call, total_tokens
                elif self.schema == PassageComparisonReasoning and parsed.BetterPassageKey in ['A', 'B']:
                    return parsed.BetterPassageKey, api_call, total_tokens
                else:
                    print(self.schema)
                    print(f"output:{parsed} is not contained in [key1, key2]; try again\n")
            except Exception as e:
                print(f"[ERROR] Attempt {api_call}: {e}")
        # return a random item
        if self.schema == ComparisonReasoning:
            return possibles.pop(), api_call, total_tokens
        return 'A', api_call, total_tokens

    async def compare(self, other, client, prompt_template, modelname):
        if self.key == other.key:
            return 0, 0, 0
        possibles = {str(self.key), str(other.key)}
        prompt = prompt_template.format(key1=str(self.key), key2=str(other.key))
        greater_key, num, tokens = await self.get_greater(client, prompt, modelname, possibles)

        if greater_key == self.key or greater_key == str(self.key) or greater_key == 'A':
            return 1, num, tokens
        return -1, num, tokens

    
    def __repr__(self):
        return str(self.key)


def create_comparator(client, prompt_template, modelname):
    def comparator(key1, key2):
        return key1.compare(key2, client, prompt_template, modelname)[0]
    return comparator
        

async def external_comparisons(data, client, prompt_template, modelname, isPassage):
    if len(data) == 0:
        return data, 0, 0
    
    api_call = 0 
    total_tokens = 0
    best_effort = None
    key_to_passage = {} # used to map key to passage
    index_to_key = {}

    if isPassage:
        assert len(data[0]) == 2, print(data[0])
        schema = PassageExternalComparisonReasoning
        datatype = str
        for index, (k, text) in enumerate(data):
            index_to_key[index+1] = k
        for key, text in data:
            key_to_passage[key] = text

        data = [(index+1, text) for index, (id, text) in enumerate(data)]
        prompt = prompt_template.format(keys = create_numbered_passages(data, True))

    else:
        schema = ExternalComparisonReasoning
        datatype = type(data[0])
        prompt = prompt_template.format(keys = str(data))

    key_hash = hash_prompt(prompt, modelname)
    if key_hash in cache:
        cached = cache[key_hash]
        parsed = schema(**cached['parsed'])
        if isPassage:
            vals = parsed.WorstToBestPassageKeys
            vals = [index_to_key[v] for v in vals]
            assert type(vals[0]) == str
            all_keys_presented = all(k in key_to_passage.keys() for k in vals)
            if len(vals) == len(data) and all_keys_presented:
                return [(k, key_to_passage[k]) for k in vals], 0, cached['tokens']
        else:
            vals = parsed.sorted_list
            if len(vals) == len(data):
                return [datatype(v) for v in vals], 0, cached['tokens']
        print("cache contain incomplete information")
        return data, 0, cached['tokens']
    
    while api_call < 5:
        api_call += 1
        prefix = ''
        if api_call > 1:
            prefix = f"Make sure the output list have length {len(data)}\n"
        try:
            # Make the API call with "type": "json_object"
            # response = client.chat.completions.create(
            if "gpt-5" in modelname:
                response = await resolve( client.responses.parse(
                    model=modelname,
                    input=[
                        {"role": "system", "content": "You are a helpful agent. Think step by step. Output a JSON object."},
                        {"role": "user", "content": prefix + prompt}],
                    text={
                        "verbosity": "low"
                    },
                    text_format=schema,
                ))
                parsed = response.output_parsed
            else:
                response = await resolve( client.responses.parse(
                    model=modelname,
                    input=[
                        {"role": "system", "content": "You are a helpful agent. Think step by step. Output a JSON object."},
                        {"role": "user", "content": prefix + prompt}],
                    temperature=0.0,
                    text_format=schema
                ))
                parsed = response.output[0].content[0].parsed
            total_tokens += response.usage.total_tokens
            if isPassage:
                vals = parsed.WorstToBestPassageKeys
                vals = [index_to_key[v] for v in vals]
                assert type(vals[0]) == str
                all_keys_presented = all(k in key_to_passage.keys() for k in vals)
                if not all_keys_presented:
                    print("not all keys are presented in the return list")
                    print(vals)
                    print(key_to_passage)
                if len(vals) != len(data):
                    print(f'api call: {api_call}: ISSUE: not the same length as input; try again\n')
                if len(vals) == len(data) and all_keys_presented:
                    cache[key_hash] = {
                        'parsed': parsed.dict(),
                        'tokens': total_tokens
                    }
                    return [(k, key_to_passage[k]) for k in vals], 0, total_tokens
            else:
                vals = parsed.sorted_list
                if len(vals) == len(data):
                    return [datatype(v) for v in vals], api_call, total_tokens
                else:
                    print(f'api call: {api_call}: ISSUE: not the same length as input; try again\n')
        except json.JSONDecodeError as jde:
            print(f"[ERROR] Attempt {api_call}: Failed to decode JSON: {jde}")
        except Exception as e:
            print(f"[ERROR] Attempt {api_call}: {e}")
    return data, api_call, total_tokens
