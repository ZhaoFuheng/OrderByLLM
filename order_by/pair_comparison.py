import json
from pydantic import BaseModel
from .utils import count_tokens, hash_prompt, resolve, create_numbered_passages, create_numbered_SQLs
from diskcache import Cache
import json
from typing import List, Callable, Dict, Tuple
import asyncio
import numpy as np
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cache = Cache(os.path.join(PROJECT_ROOT, 'sort_cache'))
# print(f"prompt cache at {os.path.join(PROJECT_ROOT, 'sort_cache')}")

np.random.seed(0)

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

class SQLComparisonReasoning(BaseModel):
    steps: list[Step]
    BetterKey: str

class SQLExternalComparisonReasoning(BaseModel):
    steps: list[Step]
    WorstToBestSQLKeys: list[int]


class Pair_Comparison_Key:
    def __init__(self, key, schema):
        if schema == ComparisonReasoning:
            self.key = key
            self.datatype = type(key)
        elif schema == PassageComparisonReasoning:
            self.key = key[1]
            self.datatype = str
            self.docid = key[0]
        elif schema == SQLComparisonReasoning:
            self.key = key[1]
            self.datatype = str
            self.sqlid = key[0]
        self.schema=schema
    
    async def get_greater(self, client, prompt, modelname, possibles):
        # print(possibles)
        key_hash = hash_prompt(prompt, modelname)

        if key_hash in cache:
            cached = cache[key_hash]
            input_tokens = 0
            if 'input_tokens' in cached:
                input_tokens = cached['input_tokens']
            else:
                input_tokens = count_tokens(prompt)
            try:
                parsed = self.schema(**cached['parsed'])
                if self.schema == ComparisonReasoning and parsed.key in possibles:
                    return self.datatype(parsed.key), 0, input_tokens, cached['tokens']-input_tokens
                if  self.schema == ComparisonReasoning and parsed.key not in possibles:
                    print('----------------------')
                    print(parsed.key)
                    print(possibles)
                    print('why is cached key not in possibles????')
                    print('----------------------')
                if self.schema == PassageComparisonReasoning and parsed.BetterPassageKey in ['A', 'B']:
                    return parsed.BetterPassageKey, 0, input_tokens, cached['tokens']-input_tokens
                if self.schema == SQLComparisonReasoning and parsed.BetterKey in ['A', 'B']:
                    print(prompt)
                    print()
                    print(parsed)
                    print('---------------')
                    return parsed.BetterKey, 0, input_tokens, cached['tokens']-input_tokens
                
            except Exception:
                print('need to delete this cached entry!\nCompute again!')
                del cache[key_hash]
        api_call = 0 
        total_tokens = 0
        while api_call < 3:
            api_call += 1
            # response = client.chat.completions.create(
            suffix = ''
            if self.schema == SQLComparisonReasoning and api_call > 1:
                suffix = '\nReturn either A or B in Betterkey.'
            try:
                if 'gpt-5' in modelname:
                    response = await resolve( client.responses.parse(
                        model=modelname,
                        input=[
                            {"role": "system", "content": "You are a helpful agent. Think step by step. Output a JSON object."},
                            {"role": "user", "content": prompt + suffix}],
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
                            {"role": "user", "content": prompt + suffix}],
                        temperature=0.0,
                        text_format=self.schema,
                    ))
                    parsed = response.output[0].content[0].parsed
                total_tokens += response.usage.total_tokens

                if self.schema == ComparisonReasoning and parsed.key in possibles:
                    cache[key_hash] = {
                        'parsed': parsed.dict(),
                        'tokens': response.usage.total_tokens,
                        'input_tokens': response.usage.input_tokens,
                        'output_tokens':response.usage.output_tokens  
                    }
                    return  self.datatype(parsed.key), api_call, response.usage.input_tokens * api_call, total_tokens - (response.usage.input_tokens * api_call)
                elif self.schema == PassageComparisonReasoning and parsed.BetterPassageKey in ['A', 'B']:
                    cache[key_hash] = {
                        'parsed': parsed.dict(),
                        'tokens': response.usage.total_tokens,
                        'input_tokens': response.usage.input_tokens,
                        'output_tokens':response.usage.output_tokens  
                    }
                    return parsed.BetterPassageKey, api_call, response.usage.input_tokens * api_call, total_tokens - (response.usage.input_tokens * api_call)
                elif self.schema == SQLComparisonReasoning and parsed.BetterKey in ['A', 'B']:
                    cache[key_hash] = {
                        'parsed': parsed.dict(),
                        'tokens': response.usage.total_tokens,
                        'input_tokens': response.usage.input_tokens,
                        'output_tokens':response.usage.output_tokens  
                    }
                    return parsed.BetterKey, api_call, response.usage.input_tokens * api_call, total_tokens - (response.usage.input_tokens * api_call)
                else:
                    print(self.schema)
                    print(f"output:{parsed} is not contained in:\n{possibles};\ntry again\n")
            except Exception as e:
                print(f"[ERROR] Attempt {api_call}: {e}")
        # return a random item
        if self.schema == ComparisonReasoning:
            return possibles.pop(), api_call, 0, total_tokens
        return 'A', api_call, 0, total_tokens

    async def compare(self, other, client, prompt_template, modelname):
        if self.key == other.key:
            return 0, 0, 0, 0
        possibles = {str(self.key), str(other.key)}
        if self.schema == PassageComparisonReasoning or self.schema == SQLComparisonReasoning :
            possibles = {'A', 'B'}
        prompt = prompt_template.format(key1=str(self.key), key2=str(other.key))
        greater_key, num, intput_tokens, output_tokens = await self.get_greater(client, prompt, modelname, possibles)

        if greater_key == self.key or greater_key == str(self.key) or greater_key == 'A':
            return 1, num, intput_tokens, output_tokens
        return -1, num, intput_tokens, output_tokens

    
    def __repr__(self):
        return str(self.key)


def create_comparator(client, prompt_template, modelname):
    def comparator(key1, key2):
        return key1.compare(key2, client, prompt_template, modelname)[0]
    return comparator
        

async def external_comparisons(data, client, prompt_template, modelname, isPassage, isSQL=False):

    if len(data) == 0:
        return data, 0, 0, 0
    
    api_call = 0 
    total_tokens = 0
    best_effort = None
    key_to_txt = {} # used to map key to passage or sql
    index_to_key = {}
    all_keys = set(data)

    if isSQL:
        assert len(data[0]) == 2, print(data[0])
        schema = SQLExternalComparisonReasoning
        datatype = str
        for index, (k, sql) in enumerate(data):
            index_to_key[index+1] = k
        for key, sql in data:
            key_to_txt[key] = sql
        data = [(index+1, text) for index, (id, text) in enumerate(data)]
        prompt = prompt_template.format(keys = create_numbered_SQLs(data, True))
    elif isPassage:
        assert len(data[0]) == 2, print(data[0])
        schema = PassageExternalComparisonReasoning
        datatype = str
        for index, (k, text) in enumerate(data):
            index_to_key[index+1] = k
        for key, text in data:
            key_to_txt[key] = text

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

        input_tokens = 0
        if 'input_tokens' in cached:
            input_tokens = cached['input_tokens']
        else:
            input_tokens = count_tokens(prompt)

        if isPassage or isSQL:
            vals = []
            if isPassage:
                vals = parsed.WorstToBestPassageKeys
            else:
                vals =  parsed.WorstToBestSQLKeys
            vals = [index_to_key[v] for v in vals]
            assert type(vals[0]) == str
            all_keys_presented = all(k in key_to_txt.keys() for k in vals)
            if len(vals) == len(data) and all_keys_presented:
                return [(k, key_to_txt[k]) for k in vals], 0, input_tokens, cached['tokens']-input_tokens
            else:
                print('need to delete this cached entry!')
                del cache[key_hash]
        else:
            all_keys_presented = True
            vals = parsed.sorted_list
            for v in vals:
                if v not in all_keys:
                    all_keys_presented = False
                    break
            if len(vals) == len(data) and all_keys_presented:
                return [datatype(v) for v in vals], 0, input_tokens, cached['tokens']-input_tokens
            else:
                print('need to delete this cached entry!')
                del cache[key_hash]
    
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
            if isPassage or isSQL:
                vals = []
                if isPassage:
                    vals = parsed.WorstToBestPassageKeys
                else:
                    vals = parsed.WorstToBestSQLKeys
                vals = [index_to_key[v] for v in vals]
                assert type(vals[0]) == str
                all_keys_presented = all(k in key_to_txt.keys() for k in vals)
                if not all_keys_presented:
                    print("not all keys are presented in the return list")
                    print(vals)
                    print(key_to_txt)
                if len(vals) != len(data):
                    print(f'api call: {api_call}: ISSUE: not the same length as input; try again\n')
                if len(vals) == len(data) and all_keys_presented:
                    cache[key_hash] = {
                        'parsed': parsed.dict(),
                        'tokens': response.usage.total_tokens,
                        'input_tokens': response.usage.input_tokens,
                        'output_tokens':response.usage.output_tokens  
                    }
                    return [(k, key_to_txt[k]) for k in vals], 0, response.usage.input_tokens * api_call, total_tokens - (response.usage.input_tokens * api_call)
            else:
                vals = parsed.sorted_list
                all_keys_presented = True
                for v in vals:
                    if v not in all_keys:
                        all_keys_presented = False
                        break

                if len(vals) == len(data) and all_keys_presented:
                    cache[key_hash] = {
                        'parsed': parsed.dict(),
                        'tokens': response.usage.total_tokens,
                        'input_tokens': response.usage.input_tokens,
                        'output_tokens':response.usage.output_tokens  
                    }
                    return [datatype(v) for v in vals], api_call, response.usage.input_tokens * api_call, total_tokens - (response.usage.input_tokens * api_call)
                else:
                    print(f'api call: {api_call}: ISSUE: not the same length as input; try again\n')
        except json.JSONDecodeError as jde:
            print(f"[ERROR] Attempt {api_call}: Failed to decode JSON: {jde}")
        except Exception as e:
            print(f"[ERROR] Attempt {api_call}: {e}")
    return data, api_call, 0, total_tokens
