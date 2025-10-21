import json
from pydantic import BaseModel
from typing import Generic, TypeVar
# from pydantic.generics import GenericModel
from .utils import count_tokens, hash_prompt, is_async_client, resolve, create_numbered_passages, create_numbered_SQLs
from diskcache import Cache
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cache = Cache(os.path.join(PROJECT_ROOT, 'sort_cache'))
# print(f"prompt cache at {os.path.join(PROJECT_ROOT, 'sort_cache')}")


T = TypeVar("T")

class Step(BaseModel):
    explanation: str

class PointwiseReasoning(BaseModel):
    key: str
    steps: list[Step]
    value: float
    confidence: int

class ExternalPointwiseReasoning(BaseModel):
    keys: list[str]
    steps: list[Step]
    values: list[float]
    confidences: list[int]


class PassagePointwiseReasoning(BaseModel):
    steps: list[Step]
    relevance_score: float
    confidence: int

class PassageExternalPointwiseReasoning(BaseModel):
    steps: list[Step]
    relevance_scores: list[float]
    confidences: list[int]

class SQLPointwiseReasoning(BaseModel):
    steps: list[Step]
    correctness_score: float
    confidence: int

class SQLExternalPointwiseReasoning(BaseModel):
    steps: list[Step]
    correctness_scores: list[float]
    confidences: list[int]

class Pointwise_Key:
    def __init__(self, key, prompt_template, schema=PointwiseReasoning):
        self.key = key
        self.prompt_template = prompt_template
        self.schema = schema
    async def value(self, client, modelname, output_type):
        api_calls = 0
        total_tokens = 0
        prompt = self.prompt_template.format(key=str(self.key))
        key_hash = hash_prompt(prompt, modelname)

        if key_hash in cache:
            cached = cache[key_hash]
            parsed = cached['parsed']
            # val, api calls, input tokens, output tokens, confidenecs
            input_tokens = 0
            if 'input_tokens' in cached:
                input_tokens = cached['input_tokens']
            else:
                input_tokens = count_tokens(prompt)
            return output_type(parsed['value']), 0, input_tokens, cached['tokens']-input_tokens, parsed['confidence']

        while api_calls < 3:
            api_calls += 1
            try:
                # response = client.chat.completions.create(
                #response = client.beta.chat.completions.parse(
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
                #response = [choice.message.content.strip() for choice in response.choices][0]                total_tokens += response.usage.total_tokens
                cache[key_hash] = {
                    'parsed': parsed.dict(),
                    'tokens': response.usage.total_tokens,
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens':response.usage.output_tokens  
                    }
                return output_type(parsed.value), api_calls, response.usage.input_tokens * api_calls, total_tokens - (response.usage.input_tokens * api_calls), parsed.confidence
            except Exception as e:
                print(e)
        return None, api_calls, 0, 0, 0

class PointwiseRelevanceKey:
    """ Used for relevance tasks, does not generate a confidence score.
    """
    def __init__(self, key, prompt_template, isPassage=True):
        self.key = key
        self.prompt_template = prompt_template
        if isPassage:
            self.schema = PassagePointwiseReasoning
        else:
            self.schema = SQLPointwiseReasoning

    async def value(self, client, modelname, output_type):
        api_calls = 0
        total_tokens = 0
        prompt = self.prompt_template.format(key=self.key)
        key_hash = hash_prompt(prompt, modelname)
        # print(prompt)

        if key_hash in cache:
            cached = cache[key_hash]
            parsed = cached['parsed']
            input_tokens = 0
            if 'input_tokens' in cached:
                input_tokens = cached['input_tokens']
            else:
                input_tokens = count_tokens(prompt)
            if self.schema == PassagePointwiseReasoning:
                return output_type(parsed['relevance_score']), 0, input_tokens, cached['tokens']-input_tokens, parsed['confidence']
            elif self.schema == SQLPointwiseReasoning:
                return output_type(parsed['correctness_score']), 0, input_tokens, cached['tokens']-input_tokens, parsed['confidence']

        while api_calls < 3:
            api_calls += 1
            try:
                if "gpt-5" in modelname:
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
                    'tokens': response.usage.total_tokens,
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens':response.usage.output_tokens  
                    }
                if self.schema == PassagePointwiseReasoning:
                    return output_type(parsed.relevance_score), api_calls, response.usage.input_tokens * api_calls, total_tokens - (response.usage.input_tokens * api_calls), int(parsed.confidence)
                elif self.schema == SQLPointwiseReasoning:
                    return output_type(parsed.correctness_score), api_calls, response.usage.input_tokens * api_calls, total_tokens - (response.usage.input_tokens * api_calls), int(parsed.confidence)
            except Exception as e:
                print(f'exception={e}')
        return 0, api_calls, 0, total_tokens, 0


async def external_values(data, client, prompt_template, modelname, output_type, schema = ExternalPointwiseReasoning):
    api_call = 0 
    total_tokens = 0
    if schema == PassageExternalPointwiseReasoning:
        prompt = prompt_template.format(keys = str(create_numbered_passages(data)))
    elif schema == SQLExternalPointwiseReasoning:
         prompt = prompt_template.format(keys = str(create_numbered_SQLs(data)))
    else:
        prompt = prompt_template.format(keys = str(data))
    best_effort = None
    key_hash = hash_prompt(prompt, modelname)

    if key_hash in cache:
        cached = cache[key_hash]
        parsed = cached['parsed']
        if schema == ExternalPointwiseReasoning:
            vals = parsed['values']
        elif schema == PassageExternalPointwiseReasoning:
            vals = parsed['relevance_scores']
        elif schema == SQLExternalPointwiseReasoning:
            vals = parsed['correctness_scores']

        input_tokens = 0
        if 'input_tokens' in cached:
            input_tokens = cached['input_tokens']
        else:
            input_tokens = count_tokens(prompt)
        if len(vals) == len(data):
            return [output_type(v) for v in vals], 0, input_tokens, cached['tokens'] - input_tokens, parsed['confidences']
        else:
            print('need to delete this cached entry!')
            del cache[key_hash]

    while api_call < 3:
        api_call += 1
        try:
            # Make the API call with "type": "json_object"
            # response = client.chat.completions.create(
            # response = client.beta.chat.completions.parse(
            if "gpt-5" in modelname:
                response = await resolve( client.responses.parse(
                    model=modelname,
                    input=[
                        {"role": "system", "content": "You are a helpful agent. Think step by step. Output a JSON object."},
                        {"role": "user", "content": prompt}
                    ],
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
                        {"role": "user", "content": prompt}],
                    temperature=0.0,
                    text_format=schema,
                ))
                parsed = response.output[0].content[0].parsed
            total_tokens += response.usage.total_tokens
            if schema == ExternalPointwiseReasoning:
                vals = parsed.values
            elif schema == PassageExternalPointwiseReasoning:
                vals = parsed.relevance_scores
            elif schema == SQLExternalPointwiseReasoning:
                vals = parsed.correctness_scores
            best_effort = vals

            cache[key_hash] = {
                'parsed': parsed.dict(),
                'tokens': response.usage.total_tokens,
                'input_tokens': response.usage.input_tokens,
                'output_tokens':response.usage.output_tokens  
            }
            if len(vals) == len(data):
                return [output_type(v) for v in vals], api_call, response.usage.input_tokens * api_call, total_tokens - (response.usage.input_tokens * api_call), parsed.confidences
            else:
                print(f'api call: {api_call}: ISSUE: not the same length as input; try again\n')
                print(f"output is of length {len(vals)}, but should be {len(data)}")
                continue
        except Exception as e:
            print(f"[ERROR] Attempt {api_call}: {e}")
    if best_effort and len(best_effort) == len(data):
        print("Best Effort Due to Error")
        return best_effort, api_call, 0, 0, total_tokens
    return [], api_call, total_tokens, 0, 0
