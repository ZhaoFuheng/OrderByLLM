import json
from pydantic import BaseModel
from typing import Generic, TypeVar
# from pydantic.generics import GenericModel
from .utils import count_tokens

T = TypeVar("T")

class Step(BaseModel):
    explanation: str

# class PointwiseReasoning(GenericModel, Generic[T]):
#     steps: list[Step]
#     value: T

# class ExternalPointwiseReasoning(GenericModel, Generic[T]):
#     steps: list[Step]
#     values: list[T]

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

class Pointwise_Key:
    def __init__(self, key):
        self.key = key
    async def value(self, client, prompt, modelname, output_type):
        api_calls = 0
        total_tokens = 0
        while api_calls < 3:
            api_calls += 1
            try:
                # response = client.chat.completions.create(
                #response = client.beta.chat.completions.parse(
                response = client.responses.parse(
                    model=modelname,
                    input=[
                        {"role": "system", "content": "You are a helpful agent. Think step by step. Output a JSON object."},
                        {"role": "user", "content": prompt}],
                    temperature=0.0,
                    text_format=PointwiseReasoning
                )
                # print(response)
                #response = [choice.message.content.strip() for choice in response.choices][0]
                parsed = response.output[0].content[0].parsed
                total_tokens += response.usage.total_tokens
                return output_type(parsed.value), api_calls, total_tokens, parsed.confidence
            except Exception as e:
                print(e)
        return None, api_calls, total_tokens, 0


async def external_values(data, client, prompt_template, modelname, output_type):
    api_call = 0 
    total_tokens = 0
    prompt = prompt_template.format(keys = str(data))
    best_effort = None
    while api_call < 3:
        api_call += 1
        try:
            # Make the API call with "type": "json_object"
            # response = client.chat.completions.create(
            # response = client.beta.chat.completions.parse(
            response = client.responses.parse(
                model=modelname,
                input=[
                    {"role": "system", "content": "You are a helpful agent. Think step by step. Output a JSON object."},
                    {"role": "user", "content": prompt}],
                temperature=0.0,
                text_format=ExternalPointwiseReasoning
            )
            parsed = response.output[0].content[0].parsed
            total_tokens += response.usage.total_tokens
            # print(response)
            vals = parsed.values
            best_effort = vals
            if len(vals) == len(data):
                return [output_type(v) for v in vals], api_call, total_tokens, parsed.confidences
            else:
                print(f'ISSUE: not the same length as input; try again\n')
                continue
        except Exception as e:
            print(f"[ERROR] Attempt {api_call}: {e}")
    if best_effort and len(best_effort) == len(data):
        return best_effort, api_call, total_tokens
    return data, api_call, total_tokens, 0
