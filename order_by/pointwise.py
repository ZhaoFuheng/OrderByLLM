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
    steps: list[Step]
    value: str

class ExternalPointwiseReasoning(BaseModel):
    steps: list[Step]
    values: list[str]

class Pointwise_Key:
    def __init__(self, key):
        self.key = key
    def value(self, client, prompt, modelname, output_type):
        api_calls = 0
        total_tokens = 0
        while api_calls < 3:
            api_calls += 1
            try:
                # response = client.chat.completions.create(
                response = client.beta.chat.completions.parse(
                    model=modelname,
                    messages=[
                        {"role": "system", "content": "You are a helpful agent. Help the user derive the value of the given key step by step."},
                        {"role": "user", "content": prompt}],
                    n = 1,
                    temperature=0.0,
                    max_tokens=4096,
                    response_format=PointwiseReasoning
                )
                response = [choice.message.content.strip() for choice in response.choices][0]
                total_tokens += count_tokens(response)
                # print(response)
                json_data = json.loads(response)
                return output_type(json_data['value']), api_calls, total_tokens
            except Exception as e:
                print(e)
        return None, api_calls, total_tokens 


def external_values(data, client, prompt_template, modelname, output_type):
    api_call = 0 
    total_tokens = 0
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
                    {"role": "system", "content": "You are a helpful agent. Help the user derive the values of the given keys step by step."},
                    {"role": "user", "content": prompt}],
                n=1,
                temperature=0.0,
                max_tokens=4096,
                response_format=ExternalPointwiseReasoning
            )
            response = [choice.message.content.strip() for choice in response.choices][0]
            total_tokens += count_tokens(response)
            # print(response)
            json_data = json.loads(response)  # Safely decode JSON response
            assert len(json_data.keys()) == 2, print(json_data)
            val = json_data['values']
            if len(val) == len(data):
                return [output_type(v) for v in val], api_call, total_tokens
            else:
                print(f'ISSUE: not the same length as input; try again\n')
                continue
        except json.JSONDecodeError as jde:
            print(f"[ERROR] Attempt {api_call}: Failed to decode JSON: {jde}")
        except Exception as e:
            print(f"[ERROR] Attempt {api_call}: {e}")
    if best_effort and len(best_effort) == len(data):
        return best_effort, api_call, total_tokens
    return data, api_call, total_tokens
