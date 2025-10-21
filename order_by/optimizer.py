from .sorting import *
from .utils import *
from collections import defaultdict
from .pointwise import *

import random
from typing import Any, List, Optional, Sequence
from pydantic import BaseModel
from diskcache import Cache
import asyncio
import math
import pytrec_eval
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cache = Cache(os.path.join(PROJECT_ROOT, 'sort_cache'))
# print(f"prompt cache at {os.path.join(PROJECT_ROOT, 'sort_cache')}")


class SeenStatus(BaseModel):
    haveSeenBefore: bool


class OrderByOptimizer:
    def __init__(
        self,
        client: Any,
        data: Sequence[str],
        membership_inference_prompt_template: str,
        pointwise_prompt_template: str,
        external_pointwise_prompt_template: str,
        pairwise_comparison_prompt_template: str,
        external_pairwise_prompt_template: str,
        dollar_budget_constraint: float,
        model_name: str,
        isPassage: bool,
        has_query_question = False
    ):
        self.client = client
        self.data = list(data)  # ensure indexable
        self.membership_inference_prompt = membership_inference_prompt_template
        self.pointwise_prompt_template = pointwise_prompt_template
        self.external_pointwise_prompt_template = external_pointwise_prompt_template
        self.pairwise_comparison_prompt_template = pairwise_comparison_prompt_template
        self.external_comparison_prompt_template = external_pairwise_prompt_template
        self.budget = dollar_budget_constraint
        self.model = model_name
        self.isPassage = isPassage
        self.has_query_question = has_query_question 

        # min max boundary
        if self.isPassage:
            if 'mini' in self.model:
                self.batch_space = [4, 6]
            else:
                self.batch_space = [4, 8]
        else:
            if 'mini' in self.model:
                self.batch_space = [4, 8]
            else:
                self.batch_space = [4, 16]


        self.membership_attack_schema = SeenStatus
        

    def _format_membership_prompt(self, example: str):
        """
        If your template includes '{example}', we substitute it.
        Otherwise we append the example at the end.
        """
        tpl = self.membership_inference_prompt
        return tpl.format(key=example)
    
    async def _call_llm(self, prompt, schema):
        key_hash = hash_prompt(prompt, self.model)
        if key_hash in cache:
            cached = cache[key_hash]
            parsed = cached['parsed']
            input_tokens = 0

            if 'input_tokens' in cached:
                input_tokens = cached['input_tokens']
            else:
                input_tokens = count_tokens(prompt)
            return parsed, 0, input_tokens, cached['tokens']-input_tokens

        if 'gpt-5' in self.model:
            response = await resolve(self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": "You are a helpful agent. Think step by step. Output a JSON object."},
                    {"role": "user", "content": prompt}],
                text={
                    "verbosity": "low"
                },
                text_format=schema,
            ))
            parsed = response.output_parsed
        else:
            response = await resolve(self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": "You are a helpful agent. Think step by step. Output a JSON object."},
                    {"role": "user", "content": prompt}],
                temperature=0.0,
                text_format=schema,
            ))
            parsed = response.output[0].content[0].parsed
        
        cache[key_hash] = {
            'parsed': parsed.dict(),
            'tokens': response.usage.total_tokens,
            'input_tokens': response.usage.input_tokens,
            'output_tokens':response.usage.output_tokens  
        }
        return parsed.dict(), 1, response.usage.input_tokens, response.usage.output_tokens

    async def process_membership_item(self, item):
        if not self.has_query_question:
            if len(item) == 2:
                item = item[1]
            assert isinstance(item, str) and not item.isdigit(), f"Invalid item: {item}"
            prompt = self._format_membership_prompt(item)
        else:
            prompt = self._format_membership_prompt(str(item))
        
        parsed, api_calls, input_tokens, output_tokens = await self._call_llm(prompt, self.membership_attack_schema)
        return parsed, input_tokens, output_tokens


    async def is_training_data(self, sample_size=5, seed=1):
        assert sample_size < len(self.data), print("sample size too large")
        total_input_tokens = 0
        total_output_tokens = 0
        membership_count = 0
        random.seed(seed)
        sampled_data = random.sample(self.data, min(sample_size, len(self.data)))

        tasks = [asyncio.create_task(self.process_membership_item(item)) for item in sampled_data]
        results = await asyncio.gather(*tasks)

        membership_count = 0
        total_input_tokens = 0
        total_output_tokens = 0

        for parsed, input_tokens, output_tokens in results:
            if parsed['haveSeenBefore']:
                membership_count += 1
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
        return membership_count > sample_size//2, total_input_tokens, total_output_tokens
    
    def estimated_total_price(self, alg_name, curr_price, sample_size):
        curr_price = float(curr_price)
        total_size = len(self.data)
        if alg_name == 'point' or alg_name == 'ext_point':
            sample_ratio = 1.0 * sample_size / total_size
            ans = curr_price / sample_ratio
        elif 'quick' == alg_name:
            # n log n
            unit_price = curr_price / (sample_size * math.log2(sample_size))
            ans = unit_price * total_size * math.log2(total_size)
        elif 'quick_3' == alg_name:
            unit_price = curr_price / (sample_size * math.log2(sample_size)) # when sample size is small, there is not much voting to be done.
            ans = unit_price * total_size * math.log2(total_size) * 1.3
        elif 'ext_bubble' in alg_name:
            batch_size = int(alg_name.split('_')[-1])
            unit_price = curr_price / ( 2 * (sample_size**2) / (batch_size**2) + (sample_size) / (batch_size))
            ans = unit_price * ( 2 * (total_size**2) / (batch_size**2) + (total_size) / (batch_size))
        elif 'merge' in alg_name:
            batch_size = int(alg_name.split('_')[-1])
            if sample_size == batch_size:
                unit_price = curr_price
            else:   
                unit_price = curr_price / ( (sample_size/batch_size) +  sample_size/batch_size * math.log2(sample_size/batch_size) )
            ans = unit_price * ( (total_size/batch_size) + total_size/batch_size * math.log2(total_size/batch_size) )
        print(alg_name, 'estimated cost', ans)
        return ans

    def determine_best_ranking_order(self, arr, policy='broda'):
        # arr is extracted = [(r[0], tokens2price(self.model, r[2], r[3]), name) for r, name in zip(results, names)]
        if 'broda':
            rankings = []
            if 'mini' in self.model:
                good_algs = ['point', 'quick', 'ext_bubble_4', 'ext_merge_4']
            else:
                good_algs = ['point', 'quick_3', 'ext_bubble_4', 'ext_merge_4']

            for sorted_data, curr_price, alg_name in arr:
                if alg_name in good_algs:
                    ranking = []
                    for data in sorted_data:
                        if type(data) == tuple:
                            ranking.append(data[0])
                        else:
                            ranking.append(data)
                    rankings.append(ranking[:])

            def borda(rankings):
                scores = defaultdict(int)
                for ranking in rankings:
                    n = len(ranking)
                    for i, item in enumerate(ranking):
                        scores[item] += n - i
                return sorted(scores, key=scores.get, reverse=True)
            return borda(rankings)
        elif 'ext_bubble_4':
            # treat the ext_bubble_4 which cost the most as the truth
            for sorted_data, curr_price, alg_name in arr:
                if alg_name == 'ext_bubble_4':
                     return sorted_data
        else:
            print('?? change a policy')
            assert False

    
    async def physical_order_by_impl(self, seed=2):
        total_input_tokens, total_output_tokens = 0, 0 

        # membership, input_tokens, output_tokens = await self.is_training_data()
        # total_input_tokens += input_tokens
        # total_output_tokens += output_tokens

        membership = False

        if membership:
            print('Membership confirmed!')
            if self.isPassage:
                data_points, scores, num_api_calls, in_tokens, out_tokens, confidences, texts =\
                    await external_pointwise_sort(self.data[:], external_values, self.client, self.external_pointwise_prompt_template,\
                                                self.model, float, 0.1, isPassage=self.isPassage)
                sorted_pairs = sorted(zip(scores, data_points, texts), key=lambda x: (x[0], -len(x[1])))
                sorted_scores, sorted_data, sorted_texts = zip(*sorted_pairs)
                total_input_tokens += in_tokens
                total_output_tokens += out_tokens
                ans = []
                for data, text in zip(sorted_data,sorted_texts):
                    ans.append((data, text))
                return ans, num_api_calls, total_input_tokens, total_output_tokens
            else:
                sorted_data, num_api_calls, in_tokens, out_tokens, confidences =\
                    await external_pointwise_sort(self.data[:], external_values, self.client, self.external_pointwise_prompt_template,\
                                                  self.model, float, 0.1)
                total_input_tokens += in_tokens
                total_output_tokens += out_tokens
                return sorted_data, num_api_calls, total_input_tokens, total_output_tokens
        else:
            random.seed(seed)
            sample_size = int(max(0.15*len(self.data), 20))
            if sample_size >= len(self.data):
                sorted_data, num_api_calls, in_tokens, out_tokens =\
                    await external_bubble_sort(self.data, external_comparisons, 4, self.client, self.external_comparison_prompt_template, self.model, self.isPassage)
                total_input_tokens += in_tokens
                total_output_tokens += out_tokens
                return sorted_data, num_api_calls, total_input_tokens, total_output_tokens

            sampled_data = random.sample(self.data, sample_size)

            # let determine the batch size for ext merge and ext bubble
            bubble_batch = -1
            for b in range(self.batch_space[0], self.batch_space[1]+1, 2):
                _, _, in_tokens, out_tokens = await self.map_algname_2_alg(sampled_data[:b], f'ext_bubble_{b}')
                total_input_tokens += in_tokens
                total_output_tokens += out_tokens
                est_price = self.estimated_total_price(f'ext_bubble_{b}', tokens2price(self.model, in_tokens, out_tokens), b)
                if est_price <= self.budget:
                    bubble_batch = b
                    break
            merge_batch = -1
            for b in range(self.batch_space[0], self.batch_space[1]+1, 2):
                _, _, in_tokens, out_tokens = await self.map_algname_2_alg(sampled_data[:b], f'ext_merge_{b}')
                total_input_tokens += in_tokens
                total_output_tokens += out_tokens
                est_price = self.estimated_total_price(f'ext_merge_{b}', tokens2price(self.model, in_tokens, out_tokens), b)
                if est_price <= self.budget:
                    merge_batch = b
                    break

            results, names = await self.invoke_all_on_samples(sampled_data, merge_batch, bubble_batch)

            for r in results:
                assert len(r) == 4, print(r)
                total_input_tokens += r[2]
                total_output_tokens += r[3]

            extracted = [(r[0], tokens2price(self.model, r[2], r[3]), name) for r, name in zip(results, names)]

            gold_sorted_data = self.determine_best_ranking_order(extracted)
            assert len(gold_sorted_data) == len(sampled_data), print("check determine best ranking order alg", gold_sorted_data)

            candidates = []
            print('as the best', gold_sorted_data[:10])

            for sorted_data, curr_price, alg_name in extracted:
                # print(alg_name, sorted_data[:10])
                est_price = self.estimated_total_price(alg_name, curr_price, sample_size)
                if est_price < self.budget:
                    gold_ids = gold_sorted_data
                    pred = sorted_data
                    if type(gold_sorted_data[0]) == tuple:
                        # use the id only
                        gold_ids = [ id for id, _ in gold_sorted_data]
                    if type(sorted_data[0]) == tuple:
                        pred = [ id for id, _ in sorted_data]

                    if not self.isPassage:
                        quality = kendalltau_distance(gold_ids, pred)
                    else:
                        gold = {
                            'Q1': {doc_id: (len(gold_ids) - i) for i, doc_id in enumerate(gold_ids)}
                        }
                        # e.g., {'Q1': {'D1': 5, 'D2': 4, 'D3': 3, ...}}

                        evaluator = pytrec_eval.RelevanceEvaluator(gold, {'ndcg_cut.10'})

                        # pred: your system’s ranked list of doc_ids for Q1, best first
                        run = {
                            'Q1': {doc_id: (len(pred) - i) for i, doc_id in enumerate(pred)}
                        }
                        metrics = evaluator.evaluate(run)
                        quality  = sum(m['ndcg_cut_10'] for m in metrics.values()) / len(metrics)
                    candidates.append((quality, alg_name))
            best_candidate = max(candidates, key=lambda x: x[0]) if candidates else None
            assert best_candidate != None
            _, best_alg = best_candidate
            print('------------------')
            print(f'decided to use {best_alg}')
            print('------------------')
            return await self.map_algname_2_alg(self.data[:], best_alg)



    async def map_algname_2_alg(self, arr, alg_name):
        input_key_class = Pointwise_Key
        if self.isPassage:
            input_key_class = PointwiseRelevanceKey

        if alg_name == 'quick_3':
            sorted_data, num_api_calls, in_tokens, out_tokens =\
                await quick_sort(arr, self.client, self.pairwise_comparison_prompt_template, self.model, self.isPassage, 3)
        elif alg_name == 'quick':
            sorted_data, num_api_calls, in_tokens, out_tokens =\
                await quick_sort(arr, self.client, self.pairwise_comparison_prompt_template, self.model, self.isPassage, 1)
        elif 'ext_merge' in alg_name:
            batch_size = int(alg_name.split('_')[-1])
            sorted_data, num_api_calls, in_tokens, out_tokens =\
                await external_merge_sort(arr, external_comparisons, batch_size, self.client, self.external_comparison_prompt_template,\
                                          self.model, self.isPassage)
        elif 'ext_bubble' in alg_name:
            batch_size = int(alg_name.split('_')[-1])
            sorted_data, num_api_calls, in_tokens, out_tokens =\
                await external_bubble_sort(arr, external_comparisons, batch_size, self.client, self.external_comparison_prompt_template,\
                                           self.model, self.isPassage)
        elif 'ext_point' in alg_name:
            if self.isPassage:
                data_points, scores, num_api_calls, in_tokens, out_tokens, confidences, texts =\
                    await external_pointwise_sort(arr, external_values, self.client, self.external_pointwise_prompt_template,\
                                                self.model, float, 0.1, isPassage=self.isPassage)
                sorted_pairs = sorted(zip(scores, data_points), key=lambda x: (x[0], -len(x[1])))
                sorted_scores, sorted_data = zip(*sorted_pairs)
            else:
                sorted_data, num_api_calls, in_tokens, out_tokens, confidences =\
                    await external_pointwise_sort(arr, external_values, self.client, self.external_pointwise_prompt_template,\
                                                  self.model, str, 0.1)
        elif alg_name == "point":
            if self.isPassage:
                data_points, scores, num_api_calls, in_tokens, out_tokens, _ =\
                    await pointwise_sort(arr, self.client, self.pointwise_prompt_template,
                                            self.model, float, input_key_class, self.isPassage)
                sorted_pairs = sorted(zip(scores, data_points), key=lambda x: (x[0], -len(x[1])))
                sorted_scores, sorted_data = zip(*sorted_pairs)
            else:
                sorted_data, num_api_calls, in_tokens, out_tokens, _ =\
                    await pointwise_sort(arr, self.client, self.pointwise_prompt_template,
                                            self.model, float, input_key_class, self.isPassage)
        else:
            print("error!!!")
            print(f"what is {alg_name} referring to?")
            assert False

        return list(sorted_data), num_api_calls, in_tokens, out_tokens



    async def invoke_all_on_samples(self, samples, merge_batch, bubble_batch):
        # run all concurrently

        names = ["point", "ext_point", "quick"]
        tasks = [
            asyncio.create_task(self.map_algname_2_alg(samples[:], "point"),
                                    ),
            asyncio.create_task(self.map_algname_2_alg(samples[:], "ext_point"),
                                    ),
            asyncio.create_task(self.map_algname_2_alg(samples[:], "quick"))
        ]
        if "mini" not in self.model:
            tasks.append(
                    asyncio.create_task(self.map_algname_2_alg(samples[:], f"quick_3"))
            )
            names.append(f'quick_3')

        if merge_batch > 2:
            tasks.append(
                    asyncio.create_task(self.map_algname_2_alg(samples[:], f"ext_merge_{merge_batch}"))
            )
            names.append(f'ext_merge_{merge_batch}')
            if merge_batch!=4:
                tasks.append(
                    asyncio.create_task(self.map_algname_2_alg(samples[:], f"ext_merge_4"))
                )
                names.append(f'ext_merge_4')
        else:
            tasks.append(
                asyncio.create_task(self.map_algname_2_alg(samples[:], f"ext_merge_4"))
            )
            names.append(f'ext_merge_4')


        if bubble_batch > 2:
            tasks.append(
                asyncio.create_task(self.map_algname_2_alg(samples[:], f"ext_bubble_{bubble_batch}"))
            )
            names.append(f'ext_bubble_{bubble_batch}')

            if bubble_batch != 4:
                tasks.append(
                    asyncio.create_task(self.map_algname_2_alg(samples[:], f"ext_bubble_{4}"))
                )
                names.append(f'ext_bubble_{4}')

        else:
            tasks.append(
                asyncio.create_task(self.map_algname_2_alg(samples[:], f"ext_bubble_{4}"))
            )
            names.append(f'ext_bubble_{4}')
        # alway include ext_bubble_4 on samples since we are not sure which is more expensive between quick_3 and ext_bubble_4
        # gather results (keeps order)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert len(results) == len(names)
        return results, names