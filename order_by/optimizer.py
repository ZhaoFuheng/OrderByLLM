from zipfile import LargeZipFile
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
from enum import Enum

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cache = Cache(os.path.join(PROJECT_ROOT, 'sort_cache'))
# print(f"prompt cache at {os.path.join(PROJECT_ROOT, 'sort_cache')}")

class status(str, Enum):
    Yes = "Yes"
    No = "No"

class SeenStatus(BaseModel):
    Explanation: str
    haveSeenBefore: status

class LLMJudgeResult(BaseModel):
    explanations: list[str]
    id: int

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
        llm_judge_prompt_template: str,
        sample_size: float = 20,
        has_id_and_row: bool = False,
        proxy_ground_truth_policy = 'borda',
        ideal_oracle = None,
        k: int = None,
        judge_model = None,
        good_algs = None,
        bm25_passage = None,
        isReview = False,
        run_all = False,
    ):
        assert proxy_ground_truth_policy in ['borda', 'llm_judge', 'ext_bubble_4', 'ideal', 'bt'], print(f"proxy_ground_truth_policy must be one of ['borda', 'llm_judge', 'ext_bubble_4', 'ideal_oracle'], but got {proxy_ground_truth_policy}")
        self.ideal_oracle = ideal_oracle
        self.llm_judge_prompt_template = llm_judge_prompt_template
        self.proxy_ground_truth_policy = proxy_ground_truth_policy
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
        self.has_id_and_row  = has_id_and_row
        self.sample_size = sample_size
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.invoked_algs = {}
        self.isReview = isReview
        self.run_all = run_all
        if self.isPassage:
            assert k is not None, print(f'k must be provided for passage ranking')
            self.k = k
            if sample_size <= self.k:
                self.k = int(self.k//2)
            assert sample_size > self.k, print(f'sample size {sample_size} must be greater than k//2 {self.k}')
        else:
            self.k = k
            if self.k is None:
                self.k = len(self.data) 
                
        if judge_model:
            self.judge_model = judge_model
        else:
            self.judge_model = self.model

        # min max boundary
        self.batch_space = [4, 16]
        self.alg_cost_est = {}

        self.membership_attack_schema = SeenStatus
        self.llm_judge_schema = LLMJudgeResult

        self.good_algs = ['quick', 'quick_3', 'ext_merge_4', 'ext_bubble_4']
        if good_algs:
            self.good_algs = good_algs

        self.bm25_passage = bm25_passage
        

    def _format_membership_prompt(self, example: str):
        """
        If your template includes '{example}', we substitute it.
        Otherwise we append the example at the end.
        """
        tpl = self.membership_inference_prompt
        return tpl.format(key=example)
    
    async def _call_llm(self, prompt, schema, llm_judge=False):
        if llm_judge:
            model = self.judge_model
        else:
            model = self.model

        key_hash = hash_prompt(prompt, model)
        if key_hash in cache:
            try:
                cached = cache[key_hash]
                parsed = schema(**cached['parsed'])
                if 'input_tokens' in cached:
                    input_tokens = cached['input_tokens']
                else:
                    input_tokens = count_tokens(prompt)
                return parsed.dict(), 0, input_tokens, cached['tokens']-input_tokens
            except Exception:
                print('need to delete this cached entry!')
                del cache[key_hash]

        i = 0
        while i < 10:
            i += 1
            try:
                if type(self.client) == SnowflakeClient:
                    response = await resolve(self.client.responses(
                        model=model,
                        prompt=prompt,
                        schema=schema,
                    ))
                    parsed = response.output_parsed 
                elif 'snowflake' in str(self.client.base_url):
                    response = await resolve(self.client.beta.chat.completions.parse(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a helpful agent. Think step by step. Output a JSON object."},
                                {"role": "user", "content": prompt}],
                            temperature=0.0,
                            response_format=schema,
                            max_completion_tokens = 20*1024,
                        ))
                    parsed = response.choices[0].message.parsed
                else:
                    if 'gpt-5' in self.model:
                        response = await resolve(self.client.responses.parse(
                            model=model,
                            input=[
                                {"role": "system", "content": "You are a helpful agent. Think step by step. Output a JSON object."},
                                {"role": "user", "content": prompt}],
                            text={
                                "verbosity": "low"
                            },
                            text_format=schema,
                            max_completion_tokens = 10*1024,
                        ))
                        parsed = response.output_parsed
                    else:
                        response = await resolve(self.client.responses.parse(
                            model=model,
                            input=[
                                {"role": "system", "content": "You are a helpful agent. Think step by step. Output a JSON object."},
                                {"role": "user", "content": prompt}],
                            temperature=0.0,
                            text_format=schema,
                            max_completion_tokens = 10*1024,
                        ))
                        parsed = response.output[0].content[0].parsed
            except Exception as e:
                print(e)
                if '429' in str(e):
                    print('rate limit exceeded, waiting for 30 second')
                    await asyncio.sleep(30)
                continue
        
        input_tokens = (
                        getattr(response.usage, "input_tokens", None)
                        or getattr(response.usage, "prompt_tokens", None)
                    )

        cache[key_hash] = {
            'parsed': parsed.dict(),
            'tokens': response.usage.total_tokens,
            'input_tokens': input_tokens,
            'output_tokens': response.usage.total_tokens - input_tokens
        }
        return parsed.dict(), 1, input_tokens, response.usage.total_tokens - input_tokens

    async def process_membership_item(self, item):
        if not self.has_id_and_row:
            if len(item) == 2:
                item = item[1]
            assert isinstance(item, str) and not item.isdigit(), f"Invalid item: {item}"
            prompt = self._format_membership_prompt(item)
        else:
            assert len(item) == 2, print(f'item must be a tuple of (id, row), but got {item}')
            item = item[1]
            prompt = self._format_membership_prompt(str(item))
        parsed, api_calls, input_tokens, output_tokens = await self._call_llm(prompt, self.membership_attack_schema)
        # print('--------------------------------')
        # print('membership inference prompt', prompt)
        # print(parsed)
        # print('--------------------------------')
        return parsed, input_tokens, output_tokens


    async def process_llm_judge_item(self, prompt):
        parsed, api_calls, input_tokens, output_tokens = await self._call_llm(prompt, self.llm_judge_schema, llm_judge=True)
        print(f'llm judge result: {parsed}')
        return parsed['id'], input_tokens, output_tokens

    async def is_training_data(self, sample_size=20, seed=1):
        assert sample_size < len(self.data), print("sample size too large")
        total_input_tokens = 0
        total_output_tokens = 0
        membership_count = 0
        random.seed(seed)
        if self.bm25_passage:
            sampled_data = self.bm25_passage[-sample_size:]
        else:
            sampled_data = random.sample(self.data, min(sample_size, len(self.data)))

        tasks = [asyncio.create_task(self.process_membership_item(item)) for item in sampled_data]
        results = await asyncio.gather(*tasks)

        membership_count = 0
        total_input_tokens = 0
        total_output_tokens = 0

        for parsed, input_tokens, output_tokens in results:
            # print(f'parsed: {parsed}')
            assert parsed['haveSeenBefore'] == 'Yes' or parsed['haveSeenBefore'] == 'No', print(f'haveSeenBefore must be Yes or No, but got {parsed["haveSeenBefore"]}')
            if parsed['haveSeenBefore'] == 'Yes':
                membership_count += 1
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
        print('membership count:', membership_count)
        # return membership_count > sample_size//2, total_input_tokens, total_output_tokens
        return membership_count >= 0.99 * sample_size, total_input_tokens, total_output_tokens

    
    def estimated_total_price(self, alg_name, curr_price, sample_size):
        curr_price = float(curr_price)
        total_size = len(self.data)
        if alg_name == 'point' or alg_name == 'ext_point':
            sample_ratio = 1.0 * sample_size / total_size
            ans = curr_price / sample_ratio
        elif 'quick' == alg_name or 'quick_3' == alg_name:
            v = 1
            if 'quick_3' == alg_name:
                v = 3
            # n log n
            if not self.isPassage and not self.isReview:
                unit_price = curr_price / (sample_size * (math.log2(sample_size)-1) )
                ans = unit_price * total_size * math.log2(total_size)
            else:
                # assume LIMIT 10
                unit_price = curr_price / (sample_size + self.k * math.log2(self.k))
                ans =  unit_price * (total_size  + self.k * math.log2(self.k))

        elif 'ext_bubble' in alg_name:
            batch_size = int(alg_name.split('_')[-1])
            print(f'ext_bubble batch_size: {batch_size}')
            if not self.isPassage and not self.isReview:
                unit_price = curr_price / ( (sample_size**2) / (batch_size**2))
                ans = unit_price * ( (total_size**2) / (batch_size**2))
            else:
                if sample_size == batch_size:
                    unit_price = curr_price
                else:
                    unit_price = curr_price / ( self.k * sample_size / (batch_size**2))
                ans = unit_price * ( self.k * (total_size) / (batch_size**2))
        elif 'merge' in alg_name:
            batch_size = int(alg_name.split('_')[-1])
            if not self.isPassage and not self.isReview:
                unit_price = curr_price / ( (sample_size/batch_size) +  sample_size/batch_size * math.log2(sample_size/batch_size) )
                ans = unit_price * ( (total_size/batch_size) + total_size/batch_size * math.log2(total_size/batch_size) )
            else:
                if sample_size == batch_size:
                    unit_price = curr_price
                else:
                    unit_price = curr_price / ( (sample_size / batch_size) * (2 + math.log2(self.k/batch_size)) )
                ans = unit_price * ( (total_size / batch_size) * (2 + math.log2(self.k/batch_size)) )
        # print(alg_name, 'estimated cost', ans)
        self.alg_cost_est[alg_name] = ans
        return ans

    async def determine_best_ranking_order(self, arr, sampled_data=None):
        # arr is extracted = [(r[0], tokens2price(self.model, r[2], r[3]), name) for r, name in zip(results, names)]
        if self.proxy_ground_truth_policy == 'borda' or self.proxy_ground_truth_policy == 'bt':
            rankings = []
            ranking_algs = []
            
            each_rank_length = len(arr[0][0])
            for sorted_data, curr_price, alg_name in arr:
                assert len(sorted_data) == each_rank_length, print(f'{alg_name} sorted data length {len(sorted_data)} is not equal to each rank length {each_rank_length}')
                if alg_name in self.good_algs or 'ext_bubble' in alg_name or 'ext_merge' in alg_name:
                    ranking = []
                    for data in sorted_data:
                        if type(data) == tuple:
                            ranking.append(data[0])
                        else:
                            ranking.append(data)
                    rankings.append(ranking[-self.k:])
                    ranking_algs.append(alg_name)
            if self.proxy_ground_truth_policy == 'bt':
                return bradley_terry(rankings, self.k)
            return borda(rankings, self.k)
        elif self.proxy_ground_truth_policy == 'llm_judge':
            rankings = []
            candidate_algs = []
            for sorted_data, est_price, alg_name in arr:
                # print(f'sorted data: {sorted_data}')
                sorted_ids = []
                for data in sorted_data:
                    if type(data) == tuple:
                        sorted_ids.append(data[0])
                    else:
                        sorted_ids.append(data)
                rankings.append(sorted_ids[-self.k:])
                candidate_algs.append(alg_name)

            if type(sampled_data[0]) == tuple:
                sampled_data_new_ids = []
                map_old_to_new = {}
                for i, sample_tuple in zip(range(len(sampled_data)), sampled_data):
                    docid, text = sample_tuple
                    sampled_data_new_ids.append(('text_id: ' + str(i+1), text))
                    map_old_to_new[docid] = 'text_id: ' + str(i+1)
                new_rankings = []
                for ranking in rankings:
                    new_ranking = []
                    for docid in ranking:
                        new_ranking.append(map_old_to_new[docid])
                    new_rankings.append(new_ranking)
                rankings = new_rankings[:]
            else:
                sampled_data_new_ids = sampled_data

            i = 0
            judge_suffix = ''
            while i < 7:
                i += 1
                
                prompt = self.llm_judge_prompt_template.format(rankings=create_numbered_rankings(rankings), keys=sampled_data_new_ids)
                
                # print('llm judge prompt', prompt)
                
                alg_index, input_tokens, output_tokens = await self.process_llm_judge_item(prompt + judge_suffix)
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                if alg_index>=0 and alg_index<=len(candidate_algs):
                    break
                else:
                    print(f'index out of range, try again {alg_index-1}, candidate algs: {candidate_algs}')
                    judge_suffix = f'Respond in JSON. The output JSON id field must be in range [1, {len(candidate_algs)}], representing each candidate.'
            return candidate_algs[alg_index-1]
        elif self.proxy_ground_truth_policy == 'ideal':
            if not self.isPassage and not self.isReview:
                assert type(self.ideal_oracle) == list, print(f'ideal oracle must be a list, but got {type(self.ideal_oracle)}')
                ideal_sorted_sample_data = []
                for item in self.ideal_oracle:
                    if item in sampled_data:
                        ideal_sorted_sample_data.append(item)
                best_quality = float('-inf')
                best_alg = None
                for sorted_data, curr_price, alg_name in arr:
                    gold_ids = ideal_sorted_sample_data[:]
                    pred = sorted_data[:]
                    if type(gold_ids[0]) == tuple:
                        gold_ids = [ id for id, _ in gold_ids]
                    if type(pred[0]) == tuple:
                        pred = [ infos[0] for infos in pred]
                    quality = kendalltau_distance(gold_ids, pred)
                    if quality > best_quality:
                        best_quality = quality
                        best_alg = alg_name
                return best_alg
            else:
                assert type(self.ideal_oracle) == dict, print(f'ideal oracle must be a dict, but got {type(self.ideal_oracle)}')
                gold = {
                    'Q1': {str(doc_id): int(score) for doc_id, score in self.ideal_oracle.items()}
                }

                evaluator = pytrec_eval.RelevanceEvaluator(gold, {f'ndcg_cut.{self.k}'})
                best_quality = float('-inf')
                best_alg = None
                for sorted_data, curr_price, alg_name in arr:
                    pred = sorted_data[:]
                    assert len(pred) == self.k, print(f'pred length {len(pred)} is not equal to k {self.k}')
                    if type(pred[0]) == tuple:
                        pred = [ infos[0] for infos in pred]
                    run = {
                            'Q1': { str(doc_id) : int(i+1) for i, doc_id in enumerate(pred)}
                        }
                    metrics = evaluator.evaluate(run)
                    quality  = sum(m[f'ndcg_cut_{self.k}'] for m in metrics.values()) / len(metrics)
                    if quality > best_quality:
                        best_quality = quality
                        best_alg = alg_name
                return best_alg
        else:
            print('?? change a policy')
            assert False
    
    async def physical_order_by_impl(self, seed=2):
        if self.proxy_ground_truth_policy == 'borda' or self.proxy_ground_truth_policy == 'bt' or self.proxy_ground_truth_policy == 'llm_judge':
            membership, input_tokens, output_tokens = await self.is_training_data()
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
        else:
            membership = False

        if membership and not self.run_all:
            print('Membership confirmed!')
            return await self.map_algname_2_alg(self.data[:], 'point', final_decision=True), 'point'
        else:
            print('Not in training data!')

        print('membership inference cost', tokens2price(self.model, self.total_input_tokens, self.total_output_tokens))
        self.budget = self.budget - tokens2price(self.model, self.total_input_tokens, self.total_output_tokens)
        print('remaining budget', self.budget)



        random.seed(seed)
        # sample_size = int(self.sample_ratio*len(self.data))
        sample_size = int(self.sample_size)
        assert sample_size <= len(self.data), print(f'sample size {sample_size} is greater than the data size {len(self.data)}')
        start_index = random.randint(0, len(self.data) - sample_size)
        if start_index + sample_size > len(self.data):
            start_index = len(self.data) - sample_size

        sampled_data = self.data[start_index:start_index+sample_size][:]


        results, names = await self.invoke_all_on_samples(sampled_data[:], None, None)

        invoked_budget = 0
        for r, n in zip(results, names):
            assert len(r) == 4, print(r, n)
            self.total_input_tokens += r[2]
            self.total_output_tokens += r[3]
            invoked_budget += tokens2price(self.model, r[2], r[3])
        print('invoked budget', invoked_budget)
        self.budget = self.budget - invoked_budget
        print('remaining budget after initial invocation', self.budget)

        extracted = [(r[0], tokens2price(self.model, r[2], r[3]), name) for r, name in zip(results, names)]


        # let determine the batch size for ext merge and ext bubble
        bubble_batch = -1
        for b in range(self.batch_space[0], self.batch_space[1]+1, 2):
            _, _, in_tokens, out_tokens = await self.map_algname_2_alg(sampled_data[:b], f'ext_bubble_{b}')
            self.total_input_tokens += in_tokens
            self.total_output_tokens += out_tokens
            est_price = self.estimated_total_price(f'ext_bubble_{b}', tokens2price(self.model, in_tokens, out_tokens), b)
            if est_price <= self.budget:
                bubble_batch = b
                break

        merge_batch = -1
        for b in range(self.batch_space[0], self.batch_space[1]+1, 2):
            _, _, in_tokens, out_tokens = await self.map_algname_2_alg(sampled_data[:b], f'ext_merge_{b}')
            self.total_input_tokens += in_tokens
            self.total_output_tokens += out_tokens
            est_price = self.estimated_total_price(f'ext_merge_{b}', tokens2price(self.model, in_tokens, out_tokens), b)
            if est_price <= self.budget:
                merge_batch = b
                break

        results, names = await self.invoke_all_on_samples(sampled_data[:], merge_batch, bubble_batch)
        additional_invoked_budget = 0
        for r, n in zip(results, names):
            assert len(r) == 4, print(r, n)
            self.total_input_tokens += r[2]
            self.total_output_tokens += r[3]
            additional_invoked_budget += tokens2price(self.model, r[2], r[3])
        print('additional_invoked_budget', additional_invoked_budget)
        extracted += [(r[0], tokens2price(self.model, r[2], r[3]), name) for r, name in zip(results, names)]

        for sorted_data, curr_price, alg_name in extracted:
            print(alg_name, curr_price)

        if self.proxy_ground_truth_policy == 'llm_judge':
            filtered_extracted = []
            for sorted_data, curr_price, alg_name in extracted:
                est_price = self.estimated_total_price(alg_name, curr_price, sample_size)
                if est_price < self.budget:
                    filtered_extracted.append((sorted_data, est_price, alg_name))
            best_alg = await self.determine_best_ranking_order(filtered_extracted, sampled_data[:])
            print('------------------')
            print(f'llm judge result: {best_alg}')
            print('------------------')
            return await self.map_algname_2_alg(self.data[:], best_alg, final_decision=True), best_alg


        elif self.proxy_ground_truth_policy == 'ideal':
            filtered_extracted = []
            for sorted_data, curr_price, alg_name in extracted:
                est_price = self.estimated_total_price(alg_name, curr_price, sample_size)
                if est_price < self.budget:
                    filtered_extracted.append((sorted_data, curr_price, alg_name))
            best_alg = await self.determine_best_ranking_order(extracted, sampled_data[:])
            print('------------------')
            print(f'ideal oracle result: {best_alg}')
            print('------------------')
            return await self.map_algname_2_alg(self.data[:], best_alg, final_decision=True), best_alg

        gold_sorted_data = await self.determine_best_ranking_order(extracted)
        assert len(gold_sorted_data) == len(sampled_data) or len(gold_sorted_data) == self.k, print("check determine best ranking order alg", gold_sorted_data, sampled_data)

        candidates = []

        for sorted_data, curr_price, alg_name in extracted:
            # print(alg_name, sorted_data[:10])
            est_price = self.estimated_total_price(alg_name, curr_price, sample_size)
            # print(f'alg_name: {alg_name}, est_price: {est_price}, budget: {self.budget}')
            if est_price < self.budget:
                gold_ids = gold_sorted_data
                pred = sorted_data
                if type(gold_sorted_data[0]) == tuple:
                    # use the id only
                    gold_ids = [ infos[0] for infos in gold_sorted_data]
                if type(sorted_data[0]) == tuple:
                    pred = [ infos[0] for infos in sorted_data]

                if not self.isPassage and not self.isReview:
                    quality = kendalltau_distance(gold_ids, pred)
                else:
                    gold = {
                        'Q1': {

                            str(doc_id): int(i+1)
                            for i, doc_id in enumerate(gold_ids)
                        }
                    }

                    evaluator = pytrec_eval.RelevanceEvaluator(gold, {f'ndcg_cut.{self.k}'})

                    run = {
                        'Q1': { str(doc_id) : int(i+1)  for i, doc_id in enumerate(pred)}
                    }
                    metrics = evaluator.evaluate(run)
                    quality  = sum(metric[f'ndcg_cut_{self.k}'] for metric in metrics.values()) / len(metrics)
                candidates.append((quality, est_price, alg_name))

        assert len(candidates) > 0, print(f'no candidates found')
        best_candidate = max(candidates, key=lambda x: (x[0], x[1])) if candidates else None
        assert best_candidate != None
        _, _, best_alg = best_candidate
        print('------------------')
        print(f'decided to use {best_alg}')
        print('------------------')
        final_ans = await self.map_algname_2_alg(self.data[:], best_alg, final_decision=True)
        return final_ans, best_alg



    async def map_algname_2_alg(self, arr, alg_name, final_decision=False):
        input_key_class = Pointwise_Key
        if self.isPassage or self.isReview:
            input_key_class = PointwiseRelevanceKey

        in_tokens = 0
        out_tokens = 0

        if alg_name == 'quick_3':
            sorted_data, num_api_calls, in_tokens, out_tokens =\
                await quick_sort(arr, self.client, self.pairwise_comparison_prompt_template, self.model, self.isPassage, 3, isReview=self.isReview, limit_k=self.k)
        elif alg_name == 'quick':
            sorted_data, num_api_calls, in_tokens, out_tokens =\
                await quick_sort(arr, self.client, self.pairwise_comparison_prompt_template, self.model, self.isPassage, 1, isReview=self.isReview, limit_k=self.k)
        elif 'ext_merge' in alg_name:
            batch_size = int(alg_name.split('_')[-1])
            sorted_data, num_api_calls, in_tokens, out_tokens =\
                await external_merge_sort(arr, external_comparisons, batch_size, self.client, self.external_comparison_prompt_template,\
                                          self.model, self.isPassage, isReview=self.isReview, limit_k=self.k)
        elif 'ext_bubble' in alg_name:
            batch_size = int(alg_name.split('_')[-1])
            sorted_data, num_api_calls, in_tokens, out_tokens =\
                await external_bubble_sort(arr, external_comparisons, batch_size, self.client, self.external_comparison_prompt_template,\
                                           self.model, self.isPassage, isReview=self.isReview, limit_k=self.k)
        elif 'ext_point' in alg_name:
            if self.isPassage or self.isReview:
                data_points, scores, num_api_calls, in_tokens, out_tokens, texts =\
                    await external_pointwise_sort(arr, external_values, self.client, self.external_pointwise_prompt_template,\
                                                self.model, float, 0.1, isPassage=self.isPassage, isReview=self.isReview, batch_size_data=self.data[:])
                sorted_pairs = sorted(zip(scores, data_points), key=lambda x: (x[0], -len(x[1])))
                sorted_scores, sorted_data = zip(*sorted_pairs)
                sorted_data = [(data, score, 'score') for data, score in zip(sorted_data, sorted_scores)]
            else:
                sorted_data, num_api_calls, in_tokens, out_tokens =\
                    await external_pointwise_sort(arr, external_values, self.client, self.external_pointwise_prompt_template,\
                                                  self.model, str, 0.1, batch_size_data=self.data[:])
        elif alg_name == "point":
            if self.isPassage or self.isReview:
                data_points, scores, num_api_calls, in_tokens, out_tokens =\
                    await pointwise_sort(arr, self.client, self.pointwise_prompt_template,
                                            self.model, float, input_key_class, self.isPassage, isReview=self.isReview)
                sorted_pairs = sorted(zip(scores, data_points), key=lambda x: (x[0], -len(x[1])))
                sorted_scores, sorted_data = zip(*sorted_pairs)
                sorted_data = [(data, score, 'score') for data, score in zip(sorted_data, sorted_scores)]
            else:
                sorted_data, num_api_calls, in_tokens, out_tokens =\
                    await pointwise_sort(arr, self.client, self.pointwise_prompt_template,
                                            self.model, float, input_key_class, self.isPassage, isReview=self.isReview)
        else:
            print("error!!!")
            print(f"what is {alg_name} referring to?")
            assert False
        if final_decision:
            print('running final decision algorithm', alg_name, 'cost', tokens2price(self.model, in_tokens, out_tokens))
            in_tokens += self.total_input_tokens
            out_tokens += self.total_output_tokens
            print('alg name', alg_name, 'real cost', tokens2price(self.model, in_tokens, out_tokens), 'alg cost est', self.alg_cost_est, 'budget', self.budget)
        else:
            if self.k:
                sorted_data = sorted_data[-self.k:]
                assert len(sorted_data) <= self.k, print(f'{alg_name} sorted data length {len(sorted_data)} is greater than k {self.k}')
        return list(sorted_data), num_api_calls, in_tokens, out_tokens



    async def invoke_all_on_samples(self, samples, merge_batch, bubble_batch):
        # run all concurrently
        tasks = []
        names = []
        for name in ["quick_3", 'quick']:
            if name not in self.invoked_algs:
                tasks.append(
                    asyncio.create_task(self.map_algname_2_alg(samples[:], name))
                )
                names.append(name)
                self.invoked_algs[name] = True

        if merge_batch is not None and merge_batch > 2:
            if f"ext_merge_{merge_batch}" not in self.invoked_algs:
                tasks.append(
                        asyncio.create_task(self.map_algname_2_alg(samples[:], f"ext_merge_{merge_batch}"))
                )
                names.append(f'ext_merge_{merge_batch}')
                self.invoked_algs[f"ext_merge_{merge_batch}"] = True



        if bubble_batch is not None and bubble_batch > 2:
            if f"ext_bubble_{bubble_batch}" not in self.invoked_algs:
                tasks.append(
                    asyncio.create_task(self.map_algname_2_alg(samples[:], f"ext_bubble_{bubble_batch}"))
                )
                names.append(f'ext_bubble_{bubble_batch}')
                self.invoked_algs[f"ext_bubble_{bubble_batch}"] = True


        for name in ['ext_point', 'point']:
            if name not in self.invoked_algs:
                tasks.append(
                    asyncio.create_task(self.map_algname_2_alg(samples[:], name))
                )
                names.append(name)
                self.invoked_algs[name] = True

        # alway include ext_bubble_4 on samples since we are not sure which is more expensive between quick_3 and ext_bubble_4
        # gather results (keeps order)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert len(results) == len(names)
        return results, names