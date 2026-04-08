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
cache = Cache(os.path.join(PROJECT_ROOT, 'sort_cache'), size_limit=50 * 1024**3, eviction_policy='least-recently-used')
# print(f"prompt cache at {os.path.join(PROJECT_ROOT, 'sort_cache')}")

class status(str, Enum):
    Yes = "Yes"
    No = "No"

class SeenStatus(BaseModel):
    Explanation: str
    isFactualKnowledge: status
    webSearchQuery: str

class LLMJudgeResult(BaseModel):
    explanations: list[str]
    id: int

class OrderByOptimizer:
    def __init__(
        self,
        client: Any,
        data: Sequence[str],
        factual_knowledge_prompt_template: str,
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
        enable_factual_web_search = True,
        external_pointwise_memory_size = 8,
        wiki_field = None,
        use_simulation_estimate: bool = False,
    ):
        assert proxy_ground_truth_policy in ['borda', 'llm_judge', 'ext_bubble_4', 'ideal'], print(f"proxy_ground_truth_policy must be one of ['borda', 'llm_judge', 'ext_bubble_4', 'ideal_oracle'], but got {proxy_ground_truth_policy}")
        self.ideal_oracle = ideal_oracle
        self.llm_judge_prompt_template = llm_judge_prompt_template
        self.proxy_ground_truth_policy = proxy_ground_truth_policy
        self.client = client
        self.data = list(data)  # ensure indexable
        self.factual_knowledge_prompt = factual_knowledge_prompt_template
        self.pointwise_prompt_template = pointwise_prompt_template
        self.external_pointwise_prompt_template = external_pointwise_prompt_template
        self.pairwise_comparison_prompt_template = pairwise_comparison_prompt_template
        self.external_comparison_prompt_template = external_pairwise_prompt_template
        self.ranking_budget = dollar_budget_constraint
        self.model = model_name
        self.isPassage = isPassage
        self.has_id_and_row  = has_id_and_row
        self.sample_size = sample_size
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.invoked_algs = {}
        self.isReview = isReview
        self.run_all = run_all
        self.enable_factual_web_search = enable_factual_web_search
        self.external_pointwise_memory_size = external_pointwise_memory_size
        self.wiki_field = wiki_field
        self.optimization_budget = 0.0
        self.use_simulation_estimate = use_simulation_estimate
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
        self.batch_space = [4, 10]
        self.alg_cost_est = {}

        self.factual_knowledge_schema = SeenStatus
        self.llm_judge_schema = LLMJudgeResult

        self.good_algs = []
        if good_algs:
            self.good_algs = good_algs

        self.bm25_passage = bm25_passage
        self.sample_results: dict = {}  # {alg_name: (sorted_data, _, in_tokens, out_tokens)}
        

    def _format_factual_knowledge_prompt(self, example: str):
        """
        If your template includes '{example}', we substitute it.
        Otherwise we append the example at the end.
        """
        tpl = self.factual_knowledge_prompt
        return tpl.format_map(defaultdict(str, key=example))
    
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
                else:
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
                break
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

    async def process_factual_knowledge_item(self, item):
        if not self.has_id_and_row:
            if len(item) == 2:
                item = item[1]
            assert isinstance(item, str) and not item.isdigit(), f"Invalid item: {item}"
            prompt = self._format_factual_knowledge_prompt(item)
        else:
            assert len(item) == 2, print(f'item must be a tuple of (id, row), but got {item}')
            item = item[1]
            prompt = self._format_factual_knowledge_prompt(str(item))
        parsed, api_calls, input_tokens, output_tokens = await self._call_llm(prompt, self.factual_knowledge_schema)
        return parsed, input_tokens, output_tokens


    async def process_llm_judge_item(self, prompt):
        parsed, api_calls, input_tokens, output_tokens = await self._call_llm(prompt, self.llm_judge_schema, llm_judge=True)
        return parsed['id'], input_tokens, output_tokens

    async def is_factual_knowledge(self, sample_size=10, seed=1):
        assert sample_size < len(self.data), print("sample size too large")
        total_input_tokens = 0
        total_output_tokens = 0
        factual_count = 0
        random.seed(seed)
        if self.bm25_passage:
            sampled_data = self.bm25_passage[-sample_size:]
        else:
            sampled_data = random.sample(self.data, min(sample_size, len(self.data)))

        tasks = [asyncio.create_task(self.process_factual_knowledge_item(item)) for item in sampled_data]
        results = await asyncio.gather(*tasks)

        factual_count = 0
        total_input_tokens = 0
        total_output_tokens = 0
        query_counter: dict[str, int] = {}

        for parsed, input_tokens, output_tokens in results:
            assert parsed['isFactualKnowledge'] == 'Yes' or parsed['isFactualKnowledge'] == 'No', print(f'isFactualKnowledge must be Yes or No, but got {parsed["isFactualKnowledge"]}')
            if parsed['isFactualKnowledge'] == 'Yes':
                factual_count += 1
            q = parsed.get('webSearchQuery', '').strip()
            if q:
                query_counter[q] = query_counter.get(q, 0) + 1
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

        most_popular_query = max(query_counter, key=query_counter.get) if query_counter else ''
        return factual_count >= 0.9 * sample_size, total_input_tokens, total_output_tokens, most_popular_query

    def estimated_total_price(self, alg_name, curr_price, sample_size, actual_sample_api_calls=None):
        curr_price = float(curr_price)
        total_size = len(self.data)
        if alg_name == 'point' or 'ext_point' in alg_name:
            sample_ratio = 1.0 * sample_size / total_size
            ans = curr_price / sample_ratio
        elif 'quick' == alg_name or 'quick_3' == alg_name:
            v = 3 if 'quick_3' == alg_name else 1
            if not self.isPassage and not self.isReview:
                unit_price = curr_price / (v * sample_size * math.log2(max(sample_size, 2)))
                ans = unit_price * v * total_size * math.log2(total_size)
            elif self.use_simulation_estimate:
                s_calls = actual_sample_api_calls if actual_sample_api_calls else quick_sort_calls_sim(sample_size, v, min(self.k, sample_size))
                total_calls = quick_sort_calls_sim(total_size, v, self.k)
                ans = curr_price * total_calls / max(s_calls, 1)
            else:
                assert actual_sample_api_calls is not None, print(f'actual_sample_api_calls is not provided for {alg_name}')
                s_calls = actual_sample_api_calls
                expected_calls = quick_calls_formula(sample_size, v, min(self.k, sample_size))
                correction_factor = expected_calls / s_calls

                total_calls = quick_calls_formula(total_size, v, self.k)
                ans = curr_price * total_calls * correction_factor / max(s_calls, 1)
            ans *= 2 # avoid underestimation

        elif 'ext_bubble' in alg_name:
            batch_size = int(alg_name.split('_')[-1])
            if not self.isPassage and not self.isReview:
                unit_price = curr_price / ( (sample_size**2) / (batch_size**2))
                ans = unit_price * ( (total_size**2) / (batch_size**2))
            elif self.use_simulation_estimate:
                s_calls = actual_sample_api_calls
                total_calls = bubble_sort_calls_sim(total_size, batch_size, self.k)
                ans = curr_price * total_calls / max(s_calls, 1)
            else:
                assert actual_sample_api_calls is not None, print(f'actual_sample_api_calls is not provided for {alg_name}')
                s_calls = actual_sample_api_calls
                total_calls = bubble_calls_formula(total_size, batch_size, self.k)
                ans = curr_price * total_calls / max(s_calls, 1)
            ans *= 1.3 # avoid underestimation
        elif 'merge' in alg_name:
            batch_size = int(alg_name.split('_')[-1])
            if not self.isPassage and not self.isReview:
                unit_price = curr_price / ( (sample_size/batch_size) +  sample_size/batch_size * math.log2(sample_size/batch_size) )
                ans = unit_price * ( (total_size/batch_size) + total_size/batch_size * math.log2(total_size/batch_size) )
            elif self.use_simulation_estimate:
                s_calls = actual_sample_api_calls                
                total_calls = merge_sort_calls_sim(total_size, batch_size, self.k)
                ans = curr_price * total_calls / max(s_calls, 1)
            else:
                assert actual_sample_api_calls is not None, print(f'actual_sample_api_calls is not provided for {alg_name}')
                s_calls = actual_sample_api_calls
                total_calls = merge_calls_formula(total_size, batch_size, self.k)
                ans = curr_price * total_calls / max(s_calls, 1)
            ans *= 1.3 # avoid underestimation
        # print(alg_name, 'estimated cost', ans)
        self.alg_cost_est[alg_name] = ans
        return ans

    async def determine_best_ranking_order(self, arr, sampled_data=None):
        # arr is extracted = [(r[0], tokens2price(self.model, r[2], r[3]), name) for r, name in zip(results, names)]
        if self.proxy_ground_truth_policy == 'borda':
            rankings = []
            ranking_algs = []
            
            each_rank_length = len(arr[0][0])
            for item in arr:
                sorted_data, curr_price, alg_name = item[0], item[1], item[2]
                assert len(sorted_data) == each_rank_length, print(f'{alg_name} sorted data length {len(sorted_data)} is not equal to each rank length {each_rank_length}')
                if 'ext_bubble' in alg_name or 'ext_merge' in alg_name or 'quick' in alg_name:
                    ranking = []
                    for data in sorted_data:
                        if type(data) == tuple:
                            ranking.append(data[0])
                        else:
                            ranking.append(data)
                    rankings.append(ranking[-self.k:])
                    ranking_algs.append(alg_name)
            return borda(rankings, self.k)
        elif self.proxy_ground_truth_policy == 'llm_judge':
            rankings = []
            candidate_algs = []
            for sorted_data, est_price, alg_name in arr:
                sorted_ids = []
                for data in sorted_data:
                    if type(data) == tuple:
                        sorted_ids.append(data[0])
                    else:
                        sorted_ids.append(data)
                rankings.append(sorted_ids[-self.k:])
                candidate_algs.append(alg_name)

            if len(candidate_algs) == 1:
                return candidate_algs[0]

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
                if alg_index>=1 and alg_index<=len(candidate_algs):
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
                for item in arr:
                    sorted_data, curr_price, alg_name = item[0], item[1], item[2]
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
                for item in arr:
                    sorted_data, curr_price, alg_name = item[0], item[1], item[2]
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
            assert False
    
    async def physical_order_by_impl(self, seed=2):
        factual_knowledge, input_tokens, output_tokens, web_search_query = await self.is_factual_knowledge()
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        if factual_knowledge:
            if self.wiki_field is not None:
                return await self.map_algname_2_alg(self.data[:], 'web_point', final_decision=True), 'web_point', self.ranking_budget, self.optimization_budget
            else:
                from .tools import ddg_search_cached
                self.web_search_context = ddg_search_cached(web_search_query) if web_search_query else ''
                print(f'Factual knowledge detected, using DuckDuckGo search on query: "{web_search_query}"')
                return await self.map_algname_2_alg(self.data[:], 'ddg_point', final_decision=True), 'ddg_point', self.ranking_budget, self.optimization_budget


        sample_size = int(self.sample_size)
        assert sample_size <= len(self.data), print(f'sample size {sample_size} is greater than the data size {len(self.data)}')
        sampled_data = self.data[:sample_size]

        results, names, invoked_budget = await self.invoke_all_on_samples(
            sampled_data[:], ['ext_merge_4', 'quick']
        )

        for r, n in zip(results, names):
            assert len(r) == 4, print(r, n)
            self.total_input_tokens += r[2]
            self.total_output_tokens += r[3]
        self.optimization_budget += invoked_budget
        # print('invoked_budget', invoked_budget, self.optimization_budget)

        extracted = [(r[0], tokens2price(self.model, r[2], r[3]), name, r[1]) for r, name in zip(results, names)]

        # Estimate quick_3 total price using quick's sample cost (quick_3 ≈ 3x quick).
        quick_entry = next(((r, name) for r, name in zip(results, names) if name == 'quick'), None)
        if quick_entry:
            q_r, _ = quick_entry
            q3_est_price = self.estimated_total_price('quick_3', tokens2price(self.model, q_r[2], q_r[3]) * 3, sample_size, actual_sample_api_calls=q_r[1] * 3)
            if q3_est_price <= self.ranking_budget:
                q3_results, q3_names, q3_budget = await self.invoke_all_on_samples(sampled_data[:], ['quick_3'])
                for r, n in zip(q3_results, q3_names):
                    assert len(r) == 4, print(r, n)
                    self.total_input_tokens += r[2]
                    self.total_output_tokens += r[3]
                self.optimization_budget += q3_budget
                extracted += [(r[0], tokens2price(self.model, r[2], r[3]), name, r[1]) for r, name in zip(q3_results, q3_names)]
       
       
       # Extract per-call cost from the batch-4 runs to estimate larger batch sizes.
        base_batch = 4
        base_costs = {}   # {prefix: (sample_price, num_calls)}
        for r, name in zip(results, names):
            if name == 'ext_merge_4':
                base_costs['ext_merge'] = (tokens2price(self.model, r[2], r[3]), r[1])
            elif name == 'ext_bubble_4':
                base_costs['ext_bubble'] = (tokens2price(self.model, r[2], r[3]), r[1])

        # Determine the minimum batch size for ext_point, ext_merge, and ext_bubble.
        # For ext_bubble and ext_merge, scale the known batch-4 sample cost by b/4
        # to estimate the per-call token cost at batch size b, avoiding extra API calls.
        ext_point_batch = -1
        for b in range(self.batch_space[0], self.batch_space[1]+1, 2):
            _, num_calls, in_tokens, out_tokens = await self.map_algname_2_alg(sampled_data[:b], f'ext_point_{b}')
            self.total_input_tokens += in_tokens
            self.total_output_tokens += out_tokens
            est_price = self.estimated_total_price(f'ext_point_{b}', tokens2price(self.model, in_tokens, out_tokens), b, actual_sample_api_calls=num_calls)
            if est_price <= self.ranking_budget:
                ext_point_batch = b
                break

        bubble_batch = -1
        if 'ext_bubble' in base_costs:
            b4_price, b4_calls = base_costs['ext_merge']
            for b in range(self.batch_space[0], self.batch_space[1]+1, 2):
                scaled_price = b4_price * (b / base_batch)
                est_price = self.estimated_total_price(f'ext_bubble_{b}', scaled_price, sample_size, actual_sample_api_calls=b4_calls)
                if est_price <= self.ranking_budget:
                    bubble_batch = b
                    break

        merge_batch = -1
        if 'ext_merge' in base_costs:
            b4_price, b4_calls = base_costs['ext_merge']
            for b in range(self.batch_space[0], self.batch_space[1]+1, 2):
                scaled_price = b4_price * (b / base_batch)
                est_price = self.estimated_total_price(f'ext_merge_{b}', scaled_price, sample_size, actual_sample_api_calls=b4_calls)
                if est_price <= self.ranking_budget:
                    merge_batch = b
                    break

        
        
        # add ext_point, ext_bubble and ext_merge with minimum batch size that satisfies the ranking budget
        self._ext_point_batch = ext_point_batch
        batch_algs = []
        if merge_batch >= 6 and f'ext_merge_{merge_batch}' not in batch_algs:
            batch_algs.append(f"ext_merge_{merge_batch}")
        elif merge_batch == 4 and self.proxy_ground_truth_policy == 'borda':
            batch_algs.append(f"ext_merge_6")

        if bubble_batch >= 6 and f'ext_bubble_{bubble_batch}' not in batch_algs:
            batch_algs.append(f"ext_bubble_{bubble_batch}")
        elif bubble_batch == 4 and self.proxy_ground_truth_policy == 'borda':
            batch_algs.append(f"ext_bubble_6")

        results, names, additional_invoked_budget = await self.invoke_all_on_samples(
            sampled_data[:], batch_algs
        )
        for r, n in zip(results, names):
            assert len(r) == 4, print(r, n)
            self.total_input_tokens += r[2]
            self.total_output_tokens += r[3]
        self.optimization_budget += additional_invoked_budget
        # print('invoked_budget', invoked_budget)
        # print('additional_invoked_budget', additional_invoked_budget, batch_algs)
        # print('total optimization budget', self.optimization_budget)
        
        extracted += [(r[0], tokens2price(self.model, r[2], r[3]), name, r[1]) for r, name in zip(results, names)]

        if self.proxy_ground_truth_policy == 'llm_judge':
            filtered_extracted = []
            filtered_algs = set()
            for sorted_data, curr_price, alg_name, num_calls in extracted:
                est_price = self.estimated_total_price(alg_name, curr_price, sample_size, actual_sample_api_calls=num_calls)
                if est_price < self.ranking_budget:
                    filtered_algs.add(alg_name)

            for sorted_data, curr_price, alg_name, num_calls in extracted:
                est_price = self.estimated_total_price(alg_name, curr_price, sample_size, actual_sample_api_calls=num_calls)
                if est_price < self.ranking_budget:
                    if 'ext_bubble' in alg_name and 'ext_bubble_4' in filtered_algs and alg_name != 'ext_bubble_4':
                        continue
                    if 'ext_merge' in alg_name and 'ext_merge_4' in filtered_algs and alg_name != 'ext_merge_4':
                        continue
                    filtered_extracted.append((sorted_data, est_price, alg_name))

            # print('number of candidates', len(filtered_extracted))
            fallback_alg = f'ext_point_{self._ext_point_batch}' if self._ext_point_batch > 0 else 'point'
            if len(filtered_extracted) == 0:
                best_alg = fallback_alg
            else:
                best_alg = await self.determine_best_ranking_order(filtered_extracted, sampled_data[:])
            return await self.map_algname_2_alg(self.data[:], best_alg, final_decision=True), best_alg, self.ranking_budget, self.optimization_budget


        elif self.proxy_ground_truth_policy == 'ideal':
            filtered_extracted = []
            for sorted_data, curr_price, alg_name, num_calls in extracted:
                est_price = self.estimated_total_price(alg_name, curr_price, sample_size, actual_sample_api_calls=num_calls)
                if est_price < self.ranking_budget:
                    filtered_extracted.append((sorted_data, curr_price, alg_name))
            fallback_alg = f'ext_point_{self._ext_point_batch}' if self._ext_point_batch > 0 else 'point'
            if len(filtered_extracted) == 0:
                best_alg = fallback_alg
            else:
                best_alg = await self.determine_best_ranking_order(filtered_extracted, sampled_data[:])
            return await self.map_algname_2_alg(self.data[:], best_alg, final_decision=True), best_alg, self.ranking_budget, self.optimization_budget

        elif self.proxy_ground_truth_policy == 'borda':
            # Build all rankings once (used for leave-one-out Borda).
            all_rankings = {}
            for sorted_data, curr_price, alg_name, num_calls in extracted:
                ranking = []
                for data in sorted_data:
                    ranking.append(data[0] if type(data) == tuple else data)
                all_rankings[alg_name] = ranking[-self.k:]

            candidates = []

            for idx, (sorted_data, curr_price, alg_name, num_calls) in enumerate(extracted):
                est_price = self.estimated_total_price(alg_name, curr_price, sample_size, actual_sample_api_calls=num_calls)
                if est_price < self.ranking_budget:
                    # Leave-one-out: Borda consensus excluding the current algorithm.
                    loo_rankings = [r for name, r in all_rankings.items() if name != alg_name]
                    if len(loo_rankings) == 0:
                        continue
                    gold_sorted_data = borda(loo_rankings, self.k)

                    gold_ids = gold_sorted_data
                    pred = sorted_data
                    if type(gold_sorted_data[0]) == tuple:
                        gold_ids = [infos[0] for infos in gold_sorted_data]
                    if type(sorted_data[0]) == tuple:
                        pred = [infos[0] for infos in sorted_data]

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
                            'Q1': {str(doc_id): int(i+1) for i, doc_id in enumerate(pred)}
                        }
                        metrics = evaluator.evaluate(run)
                        quality = sum(metric[f'ndcg_cut_{self.k}'] for metric in metrics.values()) / len(metrics)
                    candidates.append((quality, est_price, alg_name))

            all_algs = set()
            for _, _, alg_name in candidates:
                all_algs.add(alg_name)

            filtered_candidates = []
            for quality, est_price, alg_name in candidates:
                if 'ext_bubble' in alg_name and 'ext_bubble_4' in all_algs and alg_name != 'ext_bubble_4':
                    continue
                if 'ext_merge' in alg_name and 'ext_merge_4' in all_algs and alg_name != 'ext_merge_4':
                    continue
                filtered_candidates.append((quality, est_price, alg_name))
            candidates = filtered_candidates[:]


            # print('number of candidates', len(candidates))
            fallback_alg = f'ext_point_{self._ext_point_batch}' if self._ext_point_batch > 0 else 'point'
            if len(candidates) == 0:
                best_alg = fallback_alg
            else:
                best_candidate = max(candidates, key=lambda x: (x[0], x[1])) if candidates else None
                assert best_candidate != None
                _, _, best_alg = best_candidate
            final_ans = await self.map_algname_2_alg(self.data[:], best_alg, final_decision=True)
            return final_ans, best_alg, self.ranking_budget, self.optimization_budget



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
            batch_size = int(alg_name.split('_')[-1])
            if self.isPassage or self.isReview:
                data_points, scores, num_api_calls, in_tokens, out_tokens, texts =\
                    await external_pointwise_sort(arr, external_values, self.client, self.external_pointwise_prompt_template,\
                                                self.model, float, 0.1, isPassage=self.isPassage, isReview=self.isReview, memory_size=batch_size)
                sorted_pairs = sorted(zip(scores, data_points), key=lambda x: (x[0], -len(x[1])))
                sorted_scores, sorted_data = zip(*sorted_pairs)
                sorted_data = [(data, score, 'score') for data, score in zip(sorted_data, sorted_scores)]
            else:
                sorted_data, num_api_calls, in_tokens, out_tokens =\
                    await external_pointwise_sort(arr, external_values, self.client, self.external_pointwise_prompt_template,\
                                                  self.model, str, memory_size=batch_size)
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
        elif alg_name == "web_point":
            if self.isPassage or self.isReview:
                data_points, scores, num_api_calls, in_tokens, out_tokens =\
                    await pointwise_sort(arr, self.client, self.pointwise_prompt_template,
                                            self.model, float, input_key_class, self.isPassage,
                                            isReview=self.isReview, use_wiki=True,
                                            wiki_field=self.wiki_field)
                sorted_pairs = sorted(zip(scores, data_points), key=lambda x: (x[0], -len(x[1])))
                sorted_scores, sorted_data = zip(*sorted_pairs)
                sorted_data = [(data, score, 'web_score') for data, score in zip(sorted_data, sorted_scores)]
            else:
                sorted_data, num_api_calls, in_tokens, out_tokens =\
                    await pointwise_sort(arr, self.client, self.pointwise_prompt_template,
                                            self.model, float, input_key_class, self.isPassage,
                                            isReview=self.isReview, use_wiki=True,
                                            wiki_field=self.wiki_field)
        elif alg_name == "ddg_point":
            ctx = getattr(self, 'web_search_context', '')
            augmented_prompt = (
                f"Use the following web search results as context.\n\n"
                f"Web search results:\n{ctx}\n\n"
                f"{self.pointwise_prompt_template}"
            ) if ctx else self.pointwise_prompt_template
            if self.isPassage or self.isReview:
                data_points, scores, num_api_calls, in_tokens, out_tokens =\
                    await pointwise_sort(arr, self.client, augmented_prompt,
                                            self.model, float, input_key_class, self.isPassage, isReview=self.isReview)
                sorted_pairs = sorted(zip(scores, data_points), key=lambda x: (x[0], -len(x[1])))
                sorted_scores, sorted_data = zip(*sorted_pairs)
                sorted_data = [(data, score, 'ddg_score') for data, score in zip(sorted_data, sorted_scores)]
            else:
                sorted_data, num_api_calls, in_tokens, out_tokens =\
                    await pointwise_sort(arr, self.client, augmented_prompt,
                                            self.model, float, input_key_class, self.isPassage, isReview=self.isReview)
        else:
            print(f"what is {alg_name} referring to?")
            assert False
        if final_decision:
            in_tokens += self.total_input_tokens
            out_tokens += self.total_output_tokens
        else:
            if self.k:
                sorted_data = sorted_data[-self.k:]
                assert len(sorted_data) <= self.k, print(f'{alg_name} sorted data length {len(sorted_data)} is greater than k {self.k}')
        return list(sorted_data), num_api_calls, in_tokens, out_tokens



    async def invoke_all_on_samples(self, samples, alg_names: list[str]):
        """Run each algorithm in alg_names on samples concurrently.

        Skips any algorithm already present in self.invoked_algs.
        Results are stored in self.sample_results keyed by algorithm name
        and also returned as (results, names) for backward-compatible use.
        """
        tasks = []
        names = []
        for name in alg_names:
            if name not in self.invoked_algs:
                tasks.append(asyncio.create_task(self.map_algname_2_alg(samples[:], name)))
                names.append(name)
                self.invoked_algs[name] = True

        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert len(results) == len(names)

        invoked_price = 0.0
        for name, result in zip(names, results):
            self.sample_results[name] = result
            invoked_price += tokens2price(self.model, result[2], result[3])

        return results, names, invoked_price