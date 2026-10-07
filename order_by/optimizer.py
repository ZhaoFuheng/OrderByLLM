from .sorting import *
from .utils import *
from collections import defaultdict
from .pointwise import *

import random
from typing import Any, List, Optional, Sequence
from pydantic import BaseModel
from .cache import cache
from . import jev
import asyncio
import math
import pytrec_eval
import os
from enum import Enum


def _progress_write(message: str) -> None:
    try:
        from tqdm import tqdm
        tqdm.write(message)
    except Exception:
        print(message)

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
        judge_client = None,
        good_algs = None,
        bm25_passage = None,
        isReview = False,
        run_all = False,
        enable_factual_web_search = True,
        external_pointwise_memory_size = 8,
        wiki_field = None,
        use_simulation_estimate: bool = False,
        rrf_k: int = 60,
        ensemble_max_lists: int = 0,
    ):
        assert proxy_ground_truth_policy in ['borda', 'rrf', 'rrf_ensemble', 'borda_ensemble', 'llm_judge', 'ideal'], print(f"proxy_ground_truth_policy must be one of ['borda', 'rrf', 'rrf_ensemble', 'borda_ensemble', 'llm_judge', 'ideal'], but got {proxy_ground_truth_policy}")
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
        self.rrf_k = rrf_k  # reciprocal-rank-fusion constant (default 60)
        # Cap the number of lists fused in the ensemble final aggregation. 0 = no
        # cap. Keeps only the top-quality lists so a large quality gap between the
        # best and the rest can't drown the strong list under many weak votes.
        self.ensemble_max_lists = ensemble_max_lists
        self._ext_point_batch = 4
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
        # Optional separate client for the LLM judge, so the judge can run on a
        # different provider than the ranking model (e.g. haiku ranking + gpt-5-nano
        # judge). None -> reuse self.client (same provider as ranking).
        self.judge_client = judge_client

        # min max boundary
        self.batch_space = [4, 8]
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
        return tpl.format_map(defaultdict(str, example=example))
    
    async def _call_llm(self, prompt, schema, llm_judge=False):
        if llm_judge:
            model = self.judge_model
            active_client = self.judge_client if self.judge_client is not None else self.client
        else:
            model = self.model
            active_client = self.client

        key_hash = hash_prompt(prompt, model)
        if key_hash in cache:
            try:
                cached = cache[key_hash]
                cached_parsed = cached['parsed']
                if schema is SeenStatus and 'webSearchQuery' not in cached_parsed:
                    cached_parsed = {**cached_parsed, 'webSearchQuery': ''}
                parsed = schema(**cached_parsed)
                if 'input_tokens' in cached:
                    input_tokens = cached['input_tokens']
                else:
                    input_tokens = count_tokens(prompt)
                return parsed.dict(), 0, input_tokens, cached['tokens']-input_tokens
            except Exception:
                del cache[key_hash]

        response = None
        parsed = None
        for i in range(1, 11):
            suffix = ''
            if i > 2:
                suffix = " If isFactualKnowledge is 'No', still put an empty string for webSearchQuery."
            try:
                if type(active_client) == SnowflakeClient:
                    response = await resolve(active_client.responses(
                        model=model,
                        prompt=prompt + suffix,
                        schema=schema,
                    ))
                    parsed = response.output_parsed
                else:
                    response = await resolve(active_client.beta.chat.completions.parse(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a helpful agent. Think step by step. Output a JSON object."},
                                {"role": "user", "content": prompt + suffix}],
                            temperature=0.0,
                            response_format=schema,
                            max_completion_tokens = 20*1024,
                        ))
                    parsed = response.choices[0].message.parsed
                break
            except Exception as e:
                log.warning("optimizer _call_llm %s attempt %d/10: %s", model, i, e)
                if '429' in str(e):
                    await asyncio.sleep(10)
                continue

        if response is None or parsed is None:
            log.error("optimizer _call_llm %s: all retries exhausted for schema %s", model, schema.__name__)
            raise RuntimeError(f"optimizer _call_llm {model}: all 10 retries exhausted for {schema.__name__}")

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

    async def process_factual_knowledge_items(self, items):
        formatted_rows = []
        for item in items:
            if not self.has_id_and_row:
                if len(item) == 2:
                    item = item[1]
                assert isinstance(item, str) and not item.isdigit(), f"Invalid item: {item}"
                formatted_rows.append(str(item))
            else:
                assert len(item) == 2, print(f'item must be a tuple of (id, row), but got {item}')
                formatted_rows.append(str(item[1]))
        prompt = self._format_factual_knowledge_prompt("\n\n".join(formatted_rows))
        parsed, api_calls, input_tokens, output_tokens = await self._call_llm(prompt, self.factual_knowledge_schema)
        return parsed, input_tokens, output_tokens


    async def process_llm_judge_item(self, prompt):
        parsed, api_calls, input_tokens, output_tokens = await self._call_llm(prompt, self.llm_judge_schema, llm_judge=True)
        return parsed['id'], input_tokens, output_tokens

    async def is_factual_knowledge(self, sample_size=5, seed=1):
        assert sample_size < len(self.data), print("sample size too large")
        random.seed(seed)
        if self.bm25_passage:
            sampled_data = self.bm25_passage[-sample_size:]
        else:
            sampled_data = random.sample(self.data, min(sample_size, len(self.data)))

        parsed, total_input_tokens, total_output_tokens = await self.process_factual_knowledge_items(sampled_data)
        assert parsed['isFactualKnowledge'] == 'Yes' or parsed['isFactualKnowledge'] == 'No', print(f'isFactualKnowledge must be Yes or No, but got {parsed["isFactualKnowledge"]}')
        web_search_query = parsed.get('webSearchQuery', '').strip()
        return parsed['isFactualKnowledge'] == 'Yes', total_input_tokens, total_output_tokens, web_search_query

    def estimated_total_price(self, alg_name, curr_price, sample_size, actual_sample_api_calls=None):
        if 'quick_3' == alg_name:
            assert 'quick' in self.alg_cost_est
            ans = 3*self.alg_cost_est['quick']
            self.alg_cost_est[alg_name] = ans
            return ans
        curr_price = float(curr_price)
        total_size = len(self.data)
        if alg_name == 'point' or 'ext_point' in alg_name:
            sample_ratio = 1.0 * sample_size / total_size
            ans = curr_price / sample_ratio
        elif 'quick' == alg_name:
            v = 1
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
                # correction_factor = expected_calls / s_calls
                correction_factor = 1.0
                total_calls = quick_calls_formula(total_size, v, self.k)
                ans = curr_price * total_calls * correction_factor / max(s_calls, 1)
            # ans *= 1.3 # avoid underestimation
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
            # ans *= 1.3 # avoid underestimation
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
            # ans *= 1.3 # avoid underestimation
        # print(alg_name, 'estimated cost', ans)
        self.alg_cost_est[alg_name] = ans
        return ans

    async def determine_best_ranking_order(self, arr, sampled_data=None):
        if self.proxy_ground_truth_policy == 'llm_judge':
            rankings = []
            candidate_algs = []
            for sorted_data, est_price, alg_name in arr:
                if 'point' in alg_name:
                    continue
                sorted_ids = []
                for data in sorted_data:
                    if type(data) == tuple:
                        sorted_ids.append(data[0])
                    else:
                        sorted_ids.append(data)
                rankings.append(sorted_ids[-self.k:])
                candidate_algs.append(alg_name)

            if len(candidate_algs) == 0:
                # Only point-based algorithms are affordable at this budget, and the
                # judge doesn't rank those — nothing to compare. Fall back to
                # ext_point_4 instead of asking the judge to pick from an empty list
                # (which spins 7 retries then IndexErrors).
                return f'ext_point_{self._ext_point_batch}'

            if len(candidate_algs) == 1:
                return candidate_algs[0]

            judge_client = self.judge_client if self.judge_client is not None else self.client
            if jev.is_jev(judge_client):
                # Jev judges the rankings as one Choice question posed on the ranking
                # task's JevPrompt (the same object held in every *_prompt_template).
                best, _, input_tokens, output_tokens = await jev.judge_rankings(
                    judge_client, self.pairwise_comparison_prompt_template, sampled_data, rankings)
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                return candidate_algs[best]

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

    async def physical_order_by_impl(self, seed=42):
        self.rng = random.Random(seed)
        if jev.is_jev(self.client):
            # Jev tasks judge the text they are given (prompts/jev_questions.py), so the
            # factual-knowledge inquiry and the web-search algorithms do not apply.
            factual_knowledge, input_tokens, output_tokens, web_search_query = False, 0, 0, ''
        else:
            factual_knowledge, input_tokens, output_tokens, web_search_query = await self.is_factual_knowledge()
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        if factual_knowledge:
            if self.wiki_field is not None:
                return await self.map_algname_2_alg(self.data[:], 'web_point', final_decision=True), 'web_point', self.ranking_budget, self.optimization_budget
            else:
                from .tools import ddg_search_cached
                self.web_search_context = ddg_search_cached(web_search_query) if web_search_query else ''
                _progress_write(f'Factual knowledge detected, using DuckDuckGo search on query: "{web_search_query}"')
                return await self.map_algname_2_alg(self.data[:], 'ddg_point', final_decision=True), 'ddg_point', self.ranking_budget, self.optimization_budget


        sample_size = int(self.sample_size)
        assert sample_size <= len(self.data), print(f'sample size {sample_size} is greater than the data size {len(self.data)}')
        sampled_data = self.data[:sample_size]

        base_batch = 8
        init_algs = ['point',f'ext_merge_{base_batch}','quick']

        results, names, invoked_budget = await self.invoke_all_on_samples(
            sampled_data[:], init_algs
        )
        
        for r, n in zip(results, names):
            assert len(r) == 4, print(r, n)
            self.total_input_tokens += r[2]
            self.total_output_tokens += r[3]
        self.optimization_budget += invoked_budget

        extracted = [(r[0], tokens2price(self.model, r[2], r[3]), name, r[1]) for r, name in zip(results, names)]

        # Estimate quick_3 total price using quick's sample cost (quick_3 ≈ 3x quick).
        quick_entry = next(((r, name) for r, name in zip(results, names) if name == 'quick'), None)
        if quick_entry:
            _ = self.estimated_total_price('quick', tokens2price(self.model, quick_entry[0][2], quick_entry[0][3]), sample_size, actual_sample_api_calls=quick_entry[0][1])
            q_r, _ = quick_entry
            q3_est_price = self.estimated_total_price('quick_3', None, None, None)
            if q3_est_price <= self.ranking_budget:
                init_algs.append('quick_3')
                q3_results, q3_names, q3_budget = await self.invoke_all_on_samples(sampled_data[:], ['quick_3'])
                for r, n in zip(q3_results, q3_names):
                    assert len(r) == 4, print(r, n)
                    self.total_input_tokens += r[2]
                    self.total_output_tokens += r[3]
                self.optimization_budget += q3_budget
                extracted += [(r[0], tokens2price(self.model, r[2], r[3]), name, r[1]) for r, name in zip(q3_results, q3_names)]

       # Extract per-call cost from the batch-8 runs to estimate larger batch sizes.
        base_costs = {}   # {prefix: (sample_price, num_calls)}
        for r, name in zip(results, names):
            if 'ext_merge' in name:
                base_costs['ext_merge'] = (tokens2price(self.model, r[2], r[3]), r[1])
                
        most_expensive_bubble_batch = -1
        assert 'ext_merge' in base_costs, print(f'ext_merge not in base_costs: {base_costs}')
        b8_price, b8_calls = base_costs['ext_merge']
        for b in range(self.batch_space[0], self.batch_space[1]+1, 2):
            scaled_price = b8_price * (b / base_batch)
            est_price = self.estimated_total_price(f'ext_bubble_{b}', scaled_price, sample_size, actual_sample_api_calls=b8_calls)
            if est_price <= self.ranking_budget:
                most_expensive_bubble_batch = b
                break

        # Same for external merge sort: the smallest affordable batch size and
        # every larger one up to the base batch join the candidate pool.
        most_expensive_merge_batch = -1
        for b in range(self.batch_space[0], self.batch_space[1]+1, 2):
            scaled_price = b8_price * (b / base_batch)
            est_price = self.estimated_total_price(f'ext_merge_{b}', scaled_price, sample_size, actual_sample_api_calls=b8_calls)
            if est_price <= self.ranking_budget:
                most_expensive_merge_batch = b
                break

        batch_algs = []
        for prefix, smallest in (('ext_merge', most_expensive_merge_batch), ('ext_bubble', most_expensive_bubble_batch)):
            if smallest > 0:
                for b in range(smallest, self.batch_space[-1]+1, 2):
                    name = f'{prefix}_{b}'
                    if name not in init_algs:   # ext_merge_{base_batch} was already sampled
                        batch_algs.append(name)
                        init_algs.append(name)

        results, names, additional_invoked_budget = await self.invoke_all_on_samples(
            sampled_data[:], batch_algs
        )
        for r, n in zip(results, names):
            assert len(r) == 4, print(r, n)
            self.total_input_tokens += r[2]
            self.total_output_tokens += r[3]
        self.optimization_budget += additional_invoked_budget
        
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

        elif self.proxy_ground_truth_policy in ('borda', 'rrf', 'rrf_ensemble', 'borda_ensemble'):
            # Two aggregation steps, independently chosen:
            #   score_agg  -> builds the proxy-ground-truth consensus used to score candidates
            #   final_agg  -> aggregates the selected algorithms' full rankings (ensemble only)
            # borda / borda_ensemble: borda. rrf / rrf_ensemble: rrf.
            _pol = self.proxy_ground_truth_policy
            _rrf = lambda rankings: rrf(rankings, k=self.rrf_k)  # honor configurable rrf k
            score_agg = borda if _pol in ('borda', 'borda_ensemble') else _rrf
            final_agg = borda if _pol == 'borda_ensemble' else _rrf
            is_ensemble = 'ensemble' in _pol
            all_rankings = {}

            all_init_rankings = {}
            for sorted_data, curr_price, alg_name, num_calls in extracted:
                ranking = []
                for data in sorted_data:
                    ranking.append(data[0] if type(data) == tuple else data)
                all_rankings[alg_name] = ranking[-self.k:]
                if alg_name in init_algs:
                    all_init_rankings[alg_name] = ranking[-self.k:]


            # Consensus 'gold' proxy, computed ONCE from all candidate rankings.
            assert len(all_rankings) > 0, print('all_rankings is empty')
            # gold_sorted_data = score_agg(all_rankings.values())
            gold_sorted_data = score_agg(all_init_rankings.values())
            gold_sorted_data = gold_sorted_data[-self.k:]
            gold_ids = [doc_id for (doc_id, score) in gold_sorted_data]

            def _quality(pred, gold_ids):
                # Agreement of a ranking `pred` with the consensus `gold_ids`.
                if not self.isPassage and not self.isReview:
                    return kendalltau_distance(gold_ids[:], pred)
                top_k = self.k
                gold = {'Q1': {str(doc_id): int(i + 1) for i, doc_id in enumerate(gold_ids)}}
                evaluator = pytrec_eval.RelevanceEvaluator(gold, {f'ndcg_cut.{top_k}'})
                run = {'Q1': {str(doc_id): int(i + 1) for i, doc_id in enumerate(pred)}}
                metrics = evaluator.evaluate(run)
                return sum(m[f'ndcg_cut_{top_k}'] for m in metrics.values()) / len(metrics)

            candidates = []
            for idx, (sorted_data, curr_price, alg_name, num_calls) in enumerate(extracted):
                est_price = self.estimated_total_price(alg_name, curr_price, sample_size, actual_sample_api_calls=num_calls)
                if est_price < self.ranking_budget:
                    pred = sorted_data
                    if type(sorted_data[0]) == tuple:
                        pred = [infos[0] for infos in sorted_data]
                    quality = _quality(pred, gold_ids)
                    candidates.append((quality, est_price, alg_name))

            all_algs = set()
            for _, _, alg_name in candidates:
                all_algs.add(alg_name)

            filtered_candidates = []
            for quality, est_price, alg_name in candidates:
                # if 'point' in alg_name:
                #     continue
                filtered_candidates.append((quality, est_price, alg_name))
            candidates = filtered_candidates[:]

            fallback_alg = f'ext_point_{self._ext_point_batch}' if self._ext_point_batch > 0 else 'point'

            if is_ensemble:
                # Rank affordable candidates by proxy quality, greedily pick the subset
                # whose SUM of estimated costs stays under the budget, run each on the
                # FULL data, and aggregate (final_agg) their rankings into a single output.
                if len(candidates) == 0:
                    selected = [fallback_alg]
                else:
                    ranked = sorted(candidates, key=lambda x: (-x[0], x[1]))  # quality desc, cheaper tie
                    selected, cum = [], 0.0
                    for _q, price, name in ranked:
                        # Cap the number of fused lists (0 = no cap): keep only the
                        # top-quality lists so a runaway best list isn't diluted.
                        if self.ensemble_max_lists and len(selected) >= self.ensemble_max_lists:
                            break
                        if not selected or cum + price < self.ranking_budget:
                            selected.append(name)
                            cum += price
                agg_input, tot_calls, tot_in, tot_out = [], 0, 0, 0
                # The selected ensemble algorithms are independent, so run them
                # concurrently (like the sampling phase's invoke_all_on_samples).
                # gather preserves order, so agg_input stays aligned with `selected`.
                _ens_results = await asyncio.gather(*[
                    self.map_algname_2_alg(self.data[:], name, final_decision=False)
                    for name in selected])
                for sd, calls, in_t, out_t in _ens_results:
                    agg_input.append([(d[0] if type(d) == tuple else d) for d in sd])
                    tot_calls += calls
                    tot_in += in_t
                    tot_out += out_t
                if len(agg_input) == 1:
                    final_ids = agg_input[0]
                else:
                    final_ids = [doc_id for (doc_id, _s) in final_agg(agg_input)]
                # fold the shared optimization tokens in once (as final_decision would)
                final_ans = (final_ids, tot_calls,
                             tot_in + self.total_input_tokens, tot_out + self.total_output_tokens)
                return final_ans, '+'.join(selected), self.ranking_budget, self.optimization_budget

            if len(candidates) == 0:
                best_alg = fallback_alg
            else:
                best_candidate = max(candidates, key=lambda x: (x[0], x[1]))
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