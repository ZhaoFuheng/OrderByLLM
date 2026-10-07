"""Ranking primitives on Jev, TypeSafe's System One model.

Jev answers typed questions (Score / Choice / Noul) about a JSON state and
returns probabilities instead of text, so the four atomic operations the
sorting algorithms rely on are expressed as questions here rather than as
prompts:

* pointwise value          -> one Score question over the task's rubric; the
                              value is the expected score (continuous, so far
                              fewer ties than an LLM's integer grades)
* external (batched) values-> one request, one Score question per item
* pairwise comparison      -> a Choice between A and B, asked in both option
                              orders in the same request to cancel Jev's
                              first-option bias
* external comparison      -> one Score question per item, sorted by expected
                              score ("scores", the default); one Choice over all
                              items sorted by probability ("choice"); or a Choice
                              for the best item, repeated on the rest until one is
                              left ("sequential", n-1 requests per batch). On
                              DL19 "scores" beat "choice" by 0.03 / 0.014 nDCG@10
                              for merge / bubble sort.
* ranking judge            -> the optimizer's llm_judge policy as one Choice over
                              the candidate rankings, which sit in the state next
                              to the items they order

Responses are cached in their own cache (`jev_cache/`), keyed like the LLM
cache on the model name and the canonical request JSON, so a run whose
questions are all cached costs nothing and `cache_tools.py --cache jev`
exports it separately.
"""
import asyncio
import json
import os

from .cache import open_jev_cache
from .clients import RealCall
from .utils import hash_prompt
from prompts.jev_questions import JevPrompt

MODEL = "jev"                 # short name: cache key, pricing, results_<model>.json
API_MODEL = "jev-latest"      # model id sent to the API
BATCH_RANKING_METHODS = ("scores", "choice", "sequential")


class JevClient:
    """Async client for the TypeSafe API, with the Jev cache in front of it and
    a concurrency gate. Without an `sdk_client` it is cache-only: a question
    that was never answered raises RealCall instead of calling the API.

    batch_ranking selects how a batch is ordered by `external_comparisons`:
    "scores" asks one Score per item, "choice" one Choice over all items,
    "sequential" one Choice per round, removing the winner each round.
    """

    def __init__(self, sdk_client=None, max_concurrency: int | None = 16, batch_ranking: str = "scores"):
        if batch_ranking not in BATCH_RANKING_METHODS:
            raise ValueError(f"batch_ranking must be one of {BATCH_RANKING_METHODS}, got {batch_ranking!r}")
        self._sdk = sdk_client
        self._sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None
        self.batch_ranking = batch_ranking
        self.cache = open_jev_cache()
        self.misses = 0

    async def ask(self, state, questions: dict):
        """Answer `questions` about `state`. Returns (answers, input_tokens,
        output_tokens) with answers as plain dicts keyed by question name;
        served from the cache when the same request was made before."""
        request = {"model": API_MODEL, "state": state,
                   "questions": {name: q.model_dump() for name, q in questions.items()}}
        key = hash_prompt(json.dumps(request, sort_keys=True, ensure_ascii=False), MODEL)
        if key in self.cache:
            cached = self.cache[key]
            return cached["answers"], cached["input_tokens"], cached["output_tokens"]
        if self._sdk is None:
            self.misses += 1
            raise RealCall("Jev response not cached and TYPESAFE_API_KEY is not set")

        if self._sem is not None:
            async with self._sem:
                response = await self._sdk.system_one(state, questions, model=API_MODEL)
        else:
            response = await self._sdk.system_one(state, questions, model=API_MODEL)

        answers = {name: answer.model_dump() for name, answer in response.answers.items()}
        in_t = response.usage.input_tokens or 0
        out_t = response.usage.output_tokens or 0
        self.cache[key] = {"answers": answers, "tokens": in_t + out_t,
                           "input_tokens": in_t, "output_tokens": out_t}
        return answers, in_t, out_t


def build_jev_client() -> JevClient:
    """JevClient from the environment. Without TYPESAFE_API_KEY the client is
    cache-only, so cached Jev runs replay with no credentials at all.

    Env vars : TYPESAFE_API_KEY
               JEV_MAX_CONCURRENCY (default 16; set 0 to disable the gate)
               JEV_BATCH_RANKING   ("scores" (default), "choice" or "sequential", see JevClient)
    """
    mc = int(os.getenv("JEV_MAX_CONCURRENCY", "16"))
    batch_ranking = os.getenv("JEV_BATCH_RANKING", "scores")
    if not os.getenv("TYPESAFE_API_KEY"):
        return JevClient(None, batch_ranking=batch_ranking)
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
    sdk = AsyncTypeSafeClient(timeout=60.0, retry=RetryPolicy(max_retries=4, timeout=120.0))
    return JevClient(sdk, max_concurrency=mc if mc > 0 else None, batch_ranking=batch_ranking)


def is_jev(client) -> bool:
    return isinstance(client, JevClient)


# ── question builders ────────────────────────────────────────────────────────

def _score_question(prompt: JevPrompt, item_path: str):
    from typesafe_sdk import Score
    return Score(instructions=prompt.task.pointwise.format(item=item_path), criteria=list(prompt.task.levels))


def _expected_score(answer: dict, prompt: JevPrompt) -> float:
    return float(answer["score"]) + prompt.task.level_offset


def _item_ids(n: int) -> list[str]:
    return [f"item{i + 1}" for i in range(n)]


# ── the four atomic operations ───────────────────────────────────────────────

async def pointwise_value(client: JevClient, prompt: JevPrompt, text: str):
    """Expected rubric score of one item. Returns (value, 1, in_tokens, out_tokens)."""
    answers, in_t, out_t = await client.ask(prompt.state(item=text), {"score": _score_question(prompt, "item")})
    return _expected_score(answers["score"], prompt), 1, in_t, out_t


async def external_values(client: JevClient, prompt: JevPrompt, texts: list[str]):
    """Expected rubric score of each item, from one request with one Score
    question per item. Returns ([values], 1, in_tokens, out_tokens)."""
    ids = _item_ids(len(texts))
    state = prompt.state(items=dict(zip(ids, texts)))
    questions = {i: _score_question(prompt, f"items.{i}") for i in ids}
    answers, in_t, out_t = await client.ask(state, questions)
    return [_expected_score(answers[i], prompt) for i in ids], 1, in_t, out_t


async def compare(client: JevClient, prompt: JevPrompt, a: str, b: str):
    """'A' if item a is judged better than item b, else 'B'. The Choice is asked
    with the options in both orders and the probabilities averaged, because Jev
    leans toward the first option. Returns (key, 1, in_tokens, out_tokens)."""
    from typesafe_sdk import Choice
    state = prompt.state(A=a, B=b)
    opt_a = f"the {prompt.task.item} stored under `A`"
    opt_b = f"the {prompt.task.item} stored under `B`"
    questions = {
        "ab": Choice(instructions=prompt.task.pairwise, criteria={"A": opt_a, "B": opt_b}),
        "ba": Choice(instructions=prompt.task.pairwise, criteria={"B": opt_b, "A": opt_a}),
    }
    answers, in_t, out_t = await client.ask(state, questions)
    p_a = (answers["ab"]["probabilities"]["A"] + answers["ba"]["probabilities"]["A"]) / 2
    return ("A" if p_a >= 0.5 else "B"), 1, in_t, out_t


def _best_question(prompt: JevPrompt, ids: list[str]):
    from typesafe_sdk import Choice
    criteria = {i: f"the {prompt.task.item} stored under `items.{i}`" for i in ids}
    return Choice(instructions=prompt.task.best, criteria=criteria)


async def external_comparisons(client: JevClient, prompt: JevPrompt, items: list[tuple]):
    """Order a batch of (key, text) items from worst to best, by the client's
    batch_ranking method. Returns ([(key, text), ...], calls, in_tokens, out_tokens)."""
    if len(items) <= 1:
        return list(items), 0, 0, 0
    ids = _item_ids(len(items))
    texts = {i: text for i, (_, text) in zip(ids, items)}
    if client.batch_ranking == "sequential":
        return await _sequential_best(client, prompt, ids, texts, items)
    state = prompt.state(items=texts)
    if client.batch_ranking == "scores":
        questions = {i: _score_question(prompt, f"items.{i}") for i in ids}
        answers, in_t, out_t = await client.ask(state, questions)
        strength = {i: answers[i]["score"] for i in ids}
    else:
        answers, in_t, out_t = await client.ask(state, {"best": _best_question(prompt, ids)})
        strength = answers["best"]["probabilities"]
    # ascending strength = worst to best; ties keep the input order (sorted is stable)
    order = sorted(range(len(items)), key=lambda k: strength[ids[k]])
    return [items[k] for k in order], 1, in_t, out_t


async def _sequential_best(client, prompt, ids, texts, items):
    """Pick the best of the remaining items with a Choice, remove it, repeat:
    n-1 requests, each over a state holding only the remaining items (ids keep
    their original numbers). Ties go to the earliest input position."""
    remaining, best_first = list(ids), []
    calls = in_t = out_t = 0
    while len(remaining) > 1:
        state = prompt.state(items={i: texts[i] for i in remaining})
        answers, i_t, o_t = await client.ask(state, {"best": _best_question(prompt, remaining)})
        calls, in_t, out_t = calls + 1, in_t + i_t, out_t + o_t
        probs = answers["best"]["probabilities"]
        winner = max(remaining, key=lambda i: (probs[i], -remaining.index(i)))
        best_first.append(winner)
        remaining.remove(winner)
    worst_to_best = remaining + best_first[::-1]
    by_id = dict(zip(ids, items))
    return [by_id[i] for i in worst_to_best], calls, in_t, out_t


async def judge_rankings(client: JevClient, prompt: JevPrompt, items: list, rankings: list[list]):
    """Index of the candidate ranking that orders `items` best (the optimizer's
    judge). Each ranking lists item keys worst to best, as the sorting
    algorithms return them; the state holds the items and every ranking best
    first, and one Choice picks among the rankings. Ties go to the earlier
    candidate. Returns (index, 1, in_tokens, out_tokens)."""
    from typesafe_sdk import Choice
    keys = [item[0] if isinstance(item, tuple) else item for item in items]
    ids = dict(zip(keys, _item_ids(len(items))))
    texts = {ids[key]: (item[1] if isinstance(item, tuple) else item) for key, item in zip(keys, items)}
    names = [f"ranking{i + 1}" for i in range(len(rankings))]
    state = prompt.state(items=texts, rankings={n: [ids[k] for k in reversed(r)] for n, r in zip(names, rankings)})
    criteria = {n: f"the ranking stored under `rankings.{n}`" for n in names}
    answers, in_t, out_t = await client.ask(state, {"best": Choice(instructions=prompt.task.judge, criteria=criteria)})
    probs = answers["best"]["probabilities"]
    best = max(range(len(names)), key=lambda i: (probs[names[i]], -i))
    return best, 1, in_t, out_t
