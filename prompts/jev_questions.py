"""How each ranking task is posed to Jev (TypeSafe's System One model).

Jev does not read our free-text prompts: it answers typed questions about a JSON
`state`. A `JevTask` therefore carries the pieces of a task that the LLM prompt
templates in all_prompts.py spell out in prose — the rubric levels (for Score
questions) and the comparison wording (for Choice questions). `JevPrompt` binds a
task to one query and is what the runners pass to the algorithms in place of a
prompt template when the provider is `jev`.

Only tasks judged from the text in the state are defined here. Ranking by world
knowledge (country populations, player heights) is not something Jev is built
for, so those datasets have no Jev task.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class JevTask:
    name: str
    item: str
    """What one candidate is called in the questions ("passage", "review")."""
    levels: tuple[str, ...]
    """Score rubric, worst to best. Mirrors the pointwise prompt of the same task."""
    level_offset: float
    """Value of the first level, so expected scores land on the LLM prompt's scale."""
    pointwise: str
    """Score question about one item. `{item}` is the state path of the item."""
    pairwise: str
    """Choice question between two items (options A and B)."""
    best: str
    """Choice question selecting the best of several items."""
    judge: str
    """Choice question selecting the best of several candidate rankings of the items
    (the optimizer's judge). Rankings are given best item first."""
    query_field: str | None
    """State field holding the query, or None when the task has no query."""


PASSAGE = JevTask(
    name="passage",
    item="passage",
    levels=(
        "Irrelevant: the passage has nothing to do with the question.",
        "Related: the passage is on-topic but does not answer the question (e.g., discusses the wrong aspect).",
        "Highly relevant: the passage answers the question but may contain some extraneous information or lacks comprehensive detail.",
        "Perfectly relevant: the passage is dedicated to the question and contains the exact answer.",
    ),
    level_offset=0.0,
    pointwise="How well does the passage stored under `{item}` answer the question in `question`?",
    pairwise="Which passage answers the question in `question` better?",
    best="Which passage answers the question in `question` best?",
    judge="Each ranking in `rankings` lists the passages in `items` from the one that answers the "
          "question in `question` best to the one that answers it worst. Which ranking is the most accurate?",
    query_field="question",
)

REVIEW = JevTask(
    name="review",
    item="review",
    levels=(
        "Very negative: strong negative sentiment, indicating high dissatisfaction, frustration, or anger.",
        "Negative: noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.",
        "Neutral: expresses no clear positive or negative sentiment; may be factual or descriptive without emotional language.",
        "Positive: noticeably positive sentiment, indicating general satisfaction.",
        "Very positive: strong positive sentiment, indicating high satisfaction.",
    ),
    level_offset=1.0,
    pointwise="How much did the reviewer who wrote the review stored under `{item}` like the movie?",
    pairwise="Which review is more positive about the movie?",
    best="Which review is the most positive about the movie?",
    judge="Each ranking in `rankings` lists the reviews in `items` from the most positive about the "
          "movie to the most negative. Which ranking is the most accurate?",
    query_field=None,
)


@dataclass(frozen=True)
class JevPrompt:
    """A task bound to one query; passed to the algorithms as their prompt."""
    task: JevTask
    query: str | None = None

    def state(self, **items) -> dict:
        """The request state: the query (if any) plus the named items."""
        state = {}
        if self.task.query_field is not None:
            state[self.task.query_field] = self.query
        state.update(items)
        return state
