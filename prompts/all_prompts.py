nba_pointwise_prompt_template = '''Given the following nba player, key: {key}.
Determine the player height in meters as a floating-point number rounded to two decimal places
'''

nba_external_pointwise_prompt_template = '''Given a list of nba players, keys: {keys}.
Determine their heights in meters as floating-point numbers rounded to two decimal places.
'''

nba_pairwise_comparison_prompt_template = '''Which nba player is taller (in meters, rounded to two decimal places): {key1} or {key2}? 
Based on your knowledge and judgement, select the name of the taller player as the answer key.
In case of tie, return EQUAL as the output key.
Always return exactly EQUAL or {key1} or {key2} in the output JSON key.
'''

nba_external_comparison_prompt_template = '''Given a list of nba players with id at the front:
{keys}
Based on your own knowledge, sort the players in ascending order based on player height (in meters, rounded to two decimal places). 
Break ties based on the player name in alphabetical order.
Output a JSON list of integer ids, representing the player from shortest to tallest.
'''


passage_pointwise_prompt_template = """You are given a question and a passage.
Evaluate how well the passage answers the question by assigning a float relevance score from 0.0 to 3.0.

0.0 (Irrelevant): The passage has nothing to do with the question.
1.0 (Related): The passage is on-topic but does not answer the question (e.g., discusses the wrong aspect).
2.0 (Highly Relevant): The passage answers the question but may contain some extraneous information or lacks comprehensive detail.
3.0 (Perfectly Relevant): The passage is dedicated to the question and contains the exact answer.

Question: {question}
Passage: {key}
"""

passage_external_pointwise_prompt_template = """You are given a question and a list of passages.
Evaluate how well each passage answers the question.
For each passage, assign a float relevance score from 0.0 to 3.0.

0.0 (Irrelevant): The passage has nothing to do with the question.
1.0 (Related): The passage is on-topic but does not answer the question (e.g., discusses the wrong aspect).
2.0 (Highly Relevant): The passage answers the question but may contain some extraneous information or lacks comprehensive detail.
3.0 (Perfectly Relevant): The passage is dedicated to the question and contains the exact answer.

Output a JSON list of float relevance scores in the same order as the input passages.
Question: {question}
Passages:
{keys}
"""

passage_pairwise_comparison_prompt_template = """You are given a question and two passages. Determine which passage answers the question better. 
Question: {question}
Passage A: {key1}
Passage B: {key2}
If Passage A is better, select key 'A' as BetterPassageKey. If Passage B is better, select key 'B' as BetterPassageKey.
"""

passage_external_comparison_prompt_template = """You are given a question and a list of passages with passage id at the front.
Rank the passages based on how well they answer the question, from worst to best.
Question: {question}
Passages:
{keys}
Output a JSON list of integer ids, representing the ranking from worst to best.
"""



population_pointwise_prompt_template = '''Given the following country: {key}, 
what was its population in 2020, expressed as a decimal in millions?
'''

population_external_pointwise_prompt_template = '''Given a list of countries, keys: {keys},
what was the population of each country in 2020, expressed as a decimal in millions?
'''

population_pairwise_comparison_prompt_template = '''Which country had a larger population in 2020: {key1} or {key2}?
Based on your knowledge, output the name of the input country with the larger population. Do not rephrase or change the country name.
In case of tie, return EQUAL as the output key.
Always return exactly EQUAL or {key1} or {key2} in the output JSON key.
'''

population_external_comparison_prompt_template = '''Given a list of countries with id at the front:
{keys}
Sort the countries in ascending order based on their 2020 population using your own knowledge and judgement. 
Break ties based on the country name in alphabetical order.
Output a JSON list of integer ids, representing the country from least populated to most populated.
'''


tweetEval_pointwise_prompt_template = """You are given a passage.
Assign a float score from 0.0 to 1.0, where 0.0 is not {criteria} and 1.0 is completely {criteria}.
Passage: {key}
"""

tweetEval_external_pointwise_prompt_template = """You are given a list of passages.
For each passage, assign a float score from 0.0 to 1.0, where 0.0 is not {criteria} and 1.0 is completely {criteria}.
Output a JSON list of float {criteria} relevance scores in the same order as the input passages.
Passages:
{keys}
"""

tweetEval_pairwise_comparison_prompt_template = """You are given two passages.
Determine which text is more {criteria}.
A: {key1}
B: {key2}
If A is more {criteria}, select key 'A' as BetterPassageKey. If B is more {criteria}, select key 'B' as BetterPassageKey.
"""

tweetEval_external_comparison_prompt_template = """You are given a list of passages with passage id at the front.
Rank the passages based on {criteria}.
Passages:
{keys}
Output a list of passage_id from the least {criteria} passage to the most {criteria} passage.
"""


movie_pointwise_prompt_template = """Score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.
    
    Rubrics:
    5: Very positive. Strong positive sentiment, indicating high satisfaction. 
    4: Positive. Noticeably positive sentiment, indicating general satisfaction. 
    3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language. 
    2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
    1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger. 

    Review:
    {key}
"""

movie_external_pointwise_prompt_template ="""You are given a list of reviews.
For each review, assign a score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.
    
    Rubrics:
    5: Very positive. Strong positive sentiment, indicating high satisfaction. 
    4: Positive. Noticeably positive sentiment, indicating general satisfaction. 
    3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language. 
    2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
    1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger. 

    Reviews:
    {keys}

Respond in JSON and the list of scores should be in the same order as the input reviews.
"""

movie_pairwise_comparison_prompt_template = """You are given two reviews.
Determine which review is more positive.
Review A: {key1}
Review B: {key2}
If Review A is more positive, select key 'A' as betterReviewKey. If Review B is more positive, select key 'B' as betterReviewKey.
"""

movie_external_comparison_prompt_template = """You are given a list of reviews with review id at the front.
Rank the reviews based on how much the reviewer liked the movie from worst to best.
Reviews:
{keys}
Output a JSON list of review ids from most negative to most positive.
"""





direct_inquiry_factual_knowledge_prompt = """Determine whether the ranking task depends on factual world knowledge that require web search.
Task description: {description}

If this task requires factual knowledge, return 'Yes'.
If this task is mainly subjective and requires reasoning, return 'No'.
Provide a brief explanation.
"""

llm_judge_prompt = """You are an EXPERT RANKED RESULT RATER. You are given a ranking criteria and a list of CANDIDATE RANKED RESULTS.
Each candidate ranking has an integer ID. Each ranking is ordered ascending from WORST to BEST (the last item is the best).
Your task is to select the candidate ranking that best sorts the input items according to the criteria.

Items to rank: {keys}

Ranking criteria: {criteria}

CANDIDATE RANKED RESULTS (worst to best):
{rankings}

Evaluate each candidate ranking against the criteria. For each candidate, assess whether its ordering correctly places the best items last and the worst items first.
Return the integer ID of the best candidate ranking with your reasoning.
Think step by step.
"""

web_search_system_prompt = (
    "You are a factual ranking assistant. "
    "Use the provided web search results as your primary source of information. "
    "If the web context does not contain the needed information, rely on your own knowledge to provide the best estimate. "
    "Never return 0 simply because the web context is unclear — always give your best estimate. "
    "Output a JSON object."
)
