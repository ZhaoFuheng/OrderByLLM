nba_pointwise_prompt_template = '''Given a nba players, key:{key}.
Determine the player height in meters as a floating-point number rounded to two decimal places
Rate the confidence in the proposed answer on a scale of 0-10.
'''

nba_external_pointwise_prompt_template = '''Given a list of nba players, keys:{keys}.
Determine their heights in meters as floating-point numbers rounded to two decimal places.
For each answer, also provide a confidence rating on a scale of 0-10.
'''

nba_pairwise_comparison_prompt_template = '''Which nba player is taller (in meters, rounded to two decimal places): {key1} or {key2}? 
Based on your knowledge and judgement, output the name of the taller player, 
In case of tie, choose the player whose name comes second alphabetically.
'''

nba_external_comparison_prompt_template = '''Given a list of nba players: {keys}
Based on your own knowledge, sort the players in ascending order based on player height (in meters, rounded to two decimal places) and name.
'''

passage_pointwise_prompt_template = """You are given a question and a passage.
Evaluate how well the passage answers the question by assigning a float relevance score from 0.0 to 3.0, where 0.0 is not relevant and 3.0 is prefectly relevant.
Also, rate the confidence in the proposed answer on a scale of 0-10.
Question: {question}
Passage: {key}
"""

passage_external_pointwise_prompt_template = """You are given a question and a list of passages. Evaluate how well each passage answers the question.
For each passage, assign a float relevance score from 0.0 to 3.0, where 0.0 is not relevant and 3.0 is prefectly relevant.
For each passage, also provide a confidence rating on a scale of 0-10.
Output a JSON list of float relevance scores and int confidences in the same order as the input passages.
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
Output a JSON list of passage_id in ranked order (from worst to best).
"""