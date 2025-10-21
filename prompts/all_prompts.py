nba_pointwise_prompt_template = '''Given a nba players, key:{key}.
Determine the player height in meters as a floating-point number rounded to two decimal places
Rate the confidence in the proposed answer on a scale of 0-10.
'''

nba_external_pointwise_prompt_template = '''Given a list of nba players, keys:{keys}.
Determine their heights in meters as floating-point numbers rounded to two decimal places.
For each answer, also provide a confidence rating on a scale of 0-10.
'''

nba_pairwise_comparison_prompt_template = '''Which nba player is taller (in meters, rounded to two decimal places): {key1} or {key2}? 
Based on your knowledge and judgement, output the name of the taller player. Do not rephrase or change the given player name.
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
Output a JSON list of passage_id from worst passage to best passage.
"""



population_pointwise_prompt_template = '''Given the following country: {key}, 
what was its population in 2020, expressed as a decimal in millions?
Rate the confidence in the proposed answer on a scale of 0-10.
'''

population_external_pointwise_prompt_template = '''Given a list of countries, keys: {keys},
what was the population of each country in 2020, expressed as a decimal in millions?
For each answer, also provide a confidence rating on a scale of 0-10.
'''

population_pairwise_comparison_prompt_template = '''Which country had a larger population in 2020: {key1} or {key2}?
Based on your knowledge, output the name of the input country with the larger population. Do not rephrase or change the country name.
In the event of a tie, choose the country whose name appears second alphabetically.
'''

population_external_comparison_prompt_template = '''Given a list of countries: {keys},
sort the countries in ascending order based on their 2020 population. In case of a tie, sort by country name in alphabetical order.
'''


sql_pointwise_prompt_template = """You are given a database schema, a natural language question, and a candidate SQL query.
Your task is to evaluate how well the SQL query answers the question, ensuring consistency with the provided database schema.
Assign a correctness score between 0.0 and 1.0, where 0.0 is completely incorrect and 1.0 is perfectly correct.
Also, rate the confidence in the proposed answer on a scale of 0-10.
Database Schema: {schema}
Question: {question}
SQL: {key}
"""

sql_external_pointwise_prompt_template = """You are given a database schema, a natural language question, and a list of candidate SQL queries.
Evaluate how well each SQL query answers the question, ensuring consistency with the provided database schema.
For each SQL query, assign a correctness score between 0.0 and 1.0, where 0.0 is completely incorrect and 1.0 is perfectly correct.
For each SQL query, also provide a confidence rating on a scale of 0-10.
Output a JSON list of float correctness scores and int confidences in the same order as the input SQL queries.
Database Schema: {schema}
Question: {question}
SQL queries: 
{keys}
"""

sql_pairwise_comparison_prompt_template = """You are given a database schema, a natural language question, and two candidate SQL queries.
Determine which SQL query answers the question better, ensuring consistency with the provided database schema.
Database Schema: {schema}
Question: {question}
SQL A: {key1}
SQL B: {key2}
If SQL A is better, select key 'A' as BetterKey. If SQL B is better, select key 'B' as BetterKey.
"""

sql_external_comparison_prompt_template = """You are given a database schema, a natural language question, and a list of candidate SQL queries with sql_id at the front.
Rank the SQL queries based on how well they answer the question from worst to best, ensuring consistency with the provided database schema.
Question: {question}
SQL queries:
{keys}
Output a JSON list of sql_id from worst to best.
"""



tweetEval_pointwise_prompt_template = """You are given a passage.
Assign a float score from 0.0 to 1.0, where 0.0 is not {criteria} and 1.0 is completely {criteria}.
Also, rate the int confidence on a scale of 0-10.
Passage: {key}
"""

tweetEval_external_pointwise_prompt_template = """You are given a list of passages.
For each passage, assign a float score from 0.0 to 1.0, where 0.0 is not {criteria} and 1.0 is completely {criteria}.
For each passage, also provide a confidence rating on a scale of 0-10.
Output a JSON list of float {criteria} relevance scores and int confidences scores in the same order as the input passages.
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

membership_inference_prompt = """You have seen the following {description}: "{key}"?
Answer only with True or False."""