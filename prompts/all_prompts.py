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

