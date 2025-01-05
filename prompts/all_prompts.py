sort_int_prompt_template = '''Given a list of keys: {keys}
Sort the keys in ascending order.
'''


nba_pointwise_prompt_template = '''How tall is the nba player {key} in meters?
Output the player height in floating-point number.
'''

nba_external_pointwise_prompt_template = '''How tall are the following nba players {keys} in meters?
Output a JSON object with a single key containing a list of floating-point numbers, representing players' heights, and no explanation.
'''

nba_pairwise_comparison_prompt_template = '''Which nba player is taller: {key1} or {key2}? 
Output the name of the taller player, and no explanation.
'''

nba_external_comparison_prompt_template = '''Given a list of nba players: {keys}
Use your own knowledge to sort the players in ascending order based on heights.
Output a JSON object with a single key containing a list of players sorted in ascending order by height, and no explanation.
'''

