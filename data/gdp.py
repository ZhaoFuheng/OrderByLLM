import pandas as pd
import pycountry

# Read CSV
df = pd.read_csv("sorted_economy_data.csv")

# Extract country from "country_year"
countries_only = df['country_year'].str.rsplit(' ', n=1).str[0]

# Get list of all official country names
valid_country_names = [c.name for c in pycountry.countries]

def is_country(name):
    """
    Check if the name matches a real country (case-insensitive exact match).
    """
    return any(name.lower() == valid.lower() for valid in valid_country_names)

# Filter only real countries
df_filtered = df[countries_only.apply(is_country)]

# Wrap each country_year in parentheses
df_filtered['country_year'] = "(" + df_filtered['country_year'] + ")"

# Sample 200 rows (or all rows if fewer than 200)
df_sampled = df_filtered.sample(n=min(200, len(df_filtered)), random_state=42)

# Save filtered sample
df_sampled.to_csv("filtered_countries_sample.csv", index=False)

print(f"Saved {len(df_sampled)} sampled rows to filtered_countries_sample.csv")

