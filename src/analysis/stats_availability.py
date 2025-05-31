import pandas as pd
import numpy as np

# Load matches data
print("Loading data...")
df = pd.read_csv('/Users/simonemezzabotta/Coding_Projects/GoalGenius/data/raw/Matches.csv', parse_dates=['MatchDate'])

# Check statistics availability by year
stats_cols = ['HomeShots', 'AwayShots', 'HomeTarget', 'AwayTarget', 
              'HomeFouls', 'AwayFouls', 'HomeCorners', 'AwayCorners']

df['Year'] = df['MatchDate'].dt.year
stats_availability = df.groupby('Year')[stats_cols].count() / len(df.groupby('Year'))

print("\nStatistics availability by year:")
print(stats_availability.mean(axis=1))

# Find first year with complete statistics
complete_years = stats_availability[stats_availability.mean(axis=1) > 0.95].index
if len(complete_years) > 0:
    print(f"\nFirst year with complete statistics: {complete_years[0]}")
    complete_data = df[df['Year'] >= complete_years[0]]
    print(f"Number of matches from {complete_years[0]}: {len(complete_data)}")
    print(f"Total number of matches: {len(df)}")
    
    # Print some sample statistics
    print("\nSample statistics from recent matches:")
    recent = df.sort_values('MatchDate', ascending=False).head(5)
    for _, match in recent.iterrows():
        print(f"\n{match['HomeTeam']} vs {match['AwayTeam']} ({match['MatchDate'].date()}):")
        print(f"Shots: {match['HomeShots']}-{match['AwayShots']}")
        print(f"Corners: {match['HomeCorners']}-{match['AwayCorners']}")
        print(f"Score: {match['FTHome']}-{match['FTAway']}")
