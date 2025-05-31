import pandas as pd

def load_matches():
    path = "/Users/simonemezzabotta/Coding_Projects/GoalGenius/data/raw/Matches.csv"
    df = pd.read_csv(path, parse_dates=["MatchDate"])
    return df

def load_elo():
    path = "/Users/simonemezzabotta/Coding_Projects/GoalGenius/data/raw/EloRatings.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df