from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_matches():
    path = RAW_DIR / "Matches.csv"
    df = pd.read_csv(path, parse_dates=["MatchDate"])
    return df


def load_elo():
    path = RAW_DIR / "EloRatings.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df
