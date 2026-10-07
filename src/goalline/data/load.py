"""Read the raw file into the canonical match frame used by every later module."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from goalline.constants import COUNTRY, DIVISIONS, FIRST_DIVISIONS
from goalline.data.quality import validate
from goalline.settings import Settings

RAW_TO_CANONICAL = {
    "Division": "division",
    "MatchDate": "date",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHome": "home_goals",
    "FTAway": "away_goals",
    "FTResult": "result",
    "OddHome": "b365_h",
    "OddDraw": "b365_d",
    "OddAway": "b365_a",
    "MaxHome": "max_h",
    "MaxDraw": "max_d",
    "MaxAway": "max_a",
}

COLUMNS = [
    "match_id", "division", "country", "tier", "season", "date", "home", "away",
    "home_goals", "away_goals", "result",
    "b365_h", "b365_d", "b365_a", "max_h", "max_d", "max_a",
]


def read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=list(RAW_TO_CANONICAL),
        dtype={"Division": str, "HomeTeam": str, "AwayTeam": str, "FTResult": str},
    ).rename(columns=RAW_TO_CANONICAL)
    df["date"] = pd.to_datetime(df["date"])
    return df[df.division.isin(DIVISIONS)].reset_index(drop=True)


def season_of(dates: pd.Series) -> np.ndarray:
    return np.where(dates.dt.month >= 7, dates.dt.year, dates.dt.year - 1)


def season_label(season: int) -> str:
    return f"{season}-{(season + 1) % 100:02d}"


def make_match_id(division: str, date: pd.Timestamp, home: str, away: str) -> str:
    key = f"{division}|{date:%Y-%m-%d}|{home}|{away}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean, report = validate(raw)
    clean["season"] = season_of(clean.date)
    clean["tier"] = np.where(clean.division.isin(FIRST_DIVISIONS), 1, 2)
    clean["country"] = clean.division.map(COUNTRY)
    clean["match_id"] = [
        make_match_id(*row)
        for row in zip(clean.division, clean.date, clean.home, clean.away, strict=True)
    ]
    clean = clean.sort_values(["date", "division", "home"], kind="stable").reset_index(drop=True)
    return clean[COLUMNS], report


def load_matches(path: Path, settings: Settings, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mode not in ("develop", "final"):
        raise ValueError(f"mode must be 'develop' or 'final', not {mode!r}")
    frame, report = prepare(read_raw(path))
    keep = (frame.season >= settings.first_season) & (frame.season <= settings.last_season)
    if mode == "develop":
        keep &= frame.season < settings.final_first_season
    return frame[keep].reset_index(drop=True), report
