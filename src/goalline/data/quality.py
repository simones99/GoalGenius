"""Quality checks. Bad matches are dropped, bad odds are blanked, and everything is counted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from goalline.constants import ODDS_COLUMNS

OVERROUND_RANGE = (1.0, 1.25)


def validate(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    log: list[dict] = []

    def record(check: str, mask) -> None:
        mask = np.asarray(mask, dtype=bool)
        for division, n in df.loc[mask, "division"].value_counts().items():
            log.append({"check": check, "division": division, "count": int(n)})

    no_result = df.home_goals.isna() | df.away_goals.isna()
    record("no_result", no_result)
    df = df[~no_result]

    hg, ag = df.home_goals, df.away_goals
    expected = np.where(hg > ag, "H", np.where(hg < ag, "A", "D"))
    invalid = (hg < 0) | (ag < 0) | (hg % 1 != 0) | (ag % 1 != 0) | df.result.ne(expected)
    record("invalid_goals_or_result", invalid)
    df = df[~invalid]

    duplicate = df.duplicated(["division", "date", "home", "away"], keep="first")
    record("duplicate", duplicate)
    df = df[~duplicate]

    home_rows = df[["date", "home"]].rename(columns={"home": "team"})
    away_rows = df[["date", "away"]].rename(columns={"away": "team"})
    appearances = pd.DataFrame({
        "date": pd.concat([home_rows["date"], away_rows["date"]], ignore_index=True),
        "team": pd.concat([home_rows["team"], away_rows["team"]], ignore_index=True),
    })
    per_day = appearances.groupby(["date", "team"]).size()
    clashes = per_day[per_day > 1].index
    twice = pd.MultiIndex.from_arrays([df.date, df.home]).isin(clashes) | pd.MultiIndex.from_arrays(
        [df.date, df.away]
    ).isin(clashes)
    record("team_twice_same_date", twice)
    df = df[~twice].copy()

    for name, columns in ODDS_COLUMNS.items():
        odds = df[list(columns)]
        complete = odds.notna().all(axis=1)
        partial = odds.notna().any(axis=1) & ~complete
        not_above_one = complete & (odds <= 1).any(axis=1)
        blank = partial | not_above_one
        if name == "b365":
            overround = (1 / odds).sum(axis=1)
            low, high = OVERROUND_RANGE
            blank |= complete & ~not_above_one & ((overround < low) | (overround > high))
        record(f"{name}_odds_blanked", blank)
        df.loc[blank, list(columns)] = np.nan

    df["home_goals"] = df.home_goals.astype(int)
    df["away_goals"] = df.away_goals.astype(int)
    report = pd.DataFrame(log, columns=["check", "division", "count"])
    return df.reset_index(drop=True), report
