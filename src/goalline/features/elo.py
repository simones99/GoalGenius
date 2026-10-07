"""Elo ratings over all ten divisions, updated match by match, reported before each match."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

INITIAL_RATING = {1: 1500.0, 2: 1350.0}


@dataclass(frozen=True)
class EloParams:
    k: float
    home_advantage: float
    reversion: float


def expected_home(r_home, r_away, home_advantage):
    return 1.0 / (1.0 + 10.0 ** ((r_away - (r_home + home_advantage)) / 400.0))


def compute_elo(matches: pd.DataFrame, params: EloParams) -> pd.DataFrame:
    m = matches.sort_values("date", kind="stable")
    ratings: dict[str, float] = {}
    last: dict[str, tuple[int, str]] = {}
    snapshots: dict[tuple[int, str], float] = {}
    current_season = None
    out_home = np.empty(len(m))
    out_away = np.empty(len(m))

    rows = zip(
        m.season.to_numpy(), m.division.to_numpy(), m.tier.to_numpy(),
        m.home.to_numpy(), m.away.to_numpy(),
        m.home_goals.to_numpy(), m.away_goals.to_numpy(),
        strict=True,
    )
    for i, (season, division, tier, home, away, hg, ag) in enumerate(rows):
        if season != current_season:
            if current_season is not None:
                by_division: dict[str, list[float]] = defaultdict(list)
                for team, (s, d) in last.items():
                    if s == current_season:
                        by_division[d].append(ratings[team])
                for d, values in by_division.items():
                    snapshots[(current_season, d)] = float(np.mean(values))
            current_season = season
        for team in (home, away):
            if team not in ratings:
                ratings[team] = INITIAL_RATING[int(tier)]
            else:
                s, d = last[team]
                if s != season:
                    ratings[team] += params.reversion * (snapshots[(s, d)] - ratings[team])
            last[team] = (season, division)
        r_home, r_away = ratings[home], ratings[away]
        out_home[i], out_away[i] = r_home, r_away
        score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        delta = params.k * (score - expected_home(r_home, r_away, params.home_advantage))
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    result = pd.DataFrame({"elo_home": out_home, "elo_away": out_away}, index=m.index)
    return result.loc[matches.index]
