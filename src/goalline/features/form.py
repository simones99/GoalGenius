"""Recent form: mean points and goal difference over each team's previous five matches."""

from __future__ import annotations

import numpy as np
import pandas as pd

FORM_WINDOW = 5
DEFAULT_POINTS = 4 / 3


def _previous_mean(s: pd.Series) -> pd.Series:
    return s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()


def compute_form(matches: pd.DataFrame) -> pd.DataFrame:
    hg = matches.home_goals.to_numpy()
    ag = matches.away_goals.to_numpy()
    home_points = np.select([hg > ag, hg == ag], [3, 1], 0)
    away_points = np.select([hg < ag, hg == ag], [3, 1], 0)
    n = len(matches)
    long = pd.DataFrame(
        {
            "row": np.concatenate([matches.index, matches.index]),
            "side": ["home"] * n + ["away"] * n,
            "date": np.concatenate([matches.date, matches.date]),
            "division": np.concatenate([matches.division, matches.division]),
            "team": np.concatenate([matches.home, matches.away]),
            "points": np.concatenate([home_points, away_points]),
            "gd": np.concatenate([hg - ag, ag - hg]),
        }
    ).sort_values(["team", "date"], kind="stable")
    by_team = long.groupby("team", sort=False)
    long["prev_points"] = by_team["points"].transform(_previous_mean)
    long["prev_gd"] = by_team["gd"].transform(_previous_mean)
    long["n_prev"] = by_team.cumcount()

    daily = (
        long.groupby(["division", "date"])
        .agg(points=("points", "sum"), n=("points", "size"))
        .reset_index()
        .sort_values(["division", "date"])
    )
    by_division = daily.groupby("division")
    cum_points = by_division["points"].cumsum() - daily.points
    cum_n = by_division["n"].cumsum() - daily.n
    daily["division_mean"] = (cum_points / cum_n.replace(0, np.nan)).fillna(DEFAULT_POINTS)
    long = long.merge(daily[["division", "date", "division_mean"]], on=["division", "date"])

    long["form_points"] = long.prev_points.fillna(long.division_mean)
    long["form_gd"] = long.prev_gd.fillna(0.0)
    long["short_history"] = (long.n_prev < FORM_WINDOW).astype(int)

    out = pd.DataFrame(index=matches.index)
    for side in ("home", "away"):
        part = long[long.side == side].set_index("row")
        out[f"form_points_{side}"] = part.form_points
        out[f"form_gd_{side}"] = part.form_gd
        out[f"short_history_{side}"] = part.short_history
    return out[
        ["form_points_home", "form_points_away", "form_gd_home", "form_gd_away",
         "short_history_home", "short_history_away"]
    ]
