"""The feature table shared by the logistic regression and XGBoost."""

from __future__ import annotations

import pandas as pd

from goalline.constants import FIRST_DIVISIONS
from goalline.features.elo import EloParams, compute_elo
from goalline.features.form import compute_form

FEATURE_COLUMNS = [
    "elo_diff",
    "form_points_diff",
    "form_gd_diff",
    "short_history_home",
    "short_history_away",
    *[f"league_{d}" for d in FIRST_DIVISIONS],
]


def build_features(matches: pd.DataFrame, params: EloParams) -> pd.DataFrame:
    out = pd.concat([matches, compute_elo(matches, params), compute_form(matches)], axis=1)
    out["elo_diff"] = out.elo_home + params.home_advantage - out.elo_away
    out["form_points_diff"] = out.form_points_home - out.form_points_away
    out["form_gd_diff"] = out.form_gd_home - out.form_gd_away
    for division in FIRST_DIVISIONS:
        out[f"league_{division}"] = (out.division == division).astype(int)
    return out
