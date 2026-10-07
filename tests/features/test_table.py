import numpy as np
import pandas as pd

from goalline.features.elo import EloParams
from goalline.features.table import FEATURE_COLUMNS, build_features

PARAMS = EloParams(20, 60, 0.33)


def test_feature_columns_present(features):
    assert set(FEATURE_COLUMNS) <= set(features.columns)
    assert np.allclose(features.elo_diff, features.elo_home + 60 - features.elo_away)
    league = features[[c for c in FEATURE_COLUMNS if c.startswith("league_")]].sum(axis=1)
    assert (league[features.tier == 1] == 1).all()
    assert (league[features.tier == 2] == 0).all()


def test_perturbing_a_result_leaves_earlier_features_unchanged(matches):
    i = len(matches) // 2
    full = build_features(matches, PARAMS)
    changed = matches.copy()
    hg, ag = changed.loc[i, ["home_goals", "away_goals"]]
    changed.loc[i, ["home_goals", "away_goals"]] = (ag, hg) if hg != ag else (hg + 3, ag)
    perturbed = build_features(changed, PARAMS)
    upto = matches.date <= matches.loc[i, "date"]
    cols = FEATURE_COLUMNS + ["elo_home", "elo_away"]
    pd.testing.assert_frame_equal(full.loc[upto, cols], perturbed.loc[upto, cols])
    assert not full.loc[~upto, cols].equals(perturbed.loc[~upto, cols])


def test_truncating_the_data_after_a_date_changes_nothing_on_that_date(matches):
    d = matches.loc[len(matches) // 2, "date"]
    full = build_features(matches, PARAMS)
    part = build_features(matches[matches.date <= d], PARAMS)
    on_day = matches.date == d
    cols = FEATURE_COLUMNS + ["elo_home", "elo_away"]
    pd.testing.assert_frame_equal(full.loc[on_day, cols], part.loc[on_day, cols])
