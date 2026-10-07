import numpy as np
import pandas as pd
import pytest

from goalline.benchmark import Benchmark
from goalline.evaluation.walkforward import FinalSeasonError, evaluation_rows, walk_forward
from goalline.models.elo_model import DrawParams, EloModel
from goalline.models.simple import Uniform
from helpers import SMOKE


class Spy:
    name = "spy"

    def __init__(self, refit):
        self.refit = refit
        self.calls = []

    def fit(self, history):
        self.history_end = history.date.max()
        return self

    def predict_proba(self, matches):
        self.calls.append((self.history_end, matches.date.min(), matches.date.max()))
        return np.full((len(matches), 3), 1 / 3)


def test_develop_mode_refuses_test_seasons(features):
    with pytest.raises(FinalSeasonError):
        walk_forward(features, Uniform(), (2003, 2004), mode="develop", settings=SMOKE)


def test_evaluation_rows_are_first_division_with_odds(features):
    rows = evaluation_rows(features, (2004,))
    assert set(rows.tier) == {1} and set(rows.season) == {2004}
    assert rows[["b365_h", "b365_d", "b365_a"]].notna().all(axis=None)


def test_season_refit_sees_only_previous_seasons(features):
    spy = Spy("season")
    walk_forward(features, spy, (2004, 2005), mode="final", settings=SMOKE)
    assert len(spy.calls) == 2
    for history_end, first, _ in spy.calls:
        season = first.year if first.month >= 7 else first.year - 1
        assert history_end < pd.Timestamp(f"{season}-07-01")


def test_month_refit_sees_only_earlier_months(features):
    spy = Spy("month")
    walk_forward(features, spy, (2004,), mode="final", settings=SMOKE)
    assert len(spy.calls) > 2
    for history_end, first, last in spy.calls:
        assert (first.year, first.month) == (last.year, last.month)
        assert history_end < first.replace(day=1)


def test_output_columns_and_rows(features):
    out = walk_forward(features, EloModel(DrawParams(0.28, 400)), (2004,), mode="final",
                       settings=SMOKE)
    assert list(out.columns) == ["match_id", "model", "p_home", "p_draw", "p_away"]
    assert set(out.model) == {"elo"}
    assert len(out) == len(evaluation_rows(features, (2004,)))


def test_rows_without_probabilities_are_dropped(features):
    frame = features.copy()
    target = evaluation_rows(frame, (2004,)).index[0]
    frame.loc[target, ["max_h", "max_d", "max_a"]] = np.nan
    out = walk_forward(frame, Benchmark("max_odds"), (2004,), mode="final", settings=SMOKE)
    assert len(out) == len(evaluation_rows(frame, (2004,))) - 1
    assert frame.loc[target, "match_id"] not in set(out.match_id)
