"""Naive baselines: a uniform guess and the outcome shares seen in training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from goalline.models.base import first_division_rows, outcome_codes


class Uniform:
    name = "uniform"
    refit = "season"

    def fit(self, history: pd.DataFrame) -> Uniform:
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        return np.full((len(matches), 3), 1 / 3)


def _shares(codes: np.ndarray) -> np.ndarray:
    return (np.bincount(codes, minlength=3) + 1) / (len(codes) + 3)


class Frequency:
    name = "frequency"
    refit = "season"

    def __init__(self, train_first_season: int = 2002):
        self.train_first_season = train_first_season

    def fit(self, history: pd.DataFrame) -> Frequency:
        rows = first_division_rows(history, self.train_first_season)
        self.overall_ = _shares(outcome_codes(rows))
        self.by_division_ = {d: _shares(outcome_codes(g)) for d, g in rows.groupby("division")}
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        return np.array([self.by_division_.get(d, self.overall_) for d in matches.division])
