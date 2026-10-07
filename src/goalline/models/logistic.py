"""Multinomial logistic regression on the standardised feature table."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from goalline.features.table import FEATURE_COLUMNS
from goalline.models.base import first_division_rows, outcome_codes


class LogisticModel:
    name = "logistic"
    refit = "season"

    def __init__(self, c: float, train_first_season: int = 2002):
        self.c = c
        self.train_first_season = train_first_season

    def fit(self, history: pd.DataFrame) -> LogisticModel:
        rows = first_division_rows(history, self.train_first_season)
        self.n_train_ = len(rows)
        self.pipeline_ = make_pipeline(
            StandardScaler(), LogisticRegression(C=self.c, max_iter=2000)
        ).fit(rows[FEATURE_COLUMNS].to_numpy(dtype=float), outcome_codes(rows))
        if list(self.pipeline_.classes_) != [0, 1, 2]:
            raise ValueError("training data must contain home wins, draws and away wins")
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(matches[FEATURE_COLUMNS].to_numpy(dtype=float))
