"""Gradient-boosted trees on the feature table, early-stopped on the last training season."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from goalline.constants import SEED
from goalline.features.table import FEATURE_COLUMNS
from goalline.models.base import first_division_rows, outcome_codes


class XGBoostModel:
    name = "xgboost"
    refit = "season"

    def __init__(
        self,
        max_depth: int,
        learning_rate: float,
        min_child_weight: float,
        train_first_season: int = 2002,
        seed: int = SEED,
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.train_first_season = train_first_season
        self.seed = seed

    def fit(self, history: pd.DataFrame) -> XGBoostModel:
        rows = first_division_rows(history, self.train_first_season)
        self.validation_season_ = int(rows.season.max())
        train = rows[rows.season < self.validation_season_]
        valid = rows[rows.season == self.validation_season_]
        self.train_seasons_ = (int(train.season.min()), int(train.season.max()))
        self.model_ = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=1000,
            early_stopping_rounds=50,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=0.8,
            tree_method="hist",
            random_state=self.seed,
            n_jobs=-1,
        )
        self.model_.fit(
            train[FEATURE_COLUMNS].to_numpy(dtype=float), outcome_codes(train),
            eval_set=[(valid[FEATURE_COLUMNS].to_numpy(dtype=float), outcome_codes(valid))],
            verbose=False,
        )
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(matches[FEATURE_COLUMNS].to_numpy(dtype=float))
