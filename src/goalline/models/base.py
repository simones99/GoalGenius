"""The interface every model and benchmark follows, and helpers shared by the learned models."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from goalline.constants import OUTCOME_INDEX


class Model(Protocol):
    name: str
    refit: str  # "season" or "month"

    def fit(self, history: pd.DataFrame) -> Model: ...

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray: ...


def outcome_codes(frame: pd.DataFrame) -> np.ndarray:
    return frame.result.map(OUTCOME_INDEX).to_numpy(dtype=int)


def first_division_rows(history: pd.DataFrame, first_season: int) -> pd.DataFrame:
    return history[(history.tier == 1) & (history.season >= first_season)]
