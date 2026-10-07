"""Elo expected score split into three outcomes by a Gaussian draw curve on the rating gap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DrawParams:
    peak: float
    scale: float


def elo_probabilities(elo_diff: np.ndarray, draw: DrawParams) -> np.ndarray:
    expected = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    p_draw = draw.peak * np.exp(-0.5 * (elo_diff / draw.scale) ** 2)
    return np.column_stack([expected * (1 - p_draw), p_draw, (1 - expected) * (1 - p_draw)])


class EloModel:
    name = "elo"
    refit = "season"

    def __init__(self, draw: DrawParams):
        self.draw = draw

    def fit(self, history: pd.DataFrame) -> EloModel:
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        return elo_probabilities(matches.elo_diff.to_numpy(dtype=float), self.draw)
