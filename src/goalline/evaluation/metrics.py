"""Per-match scoring rules for three-way forecasts (outcome order home, draw, away)."""

from __future__ import annotations

import numpy as np


def _one_hot(y: np.ndarray) -> np.ndarray:
    return np.eye(3)[y]


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0))


def rps(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    cumulative_p = np.cumsum(p, axis=1)[:, :2]
    cumulative_o = np.cumsum(_one_hot(y), axis=1)[:, :2]
    return ((cumulative_p - cumulative_o) ** 2).sum(axis=1) / 2


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((p - _one_hot(y)) ** 2).sum(axis=1)
