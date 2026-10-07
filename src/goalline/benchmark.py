"""Turn bookmaker odds into probabilities: Shin's method and proportional normalisation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from goalline.constants import ODDS_COLUMNS


def proportional(odds: np.ndarray) -> np.ndarray:
    pi = 1.0 / odds
    return pi / pi.sum(axis=1, keepdims=True)


def shin(
    odds: np.ndarray, tol: float = 1e-13, max_iter: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Shin probabilities p_i(z) = (sqrt(z² + 4(1−z)π_i²/β) − z) / (2(1−z)), with Σ p_i(z) = 1.

    Σ p_i(0) = sqrt(β) ≥ 1 and the sum decreases in z, so z is found by bisection on [0, 1).
    """
    pi = 1.0 / odds
    beta = pi.sum(axis=1, keepdims=True)

    def probs_at(z: np.ndarray) -> np.ndarray:
        z = z[:, None]
        return (np.sqrt(z**2 + 4 * (1 - z) * pi**2 / beta) - z) / (2 * (1 - z))

    lo = np.zeros(len(pi))
    hi = np.full(len(pi), 0.999)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        too_big = probs_at(mid).sum(axis=1) > 1
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
        if np.max(hi - lo) < tol:
            break
    z = np.where(beta[:, 0] <= 1, 0.0, (lo + hi) / 2)
    p = probs_at(z)
    return p / p.sum(axis=1, keepdims=True), z


class Benchmark:
    refit = "season"

    def __init__(self, method: str):
        if method not in ("shin", "proportional", "max_odds"):
            raise ValueError(f"unknown benchmark method {method!r}")
        self.method = method
        self.name = method

    def fit(self, history: pd.DataFrame) -> Benchmark:
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        columns = ODDS_COLUMNS["max" if self.method == "max_odds" else "b365"]
        odds = matches[list(columns)].to_numpy(dtype=float)
        out = np.full((len(odds), 3), np.nan)
        ok = ~np.isnan(odds).any(axis=1)
        if ok.any():
            out[ok] = shin(odds[ok])[0] if self.method == "shin" else proportional(odds[ok])
        return out
