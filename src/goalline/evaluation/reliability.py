"""Reliability tables: predicted probability against observed frequency, per outcome."""

from __future__ import annotations

import numpy as np
import pandas as pd

from goalline.constants import OUTCOMES


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    rows = []
    for k, outcome in enumerate(OUTCOMES):
        order = np.argsort(p[:, k], kind="stable")
        for b, idx in enumerate(np.array_split(order, bins)):
            rows.append(
                {
                    "outcome": outcome,
                    "bin": b,
                    "n": len(idx),
                    "mean_predicted": float(p[idx, k].mean()),
                    "observed": float((y[idx] == k).mean()),
                }
            )
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> pd.Series:
    gap = (table.mean_predicted - table.observed).abs() * table.n
    return gap.groupby(table.outcome).sum() / table.n.groupby(table.outcome).sum()
