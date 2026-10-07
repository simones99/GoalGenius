"""Paired bootstrap of per-match loss differences, resampling whole matchdays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from goalline.constants import SEED


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    lower: float
    upper: float
    n_matches: int
    n_clusters: int


def matchday_clusters(frame: pd.DataFrame) -> np.ndarray:
    iso = frame.date.dt.isocalendar()
    return (
        frame.division.astype(str) + "-" + iso.year.astype(str) + "-" + iso.week.astype(str)
    ).to_numpy()


def bootstrap_replicates(
    diff: np.ndarray, clusters: np.ndarray, replicates: int, seed: int = SEED
) -> np.ndarray:
    codes, uniques = pd.factorize(clusters)
    sums = np.bincount(codes, weights=diff)
    counts = np.bincount(codes)
    rng = np.random.default_rng(seed)
    out = np.empty(replicates)
    for start in range(0, replicates, 200):
        stop = min(start + 200, replicates)
        idx = rng.integers(0, len(uniques), size=(stop - start, len(uniques)))
        out[start:stop] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return out


def paired_bootstrap(
    loss_model: np.ndarray,
    loss_reference: np.ndarray,
    clusters: np.ndarray,
    replicates: int = 2000,
    seed: int = SEED,
) -> BootstrapResult:
    diff = np.asarray(loss_model, dtype=float) - np.asarray(loss_reference, dtype=float)
    stats = bootstrap_replicates(diff, clusters, replicates, seed)
    lower, upper = np.percentile(stats, [2.5, 97.5])
    return BootstrapResult(
        mean=float(diff.mean()), lower=float(lower), upper=float(upper),
        n_matches=len(diff), n_clusters=len(pd.unique(clusters)),
    )
