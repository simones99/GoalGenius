"""Walk-forward prediction: every block is predicted from matches played before it started."""

from __future__ import annotations

import numpy as np
import pandas as pd

from goalline.benchmark import Benchmark
from goalline.constants import ODDS_COLUMNS
from goalline.settings import Settings


class FinalSeasonError(RuntimeError):
    pass


def evaluation_rows(frame: pd.DataFrame, seasons) -> pd.DataFrame:
    has_odds = frame[list(ODDS_COLUMNS["b365"])].notna().all(axis=1)
    return frame[(frame.tier == 1) & frame.season.isin(seasons) & has_odds]


def blocks(rows: pd.DataFrame, refit: str) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    if refit == "season":
        return [(pd.Timestamp(f"{s}-07-01"), g) for s, g in rows.groupby("season")]
    if refit == "month":
        months = rows.date.dt.to_period("M")
        return [(period.start_time, g) for period, g in rows.groupby(months)]
    raise ValueError(f"unknown refit {refit!r}")


def walk_forward(frame, model, seasons, *, mode: str, settings: Settings) -> pd.DataFrame:
    if mode not in ("develop", "final"):
        raise ValueError(f"mode must be 'develop' or 'final', got {mode!r}")
    if mode == "develop" and max(seasons) >= settings.final_first_season:
        raise FinalSeasonError(
            f"develop mode cannot score season {max(seasons)}: "
            f"the final test starts in {settings.final_first_season}"
        )
    parts = []
    for start, block in blocks(evaluation_rows(frame, seasons), model.refit):
        p = model.fit(frame[frame.date < start]).predict_proba(block)
        parts.append(
            pd.DataFrame(
                {"match_id": block.match_id.to_numpy(), "model": model.name,
                 "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2]}
            )
        )
    out = pd.concat(parts, ignore_index=True)
    finite = np.isfinite(out[["p_home", "p_draw", "p_away"]].to_numpy()).all(axis=1)
    if not finite.all():
        # Only a benchmark may legitimately fail to price a match (missing odds).
        if not isinstance(model, Benchmark):
            raise ValueError(f"{model.name} returned {(~finite).sum()} non-finite probability rows")
        out = out[finite]
    return out.reset_index(drop=True)


def predict_all(frame, models, seasons, *, mode: str, settings: Settings) -> pd.DataFrame:
    return pd.concat(
        [walk_forward(frame, m, seasons, mode=mode, settings=settings) for m in models],
        ignore_index=True,
    )
