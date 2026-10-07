"""Development: tune on walk-forward folds up to 2020-21. Final: score the frozen models once."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from goalline.evaluation.metrics import log_loss
from goalline.evaluation.selection import Selected, build_benchmarks, build_models
from goalline.evaluation.walkforward import (
    FinalSeasonError,
    evaluation_rows,
    predict_all,
    walk_forward,
)
from goalline.features.elo import EloParams, compute_elo
from goalline.features.table import build_features
from goalline.models.base import outcome_codes
from goalline.models.dixon_coles import DixonColesModel
from goalline.models.elo_model import DrawParams, EloModel, elo_probabilities
from goalline.models.logistic import LogisticModel
from goalline.models.simple import Frequency, Uniform
from goalline.models.xgb import XGBoostModel
from goalline.settings import Settings


def mean_log_loss(preds: pd.DataFrame, frame: pd.DataFrame) -> float:
    merged = preds.merge(frame[["match_id", "result"]], on="match_id")
    p = merged[["p_home", "p_draw", "p_away"]].to_numpy()
    return float(log_loss(p, outcome_codes(merged)).mean())


def tune_elo(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if frame.season.max() >= settings.final_first_season:
        raise FinalSeasonError("tune_elo must run on a develop-mode frame")
    rows = evaluation_rows(frame, settings.dev_seasons)
    y = outcome_codes(rows)
    grid = []
    for k, h, r in product(settings.elo_k, settings.elo_h, settings.elo_r):
        elo = compute_elo(frame, EloParams(k, h, r)).loc[rows.index]
        diff = (elo.elo_home + h - elo.elo_away).to_numpy()
        for peak, scale in product(settings.draw_peak, settings.draw_scale):
            p = elo_probabilities(diff, DrawParams(peak, scale))
            grid.append({"k": k, "home_advantage": h, "reversion": r, "draw_peak": peak,
                         "draw_scale": scale, "log_loss": float(log_loss(p, y).mean())})
    return pd.DataFrame(grid)


def tune(frame, factory, candidates: list[dict], settings: Settings):
    rows, best = [], None
    for params in candidates:
        preds = walk_forward(frame, factory(**params), settings.dev_seasons, mode="develop",
                             settings=settings)
        score = mean_log_loss(preds, frame)
        rows.append({**params, "log_loss": score})
        if best is None or score < best[1]:
            best = (params, score, preds)
    return pd.DataFrame(rows), best[0], best[2]


@dataclass
class DevelopResult:
    selected: Selected
    predictions: pd.DataFrame
    grids: pd.DataFrame


def develop(frame: pd.DataFrame, settings: Settings, data_sha256: str) -> DevelopResult:
    elo_grid = tune_elo(frame, settings)
    top = elo_grid.loc[elo_grid.log_loss.idxmin()]
    elo = EloParams(float(top.k), float(top.home_advantage), float(top.reversion))
    draw = DrawParams(float(top.draw_peak), float(top.draw_scale))
    features = build_features(frame, elo)
    first = settings.train_first_season

    dc_grid, dc_best, dc_preds = tune(
        features, lambda xi: DixonColesModel(xi), [{"xi": x} for x in settings.dc_xi], settings
    )
    lr_grid, lr_best, lr_preds = tune(
        features, lambda c: LogisticModel(c, first), [{"c": c} for c in settings.logistic_c],
        settings,
    )
    xgb_candidates = [
        {"max_depth": d, "learning_rate": lr, "min_child_weight": w}
        for d, lr, w in product(settings.xgb_max_depth, settings.xgb_learning_rate,
                                settings.xgb_min_child_weight)
    ]
    xgb_grid, xgb_best, xgb_preds = tune(
        features, lambda **p: XGBoostModel(**p, train_first_season=first), xgb_candidates, settings
    )

    others = predict_all(
        features, [Uniform(), Frequency(first), EloModel(draw), *build_benchmarks()],
        settings.dev_seasons, mode="develop", settings=settings,
    )
    predictions = pd.concat([others, dc_preds, lr_preds, xgb_preds], ignore_index=True)
    selected = Selected(
        elo=elo, draw=draw, dc_xi=dc_best["xi"], logistic_c=lr_best["c"],
        xgb_max_depth=int(xgb_best["max_depth"]), xgb_learning_rate=xgb_best["learning_rate"],
        xgb_min_child_weight=xgb_best["min_child_weight"],
        dev_log_loss={
            "elo": float(top.log_loss),
            "dixon_coles": float(dc_grid.log_loss.min()),
            "logistic": float(lr_grid.log_loss.min()),
            "xgboost": float(xgb_grid.log_loss.min()),
        },
        data_sha256=data_sha256,
    )
    grids = pd.concat(
        [elo_grid.assign(model="elo"), dc_grid.assign(model="dixon_coles"),
         lr_grid.assign(model="logistic"), xgb_grid.assign(model="xgboost")],
        ignore_index=True,
    )
    if not np.isclose(mean_log_loss(predictions[predictions.model == "elo"], frame), top.log_loss):
        raise RuntimeError("Elo walk-forward predictions do not reproduce the Elo grid score")
    return DevelopResult(selected=selected, predictions=predictions, grids=grids)


@dataclass(frozen=True)
class FinalRun:
    predictions: pd.DataFrame
    dixon_coles_fits: int
    dixon_coles_nonconverged: int


def final_run(frame: pd.DataFrame, selected: Selected, settings: Settings) -> FinalRun:
    features = build_features(frame, selected.elo)
    models = build_models(selected, settings) + build_benchmarks()
    predictions = predict_all(
        features, models, settings.final_seasons, mode="final", settings=settings
    )
    dc = next(m for m in models if isinstance(m, DixonColesModel))
    return FinalRun(predictions, dc.fits_total, dc.fits_nonconverged)


def final_predictions(frame: pd.DataFrame, selected: Selected, settings: Settings) -> pd.DataFrame:
    return final_run(frame, selected, settings).predictions
