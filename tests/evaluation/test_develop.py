import numpy as np
import pytest

from goalline.constants import MODEL_NAMES, REFERENCES
from goalline.data.load import load_matches
from goalline.evaluation.develop import (
    develop,
    final_predictions,
    final_run,
    mean_log_loss,
    tune,
    tune_elo,
)
from goalline.evaluation.walkforward import FinalSeasonError
from goalline.features.elo import EloParams
from goalline.features.table import build_features
from goalline.models.logistic import LogisticModel
from helpers import FIXTURE, SMOKE


@pytest.fixture(scope="module")
def dev_frame():
    frame, _ = load_matches(FIXTURE, SMOKE, mode="develop")
    return frame


def test_elo_grid_covers_every_combination(dev_frame):
    grid = tune_elo(dev_frame, SMOKE)
    assert len(grid) == 1 * 1 * 2 * 2 * 1
    assert list(grid.columns) == ["k", "home_advantage", "reversion", "draw_peak", "draw_scale",
                                  "log_loss"]


@pytest.fixture(scope="module")
def result(dev_frame):
    return develop(dev_frame, SMOKE, data_sha256="sha")


def test_develop_selects_from_the_grids(result):
    s = result.selected
    assert s.elo.reversion in SMOKE.elo_r
    assert s.draw.peak in SMOKE.draw_peak
    assert s.dc_xi in SMOKE.dc_xi
    assert s.logistic_c in SMOKE.logistic_c
    assert s.data_sha256 == "sha"
    assert set(s.dev_log_loss) == {"elo", "dixon_coles", "logistic", "xgboost"}


def test_develop_predictions_cover_dev_seasons_only(result, dev_frame):
    preds = result.predictions.merge(dev_frame[["match_id", "season"]], on="match_id")
    assert set(preds.season) == set(SMOKE.dev_seasons)
    assert set(preds.model) == set(MODEL_NAMES) | set(REFERENCES)
    assert set(result.grids.model) == {"elo", "dixon_coles", "logistic", "xgboost"}


def test_final_predictions_cover_final_seasons(result):
    frame, _ = load_matches(FIXTURE, SMOKE, mode="final")
    preds = final_predictions(frame, result.selected, SMOKE)
    seasons = set(preds.merge(frame[["match_id", "season"]], on="match_id").season)
    assert seasons == set(SMOKE.final_seasons)


def test_final_run_reports_dixon_coles_convergence(result):
    frame, _ = load_matches(FIXTURE, SMOKE, mode="final")
    run = final_run(frame, result.selected, SMOKE)
    assert run.dixon_coles_nonconverged == 0
    assert run.dixon_coles_fits > 0
    assert set(run.predictions.model) == set(MODEL_NAMES) | set(REFERENCES)


def test_tune_picks_the_argmin_and_its_predictions(dev_frame):
    features = build_features(dev_frame, EloParams(20, 60, 0.33))
    candidates = [{"c": 0.001}, {"c": 1.0}, {"c": 100.0}]
    grid, best, best_preds = tune(
        features, lambda c: LogisticModel(c, SMOKE.train_first_season), candidates, SMOKE
    )
    assert grid.log_loss.nunique() > 1
    assert best["c"] == grid.loc[grid.log_loss.idxmin(), "c"]
    assert np.isclose(mean_log_loss(best_preds, dev_frame), grid.log_loss.min())


def test_dev_predictions_match_selected_scores(result, dev_frame):
    for m in ("elo", "dixon_coles", "logistic", "xgboost"):
        preds = result.predictions[result.predictions.model == m]
        assert np.isclose(mean_log_loss(preds, dev_frame), result.selected.dev_log_loss[m])


def test_selected_elo_is_the_grid_argmin(result, dev_frame):
    grid = tune_elo(dev_frame, SMOKE)
    top = grid.loc[grid.log_loss.idxmin()]
    s = result.selected
    assert (s.elo.k, s.elo.home_advantage, s.elo.reversion) == (
        top.k, top.home_advantage, top.reversion)
    assert (s.draw.peak, s.draw.scale) == (top.draw_peak, top.draw_scale)


def test_develop_refuses_a_frame_with_final_seasons():
    frame, _ = load_matches(FIXTURE, SMOKE, mode="final")
    with pytest.raises(FinalSeasonError):
        develop(frame, SMOKE, "sha")
