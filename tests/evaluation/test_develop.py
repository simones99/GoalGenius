import pytest

from goalline.constants import MODEL_NAMES, REFERENCES
from goalline.data.load import load_matches
from goalline.evaluation.develop import develop, final_predictions, tune_elo
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
