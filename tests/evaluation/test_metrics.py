import numpy as np
import pytest

from goalline.evaluation.metrics import brier, log_loss, rps

H = np.array([0])


def test_perfect_forecast_scores_zero():
    p = np.array([[1.0, 0.0, 0.0]])
    assert log_loss(p, H)[0] == pytest.approx(0)
    assert rps(p, H)[0] == pytest.approx(0)
    assert brier(p, H)[0] == pytest.approx(0)


def test_uniform_log_loss_is_ln3():
    assert log_loss(np.full((1, 3), 1 / 3), H)[0] == pytest.approx(np.log(3))


@pytest.mark.parametrize(
    ("forecast", "expected"),
    [([0.9, 0.1, 0.0], 0.005), ([0.8, 0.1, 0.1], 0.025), ([0.5, 0.25, 0.25], 0.15625),
     ([0.35, 0.3, 0.35], 0.2725)],
)
def test_rps_hand_values_for_a_home_win(forecast, expected):
    """RPS = 1/(r−1) Σ (cumulative forecast − cumulative outcome)².

    As in Constantinou & Fenton (2012).
    """
    assert rps(np.array([forecast]), H)[0] == pytest.approx(expected)


def test_rps_rewards_the_nearer_outcome():
    draw_heavy = np.array([[0.2, 0.6, 0.2]])
    away_heavy = np.array([[0.2, 0.2, 0.6]])
    assert rps(draw_heavy, H)[0] < rps(away_heavy, H)[0]


def test_brier_sums_over_outcomes():
    assert brier(np.array([[0.5, 0.25, 0.25]]), H)[0] == pytest.approx(0.25 + 0.0625 + 0.0625)


def test_log_loss_clips_zero_probabilities():
    assert np.isfinite(log_loss(np.array([[0.0, 0.5, 0.5]]), H)[0])
