import numpy as np
import pytest

from goalline.evaluation.reliability import expected_calibration_error, reliability_table


def test_equal_count_bins_and_columns():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([2, 1, 1], size=1000)
    y = np.array([rng.choice(3, p=row) for row in p])
    table = reliability_table(p, y)
    assert list(table.columns) == ["outcome", "bin", "n", "mean_predicted", "observed"]
    assert set(table.outcome) == {"H", "D", "A"}
    assert (table.groupby("outcome").n.sum() == 1000).all()
    assert (table.n == 100).all()


def test_perfectly_calibrated_constant_forecast_has_zero_error():
    p = np.tile([0.5, 0.25, 0.25], (400, 1))
    y = np.array([0] * 200 + [1] * 100 + [2] * 100)
    ece = expected_calibration_error(reliability_table(p, y, bins=1))
    assert ece["H"] == pytest.approx(0)
    assert ece["D"] == pytest.approx(0)
