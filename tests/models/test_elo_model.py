import numpy as np
import pandas as pd

from goalline.models.elo_model import DrawParams, EloModel, elo_probabilities


def test_level_teams():
    p = elo_probabilities(np.array([0.0]), DrawParams(0.28, 400))
    assert np.allclose(p, [[0.36, 0.28, 0.36]])


def test_large_gap_favours_home_and_shrinks_draws():
    p = elo_probabilities(np.array([0.0, 400.0]), DrawParams(0.28, 400))
    assert p[1, 0] > 0.8 * (1 - p[1, 1])
    assert p[1, 1] < p[0, 1]


def test_model_reads_elo_diff():
    m = EloModel(DrawParams(0.28, 400)).fit(pd.DataFrame())
    p = m.predict_proba(pd.DataFrame({"elo_diff": [0.0]}))
    assert np.allclose(p, [[0.36, 0.28, 0.36]])
