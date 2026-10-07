import numpy as np
import pandas as pd

from goalline.models.base import first_division_rows, outcome_codes
from goalline.models.simple import Frequency, Uniform


def history(rows):
    return pd.DataFrame(rows, columns=["division", "tier", "season", "result"])


def test_outcome_codes():
    assert outcome_codes(pd.DataFrame({"result": ["H", "D", "A"]})).tolist() == [0, 1, 2]


def test_first_division_rows_filters_tier_and_season():
    h = history([("I1", 1, 2001, "H"), ("I2", 2, 2005, "H"), ("I1", 1, 2005, "D")])
    assert first_division_rows(h, 2002).result.tolist() == ["D"]


def test_uniform():
    p = Uniform().fit(history([])).predict_proba(pd.DataFrame({"division": ["I1", "E0"]}))
    assert np.allclose(p, 1 / 3)


def test_frequency_by_division_with_smoothing():
    h = history([("I1", 1, 2005, "H"), ("I1", 1, 2005, "H"), ("I1", 1, 2005, "D"),
                 ("E0", 1, 2005, "A"), ("I2", 2, 2005, "A"), ("I1", 1, 2000, "A")])
    model = Frequency(train_first_season=2002).fit(h)
    p = model.predict_proba(pd.DataFrame({"division": ["I1", "SP1"]}))
    assert np.allclose(p[0], [3 / 6, 2 / 6, 1 / 6])
    assert np.allclose(p[1], np.array([2 + 1, 1 + 1, 1 + 1]) / (4 + 3))
