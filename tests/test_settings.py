from goalline.constants import (
    COUNTRY,
    DIVISIONS,
    FIRST_DIVISIONS,
    MAIN_MODELS,
    MODEL_LABELS,
    MODEL_NAMES,
    OUTCOME_INDEX,
    OUTCOMES,
    REFERENCES,
    SECOND_DIVISIONS,
)
from goalline.settings import Settings


def test_divisions_and_countries():
    assert FIRST_DIVISIONS == ("I1", "E0", "SP1", "D1", "F1")
    assert SECOND_DIVISIONS == ("I2", "E1", "SP2", "D2", "F2")
    assert DIVISIONS == FIRST_DIVISIONS + SECOND_DIVISIONS
    assert COUNTRY["I1"] == COUNTRY["I2"] == "ITA"
    assert len(set(COUNTRY.values())) == 5


def test_outcome_order():
    assert OUTCOMES == ("H", "D", "A")
    assert OUTCOME_INDEX == {"H": 0, "D": 1, "A": 2}


def test_model_names_have_labels():
    assert set(MAIN_MODELS) <= set(MODEL_NAMES)
    for name in MODEL_NAMES + REFERENCES:
        assert name in MODEL_LABELS


def test_default_seasons_match_the_spec():
    s = Settings()
    assert s.first_season == 2000
    assert s.train_first_season == 2002
    assert s.dev_seasons == tuple(range(2005, 2021))
    assert len(s.dev_seasons) == 16
    assert s.final_seasons == tuple(range(2021, 2026))
    assert s.final_first_season == 2021
    assert s.last_season == 2025


def test_default_grids_match_the_spec():
    s = Settings()
    assert s.elo_k == (10, 15, 20, 25, 30, 40)
    assert s.elo_h == (40, 60, 80, 100)
    assert s.elo_r == (0.0, 0.1, 0.2, 0.33, 0.5)
    assert s.draw_peak == (0.22, 0.24, 0.26, 0.28, 0.30, 0.32)
    assert s.draw_scale == (200, 300, 400, 500, 600)
    assert s.dc_xi == (0.0005, 0.001, 0.0019, 0.003)
    assert s.logistic_c == (0.01, 0.1, 1.0, 10.0)
    assert s.xgb_max_depth == (2, 3, 4)
    assert s.xgb_learning_rate == (0.03, 0.1)
    assert s.xgb_min_child_weight == (1, 10)
    assert s.bootstrap_replicates == 2000
