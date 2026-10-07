import numpy as np

from goalline.models.xgb import XGBoostModel


def make():
    return XGBoostModel(max_depth=2, learning_rate=0.1, min_child_weight=1, train_first_season=2000)


def test_early_stopping_uses_the_last_training_season(features):
    model = make().fit(features[features.season < 2004])
    assert model.validation_season_ == 2003
    assert model.train_seasons_ == (2000, 2002)
    assert model.model_.best_iteration < 1000


def test_is_deterministic(features):
    history = features[features.season < 2004]
    target = features[(features.season == 2004) & (features.tier == 1)]
    a = make().fit(history).predict_proba(target)
    b = make().fit(history).predict_proba(target)
    assert np.array_equal(a, b)
