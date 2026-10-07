"""Every model and benchmark: rows sum to one, are strictly positive, and ignore row order."""

from collections.abc import Callable

import numpy as np
import pytest

from goalline.benchmark import Benchmark
from goalline.models.dixon_coles import DixonColesModel
from goalline.models.elo_model import DrawParams, EloModel
from goalline.models.simple import Frequency, Uniform

CASES: list[tuple[str, Callable]] = [
    ("uniform", lambda: Uniform()),
    ("frequency", lambda: Frequency(train_first_season=2000)),
    ("elo", lambda: EloModel(DrawParams(0.28, 400))),
    ("dixon_coles", lambda: DixonColesModel(xi=0.0019)),
    ("shin", lambda: Benchmark("shin")),
    ("proportional", lambda: Benchmark("proportional")),
    ("max_odds", lambda: Benchmark("max_odds")),
]


@pytest.mark.parametrize(("name", "factory"), CASES, ids=[c[0] for c in CASES])
def test_contract(name, factory, features):
    history = features[features.season < 2004]
    target = features[(features.season == 2004) & (features.tier == 1)]
    model = factory()
    assert model.name == name
    assert model.refit in ("season", "month")
    p = model.fit(history).predict_proba(target)
    assert p.shape == (len(target), 3)
    assert np.allclose(p.sum(axis=1), 1, atol=1e-9)
    assert (p > 0).all()
    shuffled = target.sample(frac=1, random_state=0)
    q = model.predict_proba(shuffled)
    assert np.allclose(q, p[target.index.get_indexer(shuffled.index)])
