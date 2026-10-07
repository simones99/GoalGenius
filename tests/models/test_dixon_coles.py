import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from goalline.models.dixon_coles import (
    DixonColesModel,
    fit_dixon_coles,
    negative_log_likelihood,
    outcome_probabilities,
)

TRUE_ATTACK = np.array([0.3, 0.2, 0.1, 0.0, 0.0, -0.1, -0.2, -0.3])
TRUE_DEFENCE = np.array([-0.3, -0.2, -0.1, 0.0, 0.0, 0.1, 0.2, 0.3])
TRUE_HOME = 0.3


def simulated(country="ITA", seed=1):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    day = pd.Timestamp("2010-01-01")
    for _ in range(10):
        for i in range(8):
            for j in range(8):
                if i == j:
                    continue
                lam = np.exp(TRUE_HOME + TRUE_ATTACK[i] + TRUE_DEFENCE[j])
                mu = np.exp(TRUE_ATTACK[j] + TRUE_DEFENCE[i])
                rows.append((country, day, teams[i], teams[j], rng.poisson(lam), rng.poisson(mu)))
                day += pd.Timedelta(1, "D")
    columns = ["country", "date", "home", "away", "home_goals", "away_goals"]
    return pd.DataFrame(rows, columns=columns)


def test_likelihood_hand_values():
    hi, ai = np.array([0, 1]), np.array([1, 0])
    x, y = np.array([1.0, 2.0]), np.array([0.0, 2.0])
    w = np.array([1.0, 0.5])
    theta = np.zeros(2 * 2 + 2)
    value, _ = negative_log_likelihood(theta, hi, ai, x, y, w, 2)
    assert value == pytest.approx(3.0)
    theta[-1] = 0.1
    value, _ = negative_log_likelihood(theta, hi[:1], ai[:1], np.array([0.0]), np.array([0.0]),
                                       np.array([1.0]), 2)
    assert value == pytest.approx(2 - np.log(0.9))


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    n, m = 4, 40
    hi, ai = rng.integers(0, n, m), rng.integers(0, n, m)
    x, y = rng.integers(0, 3, m).astype(float), rng.integers(0, 3, m).astype(float)
    w = rng.uniform(0.2, 1.0, m)
    theta = rng.normal(0, 0.2, 2 * n + 2)
    theta[-1] = 0.05
    _, grad = negative_log_likelihood(theta, hi, ai, x, y, w, n)
    def objective(t):
        return negative_log_likelihood(t, hi, ai, x, y, w, n)[0]

    numeric = approx_fprime(theta, objective, 1e-7)
    assert np.allclose(grad, numeric, rtol=1e-4, atol=1e-5)


def test_parameter_recovery():
    rows = simulated()
    fit = fit_dixon_coles(rows, rows.date.max() + pd.Timedelta(1, "D"), xi=0.0)
    order = [fit.teams[f"T{i}"] for i in range(8)]
    assert fit.attack.sum() == pytest.approx(0, abs=1e-9)
    assert fit.home == pytest.approx(TRUE_HOME, abs=0.1)
    assert np.corrcoef(fit.attack[order], TRUE_ATTACK)[0, 1] > 0.9
    assert np.corrcoef(fit.defence[order], TRUE_DEFENCE)[0, 1] > 0.9


def test_outcome_probabilities_sum_to_one():
    p = outcome_probabilities(np.array([1.5, 0.4]), np.array([1.0, 2.2]), 0.05)
    assert np.allclose(p.sum(axis=1), 1)
    assert p[0, 0] > p[0, 2] and p[1, 2] > p[1, 0]


def test_unknown_team_gets_average_strength():
    rows = simulated()
    model = DixonColesModel(xi=0.0).fit(rows)
    target = pd.DataFrame({"country": ["ITA"], "home": ["NEW"], "away": ["T0"]})
    p = model.predict_proba(target)
    assert np.isfinite(p).all() and np.allclose(p.sum(), 1)


def test_country_without_history_falls_back_to_uniform():
    model = DixonColesModel(xi=0.0).fit(simulated("ITA"))
    p = model.predict_proba(pd.DataFrame({"country": ["ENG"], "home": ["X"], "away": ["Y"]}))
    assert np.allclose(p, 1 / 3)
