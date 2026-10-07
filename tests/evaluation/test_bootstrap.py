import numpy as np
import pandas as pd
import pytest

from goalline.evaluation.bootstrap import bootstrap_replicates, matchday_clusters, paired_bootstrap


def test_clusters_are_league_and_iso_week():
    frame = pd.DataFrame(
        {"division": ["I1", "I1", "E0", "I1"],
         "date": pd.to_datetime(["2021-09-11", "2021-09-12", "2021-09-12", "2021-09-19"])}
    )
    c = matchday_clusters(frame)
    assert c[0] == c[1]
    assert c[1] != c[2]
    assert c[1] != c[3]


def test_identical_models_give_zero():
    loss = np.random.default_rng(0).uniform(0.5, 1.5, 300)
    clusters = np.repeat(np.arange(30), 10).astype(str)
    r = paired_bootstrap(loss, loss.copy(), clusters, replicates=200)
    assert r.mean == r.lower == r.upper == 0
    assert (r.n_matches, r.n_clusters) == (300, 30)


def test_whole_clusters_are_resampled():
    diff = np.array([0.0] + [1.0] * 99)
    clusters = np.array(["a"] + ["b"] * 99)
    stats = bootstrap_replicates(diff, clusters, replicates=500, seed=1)
    assert set(np.round(stats, 9)) <= {0.0, 0.99, 1.0}


def test_interval_contains_the_mean_and_is_seeded():
    rng = np.random.default_rng(2)
    a, b = rng.normal(1.0, 0.3, 500), rng.normal(1.05, 0.3, 500)
    clusters = np.repeat(np.arange(50), 10).astype(str)
    r1 = paired_bootstrap(a, b, clusters, replicates=500, seed=42)
    r2 = paired_bootstrap(a, b, clusters, replicates=500, seed=42)
    assert r1 == r2
    assert r1.lower < r1.mean < r1.upper
    assert r1.mean == pytest.approx((a - b).mean())


def test_sign_convention_worse_model_has_positive_mean():
    # difference = model loss minus reference loss, so a worse model gives a positive mean
    rng = np.random.default_rng(5)
    reference = rng.uniform(0.8, 1.0, 400)
    worse = reference + 0.1 + rng.normal(0, 0.01, 400)
    clusters = np.repeat(np.arange(40), 10).astype(str)
    r = paired_bootstrap(worse, reference, clusters, replicates=300)
    assert r.mean > 0 and r.lower > 0 and r.upper > 0
    flipped = paired_bootstrap(reference, worse, clusters, replicates=300)
    assert flipped.mean < 0 and flipped.upper < 0
