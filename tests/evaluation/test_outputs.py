import numpy as np
import pytest

from goalline.benchmark import Benchmark
from goalline.evaluation.outputs import (
    comparisons,
    home_advantage,
    score_predictions,
    summarise,
    write_outputs,
)
from goalline.evaluation.walkforward import predict_all
from goalline.models.elo_model import DrawParams, EloModel
from goalline.models.simple import Uniform
from helpers import SMOKE


@pytest.fixture(scope="module")
def preds(features):
    models = [Uniform(), EloModel(DrawParams(0.28, 400)), Benchmark("shin"), Benchmark("max_odds")]
    return predict_all(features, models, (2004, 2005), mode="final", settings=SMOKE)


def test_scores_have_metrics(preds, features):
    scores = score_predictions(preds, features)
    assert {"log_loss", "rps", "brier", "division", "season"} <= set(scores.columns)
    uniform = scores[scores.model == "uniform"]
    assert np.allclose(uniform.log_loss, np.log(3))


def test_summary_groups(preds, features):
    summary = summarise(score_predictions(preds, features))
    expected = ["model", "group_type", "group", "n", "log_loss", "rps", "brier"]
    assert list(summary.columns) == expected
    overall = summary[summary.group_type == "overall"].set_index("model")
    assert overall.loc["uniform", "group"] == "all"
    assert set(summary[summary.group_type == "season"].group) == {"2004", "2005"}


def test_comparisons_align_on_common_matches(preds, features):
    trimmed = preds[~((preds.model == "max_odds") & (preds.match_id == preds.match_id.iloc[0]))]
    table = comparisons(score_predictions(trimmed, features), features, replicates=50)
    vs_max = table[(table.model == "elo") & (table.reference == "max_odds")]
    vs_shin = table[(table.model == "elo") & (table.reference == "shin")]
    assert (vs_max.n_matches.to_numpy() == vs_shin.n_matches.to_numpy() - 1).all()
    assert set(table.metric) == {"log_loss", "rps"}
    assert "proportional" not in set(table.reference)


def test_home_advantage_shares(features):
    ha = home_advantage(features)
    assert np.allclose(ha[["home_share", "draw_share", "away_share"]].sum(axis=1), 1)
    assert set(ha.division) == {"I1", "E0"}


def test_write_outputs(tmp_path, preds, features):
    tables = write_outputs(tmp_path / "final", preds, features, SMOKE)
    for name in ("predictions", "metrics", "reliability", "bootstrap"):
        assert (tmp_path / "final" / f"{name}.csv").exists()
        assert not tables[name].empty
