import json

import pandas as pd
import pytest

from conftest import make_paths
from goalline.cli import Paths, main, run_final, selected_commit
from goalline.constants import MODEL_NAMES, REFERENCES
from goalline.data.manifest import ChecksumError
from helpers import SMOKE


def test_default_paths():
    p = Paths.default()
    assert str(p.data).endswith("data/raw/Matches.csv")
    assert str(p.config).endswith("config/selected.json")


def test_pipeline_writes_every_output(pipeline_run):
    out = pipeline_run.output
    assert (out / "data_quality.csv").exists()
    for stage in ("dev", "final"):
        for name in ("predictions", "metrics", "reliability", "bootstrap"):
            assert (out / stage / f"{name}.csv").exists()
    assert (out / "dev" / "grid_scores.csv").exists()
    assert (out / "home_advantage.csv").exists()
    assert pipeline_run.config.exists()


def test_final_scores_every_model_and_reference(pipeline_run):
    metrics = pd.read_csv(pipeline_run.output / "final" / "metrics.csv")
    assert set(metrics.model) == set(MODEL_NAMES) | set(REFERENCES)
    meta = json.loads((pipeline_run.output / "final" / "final_meta.json").read_text())
    assert meta["selected_commit"] == "uncommitted"
    assert meta["seasons"] == list(SMOKE.final_seasons)
    assert meta["n_matches"] > 0 and "max_odds_arbitrage" in meta


def test_elo_beats_uniform_on_data_generated_by_elo(pipeline_run):
    overall = pd.read_csv(pipeline_run.output / "final" / "metrics.csv")
    overall = overall[overall.group_type == "overall"].set_index("model")
    assert overall.loc["elo", "log_loss"] < overall.loc["uniform", "log_loss"]


def test_final_without_selection_fails_cleanly(tmp_path):
    paths = make_paths(tmp_path)
    with pytest.raises(FileNotFoundError, match="goalline develop"):
        run_final(paths, SMOKE)
    assert not paths.output.exists()


def test_final_with_other_data_fails_cleanly(tmp_path, pipeline_run):
    paths = make_paths(tmp_path)
    paths.config.parent.mkdir(parents=True)
    selected = json.loads(pipeline_run.config.read_text())
    selected["data_sha256"] = "0" * 64
    paths.config.write_text(json.dumps(selected))
    with pytest.raises(ChecksumError, match="develop"):
        run_final(paths, SMOKE)
    assert not paths.output.exists()


def test_selected_commit_outside_git(tmp_path):
    f = tmp_path / "selected.json"
    f.write_text("{}")
    assert selected_commit(f) == "uncommitted"


def test_main_rejects_unknown_command():
    with pytest.raises(SystemExit):
        main(["peek"])
