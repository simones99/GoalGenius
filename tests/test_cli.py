import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from conftest import make_paths
from goalline.cli import (
    Paths,
    UncommittedSelectionError,
    main,
    run_final,
    selected_commit,
)
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


def test_final_meta_records_code_and_environment(pipeline_run):
    meta = json.loads((pipeline_run.output / "final" / "final_meta.json").read_text())
    assert meta["code_commit"] == git_head()
    assert meta["code_dirty"] is dirty_tracked_files()
    assert meta["dixon_coles_nonconverged"] == 0 and meta["dixon_coles_fits"] > 0
    env = meta["environment"]
    assert set(env) == {"python", "numpy", "pandas", "scipy", "scikit-learn", "xgboost"}
    assert env["python"].count(".") == 2
    assert all(env.values())


def test_bootstrap_mean_is_positive_for_a_worse_model(pipeline_run):
    # difference = model loss minus reference loss: a worse model has a positive mean
    boot = pd.read_csv(pipeline_run.output / "final" / "bootstrap.csv")
    rows = boot[(boot.model == "uniform") & (boot.reference == "shin")
                & (boot.metric == "log_loss")]
    assert len(rows) == 1 and rows["mean"].iloc[0] > 0
    assert rows["lower"].iloc[0] > 0


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


def git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t",
         "-c", "commit.gpgsign=false", *args],
        cwd=repo, check=True, capture_output=True,
    )


def git_head():
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return done.stdout.strip()


def dirty_tracked_files():
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                          cwd=root, capture_output=True, text=True)
    return bool(done.stdout.strip())


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    config = tmp_path / "config"
    config.mkdir()
    (config / "selected.json").write_text("{}")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "freeze")
    return tmp_path


def test_selected_commit_of_a_clean_tracked_file(repo):
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True).stdout.strip()
    assert selected_commit(repo / "config" / "selected.json") == head


def test_selected_commit_of_a_modified_tracked_file(repo):
    path = repo / "config" / "selected.json"
    path.write_text('{"changed": true}')
    assert selected_commit(path) == "uncommitted"


def test_selected_commit_of_an_untracked_file_in_a_repo(repo):
    other = repo / "config" / "other.json"
    other.write_text("{}")
    assert selected_commit(other) == "uncommitted"


def test_final_refuses_a_modified_selection_inside_git(repo, monkeypatch, capsys):
    path = repo / "config" / "selected.json"
    path.write_text('{"changed": true}')
    monkeypatch.chdir(repo)
    assert main(["final"]) == 1
    assert "selected.json" in capsys.readouterr().err
    with pytest.raises(UncommittedSelectionError, match="commit"):
        run_final(Paths.under(repo), SMOKE)
    assert not (repo / "output").exists()
