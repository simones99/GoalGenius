import pytest

from goalline.constants import MODEL_NAMES, REFERENCES
from goalline.evaluation.selection import (
    Selected,
    build_benchmarks,
    build_models,
    read_selected,
    write_selected,
)
from goalline.features.elo import EloParams
from goalline.models.elo_model import DrawParams
from helpers import SMOKE

SELECTED = Selected(
    elo=EloParams(20, 60, 0.33), draw=DrawParams(0.28, 400), dc_xi=0.0019, logistic_c=1.0,
    xgb_max_depth=3, xgb_learning_rate=0.03, xgb_min_child_weight=10,
    dev_log_loss={"elo": 0.98}, data_sha256="abc",
)


def test_round_trip(tmp_path):
    path = tmp_path / "config" / "selected.json"
    write_selected(path, SELECTED)
    assert read_selected(path) == SELECTED


def test_missing_file_explains_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="goalline develop"):
        read_selected(tmp_path / "selected.json")


def test_model_lists():
    assert [m.name for m in build_models(SELECTED, SMOKE)] == list(MODEL_NAMES)
    assert [b.name for b in build_benchmarks()] == list(REFERENCES)
