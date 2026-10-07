"""The frozen hyperparameters: written by `develop`, read by `final`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from goalline.benchmark import Benchmark
from goalline.features.elo import EloParams
from goalline.models.dixon_coles import DixonColesModel
from goalline.models.elo_model import DrawParams, EloModel
from goalline.models.logistic import LogisticModel
from goalline.models.simple import Frequency, Uniform
from goalline.models.xgb import XGBoostModel
from goalline.settings import Settings


@dataclass(frozen=True)
class Selected:
    elo: EloParams
    draw: DrawParams
    dc_xi: float
    logistic_c: float
    xgb_max_depth: int
    xgb_learning_rate: float
    xgb_min_child_weight: float
    dev_log_loss: dict
    data_sha256: str

    def to_json(self) -> dict:
        return {
            "elo": {"k": self.elo.k, "home_advantage": self.elo.home_advantage,
                    "reversion": self.elo.reversion, "draw_peak": self.draw.peak,
                    "draw_scale": self.draw.scale},
            "dixon_coles": {"xi": self.dc_xi},
            "logistic": {"c": self.logistic_c},
            "xgboost": {"max_depth": self.xgb_max_depth, "learning_rate": self.xgb_learning_rate,
                        "min_child_weight": self.xgb_min_child_weight},
            "dev_log_loss": self.dev_log_loss,
            "data_sha256": self.data_sha256,
        }

    @classmethod
    def from_json(cls, d: dict) -> Selected:
        e = d["elo"]
        return cls(
            elo=EloParams(e["k"], e["home_advantage"], e["reversion"]),
            draw=DrawParams(e["draw_peak"], e["draw_scale"]),
            dc_xi=d["dixon_coles"]["xi"],
            logistic_c=d["logistic"]["c"],
            xgb_max_depth=d["xgboost"]["max_depth"],
            xgb_learning_rate=d["xgboost"]["learning_rate"],
            xgb_min_child_weight=d["xgboost"]["min_child_weight"],
            dev_log_loss=d["dev_log_loss"],
            data_sha256=d["data_sha256"],
        )


def write_selected(path: Path, selected: Selected) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selected.to_json(), indent=2) + "\n")


def read_selected(path: Path) -> Selected:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found: run `goalline develop`, commit the file and tag it before "
            "`goalline final`."
        )
    return Selected.from_json(json.loads(path.read_text()))


def build_models(selected: Selected, settings: Settings) -> list:
    first = settings.train_first_season
    return [
        Uniform(),
        Frequency(first),
        EloModel(selected.draw),
        DixonColesModel(selected.dc_xi),
        LogisticModel(selected.logistic_c, first),
        XGBoostModel(selected.xgb_max_depth, selected.xgb_learning_rate,
                     selected.xgb_min_child_weight, first),
    ]


def build_benchmarks() -> list:
    return [Benchmark("shin"), Benchmark("proportional"), Benchmark("max_odds")]
