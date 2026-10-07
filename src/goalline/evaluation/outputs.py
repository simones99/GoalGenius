"""Turn predictions into the CSV files the report reads."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from goalline.constants import MAIN_MODELS, MODEL_NAMES, OUTCOMES, REFERENCES, SEED
from goalline.evaluation.bootstrap import matchday_clusters, paired_bootstrap
from goalline.evaluation.metrics import brier, log_loss, rps
from goalline.evaluation.reliability import reliability_table
from goalline.models.base import outcome_codes
from goalline.settings import Settings

PROBS = ["p_home", "p_draw", "p_away"]


def score_predictions(preds: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    scores = preds.merge(
        frame[["match_id", "division", "season", "date", "result"]],
        on="match_id", how="left", validate="many_to_one",
    )
    p, y = scores[PROBS].to_numpy(), outcome_codes(scores)
    scores["log_loss"], scores["rps"], scores["brier"] = log_loss(p, y), rps(p, y), brier(p, y)
    return scores


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for group_type, key in (("overall", None), ("division", "division"), ("season", "season")):
        keys = ["model"] + ([key] if key else [])
        agg = (
            scores.groupby(keys)
            .agg(n=("log_loss", "size"), log_loss=("log_loss", "mean"),
                 rps=("rps", "mean"), brier=("brier", "mean"))
            .reset_index()
        )
        agg["group_type"] = group_type
        agg["group"] = agg[key].astype(str) if key else "all"
        parts.append(agg[["model", "group_type", "group", "n", "log_loss", "rps", "brier"]])
    return pd.concat(parts, ignore_index=True)


def reliability_all(scores: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for model, rows in scores.groupby("model"):
        table = reliability_table(rows[PROBS].to_numpy(), outcome_codes(rows))
        parts.append(table.assign(model=model))
    out = pd.concat(parts, ignore_index=True)
    return out[["model", "outcome", "bin", "n", "mean_predicted", "observed"]]


def comparisons(
    scores: pd.DataFrame, frame: pd.DataFrame, replicates: int, seed: int = SEED
) -> pd.DataFrame:
    clusters = pd.Series(matchday_clusters(frame), index=frame.match_id.to_numpy())
    present = set(scores.model)
    pairs = [(m, r) for r in REFERENCES for m in MODEL_NAMES]
    pairs += list(combinations(MAIN_MODELS, 2))
    rows = []
    for model, reference in pairs:
        if model not in present or reference not in present:
            continue
        a = scores[scores.model == model].set_index("match_id")
        b = scores[scores.model == reference].set_index("match_id")
        common = a.index.intersection(b.index)
        for metric in ("log_loss", "rps"):
            r = paired_bootstrap(
                a.loc[common, metric].to_numpy(), b.loc[common, metric].to_numpy(),
                clusters.loc[common].to_numpy(), replicates, seed,
            )
            rows.append(
                {"model": model, "reference": reference, "metric": metric, "mean": r.mean,
                 "lower": r.lower, "upper": r.upper, "n_matches": r.n_matches,
                 "n_clusters": r.n_clusters}
            )
    return pd.DataFrame(rows)


def home_advantage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame[frame.tier == 1]
    shares = pd.crosstab([rows.division, rows.season], rows.result, normalize="index")
    shares = shares.reindex(columns=list(OUTCOMES), fill_value=0.0)
    counts = rows.groupby(["division", "season"]).size().rename("n")
    out = shares.join(counts).reset_index()
    out = out.rename(columns={"H": "home_share", "D": "draw_share", "A": "away_share"})
    return out[["division", "season", "n", "home_share", "draw_share", "away_share"]]


def write_outputs(
    directory: Path, preds: pd.DataFrame, frame: pd.DataFrame, settings: Settings
) -> dict[str, pd.DataFrame]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scores = score_predictions(preds, frame)
    tables = {
        "predictions": preds,
        "metrics": summarise(scores),
        "reliability": reliability_all(scores),
        "bootstrap": comparisons(scores, frame, settings.bootstrap_replicates),
    }
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv", index=False)
    return tables
