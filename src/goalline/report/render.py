"""Build the static report page from the CSV files in output/."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape

from goalline.constants import LEAGUE_NAMES, MAIN_MODELS, MODEL_LABELS, MODEL_NAMES
from goalline.data.load import season_label
from goalline.evaluation.reliability import expected_calibration_error
from goalline.report import charts


def headline(bootstrap: pd.DataFrame) -> str:
    rows = bootstrap[
        (bootstrap.reference == "shin") & (bootstrap.metric == "log_loss")
        & bootstrap.model.isin(MAIN_MODELS)
    ]
    best = rows.loc[rows["mean"].idxmin()]
    label = MODEL_LABELS[best.model]
    if best.upper < 0:
        return (f"{label} beats the de-margined Bet365 probabilities: log loss lower by "
                f"{-best['mean']:.4f} (95% CI {-best.upper:.4f} to {-best.lower:.4f}).")
    if best.lower > 0:
        return (f"No model beats the de-margined Bet365 probabilities. The closest, {label}, "
                f"has a log loss higher by {best['mean']:.4f} "
                f"(95% CI {best.lower:.4f} to {best.upper:.4f}).")
    return (f"The closest model, {label}, is statistically indistinguishable from the "
            f"de-margined Bet365 probabilities: log-loss difference {best['mean']:+.4f} "
            f"(95% CI {best.lower:+.4f} to {best.upper:+.4f}).")


def _key_rows(metrics: pd.DataFrame, bootstrap: pd.DataFrame, group_type: str, group: str):
    in_scope = (metrics.group_type == group_type) & (metrics.group == group)
    scope = metrics[in_scope].set_index("model")
    vs = bootstrap[(bootstrap.reference == "shin") & (bootstrap.metric == "log_loss")]
    vs = vs.set_index("model")
    rows = []
    for model in (*MODEL_NAMES, "shin"):
        if model not in scope.index:
            continue
        row = {"label": MODEL_LABELS[model], "n": int(scope.loc[model, "n"]),
               "log_loss": scope.loc[model, "log_loss"], "rps": scope.loc[model, "rps"],
               "brier": scope.loc[model, "brier"], "is_reference": model == "shin"}
        if group_type == "overall" and model in vs.index:
            row |= {"diff": vs.loc[model, "mean"], "lower": vs.loc[model, "lower"],
                    "upper": vs.loc[model, "upper"]}
        rows.append(row)
    return rows


def build_context(output_dir: Path, selected_path: Path) -> dict:
    output_dir = Path(output_dir)
    final = output_dir / "final"
    metrics = pd.read_csv(final / "metrics.csv", dtype={"group": str})
    bootstrap = pd.read_csv(final / "bootstrap.csv")
    reliability = pd.read_csv(final / "reliability.csv")
    meta = json.loads((final / "final_meta.json").read_text())
    quality = pd.read_csv(output_dir / "data_quality.csv")
    home_adv = pd.read_csv(output_dir / "home_advantage.csv")
    selected = json.loads(Path(selected_path).read_text())

    leagues = metrics[metrics.group_type == "division"]
    league_table = {
        "headers": [LEAGUE_NAMES[d] for d in LEAGUE_NAMES if d in set(leagues.group)],
        "rows": [
            {"label": MODEL_LABELS[m],
             "values": [leagues[(leagues.model == m) & (leagues.group == d)].log_loss.iloc[0]
                        for d in LEAGUE_NAMES if d in set(leagues.group)]}
            for m in (*MODEL_NAMES, "shin") if m in set(leagues.model)
        ],
    }
    sensitivity = [
        {"label": MODEL_LABELS[r.model], "reference": MODEL_LABELS[r.reference],
         "diff": r["mean"], "lower": r.lower, "upper": r.upper, "n": int(r.n_matches)}
        for _, r in bootstrap[(bootstrap.metric == "log_loss")
                              & bootstrap.reference.isin(["proportional", "max_odds"])
                              & bootstrap.model.isin(MAIN_MODELS)].iterrows()
    ]
    pairs = [
        {"label": MODEL_LABELS[r.model], "reference": MODEL_LABELS[r.reference],
         "diff": r["mean"], "lower": r.lower, "upper": r.upper}
        for _, r in bootstrap[(bootstrap.metric == "log_loss")
                              & bootstrap.reference.isin(MAIN_MODELS)].iterrows()
    ]
    ece = [
        {"label": MODEL_LABELS[m], **expected_calibration_error(t).round(4).to_dict()}
        for m, t in reliability.groupby("model") if m in (*MAIN_MODELS, "shin")
    ]
    seasons = meta["seasons"]
    return {
        "headline": headline(bootstrap),
        "meta": meta,
        "season_range": f"{season_label(seasons[0])} to {season_label(seasons[-1])}",
        "key_rows": _key_rows(metrics, bootstrap, "overall", "all"),
        "serie_a_rows": _key_rows(metrics, bootstrap, "division", "I1"),
        "league_table": league_table,
        "season_svg": charts.season_chart(metrics, (*MAIN_MODELS, "shin")),
        "reliability_svg": charts.reliability_chart(reliability, ("elo", "dixon_coles", "shin")),
        "ece_rows": ece,
        "sensitivity_rows": sensitivity,
        "pair_rows": pairs,
        "selected": selected,
        "home_adv_svg": charts.home_advantage_chart(home_adv),
        "quality_rows": quality.groupby("check")["count"].sum().reset_index().to_dict("records"),
    }


def render(output_dir: Path, site_dir: Path, selected_path: Path) -> Path:
    env = Environment(
        loader=PackageLoader("goalline", "report/templates"), autoescape=select_autoescape()
    )
    html = env.get_template("index.html.j2").render(**build_context(output_dir, selected_path))
    site_dir = Path(site_dir)
    site_dir.mkdir(parents=True, exist_ok=True)
    path = site_dir / "index.html"
    path.write_text(html)
    return path
