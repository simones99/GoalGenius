import json

import pandas as pd

from goalline.cli import run_report
from goalline.report import render as render_module
from goalline.report.render import headline
from helpers import SMOKE


def boot(mean, lower, upper):
    return pd.DataFrame(
        [{"model": m, "reference": "shin", "metric": "log_loss", "mean": mean + i * 0.01,
          "lower": lower + i * 0.01, "upper": upper + i * 0.01, "n_matches": 100,
          "n_clusters": 20} for i, m in enumerate(["elo", "dixon_coles", "logistic", "xgboost"])]
    )


def test_headline_when_the_market_wins():
    text = headline(boot(0.02, 0.01, 0.03))
    assert text.startswith("No model beats") and "Elo" in text


def test_headline_when_a_model_wins():
    assert "beats the de-margined" in headline(boot(-0.02, -0.03, -0.01))


def test_headline_when_indistinguishable():
    assert "no clear difference" in headline(boot(0.0, -0.01, 0.01))


def test_headline_uses_any_model_whose_interval_is_below_zero():
    frame = boot(0.0, -0.01, 0.01)
    frame.loc[frame.model == "elo", ["mean", "lower", "upper"]] = [-0.05, -0.10, 0.01]
    frame.loc[frame.model == "dixon_coles", ["mean", "lower", "upper"]] = [-0.02, -0.03, -0.01]
    text = headline(frame)
    assert "beats the de-margined" in text and text.startswith("Dixon")


def test_template_autoescapes():
    env = render_module._environment()
    assert env.from_string("{{ x }}").render(x="<b>") == "&lt;b&gt;"
    assert "_environment" in render_module.render.__code__.co_names


def test_report_page(pipeline_run):
    run_report(pipeline_run, SMOKE)
    html = (pipeline_run.site / "index.html").read_text()
    for text in ("goal-line-calibration", "Serie A", "Bet365, Shin", "xgabora",
                 "Football-Data.co.uk", "ClubElo", "MIT", "2020-21", "Limitations",
                 "goalline develop", "uncommitted"):
        assert text in html, text
    assert html.count("<svg") >= 3
    assert "<script src" not in html


def test_report_shows_code_commit_environment_and_honest_odds_wording(pipeline_run):
    run_report(pipeline_run, SMOKE)
    html = (pipeline_run.site / "index.html").read_text()
    meta = json.loads((pipeline_run.output / "final" / "final_meta.json").read_text())
    assert html.count(meta["code_commit"]) >= 2
    assert html.count(meta["selected_commit"]) >= 2
    assert f"xgboost {meta['environment']['xgboost']}" in html
    assert "bookmakers" not in html.split("<h2>Sensitivity</h2>")[1].split("</p>")[0]
    assert "best price reported in the data" in html
