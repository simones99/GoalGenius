import pandas as pd

from goalline.cli import run_report
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
    assert "indistinguishable" in headline(boot(0.0, -0.01, 0.01))


def test_report_page(pipeline_run):
    run_report(pipeline_run, SMOKE)
    html = (pipeline_run.site / "index.html").read_text()
    for text in ("goal-line-calibration", "Serie A", "Bet365, Shin", "xgabora",
                 "Football-Data.co.uk", "ClubElo", "MIT", "2020-21", "Limitations",
                 "goalline develop", "uncommitted"):
        assert text in html, text
    assert html.count("<svg") >= 3
    assert "<script src" not in html
