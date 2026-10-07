"""Static SVG charts for the report, drawn with matplotlib."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from goalline.constants import LEAGUE_NAMES, MODEL_LABELS, OUTCOMES  # noqa: E402

GREY = "#8a8a85"  # reads on both light and dark pages

plt.rcParams.update(
    {"font.size": 9, "axes.edgecolor": GREY, "axes.labelcolor": GREY, "text.color": GREY,
     "xtick.color": GREY, "ytick.color": GREY, "svg.fonttype": "none",
     "axes.spines.top": False, "axes.spines.right": False}
)


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    text = buf.getvalue()
    return text[text.index("<svg") :]


def reliability_chart(reliability: pd.DataFrame, models) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, outcome, title in zip(axes, OUTCOMES, ("Home win", "Draw", "Away win"), strict=True):
        ax.plot([0, 1], [0, 1], color="#bbb", lw=1, ls="--")
        for model in models:
            t = reliability[(reliability.model == model) & (reliability.outcome == outcome)]
            ax.plot(t.mean_predicted, t.observed, marker="o", ms=3, lw=1.2,
                    label=MODEL_LABELS[model])
        ax.set_title(title)
        ax.set_xlabel("Predicted probability")
    axes[0].set_ylabel("Observed frequency")
    axes[-1].legend(frameon=False, fontsize=8)
    return _svg(fig)


def season_chart(metrics: pd.DataFrame, models) -> str:
    seasons = metrics[metrics.group_type == "season"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for model in models:
        t = seasons[seasons.model == model].sort_values("group")
        ax.plot(t.group.astype(int), t.log_loss, marker="o", ms=3, lw=1.2,
                label=MODEL_LABELS[model])
    ax.set_xlabel("Season (start year)")
    ax.set_ylabel("Mean log loss")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _svg(fig)


def home_advantage_chart(home_adv: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for division, t in home_adv.groupby("division"):
        t = t.sort_values("season")
        ax.plot(t.season, t.home_share, lw=1.2, label=LEAGUE_NAMES.get(division, division))
    ax.axvspan(2019.5, 2020.5, color="#ccc", alpha=0.4, lw=0)
    ax.annotate("2020-21", (2020, ax.get_ylim()[1]), ha="center", va="top", fontsize=8)
    ax.set_xlabel("Season (start year)")
    ax.set_ylabel("Home-win share")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    return _svg(fig)
