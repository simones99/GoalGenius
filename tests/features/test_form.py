import pandas as pd
import pytest

from goalline.features.form import DEFAULT_POINTS, compute_form


def frame(rows):
    df = pd.DataFrame(rows, columns=["date", "home", "away", "home_goals", "away_goals"])
    df["date"] = pd.to_datetime(df.date)
    df["division"] = "I1"
    return df


def test_hand_computed_form_and_fallbacks():
    m = frame([
        ("2000-09-01", "A", "B", 2, 0),
        ("2000-09-02", "C", "D", 1, 1),
        ("2000-09-03", "A", "C", 0, 1),
    ])
    f = compute_form(m)
    assert f.loc[0, "form_points_home"] == pytest.approx(DEFAULT_POINTS)
    assert f.loc[0, "form_gd_home"] == 0
    assert f.loc[1, "form_points_home"] == pytest.approx(1.5)
    assert f.loc[1, "form_points_away"] == pytest.approx(1.5)
    assert f.loc[2, "form_points_home"] == pytest.approx(3.0)
    assert f.loc[2, "form_gd_home"] == pytest.approx(2.0)
    assert f.loc[2, "form_points_away"] == pytest.approx(1.0)
    assert f.loc[2, "form_gd_away"] == pytest.approx(0.0)
    assert f.short_history_home.tolist() == [1, 1, 1]


def test_window_is_the_previous_five_matches():
    results = [(2, 0)] * 5 + [(0, 1)]
    rows = [(f"2000-09-{i + 1:02d}", "A", "B", h, a) for i, (h, a) in enumerate(results)]
    rows.append(("2000-09-20", "A", "B", 0, 0))
    f = compute_form(frame(rows))
    assert f.loc[6, "form_points_home"] == pytest.approx((3 * 4 + 0) / 5)
    assert f.loc[6, "short_history_home"] == 0
    assert f.loc[5, "short_history_home"] == 0
    assert f.loc[4, "short_history_home"] == 1


def test_form_carries_across_divisions():
    m = frame([("2000-09-01", "A", "B", 3, 0), ("2001-09-01", "A", "C", 0, 0)])
    m.loc[0, "division"] = "I2"
    f = compute_form(m)
    assert f.loc[1, "form_points_home"] == pytest.approx(3.0)
