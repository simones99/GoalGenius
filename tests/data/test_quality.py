import numpy as np
import pandas as pd

from goalline.data.quality import validate

BASE = dict(
    division="I1", date=pd.Timestamp("2010-09-12"), home="A", away="B",
    home_goals=2.0, away_goals=1.0, result="H",
    b365_h=1.8, b365_d=3.6, b365_a=4.5, max_h=1.9, max_d=3.8, max_a=4.8,
)


def raw(*rows):
    return pd.DataFrame([{**BASE, **r} for r in rows])


def count(report, check):
    return int(report.loc[report.check == check, "count"].sum())


def test_valid_rows_pass_untouched():
    clean, report = validate(raw({}, {"home": "C", "away": "D"}))
    assert len(clean) == 2
    assert report.empty
    assert clean.home_goals.dtype.kind == "i"


def test_rows_without_result_are_dropped():
    clean, report = validate(raw({}, {"home": "C", "away": "D", "home_goals": np.nan}))
    assert len(clean) == 1
    assert count(report, "no_result") == 1


def test_negative_goals_and_wrong_result_are_dropped():
    clean, report = validate(
        raw({}, {"home": "C", "away": "D", "home_goals": -1.0},
            {"home": "E", "away": "F", "result": "A"})
    )
    assert len(clean) == 1
    assert count(report, "invalid_goals_or_result") == 2


def test_duplicates_keep_the_first_row():
    clean, report = validate(raw({}, {}))
    assert len(clean) == 1
    assert count(report, "duplicate") == 1


def test_team_twice_on_one_date_drops_both_rows():
    clean, report = validate(raw({}, {"division": "I2", "away": "C"}, {"home": "X", "away": "Y"}))
    assert list(clean.home) == ["X"]
    assert count(report, "team_twice_same_date") == 2


def test_odds_not_above_one_are_blanked_but_the_match_is_kept():
    clean, report = validate(raw({"b365_h": 1.0}))
    assert len(clean) == 1
    assert clean[["b365_h", "b365_d", "b365_a"]].isna().all(axis=None)
    assert count(report, "b365_odds_blanked") == 1


def test_bet365_overround_outside_range_is_blanked():
    clean, report = validate(raw({"b365_h": 1.2, "b365_d": 3.0, "b365_a": 3.0}))
    assert clean[["b365_h", "b365_d", "b365_a"]].isna().all(axis=None)
    assert count(report, "b365_odds_blanked") == 1


def test_max_odds_below_one_overround_are_kept():
    clean, report = validate(raw({"max_h": 2.2, "max_d": 4.0, "max_a": 6.0}))
    assert clean[["max_h", "max_d", "max_a"]].notna().all(axis=None)
    assert count(report, "max_odds_blanked") == 0


def test_partial_odds_are_blanked():
    clean, report = validate(raw({"b365_d": np.nan}))
    assert clean[["b365_h", "b365_d", "b365_a"]].isna().all(axis=None)
    assert count(report, "b365_odds_blanked") == 1


def test_report_counts_by_division():
    _, report = validate(raw({"home_goals": np.nan}, {"division": "E0", "home_goals": np.nan}))
    assert set(report.division) == {"I1", "E0"}
    assert list(report.columns) == ["check", "division", "count"]
