import pandas as pd
import pytest

from goalline.features.elo import EloParams, compute_elo, expected_home


def frame(rows):
    """rows: (date, division, home, away, home_goals, away_goals)."""
    cols = ["date", "division", "home", "away", "home_goals", "away_goals"]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df.date)
    df["season"] = [d.year if d.month >= 7 else d.year - 1 for d in df.date]
    df["tier"] = [2 if d.endswith("2") or d == "E1" else 1 for d in df.division]
    return df


def test_expected_home_with_home_advantage():
    assert expected_home(1500, 1500, 65) == pytest.approx(0.592466, abs=1e-6)
    assert expected_home(1500, 1500, 0) == pytest.approx(0.5)


def test_hand_computed_update():
    m = frame([("2000-09-01", "I1", "A", "B", 2, 0), ("2000-09-08", "I1", "B", "A", 1, 1)])
    elo = compute_elo(m, EloParams(k=20, home_advantage=65, reversion=0.0))
    assert list(elo.loc[0]) == [1500.0, 1500.0]
    assert elo.loc[1, "elo_home"] == pytest.approx(1491.8493, abs=1e-3)
    assert elo.loc[1, "elo_away"] == pytest.approx(1508.1507, abs=1e-3)


def test_newcomer_in_second_division_starts_lower():
    m = frame([("2000-09-01", "I2", "C", "D", 0, 0)])
    elo = compute_elo(m, EloParams(20, 65, 0.0))
    assert list(elo.loc[0]) == [1350.0, 1350.0]


def test_mean_reversion_at_first_match_of_new_season():
    m = frame([("2000-09-01", "I1", "A", "B", 2, 0), ("2001-09-01", "I1", "A", "B", 0, 0)])
    elo = compute_elo(m, EloParams(k=20, home_advantage=65, reversion=0.5))
    assert elo.loc[1, "elo_home"] == pytest.approx(1504.0753, abs=1e-3)
    assert elo.loc[1, "elo_away"] == pytest.approx(1495.9247, abs=1e-3)


def test_result_keeps_the_input_index_and_order():
    m = frame([("2000-09-08", "I1", "B", "A", 1, 1), ("2000-09-01", "I1", "A", "B", 2, 0)])
    m.index = [10, 20]
    elo = compute_elo(m, EloParams(20, 65, 0.0))
    assert list(elo.index) == [10, 20]
    assert list(elo.loc[20]) == [1500.0, 1500.0]


def test_ratings_are_pre_match_on_real_shaped_data(matches):
    params = EloParams(20, 60, 0.33)
    full = compute_elo(matches, params)
    i = len(matches) // 2
    changed = matches.copy()
    hg, ag = changed.loc[i, ["home_goals", "away_goals"]]
    changed.loc[i, ["home_goals", "away_goals"]] = (ag, hg) if hg != ag else (hg + 3, ag)
    perturbed = compute_elo(changed, params)
    upto = matches.date <= matches.loc[i, "date"]
    pd.testing.assert_frame_equal(full[upto], perturbed[upto])
    assert not full[~upto].equals(perturbed[~upto])
