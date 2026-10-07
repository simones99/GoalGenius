import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from goalline.data.load import load_matches, make_match_id, read_raw, season_label, season_of
from helpers import FIXTURE, SMOKE  # noqa: E402


def test_season_starts_in_july():
    dates = pd.Series(pd.to_datetime(["2010-06-30", "2010-07-01", "2011-05-20"]))
    assert list(season_of(dates)) == [2009, 2010, 2010]


def test_season_label():
    assert season_label(2005) == "2005-06"
    assert season_label(1999) == "1999-00"


def test_match_id_is_stable_and_short():
    a = make_match_id("I1", pd.Timestamp("2010-09-12"), "Juventus", "Inter")
    assert a == make_match_id("I1", pd.Timestamp("2010-09-12"), "Juventus", "Inter")
    assert a != make_match_id("I1", pd.Timestamp("2010-09-12"), "Inter", "Juventus")
    assert len(a) == 16 and int(a, 16) >= 0


def test_read_raw_renames_and_filters(tmp_path):
    path = tmp_path / "m.csv"
    pd.read_csv(FIXTURE).assign(Division=lambda d: d.Division.replace({"E1": "N1"})).to_csv(
        path, index=False
    )
    df = read_raw(path)
    assert set(df.division) == {"I1", "I2", "E0"}
    assert {"home", "away", "home_goals", "b365_h", "max_a"} <= set(df.columns)


def test_final_mode_keeps_all_seasons_up_to_the_last(matches):
    assert matches.season.min() == 2000
    assert matches.season.max() == SMOKE.last_season
    assert matches.date.is_monotonic_increasing
    assert list(matches.index) == list(range(len(matches)))
    assert set(matches.tier) == {1, 2}
    assert set(matches.country) == {"ITA", "ENG"}
    assert matches.match_id.is_unique


def test_develop_mode_never_returns_test_seasons():
    frame, quality = load_matches(FIXTURE, SMOKE, mode="develop")
    assert frame.season.max() == SMOKE.final_first_season - 1
    assert list(quality.columns) == ["check", "division", "count"]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        load_matches(FIXTURE, SMOKE, mode="peek")
