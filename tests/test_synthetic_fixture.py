import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
import make_synthetic  # noqa: E402

from helpers import FIXTURE  # noqa: E402


def test_fixture_is_reproducible():
    assert make_synthetic.generate().to_csv(index=False) == FIXTURE.read_text()


def test_fixture_shape():
    df = pd.read_csv(FIXTURE)
    assert set(df.Division) == {"I1", "I2", "E0", "E1"}
    assert len(df) == 6 * 4 * 30
    assert df.groupby(["MatchDate", "HomeTeam"]).size().max() == 1
    assert (1 / df[["OddHome", "OddDraw", "OddAway"]]).sum(axis=1).between(1.0, 1.25).all()


def test_promotion_happens():
    df = pd.read_csv(FIXTURE)
    teams = lambda season, div: set(  # noqa: E731
        df[(df.Division == div) & df.MatchDate.str.startswith(str(season))].HomeTeam
    )
    assert teams(2000, "I1") != teams(2001, "I1")
