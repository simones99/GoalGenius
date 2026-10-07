import pytest

from goalline.data.load import load_matches
from goalline.features.elo import EloParams
from goalline.features.table import build_features
from helpers import FIXTURE, SMOKE


@pytest.fixture(scope="session")
def matches():
    frame, _ = load_matches(FIXTURE, SMOKE, mode="final")
    return frame


@pytest.fixture(scope="session")
def features(matches):
    return build_features(matches, EloParams(k=20, home_advantage=60, reversion=0.33))
