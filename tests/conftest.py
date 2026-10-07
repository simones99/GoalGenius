import pytest

from goalline.data.load import load_matches
from helpers import FIXTURE, SMOKE


@pytest.fixture(scope="session")
def matches():
    frame, _ = load_matches(FIXTURE, SMOKE, mode="final")
    return frame
