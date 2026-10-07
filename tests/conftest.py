import json
import shutil

import pytest

from goalline.cli import Paths, run_develop, run_final, run_validate
from goalline.data.load import load_matches
from goalline.data.manifest import sha256_of
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


def make_paths(root):
    paths = Paths.under(root)
    paths.data.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE, paths.data)
    paths.manifest.write_text(json.dumps(
        {"url": paths.data.as_uri(), "commit": "synthetic", "sha256": sha256_of(paths.data),
         "retrieved": "2026-10-07"}
    ))
    return paths


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory):
    paths = make_paths(tmp_path_factory.mktemp("run"))
    run_validate(paths, SMOKE)
    run_develop(paths, SMOKE)
    run_final(paths, SMOKE)
    return paths
