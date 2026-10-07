# goal-line-calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the GoalGenius repository as `goalline`, a tested Python package that forecasts home/draw/away probabilities for the five big European leagues with four models and scores them against de-margined Bet365 odds, publishing a static report.

**Architecture:** A one-way pipeline: `data` (pinned download, quality checks, canonical frame) → `features` (Elo, form) → `models` and `benchmark` (common `fit` / `predict_proba` interface) → `evaluation` (walk-forward, metrics, bootstrap, develop/final orchestration) → `report` (static HTML), driven by a small CLI. Each module depends only on modules to its left and is tested on a synthetic league system committed in `tests/fixtures/`.

**Tech Stack:** Python ≥ 3.12, pandas, numpy, scipy, scikit-learn, xgboost, jinja2, matplotlib; pytest and ruff; GitHub Actions and GitHub Pages.

**Spec:** `docs/superpowers/specs/2026-10-07-goal-line-calibration-design.md`

## Global Constraints

- Package `src/goalline/`, CLI entry point `goalline`, `python -m goalline` also works.
- `requires-python = ">=3.12"`; CI runs Python 3.12; local development may use 3.13.
- First divisions `I1, E0, SP1, D1, F1`; second divisions `I2, E1, SP2, D2, F2` (ratings only).
- Outcome order everywhere: home, draw, away (`H, D, A`, indices 0, 1, 2).
- Season `YYYY` means `YYYY-(YY+1)`; a match dated in July or later belongs to the season starting that year.
- Burn-in 2000-01 to 2004-05; development 2005-06 to 2020-21; final test 2021-22 to 2025-26; 2026-27 out of scope.
- Learned models (frequency, logistic, XGBoost) train on first-division rows from 2002-03 onwards.
- Data source pinned: xgabora/Club-Football-Match-Data commit `25882a58a736daf7ece3781940eac17ae1117a66`, SHA-256 `ef224cf2c252f07a842b3bcfd4ba5c718c25cedd8937ffa174a74b86b5ba4221`. No data in git.
- Seeds fixed at 42; bootstrap 2,000 replicates.
- No betting simulation, no ROI.
- Commits: Conventional Commits, **no `Co-Authored-By` line**.
- TDD: every production unit starts from a failing test that was run and seen to fail.
- Before every commit run `ruff check --fix .`: it sorts imports in new test files automatically; a remaining ruff error must be fixed by hand.
- Every shell command below assumes the repository root `/Users/simonemezzabotta/Coding_Projects/goal-line-calibration` and an active virtualenv (`source .venv/bin/activate`).

## Review Focus

1. **A team the Dixon–Coles fit has never seen** (e.g. promoted from a division outside the data) must get finite probabilities from average strength, not a crash or NaN. Test: Task 11, `test_unknown_team_gets_average_strength`.
2. **A country with no matches in the Dixon–Coles window** must fall back to uniform probabilities for its matches. Test: Task 11, `test_country_without_history_falls_back_to_uniform`.
3. **Missing maximum odds on some evaluation matches** must drop only those rows from the max-odds benchmark, and every comparison must use the intersection of matches with the right count. Tests: Task 16, `test_rows_without_probabilities_are_dropped`; Task 17, `test_comparisons_align_on_common_matches`.
4. **`final` without a committed `config/selected.json`, or with a data file whose SHA-256 differs from the one used in `develop`,** must stop with a clear message and write nothing. Tests: Task 19, `test_final_without_selection_fails_cleanly`, `test_final_with_other_data_fails_cleanly`.
5. **An interrupted or corrupted download** must raise a checksum error and leave no file at the destination. Test: Task 3, `test_download_with_wrong_checksum_leaves_no_file`.

---

## File structure

```
pyproject.toml                      packaging, deps, pytest/ruff config
.gitignore
README.md                           placeholder in Task 1, final in Task 23
data/manifest.json                  pinned source (Task 3)
config/selected.json                frozen hyperparameters (Task 22)
src/goalline/
  __init__.py                       version
  __main__.py                       python -m goalline
  constants.py                      divisions, outcomes, labels, odds columns
  settings.py                       Settings: seasons and grids
  cli.py                            Paths, run_* functions, argparse main
  data/__init__.py
  data/manifest.py                  Manifest, sha256_of, verify, download
  data/quality.py                   validate(raw) -> clean, report
  data/load.py                      read_raw, season_of, season_label, prepare, load_matches
  features/__init__.py
  features/elo.py                   EloParams, expected_home, compute_elo
  features/form.py                  compute_form
  features/table.py                 FEATURE_COLUMNS, build_features
  benchmark.py                      proportional, shin, Benchmark
  models/__init__.py
  models/base.py                    Model protocol, outcome_codes, first_division_rows
  models/simple.py                  Uniform, Frequency
  models/elo_model.py               DrawParams, elo_probabilities, EloModel
  models/dixon_coles.py             negative_log_likelihood, fit_dixon_coles, DixonColesModel
  models/logistic.py                LogisticModel
  models/xgb.py                     XGBoostModel
  evaluation/__init__.py
  evaluation/metrics.py             log_loss, rps, brier
  evaluation/reliability.py         reliability_table, expected_calibration_error
  evaluation/bootstrap.py           matchday_clusters, bootstrap_replicates, paired_bootstrap
  evaluation/walkforward.py         FinalSeasonError, evaluation_rows, blocks, walk_forward, predict_all
  evaluation/outputs.py             score_predictions, summarise, reliability_all, comparisons, home_advantage, write_outputs
  evaluation/selection.py           Selected, read/write_selected, build_models, build_benchmarks
  evaluation/develop.py             tune_elo, tune, develop, final_predictions
  report/__init__.py
  report/charts.py                  SVG charts
  report/render.py                  headline, build_context, render
  report/templates/index.html.j2
tests/
  helpers.py                        FIXTURE path, SMOKE settings
  conftest.py                       matches, features, pipeline_run fixtures
  fixtures/make_synthetic.py        synthetic league generator
  fixtures/synthetic_matches.csv    generated, committed
  ... one test module per source module
.github/workflows/ci.yml
.github/workflows/publish.yml
```

---

### Task 1: Clean slate and scaffold

**Files:**
- Delete (git): `models/`, `src/features/`, `src/ingest/`, `src/analysis/`, `notebooks/`, `setup.py`, `data/processed/`, `data/results/`
- Delete (untracked): `venv/`, `.venv/`, `goalgenius.egg-info/`, `data/models/`; move `data/raw/*.csv` to `../_private_backup/goalgenius-old-data/`
- Create: `pyproject.toml`, `.gitignore` (replace), `README.md` (replace), `src/goalline/__init__.py`, `src/goalline/constants.py`, `src/goalline/settings.py`
- Test: `tests/test_settings.py`

**Interfaces:**
- Produces: `goalline.constants` — `FIRST_DIVISIONS`, `SECOND_DIVISIONS`, `DIVISIONS`, `COUNTRY`, `LEAGUE_NAMES`, `OUTCOMES`, `OUTCOME_INDEX`, `ODDS_COLUMNS`, `MODEL_NAMES`, `MAIN_MODELS`, `REFERENCES`, `MODEL_LABELS`, `SEED`. `goalline.settings.Settings` (frozen dataclass) with fields `first_season, train_first_season, dev_seasons, final_seasons, elo_k, elo_h, elo_r, draw_peak, draw_scale, dc_xi, logistic_c, xgb_max_depth, xgb_learning_rate, xgb_min_child_weight, bootstrap_replicates` and properties `final_first_season`, `last_season`.

- [ ] **Step 1: Remove the old code in its own commit**

```bash
git rm -r -q models src notebooks setup.py data/processed data/results
rm -rf venv .venv goalgenius.egg-info data/models
mkdir -p ../_private_backup/goalgenius-old-data && mv data/raw/*.csv ../_private_backup/goalgenius-old-data/ 2>/dev/null; rmdir data/raw 2>/dev/null; true
```

Replace `README.md` with:

```markdown
# goal-line-calibration

Rewrite in progress: this repository (formerly GoalGenius) is becoming a
probabilistic forecasting study of football match outcomes, scored against
bookmaker odds. Design: [docs/superpowers/specs/2026-10-07-goal-line-calibration-design.md](docs/superpowers/specs/2026-10-07-goal-line-calibration-design.md).
```

```bash
git add README.md
git commit -m "chore: remove the GoalGenius code before the rewrite"
```

- [ ] **Step 2: Create the virtualenv and packaging files**

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "goal-line-calibration"
version = "0.1.0"
description = "Football outcome forecasts scored against de-margined bookmaker odds"
requires-python = ">=3.12"
license = { text = "MIT" }
dependencies = [
  "pandas>=2.2,<3",
  "numpy>=2.0,<3",
  "scipy>=1.13",
  "scikit-learn>=1.5,<1.8",
  "xgboost>=2.1,<4",
  "jinja2>=3.1",
  "matplotlib>=3.9",
]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6"]

[project.scripts]
goalline = "goalline.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
goalline = ["report/templates/*.j2"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

`.gitignore`:

```
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
.pytest_cache/
.ruff_cache/
.DS_Store
data/raw/
output/
site/
```

`src/goalline/__init__.py`:

```python
"""Football outcome forecasts scored against de-margined bookmaker odds."""

__version__ = "0.1.0"
```

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -q -e ".[dev]"
```

- [ ] **Step 3: Write the failing test**

`tests/test_settings.py`:

```python
from goalline.constants import (
    COUNTRY,
    DIVISIONS,
    FIRST_DIVISIONS,
    MAIN_MODELS,
    MODEL_LABELS,
    MODEL_NAMES,
    OUTCOME_INDEX,
    OUTCOMES,
    REFERENCES,
    SECOND_DIVISIONS,
)
from goalline.settings import Settings


def test_divisions_and_countries():
    assert FIRST_DIVISIONS == ("I1", "E0", "SP1", "D1", "F1")
    assert SECOND_DIVISIONS == ("I2", "E1", "SP2", "D2", "F2")
    assert DIVISIONS == FIRST_DIVISIONS + SECOND_DIVISIONS
    assert COUNTRY["I1"] == COUNTRY["I2"] == "ITA"
    assert len(set(COUNTRY.values())) == 5


def test_outcome_order():
    assert OUTCOMES == ("H", "D", "A")
    assert OUTCOME_INDEX == {"H": 0, "D": 1, "A": 2}


def test_model_names_have_labels():
    assert set(MAIN_MODELS) <= set(MODEL_NAMES)
    for name in MODEL_NAMES + REFERENCES:
        assert name in MODEL_LABELS


def test_default_seasons_match_the_spec():
    s = Settings()
    assert s.first_season == 2000
    assert s.train_first_season == 2002
    assert s.dev_seasons == tuple(range(2005, 2021))
    assert len(s.dev_seasons) == 16
    assert s.final_seasons == tuple(range(2021, 2026))
    assert s.final_first_season == 2021
    assert s.last_season == 2025


def test_default_grids_match_the_spec():
    s = Settings()
    assert s.elo_k == (10, 15, 20, 25, 30, 40)
    assert s.elo_h == (40, 60, 80, 100)
    assert s.elo_r == (0.0, 0.1, 0.2, 0.33, 0.5)
    assert s.draw_peak == (0.22, 0.24, 0.26, 0.28, 0.30, 0.32)
    assert s.draw_scale == (200, 300, 400, 500, 600)
    assert s.dc_xi == (0.0005, 0.001, 0.0019, 0.003)
    assert s.logistic_c == (0.01, 0.1, 1.0, 10.0)
    assert s.xgb_max_depth == (2, 3, 4)
    assert s.xgb_learning_rate == (0.03, 0.1)
    assert s.xgb_min_child_weight == (1, 10)
    assert s.bootstrap_replicates == 2000
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_settings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goalline.constants'`

- [ ] **Step 5: Write minimal implementation**

`src/goalline/constants.py`:

```python
"""Names shared by every module."""

FIRST_DIVISIONS = ("I1", "E0", "SP1", "D1", "F1")
SECOND_DIVISIONS = ("I2", "E1", "SP2", "D2", "F2")
DIVISIONS = FIRST_DIVISIONS + SECOND_DIVISIONS

COUNTRY = {
    "I1": "ITA", "I2": "ITA",
    "E0": "ENG", "E1": "ENG",
    "SP1": "ESP", "SP2": "ESP",
    "D1": "GER", "D2": "GER",
    "F1": "FRA", "F2": "FRA",
}

LEAGUE_NAMES = {
    "I1": "Serie A",
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}

OUTCOMES = ("H", "D", "A")
OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}

ODDS_COLUMNS = {
    "b365": ("b365_h", "b365_d", "b365_a"),
    "max": ("max_h", "max_d", "max_a"),
}

MODEL_NAMES = ("uniform", "frequency", "elo", "dixon_coles", "logistic", "xgboost")
MAIN_MODELS = ("elo", "dixon_coles", "logistic", "xgboost")
REFERENCES = ("shin", "proportional", "max_odds")

MODEL_LABELS = {
    "uniform": "Uniform",
    "frequency": "Frequency",
    "elo": "Elo",
    "dixon_coles": "Dixon–Coles",
    "logistic": "Logistic regression",
    "xgboost": "XGBoost",
    "shin": "Bet365, Shin",
    "proportional": "Bet365, proportional",
    "max_odds": "Best odds, proportional",
}

SEED = 42
```

`src/goalline/settings.py`:

```python
"""Season ranges and hyperparameter grids. Defaults are the spec; tests use smaller ones."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    first_season: int = 2000
    train_first_season: int = 2002
    dev_seasons: tuple[int, ...] = tuple(range(2005, 2021))
    final_seasons: tuple[int, ...] = tuple(range(2021, 2026))
    elo_k: tuple[float, ...] = (10, 15, 20, 25, 30, 40)
    elo_h: tuple[float, ...] = (40, 60, 80, 100)
    elo_r: tuple[float, ...] = (0.0, 0.1, 0.2, 0.33, 0.5)
    draw_peak: tuple[float, ...] = (0.22, 0.24, 0.26, 0.28, 0.30, 0.32)
    draw_scale: tuple[float, ...] = (200, 300, 400, 500, 600)
    dc_xi: tuple[float, ...] = (0.0005, 0.001, 0.0019, 0.003)
    logistic_c: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    xgb_max_depth: tuple[int, ...] = (2, 3, 4)
    xgb_learning_rate: tuple[float, ...] = (0.03, 0.1)
    xgb_min_child_weight: tuple[float, ...] = (1, 10)
    bootstrap_replicates: int = 2000

    @property
    def final_first_season(self) -> int:
        return self.final_seasons[0]

    @property
    def last_season(self) -> int:
        return self.final_seasons[-1]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_settings.py -v && ruff check .`
Expected: 5 passed; ruff `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore src/goalline tests/test_settings.py
git commit -m "build: scaffold the goalline package with settings and constants"
```

---

### Task 2: Synthetic league fixture

**Files:**
- Create: `tests/fixtures/make_synthetic.py`, `tests/fixtures/synthetic_matches.csv` (generated), `tests/helpers.py`
- Test: `tests/test_synthetic_fixture.py`

**Interfaces:**
- Produces: `tests/fixtures/synthetic_matches.csv` in the raw xgabora schema (columns `Division, MatchDate, HomeTeam, AwayTeam, FTHome, FTAway, FTResult, OddHome, OddDraw, OddAway, MaxHome, MaxDraw, MaxAway`), seasons 2000–2005, divisions I1, I2, E0, E1, 6 teams each, one promotion/relegation per country per season, results drawn from a known Elo-plus-draw process. `tests/helpers.py`: `FIXTURE: Path`, `SMOKE: Settings` (dev 2002–2003, final 2004–2005, one- or two-value grids, 200 bootstrap replicates).

- [ ] **Step 1: Write the failing test**

`tests/test_synthetic_fixture.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_synthetic_fixture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'make_synthetic'`

- [ ] **Step 3: Write the generator, helpers and generate the CSV**

`tests/fixtures/make_synthetic.py`:

```python
"""Generate synthetic_matches.csv: a small two-country league system driven by known Elo strengths.

Run `python tests/fixtures/make_synthetic.py` to rewrite the CSV. The output is deterministic.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
SEASONS = range(2000, 2006)
COUNTRIES = {"ITA": ("I1", "I2"), "ENG": ("E0", "E1")}
TEAMS_PER_DIVISION = 6
HOME_ADVANTAGE = 60.0
DRAW_PEAK, DRAW_SCALE = 0.27, 400.0
MARGIN_B365, MARGIN_MAX = 1.05, 1.01
ROUND_SPACING_DAYS = 21
PATH = Path(__file__).with_name("synthetic_matches.csv")


def round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    """Double round robin by the circle method: each team plays once per round."""
    t = list(teams)
    n = len(t)
    rounds = []
    for r in range(n - 1):
        pairs = [(t[i], t[n - 1 - i]) for i in range(n // 2)]
        rounds.append(pairs if r % 2 == 0 else [(b, a) for a, b in pairs])
        t = [t[0], t[-1], *t[1:-1]]
    return rounds + [[(b, a) for a, b in rnd] for rnd in rounds]


def probabilities(r_home: float, r_away: float) -> np.ndarray:
    d = r_home + HOME_ADVANTAGE - r_away
    e = 1 / (1 + 10 ** (-d / 400))
    draw = DRAW_PEAK * np.exp(-0.5 * (d / DRAW_SCALE) ** 2)
    return np.array([e * (1 - draw), draw, (1 - e) * (1 - draw)])


def goals(rng: np.random.Generator, outcome: str) -> tuple[int, int]:
    if outcome == "D":
        g = int(rng.poisson(1.1))
        return g, g
    loser = int(rng.poisson(0.8))
    winner = loser + 1 + int(rng.poisson(0.7))
    return (winner, loser) if outcome == "H" else (loser, winner)


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for country, (top, second) in COUNTRIES.items():
        names = [f"{country}-{i:02d}" for i in range(2 * TEAMS_PER_DIVISION)]
        strength = dict(zip(names, rng.normal(1500, 200, len(names)), strict=True))
        ranked = sorted(names, key=strength.get, reverse=True)
        members = {top: ranked[:TEAMS_PER_DIVISION], second: ranked[TEAMS_PER_DIVISION:]}
        for season in SEASONS:
            start = dt.date(season, 8, 20)
            table: dict[str, int] = {}
            for division in (top, second):
                for k, rnd in enumerate(round_robin(members[division])):
                    day = start + dt.timedelta(days=ROUND_SPACING_DAYS * k)
                    for home, away in rnd:
                        p = probabilities(strength[home], strength[away])
                        outcome = "HDA"[rng.choice(3, p=p)]
                        hg, ag = goals(rng, outcome)
                        hp, ap = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}[outcome]
                        table[home] = table.get(home, 0) + hp
                        table[away] = table.get(away, 0) + ap
                        b365 = 1 / (p * MARGIN_B365)
                        best = 1 / (p * MARGIN_MAX)
                        rows.append(
                            {
                                "Division": division,
                                "MatchDate": day.isoformat(),
                                "HomeTeam": home,
                                "AwayTeam": away,
                                "FTHome": hg,
                                "FTAway": ag,
                                "FTResult": outcome,
                                "OddHome": round(b365[0], 2),
                                "OddDraw": round(b365[1], 2),
                                "OddAway": round(b365[2], 2),
                                "MaxHome": round(best[0], 2),
                                "MaxDraw": round(best[1], 2),
                                "MaxAway": round(best[2], 2),
                            }
                        )
            worst = min(members[top], key=table.get)
            best_second = max(members[second], key=table.get)
            members[top] = [t for t in members[top] if t != worst] + [best_second]
            members[second] = [t for t in members[second] if t != best_second] + [worst]
            for t in names:
                strength[t] += rng.normal(0, 15)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    generate().to_csv(PATH, index=False)
```

`tests/helpers.py`:

```python
from pathlib import Path

from goalline.settings import Settings

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_matches.csv"

SMOKE = Settings(
    first_season=2000,
    train_first_season=2000,
    dev_seasons=(2002, 2003),
    final_seasons=(2004, 2005),
    elo_k=(20,),
    elo_h=(60,),
    elo_r=(0.0, 0.33),
    draw_peak=(0.26, 0.28),
    draw_scale=(400,),
    dc_xi=(0.0019,),
    logistic_c=(1.0,),
    xgb_max_depth=(2,),
    xgb_learning_rate=(0.1,),
    xgb_min_child_weight=(1,),
    bootstrap_replicates=200,
)
```

```bash
python tests/fixtures/make_synthetic.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_synthetic_fixture.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures tests/helpers.py tests/test_synthetic_fixture.py
git commit -m "test: add a synthetic two-country league system as fixture"
```

---

### Task 3: Pinned manifest and download

**Files:**
- Create: `src/goalline/data/__init__.py` (empty), `src/goalline/data/manifest.py`, `data/manifest.json`
- Test: `tests/data/test_manifest.py`

**Interfaces:**
- Produces: `Manifest` (frozen dataclass: `url: str, commit: str, sha256: str, retrieved: str`), `read_manifest(path: Path) -> Manifest`, `sha256_of(path: Path) -> str`, `ChecksumError(RuntimeError)`, `verify(path: Path, manifest: Manifest) -> None`, `download(manifest: Manifest, dest: Path) -> Path`.

- [ ] **Step 1: Write the failing test**

`tests/data/test_manifest.py`:

```python
import hashlib
import json

import pytest

from goalline.data.manifest import (
    ChecksumError,
    Manifest,
    download,
    read_manifest,
    sha256_of,
    verify,
)


def write(tmp_path, text="a,b\n1,2\n"):
    path = tmp_path / "source.csv"
    path.write_text(text)
    return path


def test_sha256_of_matches_hashlib(tmp_path):
    path = write(tmp_path)
    assert sha256_of(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_read_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"url": "u", "commit": "c", "sha256": "s", "retrieved": "r"}))
    assert read_manifest(path) == Manifest(url="u", commit="c", sha256="s", retrieved="r")


def test_verify_accepts_matching_file(tmp_path):
    path = write(tmp_path)
    verify(path, Manifest("u", "c", sha256_of(path), "r"))


def test_verify_rejects_changed_file(tmp_path):
    path = write(tmp_path)
    with pytest.raises(ChecksumError, match="SHA-256"):
        verify(path, Manifest("u", "c", "0" * 64, "r"))


def test_download_fetches_and_verifies(tmp_path):
    source = write(tmp_path)
    dest = tmp_path / "raw" / "Matches.csv"
    manifest = Manifest(source.as_uri(), "c", sha256_of(source), "r")
    assert download(manifest, dest) == dest
    assert dest.read_text() == source.read_text()


def test_download_with_wrong_checksum_leaves_no_file(tmp_path):
    source = write(tmp_path)
    dest = tmp_path / "raw" / "Matches.csv"
    with pytest.raises(ChecksumError):
        download(Manifest(source.as_uri(), "c", "0" * 64, "r"), dest)
    assert not dest.exists()
    assert list(dest.parent.iterdir()) == []


def test_repository_manifest_is_pinned():
    m = read_manifest(__import__("pathlib").Path("data/manifest.json"))
    assert m.commit == "25882a58a736daf7ece3781940eac17ae1117a66"
    assert m.commit in m.url
    assert m.sha256 == "ef224cf2c252f07a842b3bcfd4ba5c718c25cedd8937ffa174a74b86b5ba4221"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goalline.data'`

- [ ] **Step 3: Write minimal implementation**

`src/goalline/data/manifest.py`:

```python
"""Pinned source file: where it comes from and how to check it is the same file."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


class ChecksumError(RuntimeError):
    pass


@dataclass(frozen=True)
class Manifest:
    url: str
    commit: str
    sha256: str
    retrieved: str


def read_manifest(path: Path) -> Manifest:
    data = json.loads(Path(path).read_text())
    return Manifest(data["url"], data["commit"], data["sha256"], data["retrieved"])


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, manifest: Manifest) -> None:
    actual = sha256_of(path)
    if actual != manifest.sha256:
        raise ChecksumError(
            f"SHA-256 of {path} is {actual}, the manifest pins {manifest.sha256}. "
            "Delete the file and run `goalline download` again."
        )


def download(manifest: Manifest, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        urllib.request.urlretrieve(manifest.url, tmp)
        verify(tmp, manifest)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest
```

`data/manifest.json`:

```json
{
  "url": "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data/25882a58a736daf7ece3781940eac17ae1117a66/data/Matches.csv",
  "commit": "25882a58a736daf7ece3781940eac17ae1117a66",
  "sha256": "ef224cf2c252f07a842b3bcfd4ba5c718c25cedd8937ffa174a74b86b5ba4221",
  "retrieved": "2026-10-07",
  "licence": "MIT (xgabora/Club-Football-Match-Data); upstream sources Football-Data.co.uk and ClubElo"
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_manifest.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/goalline/data data/manifest.json tests/data/test_manifest.py
git commit -m "feat(data): pin the source file by commit and SHA-256"
```

---

### Task 4: Quality checks

**Files:**
- Create: `src/goalline/data/quality.py`
- Test: `tests/data/test_quality.py`

**Interfaces:**
- Consumes: `goalline.constants.ODDS_COLUMNS`.
- Produces: `OVERROUND_RANGE = (1.0, 1.25)`; `validate(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]`. Input columns: `division, date, home, away, home_goals, away_goals, result, b365_h, b365_d, b365_a, max_h, max_d, max_a`. Output 1: clean rows, goals as `int`. Output 2: report with columns `check, division, count` and check names `no_result`, `invalid_goals_or_result`, `duplicate`, `team_twice_same_date`, `b365_odds_blanked`, `max_odds_blanked`.

- [ ] **Step 1: Write the failing test**

`tests/data/test_quality.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_quality.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goalline.data.quality'`

- [ ] **Step 3: Write minimal implementation**

`src/goalline/data/quality.py`:

```python
"""Quality checks. Bad matches are dropped, bad odds are blanked, and everything is counted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from goalline.constants import ODDS_COLUMNS

OVERROUND_RANGE = (1.0, 1.25)


def validate(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    log: list[dict] = []

    def record(check: str, mask) -> None:
        mask = np.asarray(mask, dtype=bool)
        for division, n in df.loc[mask, "division"].value_counts().items():
            log.append({"check": check, "division": division, "count": int(n)})

    no_result = df.home_goals.isna() | df.away_goals.isna()
    record("no_result", no_result)
    df = df[~no_result]

    hg, ag = df.home_goals, df.away_goals
    expected = np.where(hg > ag, "H", np.where(hg < ag, "A", "D"))
    invalid = (hg < 0) | (ag < 0) | (hg % 1 != 0) | (ag % 1 != 0) | df.result.ne(expected)
    record("invalid_goals_or_result", invalid)
    df = df[~invalid]

    duplicate = df.duplicated(["division", "date", "home", "away"], keep="first")
    record("duplicate", duplicate)
    df = df[~duplicate]

    appearances = pd.concat(
        [
            df[["date", "home"]].rename(columns={"home": "team"}),
            df[["date", "away"]].rename(columns={"away": "team"}),
        ]
    )
    per_day = appearances.groupby(["date", "team"]).size()
    clashes = per_day[per_day > 1].index
    twice = pd.MultiIndex.from_arrays([df.date, df.home]).isin(clashes) | pd.MultiIndex.from_arrays(
        [df.date, df.away]
    ).isin(clashes)
    record("team_twice_same_date", twice)
    df = df[~twice].copy()

    for name, columns in ODDS_COLUMNS.items():
        odds = df[list(columns)]
        complete = odds.notna().all(axis=1)
        partial = odds.notna().any(axis=1) & ~complete
        not_above_one = complete & (odds <= 1).any(axis=1)
        blank = partial | not_above_one
        if name == "b365":
            overround = (1 / odds).sum(axis=1)
            low, high = OVERROUND_RANGE
            blank |= complete & ~not_above_one & ((overround < low) | (overround > high))
        record(f"{name}_odds_blanked", blank)
        df.loc[blank, list(columns)] = np.nan

    df["home_goals"] = df.home_goals.astype(int)
    df["away_goals"] = df.away_goals.astype(int)
    report = pd.DataFrame(log, columns=["check", "division", "count"])
    return df.reset_index(drop=True), report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_quality.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/goalline/data/quality.py tests/data/test_quality.py
git commit -m "feat(data): quality checks that drop bad matches and blank bad odds"
```

---

### Task 5: Loading the canonical frame

**Files:**
- Create: `src/goalline/data/load.py`, `tests/conftest.py`
- Test: `tests/data/test_load.py`

**Interfaces:**
- Consumes: `validate` (Task 4), `Settings` (Task 1), constants.
- Produces: `RAW_TO_CANONICAL: dict`, `read_raw(path) -> pd.DataFrame`, `season_of(dates: pd.Series) -> np.ndarray`, `season_label(season: int) -> str`, `make_match_id(division, date, home, away) -> str`, `prepare(raw) -> tuple[pd.DataFrame, pd.DataFrame]`, `load_matches(path, settings, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]`. Canonical frame columns: `match_id, division, country, tier, season, date, home, away, home_goals, away_goals, result, b365_h, b365_d, b365_a, max_h, max_d, max_a`, sorted by `date, division, home`, index `0..n-1`. Pytest fixture `matches` (synthetic fixture, `SMOKE`, mode `final`).

- [ ] **Step 1: Write the failing test**

`tests/conftest.py`:

```python
import pytest
from helpers import FIXTURE, SMOKE

from goalline.data.load import load_matches


@pytest.fixture(scope="session")
def matches():
    frame, _ = load_matches(FIXTURE, SMOKE, mode="final")
    return frame
```

`tests/data/test_load.py`:

```python
import pandas as pd
import pytest
from helpers import FIXTURE, SMOKE

from goalline.data.load import load_matches, make_match_id, read_raw, season_label, season_of


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_load.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goalline.data.load'`

- [ ] **Step 3: Write minimal implementation**

`src/goalline/data/load.py`:

```python
"""Read the raw file into the canonical match frame used by every later module."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from goalline.constants import COUNTRY, DIVISIONS, FIRST_DIVISIONS
from goalline.data.quality import validate
from goalline.settings import Settings

RAW_TO_CANONICAL = {
    "Division": "division",
    "MatchDate": "date",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHome": "home_goals",
    "FTAway": "away_goals",
    "FTResult": "result",
    "OddHome": "b365_h",
    "OddDraw": "b365_d",
    "OddAway": "b365_a",
    "MaxHome": "max_h",
    "MaxDraw": "max_d",
    "MaxAway": "max_a",
}

COLUMNS = [
    "match_id", "division", "country", "tier", "season", "date", "home", "away",
    "home_goals", "away_goals", "result",
    "b365_h", "b365_d", "b365_a", "max_h", "max_d", "max_a",
]


def read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=list(RAW_TO_CANONICAL),
        dtype={"Division": str, "HomeTeam": str, "AwayTeam": str, "FTResult": str},
    ).rename(columns=RAW_TO_CANONICAL)
    df["date"] = pd.to_datetime(df["date"])
    return df[df.division.isin(DIVISIONS)].reset_index(drop=True)


def season_of(dates: pd.Series) -> np.ndarray:
    return np.where(dates.dt.month >= 7, dates.dt.year, dates.dt.year - 1)


def season_label(season: int) -> str:
    return f"{season}-{(season + 1) % 100:02d}"


def make_match_id(division: str, date: pd.Timestamp, home: str, away: str) -> str:
    key = f"{division}|{date:%Y-%m-%d}|{home}|{away}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean, report = validate(raw)
    clean["season"] = season_of(clean.date)
    clean["tier"] = np.where(clean.division.isin(FIRST_DIVISIONS), 1, 2)
    clean["country"] = clean.division.map(COUNTRY)
    clean["match_id"] = [
        make_match_id(*row)
        for row in zip(clean.division, clean.date, clean.home, clean.away, strict=True)
    ]
    clean = clean.sort_values(["date", "division", "home"], kind="stable").reset_index(drop=True)
    return clean[COLUMNS], report


def load_matches(path: Path, settings: Settings, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mode not in ("develop", "final"):
        raise ValueError(f"mode must be 'develop' or 'final', not {mode!r}")
    frame, report = prepare(read_raw(path))
    keep = (frame.season >= settings.first_season) & (frame.season <= settings.last_season)
    if mode == "develop":
        keep &= frame.season < settings.final_first_season
    return frame[keep].reset_index(drop=True), report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data -v`
Expected: all data tests pass (24 passed)

- [ ] **Step 5: Commit**

```bash
git add src/goalline/data/load.py tests/conftest.py tests/data/test_load.py
git commit -m "feat(data): canonical match frame with seasons, ids and a develop-mode guard"
```

---

### Task 6: Elo ratings

**Files:**
- Create: `src/goalline/features/__init__.py` (empty), `src/goalline/features/elo.py`
- Test: `tests/features/test_elo.py`

**Interfaces:**
- Consumes: canonical frame columns `date, season, division, tier, home, away, home_goals, away_goals`.
- Produces: `EloParams(k: float, home_advantage: float, reversion: float)` (frozen dataclass); `INITIAL_RATING = {1: 1500.0, 2: 1350.0}`; `expected_home(r_home, r_away, home_advantage) -> float | np.ndarray`; `compute_elo(matches, params) -> pd.DataFrame` with columns `elo_home, elo_away` (pre-match ratings), same index as `matches`.

- [ ] **Step 1: Write the failing test**

`tests/features/test_elo.py`:

```python
import pandas as pd
import pytest

from goalline.features.elo import EloParams, compute_elo, expected_home


def frame(rows):
    """rows: (date, division, home, away, home_goals, away_goals)."""
    df = pd.DataFrame(rows, columns=["date", "division", "home", "away", "home_goals", "away_goals"])
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/features/test_elo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goalline.features'`

- [ ] **Step 3: Write minimal implementation**

`src/goalline/features/elo.py`:

```python
"""Elo ratings over all ten divisions, updated match by match, reported before each match."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

INITIAL_RATING = {1: 1500.0, 2: 1350.0}


@dataclass(frozen=True)
class EloParams:
    k: float
    home_advantage: float
    reversion: float


def expected_home(r_home, r_away, home_advantage):
    return 1.0 / (1.0 + 10.0 ** ((r_away - (r_home + home_advantage)) / 400.0))


def compute_elo(matches: pd.DataFrame, params: EloParams) -> pd.DataFrame:
    m = matches.sort_values("date", kind="stable")
    ratings: dict[str, float] = {}
    last: dict[str, tuple[int, str]] = {}
    snapshots: dict[tuple[int, str], float] = {}
    current_season = None
    out_home = np.empty(len(m))
    out_away = np.empty(len(m))

    rows = zip(
        m.season.to_numpy(), m.division.to_numpy(), m.tier.to_numpy(),
        m.home.to_numpy(), m.away.to_numpy(),
        m.home_goals.to_numpy(), m.away_goals.to_numpy(),
        strict=True,
    )
    for i, (season, division, tier, home, away, hg, ag) in enumerate(rows):
        if season != current_season:
            if current_season is not None:
                by_division: dict[str, list[float]] = defaultdict(list)
                for team, (s, d) in last.items():
                    if s == current_season:
                        by_division[d].append(ratings[team])
                for d, values in by_division.items():
                    snapshots[(current_season, d)] = float(np.mean(values))
            current_season = season
        for team in (home, away):
            if team not in ratings:
                ratings[team] = INITIAL_RATING[int(tier)]
            else:
                s, d = last[team]
                if s != season:
                    ratings[team] += params.reversion * (snapshots[(s, d)] - ratings[team])
            last[team] = (season, division)
        r_home, r_away = ratings[home], ratings[away]
        out_home[i], out_away[i] = r_home, r_away
        score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        delta = params.k * (score - expected_home(r_home, r_away, params.home_advantage))
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    result = pd.DataFrame({"elo_home": out_home, "elo_away": out_away}, index=m.index)
    return result.loc[matches.index]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/features/test_elo.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/goalline/features tests/features/test_elo.py
git commit -m "feat(features): Elo ratings with mean reversion and tier-based starting values"
```

---

### Task 7: Form and the feature table

**Files:**
- Create: `src/goalline/features/form.py`, `src/goalline/features/table.py`
- Modify: `tests/conftest.py` (add `features` fixture)
- Test: `tests/features/test_form.py`, `tests/features/test_table.py`

**Interfaces:**
- Consumes: `EloParams`, `compute_elo` (Task 6).
- Produces: `FORM_WINDOW = 5`, `DEFAULT_POINTS = 4 / 3`, `compute_form(matches) -> pd.DataFrame` with columns `form_points_home, form_points_away, form_gd_home, form_gd_away, short_history_home, short_history_away` (same index). `FEATURE_COLUMNS: list[str]` = `["elo_diff", "form_points_diff", "form_gd_diff", "short_history_home", "short_history_away", "league_I1", "league_E0", "league_SP1", "league_D1", "league_F1"]`; `build_features(matches, params) -> pd.DataFrame` = `matches` plus the Elo and form columns, plus every column in `FEATURE_COLUMNS`. Pytest fixture `features`.

- [ ] **Step 1: Write the failing tests**

`tests/features/test_form.py`:

```python
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
```

`tests/features/test_table.py`:

```python
import numpy as np
import pandas as pd

from goalline.features.elo import EloParams
from goalline.features.table import FEATURE_COLUMNS, build_features

PARAMS = EloParams(20, 60, 0.33)


def test_feature_columns_present(features):
    assert set(FEATURE_COLUMNS) <= set(features.columns)
    assert np.allclose(features.elo_diff, features.elo_home + 60 - features.elo_away)
    league = features[[c for c in FEATURE_COLUMNS if c.startswith("league_")]].sum(axis=1)
    assert (league[features.tier == 1] == 1).all()
    assert (league[features.tier == 2] == 0).all()


def test_perturbing_a_result_leaves_earlier_features_unchanged(matches):
    i = len(matches) // 2
    full = build_features(matches, PARAMS)
    changed = matches.copy()
    hg, ag = changed.loc[i, ["home_goals", "away_goals"]]
    changed.loc[i, ["home_goals", "away_goals"]] = (ag, hg) if hg != ag else (hg + 3, ag)
    perturbed = build_features(changed, PARAMS)
    upto = matches.date <= matches.loc[i, "date"]
    cols = FEATURE_COLUMNS + ["elo_home", "elo_away"]
    pd.testing.assert_frame_equal(full.loc[upto, cols], perturbed.loc[upto, cols])
    assert not full.loc[~upto, cols].equals(perturbed.loc[~upto, cols])


def test_truncating_the_data_after_a_date_changes_nothing_on_that_date(matches):
    d = matches.loc[len(matches) // 2, "date"]
    full = build_features(matches, PARAMS)
    part = build_features(matches[matches.date <= d], PARAMS)
    on_day = matches.date == d
    cols = FEATURE_COLUMNS + ["elo_home", "elo_away"]
    pd.testing.assert_frame_equal(full.loc[on_day, cols], part.loc[on_day, cols])
```

Add to `tests/conftest.py`:

```python
from goalline.features.elo import EloParams
from goalline.features.table import build_features


@pytest.fixture(scope="session")
def features(matches):
    return build_features(matches, EloParams(k=20, home_advantage=60, reversion=0.33))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/features -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goalline.features.form'`

- [ ] **Step 3: Write minimal implementation**

`src/goalline/features/form.py`:

```python
"""Recent form: mean points and goal difference over each team's previous five matches."""

from __future__ import annotations

import numpy as np
import pandas as pd

FORM_WINDOW = 5
DEFAULT_POINTS = 4 / 3


def _previous_mean(s: pd.Series) -> pd.Series:
    return s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()


def compute_form(matches: pd.DataFrame) -> pd.DataFrame:
    hg = matches.home_goals.to_numpy()
    ag = matches.away_goals.to_numpy()
    home_points = np.select([hg > ag, hg == ag], [3, 1], 0)
    away_points = np.select([hg < ag, hg == ag], [3, 1], 0)
    n = len(matches)
    long = pd.DataFrame(
        {
            "row": np.concatenate([matches.index, matches.index]),
            "side": ["home"] * n + ["away"] * n,
            "date": np.concatenate([matches.date, matches.date]),
            "division": np.concatenate([matches.division, matches.division]),
            "team": np.concatenate([matches.home, matches.away]),
            "points": np.concatenate([home_points, away_points]),
            "gd": np.concatenate([hg - ag, ag - hg]),
        }
    ).sort_values(["team", "date"], kind="stable")
    by_team = long.groupby("team", sort=False)
    long["prev_points"] = by_team["points"].transform(_previous_mean)
    long["prev_gd"] = by_team["gd"].transform(_previous_mean)
    long["n_prev"] = by_team.cumcount()

    daily = (
        long.groupby(["division", "date"])
        .agg(points=("points", "sum"), n=("points", "size"))
        .reset_index()
        .sort_values(["division", "date"])
    )
    by_division = daily.groupby("division")
    cum_points = by_division["points"].cumsum() - daily.points
    cum_n = by_division["n"].cumsum() - daily.n
    daily["division_mean"] = (cum_points / cum_n.replace(0, np.nan)).fillna(DEFAULT_POINTS)
    long = long.merge(daily[["division", "date", "division_mean"]], on=["division", "date"])

    long["form_points"] = long.prev_points.fillna(long.division_mean)
    long["form_gd"] = long.prev_gd.fillna(0.0)
    long["short_history"] = (long.n_prev < FORM_WINDOW).astype(int)

    out = pd.DataFrame(index=matches.index)
    for side in ("home", "away"):
        part = long[long.side == side].set_index("row")
        out[f"form_points_{side}"] = part.form_points
        out[f"form_gd_{side}"] = part.form_gd
        out[f"short_history_{side}"] = part.short_history
    return out[
        ["form_points_home", "form_points_away", "form_gd_home", "form_gd_away",
         "short_history_home", "short_history_away"]
    ]
```

`src/goalline/features/table.py`:

```python
"""The feature table shared by the logistic regression and XGBoost."""

from __future__ import annotations

import pandas as pd

from goalline.constants import FIRST_DIVISIONS
from goalline.features.elo import EloParams, compute_elo
from goalline.features.form import compute_form

FEATURE_COLUMNS = [
    "elo_diff",
    "form_points_diff",
    "form_gd_diff",
    "short_history_home",
    "short_history_away",
    *[f"league_{d}" for d in FIRST_DIVISIONS],
]


def build_features(matches: pd.DataFrame, params: EloParams) -> pd.DataFrame:
    out = pd.concat([matches, compute_elo(matches, params), compute_form(matches)], axis=1)
    out["elo_diff"] = out.elo_home + params.home_advantage - out.elo_away
    out["form_points_diff"] = out.form_points_home - out.form_points_away
    out["form_gd_diff"] = out.form_gd_home - out.form_gd_away
    for division in FIRST_DIVISIONS:
        out[f"league_{division}"] = (out.division == division).astype(int)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/features -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/goalline/features tests/features tests/conftest.py
git commit -m "feat(features): recent form and a leakage-tested feature table"
```

---

### Task 8: Bookmaker benchmark

**Files:**
- Create: `src/goalline/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `ODDS_COLUMNS`.
- Produces: `proportional(odds: np.ndarray) -> np.ndarray`, `shin(odds: np.ndarray) -> tuple[np.ndarray, np.ndarray]` (probabilities n×3, z of length n), `Benchmark(method)` with `method ∈ {"shin", "proportional", "max_odds"}`, attributes `name == method`, `refit = "season"`, `fit(history) -> self`, `predict_proba(matches) -> np.ndarray` (rows with missing odds are NaN).

- [ ] **Step 1: Write the failing test**

`tests/test_benchmark.py`:

```python
import numpy as np
import pandas as pd
import pytest

from goalline.benchmark import Benchmark, proportional, shin


def shin_odds(p, z):
    """Odds generated by Shin's model from true probabilities p and insider share z."""
    root = np.sqrt(z * p + (1 - z) * p**2)
    return 1 / (root.sum() * root)


def test_proportional_divides_by_the_overround():
    odds = np.array([[1.8, 3.6, 4.5]])
    pi = 1 / odds
    assert np.allclose(proportional(odds), pi / pi.sum())


def test_shin_with_zero_margin_returns_the_implied_probabilities():
    p = np.array([0.5, 0.3, 0.2])
    probs, z = shin((1 / p)[None, :])
    assert np.allclose(probs[0], p, atol=1e-9)
    assert z[0] == pytest.approx(0, abs=1e-9)


def test_shin_recovers_the_parameters_of_its_own_model():
    p = np.array([0.5, 0.3, 0.2])
    probs, z = shin(shin_odds(p, 0.03)[None, :])
    assert np.allclose(probs[0], p, atol=1e-9)
    assert z[0] == pytest.approx(0.03, abs=1e-9)


def test_shin_moves_probability_towards_the_favourite():
    odds = np.array([[1.30, 5.50, 11.0]])
    s, _ = shin(odds)
    q = proportional(odds)
    assert s[0, 0] > q[0, 0]
    assert s[0, 2] < q[0, 2]


def test_shin_sums_to_one_for_any_realistic_odds():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([3, 2, 2], size=500)
    odds = 1 / (p * rng.uniform(1.02, 1.20, size=(500, 1)))
    probs, z = shin(odds)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-9)
    assert (probs > 0).all() and (z >= 0).all()


def test_benchmark_returns_nan_where_odds_are_missing():
    frame = pd.DataFrame(
        {"b365_h": [1.8, np.nan], "b365_d": [3.6, np.nan], "b365_a": [4.5, np.nan],
         "max_h": [1.9, 2.0], "max_d": [3.8, 3.5], "max_a": [4.8, 4.0]}
    )
    shin_p = Benchmark("shin").fit(frame).predict_proba(frame)
    assert np.isnan(shin_p[1]).all() and np.allclose(shin_p[0].sum(), 1)
    max_p = Benchmark("max_odds").predict_proba(frame)
    assert np.isfinite(max_p).all()


def test_benchmark_rejects_unknown_method():
    with pytest.raises(ValueError):
        Benchmark("closing")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goalline.benchmark'`

- [ ] **Step 3: Write minimal implementation**

`src/goalline/benchmark.py`:

```python
"""Turn bookmaker odds into probabilities: Shin's method and proportional normalisation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from goalline.constants import ODDS_COLUMNS


def proportional(odds: np.ndarray) -> np.ndarray:
    pi = 1.0 / odds
    return pi / pi.sum(axis=1, keepdims=True)


def shin(odds: np.ndarray, tol: float = 1e-13, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Shin probabilities p_i(z) = (sqrt(z² + 4(1−z)π_i²/β) − z) / (2(1−z)), with Σ p_i(z) = 1.

    Σ p_i(0) = sqrt(β) ≥ 1 and the sum decreases in z, so z is found by bisection on [0, 1).
    """
    pi = 1.0 / odds
    beta = pi.sum(axis=1, keepdims=True)

    def probs_at(z: np.ndarray) -> np.ndarray:
        z = z[:, None]
        return (np.sqrt(z**2 + 4 * (1 - z) * pi**2 / beta) - z) / (2 * (1 - z))

    lo = np.zeros(len(pi))
    hi = np.full(len(pi), 0.999)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        too_big = probs_at(mid).sum(axis=1) > 1
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
        if np.max(hi - lo) < tol:
            break
    z = np.where(beta[:, 0] <= 1, 0.0, (lo + hi) / 2)
    p = probs_at(z)
    return p / p.sum(axis=1, keepdims=True), z


class Benchmark:
    refit = "season"

    def __init__(self, method: str):
        if method not in ("shin", "proportional", "max_odds"):
            raise ValueError(f"unknown benchmark method {method!r}")
        self.method = method
        self.name = method

    def fit(self, history: pd.DataFrame) -> Benchmark:
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        columns = ODDS_COLUMNS["max" if self.method == "max_odds" else "b365"]
        odds = matches[list(columns)].to_numpy(dtype=float)
        out = np.full((len(odds), 3), np.nan)
        ok = ~np.isnan(odds).any(axis=1)
        if ok.any():
            out[ok] = shin(odds[ok])[0] if self.method == "shin" else proportional(odds[ok])
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_benchmark.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/goalline/benchmark.py tests/test_benchmark.py
git commit -m "feat(benchmark): Shin and proportional de-margining of bookmaker odds"
```

---

### Task 9: Model contract, uniform and frequency models

**Files:**
- Create: `src/goalline/models/__init__.py` (empty), `src/goalline/models/base.py`, `src/goalline/models/simple.py`
- Test: `tests/models/test_simple.py`, `tests/models/test_contract.py`

**Interfaces:**
- Consumes: `OUTCOME_INDEX`, `Benchmark` (Task 8), fixture `features`.
- Produces: `Model` (typing `Protocol` with `name: str`, `refit: str`, `fit(history) -> Model`, `predict_proba(matches) -> np.ndarray`); `outcome_codes(frame) -> np.ndarray[int]`; `first_division_rows(history, first_season: int) -> pd.DataFrame`; `Uniform()`; `Frequency(train_first_season: int = 2002)` with Laplace smoothing `(count + 1) / (n + 3)` per division and an overall fallback. `tests/models/test_contract.py` defines `CASES: list[tuple[str, Callable[[], Model]]]`, which later tasks extend.

- [ ] **Step 1: Write the failing tests**

`tests/models/test_simple.py`:

```python
import numpy as np
import pandas as pd

from goalline.models.base import first_division_rows, outcome_codes
from goalline.models.simple import Frequency, Uniform


def history(rows):
    return pd.DataFrame(rows, columns=["division", "tier", "season", "result"])


def test_outcome_codes():
    assert outcome_codes(pd.DataFrame({"result": ["H", "D", "A"]})).tolist() == [0, 1, 2]


def test_first_division_rows_filters_tier_and_season():
    h = history([("I1", 1, 2001, "H"), ("I2", 2, 2005, "H"), ("I1", 1, 2005, "D")])
    assert first_division_rows(h, 2002).result.tolist() == ["D"]


def test_uniform():
    p = Uniform().fit(history([])).predict_proba(pd.DataFrame({"division": ["I1", "E0"]}))
    assert np.allclose(p, 1 / 3)


def test_frequency_by_division_with_smoothing():
    h = history([("I1", 1, 2005, "H"), ("I1", 1, 2005, "H"), ("I1", 1, 2005, "D"),
                 ("E0", 1, 2005, "A"), ("I2", 2, 2005, "A"), ("I1", 1, 2000, "A")])
    model = Frequency(train_first_season=2002).fit(h)
    p = model.predict_proba(pd.DataFrame({"division": ["I1", "SP1"]}))
    assert np.allclose(p[0], [3 / 6, 2 / 6, 1 / 6])
    assert np.allclose(p[1], np.array([2 + 1, 1 + 1, 1 + 1]) / (4 + 3))
```

`tests/models/test_contract.py`:

```python
"""Every model and benchmark: rows sum to one, are strictly positive, and ignore row order."""

from collections.abc import Callable

import numpy as np
import pytest

from goalline.benchmark import Benchmark
from goalline.models.simple import Frequency, Uniform

CASES: list[tuple[str, Callable]] = [
    ("uniform", lambda: Uniform()),
    ("frequency", lambda: Frequency(train_first_season=2000)),
    ("shin", lambda: Benchmark("shin")),
    ("proportional", lambda: Benchmark("proportional")),
    ("max_odds", lambda: Benchmark("max_odds")),
]


@pytest.mark.parametrize(("name", "factory"), CASES, ids=[c[0] for c in CASES])
def test_contract(name, factory, features):
    history = features[features.season < 2004]
    target = features[(features.season == 2004) & (features.tier == 1)]
    model = factory()
    assert model.name == name
    assert model.refit in ("season", "month")
    p = model.fit(history).predict_proba(target)
    assert p.shape == (len(target), 3)
    assert np.allclose(p.sum(axis=1), 1, atol=1e-9)
    assert (p > 0).all()
    shuffled = target.sample(frac=1, random_state=0)
    q = model.predict_proba(shuffled)
    assert np.allclose(q, p[target.index.get_indexer(shuffled.index)])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/models -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goalline.models'`

- [ ] **Step 3: Write minimal implementation**

`src/goalline/models/base.py`:

```python
"""The interface every model and benchmark follows, and helpers shared by the learned models."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from goalline.constants import OUTCOME_INDEX


class Model(Protocol):
    name: str
    refit: str  # "season" or "month"

    def fit(self, history: pd.DataFrame) -> Model: ...

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray: ...


def outcome_codes(frame: pd.DataFrame) -> np.ndarray:
    return frame.result.map(OUTCOME_INDEX).to_numpy(dtype=int)


def first_division_rows(history: pd.DataFrame, first_season: int) -> pd.DataFrame:
    return history[(history.tier == 1) & (history.season >= first_season)]
```

`src/goalline/models/simple.py`:

```python
"""Naive baselines: a uniform guess and the outcome shares seen in training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from goalline.models.base import first_division_rows, outcome_codes


class Uniform:
    name = "uniform"
    refit = "season"

    def fit(self, history: pd.DataFrame) -> Uniform:
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        return np.full((len(matches), 3), 1 / 3)


def _shares(codes: np.ndarray) -> np.ndarray:
    return (np.bincount(codes, minlength=3) + 1) / (len(codes) + 3)


class Frequency:
    name = "frequency"
    refit = "season"

    def __init__(self, train_first_season: int = 2002):
        self.train_first_season = train_first_season

    def fit(self, history: pd.DataFrame) -> Frequency:
        rows = first_division_rows(history, self.train_first_season)
        self.overall_ = _shares(outcome_codes(rows))
        self.by_division_ = {d: _shares(outcome_codes(g)) for d, g in rows.groupby("division")}
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        return np.array([self.by_division_.get(d, self.overall_) for d in matches.division])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/models -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/goalline/models tests/models
git commit -m "feat(models): model contract with uniform and frequency baselines"
```

---

### Task 10: Elo model with draw curve

**Files:**
- Create: `src/goalline/models/elo_model.py`
- Modify: `tests/models/test_contract.py` (add a case)
- Test: `tests/models/test_elo_model.py`

**Interfaces:**
- Consumes: feature column `elo_diff`.
- Produces: `DrawParams(peak: float, scale: float)` (frozen dataclass); `elo_probabilities(elo_diff: np.ndarray, draw: DrawParams) -> np.ndarray`; `EloModel(draw)` with `name = "elo"`, `refit = "season"`.

- [ ] **Step 1: Write the failing test**

`tests/models/test_elo_model.py`:

```python
import numpy as np
import pandas as pd

from goalline.models.elo_model import DrawParams, EloModel, elo_probabilities


def test_level_teams():
    p = elo_probabilities(np.array([0.0]), DrawParams(0.28, 400))
    assert np.allclose(p, [[0.36, 0.28, 0.36]])


def test_large_gap_favours_home_and_shrinks_draws():
    p = elo_probabilities(np.array([0.0, 400.0]), DrawParams(0.28, 400))
    assert p[1, 0] > 0.8 * (1 - p[1, 1])
    assert p[1, 1] < p[0, 1]


def test_model_reads_elo_diff():
    m = EloModel(DrawParams(0.28, 400)).fit(pd.DataFrame())
    p = m.predict_proba(pd.DataFrame({"elo_diff": [0.0]}))
    assert np.allclose(p, [[0.36, 0.28, 0.36]])
```

Add to `CASES` in `tests/models/test_contract.py` (and the import `from goalline.models.elo_model import DrawParams, EloModel`):

```python
    ("elo", lambda: EloModel(DrawParams(0.28, 400))),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/models -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goalline.models.elo_model'`

- [ ] **Step 3: Write minimal implementation**

`src/goalline/models/elo_model.py`:

```python
"""Elo expected score split into three outcomes by a Gaussian draw curve on the rating gap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DrawParams:
    peak: float
    scale: float


def elo_probabilities(elo_diff: np.ndarray, draw: DrawParams) -> np.ndarray:
    expected = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    p_draw = draw.peak * np.exp(-0.5 * (elo_diff / draw.scale) ** 2)
    return np.column_stack([expected * (1 - p_draw), p_draw, (1 - expected) * (1 - p_draw)])


class EloModel:
    name = "elo"
    refit = "season"

    def __init__(self, draw: DrawParams):
        self.draw = draw

    def fit(self, history: pd.DataFrame) -> EloModel:
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        return elo_probabilities(matches.elo_diff.to_numpy(dtype=float), self.draw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/models -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add src/goalline/models/elo_model.py tests/models
git commit -m "feat(models): Elo model with a Gaussian draw curve"
```

---

### Task 11: Dixon–Coles

**Files:**
- Create: `src/goalline/models/dixon_coles.py`
- Modify: `tests/models/test_contract.py` (add a case)
- Test: `tests/models/test_dixon_coles.py`

**Interfaces:**
- Consumes: canonical columns `country, date, home, away, home_goals, away_goals`.
- Produces: `MAX_GOALS = 10`, `WINDOW_DAYS = 1096`, `MIN_MATCHES = 20`; `negative_log_likelihood(theta, hi, ai, x, y, w, n_teams) -> tuple[float, np.ndarray]` (without the constant log-factorial terms); `DixonColesFit` (dataclass: `teams: dict[str, int], attack, defence, home: float, rho: float`, method `rates(home, away) -> tuple[np.ndarray, np.ndarray]`); `fit_dixon_coles(rows, reference: pd.Timestamp, xi: float) -> DixonColesFit`; `outcome_probabilities(lam, mu, rho) -> np.ndarray`; `DixonColesModel(xi)` with `name = "dixon_coles"`, `refit = "month"`. The reference date for the weights is the day after the latest match in `history`.

- [ ] **Step 1: Write the failing test**

`tests/models/test_dixon_coles.py`:

```python
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from goalline.models.dixon_coles import (
    DixonColesModel,
    fit_dixon_coles,
    negative_log_likelihood,
    outcome_probabilities,
)

TRUE_ATTACK = np.array([0.3, 0.2, 0.1, 0.0, 0.0, -0.1, -0.2, -0.3])
TRUE_DEFENCE = np.array([-0.3, -0.2, -0.1, 0.0, 0.0, 0.1, 0.2, 0.3])
TRUE_HOME = 0.3


def simulated(country="ITA", seed=1):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    day = pd.Timestamp("2010-01-01")
    for _ in range(10):
        for i in range(8):
            for j in range(8):
                if i == j:
                    continue
                lam = np.exp(TRUE_HOME + TRUE_ATTACK[i] + TRUE_DEFENCE[j])
                mu = np.exp(TRUE_ATTACK[j] + TRUE_DEFENCE[i])
                rows.append((country, day, teams[i], teams[j], rng.poisson(lam), rng.poisson(mu)))
                day += pd.Timedelta(days=1)
    return pd.DataFrame(rows, columns=["country", "date", "home", "away", "home_goals", "away_goals"])


def test_likelihood_hand_values():
    hi, ai = np.array([0, 1]), np.array([1, 0])
    x, y = np.array([1.0, 2.0]), np.array([0.0, 2.0])
    w = np.array([1.0, 0.5])
    theta = np.zeros(2 * 2 + 2)
    value, _ = negative_log_likelihood(theta, hi, ai, x, y, w, 2)
    assert value == pytest.approx(3.0)
    theta[-1] = 0.1
    value, _ = negative_log_likelihood(theta, hi[:1], ai[:1], np.array([0.0]), np.array([0.0]),
                                       np.array([1.0]), 2)
    assert value == pytest.approx(2 - np.log(0.9))


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    n, m = 4, 40
    hi, ai = rng.integers(0, n, m), rng.integers(0, n, m)
    x, y = rng.integers(0, 3, m).astype(float), rng.integers(0, 3, m).astype(float)
    w = rng.uniform(0.2, 1.0, m)
    theta = rng.normal(0, 0.2, 2 * n + 2)
    theta[-1] = 0.05
    _, grad = negative_log_likelihood(theta, hi, ai, x, y, w, n)
    numeric = approx_fprime(theta, lambda t: negative_log_likelihood(t, hi, ai, x, y, w, n)[0], 1e-7)
    assert np.allclose(grad, numeric, rtol=1e-4, atol=1e-5)


def test_parameter_recovery():
    rows = simulated()
    fit = fit_dixon_coles(rows, rows.date.max() + pd.Timedelta(days=1), xi=0.0)
    order = [fit.teams[f"T{i}"] for i in range(8)]
    assert fit.attack.sum() == pytest.approx(0, abs=1e-9)
    assert fit.home == pytest.approx(TRUE_HOME, abs=0.1)
    assert np.corrcoef(fit.attack[order], TRUE_ATTACK)[0, 1] > 0.9
    assert np.corrcoef(fit.defence[order], TRUE_DEFENCE)[0, 1] > 0.9


def test_outcome_probabilities_sum_to_one():
    p = outcome_probabilities(np.array([1.5, 0.4]), np.array([1.0, 2.2]), 0.05)
    assert np.allclose(p.sum(axis=1), 1)
    assert p[0, 0] > p[0, 2] and p[1, 2] > p[1, 0]


def test_unknown_team_gets_average_strength():
    rows = simulated()
    model = DixonColesModel(xi=0.0).fit(rows)
    target = pd.DataFrame({"country": ["ITA"], "home": ["NEW"], "away": ["T0"]})
    p = model.predict_proba(target)
    assert np.isfinite(p).all() and np.allclose(p.sum(), 1)


def test_country_without_history_falls_back_to_uniform():
    model = DixonColesModel(xi=0.0).fit(simulated("ITA"))
    p = model.predict_proba(pd.DataFrame({"country": ["ENG"], "home": ["X"], "away": ["Y"]}))
    assert np.allclose(p, 1 / 3)
```

Add to `CASES` in `tests/models/test_contract.py` (and the import `from goalline.models.dixon_coles import DixonColesModel`):

```python
    ("dixon_coles", lambda: DixonColesModel(xi=0.0019)),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/models/test_dixon_coles.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goalline.models.dixon_coles'`

- [ ] **Step 3: Write minimal implementation**

`src/goalline/models/dixon_coles.py`:

```python
"""Dixon–Coles: Poisson goals with team attack/defence, home effect and low-score correction ρ.

log λ = home + attack[home team] + defence[away team]; log μ = attack[away team] + defence[home team].
One fit per country on its first and second divisions, weighted by exp(−ξ · days before the
reference date), over the last WINDOW_DAYS (about three seasons).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10
WINDOW_DAYS = 1096
MIN_MATCHES = 20


def negative_log_likelihood(theta, hi, ai, x, y, w, n_teams):
    n = n_teams
    attack, defence = theta[:n], theta[n : 2 * n]
    home, rho = theta[2 * n], theta[2 * n + 1]
    eta_h = home + attack[hi] + defence[ai]
    eta_a = attack[ai] + defence[hi]
    lam, mu = np.exp(eta_h), np.exp(eta_a)

    tau = np.ones_like(lam)
    d_h = np.zeros_like(lam)
    d_a = np.zeros_like(lam)
    d_rho = np.zeros_like(lam)
    m = (x == 0) & (y == 0)
    tau[m] = 1 - lam[m] * mu[m] * rho
    d_h[m] = d_a[m] = -lam[m] * mu[m] * rho
    d_rho[m] = -lam[m] * mu[m]
    m = (x == 0) & (y == 1)
    tau[m] = 1 + lam[m] * rho
    d_h[m] = lam[m] * rho
    d_rho[m] = lam[m]
    m = (x == 1) & (y == 0)
    tau[m] = 1 + mu[m] * rho
    d_a[m] = mu[m] * rho
    d_rho[m] = mu[m]
    m = (x == 1) & (y == 1)
    tau[m] = 1 - rho
    d_rho[m] = -1.0
    tau = np.maximum(tau, 1e-10)

    ll = w * (np.log(tau) + x * eta_h - lam + y * eta_a - mu)
    g_h = w * (x - lam + d_h / tau)
    g_a = w * (y - mu + d_a / tau)
    grad = np.concatenate(
        [
            np.bincount(hi, g_h, n) + np.bincount(ai, g_a, n),
            np.bincount(ai, g_h, n) + np.bincount(hi, g_a, n),
            [g_h.sum(), (w * d_rho / tau).sum()],
        ]
    )
    return -ll.sum(), -grad


@dataclass
class DixonColesFit:
    teams: dict[str, int]
    attack: np.ndarray
    defence: np.ndarray
    home: float
    rho: float

    def rates(self, home, away) -> tuple[np.ndarray, np.ndarray]:
        mean_defence = float(self.defence.mean())

        def lookup(values, teams, default):
            return np.array([values[self.teams[t]] if t in self.teams else default for t in teams])

        att_h, att_a = lookup(self.attack, home, 0.0), lookup(self.attack, away, 0.0)
        def_h = lookup(self.defence, home, mean_defence)
        def_a = lookup(self.defence, away, mean_defence)
        return np.exp(self.home + att_h + def_a), np.exp(att_a + def_h)


def fit_dixon_coles(rows: pd.DataFrame, reference: pd.Timestamp, xi: float) -> DixonColesFit:
    teams = pd.Index(sorted(set(rows.home) | set(rows.away)))
    n = len(teams)
    hi, ai = teams.get_indexer(rows.home), teams.get_indexer(rows.away)
    x = rows.home_goals.to_numpy(dtype=float)
    y = rows.away_goals.to_numpy(dtype=float)
    w = np.exp(-xi * (reference - rows.date).dt.days.to_numpy(dtype=float))
    theta0 = np.zeros(2 * n + 2)
    theta0[2 * n] = 0.25
    bounds = [(-3.0, 3.0)] * (2 * n) + [(-1.0, 1.0), (-0.2, 0.2)]
    res = minimize(
        negative_log_likelihood, theta0, args=(hi, ai, x, y, w, n),
        jac=True, method="L-BFGS-B", bounds=bounds,
    )
    attack, defence = res.x[:n].copy(), res.x[n : 2 * n].copy()
    shift = attack.mean()
    attack -= shift
    defence += shift
    return DixonColesFit(
        teams=dict(zip(teams, range(n), strict=True)), attack=attack, defence=defence,
        home=float(res.x[2 * n]), rho=float(res.x[2 * n + 1]),
    )


def outcome_probabilities(lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    goals = np.arange(MAX_GOALS + 1)
    px = poisson.pmf(goals[None, :], lam[:, None])
    py = poisson.pmf(goals[None, :], mu[:, None])
    grid = px[:, :, None] * py[:, None, :]
    grid[:, 0, 0] *= 1 - lam * mu * rho
    grid[:, 0, 1] *= 1 + lam * rho
    grid[:, 1, 0] *= 1 + mu * rho
    grid[:, 1, 1] *= 1 - rho
    grid = np.clip(grid, 0, None)
    grid /= grid.sum(axis=(1, 2), keepdims=True)
    home_wins = goals[:, None] > goals[None, :]
    return np.column_stack(
        [
            (grid * home_wins).sum(axis=(1, 2)),
            np.trace(grid, axis1=1, axis2=2),
            (grid * home_wins.T).sum(axis=(1, 2)),
        ]
    )


class DixonColesModel:
    name = "dixon_coles"
    refit = "month"

    def __init__(self, xi: float):
        self.xi = xi

    def fit(self, history: pd.DataFrame) -> DixonColesModel:
        self.fits_: dict[str, DixonColesFit] = {}
        if history.empty:
            return self
        reference = history.date.max() + pd.Timedelta(days=1)
        start = reference - pd.Timedelta(days=WINDOW_DAYS)
        for country, rows in history[history.date >= start].groupby("country"):
            if len(rows) >= MIN_MATCHES:
                self.fits_[country] = fit_dixon_coles(rows, reference, self.xi)
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        out = np.full((len(matches), 3), 1 / 3)
        for country, rows in matches.groupby("country"):
            fit = self.fits_.get(country)
            if fit is None:
                continue
            lam, mu = fit.rates(rows.home, rows.away)
            out[matches.index.get_indexer(rows.index)] = outcome_probabilities(lam, mu, fit.rho)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/models -v`
Expected: all model tests pass (20 passed)

- [ ] **Step 5: Commit**

```bash
git add src/goalline/models/dixon_coles.py tests/models
git commit -m "feat(models): Dixon-Coles with analytic gradient, monthly refit and fallbacks"
```

---

### Task 12: Logistic regression

**Files:**
- Create: `src/goalline/models/logistic.py`
- Modify: `tests/models/test_contract.py` (add a case)
- Test: `tests/models/test_logistic.py`

**Interfaces:**
- Consumes: `FEATURE_COLUMNS` (Task 7), `first_division_rows`, `outcome_codes` (Task 9).
- Produces: `LogisticModel(c: float, train_first_season: int = 2002)` with `name = "logistic"`, `refit = "season"`.

- [ ] **Step 1: Write the failing test**

`tests/models/test_logistic.py`:

```python
from goalline.models.logistic import LogisticModel


def test_higher_elo_gap_raises_home_win_probability(features):
    model = LogisticModel(c=1.0, train_first_season=2000).fit(features[features.season < 2004])
    row = features[(features.season == 2004) & (features.tier == 1)].iloc[[0, 0]].copy()
    row["elo_diff"] = [-400.0, 400.0]
    p = model.predict_proba(row)
    assert p[1, 0] > p[0, 0]
    assert p[1, 2] < p[0, 2]


def test_trains_only_on_first_division_rows_from_the_first_season(features):
    model = LogisticModel(c=1.0, train_first_season=2001).fit(features[features.season < 2004])
    n_expected = len(features[(features.season.between(2001, 2003)) & (features.tier == 1)])
    assert model.n_train_ == n_expected
```

Add to `CASES` (and `from goalline.models.logistic import LogisticModel`):

```python
    ("logistic", lambda: LogisticModel(c=1.0, train_first_season=2000)),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/models/test_logistic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goalline.models.logistic'`

- [ ] **Step 3: Write minimal implementation**

`src/goalline/models/logistic.py`:

```python
"""Multinomial logistic regression on the standardised feature table."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from goalline.features.table import FEATURE_COLUMNS
from goalline.models.base import first_division_rows, outcome_codes


class LogisticModel:
    name = "logistic"
    refit = "season"

    def __init__(self, c: float, train_first_season: int = 2002):
        self.c = c
        self.train_first_season = train_first_season

    def fit(self, history: pd.DataFrame) -> LogisticModel:
        rows = first_division_rows(history, self.train_first_season)
        self.n_train_ = len(rows)
        self.pipeline_ = make_pipeline(
            StandardScaler(), LogisticRegression(C=self.c, max_iter=2000)
        ).fit(rows[FEATURE_COLUMNS].to_numpy(dtype=float), outcome_codes(rows))
        if list(self.pipeline_.classes_) != [0, 1, 2]:
            raise ValueError("training data must contain home wins, draws and away wins")
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(matches[FEATURE_COLUMNS].to_numpy(dtype=float))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/models -v`
Expected: all model tests pass (23 passed)

- [ ] **Step 5: Commit**

```bash
git add src/goalline/models/logistic.py tests/models
git commit -m "feat(models): multinomial logistic regression on the feature table"
```

---

### Task 13: XGBoost

**Files:**
- Create: `src/goalline/models/xgb.py`
- Modify: `tests/models/test_contract.py` (add a case)
- Test: `tests/models/test_xgb.py`

**Interfaces:**
- Consumes: as Task 12.
- Produces: `XGBoostModel(max_depth: int, learning_rate: float, min_child_weight: float, train_first_season: int = 2002, seed: int = 42)` with `name = "xgboost"`, `refit = "season"`; after `fit`, attributes `validation_season_: int`, `train_seasons_: tuple[int, int]`, `model_` (fitted `xgboost.XGBClassifier`).

- [ ] **Step 1: Write the failing test**

`tests/models/test_xgb.py`:

```python
import numpy as np

from goalline.models.xgb import XGBoostModel


def make():
    return XGBoostModel(max_depth=2, learning_rate=0.1, min_child_weight=1, train_first_season=2000)


def test_early_stopping_uses_the_last_training_season(features):
    model = make().fit(features[features.season < 2004])
    assert model.validation_season_ == 2003
    assert model.train_seasons_ == (2000, 2002)
    assert model.model_.best_iteration < 1000


def test_is_deterministic(features):
    history = features[features.season < 2004]
    target = features[(features.season == 2004) & (features.tier == 1)]
    a = make().fit(history).predict_proba(target)
    b = make().fit(history).predict_proba(target)
    assert np.array_equal(a, b)
```

Add to `CASES` (and `from goalline.models.xgb import XGBoostModel`):

```python
    ("xgboost", lambda: XGBoostModel(2, 0.1, 1, train_first_season=2000)),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/models/test_xgb.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goalline.models.xgb'`

- [ ] **Step 3: Write minimal implementation**

`src/goalline/models/xgb.py`:

```python
"""Gradient-boosted trees on the feature table, early-stopped on the last training season."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from goalline.constants import SEED
from goalline.features.table import FEATURE_COLUMNS
from goalline.models.base import first_division_rows, outcome_codes


class XGBoostModel:
    name = "xgboost"
    refit = "season"

    def __init__(
        self,
        max_depth: int,
        learning_rate: float,
        min_child_weight: float,
        train_first_season: int = 2002,
        seed: int = SEED,
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.train_first_season = train_first_season
        self.seed = seed

    def fit(self, history: pd.DataFrame) -> XGBoostModel:
        rows = first_division_rows(history, self.train_first_season)
        self.validation_season_ = int(rows.season.max())
        train = rows[rows.season < self.validation_season_]
        valid = rows[rows.season == self.validation_season_]
        self.train_seasons_ = (int(train.season.min()), int(train.season.max()))
        self.model_ = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=1000,
            early_stopping_rounds=50,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=0.8,
            tree_method="hist",
            random_state=self.seed,
            n_jobs=-1,
        )
        self.model_.fit(
            train[FEATURE_COLUMNS].to_numpy(dtype=float), outcome_codes(train),
            eval_set=[(valid[FEATURE_COLUMNS].to_numpy(dtype=float), outcome_codes(valid))],
            verbose=False,
        )
        return self

    def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(matches[FEATURE_COLUMNS].to_numpy(dtype=float))
```

The `hist` tree method is deterministic with several threads; `test_is_deterministic` guards this.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/models -v`
Expected: all model tests pass (26 passed)

- [ ] **Step 5: Commit**

```bash
git add src/goalline/models/xgb.py tests/models
git commit -m "feat(models): XGBoost with early stopping on the last training season"
```

---

### Task 14: Metrics and reliability

**Files:**
- Create: `src/goalline/evaluation/__init__.py` (empty), `src/goalline/evaluation/metrics.py`, `src/goalline/evaluation/reliability.py`
- Test: `tests/evaluation/test_metrics.py`, `tests/evaluation/test_reliability.py`

**Interfaces:**
- Produces: `log_loss(p, y) -> np.ndarray`, `rps(p, y) -> np.ndarray`, `brier(p, y) -> np.ndarray` (all per match; `p` n×3, `y` int codes). `reliability_table(p, y, bins=10) -> pd.DataFrame[outcome, bin, n, mean_predicted, observed]`; `expected_calibration_error(table) -> pd.Series` indexed by outcome.

- [ ] **Step 1: Write the failing tests**

`tests/evaluation/test_metrics.py`:

```python
import numpy as np
import pytest

from goalline.evaluation.metrics import brier, log_loss, rps

H = np.array([0])


def test_perfect_forecast_scores_zero():
    p = np.array([[1.0, 0.0, 0.0]])
    assert log_loss(p, H)[0] == pytest.approx(0)
    assert rps(p, H)[0] == pytest.approx(0)
    assert brier(p, H)[0] == pytest.approx(0)


def test_uniform_log_loss_is_ln3():
    assert log_loss(np.full((1, 3), 1 / 3), H)[0] == pytest.approx(np.log(3))


@pytest.mark.parametrize(
    ("forecast", "expected"),
    [([0.9, 0.1, 0.0], 0.005), ([0.8, 0.1, 0.1], 0.025), ([0.5, 0.25, 0.25], 0.15625),
     ([0.35, 0.3, 0.35], 0.2725)],
)
def test_rps_hand_values_for_a_home_win(forecast, expected):
    """RPS = 1/(r−1) Σ (cumulative forecast − cumulative outcome)², as in Constantinou & Fenton (2012)."""
    assert rps(np.array([forecast]), H)[0] == pytest.approx(expected)


def test_rps_rewards_the_nearer_outcome():
    draw_heavy = np.array([[0.2, 0.6, 0.2]])
    away_heavy = np.array([[0.2, 0.2, 0.6]])
    assert rps(draw_heavy, H)[0] < rps(away_heavy, H)[0]


def test_brier_sums_over_outcomes():
    assert brier(np.array([[0.5, 0.25, 0.25]]), H)[0] == pytest.approx(0.25 + 0.0625 + 0.0625)


def test_log_loss_clips_zero_probabilities():
    assert np.isfinite(log_loss(np.array([[0.0, 0.5, 0.5]]), H)[0])
```

`tests/evaluation/test_reliability.py`:

```python
import numpy as np
import pytest

from goalline.evaluation.reliability import expected_calibration_error, reliability_table


def test_equal_count_bins_and_columns():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([2, 1, 1], size=1000)
    y = np.array([rng.choice(3, p=row) for row in p])
    table = reliability_table(p, y)
    assert list(table.columns) == ["outcome", "bin", "n", "mean_predicted", "observed"]
    assert set(table.outcome) == {"H", "D", "A"}
    assert (table.groupby("outcome").n.sum() == 1000).all()
    assert (table.n == 100).all()


def test_perfectly_calibrated_constant_forecast_has_zero_error():
    p = np.tile([0.5, 0.25, 0.25], (400, 1))
    y = np.array([0] * 200 + [1] * 100 + [2] * 100)
    ece = expected_calibration_error(reliability_table(p, y, bins=1))
    assert ece["H"] == pytest.approx(0)
    assert ece["D"] == pytest.approx(0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/evaluation -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goalline.evaluation'`

- [ ] **Step 3: Write minimal implementation**

`src/goalline/evaluation/metrics.py`:

```python
"""Per-match scoring rules for three-way forecasts (outcome order home, draw, away)."""

from __future__ import annotations

import numpy as np


def _one_hot(y: np.ndarray) -> np.ndarray:
    return np.eye(3)[y]


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0))


def rps(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    cumulative_p = np.cumsum(p, axis=1)[:, :2]
    cumulative_o = np.cumsum(_one_hot(y), axis=1)[:, :2]
    return ((cumulative_p - cumulative_o) ** 2).sum(axis=1) / 2


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((p - _one_hot(y)) ** 2).sum(axis=1)
```

`src/goalline/evaluation/reliability.py`:

```python
"""Reliability tables: predicted probability against observed frequency, per outcome."""

from __future__ import annotations

import numpy as np
import pandas as pd

from goalline.constants import OUTCOMES


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    rows = []
    for k, outcome in enumerate(OUTCOMES):
        order = np.argsort(p[:, k], kind="stable")
        for b, idx in enumerate(np.array_split(order, bins)):
            rows.append(
                {
                    "outcome": outcome,
                    "bin": b,
                    "n": len(idx),
                    "mean_predicted": float(p[idx, k].mean()),
                    "observed": float((y[idx] == k).mean()),
                }
            )
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> pd.Series:
    gap = (table.mean_predicted - table.observed).abs() * table.n
    return gap.groupby(table.outcome).sum() / table.n.groupby(table.outcome).sum()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/evaluation -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/goalline/evaluation tests/evaluation
git commit -m "feat(evaluation): log loss, RPS, Brier and reliability tables"
```

---

### Task 15: Paired block bootstrap

**Files:**
- Create: `src/goalline/evaluation/bootstrap.py`
- Test: `tests/evaluation/test_bootstrap.py`

**Interfaces:**
- Produces: `matchday_clusters(frame) -> np.ndarray[str]` (`"{division}-{iso_year}-{iso_week}"`); `bootstrap_replicates(diff, clusters, replicates, seed) -> np.ndarray`; `BootstrapResult` (frozen dataclass: `mean, lower, upper: float; n_matches, n_clusters: int`); `paired_bootstrap(loss_model, loss_reference, clusters, replicates=2000, seed=42) -> BootstrapResult`.

- [ ] **Step 1: Write the failing test**

`tests/evaluation/test_bootstrap.py`:

```python
import numpy as np
import pandas as pd
import pytest

from goalline.evaluation.bootstrap import bootstrap_replicates, matchday_clusters, paired_bootstrap


def test_clusters_are_league_and_iso_week():
    frame = pd.DataFrame(
        {"division": ["I1", "I1", "E0", "I1"],
         "date": pd.to_datetime(["2021-09-11", "2021-09-12", "2021-09-12", "2021-09-19"])}
    )
    c = matchday_clusters(frame)
    assert c[0] == c[1]
    assert c[1] != c[2]
    assert c[1] != c[3]


def test_identical_models_give_zero():
    loss = np.random.default_rng(0).uniform(0.5, 1.5, 300)
    clusters = np.repeat(np.arange(30), 10).astype(str)
    r = paired_bootstrap(loss, loss.copy(), clusters, replicates=200)
    assert r.mean == r.lower == r.upper == 0
    assert (r.n_matches, r.n_clusters) == (300, 30)


def test_whole_clusters_are_resampled():
    diff = np.array([0.0] + [1.0] * 99)
    clusters = np.array(["a"] + ["b"] * 99)
    stats = bootstrap_replicates(diff, clusters, replicates=500, seed=1)
    assert set(np.round(stats, 9)) <= {0.0, 0.99, 1.0}


def test_interval_contains_the_mean_and_is_seeded():
    rng = np.random.default_rng(2)
    a, b = rng.normal(1.0, 0.3, 500), rng.normal(1.05, 0.3, 500)
    clusters = np.repeat(np.arange(50), 10).astype(str)
    r1 = paired_bootstrap(a, b, clusters, replicates=500, seed=42)
    r2 = paired_bootstrap(a, b, clusters, replicates=500, seed=42)
    assert r1 == r2
    assert r1.lower < r1.mean < r1.upper
    assert r1.mean == pytest.approx((a - b).mean())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/evaluation/test_bootstrap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goalline.evaluation.bootstrap'`

- [ ] **Step 3: Write minimal implementation**

`src/goalline/evaluation/bootstrap.py`:

```python
"""Paired bootstrap of per-match loss differences, resampling whole matchdays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from goalline.constants import SEED


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    lower: float
    upper: float
    n_matches: int
    n_clusters: int


def matchday_clusters(frame: pd.DataFrame) -> np.ndarray:
    iso = frame.date.dt.isocalendar()
    return (
        frame.division.astype(str) + "-" + iso.year.astype(str) + "-" + iso.week.astype(str)
    ).to_numpy()


def bootstrap_replicates(
    diff: np.ndarray, clusters: np.ndarray, replicates: int, seed: int = SEED
) -> np.ndarray:
    codes, uniques = pd.factorize(clusters)
    sums = np.bincount(codes, weights=diff)
    counts = np.bincount(codes)
    rng = np.random.default_rng(seed)
    out = np.empty(replicates)
    for start in range(0, replicates, 200):
        stop = min(start + 200, replicates)
        idx = rng.integers(0, len(uniques), size=(stop - start, len(uniques)))
        out[start:stop] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return out


def paired_bootstrap(
    loss_model: np.ndarray,
    loss_reference: np.ndarray,
    clusters: np.ndarray,
    replicates: int = 2000,
    seed: int = SEED,
) -> BootstrapResult:
    diff = np.asarray(loss_model, dtype=float) - np.asarray(loss_reference, dtype=float)
    stats = bootstrap_replicates(diff, clusters, replicates, seed)
    lower, upper = np.percentile(stats, [2.5, 97.5])
    return BootstrapResult(
        mean=float(diff.mean()), lower=float(lower), upper=float(upper),
        n_matches=len(diff), n_clusters=len(pd.unique(clusters)),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/evaluation/test_bootstrap.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/goalline/evaluation/bootstrap.py tests/evaluation/test_bootstrap.py
git commit -m "feat(evaluation): paired bootstrap that resamples whole matchdays"
```

---

### Task 16: Walk-forward

**Files:**
- Create: `src/goalline/evaluation/walkforward.py`
- Test: `tests/evaluation/test_walkforward.py`

**Interfaces:**
- Consumes: `Settings`, any `Model`, `Benchmark`.
- Produces: `FinalSeasonError(RuntimeError)`; `evaluation_rows(frame, seasons) -> pd.DataFrame` (first division, all three Bet365 odds present, season in `seasons`); `blocks(rows, refit) -> list[tuple[pd.Timestamp, pd.DataFrame]]`; `walk_forward(frame, model, seasons, *, mode, settings) -> pd.DataFrame[match_id, model, p_home, p_draw, p_away]`; `predict_all(frame, models, seasons, *, mode, settings) -> pd.DataFrame` (concatenation).

- [ ] **Step 1: Write the failing test**

`tests/evaluation/test_walkforward.py`:

```python
import numpy as np
import pandas as pd
import pytest
from helpers import SMOKE

from goalline.benchmark import Benchmark
from goalline.evaluation.walkforward import FinalSeasonError, evaluation_rows, walk_forward
from goalline.models.elo_model import DrawParams, EloModel
from goalline.models.simple import Uniform


class Spy:
    name = "spy"

    def __init__(self, refit):
        self.refit = refit
        self.calls = []

    def fit(self, history):
        self.history_end = history.date.max()
        return self

    def predict_proba(self, matches):
        self.calls.append((self.history_end, matches.date.min(), matches.date.max()))
        return np.full((len(matches), 3), 1 / 3)


def test_develop_mode_refuses_test_seasons(features):
    with pytest.raises(FinalSeasonError):
        walk_forward(features, Uniform(), (2003, 2004), mode="develop", settings=SMOKE)


def test_evaluation_rows_are_first_division_with_odds(features):
    rows = evaluation_rows(features, (2004,))
    assert set(rows.tier) == {1} and set(rows.season) == {2004}
    assert rows[["b365_h", "b365_d", "b365_a"]].notna().all(axis=None)


def test_season_refit_sees_only_previous_seasons(features):
    spy = Spy("season")
    walk_forward(features, spy, (2004, 2005), mode="final", settings=SMOKE)
    assert len(spy.calls) == 2
    for history_end, first, _ in spy.calls:
        assert history_end < pd.Timestamp(f"{first.year if first.month >= 7 else first.year - 1}-07-01")


def test_month_refit_sees_only_earlier_months(features):
    spy = Spy("month")
    walk_forward(features, spy, (2004,), mode="final", settings=SMOKE)
    assert len(spy.calls) > 2
    for history_end, first, last in spy.calls:
        assert (first.year, first.month) == (last.year, last.month)
        assert history_end < first.replace(day=1)


def test_output_columns_and_rows(features):
    out = walk_forward(features, EloModel(DrawParams(0.28, 400)), (2004,), mode="final",
                       settings=SMOKE)
    assert list(out.columns) == ["match_id", "model", "p_home", "p_draw", "p_away"]
    assert set(out.model) == {"elo"}
    assert len(out) == len(evaluation_rows(features, (2004,)))


def test_rows_without_probabilities_are_dropped(features):
    frame = features.copy()
    target = evaluation_rows(frame, (2004,)).index[0]
    frame.loc[target, ["max_h", "max_d", "max_a"]] = np.nan
    out = walk_forward(frame, Benchmark("max_odds"), (2004,), mode="final", settings=SMOKE)
    assert len(out) == len(evaluation_rows(frame, (2004,))) - 1
    assert frame.loc[target, "match_id"] not in set(out.match_id)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/evaluation/test_walkforward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goalline.evaluation.walkforward'`

- [ ] **Step 3: Write minimal implementation**

`src/goalline/evaluation/walkforward.py`:

```python
"""Walk-forward prediction: every block is predicted from matches played before it started."""

from __future__ import annotations

import numpy as np
import pandas as pd

from goalline.constants import ODDS_COLUMNS
from goalline.settings import Settings


class FinalSeasonError(RuntimeError):
    pass


def evaluation_rows(frame: pd.DataFrame, seasons) -> pd.DataFrame:
    has_odds = frame[list(ODDS_COLUMNS["b365"])].notna().all(axis=1)
    return frame[(frame.tier == 1) & frame.season.isin(seasons) & has_odds]


def blocks(rows: pd.DataFrame, refit: str) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    if refit == "season":
        return [(pd.Timestamp(f"{s}-07-01"), g) for s, g in rows.groupby("season")]
    if refit == "month":
        months = rows.date.dt.to_period("M")
        return [(period.start_time, g) for period, g in rows.groupby(months)]
    raise ValueError(f"unknown refit {refit!r}")


def walk_forward(frame, model, seasons, *, mode: str, settings: Settings) -> pd.DataFrame:
    if mode == "develop" and max(seasons) >= settings.final_first_season:
        raise FinalSeasonError(
            f"develop mode cannot score season {max(seasons)}: "
            f"the final test starts in {settings.final_first_season}"
        )
    parts = []
    for start, block in blocks(evaluation_rows(frame, seasons), model.refit):
        p = model.fit(frame[frame.date < start]).predict_proba(block)
        parts.append(
            pd.DataFrame(
                {"match_id": block.match_id.to_numpy(), "model": model.name,
                 "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2]}
            )
        )
    out = pd.concat(parts, ignore_index=True)
    keep = np.isfinite(out[["p_home", "p_draw", "p_away"]].to_numpy()).all(axis=1)
    return out[keep].reset_index(drop=True)


def predict_all(frame, models, seasons, *, mode: str, settings: Settings) -> pd.DataFrame:
    return pd.concat(
        [walk_forward(frame, m, seasons, mode=mode, settings=settings) for m in models],
        ignore_index=True,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/evaluation/test_walkforward.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/goalline/evaluation/walkforward.py tests/evaluation/test_walkforward.py
git commit -m "feat(evaluation): walk-forward prediction with season and month refits"
```

---

### Task 17: Scores, summaries, comparisons and home advantage

**Files:**
- Create: `src/goalline/evaluation/outputs.py`
- Test: `tests/evaluation/test_outputs.py`

**Interfaces:**
- Consumes: metrics (Task 14), reliability (Task 14), bootstrap (Task 15), `outcome_codes`, constants `MODEL_NAMES`, `MAIN_MODELS`, `REFERENCES`.
- Produces:
  - `score_predictions(preds, frame) -> pd.DataFrame`: predictions plus `division, season, date, result, log_loss, rps, brier`.
  - `summarise(scores) -> pd.DataFrame[model, group_type, group, n, log_loss, rps, brier]`, with `group_type ∈ {overall, division, season}`; `group` is a string (`"all"`, a division code, or the season start year).
  - `reliability_all(scores) -> pd.DataFrame[model, outcome, bin, n, mean_predicted, observed]`.
  - `comparisons(scores, frame, replicates, seed=42) -> pd.DataFrame[model, reference, metric, mean, lower, upper, n_matches, n_clusters]`: every model in `MODEL_NAMES` against each of `REFERENCES`, plus the six pairs of `MAIN_MODELS`, for the metrics `log_loss` and `rps`.
  - `home_advantage(frame) -> pd.DataFrame[division, season, n, home_share, draw_share, away_share]`.
  - `write_outputs(directory, preds, frame, settings) -> dict[str, pd.DataFrame]`: writes `predictions.csv`, `metrics.csv`, `reliability.csv`, `bootstrap.csv`.

- [ ] **Step 1: Write the failing test**

`tests/evaluation/test_outputs.py`:

```python
import numpy as np
import pandas as pd
import pytest
from helpers import SMOKE

from goalline.benchmark import Benchmark
from goalline.evaluation.outputs import (
    comparisons,
    home_advantage,
    score_predictions,
    summarise,
    write_outputs,
)
from goalline.evaluation.walkforward import predict_all
from goalline.models.elo_model import DrawParams, EloModel
from goalline.models.simple import Uniform


@pytest.fixture(scope="module")
def preds(features):
    models = [Uniform(), EloModel(DrawParams(0.28, 400)), Benchmark("shin"), Benchmark("max_odds")]
    return predict_all(features, models, (2004, 2005), mode="final", settings=SMOKE)


def test_scores_have_metrics(preds, features):
    scores = score_predictions(preds, features)
    assert {"log_loss", "rps", "brier", "division", "season"} <= set(scores.columns)
    uniform = scores[scores.model == "uniform"]
    assert np.allclose(uniform.log_loss, np.log(3))


def test_summary_groups(preds, features):
    summary = summarise(score_predictions(preds, features))
    assert list(summary.columns) == ["model", "group_type", "group", "n", "log_loss", "rps", "brier"]
    overall = summary[summary.group_type == "overall"].set_index("model")
    assert overall.loc["uniform", "group"] == "all"
    assert set(summary[summary.group_type == "season"].group) == {"2004", "2005"}


def test_comparisons_align_on_common_matches(preds, features):
    trimmed = preds[~((preds.model == "max_odds") & (preds.match_id == preds.match_id.iloc[0]))]
    table = comparisons(score_predictions(trimmed, features), features, replicates=50)
    vs_max = table[(table.model == "elo") & (table.reference == "max_odds")]
    vs_shin = table[(table.model == "elo") & (table.reference == "shin")]
    assert (vs_max.n_matches == vs_shin.n_matches - 1).all()
    assert set(table.metric) == {"log_loss", "rps"}
    assert "proportional" not in set(table.reference)


def test_home_advantage_shares(features):
    ha = home_advantage(features)
    assert np.allclose(ha[["home_share", "draw_share", "away_share"]].sum(axis=1), 1)
    assert set(ha.division) == {"I1", "E0"}


def test_write_outputs(tmp_path, preds, features):
    tables = write_outputs(tmp_path / "final", preds, features, SMOKE)
    for name in ("predictions", "metrics", "reliability", "bootstrap"):
        assert (tmp_path / "final" / f"{name}.csv").exists()
        assert not tables[name].empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/evaluation/test_outputs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goalline.evaluation.outputs'`

- [ ] **Step 3: Write minimal implementation**

`src/goalline/evaluation/outputs.py`:

```python
"""Turn predictions into the CSV files the report reads."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from goalline.constants import MAIN_MODELS, MODEL_NAMES, OUTCOMES, REFERENCES, SEED
from goalline.evaluation.bootstrap import matchday_clusters, paired_bootstrap
from goalline.evaluation.metrics import brier, log_loss, rps
from goalline.evaluation.reliability import reliability_table
from goalline.models.base import outcome_codes
from goalline.settings import Settings

PROBS = ["p_home", "p_draw", "p_away"]


def score_predictions(preds: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    scores = preds.merge(
        frame[["match_id", "division", "season", "date", "result"]],
        on="match_id", how="left", validate="many_to_one",
    )
    p, y = scores[PROBS].to_numpy(), outcome_codes(scores)
    scores["log_loss"], scores["rps"], scores["brier"] = log_loss(p, y), rps(p, y), brier(p, y)
    return scores


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for group_type, key in (("overall", None), ("division", "division"), ("season", "season")):
        keys = ["model"] + ([key] if key else [])
        agg = (
            scores.groupby(keys)
            .agg(n=("log_loss", "size"), log_loss=("log_loss", "mean"),
                 rps=("rps", "mean"), brier=("brier", "mean"))
            .reset_index()
        )
        agg["group_type"] = group_type
        agg["group"] = agg[key].astype(str) if key else "all"
        parts.append(agg[["model", "group_type", "group", "n", "log_loss", "rps", "brier"]])
    return pd.concat(parts, ignore_index=True)


def reliability_all(scores: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for model, rows in scores.groupby("model"):
        table = reliability_table(rows[PROBS].to_numpy(), outcome_codes(rows))
        parts.append(table.assign(model=model))
    out = pd.concat(parts, ignore_index=True)
    return out[["model", "outcome", "bin", "n", "mean_predicted", "observed"]]


def comparisons(
    scores: pd.DataFrame, frame: pd.DataFrame, replicates: int, seed: int = SEED
) -> pd.DataFrame:
    clusters = pd.Series(matchday_clusters(frame), index=frame.match_id.to_numpy())
    present = set(scores.model)
    pairs = [(m, r) for r in REFERENCES for m in MODEL_NAMES]
    pairs += list(combinations(MAIN_MODELS, 2))
    rows = []
    for model, reference in pairs:
        if model not in present or reference not in present:
            continue
        a = scores[scores.model == model].set_index("match_id")
        b = scores[scores.model == reference].set_index("match_id")
        common = a.index.intersection(b.index)
        for metric in ("log_loss", "rps"):
            r = paired_bootstrap(
                a.loc[common, metric].to_numpy(), b.loc[common, metric].to_numpy(),
                clusters.loc[common].to_numpy(), replicates, seed,
            )
            rows.append(
                {"model": model, "reference": reference, "metric": metric, "mean": r.mean,
                 "lower": r.lower, "upper": r.upper, "n_matches": r.n_matches,
                 "n_clusters": r.n_clusters}
            )
    return pd.DataFrame(rows)


def home_advantage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame[frame.tier == 1]
    shares = pd.crosstab([rows.division, rows.season], rows.result, normalize="index")
    shares = shares.reindex(columns=list(OUTCOMES), fill_value=0.0)
    counts = rows.groupby(["division", "season"]).size().rename("n")
    out = shares.join(counts).reset_index()
    out = out.rename(columns={"H": "home_share", "D": "draw_share", "A": "away_share"})
    return out[["division", "season", "n", "home_share", "draw_share", "away_share"]]


def write_outputs(
    directory: Path, preds: pd.DataFrame, frame: pd.DataFrame, settings: Settings
) -> dict[str, pd.DataFrame]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scores = score_predictions(preds, frame)
    tables = {
        "predictions": preds,
        "metrics": summarise(scores),
        "reliability": reliability_all(scores),
        "bootstrap": comparisons(scores, frame, settings.bootstrap_replicates),
    }
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv", index=False)
    return tables
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/evaluation/test_outputs.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/goalline/evaluation/outputs.py tests/evaluation/test_outputs.py
git commit -m "feat(evaluation): scores, summaries, bootstrap comparisons and home advantage"
```

---

### Task 18: Hyperparameter selection, develop and final

**Files:**
- Create: `src/goalline/evaluation/selection.py`, `src/goalline/evaluation/develop.py`
- Test: `tests/evaluation/test_selection.py`, `tests/evaluation/test_develop.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `Selected` (frozen dataclass: `elo: EloParams, draw: DrawParams, dc_xi: float, logistic_c: float, xgb_max_depth: int, xgb_learning_rate: float, xgb_min_child_weight: float, dev_log_loss: dict[str, float], data_sha256: str`), with `to_json() -> dict` and `Selected.from_json(d)`.
  - `write_selected(path, selected)` and `read_selected(path) -> Selected`. `read_selected` raises `FileNotFoundError` with a message naming `goalline develop`.
  - `build_models(selected, settings) -> list` (six models, in `MODEL_NAMES` order) and `build_benchmarks() -> list` (three, in `REFERENCES` order).
  - In `develop.py`:
    - `mean_log_loss(preds, frame) -> float`;
    - `tune_elo(frame, settings) -> pd.DataFrame[k, home_advantage, reversion, draw_peak, draw_scale, log_loss]`;
    - `tune(frame, factory, candidates: list[dict], settings) -> tuple[pd.DataFrame, dict, pd.DataFrame]`;
    - `DevelopResult` (dataclass: `selected, predictions, grids`);
    - `develop(frame, settings, data_sha256) -> DevelopResult`;
    - `final_predictions(frame, selected, settings) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests**

`tests/evaluation/test_selection.py`:

```python
import pytest
from helpers import SMOKE

from goalline.constants import MODEL_NAMES, REFERENCES
from goalline.evaluation.selection import (
    Selected,
    build_benchmarks,
    build_models,
    read_selected,
    write_selected,
)
from goalline.features.elo import EloParams
from goalline.models.elo_model import DrawParams

SELECTED = Selected(
    elo=EloParams(20, 60, 0.33), draw=DrawParams(0.28, 400), dc_xi=0.0019, logistic_c=1.0,
    xgb_max_depth=3, xgb_learning_rate=0.03, xgb_min_child_weight=10,
    dev_log_loss={"elo": 0.98}, data_sha256="abc",
)


def test_round_trip(tmp_path):
    path = tmp_path / "config" / "selected.json"
    write_selected(path, SELECTED)
    assert read_selected(path) == SELECTED


def test_missing_file_explains_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="goalline develop"):
        read_selected(tmp_path / "selected.json")


def test_model_lists():
    assert [m.name for m in build_models(SELECTED, SMOKE)] == list(MODEL_NAMES)
    assert [b.name for b in build_benchmarks()] == list(REFERENCES)
```

`tests/evaluation/test_develop.py`:

```python
import pytest
from helpers import FIXTURE, SMOKE

from goalline.constants import MODEL_NAMES, REFERENCES
from goalline.data.load import load_matches
from goalline.evaluation.develop import develop, final_predictions, tune_elo


@pytest.fixture(scope="module")
def dev_frame():
    frame, _ = load_matches(FIXTURE, SMOKE, mode="develop")
    return frame


def test_elo_grid_covers_every_combination(dev_frame):
    grid = tune_elo(dev_frame, SMOKE)
    assert len(grid) == 1 * 1 * 2 * 2 * 1
    assert list(grid.columns) == ["k", "home_advantage", "reversion", "draw_peak", "draw_scale",
                                  "log_loss"]


@pytest.fixture(scope="module")
def result(dev_frame):
    return develop(dev_frame, SMOKE, data_sha256="sha")


def test_develop_selects_from_the_grids(result):
    s = result.selected
    assert s.elo.reversion in SMOKE.elo_r
    assert s.draw.peak in SMOKE.draw_peak
    assert s.dc_xi in SMOKE.dc_xi
    assert s.logistic_c in SMOKE.logistic_c
    assert s.data_sha256 == "sha"
    assert set(s.dev_log_loss) == {"elo", "dixon_coles", "logistic", "xgboost"}


def test_develop_predictions_cover_dev_seasons_only(result, dev_frame):
    preds = result.predictions.merge(dev_frame[["match_id", "season"]], on="match_id")
    assert set(preds.season) == set(SMOKE.dev_seasons)
    assert set(preds.model) == set(MODEL_NAMES) | set(REFERENCES)
    assert set(result.grids.model) == {"elo", "dixon_coles", "logistic", "xgboost"}


def test_final_predictions_cover_final_seasons(result):
    frame, _ = load_matches(FIXTURE, SMOKE, mode="final")
    preds = final_predictions(frame, result.selected, SMOKE)
    seasons = set(preds.merge(frame[["match_id", "season"]], on="match_id").season)
    assert seasons == set(SMOKE.final_seasons)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/evaluation/test_selection.py tests/evaluation/test_develop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goalline.evaluation.selection'`

- [ ] **Step 3: Write minimal implementation**

`src/goalline/evaluation/selection.py`:

```python
"""The frozen hyperparameters: written by `develop`, read by `final`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from goalline.benchmark import Benchmark
from goalline.features.elo import EloParams
from goalline.models.dixon_coles import DixonColesModel
from goalline.models.elo_model import DrawParams, EloModel
from goalline.models.logistic import LogisticModel
from goalline.models.simple import Frequency, Uniform
from goalline.models.xgb import XGBoostModel
from goalline.settings import Settings


@dataclass(frozen=True)
class Selected:
    elo: EloParams
    draw: DrawParams
    dc_xi: float
    logistic_c: float
    xgb_max_depth: int
    xgb_learning_rate: float
    xgb_min_child_weight: float
    dev_log_loss: dict
    data_sha256: str

    def to_json(self) -> dict:
        return {
            "elo": {"k": self.elo.k, "home_advantage": self.elo.home_advantage,
                    "reversion": self.elo.reversion, "draw_peak": self.draw.peak,
                    "draw_scale": self.draw.scale},
            "dixon_coles": {"xi": self.dc_xi},
            "logistic": {"c": self.logistic_c},
            "xgboost": {"max_depth": self.xgb_max_depth, "learning_rate": self.xgb_learning_rate,
                        "min_child_weight": self.xgb_min_child_weight},
            "dev_log_loss": self.dev_log_loss,
            "data_sha256": self.data_sha256,
        }

    @classmethod
    def from_json(cls, d: dict) -> Selected:
        e = d["elo"]
        return cls(
            elo=EloParams(e["k"], e["home_advantage"], e["reversion"]),
            draw=DrawParams(e["draw_peak"], e["draw_scale"]),
            dc_xi=d["dixon_coles"]["xi"],
            logistic_c=d["logistic"]["c"],
            xgb_max_depth=d["xgboost"]["max_depth"],
            xgb_learning_rate=d["xgboost"]["learning_rate"],
            xgb_min_child_weight=d["xgboost"]["min_child_weight"],
            dev_log_loss=d["dev_log_loss"],
            data_sha256=d["data_sha256"],
        )


def write_selected(path: Path, selected: Selected) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selected.to_json(), indent=2) + "\n")


def read_selected(path: Path) -> Selected:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found: run `goalline develop`, commit the file and tag it before "
            "`goalline final`."
        )
    return Selected.from_json(json.loads(path.read_text()))


def build_models(selected: Selected, settings: Settings) -> list:
    first = settings.train_first_season
    return [
        Uniform(),
        Frequency(first),
        EloModel(selected.draw),
        DixonColesModel(selected.dc_xi),
        LogisticModel(selected.logistic_c, first),
        XGBoostModel(selected.xgb_max_depth, selected.xgb_learning_rate,
                     selected.xgb_min_child_weight, first),
    ]


def build_benchmarks() -> list:
    return [Benchmark("shin"), Benchmark("proportional"), Benchmark("max_odds")]
```

`src/goalline/evaluation/develop.py`:

```python
"""Development: tune on walk-forward folds up to 2020-21. Final: score the frozen models once."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from goalline.evaluation.metrics import log_loss
from goalline.evaluation.selection import Selected, build_benchmarks, build_models
from goalline.evaluation.walkforward import FinalSeasonError, evaluation_rows, predict_all, walk_forward
from goalline.features.elo import EloParams, compute_elo
from goalline.features.table import build_features
from goalline.models.base import outcome_codes
from goalline.models.dixon_coles import DixonColesModel
from goalline.models.elo_model import DrawParams, EloModel, elo_probabilities
from goalline.models.logistic import LogisticModel
from goalline.models.simple import Frequency, Uniform
from goalline.models.xgb import XGBoostModel
from goalline.settings import Settings


def mean_log_loss(preds: pd.DataFrame, frame: pd.DataFrame) -> float:
    merged = preds.merge(frame[["match_id", "result"]], on="match_id")
    p = merged[["p_home", "p_draw", "p_away"]].to_numpy()
    return float(log_loss(p, outcome_codes(merged)).mean())


def tune_elo(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if frame.season.max() >= settings.final_first_season:
        raise FinalSeasonError("tune_elo must run on a develop-mode frame")
    rows = evaluation_rows(frame, settings.dev_seasons)
    y = outcome_codes(rows)
    grid = []
    for k, h, r in product(settings.elo_k, settings.elo_h, settings.elo_r):
        elo = compute_elo(frame, EloParams(k, h, r)).loc[rows.index]
        diff = (elo.elo_home + h - elo.elo_away).to_numpy()
        for peak, scale in product(settings.draw_peak, settings.draw_scale):
            p = elo_probabilities(diff, DrawParams(peak, scale))
            grid.append({"k": k, "home_advantage": h, "reversion": r, "draw_peak": peak,
                         "draw_scale": scale, "log_loss": float(log_loss(p, y).mean())})
    return pd.DataFrame(grid)


def tune(frame, factory, candidates: list[dict], settings: Settings):
    rows, best = [], None
    for params in candidates:
        preds = walk_forward(frame, factory(**params), settings.dev_seasons, mode="develop",
                             settings=settings)
        score = mean_log_loss(preds, frame)
        rows.append({**params, "log_loss": score})
        if best is None or score < best[1]:
            best = (params, score, preds)
    return pd.DataFrame(rows), best[0], best[2]


@dataclass
class DevelopResult:
    selected: Selected
    predictions: pd.DataFrame
    grids: pd.DataFrame


def develop(frame: pd.DataFrame, settings: Settings, data_sha256: str) -> DevelopResult:
    elo_grid = tune_elo(frame, settings)
    top = elo_grid.loc[elo_grid.log_loss.idxmin()]
    elo = EloParams(float(top.k), float(top.home_advantage), float(top.reversion))
    draw = DrawParams(float(top.draw_peak), float(top.draw_scale))
    features = build_features(frame, elo)
    first = settings.train_first_season

    dc_grid, dc_best, dc_preds = tune(
        features, lambda xi: DixonColesModel(xi), [{"xi": x} for x in settings.dc_xi], settings
    )
    lr_grid, lr_best, lr_preds = tune(
        features, lambda c: LogisticModel(c, first), [{"c": c} for c in settings.logistic_c],
        settings,
    )
    xgb_candidates = [
        {"max_depth": d, "learning_rate": lr, "min_child_weight": w}
        for d, lr, w in product(settings.xgb_max_depth, settings.xgb_learning_rate,
                                settings.xgb_min_child_weight)
    ]
    xgb_grid, xgb_best, xgb_preds = tune(
        features, lambda **p: XGBoostModel(**p, train_first_season=first), xgb_candidates, settings
    )

    others = predict_all(
        features, [Uniform(), Frequency(first), EloModel(draw), *build_benchmarks()],
        settings.dev_seasons, mode="develop", settings=settings,
    )
    predictions = pd.concat([others, dc_preds, lr_preds, xgb_preds], ignore_index=True)
    selected = Selected(
        elo=elo, draw=draw, dc_xi=dc_best["xi"], logistic_c=lr_best["c"],
        xgb_max_depth=int(xgb_best["max_depth"]), xgb_learning_rate=xgb_best["learning_rate"],
        xgb_min_child_weight=xgb_best["min_child_weight"],
        dev_log_loss={
            "elo": float(top.log_loss),
            "dixon_coles": float(dc_grid.log_loss.min()),
            "logistic": float(lr_grid.log_loss.min()),
            "xgboost": float(xgb_grid.log_loss.min()),
        },
        data_sha256=data_sha256,
    )
    grids = pd.concat(
        [elo_grid.assign(model="elo"), dc_grid.assign(model="dixon_coles"),
         lr_grid.assign(model="logistic"), xgb_grid.assign(model="xgboost")],
        ignore_index=True,
    )
    assert np.isclose(mean_log_loss(predictions[predictions.model == "elo"], frame), top.log_loss)
    return DevelopResult(selected=selected, predictions=predictions, grids=grids)


def final_predictions(frame: pd.DataFrame, selected: Selected, settings: Settings) -> pd.DataFrame:
    features = build_features(frame, selected.elo)
    models = build_models(selected, settings) + build_benchmarks()
    return predict_all(features, models, settings.final_seasons, mode="final", settings=settings)
```

The `assert` at the end of `develop` checks that the Elo walk-forward predictions reproduce the Elo grid score. If it fails, the feature table and the grid disagree on `elo_diff`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/evaluation -v`
Expected: all evaluation tests pass (33 passed)

- [ ] **Step 5: Commit**

```bash
git add src/goalline/evaluation/selection.py src/goalline/evaluation/develop.py tests/evaluation
git commit -m "feat(evaluation): tune on development folds, freeze, and score the final test"
```

---

### Task 19: Command line and end-to-end pipeline

**Files:**
- Create: `src/goalline/cli.py`, `src/goalline/__main__.py`
- Modify: `tests/conftest.py` (add `pipeline_run` fixture)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `Paths` (dataclass: `manifest, data, config, output, site: Path`) with `Paths.default()` (relative to the working directory: `data/manifest.json`, `data/raw/Matches.csv`, `config/selected.json`, `output`, `site`) and `Paths.under(root)`.
  - `selected_commit(config: Path) -> str` (`"uncommitted"` outside git or for an untracked file).
  - `run_download(paths)`, `run_validate(paths, settings)`, `run_develop(paths, settings)`, `run_final(paths, settings)`. `run_report` comes in Task 20.
  - `main(argv: list[str] | None = None) -> int`.
  - Files written: `output/data_quality.csv` (validate); `output/dev/{predictions,metrics,reliability,bootstrap,grid_scores}.csv` and `config/selected.json` (develop); `output/final/*.csv`, `output/final/final_meta.json` and `output/home_advantage.csv` (final).
  - Pytest fixture `pipeline_run` returns `Paths` after validate, develop and final on the synthetic fixture with `SMOKE`.

- [ ] **Step 1: Write the failing test**

Add to `tests/conftest.py`:

```python
import json
import shutil

from goalline.cli import Paths, run_develop, run_final, run_validate
from goalline.data.manifest import sha256_of


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
```

`tests/test_cli.py`:

```python
import json

import pandas as pd
import pytest
from conftest import make_paths
from helpers import SMOKE

from goalline.cli import Paths, main, run_final, selected_commit
from goalline.constants import MODEL_NAMES, REFERENCES
from goalline.data.manifest import ChecksumError


def test_default_paths():
    p = Paths.default()
    assert str(p.data).endswith("data/raw/Matches.csv")
    assert str(p.config).endswith("config/selected.json")


def test_pipeline_writes_every_output(pipeline_run):
    out = pipeline_run.output
    assert (out / "data_quality.csv").exists()
    for stage in ("dev", "final"):
        for name in ("predictions", "metrics", "reliability", "bootstrap"):
            assert (out / stage / f"{name}.csv").exists()
    assert (out / "dev" / "grid_scores.csv").exists()
    assert (out / "home_advantage.csv").exists()
    assert pipeline_run.config.exists()


def test_final_scores_every_model_and_reference(pipeline_run):
    metrics = pd.read_csv(pipeline_run.output / "final" / "metrics.csv")
    assert set(metrics.model) == set(MODEL_NAMES) | set(REFERENCES)
    meta = json.loads((pipeline_run.output / "final" / "final_meta.json").read_text())
    assert meta["selected_commit"] == "uncommitted"
    assert meta["seasons"] == list(SMOKE.final_seasons)
    assert meta["n_matches"] > 0 and "max_odds_arbitrage" in meta


def test_elo_beats_uniform_on_data_generated_by_elo(pipeline_run):
    overall = pd.read_csv(pipeline_run.output / "final" / "metrics.csv")
    overall = overall[overall.group_type == "overall"].set_index("model")
    assert overall.loc["elo", "log_loss"] < overall.loc["uniform", "log_loss"]


def test_final_without_selection_fails_cleanly(tmp_path):
    paths = make_paths(tmp_path)
    with pytest.raises(FileNotFoundError, match="goalline develop"):
        run_final(paths, SMOKE)
    assert not paths.output.exists()


def test_final_with_other_data_fails_cleanly(tmp_path, pipeline_run):
    paths = make_paths(tmp_path)
    paths.config.parent.mkdir(parents=True)
    selected = json.loads(pipeline_run.config.read_text())
    selected["data_sha256"] = "0" * 64
    paths.config.write_text(json.dumps(selected))
    with pytest.raises(ChecksumError, match="develop"):
        run_final(paths, SMOKE)
    assert not paths.output.exists()


def test_selected_commit_outside_git(tmp_path):
    f = tmp_path / "selected.json"
    f.write_text("{}")
    assert selected_commit(f) == "uncommitted"


def test_main_rejects_unknown_command():
    with pytest.raises(SystemExit):
        main(["peek"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'goalline.cli'`

- [ ] **Step 3: Write minimal implementation**

`src/goalline/cli.py`:

```python
"""Command line: download, validate, develop, final, report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from goalline.constants import ODDS_COLUMNS
from goalline.data.load import load_matches, prepare, read_raw
from goalline.data.manifest import ChecksumError, download, read_manifest, verify
from goalline.evaluation.develop import develop, final_predictions
from goalline.evaluation.outputs import home_advantage, write_outputs
from goalline.evaluation.selection import read_selected, write_selected
from goalline.evaluation.walkforward import evaluation_rows
from goalline.settings import Settings


@dataclass(frozen=True)
class Paths:
    manifest: Path
    data: Path
    config: Path
    output: Path
    site: Path

    @classmethod
    def under(cls, root: Path) -> Paths:
        root = Path(root)
        return cls(
            manifest=root / "data" / "manifest.json",
            data=root / "data" / "raw" / "Matches.csv",
            config=root / "config" / "selected.json",
            output=root / "output",
            site=root / "site",
        )

    @classmethod
    def default(cls) -> Paths:
        return cls.under(Path.cwd())


def selected_commit(config: Path) -> str:
    try:
        done = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", Path(config).name],
            cwd=Path(config).parent, capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return "uncommitted"
    sha = done.stdout.strip()
    return sha if done.returncode == 0 and sha else "uncommitted"


def _verified(paths: Paths) -> str:
    manifest = read_manifest(paths.manifest)
    verify(paths.data, manifest)
    return manifest.sha256


def run_download(paths: Paths) -> None:
    manifest = read_manifest(paths.manifest)
    if paths.data.exists():
        try:
            verify(paths.data, manifest)
            print(f"{paths.data} already matches the manifest")
            return
        except ChecksumError:
            paths.data.unlink()
    download(manifest, paths.data)
    print(f"downloaded {paths.data}")


def run_validate(paths: Paths, settings: Settings) -> None:
    _verified(paths)
    _, report = prepare(read_raw(paths.data))
    paths.output.mkdir(parents=True, exist_ok=True)
    report.to_csv(paths.output / "data_quality.csv", index=False)
    print(f"{int(report['count'].sum()) if not report.empty else 0} rows dropped or odds blanked")


def run_develop(paths: Paths, settings: Settings) -> None:
    sha = _verified(paths)
    frame, _ = load_matches(paths.data, settings, mode="develop")
    result = develop(frame, settings, data_sha256=sha)
    write_selected(paths.config, result.selected)
    tables = write_outputs(paths.output / "dev", result.predictions, frame, settings)
    result.grids.to_csv(paths.output / "dev" / "grid_scores.csv", index=False)
    print(f"selected hyperparameters written to {paths.config}")
    print(tables["metrics"].query("group_type == 'overall'").to_string(index=False))


def run_final(paths: Paths, settings: Settings) -> None:
    selected = read_selected(paths.config)
    sha = _verified(paths)
    if sha != selected.data_sha256:
        raise ChecksumError(
            f"the data file ({sha}) is not the one used in develop ({selected.data_sha256}); "
            "a new data file needs a new develop run"
        )
    frame, _ = load_matches(paths.data, settings, mode="final")
    preds = final_predictions(frame, selected, settings)
    write_outputs(paths.output / "final", preds, frame, settings)
    home_advantage(frame).to_csv(paths.output / "home_advantage.csv", index=False)
    scored = evaluation_rows(frame, settings.final_seasons)
    max_odds = scored[list(ODDS_COLUMNS["max"])].dropna()
    meta = {
        "selected_commit": selected_commit(paths.config),
        "data_sha256": sha,
        "generated": dt.date.today().isoformat(),
        "seasons": list(settings.final_seasons),
        "n_matches": int(len(scored)),
        "max_odds_arbitrage": int(((1 / max_odds).sum(axis=1) < 1).sum()),
    }
    (paths.output / "final" / "final_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(pd.read_csv(paths.output / "final" / "metrics.csv").query("group_type == 'overall'"))


COMMANDS = {
    "download": lambda paths, settings: run_download(paths),
    "validate": run_validate,
    "develop": run_develop,
    "final": run_final,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="goalline", description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    args = parser.parse_args(argv)
    COMMANDS[args.command](Paths.default(), Settings())
    return 0
```

`src/goalline/__main__.py`:

```python
from goalline.cli import main

raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py -v`
Expected: 8 passed

- [ ] **Step 5: Run the whole suite and lint**

Run: `python -m pytest && ruff check .`
Expected: all tests pass; ruff `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/goalline/cli.py src/goalline/__main__.py tests/conftest.py tests/test_cli.py
git commit -m "feat(cli): download, validate, develop and final commands with safety checks"
```

---

### Task 20: Static report

**Files:**
- Create: `src/goalline/report/__init__.py` (empty), `src/goalline/report/charts.py`, `src/goalline/report/render.py`, `src/goalline/report/templates/index.html.j2`
- Modify: `src/goalline/cli.py` (add `run_report` and the `report` command)
- Test: `tests/report/test_render.py`

**Interfaces:**
- Consumes: the CSV and JSON files from Task 19, `config/selected.json`.
- Produces:
  - `charts.reliability_chart(reliability, models) -> str`, `charts.season_chart(metrics, models) -> str` and `charts.home_advantage_chart(home_adv) -> str` (inline SVG strings);
  - `render.headline(bootstrap) -> str`, `render.build_context(output_dir, selected_path) -> dict` and `render.render(output_dir, site_dir, selected_path) -> Path`;
  - `cli.run_report(paths, settings)`, which writes `site/index.html`.

- [ ] **Step 1: Write the failing test**

`tests/report/test_render.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/report -v`
Expected: FAIL with `ImportError: cannot import name 'run_report' from 'goalline.cli'`

- [ ] **Step 3: Write the charts**

`src/goalline/report/charts.py`:

```python
"""Static SVG charts for the report, drawn with matplotlib."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from goalline.constants import LEAGUE_NAMES, MODEL_LABELS, OUTCOMES  # noqa: E402

plt.rcParams.update(
    {"font.size": 9, "axes.edgecolor": "#888", "axes.labelcolor": "#666", "text.color": "#666",
     "xtick.color": "#666", "ytick.color": "#666", "svg.fonttype": "none",
     "axes.spines.top": False, "axes.spines.right": False}
)


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    text = buf.getvalue()
    return text[text.index("<svg") :]


def reliability_chart(reliability: pd.DataFrame, models) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
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
        ax.plot(t.group.astype(int), t.log_loss, marker="o", ms=3, lw=1.2, label=MODEL_LABELS[model])
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
```

- [ ] **Step 4: Write the renderer and template**

`src/goalline/report/render.py`:

```python
"""Build the static report page from the CSV files in output/."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape

from goalline.constants import LEAGUE_NAMES, MAIN_MODELS, MODEL_LABELS, MODEL_NAMES
from goalline.data.load import season_label
from goalline.evaluation.reliability import expected_calibration_error
from goalline.report import charts


def headline(bootstrap: pd.DataFrame) -> str:
    rows = bootstrap[
        (bootstrap.reference == "shin") & (bootstrap.metric == "log_loss")
        & bootstrap.model.isin(MAIN_MODELS)
    ]
    best = rows.loc[rows["mean"].idxmin()]
    label = MODEL_LABELS[best.model]
    if best.upper < 0:
        return (f"{label} beats the de-margined Bet365 probabilities: log loss lower by "
                f"{-best['mean']:.4f} (95% CI {-best.upper:.4f} to {-best.lower:.4f}).")
    if best.lower > 0:
        return (f"No model beats the de-margined Bet365 probabilities. The closest, {label}, "
                f"has a log loss higher by {best['mean']:.4f} "
                f"(95% CI {best.lower:.4f} to {best.upper:.4f}).")
    return (f"The closest model, {label}, is statistically indistinguishable from the "
            f"de-margined Bet365 probabilities: log-loss difference {best['mean']:+.4f} "
            f"(95% CI {best.lower:+.4f} to {best.upper:+.4f}).")


def _key_rows(metrics: pd.DataFrame, bootstrap: pd.DataFrame, group_type: str, group: str):
    scope = metrics[(metrics.group_type == group_type) & (metrics.group == group)].set_index("model")
    vs = bootstrap[(bootstrap.reference == "shin") & (bootstrap.metric == "log_loss")]
    vs = vs.set_index("model")
    rows = []
    for model in (*MODEL_NAMES, "shin"):
        if model not in scope.index:
            continue
        row = {"label": MODEL_LABELS[model], "n": int(scope.loc[model, "n"]),
               "log_loss": scope.loc[model, "log_loss"], "rps": scope.loc[model, "rps"],
               "brier": scope.loc[model, "brier"], "is_reference": model == "shin"}
        if group_type == "overall" and model in vs.index:
            row |= {"diff": vs.loc[model, "mean"], "lower": vs.loc[model, "lower"],
                    "upper": vs.loc[model, "upper"]}
        rows.append(row)
    return rows


def build_context(output_dir: Path, selected_path: Path) -> dict:
    output_dir = Path(output_dir)
    final = output_dir / "final"
    metrics = pd.read_csv(final / "metrics.csv", dtype={"group": str})
    bootstrap = pd.read_csv(final / "bootstrap.csv")
    reliability = pd.read_csv(final / "reliability.csv")
    meta = json.loads((final / "final_meta.json").read_text())
    quality = pd.read_csv(output_dir / "data_quality.csv")
    home_adv = pd.read_csv(output_dir / "home_advantage.csv")
    selected = json.loads(Path(selected_path).read_text())

    leagues = metrics[metrics.group_type == "division"]
    league_table = {
        "headers": [LEAGUE_NAMES[d] for d in LEAGUE_NAMES if d in set(leagues.group)],
        "rows": [
            {"label": MODEL_LABELS[m],
             "values": [leagues[(leagues.model == m) & (leagues.group == d)].log_loss.iloc[0]
                        for d in LEAGUE_NAMES if d in set(leagues.group)]}
            for m in (*MODEL_NAMES, "shin") if m in set(leagues.model)
        ],
    }
    sensitivity = [
        {"label": MODEL_LABELS[r.model], "reference": MODEL_LABELS[r.reference],
         "diff": r["mean"], "lower": r.lower, "upper": r.upper, "n": int(r.n_matches)}
        for _, r in bootstrap[(bootstrap.metric == "log_loss")
                              & bootstrap.reference.isin(["proportional", "max_odds"])
                              & bootstrap.model.isin(MAIN_MODELS)].iterrows()
    ]
    pairs = [
        {"label": MODEL_LABELS[r.model], "reference": MODEL_LABELS[r.reference],
         "diff": r["mean"], "lower": r.lower, "upper": r.upper}
        for _, r in bootstrap[(bootstrap.metric == "log_loss")
                              & bootstrap.reference.isin(MAIN_MODELS)].iterrows()
    ]
    ece = [
        {"label": MODEL_LABELS[m], **expected_calibration_error(t).round(4).to_dict()}
        for m, t in reliability.groupby("model") if m in (*MAIN_MODELS, "shin")
    ]
    seasons = meta["seasons"]
    return {
        "headline": headline(bootstrap),
        "meta": meta,
        "season_range": f"{season_label(seasons[0])} to {season_label(seasons[-1])}",
        "key_rows": _key_rows(metrics, bootstrap, "overall", "all"),
        "serie_a_rows": _key_rows(metrics, bootstrap, "division", "I1"),
        "league_table": league_table,
        "season_svg": charts.season_chart(metrics, (*MAIN_MODELS, "shin")),
        "reliability_svg": charts.reliability_chart(reliability, ("elo", "dixon_coles", "shin")),
        "ece_rows": ece,
        "sensitivity_rows": sensitivity,
        "pair_rows": pairs,
        "selected": selected,
        "home_adv_svg": charts.home_advantage_chart(home_adv),
        "quality_rows": quality.groupby("check")["count"].sum().reset_index().to_dict("records"),
    }


def render(output_dir: Path, site_dir: Path, selected_path: Path) -> Path:
    env = Environment(
        loader=PackageLoader("goalline", "report/templates"), autoescape=select_autoescape()
    )
    html = env.get_template("index.html.j2").render(**build_context(output_dir, selected_path))
    site_dir = Path(site_dir)
    site_dir.mkdir(parents=True, exist_ok=True)
    path = site_dir / "index.html"
    path.write_text(html)
    return path
```

`src/goalline/report/templates/index.html.j2`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>goal-line-calibration</title>
<style>
:root { --bg:#fbfbf9; --fg:#1d1d1b; --muted:#666; --line:#ddd; --accent:#0b6e4f; --ref:#f1efe8; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#151514; --fg:#ecebe6; --muted:#a3a29c; --line:#333; --accent:#5cc39a; --ref:#22221f; }
}
body { background:var(--bg); color:var(--fg); font:16px/1.55 system-ui, sans-serif; margin:0; }
main { max-width:960px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:1.8rem; margin:0 0 4px; } h2 { margin-top:2.2rem; border-bottom:1px solid var(--line); }
.lede { font-size:1.15rem; } .muted { color:var(--muted); font-size:.9rem; }
.table-wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; font-size:.92rem; }
th, td { padding:6px 8px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
tr.ref td { background:var(--ref); font-weight:600; }
svg { max-width:100%; height:auto; }
code { font-size:.9em; }
</style>
</head>
<body>
<main>
<h1>goal-line-calibration</h1>
<p class="muted">How close do statistical and machine-learning models get to the bookmakers' probabilities for home win, draw and away win? Final test {{ season_range }}, {{ meta.n_matches }} first-division matches, scored once with hyperparameters frozen in commit <code>{{ meta.selected_commit }}</code>.</p>
<p class="lede">{{ headline }}</p>

<h2>Final test</h2>
<p class="muted">Mean per match; lower is better. The difference is model minus Bet365 (Shin) log loss, with a 95% interval from a paired bootstrap over matchdays.</p>
<div class="table-wrap"><table>
<tr><th>Model</th><th>Log loss</th><th>RPS</th><th>Brier</th><th>Δ log loss vs Shin</th><th>95% CI</th></tr>
{% for r in key_rows %}
<tr{% if r.is_reference %} class="ref"{% endif %}><td>{{ r.label }}</td><td>{{ "%.4f"|format(r.log_loss) }}</td><td>{{ "%.4f"|format(r.rps) }}</td><td>{{ "%.4f"|format(r.brier) }}</td>
<td>{% if r.diff is defined %}{{ "%+.4f"|format(r.diff) }}{% else %}—{% endif %}</td>
<td>{% if r.diff is defined %}{{ "%+.4f"|format(r.lower) }} to {{ "%+.4f"|format(r.upper) }}{% else %}—{% endif %}</td></tr>
{% endfor %}
</table></div>

<h2>Serie A</h2>
<div class="table-wrap"><table>
<tr><th>Model</th><th>Matches</th><th>Log loss</th><th>RPS</th><th>Brier</th></tr>
{% for r in serie_a_rows %}
<tr{% if r.is_reference %} class="ref"{% endif %}><td>{{ r.label }}</td><td>{{ r.n }}</td><td>{{ "%.4f"|format(r.log_loss) }}</td><td>{{ "%.4f"|format(r.rps) }}</td><td>{{ "%.4f"|format(r.brier) }}</td></tr>
{% endfor %}
</table></div>

<h2>By league and by season</h2>
<div class="table-wrap"><table>
<tr><th>Log loss</th>{% for h in league_table.headers %}<th>{{ h }}</th>{% endfor %}</tr>
{% for r in league_table.rows %}<tr><td>{{ r.label }}</td>{% for v in r["values"] %}<td>{{ "%.4f"|format(v) }}</td>{% endfor %}</tr>{% endfor %}
</table></div>
{{ season_svg|safe }}

<h2>Calibration</h2>
<p class="muted">Ten equal-count bins per outcome. Points on the dashed line are perfectly calibrated.</p>
{{ reliability_svg|safe }}
<div class="table-wrap"><table>
<tr><th>Expected calibration error</th><th>Home</th><th>Draw</th><th>Away</th></tr>
{% for r in ece_rows %}<tr><td>{{ r.label }}</td><td>{{ r.H }}</td><td>{{ r.D }}</td><td>{{ r.A }}</td></tr>{% endfor %}
</table></div>

<h2>Sensitivity</h2>
<p class="muted">The same comparison against proportionally de-margined Bet365 odds and against the best price across about 17 bookmakers. {{ meta.max_odds_arbitrage }} test matches have best prices whose implied probabilities sum to less than one; they are normalised anyway.</p>
<div class="table-wrap"><table>
<tr><th>Model</th><th>Reference</th><th>Matches</th><th>Δ log loss</th><th>95% CI</th></tr>
{% for r in sensitivity_rows %}<tr><td>{{ r.label }}</td><td>{{ r.reference }}</td><td>{{ r.n }}</td><td>{{ "%+.4f"|format(r.diff) }}</td><td>{{ "%+.4f"|format(r.lower) }} to {{ "%+.4f"|format(r.upper) }}</td></tr>{% endfor %}
</table></div>
<h3>Models against each other</h3>
<div class="table-wrap"><table>
<tr><th>Model</th><th>Against</th><th>Δ log loss</th><th>95% CI</th></tr>
{% for r in pair_rows %}<tr><td>{{ r.label }}</td><td>{{ r.reference }}</td><td>{{ "%+.4f"|format(r.diff) }}</td><td>{{ "%+.4f"|format(r.lower) }} to {{ "%+.4f"|format(r.upper) }}</td></tr>{% endfor %}
</table></div>
<p class="muted">Intervals, not p-values: with this many comparisons, an interval that excludes zero is evidence, not proof.</p>

<h2>Development and frozen hyperparameters</h2>
<p class="muted">Chosen by mean log loss over the walk-forward development seasons 2005-06 to 2020-21, then frozen before the final test.</p>
<div class="table-wrap"><table>
<tr><th>Model</th><th>Hyperparameters</th><th>Development log loss</th></tr>
<tr><td>Elo</td><td>K {{ selected.elo.k }}, H {{ selected.elo.home_advantage }}, reversion {{ selected.elo.reversion }}, draw peak {{ selected.elo.draw_peak }}, scale {{ selected.elo.draw_scale }}</td><td>{{ "%.4f"|format(selected.dev_log_loss.elo) }}</td></tr>
<tr><td>Dixon–Coles</td><td>ξ {{ selected.dixon_coles.xi }} per day</td><td>{{ "%.4f"|format(selected.dev_log_loss.dixon_coles) }}</td></tr>
<tr><td>Logistic regression</td><td>C {{ selected.logistic.c }}</td><td>{{ "%.4f"|format(selected.dev_log_loss.logistic) }}</td></tr>
<tr><td>XGBoost</td><td>depth {{ selected.xgboost.max_depth }}, learning rate {{ selected.xgboost.learning_rate }}, min child weight {{ selected.xgboost.min_child_weight }}</td><td>{{ "%.4f"|format(selected.dev_log_loss.xgboost) }}</td></tr>
</table></div>

<h2>Home advantage and 2020-21</h2>
<p class="muted">Share of home wins per season. In 2020-21 most matches were played without crowds; the models use one home advantage for every league and season and do not adjust for it.</p>
{{ home_adv_svg|safe }}

<h2>Data</h2>
<p>Matches and odds come from <a href="https://github.com/xgabora/Club-Football-Match-Data">xgabora/Club-Football-Match-Data</a> (MIT licence), which collects results and odds from <a href="https://www.football-data.co.uk/">Football-Data.co.uk</a> and Elo snapshots from <a href="http://clubelo.com/">ClubElo</a>. Only results, dates and odds are used; Elo and form are recomputed here. File SHA-256 <code>{{ meta.data_sha256 }}</code>.</p>
<div class="table-wrap"><table>
<tr><th>Quality check</th><th>Rows</th></tr>
{% for r in quality_rows %}<tr><td>{{ r.check }}</td><td>{{ r["count"] }}</td></tr>{% else %}<tr><td>No rows dropped</td><td>0</td></tr>{% endfor %}
</table></div>

<h2>Limitations</h2>
<ul>
<li>The bookmakers know things the models do not: injuries, line-ups, news.</li>
<li>The Bet365 columns are documented only as match odds; they need not be closing odds, which would be a harder benchmark.</li>
<li>Elo and Dixon–Coles see only domestic league matches; cups and European games are missing.</li>
<li>One home advantage for all leagues, and no special treatment of the 2020-21 season.</li>
<li>This is a forecast evaluation, not a betting strategy.</li>
</ul>

<h2>Reproduce</h2>
<pre><code>pip install -e .
goalline download    # pinned file, SHA-256 checked
goalline validate
goalline develop     # slow: tunes every model on 2005-06 to 2020-21
goalline final       # uses config/selected.json
goalline report</code></pre>
<p class="muted">Generated {{ meta.generated }}.</p>
</main>
</body>
</html>
```

Add to `src/goalline/cli.py` (import at the top: `from goalline.report.render import render`):

```python
def run_report(paths: Paths, settings: Settings) -> None:
    path = render(paths.output, paths.site, paths.config)
    print(f"report written to {path}")
```

and register it in `COMMANDS`:

```python
    "report": run_report,
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/report -v && python -m pytest && ruff check .`
Expected: 4 passed; full suite passes; ruff clean

- [ ] **Step 6: Commit**

```bash
git add src/goalline/report src/goalline/cli.py tests/report
git commit -m "feat(report): static HTML report with tables, calibration and attribution"
```

---

### Task 21: Continuous integration and publishing workflows

**Files:**
- Create: `.github/workflows/ci.yml`, `.github/workflows/publish.yml`
- Test: `tests/test_workflows.py`

**Interfaces:**
- Produces: CI on every push and pull request (ruff, pytest); publish on `workflow_dispatch` and `v*` tags (download → validate → final → report → Pages).

- [ ] **Step 1: Write the failing test**

`tests/test_workflows.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return (ROOT / ".github" / "workflows" / name).read_text()


def test_ci_runs_lint_and_tests_on_python_312():
    ci = read("ci.yml")
    for text in ("push", "pull_request", "python-version: \"3.12\"", "ruff check", "pytest"):
        assert text in ci, text


def test_publish_runs_final_not_develop():
    pub = read("publish.yml")
    for text in ("workflow_dispatch", "tags:", "v*", "fetch-depth: 0", "goalline download",
                 "goalline validate", "goalline final", "goalline report", "deploy-pages"):
        assert text in pub, text
    assert "goalline develop" not in pub
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflows.py -v`
Expected: FAIL with `FileNotFoundError` for `ci.yml`

- [ ] **Step 3: Write the workflows**

`.github/workflows/ci.yml`:

```yaml
name: ci

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: pytest
```

`.github/workflows/publish.yml`:

```yaml
name: publish

on:
  workflow_dispatch:
  push:
    tags:
      - "v*"

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -e .
      - run: goalline download
      - run: goalline validate
      - run: goalline final
      - run: goalline report
      - uses: actions/upload-pages-artifact@v3
        with:
          path: site
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_workflows.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add .github tests/test_workflows.py
git commit -m "ci: lint and test on every push, publish the report on tags"
```

---

### Task 22: Real data — validate, develop and freeze

This task runs the pipeline on the real file. There is no new production code. The "test" is that each command finishes, and that the outputs pass the checks listed.

**Files:**
- Create (committed): `config/selected.json`
- Create (ignored): `data/raw/Matches.csv`, `output/`

- [ ] **Step 1: Download and validate**

Run:

```bash
goalline download && goalline validate
python -c "import pandas as pd; print(pd.read_csv('output/data_quality.csv').groupby('check')['count'].sum())"
```

Expected: `downloaded data/raw/Matches.csv`. The quality summary should be roughly what the data profile on 2026-10-07 found: `no_result` 1 (F2); `b365_odds_blanked` small (only partial odds sets count, plus the one F1 overround above 1.25); no duplicates and no result mismatches. About 450 first-division matches have no Bet365 odds at all: they are not counted as blanked and simply stay out of the evaluation set. A count far off these means the file or the checks changed: stop and investigate.

- [ ] **Step 2: Run develop in the background and time it**

Run (background, log to file):

```bash
( time goalline develop ) > output/develop.log 2>&1
```

Expected: finishes, prints the overall development metrics, writes `config/selected.json`. If it runs longer than about an hour, stop it, reduce `xgb_max_depth` to `(2, 3)` in `Settings`, record the change and the reason in spec section 14, commit that change on its own, and rerun.

- [ ] **Step 3: Sanity-check the development results**

Run:

```bash
python - <<'EOF'
import pandas as pd
m = pd.read_csv("output/dev/metrics.csv")
o = m[m.group_type == "overall"].set_index("model").log_loss.sort_values()
print(o)
assert o["uniform"] > o["frequency"] > o["elo"], "baselines should rank below Elo"
assert o["shin"] < o["uniform"]
EOF
cat config/selected.json
```

Expected: uniform ≈ 1.0986, frequency ≈ 1.05–1.07, the four models between frequency and Shin. Shin is expected to be the lowest (around 0.97–1.00). If a model beats Shin by a wide margin during development, suspect leakage: stop and investigate before freezing.

- [ ] **Step 4: Freeze**

```bash
git add config/selected.json
git commit -m "chore: freeze hyperparameters selected on 2005-06 to 2020-21"
git tag -a frozen-v1 -m "Hyperparameters frozen before the final test"
```

---

### Task 23: Final test, report, README and publication

**Files:**
- Modify: `README.md` (final version)
- Create (ignored): `output/final/`, `site/index.html`

- [ ] **Step 1: Run the final test once**

```bash
goalline final && goalline report
python -c "import json; print(json.load(open('output/final/final_meta.json')))"
```

Expected: `selected_commit` equals the SHA of the `frozen-v1` commit (`git rev-list -n1 frozen-v1`). `seasons` is `[2021, 2022, 2023, 2024, 2025]`. `site/index.html` opens in a browser and shows every section.

- [ ] **Step 2: Print the key table as Markdown**

```bash
python - <<'EOF'
import pandas as pd
from goalline.constants import MODEL_LABELS, MODEL_NAMES
m = pd.read_csv("output/final/metrics.csv", dtype={"group": str})
b = pd.read_csv("output/final/bootstrap.csv")
o = m[m.group_type == "overall"].set_index("model")
v = b[(b.reference == "shin") & (b.metric == "log_loss")].set_index("model")
print("| Model | Log loss | RPS | Δ log loss vs Shin (95% CI) |")
print("|---|---|---|---|")
for name in (*MODEL_NAMES, "shin"):
    ci = (f"{v.loc[name, 'mean']:+.4f} ({v.loc[name, 'lower']:+.4f} to {v.loc[name, 'upper']:+.4f})"
          if name in v.index else "reference")
    print(f"| {MODEL_LABELS[name]} | {o.loc[name, 'log_loss']:.4f} | {o.loc[name, 'rps']:.4f} | {ci} |")
EOF
```

- [ ] **Step 3: Write the final README**

Replace `README.md` with this text. Paste the table printed in Step 2 where marked, and the headline sentence from `site/index.html` (the `.lede` paragraph) as the one-line answer:

````markdown
# goal-line-calibration

![CI](https://github.com/simones99/goal-line-calibration/actions/workflows/ci.yml/badge.svg)

How close do statistical and machine-learning models get to the bookmakers'
probabilities for home win, draw and away win in the five big European leagues?

**Answer (final test 2021-22 to 2025-26, scored once):** <headline sentence from the report>

<table from Step 2>

**Report:** https://simones99.github.io/goal-line-calibration/

## Question and method

- **Data:** about 45,000 first-division matches with Bet365 odds (Serie A, Premier League,
  La Liga, Bundesliga, Ligue 1), 2000-01 to 2025-26, plus the second divisions to rate
  promoted teams. Source: [xgabora/Club-Football-Match-Data](https://github.com/xgabora/Club-Football-Match-Data)
  (MIT), collecting [Football-Data.co.uk](https://www.football-data.co.uk/) and
  [ClubElo](http://clubelo.com/); pinned by commit and SHA-256, never stored in git.
- **Models:** Elo with a draw curve, Dixon–Coles, multinomial logistic regression,
  XGBoost, plus uniform and frequency baselines.
- **Benchmark:** Bet365 odds de-margined with Shin's method; proportional normalisation
  and the best price across about 17 bookmakers as sensitivity checks.
- **Validation:** walk-forward by season. Hyperparameters chosen on 2005-06 to 2020-21,
  frozen in tag `frozen-v1`, then the five test seasons are scored once.
- **Metrics:** log loss (primary), ranked probability score, Brier score, reliability
  tables, and a paired bootstrap over matchdays for every comparison.

This is a forecast evaluation, not a betting strategy: there is no betting simulation.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                 # tests run on a synthetic league, no download needed
goalline download      # pinned data file, SHA-256 checked
goalline validate
goalline develop       # slow: tunes every model on the development seasons
goalline final         # uses the committed config/selected.json
goalline report        # writes site/index.html
```

## Structure

`src/goalline/`: `data` (download, quality checks), `features` (Elo, form), `models`,
`benchmark`, `evaluation` (walk-forward, metrics, bootstrap), `report`, `cli`.
Design: [docs/superpowers/specs](docs/superpowers/specs/2026-10-07-goal-line-calibration-design.md).

## Limitations

The bookmakers know about injuries, line-ups and news; the models do not. The Bet365
columns need not be closing odds. Only domestic league matches are used. One home
advantage for every league, with no adjustment for the 2020-21 season played without
crowds.

Started in 2025 as GoalGenius, rewritten in 2026.
````

- [ ] **Step 4: Commit and push**

```bash
git add README.md
git commit -m "docs: README with the final-test results and how to reproduce them"
git push origin main --follow-tags
```

- [ ] **Step 5: Publish (the user runs these)**

The first two commands change the public repository's settings, so the user runs them:

```
! gh api -X POST repos/simones99/goal-line-calibration/pages -f build_type=workflow
! gh workflow run publish.yml -R simones99/goal-line-calibration
```

Then check:

```bash
gh run list -R simones99/goal-line-calibration --limit 2
curl -s -o /dev/null -w "%{http_code}\n" https://simones99.github.io/goal-line-calibration/
```

Expected: `ci` and `publish` succeed, the page returns 200 and shows the same numbers as the README.

- [ ] **Step 6: Tag the release**

```bash
git tag -a v1.0.0 -m "First published version"
git push origin v1.0.0
```

Expected: the tag push triggers `publish` again and produces an identical page, because the final test is deterministic.
