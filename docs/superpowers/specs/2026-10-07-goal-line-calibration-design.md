# goal-line-calibration — design

Date: 2026-10-07. Status: approved in conversation, section by section; this
document is the written spec for review.

## 1. Purpose

The repository (formerly GoalGenius) is rewritten as a probabilistic
forecasting study with one question:

> How close do statistical and machine-learning models get to the
> bookmakers' probabilities for home win / draw / away win in the five big
> European leagues, and where do they fall short?

Audience: recruiters and reviewers for data roles (EU institutions in
particular). The project must show method: leakage-safe features,
time-ordered validation, a test set scored once, calibration, uncertainty on
the comparisons, and honest reporting. It is a forecast-evaluation project,
not a betting project: there is no betting simulation and no ROI.

Expected result, stated up front in the README: the market has information the
models do not (injuries, line-ups, news), so the models will probably not beat
it. The interesting output is how large the gap is and where it lies: by
league, by season, by probability range.

## 2. Decisions

| Topic | Decision |
|---|---|
| Repository | Renamed `simones99/goal-line-calibration`; history kept |
| Code | Clean rewrite as the package `src/goalline/`; old code deleted in a dedicated commit |
| Scope | Five first divisions (I1, E0, SP1, D1, F1), Serie A highlighted; second divisions (I2, E1, SP2, D2, F2) only feed the ratings |
| Models | Uniform, frequency, Elo with draw model, Dixon–Coles, multinomial logistic regression, XGBoost; stacking removed |
| Features | Elo and form recomputed from results; the dataset's Elo and form columns are not used |
| Benchmark | Bet365 odds de-margined with Shin's method; proportional normalisation and maximum odds as sensitivity checks |
| Validation | Walk-forward development up to 2020-21; final test 2021-22 to 2025-26, scored once with frozen hyperparameters |
| Metrics | Log loss (primary), RPS, Brier, reliability tables, paired block bootstrap |
| Output | Static HTML report on GitHub Pages, built by a workflow run by hand or on `v*` tags |
| Method | Test-driven development: every production unit starts from a failing test |

## 3. Data — `goalline.data`

**Source.** `data/Matches.csv` from
[xgabora/Club-Football-Match-Data](https://github.com/xgabora/Club-Football-Match-Data)
(MIT licence), pinned at commit `25882a58a736daf7ece3781940eac17ae1117a66`
(2026-09-06). Upstream sources: match results and odds from
[Football-Data.co.uk](https://www.football-data.co.uk/), Elo snapshots from
[ClubElo](http://clubelo.com/). Neither upstream site states a licence or a
redistribution restriction. The project therefore stores no data in git and
downloads the file at run time.

**Manifest.** `data/manifest.json` holds the URL, the commit, the SHA-256 of
the file and the retrieval date. `goalline download` writes
`data/raw/Matches.csv`, which git ignores. Every later step verifies the
SHA-256 and stops if it differs.

**Columns used.** `Division`, `MatchDate`, `HomeTeam`, `AwayTeam`, `FTHome`,
`FTAway`, `FTResult`, `OddHome`, `OddDraw`, `OddAway`, `MaxHome`, `MaxDraw`,
`MaxAway`. All other columns are ignored, including `HomeElo`, `AwayElo` and
`Form*`. After 2025-06-01 those columns are a provisional continuation by the
dataset author, and before that the Elo values are snapshots up to two weeks
old.

**Season.** A match dated in July or later belongs to season `YYYY-(YY+1)`,
otherwise to `(YYYY-1)-YY`.

**Match id.** A stable hash of (division, date, home team, away team).

**Quality checks** (`goalline validate`; each check has a unit test):

- no duplicate (division, date, home, away);
- no team plays twice on the same date;
- goals are non-negative integers and `FTResult` agrees with the goals;
- odds, where present, are greater than 1;
- the Bet365 overround (sum of 1/odds) lies in [1.00, 1.25].

Rows that fail one of the first three checks are dropped. When only the odds
checks fail, the odds of that set (Bet365 or maximum) are set to missing: the
match still updates the ratings but leaves the evaluation set. Every drop and
every blanked set of odds is counted per check and per division. The counts go
to `output/data_quality.csv` and to the report.

**Evaluation set.** First-division matches that have all three Bet365 odds.
Matches without odds still update the ratings.

**Test-season guard.** Data loading takes a mode. In `develop` mode, a request
for any season from 2021-22 onwards raises an error.

## 4. Features — `goalline.features`

All features are pure functions of matches sorted by date, and each match gets
the values from before it was played.

**Elo.**

- One rating pool for all ten divisions. Teams from different countries never
  meet in this data, so ratings are compared only within a country.
- A team's first rating is 1500 if it first appears in a first division and
  1350 if it first appears in a second division. Both values are fixed, not
  tuned; the burn-in seasons absorb the choice.
- Expected home score: `E = 1 / (1 + 10^((R_away − (R_home + H)) / 400))`.
- Zero-sum update with constant `K`: `R_home += K·(S − E)`, `R_away −= K·(S − E)`,
  where S is 1 for a home win, 0.5 for a draw and 0 for an away win.
- Mean reversion: at a team's first match of a new season,
  `R ← R + r·(m − R)`, where `m` is the mean end-of-season rating of the
  division the team played in during the previous season.
- One home advantage `H` for every league. The 2020-21 season, played mostly
  without crowds, is described in the report, not modelled.
- `K`, `H` and `r` are tuned in development (section 6).

**Form.** Mean points per match (W = 3, D = 1, L = 0) and mean goal
difference per match over the team's previous 5 matches in any division, so a
promoted team carries its second-division form. With fewer than 5 previous
matches the mean of the available ones is used. With none, points take the
mean points per match of the team's division over all matches before that
date, and goal difference takes 0. A `short_history` flag (fewer than 5
previous matches) is added for each side.

**Feature table** (input to the logistic regression and XGBoost):
`elo_diff = R_home + H − R_away`, `form_points_diff`, `form_gd_diff`,
`short_history_home`, `short_history_away`, and league one-hot columns.

**Leakage tests.**

1. *Perturbation*: changing the result of match *m* leaves the features of *m*
   and of every earlier match unchanged.
2. *Truncation*: features of *m* computed on all data equal those computed on
   data up to the day before *m*.
3. *Guard*: `develop` mode cannot load a test season.

## 5. Models and benchmark — `goalline.models`, `goalline.benchmark`

**Common interface.** `fit(history) -> self`, where `history` holds only
earlier matches, and `predict_proba(matches) -> ndarray (n, 3)` in the order
home, draw, away. A single contract test runs on every model and benchmark:

- rows sum to 1 (tolerance 1e-9);
- every probability is strictly positive;
- predictions do not depend on row order.

**Models.**

1. **Uniform**: 1/3 each.
2. **Frequency**: home/draw/away shares in the training data, per league.
3. **Elo with draw model**: `d = R_home + H − R_away`,
   `P(draw) = peak · exp(−0.5 · (d / scale)²)`,
   `P(home) = E · (1 − P(draw))`, `P(away) = (1 − E) · (1 − P(draw))`.
   The draw constants only change the probabilities, not the ratings, so they
   are tuned on stored ratings without recomputing them.
4. **Dixon–Coles**:
   - home and away goals are Poisson with team attack and defence strengths,
     a home effect and the ρ correction for 0-0, 1-0, 0-1 and 1-1;
   - fitted by weighted maximum likelihood with weights `exp(−ξ · days)` on
     the last three seasons;
   - one fit per country on its first and second divisions together, so
     promotions link the two divisions and a promoted team already has
     parameters. Identifiability: the attack strengths sum to zero;
   - refitted on the first day of every month, using only matches before that
     day;
   - outcome probabilities come from the score matrix up to 10 goals per side.
5. **Multinomial logistic regression**: the feature table, standardised, with
   L2 penalty `C`. Refitted once per season.
6. **XGBoost**: the feature table, `objective = multi:softprob`,
   `eval_metric = mlogloss` (the old `auc_mu` setting was the bug). Early
   stopping (50 rounds) uses the last training season of each fold as the
   inner validation set, never the season being evaluated. Refitted once per
   season.

**Benchmark** (no fitting; it reads the odds columns).

- **Shin** on Bet365. With `π_i = 1/odds_i` and `β = Σπ_i`,
  `p_i(z) = (sqrt(z² + 4(1 − z)·π_i²/β) − z) / (2(1 − z))`. `z ∈ [0, 1)` is
  found by bisection so that `Σp_i(z) = 1`.
- **Proportional** on Bet365: `p_i = π_i / β`.
- **Maximum odds**: proportional normalisation of `MaxHome/MaxDraw/MaxAway`.
  Where `Σ1/odds < 1` (an arbitrage across bookmakers) it still normalises,
  and these cases are counted in the report.

## 6. Evaluation — `goalline.evaluation`

**Seasons.**

| Period | Seasons | Use |
|---|---|---|
| Burn-in | 2000-01 to 2004-05 | Ratings and Dixon–Coles settle; nothing is scored |
| Development | 2005-06 to 2020-21 (16 folds) | Walk-forward scoring and hyperparameter selection |
| Final test | 2021-22 to 2025-26 (5 folds) | Walk-forward, hyperparameters frozen, scored once |

2026-27 is out of scope. Fold *s* trains on every season before *s*, and in
the final test that includes earlier test seasons. Rows from 2000-01 and
2001-02 are excluded from the training data of the learned models (frequency,
logistic, XGBoost), because their features are still warming up.

**Hyperparameter selection** (`goalline develop`): mean log loss over the 16
development folds, five leagues pooled. Grids:

| Model | Grid |
|---|---|
| Elo, joint grid | K ∈ {10, 15, 20, 25, 30, 40}; H ∈ {40, 60, 80, 100}; r ∈ {0, 0.1, 0.2, 0.33, 0.5}; peak ∈ {0.22, 0.24, …, 0.32}; scale ∈ {200, 300, 400, 500, 600}. Ratings are computed once per (K, H, r), i.e. 120 runs, and all 30 draw settings are scored on each of them |
| Dixon–Coles | ξ per day ∈ {0.0005, 0.001, 0.0019, 0.003} |
| Logistic | C ∈ {0.01, 0.1, 1, 10} |
| XGBoost | max_depth ∈ {2, 3, 4}; learning_rate ∈ {0.03, 0.1}; min_child_weight ∈ {1, 10}; subsample = 0.8; up to 1000 trees with early stopping |

Order: the Elo parameters are chosen first and frozen. The feature table is
then built with them, and the logistic regression and XGBoost are tuned on it.

**Freezing.**

- `develop` writes `config/selected.json` with the chosen values and their
  development scores.
- The file is committed and tagged `frozen-v1`.
- `goalline final` refuses to run without `config/selected.json`, and records
  in its output the commit SHA in which that file was last changed.
- Seeds are fixed (42), so re-running `final` reproduces the same numbers.

**Metrics** (per match, then averaged):

- log loss (natural log, with probabilities clipped at 1e-15 only inside the
  metric);
- RPS over the ordered outcomes home > draw > away;
- Brier score summed over the three outcomes.

Averages are reported overall, per league and per season, with Serie A shown
separately.

**Reliability.** For each outcome, 10 equal-count bins of predicted
probability, comparing the mean predicted value with the observed frequency.
The summary is the expected calibration error, the bin-size-weighted mean
absolute gap.

**Paired block bootstrap.**

- Per-match loss differences, model minus reference.
- Resampling unit: the matchday cluster (league × ISO year-week); 2,000
  replicates, seed 42.
- Percentile 95% confidence intervals for the mean difference in log loss and
  in RPS.
- Comparisons: every model against Shin, and the six pairs among Elo,
  Dixon–Coles, logistic regression and XGBoost.
- The main table is repeated with the proportional and maximum-odds
  references.
- The report shows intervals, not p-values, and says so.

**Descriptive.** Home-win share per season and league, with 2020-21
highlighted.

**Outputs** (`output/dev/` and `output/final/`, all ignored by git):
`predictions.csv` (match_id, model, p_home, p_draw, p_away), `metrics.csv`,
`reliability.csv`, `bootstrap.csv`, plus `output/data_quality.csv` and
`output/home_advantage.csv`.

## 7. Report — `goalline.report`

`goalline report` renders `site/index.html` from the CSVs in `output/` with a
Jinja2 template and inline matplotlib SVG charts. It does no modelling. The
page is self-contained and in English. Sections:

1. Question and answer in two lines, with the key final-test table: models
   against Shin, log loss, RPS, Brier and the CI of the difference.
2. Serie A panel.
3. Results by league and by season.
4. Reliability charts.
5. Sensitivity: proportional and maximum-odds references.
6. Development summary and frozen hyperparameters, with the commit SHA.
7. Home advantage and 2020-21.
8. Data, licences and attribution (xgabora under MIT, Football-Data.co.uk,
   ClubElo) and the dropped-row counts.
9. Limitations.
10. How to reproduce.

## 8. Command line

`python -m goalline <command>`, also installed as the `goalline` entry point:

| Command | Does |
|---|---|
| `download` | Fetch the pinned file and verify the SHA-256 |
| `validate` | Run the quality checks and write `output/data_quality.csv` |
| `develop` | Walk-forward development, write `config/selected.json` |
| `final` | Final test with frozen hyperparameters |
| `report` | Build `site/index.html` |

## 9. CI and publishing

- **`.github/workflows/ci.yml`**, on every push and pull request: ruff, pytest,
  and an end-to-end smoke test on a small synthetic dataset in
  `tests/fixtures/`. It runs from loading to the report, with no download.
- **`.github/workflows/publish.yml`**, run by hand or on `v*` tags: `download`,
  `validate`, `final` with the committed `config/selected.json`, `report`,
  then deploy to GitHub Pages. `develop` is heavy, so it runs locally. Its
  command is documented in the README.

## 10. README

In English and short:

- one sentence and the CI badge;
- the key final-test table and the link to the report, at the top;
- the question, the data and licences, the method in five points, how to
  reproduce, the structure and the limitations;
- a closing line: "Started in 2025 as GoalGenius, rewritten in 2026."

## 11. Testing strategy

| Level | Content |
|---|---|
| Unit, hand-computed values | Elo update over 2–3 matches; Shin on a worked example; Dixon–Coles log-likelihood on 2 matches; metrics: a perfect forecast gives log loss 0, the uniform forecast gives log loss ln 3 ≈ 1.0986, RPS matches the worked example in Constantinou & Fenton (2012) |
| Properties and parameter recovery | Elo and Dixon–Coles recover the parameters used to simulate synthetic data; Shin sums to 1 for any odds; zero-margin odds give z = 0 and p = 1/odds |
| Contract | One test applied to every model and benchmark (section 5) |
| Leakage | Perturbation, truncation and guard (section 4) |
| Bootstrap | Identical models give a difference of 0 and an interval containing 0; whole matchday clusters are resampled |
| Integration | The full pipeline on the synthetic fixture, generated from a known process (Elo-driven), in which the Elo model beats the uniform one |
| Real data (locally and in `publish`, not in CI) | `validate` on the real file |

## 12. Repository clean-up

Each step is a dedicated commit, with no Co-Authored-By line:

- delete `models/`, `src/features/`, `src/ingest/`, `src/analysis/`,
  `notebooks/`, `setup.py` and `goalgenius.egg-info/`;
- stop tracking `data/processed/matches_processed.csv`, `data/results/` and
  every `.DS_Store`;
- `.gitignore` covers `data/raw/`, `output/`, `site/`, `venv/`, `.venv/` and
  `.DS_Store`;
- `pyproject.toml` with Python 3.12 and these dependencies: pandas, numpy,
  scipy, scikit-learn, xgboost, jinja2, matplotlib; dev dependencies: pytest,
  ruff;
- the MIT `LICENSE` stays.

## 13. Out of scope

- Betting simulation or ROI.
- The 2026-27 season and live forecasts.
- A comparison between our Elo and ClubElo's.
- Bayesian models, stacking, LightGBM.
- Any change to the juventum code. A one-line link in juventum's README is
  optional.

## 14. Risks

- **Run time**: about 1,000 Dixon–Coles fits and the XGBoost grid. If
  `develop` takes longer than about an hour locally, the XGBoost grid is
  reduced first; the change is recorded in this document.
- **Data drift upstream**: the commit pin and the SHA-256 make the input
  immutable. Moving to a newer commit means a new manifest and a new frozen
  run.
- **Odds timing**: the dataset documents the Bet365 columns only as Bet365
  match odds. They are treated as pre-match odds that need not be closing
  odds. The report states this, because closing odds would be a harder
  benchmark.
