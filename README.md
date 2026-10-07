# goal-line-calibration

![CI](https://github.com/simones99/goal-line-calibration/actions/workflows/ci.yml/badge.svg)

How close do statistical and machine-learning models get to the bookmakers'
probabilities for home win, draw and away win in the five big European leagues?

**Answer (final test 2021-22 to 2025-26, scored once):** No model beats the de-margined Bet365 probabilities. The closest, Logistic regression, has a log loss higher by 0.0179 (95% CI 0.0145 to 0.0213).

| Model | Log loss | RPS | Δ log loss vs Shin (95% CI) |
|---|---|---|---|
| Uniform | 1.0986 | 0.2354 | +0.1283 (+0.1190 to +0.1378) |
| Frequency | 1.0752 | 0.2305 | +0.1049 (+0.0968 to +0.1128) |
| Elo | 0.9939 | 0.2025 | +0.0235 (+0.0197 to +0.0272) |
| Dixon–Coles | 0.9886 | 0.2009 | +0.0182 (+0.0147 to +0.0215) |
| Logistic regression | 0.9882 | 0.2006 | +0.0179 (+0.0145 to +0.0213) |
| XGBoost | 0.9901 | 0.2011 | +0.0197 (+0.0162 to +0.0235) |
| Bet365, Shin | 0.9703 | 0.1952 | reference |

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
  and the best price reported in the data as sensitivity checks.
- **Validation:** walk-forward by season. Hyperparameters chosen on 2005-06 to 2020-21,
  frozen in tag `frozen-v1`, then the five test seasons are scored once.
- **Metrics:** log loss (primary), ranked probability score, Brier score, reliability
  tables, and a paired bootstrap over matchdays for every comparison.

This is a forecast evaluation, not a betting strategy: there is no betting simulation.

## Reproduce

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt && pip install -e . --no-deps
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
