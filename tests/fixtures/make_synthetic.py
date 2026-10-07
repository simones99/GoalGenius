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
