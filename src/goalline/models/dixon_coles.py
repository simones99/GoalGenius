"""Dixon–Coles: Poisson goals with team attack/defence, home effect and low-score correction ρ.

log λ = home + attack[home team] + defence[away team];
log μ = attack[away team] + defence[home team].
One fit per country on its first and second divisions, weighted by exp(−ξ · days before the
reference date), over the last WINDOW_DAYS (about three seasons).
"""

from __future__ import annotations

import warnings
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


class ConvergenceWarning(RuntimeWarning):
    """The Dixon–Coles optimiser stopped without meeting its convergence criteria."""


@dataclass
class DixonColesFit:
    teams: dict[str, int]
    attack: np.ndarray
    defence: np.ndarray
    home: float
    rho: float
    converged: bool = True

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
    if not res.success:
        warnings.warn(
            f"Dixon–Coles fit did not converge ({res.message}) for {n} teams "
            f"as of {reference.date()}",
            ConvergenceWarning,
            stacklevel=2,
        )
    attack, defence = res.x[:n].copy(), res.x[n : 2 * n].copy()
    shift = attack.mean()
    attack -= shift
    defence += shift
    return DixonColesFit(
        teams=dict(zip(teams, range(n), strict=True)), attack=attack, defence=defence,
        home=float(res.x[2 * n]), rho=float(res.x[2 * n + 1]),
        converged=bool(res.success),
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
        self.fits_total = 0
        self.fits_nonconverged = 0

    def fit(self, history: pd.DataFrame) -> DixonColesModel:
        self.fits_: dict[str, DixonColesFit] = {}
        if history.empty:
            return self
        reference = history.date.max() + pd.Timedelta(1, "D")
        start = reference - pd.Timedelta(WINDOW_DAYS, "D")
        for country, rows in history[history.date >= start].groupby("country"):
            if len(rows) >= MIN_MATCHES:
                fit = fit_dixon_coles(rows, reference, self.xi)
                self.fits_[country] = fit
                self.fits_total += 1
                self.fits_nonconverged += not fit.converged
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
