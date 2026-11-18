"""GARCH(1,1) and DCC (Engle 2002) with pairwise composite likelihood
(Engle, Ledoit & Wolf 2019 style) -- paper Section 3.2, eq. (1)-(4)."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------- GARCH(1,1)
def _garch_filter(eps: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    h = np.empty_like(eps)
    h[0] = eps.var()
    for t in range(1, len(eps)):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
    return h


def fit_garch11(r: np.ndarray) -> dict:
    """Gaussian QML estimate of h_t = w + a e_{t-1}^2 + b h_{t-1} (eq. 2), a+b<1."""
    x = np.asarray(r, float)
    mu = x.mean()
    eps = x - mu
    var = eps.var()

    def nll(p):
        w, a, b = p
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.9999:
            return 1e12
        h = _garch_filter(eps, w, a, b)
        return 0.5 * np.sum(np.log(h) + eps ** 2 / h)

    best = None
    for a0, b0 in [(0.05, 0.90), (0.10, 0.85), (0.03, 0.95)]:
        res = minimize(nll, [var * (1 - a0 - b0), a0, b0], method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res
    w, a, b = best.x
    h = _garch_filter(eps, w, a, b)
    return {"mu": mu, "omega": w, "alpha": a, "beta": b, "h": h, "z": eps / np.sqrt(h)}


def standardize(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a GARCH(1,1) per series; return standardized residuals and parameters."""
    r = returns.dropna()
    Z, params = {}, {}
    for c in r.columns:
        g = fit_garch11(r[c].values)
        Z[c] = g["z"]
        params[c] = {k: g[k] for k in ("mu", "omega", "alpha", "beta")}
    return pd.DataFrame(Z, index=r.index), pd.DataFrame(params).T


# ---------------------------------------------------------------- DCC
def _pair_paths(z1, z2, a, b, qbar):
    """Vectorised bivariate DCC recursion over P pairs at once.
    z1, z2: (T, P). qbar: (P,) unconditional correlation of each pair."""
    T, P = z1.shape
    q11 = np.ones(P); q22 = np.ones(P); q12 = qbar.copy()
    rho = np.empty((T, P))
    c = 1.0 - a - b
    for t in range(T):
        rho[t] = q12 / np.sqrt(q11 * q22)
        q11 = c * 1.0 + a * z1[t] ** 2 + b * q11
        q22 = c * 1.0 + a * z2[t] ** 2 + b * q22
        q12 = c * qbar + a * z1[t] * z2[t] + b * q12
    return rho


def _composite_nll(params, z1, z2, qbar):
    a, b = params
    if a < 0 or b < 0 or a + b >= 0.999:
        return 1e12
    rho = np.clip(_pair_paths(z1, z2, a, b, qbar), -0.9999, 0.9999)
    one_m = 1.0 - rho ** 2
    ll = np.log(one_m) + (z1 ** 2 + z2 ** 2 - 2 * rho * z1 * z2) / one_m - (z1 ** 2 + z2 ** 2)
    return 0.5 * np.sum(ll)


def fit_dcc(Z: pd.DataFrame, pairs: list[tuple[str, str]] | None = None) -> dict:
    """Estimate scalar DCC(1,1) parameters (alpha, beta) by composite likelihood
    over all pairs (or the supplied pairs) and return conditional correlations."""
    cols = list(Z.columns)
    pairs = pairs or list(itertools.combinations(cols, 2))
    z1 = Z[[p[0] for p in pairs]].values
    z2 = Z[[p[1] for p in pairs]].values
    qbar = np.array([np.mean(z1[:, i] * z2[:, i]) for i in range(len(pairs))])
    res = minimize(_composite_nll, [0.02, 0.95], args=(z1, z2, qbar), method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-7})
    a, b = res.x
    rho = _pair_paths(z1, z2, a, b, qbar)
    names = [f"{p[0]}|{p[1]}" for p in pairs]
    return {"alpha": a, "beta": b, "nll": res.fun,
            "corr": pd.DataFrame(rho, index=Z.index, columns=names)}


def dcc_forecast(rho_t: float, qbar: float, a: float, b: float, h: int) -> float:
    """h-step-ahead DCC correlation forecast (standard approximation:
    mean reversion of the correlation at rate (a+b)^h toward its long-run level)."""
    return qbar + (a + b) ** h * (rho_t - qbar)
