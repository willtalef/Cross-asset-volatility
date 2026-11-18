"""Bivariate copulas (paper Section 3.3, eq. 5) and copula-based VaR (Section 7.3).

Families: Gaussian, Student-t, Clayton (lower-tail / crash dependence),
Gumbel (upper-tail / boom dependence). Selection by AIC. Margins are
estimated non-parametrically (pseudo-observations from ranks; the paper uses a
kernel-smoothed empirical CDF, which gives very similar pseudo-observations).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar, minimize


def pseudo_obs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return stats.rankdata(x) / (len(x) + 1.0)


# ---------------------------------------------------------- log-densities
def ll_gaussian(u, v, rho):
    x, y = stats.norm.ppf(u), stats.norm.ppf(v)
    r2 = 1 - rho ** 2
    return np.sum(-0.5 * np.log(r2) - (rho ** 2 * (x ** 2 + y ** 2) - 2 * rho * x * y) / (2 * r2))


def ll_student(u, v, rho, nu):
    x, y = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
    r2 = 1 - rho ** 2
    from scipy.special import gammaln
    lc = (gammaln((nu + 2) / 2) + gammaln(nu / 2) - 2 * gammaln((nu + 1) / 2) - 0.5 * np.log(r2)
          - (nu + 2) / 2 * np.log1p((x ** 2 + y ** 2 - 2 * rho * x * y) / (nu * r2))
          + (nu + 1) / 2 * (np.log1p(x ** 2 / nu) + np.log1p(y ** 2 / nu)))
    return np.sum(lc)


def ll_clayton(u, v, th):
    return np.sum(np.log1p(th) - (th + 1) * (np.log(u) + np.log(v))
                  - (2 + 1 / th) * np.log(u ** -th + v ** -th - 1))


def ll_gumbel(u, v, th):
    lu, lv = -np.log(u), -np.log(v)
    A = lu ** th + lv ** th
    lC = -A ** (1 / th)
    return np.sum(lC - np.log(u) - np.log(v) + (th - 1) * (np.log(lu) + np.log(lv))
                  + (2 / th - 2) * np.log(A) + np.log1p((th - 1) * A ** (-1 / th)))


# ---------------------------------------------------------- fitting
def fit_all(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Fit the four families on a pair of return series; report parameter,
    log-likelihood, AIC and tail-dependence coefficients."""
    u, v = pseudo_obs(x), pseudo_obs(y)
    out = {}
    tau = stats.kendalltau(u, v)[0]
    r = minimize_scalar(lambda p: -ll_gaussian(u, v, p), bounds=(-0.99, 0.99), method="bounded")
    out["gaussian"] = {"param": r.x, "ll": -r.fun, "k": 1, "lambda_L": 0.0, "lambda_U": 0.0}
    rt = minimize(lambda p: -ll_student(u, v, p[0], p[1]), [np.sin(np.pi * tau / 2), 8.0],
                  bounds=[(-0.99, 0.99), (2.05, 100)], method="L-BFGS-B")
    rho, nu = rt.x
    lam_t = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
    out["student_t"] = {"param": (rho, nu), "ll": -rt.fun, "k": 2, "lambda_L": lam_t, "lambda_U": lam_t}
    if tau > 0:
        rc = minimize_scalar(lambda p: -ll_clayton(u, v, p), bounds=(1e-4, 30), method="bounded")
        out["clayton"] = {"param": rc.x, "ll": -rc.fun, "k": 1,
                          "lambda_L": 2 ** (-1 / rc.x), "lambda_U": 0.0}
        rg = minimize_scalar(lambda p: -ll_gumbel(u, v, p), bounds=(1.0001, 30), method="bounded")
        out["gumbel"] = {"param": rg.x, "ll": -rg.fun, "k": 1,
                         "lambda_L": 0.0, "lambda_U": 2 - 2 ** (1 / rg.x)}
    df = pd.DataFrame(out).T
    df["aic"] = 2 * df["k"].astype(float) - 2 * df["ll"].astype(float)
    df["best_by_aic"] = df["aic"] == df["aic"].min()
    return df


def rolling_clayton(x: pd.Series, y: pd.Series, window: int = 36, step: int = 1) -> pd.Series:
    """Time-varying Clayton parameter from rolling windows (paper: 3-year window,
    1-month step, on monthly returns)."""
    d = pd.concat([x, y], axis=1).dropna()
    out = {}
    for end in range(window, len(d) + 1, step):
        w = d.iloc[end - window:end]
        u, v = pseudo_obs(w.iloc[:, 0].values), pseudo_obs(w.iloc[:, 1].values)
        if stats.kendalltau(u, v)[0] <= 0:
            out[w.index[-1]] = np.nan; continue
        r = minimize_scalar(lambda p: -ll_clayton(u, v, p), bounds=(1e-4, 30), method="bounded")
        out[w.index[-1]] = r.x
    return pd.Series(out, name="clayton_theta")


# ---------------------------------------------------------- simulation & VaR
def sim_gaussian(n, rho, rng):
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    return stats.norm.cdf(z)


def sim_clayton(n, th, rng):
    """Marshall-Olkin algorithm."""
    V = rng.gamma(1 / th, 1.0, size=n)
    E = rng.exponential(size=(n, 2))
    return (1 + E / V[:, None]) ** (-1 / th)


def copula_var(r_eq: pd.Series, r_cmd: pd.Series, w_cmd: float = 0.125, horizon: int = 10,
               alpha: float = 0.01, n_sims: int = 200_000, clayton_theta: float | None = None,
               seed: int = 42) -> pd.Series:
    """10-day 1% VaR of a portfolio (1-w_cmd) equity + w_cmd commodity, with
    empirical margins joined by (i) a Gaussian copula and (ii) a Clayton copula.
    Returns both VaRs (positive numbers = loss) and the % increase (paper: +8-14%)."""
    d = pd.concat([r_eq, r_cmd], axis=1).dropna()
    x, y = d.iloc[:, 0].values, d.iloc[:, 1].values
    fits = fit_all(x, y)
    rho = float(fits.loc["gaussian", "param"])
    if clayton_theta is None and "clayton" not in fits.index:
        # Kendall's tau <= 0: no positive (crash) dependence, Clayton not applicable
        return pd.Series({"VaR_gaussian": np.nan, "VaR_clayton": np.nan, "increase_pct": np.nan,
                          "rho_gaussian": float(fits.loc["gaussian", "param"]), "theta_clayton": np.nan,
                          "note": "tau<=0: Clayton not applicable"})
    th = clayton_theta if clayton_theta is not None else float(fits.loc["clayton", "param"])
    rng = np.random.default_rng(seed)
    qx, qy = np.sort(x), np.sort(y)

    def inv(q, u):
        return np.quantile(q, u, method="linear")

    res = {}
    for name, sim in (("gaussian", lambda n: sim_gaussian(n, rho, rng)),
                      ("clayton", lambda n: sim_clayton(n, th, rng))):
        U = sim(n_sims * horizon)
        port = (1 - w_cmd) * inv(qx, U[:, 0]) + w_cmd * inv(qy, U[:, 1])
        h = port.reshape(n_sims, horizon).sum(1)          # i.i.d. daily draws aggregated
        res[f"VaR_{name}"] = -np.quantile(h, alpha)
    res["increase_pct"] = 100 * (res["VaR_clayton"] / res["VaR_gaussian"] - 1)
    res["rho_gaussian"], res["theta_clayton"] = rho, th
    return pd.Series(res)
