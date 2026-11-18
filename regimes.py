"""Correlation regimes (paper Section 4.2-4.3, eq. 15).

* GaussianHMM: univariate Gaussian hidden Markov model fitted by EM
  (Baum-Welch) on the DCC correlation series; Viterbi decoding.
* Two-state analysis reports the transition matrix, expected durations
  1/(1-p_ii) and the stationary distribution.

Note: the paper's RS-DCC (Pelletier 2006) estimates regimes inside the
correlation model. Here the two-state model is fitted on the DCC series,
which is a simpler two-step approximation.
"""
from __future__ import annotations
from cross_asset import MONTH_END
import numpy as np
import pandas as pd
from scipy.stats import norm


class GaussianHMM:
    def __init__(self, n_states: int = 2, n_iter: int = 500, tol: float = 1e-8, seed: int = 0):
        self.k, self.n_iter, self.tol, self.seed = n_states, n_iter, tol, seed

    def _emissions(self, x):
        return np.column_stack([norm.pdf(x, self.mu[j], self.sd[j]) for j in range(self.k)]) + 1e-300

    def fit(self, x: np.ndarray):
        x = np.asarray(x, float)
        T, k = len(x), self.k
        qs = np.quantile(x, np.linspace(0.1, 0.9, k))
        self.mu, self.sd = qs.copy(), np.full(k, x.std() / k + 1e-6)
        self.A = np.full((k, k), 0.05 / max(k - 1, 1)); np.fill_diagonal(self.A, 0.95)
        self.pi = np.full(k, 1.0 / k)
        prev = -np.inf
        for _ in range(self.n_iter):
            B = self._emissions(x)
            alpha = np.zeros((T, k)); c = np.zeros(T)
            alpha[0] = self.pi * B[0]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ self.A) * B[t]; c[t] = alpha[t].sum(); alpha[t] /= c[t]
            beta = np.ones((T, k))
            for t in range(T - 2, -1, -1):
                beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
            gamma = alpha * beta; gamma /= gamma.sum(1, keepdims=True)
            xi = np.zeros((k, k))
            for t in range(T - 1):
                m = np.outer(alpha[t], B[t + 1] * beta[t + 1]) * self.A / c[t + 1]
                xi += m / m.sum()
            self.pi = gamma[0]
            self.A = xi / xi.sum(1, keepdims=True)
            w = gamma.sum(0)
            self.mu = (gamma * x[:, None]).sum(0) / w
            self.sd = np.sqrt((gamma * (x[:, None] - self.mu) ** 2).sum(0) / w) + 1e-6
            ll = np.log(c).sum()
            if abs(ll - prev) < self.tol:
                break
            prev = ll
        order = np.argsort(self.mu)                     # state 0 = lowest mean correlation
        self.mu, self.sd = self.mu[order], self.sd[order]
        self.A, self.pi = self.A[np.ix_(order, order)], self.pi[order]
        self.loglik, self.gamma = ll, gamma[:, order]
        return self

    def viterbi(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float); T, k = len(x), self.k
        lB, lA = np.log(self._emissions(x)), np.log(self.A + 1e-300)
        d = np.log(self.pi + 1e-300) + lB[0]; psi = np.zeros((T, k), int)
        for t in range(1, T):
            s = d[:, None] + lA; psi[t] = s.argmax(0); d = s.max(0) + lB[t]
        path = np.zeros(T, int); path[-1] = d.argmax()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path


def regime_summary(series: pd.Series, n_states: int) -> dict:
    """Fit an n-state HMM; return state means, transition matrix, expected
    durations (in periods) and stationary probabilities."""
    x = series.dropna()
    hmm = GaussianHMM(n_states).fit(x.values)
    states = pd.Series(hmm.viterbi(x.values), index=x.index, name="state")
    A = hmm.A
    w, v = np.linalg.eig(A.T)
    stat = np.real(v[:, np.argmin(np.abs(w - 1))]); stat = stat / stat.sum()
    return {"model": hmm, "states": states,
            "state_means": pd.Series(hmm.mu, name="mean_corr"),
            "transition": pd.DataFrame(A),
            "expected_duration": pd.Series(1 / (1 - np.diag(A)), name="periods"),
            "stationary": pd.Series(stat, name="share_of_time"),
            "episodes": _episodes(states, x)}


def _episodes(states: pd.Series, x: pd.Series) -> pd.DataFrame:
    """Contiguous regime spells with their mean correlation."""
    g = (states != states.shift()).cumsum()
    rows = []
    for _, s in states.groupby(g):
        rows.append({"start": s.index[0], "end": s.index[-1], "state": int(s.iloc[0]),
                     "mean_corr": float(x.loc[s.index].mean()), "length": len(s)})
    return pd.DataFrame(rows)


def threshold_regime(series: pd.Series, low: float = 0.20, high: float = 0.40) -> pd.Series:
    """Paper's two-regime labelling: L if rho<0.20, H if rho>0.40 (else NaN)."""
    lab = pd.Series(np.nan, index=series.index)
    lab[series < low] = 0; lab[series > high] = 1
    return lab


def period_means(series: pd.Series, periods: dict[str, tuple[str, str]]) -> pd.Series:
    """Mean correlation over user-defined eras (e.g. the paper's four regimes)."""
    return pd.Series({k: series.loc[a:b].mean() for k, (a, b) in periods.items()})


def crisis_table(corr: pd.DataFrame, crises: dict[str, tuple[str, str]],
                 normal_level: pd.Series | None = None, tol: float = 0.05) -> pd.DataFrame:
    """Peak correlation per crisis window (paper Table 3), months from window
    start to peak, and months from peak until the correlation is back within
    `tol` of its pre-crisis 12-month average (normalisation)."""
    m = corr.resample(MONTH_END).mean()
    out = []
    for name, (a, b) in crises.items():
        row = {"crisis": name}
        for col in m.columns:
            w = m.loc[a:b, col]
            if w.empty:
                continue
            peak_date = w.idxmax()
            row[f"peak_{col}"] = w.max()
            pre = m[col].loc[:a].iloc[-13:-1].mean()
            after = m[col].loc[peak_date:]
            back = after[after <= pre + tol]
            row[f"m_to_peak_{col}"] = len(m.loc[a:peak_date]) - 1
            row[f"m_to_norm_{col}"] = (len(m.loc[peak_date:back.index[0]]) - 1) if len(back) else np.nan
        out.append(row)
    return pd.DataFrame(out).set_index("crisis")
