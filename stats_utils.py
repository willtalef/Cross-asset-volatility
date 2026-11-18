"""Unconditional correlations with Newey-West (HAC) significance (paper Table 2)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def nw_corr_test(x: np.ndarray, y: np.ndarray, lags: int | None = None) -> tuple[float, float]:
    """Correlation and HAC t-statistic, from the regression of standardized y on standardized x."""
    d = np.column_stack([x, y]); d = d[~np.isnan(d).any(1)]
    a = (d[:, 0] - d[:, 0].mean()) / d[:, 0].std()
    b = (d[:, 1] - d[:, 1].mean()) / d[:, 1].std()
    n = len(a); beta = np.dot(a, b) / np.dot(a, a)
    e = b - beta * a; u = a * e
    L = lags if lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    s = np.dot(u, u)
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        s += 2 * w * np.dot(u[l:], u[:-l])
    se = np.sqrt(s) / np.dot(a, a)
    return float(np.corrcoef(a, b)[0, 1]), float(beta / se)


def corr_table(r: pd.DataFrame, rows: list[str], cols: list[str], crit: float = 1.96) -> pd.DataFrame:
    out = pd.DataFrame(index=rows, columns=cols, dtype=object)
    for i in rows:
        for j in cols:
            c, t = nw_corr_test(r[i].values, r[j].values)
            out.loc[i, j] = f"{c:.2f}{'*' if abs(t) > crit else ''}"
    return out
