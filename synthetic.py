"""Synthetic data generator used ONLY to smoke-test the pipeline.
Its numbers have no relation to the paper's results."""
from __future__ import annotations
from cross_asset import MONTH_END
import numpy as np
import pandas as pd


def make_synthetic(start="1990-01-01", end="2026-03-31", seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    T = len(idx)
    # piecewise "true" correlation regimes between equity and commodity
    eras = [("1990", "2003", 0.08), ("2004", "2011", 0.38), ("2012", "2019", 0.31), ("2020", "2026", 0.42)]
    rho = np.zeros(T)
    for a, b, r in eras:
        m = (idx >= a) & (idx <= f"{b}-12-31")
        rho[m] = r
    for a, b, peak in [("2008-09-01", "2009-03-31", 0.72), ("2020-02-15", "2020-05-31", 0.78)]:
        m = (idx >= a) & (idx <= b); rho[m] = peak
    # GARCH-like volatility
    vol = np.zeros((T, 3)); vol[0] = 0.01
    eps = np.zeros((T, 3))
    for t in range(T):
        if t:
            vol[t] = np.sqrt(1e-6 + 0.06 * eps[t - 1] ** 2 + 0.92 * vol[t - 1] ** 2)
        r = rho[t]
        C = np.array([[1, r, -0.05], [r, 1, 0.0], [-0.05, 0.0, 1]])
        z = np.linalg.cholesky(C) @ rng.standard_normal(3)
        eps[t] = vol[t] * z
    prices = pd.DataFrame(100 * np.exp(np.cumsum(eps + [0.0003, 0.0001, 0.0001], 0)), index=idx,
                          columns=["EQUITY", "COMMODITY", "GOLD"])
    prices.index.name = "date"
    months = pd.date_range(start, end, freq=MONTH_END)
    feats = pd.DataFrame({"VIX": 18 + 8 * rng.standard_normal(len(months)).cumsum() * 0.05,
                          "HY_SPREAD": 4 + rng.standard_normal(len(months)).cumsum() * 0.1,
                          "TERM_SPREAD": rng.standard_normal(len(months)),
                          "CPI_YOY": 2 + rng.standard_normal(len(months)) * 0.5,
                          "CFTC_OIL_NET": rng.standard_normal(len(months)).cumsum(),
                          "CHINA_PMI": 50 + rng.standard_normal(len(months))}, index=months)
    feats.index.name = "date"
    # mimic late-starting series and their long-history proxies
    proxies = pd.DataFrame({"HY_SPREAD": feats["HY_SPREAD"] * 0.8 + rng.standard_normal(len(months)) * 0.2},
                           index=months)
    feats.loc[:"1996-11-30", "HY_SPREAD"] = np.nan
    feats.loc[:"2004-12-31", "CHINA_PMI"] = np.nan
    return prices, feats, proxies
