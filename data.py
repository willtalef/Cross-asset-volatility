"""Data loading and return construction (paper Section 3.1)."""
from __future__ import annotations
from cross_asset import MONTH_END
import numpy as np
import pandas as pd


def load_prices(path: str) -> pd.DataFrame:
    """Load a wide CSV of prices/index levels in USD.

    Expected layout: first column 'date', one column per series
    (e.g. MSCI_ACWI, SP500, BCOM_ENERGY, GOLD, LME_COPPER, ...).
    Total-return (dividend-adjusted) levels should be used where available.
    """
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def log_returns(prices: pd.DataFrame, freq: str | None = None) -> pd.DataFrame:
    """Log returns. freq=None keeps native (daily) frequency;
    freq='ME' uses end-of-month prices (paper's main analysis)."""
    p = prices if freq is None else prices.resample(freq).last()
    r = np.log(p).diff()
    return r.dropna(how="all")


def geltner_unsmooth(r: pd.Series, rho: float | None = None) -> pd.Series:
    """Geltner (1993) desmoothing for appraisal-based series (private equity).
    r_true_t = (r_obs_t - rho * r_obs_{t-1}) / (1 - rho), rho = AR(1) coefficient."""
    x = r.dropna()
    if rho is None:
        rho = float(np.corrcoef(x.values[1:], x.values[:-1])[0, 1])
    out = (x - rho * x.shift(1)) / (1.0 - rho)
    return out.dropna()


def load_features(path: str) -> pd.DataFrame:
    """Load monthly macro / financial-conditions features (paper Section 6.1),
    e.g. VIX, HY_SPREAD, IG_SPREAD, TERM_SPREAD, CPI_YOY, ISM_PMI, CHINA_PMI,
    USD_INDEX, CFTC_OIL_NET, CFTC_GOLD_NET, BDI, ... First column 'date'."""
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df.resample(MONTH_END).last()
