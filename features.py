"""Treatment of forecasting features that start after January 1990
(revision added November 2026 – see DATA_APPENDIX.md).

Three strategies, all chosen BEFORE looking at forecasting results:

1. "proxy" (main specification): before a series exists, use a documented
   long-history proxy, re-scaled by OLS on the overlap period (splicing).
   A 0/1 flag column records which months use the proxy.
2. "native": leave the months before the series starts as missing; tree
   models handle missing values natively, FAVAR uses only the features
   available over its estimation window.
3. "start_late": drop the late-starting features entirely (robustness check).

Results must be reported for the main specification, with the other two as
robustness checks, whatever they show.
"""
from __future__ import annotations
from cross_asset import MONTH_END
import numpy as np
import pandas as pd

# Late-starting series and their documented proxies (all available from 1990).
# name -> (proxy description, FRED/computed source of the proxy)
PROXIES = {
    "HY_SPREAD": ("Moody's Baa corporate yield minus 10-year Treasury", "FRED: BAA10Y (from 1986)"),
    "IG_SPREAD": ("Moody's Aaa corporate yield minus 10-year Treasury", "FRED: AAA10Y (from 1983)"),
    "REAL_RATE": ("Cleveland Fed 10-year real interest rate model", "FRED: REAINTRATREARAT10Y (from 1982)"),
    "OVX": ("30-day realised volatility of WTI crude, annualised", "computed from daily WTI prices (EIA, from 1986)"),
    "GVZ": ("30-day realised volatility of gold, annualised", "computed from daily gold prices (LBMA)"),
    # CHINA_PMI (2005): no reliable monthly proxy before 2005 -> left missing
    # under 'proxy' and 'native'; dropped under 'start_late'.
}
NO_PROXY = ["CHINA_PMI", "CITI_SURPRISE"]


def realised_vol(prices: pd.Series, window: int = 21) -> pd.Series:
    """Annualised rolling realised volatility (in %, like OVX/GVZ), month-end."""
    r = np.log(prices).diff()
    return (r.rolling(window).std() * np.sqrt(252) * 100).resample(MONTH_END).last()


def splice(target: pd.Series, proxy: pd.Series, min_overlap: int = 36) -> tuple[pd.Series, pd.Series, dict]:
    """Fill `target` before its first valid date with a + b * proxy, where a, b
    are estimated by OLS on the overlap. Returns spliced series, 0/1 proxy flag,
    and the fit statistics (to be reported in the data appendix)."""
    t, p = target.copy(), proxy.reindex(target.index.union(proxy.index))
    t = t.reindex(p.index)
    ov = pd.concat([t, p], axis=1).dropna()
    if len(ov) < min_overlap:
        raise ValueError(f"overlap too short ({len(ov)} months) to splice")
    b, a = np.polyfit(ov.iloc[:, 1], ov.iloc[:, 0], 1)
    r = np.corrcoef(ov.iloc[:, 0], ov.iloc[:, 1])[0, 1]
    first = target.first_valid_index()
    fitted = a + b * p
    out = t.copy()
    early = out.index < first
    out[early] = fitted[early]
    flag = pd.Series(early.astype(int), index=out.index)
    return out, flag, {"intercept": a, "slope": b, "overlap_corr": r, "overlap_months": len(ov),
                       "proxy_until": str(first.date())}


def prepare(features: pd.DataFrame, proxies: pd.DataFrame | None, strategy: str = "proxy"):
    """Apply one of the three strategies. `proxies` holds monthly proxy series
    with the same column names as the targets they replace."""
    f = features.copy()
    report = {}
    late = [c for c in f.columns if f[c].first_valid_index() is not None
            and f[c].first_valid_index() > f.index[0] + pd.offsets.MonthEnd(6)]
    if strategy == "start_late":
        report = {c: "dropped (starts " + str(f[c].first_valid_index().date()) + ")" for c in late}
        return f.drop(columns=late), report
    if strategy == "proxy" and proxies is not None:
        for c in late:
            if c in proxies.columns and c in PROXIES:
                f[c], flag, stats = splice(f[c], proxies[c])
                f[c + "_is_proxy"] = flag.reindex(f.index).fillna(0)
                report[c] = {"proxy": PROXIES[c][0], "source": PROXIES[c][1], **stats}
            else:
                report[c] = "left missing before " + str(f[c].first_valid_index().date())
        return f, report
    report = {c: "left missing before " + str(f[c].first_valid_index().date()) for c in late}
    return f, report
