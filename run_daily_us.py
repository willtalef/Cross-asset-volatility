"""US daily extension, 1990-2026 (data received 2026-11-02).

Equity: S&P 500 total return (Yahoo ^SP500TR, daily from 1990-01-02).
Commodities: S&P GSCI (Yahoo ^SPGSCI, daily from 1990-01-02); Brent spot
(FRED DCOILBRENTEU, daily from 1990; WTI is not used for log returns because
it traded at a negative price on 2020-04-20); gold, copper and corn front-month
futures (Yahoo GC=F, HG=F, ZC=F, daily from 2000 only).
Forecast features: FRED monthly file (see run_collected_data.py).
"""
from __future__ import annotations
import os as _os, sys as _sys
try:  # always run from the project folder, whatever the current directory (Spyder, PyCharm, Jupyter %run)
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
except NameError:
    _HERE = _os.getcwd()
_os.chdir(_HERE); _sys.path.insert(0, _HERE)
from cross_asset import MONTH_END
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cross_asset import dcc, regimes, copulas, forecasting, stats_utils

OUT = Path("results_daily_us"); OUT.mkdir(exist_ok=True)
y = pd.read_csv("data/collected_yahoo_daily.csv", parse_dates=["date"], index_col="date")
fd = pd.read_csv("data/collected_fred_daily.csv", parse_dates=["date"], index_col="date")
px = pd.DataFrame({"SP500TR": y["^SP500TR"], "GSCI": y["^SPGSCI"], "BRENT": fd["DCOILBRENTEU"],
                   "GOLD": y["GC=F"], "COPPER": y["HG=F"], "CORN": y["ZC=F"]}).loc["1990-01-01":"2026-03-31"]
px.to_csv(OUT / "prices_used.csv")
log = []

# daily log returns on common trading days (no filling across missing days)
def rets(cols, start=None):
    p = px[cols].loc[start:] if start else px[cols]
    p = p.dropna()
    return np.log(p).diff().dropna()

r3 = rets(["SP500TR", "GSCI", "BRENT"])
log.append(f"Main daily sample: {r3.index[0].date()} to {r3.index[-1].date()} ({len(r3)} common trading days)")
r_m = np.log(px.resample(MONTH_END).last()).diff().loc["1990-02-28":]

# ---- Table 2 analogue (monthly, end-of-month)
stats_utils.corr_table(r_m, ["SP500TR"], ["GSCI", "BRENT", "GOLD", "COPPER", "CORN"]).to_csv(OUT / "table2_unconditional_corr_monthly.csv")

# ---- DCC daily
Z, gp = dcc.standardize(r3); gp.to_csv(OUT / "garch_params.csv")
D = dcc.fit_dcc(Z, pairs=[("SP500TR", "GSCI"), ("SP500TR", "BRENT")])
D["corr"].to_csv(OUT / "dcc_daily.csv")
log.append(f"DCC daily alpha={D['alpha']:.4f} beta={D['beta']:.4f}")
r2k = rets(["SP500TR", "GOLD", "COPPER", "CORN"], "2000-09-01")
Z2, _ = dcc.standardize(r2k)
D2 = dcc.fit_dcc(Z2, pairs=[("SP500TR", "GOLD"), ("SP500TR", "COPPER"), ("SP500TR", "CORN")])
D2["corr"].to_csv(OUT / "dcc_daily_2000plus.csv")
allc = pd.concat([D["corr"], D2["corr"]], axis=1, sort=True)
cm = allc.resample(MONTH_END).mean()

eras = {"1990-2003 (Regime 1)": ("1990-01-01", "2003-12-31"), "2004-2011 (Regime 2)": ("2004-01-01", "2011-12-31"),
        "2012-2019 (Regime 3)": ("2012-01-01", "2019-12-31"), "2020-2026 (Regime 4)": ("2020-01-01", "2026-03-31")}
era = pd.DataFrame({k: allc.loc[a:b].mean() for k, (a, b) in eras.items()}).T; era.to_csv(OUT / "era_means.csv")

rho = cm["SP500TR|GSCI"].dropna()
h4 = regimes.regime_summary(rho, 4); h4["episodes"].to_csv(OUT / "hmm4_episodes.csv", index=False)
h2 = regimes.regime_summary(rho, 2)
hm = pd.concat([h2["transition"], h2["expected_duration"], h2["stationary"], h2["state_means"]], axis=1); hm.to_csv(OUT / "hmm2.csv")

crises = {"Gulf War I": ("1990-08-01", "1991-03-31"), "Asia/Russia/LTCM": ("1997-07-01", "1998-12-31"),
          "Dot-Com Bust": ("2000-03-01", "2002-10-31"), "Global Financial Crisis": ("2008-09-01", "2009-06-30"),
          "Euro Debt Crisis": ("2011-07-01", "2012-07-31"), "Oil Price Collapse": ("2014-07-01", "2016-02-29"),
          "COVID-19 Shock": ("2020-02-01", "2020-06-30"), "Inflation Surge": ("2021-06-01", "2023-06-30"),
          "Rate Cycle Peak": ("2023-07-01", "2024-12-31")}
pk = pd.DataFrame({k: cm.loc[a:b, ["SP500TR|GSCI", "SP500TR|BRENT", "SP500TR|GOLD", "SP500TR|CORN"]].max()
                   for k, (a, b) in crises.items()}).T
pk_daily = pd.DataFrame({k: allc.loc[a:b, ["SP500TR|GSCI", "SP500TR|BRENT"]].max() for k, (a, b) in crises.items()}).T
pk.to_csv(OUT / "table3_crisis_peaks_monthly_avg.csv"); pk_daily.to_csv(OUT / "table3_crisis_peaks_daily_max.csv")

# ---- copulas & 10-day VaR
cop = pd.concat({c: copulas.fit_all(r_m["SP500TR"].dropna().reindex(r_m[c].dropna().index).dropna().values,
                                     r_m[c].reindex(r_m["SP500TR"].dropna().index).dropna().values)
                 for c in ["GSCI", "BRENT"]})
cop.to_csv(OUT / "copula_fits_monthly.csv")
copd = pd.concat({c: copulas.fit_all(r3["SP500TR"].values, r3[c].values) for c in ["GSCI", "BRENT"]})
copd.to_csv(OUT / "copula_fits_daily.csv")
var = {}
for c in ["GSCI", "BRENT"]:
    for w in (0.10, 0.125, 0.15):
        var[f"{c}_w{w}"] = copulas.copula_var(r3["SP500TR"], r3[c], w_cmd=w, horizon=10)
var = pd.DataFrame(var).T; var.to_csv(OUT / "var10d_gaussian_vs_clayton.csv")
# time-varying Clayton: last 3 years of daily data
last = r3.loc["2023-04-01":]
var_tv = pd.DataFrame({f"{c}_last3y": copulas.copula_var(last["SP500TR"], last[c], w_cmd=0.125, horizon=10)
                       for c in ["GSCI", "BRENT"]}).T
var_tv.to_csv(OUT / "var10d_last3y.csv")

# ---- forecasting (monthly average of daily DCC, h=3)
f2 = pd.read_csv("data/collected_fred_monthly_v2.csv", parse_dates=["date"], index_col="date")
f2.index = f2.index.to_period("M").to_timestamp("M")
cpi = f2["CPIAUCSL"].interpolate(limit=1, limit_area="inside")
feats = pd.DataFrame({"VIX": f2["VIXCLS"], "OVX": f2["OVXCLS"], "GVZ": f2["GVZCLS"],
                      "CREDIT_SPREAD_BAA10Y": f2["BAA10Y"], "CREDIT_SPREAD_AAA10Y": f2["AAA10Y"],
                      "TED_SPREAD": f2["TEDRATE"], "TERM_SPREAD_10Y2Y": f2["T10Y2Y"],
                      "REAL_RATE": f2["DFII10"].combine_first(f2["REAINTRATREARAT10Y"]),
                      "CPI_YOY": 100 * np.log(cpi).diff(12), "INDPRO_YOY": 100 * np.log(f2["INDPRO"]).diff(12),
                      "USREC": f2["USREC"]})
feats.loc[feats.index > f2["TEDRATE"].last_valid_index(), "TED_SPREAD"] = np.nan
pers = (D["alpha"] + D["beta"]) ** 21
res = {}
for name, f in (("main", feats), ("start_late_robustness", feats.drop(columns=["OVX", "GVZ", "TED_SPREAD"])),
                ("correlation_features_only", None)):
    bt = forecasting.backtest(rho, f, h=3, min_train=120, refit_every=12, dcc_persistence=pers, n_factors=5)
    bt["forecasts"].to_csv(OUT / f"oos_forecasts_{name}.csv"); bt["metrics"].to_csv(OUT / f"table4_{name}.csv")
    res[name] = bt
fc = res["main"]["forecasts"]
log.append(f"Forecast OOS: {fc.index[0].date()} to {fc.index[-1].date()} ({len(fc)} months)")


print("\n".join(log))
print("\nTABLE2\n", pd.read_csv(OUT / "table2_unconditional_corr_monthly.csv", index_col=0).to_string())
print("\nERAS\n", era.round(3).to_string())
print("\nHMM2\n", hm.round(3).to_string(), "\nHMM4 means", h4["state_means"].round(3).tolist())
print("\nPEAKS (monthly avg of daily DCC)\n", pk.round(3).to_string())
print("\nPEAKS (daily max)\n", pk_daily.round(3).to_string())
print("\nCOPULA daily best\n", copd[["param", "aic", "best_by_aic"]].to_string())
print("\nCOPULA monthly best\n", cop[["param", "aic", "best_by_aic"]].to_string())
print("\nVAR 10d\n", var.round(4).to_string(), "\n", var_tv.round(4).to_string())
for k, v in res.items():
    print("\nTABLE4", k, "\n", v["metrics"].round(3).to_string())
(OUT / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
