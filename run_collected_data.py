"""Run the methodology on the data actually collected on 2026-11-02.

Data (see DATA_PROVENANCE.md):
* MSCI ACWI, World, EM and 47 country indices – msci.com, gross USD,
  month-end levels, 1998-12 to 2026-03 (public history starts 1998-12).
* World Bank Commodity Price Data ("Pink Sheet"), monthly AVERAGE prices and
  indices, 1990-01 to 2026-03.
* FRED monthly: CPIAUCSL, INDPRO, USREC, REAINTRATREARAT10Y.

Differences from the paper's design, imposed by the data:
* Monthly frequency only (no daily data): GARCH/DCC are estimated on monthly
  returns, and VaR is a 1-month 1% VaR instead of a 10-day 1% VaR.
* Equity data start in 1998-12, so the sample is 1999-01 to 2026-03
  (not 1990-2026).
* World Bank series are monthly averages, not month-end prices; averaging
  smooths returns and tends to lower measured volatility and correlation.
* Forecasting uses 4 macro features (not 47): CPI and industrial production
  growth, NBER recession dummy, 10-year real rate.
"""
from __future__ import annotations
import os as _os, sys as _sys
try:  # always run from the project folder, whatever the current directory (Spyder, PyCharm, Jupyter %run)
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
except NameError:
    _HERE = _os.getcwd()
_os.chdir(_HERE); _sys.path.insert(0, _HERE)

import json
from pathlib import Path
import numpy as np
import pandas as pd
from cross_asset import dcc, regimes, copulas, forecasting, stats_utils

OUT = Path("results_collected"); OUT.mkdir(exist_ok=True)
msci = pd.read_csv("data/collected_msci_monthly_eom.csv", parse_dates=["date"], index_col="date")
wb = pd.read_csv("data/collected_worldbank_monthly_avg.csv", parse_dates=["date"], index_col="date")
fred = pd.read_csv("data/collected_fred_monthly.csv", parse_dates=["date"], index_col="date")

CMD = {"WB_TOTAL": "Broad commodities", "WB_ENERGY": "Energy", "WB_PRECIOUS": "Precious metals",
       "WB_BASE_METALS": "Base metals", "WB_AGRI": "Agriculture", "WB_GOLD": "Gold"}
r = np.log(pd.concat([msci, wb[list(CMD)]], axis=1)).diff()
r = r.loc["1999-01-31":"2026-03-31"]
log = [f"Sample: {r.index[0].date()} to {r.index[-1].date()} ({len(r)} monthly returns)"]

# ---- Table 2 analogue
eqs = ["MSCI_ACWI", "MSCI_World", "MSCI_EM", "MSCI_USA"]
t2 = stats_utils.corr_table(r, eqs, list(CMD)); t2.to_csv(OUT / "table2_unconditional_corr.csv")

# ---- countries
countries = [c for c in msci.columns if c not in ("MSCI_ACWI", "MSCI_World", "MSCI_EM")]
rows = []
for c in countries:
    d = r[[c, "WB_TOTAL", "WB_ENERGY"]].dropna()
    rows.append({"country": c.replace("MSCI_", "").replace("_", " "), "n_months": len(d),
                 "corr_broad": d[c].corr(d["WB_TOTAL"]), "corr_energy": d[c].corr(d["WB_ENERGY"])})
ct = pd.DataFrame(rows).sort_values("corr_broad", ascending=False); ct.to_csv(OUT / "country_commodity_corr.csv", index=False)
log.append(f"Countries with data: {len(countries)} (paper: 42)")

# ---- GARCH + DCC (monthly)
series = ["MSCI_ACWI"] + [c for c in CMD if c != "WB_GOLD"]
Z, gp = dcc.standardize(r[series].dropna()); gp.to_csv(OUT / "garch_params.csv")
D = dcc.fit_dcc(Z, pairs=[("MSCI_ACWI", c) for c in series[1:]])
D["corr"].to_csv(OUT / "dcc_monthly_correlations.csv")
log.append(f"DCC (monthly) alpha={D['alpha']:.4f} beta={D['beta']:.4f}")
rho = D["corr"]["MSCI_ACWI|WB_TOTAL"]

eras = {"1999-2003 (part of paper Regime 1)": ("1999-01-31", "2003-12-31"),
        "2004-2011 (Regime 2)": ("2004-01-31", "2011-12-31"),
        "2012-2019 (Regime 3)": ("2012-01-31", "2019-12-31"),
        "2020-2026 (Regime 4)": ("2020-01-31", "2026-03-31")}
era_tab = pd.DataFrame({k: D["corr"].loc[a:b].mean() for k, (a, b) in eras.items()}).T
era_tab.to_csv(OUT / "era_mean_dcc_correlations.csv")

r4 = regimes.regime_summary(rho, 4); r4["episodes"].to_csv(OUT / "hmm4_episodes.csv", index=False)
r4["state_means"].to_csv(OUT / "hmm4_state_means.csv")
r2 = regimes.regime_summary(rho, 2)
tr = pd.concat([r2["transition"], r2["expected_duration"], r2["stationary"], r2["state_means"]], axis=1)
tr.to_csv(OUT / "hmm2_transition_durations.csv")

crises = {"Dot-Com Bust": ("2000-03-31", "2002-10-31"), "Global Financial Crisis": ("2008-09-30", "2009-06-30"),
          "Euro Debt Crisis": ("2011-07-31", "2012-07-31"), "Oil Price Collapse": ("2014-07-31", "2016-02-29"),
          "COVID-19 Shock": ("2020-02-29", "2020-06-30"), "Inflation Surge": ("2021-06-30", "2023-06-30"),
          "Rate Cycle Peak": ("2023-07-31", "2024-12-31")}
cr_pairs = ["MSCI_ACWI|WB_ENERGY", "MSCI_ACWI|WB_PRECIOUS", "MSCI_ACWI|WB_AGRI"]
peaks = pd.DataFrame({k: D["corr"].loc[a:b, cr_pairs].max() for k, (a, b) in crises.items()}).T
peaks.columns = ["peak_eq_energy", "peak_eq_precious", "peak_eq_agri"]; peaks.to_csv(OUT / "table3_crisis_peaks.csv")

# ---- copulas & VaR (1-month)
cop = {}
for c in ["WB_TOTAL", "WB_ENERGY", "WB_GOLD"]:
    d = r[["MSCI_ACWI", c]].dropna(); cop[c] = copulas.fit_all(d.iloc[:, 0].values, d.iloc[:, 1].values)
pd.concat(cop).to_csv(OUT / "copula_fits.csv")
var = {}
for c in ["WB_TOTAL", "WB_ENERGY"]:
    for w in (0.10, 0.125, 0.15):
        var[f"{c}_w{w}"] = copulas.copula_var(r["MSCI_ACWI"], r[c], w_cmd=w, horizon=1)
var = pd.DataFrame(var).T; var.to_csv(OUT / "var_gaussian_vs_clayton_1m.csv")
th = copulas.rolling_clayton(r["MSCI_ACWI"], r["WB_TOTAL"], 36, 1); th.to_csv(OUT / "rolling_clayton_theta.csv")

# ---- forecasting features (FRED monthly file v2, received 2026-11-02)
f2 = pd.read_csv("data/collected_fred_monthly_v2.csv", parse_dates=["date"], index_col="date")
f2.index = f2.index.to_period("M").to_timestamp("M")
cpi = f2["CPIAUCSL"].interpolate(limit=1, limit_area="inside")   # Oct-2025 not published (shutdown): linear interpolation, flagged
feats = pd.DataFrame({
    "VIX": f2["VIXCLS"],
    "OVX": f2["OVXCLS"],                 # from 2007-06: left missing before (no daily oil data for a realised-vol proxy)
    "GVZ": f2["GVZCLS"],                 # from 2008-06: idem
    "CREDIT_SPREAD_BAA10Y": f2["BAA10Y"],  # full sample; replaces HY OAS (ICE data on FRED only from 2023-11)
    "CREDIT_SPREAD_AAA10Y": f2["AAA10Y"],  # full sample; replaces IG OAS
    "TED_SPREAD": f2["TEDRATE"],         # discontinued after 2021-12: forward-filled is NOT applied beyond the last value
    "TERM_SPREAD_10Y2Y": f2["T10Y2Y"],
    "REAL_RATE": f2["DFII10"].combine_first(f2["REAINTRATREARAT10Y"]),  # TIPS from 2003, Cleveland Fed model before
    "CPI_YOY": 100 * np.log(cpi).diff(12),
    "INDPRO_YOY": 100 * np.log(f2["INDPRO"]).diff(12),
    "USREC": f2["USREC"]})
feats.loc[feats.index > f2["TEDRATE"].last_valid_index(), "TED_SPREAD"] = np.nan
feats.to_csv(OUT / "forecast_features_used.csv")
pers = D["alpha"] + D["beta"]
res = {}
for name, f in (("with_macro_features", feats),
                ("start_late_robustness", feats.drop(columns=["OVX", "GVZ", "TED_SPREAD"])),
                ("correlation_features_only", None)):
    bt = forecasting.backtest(rho, f, h=3, min_train=120, refit_every=12, dcc_persistence=pers, n_factors=2)
    bt["forecasts"].to_csv(OUT / f"oos_forecasts_{name}.csv"); bt["metrics"].to_csv(OUT / f"table4_oos_metrics_{name}.csv")
    res[name] = bt
log.append(f"Forecast OOS period: {res['with_macro_features']['forecasts'].index[0].date()} to "
           f"{res['with_macro_features']['forecasts'].index[-1].date()}")
log.append(f"Backends: GBRT={res['with_macro_features']['gbrt_backend']}, LSTM={res['with_macro_features']['lstm_backend']}")

summary = {"log": log, "era_means": era_tab.round(3).to_dict(), "transition": r2["transition"].round(3).values.tolist(),
           "durations": r2["expected_duration"].round(1).tolist(), "state_means_2": r2["state_means"].round(3).tolist(),
           "state_means_4": r4["state_means"].round(3).tolist(), "crisis_peaks": peaks.round(3).to_dict(),
           "var": var[["VaR_gaussian", "VaR_clayton", "increase_pct"]].round(4).to_dict(),
           "table4": {k: v["metrics"].round(3).to_dict() for k, v in res.items()}}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
print("\n".join(log))
print("\nTABLE 2\n", t2.to_string())
print("\nERA MEANS\n", era_tab.round(3).to_string())
print("\nHMM2\n", tr.round(3).to_string())
print("\nHMM4 state means", r4["state_means"].round(3).tolist())
print("\nCRISIS PEAKS\n", peaks.round(3).to_string())
print("\nCOPULA AIC (best)\n", pd.concat(cop)[["param", "aic", "best_by_aic"]].to_string())
print("\nVAR\n", var.round(4).to_string())
for k, v in res.items():
    print("\nTABLE4", k, "\n", v["metrics"].round(3).to_string())
print("\nCOUNTRIES top/bottom\n", ct.head(5).round(2).to_string(), "\n", ct.tail(5).round(2).to_string())
