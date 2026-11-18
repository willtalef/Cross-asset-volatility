"""Run the full pipeline and write every table to results/.

Usage:
    python run_all.py                 # uses config.json and the CSVs it points to
    python run_all.py --synthetic     # smoke test on synthetic data (NOT paper results)
"""
from __future__ import annotations
import os as _os, sys as _sys
try:  # always run from the project folder, whatever the current directory (Spyder, PyCharm, Jupyter %run)
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
except NameError:
    _HERE = _os.getcwd()
_os.chdir(_HERE); _sys.path.insert(0, _HERE)
from cross_asset import MONTH_END
import argparse, json, platform, sys, datetime
from pathlib import Path
import numpy as np
import pandas as pd

from cross_asset import data, dcc, regimes, copulas, forecasting, stats_utils, features
from cross_asset.synthetic import make_synthetic

PAPER = {  # figures reported in the paper, for side-by-side comparison
    "era_means": [0.08, 0.38, 0.31, 0.42],
    "transition": [[0.94, 0.06], [0.08, 0.92]],
    "var_increase_pct": "8-14",
    "ensemble_vs_rw_pct": 42, "ensemble_r2_oos": 0.74,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--missing", default="proxy", choices=["proxy", "native", "start_late"],
                    help="treatment of features starting after 1990 (main spec: proxy)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out); out.mkdir(exist_ok=True)
    log = []

    if a.synthetic:
        prices, feats, proxies = make_synthetic()
        log.append("DATA: SYNTHETIC (smoke test only - numbers are meaningless)")
    else:
        prices = data.load_prices(cfg["prices_csv"])
        feats = data.load_features(cfg["features_csv"]) if Path(cfg["features_csv"]).exists() else None
        pp = cfg.get("proxies_csv", "data/proxies_monthly.csv")
        proxies = data.load_features(pp) if Path(pp).exists() else None
        log.append(f"DATA: {cfg['prices_csv']} ({prices.index[0].date()} to {prices.index[-1].date()})")

    eq, cm = cfg["equity"], cfg["commodity"]
    cols = [eq, cm] + [c for c in cfg.get("extra_series", []) if c in prices]
    r_d = data.log_returns(prices[cols]).dropna()
    r_m = data.log_returns(prices[cols], MONTH_END).dropna()

    # 4.1 Unconditional correlations (monthly, Newey-West)
    stats_utils.corr_table(r_m, [eq], [c for c in cols if c != eq]).to_csv(out / "table2_unconditional_corr.csv")

    # 3.2 GARCH + DCC (daily)
    Z, gp = dcc.standardize(r_d)
    gp.to_csv(out / "garch_params.csv")
    d = dcc.fit_dcc(Z)
    d["corr"].to_csv(out / "dcc_daily_correlations.csv")
    log.append(f"DCC alpha={d['alpha']:.4f} beta={d['beta']:.4f}")
    pair = f"{eq}|{cm}"
    rho_m = d["corr"][pair].resample(MONTH_END).mean()

    # 4.2 Regimes
    regimes.period_means(rho_m, {k: tuple(v) for k, v in cfg["eras"].items()}).rename("mean_corr").rename_axis("era").to_csv(out / "era_mean_correlations.csv")
    r4 = regimes.regime_summary(rho_m, 4)
    r4["episodes"].to_csv(out / "hmm4_episodes.csv"); r4["state_means"].to_csv(out / "hmm4_state_means.csv")
    r2 = regimes.regime_summary(rho_m, 2)
    pd.concat([r2["transition"], r2["expected_duration"], r2["stationary"]], axis=1).to_csv(out / "hmm2_transition_durations.csv")
    # 4.3 Crisis table
    regimes.crisis_table(d["corr"][[pair]], {k: tuple(v) for k, v in cfg["crises"].items()}).to_csv(out / "table3_crisis_peaks.csv")

    # 3.3 / 7.3 Copulas and VaR
    copulas.fit_all(r_m[eq].values, r_m[cm].values).to_csv(out / "copula_fits_monthly.csv")
    th = copulas.rolling_clayton(r_m[eq], r_m[cm], 36, 1)
    th.to_csv(out / "rolling_clayton_theta.csv")
    var = copulas.copula_var(r_d[eq], r_d[cm], w_cmd=cfg["commodity_weight_for_var"])
    var.to_csv(out / "var_gaussian_vs_clayton.csv")
    log.append(f"VaR 1% 10d: Gaussian={var['VaR_gaussian']:.4f}  Clayton={var['VaR_clayton']:.4f}  increase={var['increase_pct']:.1f}%")

    # 6 Forecasting: main specification + the two robustness checks
    fc = cfg["forecast"]
    order = [a.missing] + [m for m in ("proxy", "native", "start_late") if m != a.missing]
    robust = {}
    for strat in order:
        f_s, rep = features.prepare(feats, proxies, strat) if feats is not None else (None, {})
        (out / f"feature_treatment_{strat}.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        bt = forecasting.backtest(rho_m, f_s, h=fc["horizon_months"], min_train=fc["min_train_months"],
                                  refit_every=fc["refit_every_months"], dcc_persistence=(d["alpha"] + d["beta"]) ** 21)
        tag = "" if strat == a.missing else f"_robust_{strat}"
        bt["forecasts"].to_csv(out / f"oos_forecasts{tag}.csv"); bt["metrics"].to_csv(out / f"table4_oos_metrics{tag}.csv")
        robust[strat] = bt["metrics"].loc["ENSEMBLE", ["R2_oos", "vs_RW_pct"]]
        if strat == a.missing:
            main_bt = bt
    bt = main_bt
    log.append(f"Main specification for late-starting features: {a.missing}")
    log.append(f"GBRT backend: {bt['gbrt_backend']} | LSTM backend: {bt['lstm_backend']}")
    pd.DataFrame(robust).T.to_csv(out / "robustness_feature_treatment.csv")

    # Environment record for reproducibility
    import sklearn, scipy
    env = {"run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "python": sys.version, "platform": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    lines = ["# Run summary", "", *[f"- {l}" for l in log], "",
             "## Era mean correlations (paper: 0.08 / 0.38 / 0.31 / 0.42)",
             pd.read_csv(out / "era_mean_correlations.csv").to_markdown(index=False) if _has_tabulate() else
             pd.read_csv(out / "era_mean_correlations.csv").to_string(index=False), "",
             "## Two-state transition matrix (paper: 0.94/0.06, 0.08/0.92)",
             r2["transition"].round(3).to_string(), "",
             "## Out-of-sample metrics (paper ensemble: R2_oos 0.74, +42% vs RW)",
             bt["metrics"].round(3).to_string(), "",
             "## Ensemble under the three feature treatments (report all of them)",
             pd.DataFrame(robust).T.round(3).to_string()]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def _has_tabulate():
    try:
        import tabulate  # noqa
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
