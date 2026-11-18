# Cross-Asset Correlation Dynamics – methodology implementation

**Quick start:** extract the zip, then run `python lancer_tout.py` from the project folder (Windows instructions in LISEZMOI_WINDOWS.md). It runs every analysis, the figures and the Excel summary.

Python implementation of the methods described in
*Cross-Asset Correlation Dynamics and Commodity Market Predictability*
(W. L. R. Talefo Taffo, SKEMA Business School Working Paper 14(2), May 2026).

## ⚠️ Status of this code – read first

* **This code was written in November 2026, after the paper, from the
  paper's text. It is not the original code used to produce the published
  results.** It must not be described as such.
* It implements the methods the paper describes, so that a third party can
  run them on documented data. Results will match the paper's figures only if
  the same data, the same sample and the same settings are used. Any
  difference between this code's output and the paper must be reported as
  such, not adjusted.
* It has only been tested end-to-end on **synthetic data**
  (`python run_all.py --synthetic`). It has not yet been run on the paper's
  data.

## What is implemented (paper section → module)

| Paper | Method | Module |
|---|---|---|
| §3.1 | Log returns in USD, monthly/daily; Geltner unsmoothing | `data.py` |
| §4.1, Table 2 | Unconditional correlations, Newey-West significance | `stats_utils.py` |
| §3.2, eq. 1–4 | GARCH(1,1) + scalar DCC(1,1), pairwise composite likelihood | `dcc.py` |
| §4.2, eq. 15 | 4-state and 2-state Gaussian HMM on the DCC series; transition matrix, expected durations 1/(1−pᵢᵢ), stationary shares; era means | `regimes.py` |
| §4.3, Table 3 | Peak correlation per crisis window, months to peak / normalisation | `regimes.py` |
| §3.3 | Gaussian, Student-t, Clayton, Gumbel copulas; AIC selection; rolling 36-month Clayton parameter | `copulas.py` |
| §7.3 | 1% 10-day VaR, Gaussian vs Clayton copula, 12.5% commodity weight | `copulas.py` |
| §3.5, §6, Table 4 | 3-month-ahead forecasts: random walk, rolling 12m mean, DCC mean reversion, FAVAR (PCA + VAR(1)), gradient-boosted trees, LSTM, ensemble; expanding window, ≥10-year training; RMSE, MAE, R²_oos, improvement vs RW, regime accuracy | `forecasting.py` |
| eq. 13 | SHAP feature importance (if `shap` installed; else permutation importance) | `forecasting.py` |

## Features that begin after 1990

See `DATA_APPENDIX.md`. Main specification: spliced long-history proxies
(`--missing proxy`, default). Robustness checks (`native`, `start_late`) are
run automatically and reported in `results/robustness_feature_treatment.csv`.
Put the proxy series in `data/proxies_monthly.csv` (template provided).

## Not implemented / simplified

* Wavelet coherence (§3.4) – not implemented.
* RS-DCC of Pelletier (2006) – approximated by a two-state HMM fitted on the
  DCC correlation series (two-step instead of joint estimation).
* Structural-break tests (Bai–Perron), scenario projections (§6.3),
  efficient-frontier analysis (§7.1) and the regime-based allocation backtest
  (§7.2) – not implemented.
* Copula margins use rank-based pseudo-observations (the paper mentions a
  kernel-smoothed empirical CDF).
* DCC h-step forecast uses the standard mean-reversion approximation.
* If `xgboost` / `torch` are not installed, the code falls back to
  scikit-learn `HistGradientBoostingRegressor` / an `MLPRegressor`, and the
  run summary states which backend was used.

## How to run

```bash
pip install -r requirements.txt
# smoke test (synthetic data, numbers are meaningless):
python run_all.py --synthetic
# real run: put the data in data/ following the *_TEMPLATE.csv layouts,
# complete DATA_PROVENANCE.md, adjust config.json, then:
python run_all.py
```

All tables are written to `results/` together with `environment.json`
(package versions) and `SUMMARY.md` (side-by-side with the paper's reported
figures).

## Data

The data are not included (licensed vendor data). `DATA_PROVENANCE.md` must
record, for each series, the vendor, ticker, access method and retrieval date.
