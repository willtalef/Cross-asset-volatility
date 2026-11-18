# Data provenance (to be completed by the author)

For each series used in the paper, record where it came from and how it was
accessed. Only list sources that were actually used.

| Series (column name) | Description | Source / vendor | Ticker or code | Access method (e.g. school Bloomberg terminal, FRED public download, CFTC website) | Frequency | Date range | Date retrieved |
|---|---|---|---|---|---|---|---|
| EQUITY | e.g. MSCI ACWI total return, USD | | | | daily | | |
| COMMODITY | e.g. broad commodity index, USD | | | | daily | | |
| GOLD | e.g. LBMA gold fix | | | | daily | | |
| VIX | CBOE VIX | | | | monthly | | |
| HY_SPREAD | ICE BofA US HY OAS | | | | monthly | | |
| HY_SPREAD proxy | Moody's Baa – 10Y Treasury | FRED | BAA10Y | public download | monthly | 1990–1996 | |
| IG_SPREAD proxy | Moody's Aaa – 10Y Treasury | FRED | AAA10Y | public download | monthly | 1990–1996 | |
| REAL_RATE proxy | Cleveland Fed 10Y real rate | FRED | REAINTRATREARAT10Y | public download | monthly | 1990–2002 | |
| OVX proxy | WTI daily spot (for realised vol) | EIA / FRED | DCOILWTICO | public download | daily | 1990–2007 | |
| GVZ proxy | Gold daily price (for realised vol) | LBMA | | | daily | 1990–2008 | |
| ... | | | | | | | |

Notes on transformations (USD conversion, total-return adjustment, Geltner
unsmoothing of private equity, survivorship treatment of hedge fund indices):

-


## Data received on 2026-11-02 and known limitations

* `data/collected_msci_monthly_eom.csv` – msci.com, gross USD, month-end, 1998-12 → 2026-03.
* `data/collected_worldbank_monthly_avg.csv` – World Bank Pink Sheet (update of 2 Oct 2026), monthly averages, 1990-01 → 2026-03.
* `data/collected_fred_monthly_v2.csv` – FRED monthly, 1990-01 → 2026-03.

Limitations (reported in the results workbook, sheet "Limites des données"):
1. ICE high-yield and investment-grade spreads and the HY total-return index are available on FRED only for 29 months (2023-11 → 2026-03) because of ICE's restriction on FRED; BAA10Y and AAA10Y cover all 435 months and are used instead.
2. CPI for October 2025 is missing: it was not published during the US government shutdown (interpolated for the year-on-year growth rate; flagged).
3. An empty cell means no observation (holiday or series not yet started), never zero.
