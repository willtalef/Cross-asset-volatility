# Data appendix – features that begin after January 1990
*(Revision prepared November 2026. To be added to the paper as a dated
revision; it describes how the replication code treats these series. It does
not describe the original May 2026 computation.)*

Several forecasting features used in Section 6.1 are not available for the
full 1990–2026 sample: the ICE BofA US High Yield and Investment Grade
option-adjusted spreads begin in December 1996, the 10-year TIPS real yield in
2003, the China NBS manufacturing PMI in 2005, the Citi Economic Surprise
Index in 2003, and the CBOE crude-oil and gold volatility indices (OVX, GVZ)
in 2007–2008.

**Main specification (spliced proxies).** Before each series begins, it is
replaced by a long-history proxy, re-scaled by an ordinary least-squares
regression of the series on the proxy over their overlapping period:

| Feature | Proxy before the series begins | Proxy source |
|---|---|---|
| High-yield spread | Moody's Baa yield minus 10-year Treasury | FRED BAA10Y (from 1986) |
| Investment-grade spread | Moody's Aaa yield minus 10-year Treasury | FRED AAA10Y (from 1983) |
| Real interest rate | Cleveland Fed 10-year real interest rate | FRED REAINTRATREARAT10Y (from 1982) |
| OVX | 21-day realised volatility of WTI crude, annualised | EIA daily WTI (from 1986) |
| GVZ | 21-day realised volatility of gold, annualised | LBMA daily gold |

A 0/1 indicator records the months that use a proxy. The regression
coefficients, overlap correlation and overlap length are reported in
`results/feature_treatment_proxy.json`. No reliable monthly proxy exists for
the China PMI or the Citi Economic Surprise Index before they begin; they are
treated as missing before that date.

**Handling of missing values in the models.** Gradient-boosted trees handle
missing values natively. The LSTM sets a missing standardised feature to zero
(its training mean). The FAVAR extracts its factors only from the features
that are fully observed over each estimation window.

**Robustness checks.** Results are also computed (i) leaving all
late-starting features missing before they begin, without proxies, and
(ii) excluding the late-starting features entirely. All three sets of results
are reported in `results/robustness_feature_treatment.csv`.

These choices were fixed before the forecasting results were computed.
Whatever the results, the figures reported in the paper, the petition and the
CV must be those produced by the main specification on the documented data.

**Scope.** This treatment affects only the forecasting results of Section 6
(Table 4). The correlation regimes (Section 4.2), the crisis peaks
(Section 4.3) and the copula-based VaR comparison (Section 7.3) use only the
asset-price series, which are available from 1990, and are not affected.
