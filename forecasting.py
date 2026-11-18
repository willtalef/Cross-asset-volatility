"""One-quarter-ahead correlation forecasting (paper Sections 3.5 and 6).

Target: monthly-average DCC correlation h=3 months ahead.
Models: random walk, rolling 12-month average, DCC mean reversion,
FAVAR (PCA factors + VAR(1)), gradient-boosted trees (XGBoost if installed,
else scikit-learn HistGradientBoosting), LSTM (PyTorch if installed, else a
scikit-learn MLP on the same lag window -- reported as such), and an
equal-weight ensemble.
Evaluation: expanding window, minimum 10 years of training data; no
observation whose target date is after the forecast origin is ever used.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


# ------------------------------------------------------------- features
def build_features(rho: pd.Series, exog: pd.DataFrame | None = None, h: int = 3) -> pd.DataFrame:
    """Lagged correlation (AR1-AR4), momentum (3m, 6m), mean reversion (vs 60m
    average) + optional exogenous monthly features. Target column 'y' = rho_{t+h}."""
    f = pd.DataFrame(index=rho.index)
    for k in range(4):
        f[f"ar{k + 1}"] = rho.shift(k)
    f["mom3"] = rho - rho.shift(3)
    f["mom6"] = rho - rho.shift(6)
    f["mr60"] = rho - rho.rolling(60, min_periods=24).mean()
    if exog is not None:
        f = f.join(exog.reindex(f.index).ffill())
    f["y"] = rho.shift(-h)
    return f


# ------------------------------------------------------------- models
def _gbrt():
    if HAS_XGB:
        return xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
                                subsample=0.8, colsample_bytree=0.8, random_state=0)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0)


class LSTMReg:
    """Two stacked LSTM layers (64, 32), dropout 0.20, Adam, early stopping
    (paper Section 3.5.1). Falls back to an MLP when PyTorch is unavailable."""

    def __init__(self, window: int = 12, epochs: int = 200, seed: int = 0):
        self.window, self.epochs, self.seed = window, epochs, seed
        self.backend = "pytorch-lstm" if HAS_TORCH else "sklearn-mlp-fallback"

    def _seq(self, X):
        w = self.window
        return np.stack([X[i - w + 1:i + 1] for i in range(w - 1, len(X))])

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)              # ignores NaN when fitting
        Xs = np.nan_to_num(self.sc.transform(X))      # missing -> 0 = feature mean
        self.ym, self.ys = float(np.mean(y)), float(np.std(y) + 1e-9)
        S, Y = self._seq(Xs), (y[self.window - 1:] - self.ym) / self.ys
        if HAS_TORCH:
            torch.manual_seed(self.seed)
            nf = S.shape[2]

            class Net(torch.nn.Module):
                def __init__(s):
                    super().__init__()
                    s.l1 = torch.nn.LSTM(nf, 64, batch_first=True)
                    s.l2 = torch.nn.LSTM(64, 32, batch_first=True)
                    s.dr = torch.nn.Dropout(0.2)
                    s.out = torch.nn.Linear(32, 1)

                def forward(s, x):
                    o, _ = s.l1(x); o = s.dr(o)
                    o, _ = s.l2(o); o = s.dr(o[:, -1])
                    return s.out(o).squeeze(-1)

            self.net = Net()
            n_val = max(1, int(0.15 * len(S)))
            Xt = torch.tensor(S[:-n_val], dtype=torch.float32); yt = torch.tensor(Y[:-n_val], dtype=torch.float32)
            Xv = torch.tensor(S[-n_val:], dtype=torch.float32); yv = torch.tensor(Y[-n_val:], dtype=torch.float32)
            opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
            best, best_state, patience = np.inf, None, 0
            for _ in range(self.epochs):
                self.net.train(); opt.zero_grad()
                loss = torch.mean((self.net(Xt) - yt) ** 2); loss.backward(); opt.step()
                self.net.eval()
                with torch.no_grad():
                    vl = float(torch.mean((self.net(Xv) - yv) ** 2))
                if vl < best - 1e-6:
                    best, patience = vl, 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    patience += 1
                    if patience >= 20:
                        break
            self.net.load_state_dict(best_state)
        else:
            from sklearn.neural_network import MLPRegressor
            self.net = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                    max_iter=2000, random_state=self.seed)
            self.net.fit(S.reshape(len(S), -1), Y)
        return self

    def predict_last(self, X_hist):
        """Predict from the last `window` rows of X_hist."""
        s = np.nan_to_num(self.sc.transform(X_hist[-self.window:]))[None]
        if HAS_TORCH:
            self.net.eval()
            with torch.no_grad():
                return self.ym + self.ys * float(self.net(torch.tensor(s, dtype=torch.float32))[0])
        return self.ym + self.ys * float(self.net.predict(s.reshape(1, -1))[0])


def favar_forecast(factors: np.ndarray, rho: np.ndarray, h: int = 3) -> float:
    """VAR(1) on [F_t, rho_t] estimated by OLS (eq. 14), iterated h steps."""
    Y = np.column_stack([factors, rho])
    X = np.column_stack([np.ones(len(Y) - 1), Y[:-1]])
    B = np.linalg.lstsq(X, Y[1:], rcond=None)[0]
    y = Y[-1]
    for _ in range(h):
        y = np.r_[1.0, y] @ B
    return float(y[-1])


# ------------------------------------------------------------- backtest
def backtest(rho: pd.Series, exog: pd.DataFrame | None = None, h: int = 3,
             min_train: int = 120, refit_every: int = 12, dcc_persistence: float | None = None,
             n_factors: int = 5, ensemble=("DCC", "FAVAR", "GBRT", "LSTM")) -> dict:
    """Expanding-window out-of-sample backtest. Returns forecasts and metrics."""
    rho = rho.dropna()
    F = build_features(rho, exog, h)
    feat_cols = [c for c in F.columns if c != "y"]
    exog_cols = [c for c in (exog.columns if exog is not None else [])]
    F[feat_cols] = F[feat_cols].ffill(limit=2)   # fill short gaps only (e.g. one unpublished month)
    core = [c for c in feat_cols if c not in exog_cols]
    valid = F.dropna(subset=core).index          # exogenous features may be missing early
    preds, gbrt, lstm, lstm_backend, use = [], None, None, None, feat_cols
    persistence = dcc_persistence if dcc_persistence is not None else _ar1(rho)
    for i, t in enumerate(valid):
        pos = F.index.get_loc(t)
        if pos + h >= len(F) or i < min_train:
            continue
        train = F.loc[valid[: i - h + 1]].dropna(subset=["y"])   # targets known at t
        # (missing exogenous values: handled natively by the tree model;
        #  zero-filled after standardisation for the LSTM, i.e. set to the mean)
        if len(train) < min_train - h:
            continue
        hist_rho = rho.loc[:t]
        row = {"date": t, "actual": F["y"].iloc[pos], "RW": hist_rho.iloc[-1],
               "ROLL12": hist_rho.iloc[-12:].mean(), "HISTMEAN": hist_rho.mean()}
        lr = hist_rho.mean()
        row["DCC"] = lr + persistence ** h * (hist_rho.iloc[-1] - lr)
        if exog_cols:
            E = F.loc[:t, exog_cols]
            E = E.loc[:, E.notna().mean() > 0.999].dropna()   # FAVAR: only fully observed features
            if E.shape[1] >= 1 and len(E) > 24:
                sc = StandardScaler().fit(E.values)
                k = min(n_factors, E.shape[1])
                fac = PCA(k).fit_transform(sc.transform(E.values))
                row["FAVAR"] = favar_forecast(fac, hist_rho.reindex(E.index).values, h)
            else:
                row["FAVAR"] = np.nan
        else:
            row["FAVAR"] = np.nan
        if gbrt is None or i % refit_every == 0:
            use = [c for c in feat_cols if train[c].notna().sum() >= 24]   # features observed long enough
            gbrt = _gbrt().fit(train[use].values, train["y"].values)
            lstm = LSTMReg().fit(train[use].values, train["y"].values)
            lstm_backend = lstm.backend
        row["GBRT"] = float(gbrt.predict(F.loc[[t], use].values)[0])
        row["LSTM"] = lstm.predict_last(F.loc[:t, use].values)
        row["ENSEMBLE"] = np.nanmean([row[m] for m in ensemble])
        preds.append(row)
    P = pd.DataFrame(preds).set_index("date")
    return {"forecasts": P, "metrics": evaluate(P), "lstm_backend": lstm_backend,
            "gbrt_backend": "xgboost" if HAS_XGB else "sklearn-histgb-fallback",
            "feature_names": feat_cols, "last_gbrt": gbrt}


def _ar1(s: pd.Series) -> float:
    x = s.dropna().values
    return float(np.clip(np.corrcoef(x[1:], x[:-1])[0, 1], 0, 0.999))


def evaluate(P: pd.DataFrame, high: float = 0.40, crisis_mask: pd.Series | None = None) -> pd.DataFrame:
    """RMSE, MAE, out-of-sample R^2 (vs expanding historical mean, Campbell-
    Thompson), RMSE improvement vs random walk (1 - RMSE/RMSE_RW; this matches
    the 'vs. RW (%)' column of the paper's Table 4), and regime accuracy."""
    a = P["actual"]
    rw = np.sqrt(np.mean((P["RW"] - a) ** 2))
    sse_hm = np.sum((P["HISTMEAN"] - a) ** 2)
    rows = {}
    for m in ["RW", "ROLL12", "DCC", "FAVAR", "GBRT", "LSTM", "ENSEMBLE"]:
        if m not in P or P[m].isna().all():
            continue
        e = P[m] - a
        rmse = np.sqrt(np.mean(e ** 2))
        rows[m] = {"RMSE": rmse, "MAE": np.mean(np.abs(e)),
                   "R2_oos": 1 - np.sum(e ** 2) / sse_hm,
                   "vs_RW_pct": 100 * (1 - rmse / rw),
                   "regime_acc": np.mean((P[m] > high) == (a > high))}
        if crisis_mask is not None:
            cm = crisis_mask.reindex(P.index).fillna(False).astype(bool)
            rows[m]["crisis_acc"] = np.mean((P.loc[cm, m] > high) == (a[cm] > high))
    return pd.DataFrame(rows).T


def feature_importance(model, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Mean |SHAP| if `shap` is installed (paper eq. 13), else permutation importance."""
    try:
        import shap
        sv = shap.TreeExplainer(model).shap_values(X)
        return pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
    except Exception:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, X, y, n_repeats=10, random_state=0)
        return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
