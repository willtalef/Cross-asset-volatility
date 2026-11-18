"""Additional diagnostics for the November 2026 revision (descriptives, breaks, DM tests, subperiods, importance)."""
import os as _os, sys as _sys
try:  # always run from the project folder, whatever the current directory (Spyder, PyCharm, Jupyter %run)
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
except NameError:
    _HERE = _os.getcwd()
_os.chdir(_HERE); _sys.path.insert(0, _HERE)
from cross_asset import MONTH_END
import json, numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
from cross_asset import forecasting, regimes, copulas
OUT=Path("results_extra"); OUT.mkdir(exist_ok=True)
px=pd.read_csv("results_daily_us/prices_used.csv",parse_dates=["date"],index_col="date")
def desc(r,ann):
    return pd.DataFrame({"N":r.count(),"Mean (ann.,%)":100*ann*r.mean(),"Vol (ann.,%)":100*np.sqrt(ann)*r.std(),
        "Skew":r.skew(),"Ex. kurt":r.kurt(),"Min (%)":100*r.min(),"Max (%)":100*r.max(),
        "JB":[stats.jarque_bera(r[c].dropna())[0] for c in r]})
rd=np.log(px).diff()
# daily: per-series availability (no common-sample restriction)
d1=desc(rd,252); d1.to_csv(OUT/"desc_us_daily.csv")
msci=pd.read_csv("data/collected_msci_monthly_eom.csv",parse_dates=["date"],index_col="date")
wb=pd.read_csv("data/collected_worldbank_monthly_avg.csv",parse_dates=["date"],index_col="date")
g=np.log(pd.concat([msci[["MSCI_ACWI","MSCI_World","MSCI_EM","MSCI_USA"]],wb[["WB_TOTAL","WB_ENERGY","WB_PRECIOUS","WB_BASE_METALS","WB_AGRI","WB_GOLD"]]],axis=1)).diff().loc["1999-01-01":"2026-03-31"]
d2=desc(g,12); d2.to_csv(OUT/"desc_global_monthly.csv")
# ---- structural break in mean of monthly S&P-GSCI DCC (Andrews sup-Wald, HAC)
dcc=pd.read_csv("results_daily_us/dcc_daily.csv",parse_dates=["date"],index_col="date")
cm=dcc.resample(MONTH_END).mean()
def nw_var(u,L):
    u=u-u.mean(); T=len(u); s=u@u/T
    for l in range(1,L+1): s+=2*(1-l/(L+1))*(u[l:]@u[:-l])/T
    return s
def wald_break(y,k,L):
    y=np.asarray(y); T=len(y); m1,m2=y[:k].mean(),y[k:].mean()
    u=np.r_[y[:k]-m1,y[k:]-m2]; s2=nw_var(u,L)
    return (m2-m1)**2/(s2*(1/k+1/(T-k))),m1,m2
res={}
for pair in ["SP500TR|GSCI","SP500TR|BRENT"]:
    y=cm[pair].dropna(); T=len(y); L=int(4*(T/100)**(2/9))
    ks=range(int(0.15*T),int(0.85*T)); W=[wald_break(y.values,k,L)[0] for k in ks]
    kstar=list(ks)[int(np.argmax(W))]; w,m1,m2=wald_break(y.values,kstar,L)
    k04=int((y.index<"2004-01-01").sum()); w04,a1,a2=wald_break(y.values,k04,L)
    res[pair]={"T":T,"supW":float(max(W)),"break_date":str(y.index[kstar].date()),"mean_before":float(m1),"mean_after":float(m2),
               "chow2004_W":float(w04),"chow2004_p":float(1-stats.chi2.cdf(w04,1)),"mean_pre2004":float(a1),"mean_post2004":float(a2),"NW_lags":L}
json.dump(res,open(OUT/"breaks.json","w",encoding="utf-8"),indent=2)
# ---- Diebold-Mariano vs RW, subperiod RMSE
f=pd.read_csv("results_daily_us/oos_forecasts_main.csv",parse_dates=["date"],index_col="date")
dm={}
for m in ["ROLL12","DCC","FAVAR","GBRT","LSTM","ENSEMBLE"]:
    d=(f["RW"]-f["actual"])**2-(f[m]-f["actual"])**2; d=d.dropna().values; T=len(d)
    s=nw_var(d,2); stat=d.mean()/np.sqrt(s/T)
    # Harvey-Leybourne-Newbold small-sample correction, h=3
    h=3; stat*=np.sqrt((T+1-2*h+h*(h-1)/T)/T)
    dm[m]={"DM":float(stat),"p_two_sided":float(2*(1-stats.t.cdf(abs(stat),T-1)))}
pd.DataFrame(dm).T.to_csv(OUT/"dm_tests_us.csv")
per={"2001-2007":("2001-01-01","2007-12-31"),"2008-2012":("2008-01-01","2012-12-31"),"2013-2019":("2013-01-01","2019-12-31"),"2020-2025":("2020-01-01","2025-12-31")}
sub=pd.DataFrame({k:{m:float(np.sqrt(((f.loc[a:b,m]-f.loc[a:b,"actual"])**2).mean())) for m in ["RW","ROLL12","DCC","FAVAR","GBRT","LSTM","ENSEMBLE"]} for k,(a,b) in per.items()})
sub.loc["N"]=[len(f.loc[a:b]) for a,b in per.values()]
sub.to_csv(OUT/"subperiod_rmse_us.csv")
fg=pd.read_csv("results_collected/oos_forecasts_with_macro_features.csv",parse_dates=["date"],index_col="date") if Path("results_collected/oos_forecasts_with_macro_features.csv").exists() else None
if fg is not None:
    dmg={}
    for m in ["DCC","FAVAR","GBRT","LSTM","ENSEMBLE"]:
        d=((fg["RW"]-fg["actual"])**2-(fg[m]-fg["actual"])**2).dropna().values;T=len(d);s=nw_var(d,2);st=d.mean()/np.sqrt(s/T)*np.sqrt((T+1-6+6/T)/T)
        dmg[m]={"DM":float(st),"p_two_sided":float(2*(1-stats.t.cdf(abs(st),T-1)))}
    pd.DataFrame(dmg).T.to_csv(OUT/"dm_tests_global.csv")
    open(OUT/"global_oos_span.txt","w",encoding="utf-8").write(f"{fg.index[0].date()} {fg.index[-1].date()} {len(fg)}")
# ---- HMM4 state table (US)
rho=cm["SP500TR|GSCI"].dropna()
h4=regimes.regime_summary(rho,4)
st=pd.concat([h4["state_means"].rename("mean"),h4["expected_duration"].rename("dur"),h4["stationary"].rename("stat")],axis=1)
st["share"]=pd.Series(h4["states"]).value_counts(normalize=True).sort_index().values if hasattr(h4["states"],"__len__") else np.nan
st.to_csv(OUT/"hmm4_states_us.csv"); h4["transition"].to_csv(OUT/"hmm4_transition_us.csv")
pd.Series(np.asarray(h4["states"]),index=rho.index).to_csv(OUT/"hmm4_path_us.csv")
h2=regimes.regime_summary(rho,2); pd.Series(np.asarray(h2["states"]),index=rho.index).to_csv(OUT/"hmm2_path_us.csv")
# ---- rolling 60m Clayton on monthly US returns (S&P vs GSCI)
rm=np.log(px.resample(MONTH_END).last()).diff().dropna(subset=["SP500TR","GSCI"])
rc={}
for i in range(60,len(rm)+1):
    w=rm.iloc[i-60:i]; tau=stats.kendalltau(w["SP500TR"],w["GSCI"])[0]
    rc[w.index[-1]]={"tau":tau,"theta":2*tau/(1-tau) if tau>0 else np.nan}
pd.DataFrame(rc).T.to_csv(OUT/"rolling60_tau_us.csv")
# ---- feature importance: GBRT on full sample (descriptive, in-sample permutation)
f2=pd.read_csv("data/collected_fred_monthly_v2.csv",parse_dates=["date"],index_col="date"); f2.index=f2.index.to_period("M").to_timestamp("M")
cpi=f2["CPIAUCSL"].interpolate(limit=1,limit_area="inside")
feats=pd.DataFrame({"VIX":f2["VIXCLS"],"OVX":f2["OVXCLS"],"GVZ":f2["GVZCLS"],"BAA10Y":f2["BAA10Y"],"AAA10Y":f2["AAA10Y"],"TED":f2["TEDRATE"],"T10Y2Y":f2["T10Y2Y"],
  "REAL_RATE":f2["DFII10"].combine_first(f2["REAINTRATREARAT10Y"]),"CPI_YOY":100*np.log(cpi).diff(12),"INDPRO_YOY":100*np.log(f2["INDPRO"]).diff(12),"USREC":f2["USREC"]})
F=forecasting.build_features(rho,feats.drop(columns=["OVX","GVZ","TED"]),3).dropna()
from sklearn.ensemble import HistGradientBoostingRegressor
X=F.drop(columns="y"); yv=F["y"]
mdl=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=3,random_state=0).fit(X,yv)
imp=forecasting.feature_importance(mdl,X,yv); imp.to_csv(OUT/"feature_importance_gbrt_us.csv")
# simple correlations of predictors with future correlation
pd.Series({c:X[c].corr(yv) for c in X}).to_csv(OUT/"predictor_corr_with_target.csv")
print(json.dumps(res,indent=1)); print(pd.DataFrame(dm).T.round(3)); print(sub.round(3)); print(st.round(3)); print(imp.round(4).head(12))
print(pd.read_csv(OUT/"dm_tests_global.csv",index_col=0).round(3)); print(open(OUT/"global_oos_span.txt").read())
print(d1.round(2).to_string()); print(d2.round(2).to_string())
