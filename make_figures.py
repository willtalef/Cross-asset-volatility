"""Draw the figures of the paper into the folder figures/ (run after the analysis scripts)."""
import os as _os, sys as _sys
try:
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
except NameError:
    _HERE = _os.getcwd()
_os.chdir(_HERE); _sys.path.insert(0, _HERE)
_os.makedirs("figures", exist_ok=True)
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=""
plt.rcParams.update({"font.family":"serif","font.size":9})
d=pd.read_csv(R+"results_daily_us/dcc_daily.csv",parse_dates=["date"],index_col="date")
cr={"Gulf":("1990-08-01","1991-03-31"),"LTCM":("1997-07-01","1998-12-31"),"Dot-com":("2000-03-01","2002-10-31"),"GFC":("2008-09-01","2009-06-30"),"Euro":("2011-07-01","2012-07-31"),"Oil":("2014-07-01","2016-02-29"),"COVID":("2020-02-01","2020-06-30"),"Infl.":("2021-06-01","2023-06-30"),"Rates":("2023-07-01","2024-12-31")}
fig,ax=plt.subplots(figsize=(6.5,2.8))
for k,(a,b) in cr.items(): ax.axvspan(pd.Timestamp(a),pd.Timestamp(b),color="0.88",lw=0)
ax.plot(d.index,d["SP500TR|GSCI"],lw=0.5,color="#1f3b73",label="S&P 500 TR – S&P GSCI")
ax.plot(d.index,d["SP500TR|BRENT"],lw=0.5,color="#c0504d",alpha=0.75,label="S&P 500 TR – Brent")
ax.axvline(pd.Timestamp("2004-01-01"),ls="--",lw=0.8,color="k"); ax.text(pd.Timestamp("2004-03-01"),-0.25,"2004",fontsize=7)
ax.axhline(0,lw=0.5,color="k"); ax.set_ylabel("DCC correlation"); ax.legend(loc="upper left",fontsize=7,frameon=False)
ax.set_xlim(d.index[0],d.index[-1]); fig.tight_layout(); fig.savefig("figures/fig_dcc_daily.pdf")
c=pd.read_csv(R+"results_collected/country_commodity_corr.csv").sort_values("corr_broad")
fig,ax=plt.subplots(figsize=(6.5,5.6))
ax.barh(c["country"],c["corr_broad"],color="#1f3b73",height=0.7,label="Broad commodity index")
ax.plot(c["corr_energy"],c["country"],"o",ms=2.5,color="#c0504d",label="Energy index")
ax.axvline(c["corr_broad"].median(),ls="--",lw=0.8,color="k"); ax.tick_params(axis="y",labelsize=6.5)
ax.set_xlabel("Correlation of monthly MSCI USD returns with World Bank commodity index returns"); ax.legend(fontsize=7,frameon=False,loc="lower right")
fig.tight_layout(); fig.savefig("figures/fig_countries.pdf")
f=pd.read_csv(R+"results_daily_us/oos_forecasts_main.csv",parse_dates=["date"],index_col="date")
fig,ax=plt.subplots(figsize=(6.5,2.6))
ax.plot(f.index,f["actual"],color="k",lw=1,label="Realised (t+3)")
ax.plot(f.index,f["DCC"],color="#1f3b73",lw=0.8,label="DCC mean reversion")
ax.plot(f.index,f["ENSEMBLE"],color="#c0504d",lw=0.8,ls="--",label="ML ensemble")
ax.set_ylabel("S&P 500 TR – GSCI corr."); ax.legend(fontsize=7,frameon=False,ncol=3,loc="upper left"); fig.tight_layout(); fig.savefig("figures/fig_oos.pdf")

import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=""; X="results_extra/"
plt.rcParams.update({"font.family":"serif","font.size":9})
d=pd.read_csv(R+"results_daily_us/dcc_daily_2000plus.csv",parse_dates=["date"],index_col="date")
fig,ax=plt.subplots(figsize=(6.5,2.7))
for c,col in zip(d.columns,["#b8860b","#c0504d","#4f8a3c"]): ax.plot(d.index,d[c],lw=0.5,color=col,label=c.replace("SP500TR|","S&P 500 TR – ").title().replace("S&P 500 Tr","S&P 500 TR"))
ax.axhline(0,lw=.5,color="k");ax.set_ylabel("DCC correlation");ax.legend(fontsize=7,frameon=False,ncol=3,loc="upper left");ax.set_xlim(d.index[0],d.index[-1]);fig.tight_layout();fig.savefig("figures/fig_dcc_2000.pdf")
dc=pd.read_csv(R+"results_daily_us/dcc_daily.csv",parse_dates=["date"],index_col="date").resample("ME").mean()["SP500TR|GSCI"]
s2=pd.read_csv(X+"hmm2_path_us.csv",parse_dates=[0],index_col=0).iloc[:,0]
s4=pd.read_csv(X+"hmm4_path_us.csv",parse_dates=[0],index_col=0).iloc[:,0]
fig,ax=plt.subplots(2,1,figsize=(6.5,4.2),sharex=True)
hi=s2==s2.groupby(s2).apply(lambda z: dc.loc[z.index].mean()).idxmax()
for t,v in hi.items():
    if v: ax[0].axvspan(t-pd.offsets.MonthEnd(1),t,color="#f2c4c4",lw=0)
ax[0].plot(dc.index,dc,color="#1f3b73",lw=.9);ax[0].set_ylabel("Monthly avg. DCC");ax[0].set_title("Two-state HMM: shaded = high-correlation state",fontsize=8)
means=dc.groupby(s4).mean().sort_values(); rank={k:i for i,k in enumerate(means.index)}
ax[1].step(s4.index,[rank[v] for v in s4],where="post",color="k",lw=.8);ax[1].set_yticks(range(4));ax[1].set_yticklabels(["Negative","Low","Moderate","High"]);ax[1].set_title("Four-state HMM: Viterbi path",fontsize=8)
fig.tight_layout();fig.savefig("figures/fig_hmm.pdf")
r=pd.read_csv(X+"rolling60_tau_us.csv",index_col=0,parse_dates=True)
fig,ax=plt.subplots(figsize=(6.5,2.4));ax.plot(r.index,r["tau"],color="#1f3b73",lw=1,label="Kendall's τ (60-month)");ax2=ax.twinx();ax2.plot(r.index,r["theta"],color="#c0504d",lw=.8,ls="--",label="Implied Clayton θ")
ax.axhline(0,lw=.5,color="k");ax.set_ylabel("τ");ax2.set_ylabel("θ");h1,l1=ax.get_legend_handles_labels();h2,l2=ax2.get_legend_handles_labels();ax.legend(h1+h2,l1+l2,fontsize=7,frameon=False,loc="upper left");fig.tight_layout();fig.savefig("figures/fig_rolling_tau.pdf")
g=pd.read_csv(R+"results_collected/dcc_monthly_correlations.csv",parse_dates=["date"],index_col="date")
fig,ax=plt.subplots(figsize=(6.5,2.6))
for c,col in zip(g.columns,["#1f3b73","#c0504d","#b8860b","#7f7f7f","#4f8a3c"]): ax.plot(g.index,g[c],lw=.9,color=col,label=c.replace("MSCI_ACWI|WB_","").replace("_"," ").title())
ax.set_ylabel("DCC correlation with MSCI ACWI");ax.legend(fontsize=7,frameon=False,ncol=5,loc="upper left");ax.set_ylim(None,0.5);fig.tight_layout();fig.savefig("figures/fig_global_dcc.pdf")
f=pd.read_csv(R+"results_daily_us/oos_forecasts_main.csv",parse_dates=["date"],index_col="date")
fig,ax=plt.subplots(figsize=(6.5,2.6))
for m,col,ls in [("DCC","#1f3b73","-"),("FAVAR","#4f8a3c","-"),("GBRT","#b8860b","--"),("ENSEMBLE","#c0504d","--")]:
    ax.plot(f.index,((f["RW"]-f["actual"])**2-(f[m]-f["actual"])**2).cumsum(),color=col,ls=ls,lw=1,label=m.title() if m!="DCC" else "DCC")
ax.axhline(0,lw=.5,color="k");ax.set_ylabel("Cum. SE(RW) − SE(model)");ax.legend(fontsize=7,frameon=False,ncol=4,loc="lower left");fig.tight_layout();fig.savefig("figures/fig_cssed.pdf")
imp=pd.read_csv(X+"feature_importance_gbrt_us.csv",index_col=0).iloc[:,0].sort_values()
fig,ax=plt.subplots(figsize=(5.5,3.2));ax.barh(imp.index,imp.values,color="#1f3b73");ax.set_xlabel("Permutation importance (drop in R²)");fig.tight_layout();fig.savefig("figures/fig_importance.pdf")
