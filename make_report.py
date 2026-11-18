import os as _os, sys as _sys
try:  # always run from the project folder, whatever the current directory (Spyder, PyCharm, Jupyter %run)
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
except NameError:
    _HERE = _os.getcwd()
_os.chdir(_HERE); _sys.path.insert(0, _HERE)

import pandas as pd, sys
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
G="results_collected/"; U="results_daily_us/"; OUT=sys.argv[1] if len(sys.argv)>1 else "Resultats_CrossAsset.xlsx"
comp=pd.DataFrame([
 ["Données","Quotidien + mensuel, 1990–2026, 42 pays, 18 indices","Mensuel, 1999–2026 ; MSCI ACWI + 47 pays ; Banque mondiale (moyennes mensuelles)","Quotidien, 1990–2026 ; S&P 500 rendement total, S&P GSCI, Brent ; or, cuivre, maïs depuis 2000","—"],
 ["Corrélation actions–matières premières, ère 1 (1990–2003)","0,08","0,16 (1999–2003 seulement)","0,04","Retrouvé (US quotidien)"],
 ["Ère 2 (2004–2011)","0,38","0,17","0,20","Plus faible"],
 ["Ère 3 (2012–2019)","0,31","0,23","0,28","Proche (US quotidien)"],
 ["Ère 4 (2020–2026)","0,42","0,19","0,22","Plus faible"],
 ["Hausse après 2004 (financiarisation)","0,08 → plus de 0,35","Faible","0,04 → 0,20–0,28","Retrouvé dans le sens, d'ampleur moindre"],
 ["Pic actions–énergie, crise 2008","0,72","0,27 (DCC mensuel)","S&P–GSCI 0,54 ; S&P–Brent 0,40 (DCC quotidien)","Non retrouvé (même sens, plus faible)"],
 ["Pic actions–énergie, COVID 2020","0,78","0,37","S&P–GSCI 0,56 ; S&P–Brent 0,48","Non retrouvé (même sens, plus faible)"],
 ["Pic le plus élevé observé","—","—","S&P–GSCI 0,62 (crise de la dette euro 2011–2012)","Nouveau résultat"],
 ["Régime à 2 états : persistance P(L→L) / P(H→H)","0,94 / 0,92","0,984 / 0,988","0,976 / 0,952","Plus persistant"],
 ["Part du temps en régime bas / haut","57 % / 43 %","44 % / 56 %","67 % / 33 %","Différent"],
 ["Copule préférée (AIC)","Clayton (crash)","Clayton","Student-t (queues symétriques) ; Clayton meilleure que Gaussienne","Partiellement retrouvé"],
 ["Hausse de VaR Clayton vs Gaussienne","+8 à 14 % (VaR 10 jours, 10–15 % matières premières)","+3,8 à 7,3 % (VaR 1 mois)","+1,7 à 2,7 % (VaR 10 jours)","Non retrouvé (même sens, plus faible)"],
 ["Prévision : ensemble vs marche aléatoire","+42 % ; R² hors échantillon 0,74","−13 %","−5,6 % ; meilleurs : DCC +5,7 %, FAVAR +3,8 % (R² 0,56)","Non retrouvé"],
],columns=["Résultat","Papier (mai 2026)","Global mensuel 1999–2026","Extension US quotidienne 1990–2026","Verdict"])
lim=pd.DataFrame([
 ["Spreads HY et IG (ICE) et rendement total HY","Seulement 29 mois sur FRED (nov. 2023 → mars 2026), à cause de la restriction d'ICE sur FRED.","Non utilisés. Remplacés sur toute la période par BAA10Y et AAA10Y (435 mois). 29 mois de recouvrement ne suffisent pas pour un raccordement fiable (minimum fixé : 36 mois)."],
 ["Inflation (CPIAUCSL)","Octobre 2025 vide : chiffre non publié pendant le shutdown du gouvernement américain.","Interpolé entre septembre et novembre 2025 pour le seul calcul du glissement annuel ; la cellule source reste vide."],
 ["Cellules vides (tous fichiers)","Une cellule vide = pas d'observation (jour férié ou série pas encore lancée). Ce ne sont pas des zéros.","Toujours traitées comme manquantes. Les rendements quotidiens sont calculés uniquement sur les jours communs (pas de remplissage)."],
 ["TED spread","Arrêté fin 2021 / janvier 2022.","Utilisé jusqu'à sa dernière valeur, manquant ensuite."],
 ["OVX et GVZ","Depuis juin 2007 et juin 2008.","Manquants avant ces dates (test de robustesse sans ces variables fourni)."],
 ["Taux réel","TIPS 10 ans (DFII10) depuis 2003 seulement.","Cleveland Fed (REAINTRATREARAT10Y) avant 2003."],
 ["Actions MSCI","Historique public depuis décembre 1998.","Analyse globale sur 1999–2026 ; extension US quotidienne sur 1990–2026."],
 ["Matières premières Banque mondiale","Prix MOYENS mensuels.","Lissent les rendements (corrélations probablement sous-estimées) ; utilisées seulement dans l'analyse globale mensuelle."],
 ["S&P GSCI (Yahoo ^SPGSCI)","Version de l'indice (spot, excess return ou total return) non précisée par Yahoo.","À vérifier avant publication ; seuls les rendements de prix sont utilisés."],
 ["Pétrole WTI","Prix négatif le 20 avril 2020 : logarithme impossible.","Brent (FRED DCOILBRENTEU) utilisé comme série énergie quotidienne."],
 ["Futures Yahoo (GC=F, HG=F, ZC=F, CL=F…)","Contrats front-month continus, depuis 2000 seulement ; sauts possibles aux changements d'échéance.","Utilisés uniquement à partir de 2000 (or, cuivre, maïs)."],
 ["Indices boursiers étrangers (Yahoo)","En monnaie locale ; plusieurs commencent après 1990 ; pas de taux de change quotidien de l'euro avant 1999 sur FRED.","Pas encore utilisés dans cette version."],
 ["Sous-indices GSCI (énergie, métaux, agriculture)","Non disponibles gratuitement en quotidien.","Remplacés par Brent (énergie), or, cuivre, maïs (2000+)."],
],columns=["Donnée","Limite","Traitement dans le code"])
with pd.ExcelWriter(OUT) as w:
    comp.to_excel(w,sheet_name="Comparaison papier",index=False)
    lim.to_excel(w,sheet_name="Limites des données",index=False)
    pd.read_csv(U+"table2_unconditional_corr_monthly.csv",index_col=0).to_excel(w,sheet_name="US Table 2")
    pd.read_csv(U+"era_means.csv",index_col=0).round(3).to_excel(w,sheet_name="US Régimes (ères)")
    pd.read_csv(U+"hmm2.csv",index_col=0).round(3).to_excel(w,sheet_name="US HMM 2 états")
    pd.read_csv(U+"hmm4_episodes.csv").round(3).to_excel(w,sheet_name="US HMM 4 états",index=False)
    pd.read_csv(U+"table3_crisis_peaks_daily_max.csv",index_col=0).round(3).to_excel(w,sheet_name="US Table 3 (pics quotid.)")
    pd.read_csv(U+"table3_crisis_peaks_monthly_avg.csv",index_col=0).round(3).to_excel(w,sheet_name="US Table 3 (moy. mens.)")
    pd.read_csv(U+"copula_fits_daily.csv").to_excel(w,sheet_name="US Copules",index=False)
    pd.read_csv(U+"var10d_gaussian_vs_clayton.csv",index_col=0).round(4).to_excel(w,sheet_name="US VaR 10 jours")
    pd.read_csv(U+"table4_main.csv",index_col=0).round(3).to_excel(w,sheet_name="US Table 4 (principale)")
    pd.read_csv(U+"table4_start_late_robustness.csv",index_col=0).round(3).to_excel(w,sheet_name="US Table 4 (robustesse)")
    pd.read_csv(U+"table4_correlation_features_only.csv",index_col=0).round(3).to_excel(w,sheet_name="US Table 4 (sans macro)")
    pd.read_csv(G+"table2_unconditional_corr.csv",index_col=0).to_excel(w,sheet_name="Global Table 2")
    pd.read_csv(G+"era_mean_dcc_correlations.csv",index_col=0).round(3).to_excel(w,sheet_name="Global Régimes (ères)")
    pd.read_csv(G+"table3_crisis_peaks.csv",index_col=0).round(3).to_excel(w,sheet_name="Global Table 3")
    pd.read_csv(G+"var_gaussian_vs_clayton_1m.csv",index_col=0).round(4).to_excel(w,sheet_name="Global VaR 1 mois")
    pd.read_csv(G+"table4_oos_metrics_with_macro_features.csv",index_col=0).round(3).to_excel(w,sheet_name="Global Table 4")
    pd.read_csv(G+"country_commodity_corr.csv").round(3).to_excel(w,sheet_name="Global 47 pays",index=False)
wb=load_workbook(OUT)
for ws in wb.worksheets:
    for c in ws[1]:
        c.font=Font(bold=True,color="FFFFFF"); c.fill=PatternFill("solid",fgColor="1B2A4A"); c.alignment=Alignment(wrap_text=True)
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width=min(55,max(12,max(len(str(x.value or "")) for x in col[:40])+2))
    if ws.title in ("Comparaison papier","Limites des données"):
        for row in ws.iter_rows(min_row=2):
            for c in row: c.alignment=Alignment(wrap_text=True,vertical="top")
    ws.freeze_panes="B2"
wb.save(OUT); print("saved")
