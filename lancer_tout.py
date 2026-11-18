"""LANCER TOUT / RUN EVERYTHING

Double-cliquez ou, dans Anaconda Prompt :   python lancer_tout.py
Dans Spyder / PyCharm : ouvrez ce fichier et cliquez sur Run (F5).
Dans Jupyter : %run lancer_tout.py

Etapes (5 a 20 minutes selon l'ordinateur) :
 1. verification des bibliotheques
 2. analyse mondiale mensuelle (MSCI + Banque mondiale)  -> results_collected/
 3. analyse US quotidienne (S&P 500, GSCI, Brent, ...)   -> results_daily_us/
 4. tests complementaires (ruptures, Diebold-Mariano...) -> results_extra/
 5. figures du papier                                     -> figures/
 6. classeur Excel de synthese                            -> Resultats_CrossAsset.xlsx
"""
import os, sys, runpy, time, traceback
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
os.chdir(HERE); sys.path.insert(0, HERE)

print("Python", sys.version.split()[0], "| dossier :", HERE)
if sys.version_info < (3, 9):
    sys.exit("Python 3.9 ou plus recent est necessaire.")
missing = []
for mod, pipname in [("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("sklearn", "scikit-learn"),
                     ("matplotlib", "matplotlib"), ("openpyxl", "openpyxl")]:
    try:
        __import__(mod)
    except ImportError:
        missing.append(pipname)
if missing:
    sys.exit("Bibliotheques manquantes : " + ", ".join(missing) +
             "\nDans Anaconda Prompt, tapez :  pip install " + " ".join(missing))
if not os.path.isdir(os.path.join(HERE, "cross_asset")) or not os.path.isdir(os.path.join(HERE, "data")):
    sys.exit("Les dossiers 'cross_asset' et 'data' sont introuvables a cote de ce fichier.\n"
             "Avez-vous bien EXTRAIT le zip (clic droit > Extraire tout) au lieu de l'ouvrir directement ?")

steps = [("Analyse mondiale mensuelle", "run_collected_data.py"),
         ("Analyse US quotidienne", "run_daily_us.py"),
         ("Tests complementaires", "extra_analysis.py"),
         ("Figures", "make_figures.py"),
         ("Classeur Excel", "make_report.py")]
for i, (name, script) in enumerate(steps, 1):
    t0 = time.time()
    print(f"\n=== {i}/{len(steps)} {name} ({script}) ===", flush=True)
    try:
        sys.argv = [script]
        runpy.run_path(os.path.join(HERE, script), run_name="__main__")
    except SystemExit:
        pass
    except Exception:
        traceback.print_exc()
        sys.exit(f"\nERREUR a l'etape '{name}'. Copiez le message ci-dessus et envoyez-le.")
    print(f"--- termine en {time.time() - t0:.0f} s")
print("\nTermine. Resultats dans results_collected/, results_daily_us/, results_extra/, figures/ et Resultats_CrossAsset.xlsx")
if os.name == "nt" and not sys.stdin is None and sys.stdin.isatty():
    input("Appuyez sur Entree pour fermer...")
