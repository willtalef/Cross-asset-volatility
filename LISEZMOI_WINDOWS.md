# Lancer le code sur Windows (Anaconda)

1. **Extraire le zip** : clic droit sur `cross_asset_replication.zip` > *Extraire tout...*
   Ne lancez pas les fichiers depuis l'intérieur du zip : Python ne voit alors ni le dossier `data` ni le dossier `cross_asset`.
2. Ouvrez **Anaconda Prompt** (menu Démarrer), puis allez dans le dossier extrait, par exemple :
   ```
   cd %USERPROFILE%\Downloads\cross_asset_replication\cross_asset_replication
   ```
3. Lancez tout :
   ```
   python lancer_tout.py
   ```
   Durée : 2 à 10 minutes. Les résultats arrivent dans `results_collected/`, `results_daily_us/`, `results_extra/`, `figures/` et `Resultats_CrossAsset.xlsx`.

**Spyder / PyCharm** : ouvrez `lancer_tout.py` et cliquez sur Run (F5).
**Jupyter** : dans une cellule, `%run chemin\vers\cross_asset_replication\lancer_tout.py`

Si une bibliothèque manque, le script affiche la commande `pip install ...` à taper.
Optionnel (vrai réseau LSTM et XGBoost) : `pip install -r requirements-optional.txt`.
Si une erreur apparaît, copiez le message rouge complet et envoyez-le.
