# ============================================================
# ALGORITHME 1 — RÉGRESSION LINÉAIRE
# Type : Régression — prédiction d'une valeur numérique
# Objectif : prédire le prix d'une maison
# ============================================================

from backend.ml.regression_lineaire import RegressionLineaire

# 1. Créer le modèle et l'entraîner sur le petit jeu de maisons.
algo = RegressionLineaire()
trained = algo.train()

# 2. Afficher les métriques sur le jeu de test (MAE et R²).
print("Valeurs réelles :", trained["metrics"]["y_true"])
print("Prédictions :", trained["metrics"]["y_pred"])
print("Erreur absolue moyenne :", trained["metrics"]["mae"])
print("Coefficient R² :", trained["metrics"]["r2"])

# 3. Prédire le prix d'une nouvelle maison de 100 m² et 3 chambres.
example = algo.predict({"surface": 100, "chambres": 3})
print(
    "Prix prédit pour la nouvelle maison :",
    example["predicted_price"],
    example["unit"],
)
