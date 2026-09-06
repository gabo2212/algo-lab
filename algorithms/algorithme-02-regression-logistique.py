# ============================================================
# ALGORITHME 2 — RÉGRESSION LOGISTIQUE
# Type : Classification binaire — oui ou non
# Objectif : prédire la réussite ou l'échec à un examen
# ============================================================

from backend.ml.regression_logistique import RegressionLogistique

# 1. Entraîner le pipeline (normalisation + régression logistique).
algo = RegressionLogistique()
trained = algo.train()

# 2. Lire l'exactitude et la matrice de confusion sur le jeu de test.
print("Valeurs réelles :", trained["metrics"]["y_true"])
print("Classes prédites :", trained["metrics"]["y_pred"])
print("Exactitude :", trained["metrics"]["accuracy"])
print("Matrice de confusion :")
print(trained["metrics"]["confusion_matrix"])

# 3. Tester un nouvel étudiant : 5 heures d'étude, 74 % de présence.
example = algo.predict({"heures_etude": 5, "presence": 74})
print("Classe prédite :", example["predicted_class"])
print("Probabilité de réussite :", example["success_probability_percent"], "%")
print("Résultat :", example["result"])
