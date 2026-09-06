# ============================================================
# ALGORITHME 5 — FORÊT ALÉATOIRE
# Type : Classification par ensemble d'arbres
# Objectif : reconnaître une catégorie de vin
# ============================================================

from backend.ml.foret_aleatoire import ForetAleatoire

# 1. Entraîner une forêt de 100 arbres, profondeur maximale 5.
algo = ForetAleatoire()
trained = algo.train(n_estimators=100, max_depth=5)

# 2. Évaluer le modèle et afficher les 5 caractéristiques les plus importantes.
print("Exactitude :", trained["metrics"]["accuracy"])
print("Matrice de confusion :")
print(trained["metrics"]["confusion_matrix"])
print("Cinq caractéristiques les plus importantes :")
for row in trained["metrics"]["top_features"]:
    print(row["caracteristique"], row["importance"])

# 3. Comparer la classe réelle et la classe prédite du premier exemple de test.
example = trained["metrics"]["test_example"]
print("Classe réelle :", example["real_class"])
print("Classe prédite :", example["predicted_class"])
