# ============================================================
# ALGORITHME 3 — ARBRE DE DÉCISION
# Type : Classification multiclasse
# Objectif : reconnaître l'espèce d'une fleur Iris
# ============================================================

from backend.ml.arbre_decision import ArbreDecision

# 1. Entraîner un arbre limité à trois niveaux.
algo = ArbreDecision()
trained = algo.train(max_depth=3)

# 2. Évaluer l'arbre et afficher les règles apprises.
print("Exactitude :", trained["metrics"]["accuracy"])
print("Rapport de classification :")
print(trained["metrics"]["classification_report"])
print("Règles apprises :")
print(trained["metrics"]["rules"])

# 3. Prédire une nouvelle fleur avec les mesures de l'exemple du devoir.
example = algo.predict(
    {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2,
    }
)
print("Espèce prédite :", example["predicted_species"])
