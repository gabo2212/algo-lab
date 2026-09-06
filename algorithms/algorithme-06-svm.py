# ============================================================
# ALGORITHME 6 — MACHINE À VECTEURS DE SUPPORT (SVM)
# Type : Classification binaire avec séparation optimale
# Objectif : classer une tumeur comme maligne ou bénigne
# Exemple pédagogique uniquement — ce n'est pas un diagnostic médical
# ============================================================

from backend.ml.svm import Svm

# 1. Entraîner un SVM RBF avec normalisation.
algo = Svm()
trained = algo.train(kernel="rbf")

# 2. Lire l'exactitude et le rapport de classification.
print("Exactitude :", trained["metrics"]["accuracy"])
print("Rapport de classification :")
print(trained["metrics"]["classification_report"])

# 3. Tester une observation du jeu de test.
example = trained["metrics"]["test_example"]
print("Classe réelle :", example["real_class"])
print("Classe prédite :", example["predicted_class"])
print("Probabilités :")
for name, probability in example["probabilities_percent"].items():
    print(name, ":", probability, "%")
