# ============================================================
# ALGORITHME 4 — K PLUS PROCHES VOISINS (KNN)
# Type : Classification fondée sur la proximité
# Objectif : classer une fleur selon ses voisines
# ============================================================

from backend.ml.knn import Knn

# 1. Entraîner KNN avec K=5, après normalisation des distances.
algo = Knn()
trained = algo.train(n_neighbors=5)

# 2. Lire l'exactitude et la matrice de confusion.
print("Exactitude :", trained["metrics"]["accuracy"])
print("Matrice de confusion :")
print(trained["metrics"]["confusion_matrix"])

# 3. Prédire une nouvelle fleur et afficher les probabilités par espèce.
example = algo.predict(
    {
        "sepal length (cm)": 6.0,
        "sepal width (cm)": 2.9,
        "petal length (cm)": 4.5,
        "petal width (cm)": 1.5,
    }
)
print("Espèce prédite :", example["predicted_species"])
print("Probabilités par espèce :")
for name, probability in example["probabilities_percent"].items():
    print(name, ":", probability, "%")
