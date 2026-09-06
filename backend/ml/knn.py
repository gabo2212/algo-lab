"""ALGORITHME 4 — K plus proches voisins (KNN)

Classer une fleur selon les classes de ses K voisines les plus proches.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.json_safe import json_safe
from backend.ml.base import SupervisedAlgorithm


class Knn(SupervisedAlgorithm):
    slug = "knn"
    name = "K plus proches voisins (KNN)"
    kind = "classification-proximite"
    objective = "Classer une fleur en observant les classes de ses cinq voisines les plus proches."
    principle = (
        "KNN ne construit pas une formule complexe. Pour prédire une nouvelle observation, "
        "il cherche les K exemples les plus proches dans les données d'entraînement et "
        "utilise le vote majoritaire."
    )
    how_to_read = (
        "La normalisation est importante pour KNN, car l'algorithme calcule des distances. "
        "Sans elle, une caractéristique ayant de grandes valeurs pourrait dominer les autres."
    )
    file_name = "algorithme-04-knn.py"
    exercise = {
        "prompt": (
            "Remplacez n_neighbors=5 par n_neighbors=3. Exécutez de nouveau le programme "
            "et comparez l'exactitude."
        ),
        "changed": {"n_neighbors": 3},
        "hyperparameters": {"n_neighbors": 3},
        "input": {
            "sepal length (cm)": 6.0,
            "sepal width (cm)": 2.9,
            "petal length (cm)": 4.5,
            "petal width (cm)": 1.5,
        },
    }

    def __init__(self) -> None:
        super().__init__()
        iris = load_iris()
        self._iris = iris
        self.feature_names = list(iris.feature_names)
        self.target_name = "espece"
        self.class_names = list(iris.target_names)
        self.sample_input = dict(zip(self.feature_names, [6.0, 2.9, 4.5, 1.5]))
        self.hyperparameters = {"n_neighbors": 5}

    def train(self, **hyperparameters: Any) -> dict[str, Any]:
        custom = self.try_custom_train(**hyperparameters)
        if custom is not None:
            return custom
        n_neighbors = int(hyperparameters.get("n_neighbors", 5))
        self.hyperparameters = {"n_neighbors": n_neighbors}

        X = pd.DataFrame(self._iris.data, columns=self.feature_names)
        y = self._iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.model = Pipeline(
            [
                ("normalisation", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
            ]
        )
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        self.metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
            "confusion_matrix": json_safe(confusion_matrix(y_test, predictions)),
        }
        self.trained = True
        return {"metrics": self.metrics, "hyperparameters": self.hyperparameters}

