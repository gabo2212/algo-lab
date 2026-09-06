"""ALGORITHME 3 — Arbre de décision

Reconnaître l'espèce d'une fleur Iris à partir de quatre mesures.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from backend.json_safe import json_safe
from backend.ml.base import SupervisedAlgorithm


class ArbreDecision(SupervisedAlgorithm):
    slug = "arbre-decision"
    name = "Arbre de décision"
    kind = "classification-multiclasse"
    objective = "Reconnaître l'espèce d'une fleur à partir de quatre mesures physiques."
    principle = (
        "Un arbre de décision construit une suite de règles de type « si… alors… ». "
        "Chaque séparation utilise une caractéristique pour rapprocher les observations "
        "de leur classe correcte."
    )
    how_to_read = (
        "Le rapport présente la précision, le rappel et le score F1 pour chaque espèce. "
        "Les règles affichées montrent les décisions apprises par l'arbre."
    )
    file_name = "algorithme-03-arbre-decision.py"
    exercise = {
        "prompt": (
            "Remplacez les mesures de la nouvelle fleur par [6.5, 3.0, 5.2, 2.0], "
            "puis observez l'espèce prédite."
        ),
        "changed": {
            "sepal length (cm)": 6.5,
            "sepal width (cm)": 3.0,
            "petal length (cm)": 5.2,
            "petal width (cm)": 2.0,
        },
        "input": {
            "sepal length (cm)": 6.5,
            "sepal width (cm)": 3.0,
            "petal length (cm)": 5.2,
            "petal width (cm)": 2.0,
        },
    }

    def __init__(self) -> None:
        super().__init__()
        iris = load_iris()
        self._iris = iris
        self.feature_names = list(iris.feature_names)
        self.target_name = "espece"
        self.class_names = list(iris.target_names)
        self.sample_input = dict(zip(self.feature_names, [5.1, 3.5, 1.4, 0.2]))
        self.hyperparameters = {"max_depth": 3}

    def train(self, **hyperparameters: Any) -> dict[str, Any]:
        max_depth = int(hyperparameters.get("max_depth", 3))
        self.hyperparameters = {"max_depth": max_depth}

        X = pd.DataFrame(self._iris.data, columns=self.feature_names)
        y = self._iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        self.metrics = json_safe(
            {
                "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
                "classification_report": classification_report(
                    y_test,
                    predictions,
                    target_names=self.class_names,
                    output_dict=True,
                ),
                "rules": export_text(self.model, feature_names=self.feature_names),
            }
        )
        self.trained = True
        return {"metrics": self.metrics, "hyperparameters": self.hyperparameters}

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        if not self.trained:
            self.train()
        row = pd.DataFrame(
            [[float(features[name]) for name in self.feature_names]],
            columns=self.feature_names,
        )
        predicted_number = int(self.model.predict(row)[0])
        return {
            "input": {name: row.at[0, name] for name in self.feature_names},
            "predicted_class": predicted_number,
            "predicted_species": self.class_names[predicted_number],
        }
