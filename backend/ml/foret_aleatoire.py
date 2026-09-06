"""ALGORITHME 5 — Forêt aléatoire

Reconnaître une catégorie de vin et mesurer l'importance des caractéristiques.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from backend.json_safe import json_safe
from backend.ml.base import SupervisedAlgorithm


class ForetAleatoire(SupervisedAlgorithm):
    slug = "foret-aleatoire"
    name = "Forêt aléatoire"
    kind = "classification-ensemble"
    objective = (
        "Reconnaître une catégorie de vin et découvrir les caractéristiques les plus importantes."
    )
    principle = (
        "Une forêt aléatoire entraîne plusieurs arbres de décision sur différentes parties "
        "des données. Les arbres votent ensuite pour produire une prédiction finale plus robuste."
    )
    how_to_read = (
        "L'importance d'une caractéristique mesure sa contribution aux décisions prises "
        "par les arbres. Une valeur élevée signifie que la forêt s'appuie souvent sur cette "
        "caractéristique."
    )
    file_name = "algorithme-05-foret-aleatoire.py"
    exercise = {
        "prompt": (
            "Remplacez n_estimators=100 par n_estimators=20, puis comparez l'exactitude. "
            "Expliquez si le résultat change."
        ),
        "changed": {"n_estimators": 20},
        "hyperparameters": {"n_estimators": 20},
    }

    def __init__(self) -> None:
        super().__init__()
        wine = load_wine()
        self._wine = wine
        self.feature_names = list(wine.feature_names)
        self.target_name = "categorie"
        self.class_names = list(wine.target_names)
        self.sample_input = dict(zip(self.feature_names, wine.data[0].tolist()))
        self.hyperparameters = {"n_estimators": 100, "max_depth": 5}
        self._example_true_class: int | None = None

    def train(self, **hyperparameters: Any) -> dict[str, Any]:
        n_estimators = int(hyperparameters.get("n_estimators", 100))
        max_depth = int(hyperparameters.get("max_depth", 5))
        self.hyperparameters = {"n_estimators": n_estimators, "max_depth": max_depth}

        X = pd.DataFrame(self._wine.data, columns=self.feature_names)
        y = self._wine.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        importance_table = (
            pd.DataFrame(
                {
                    "caracteristique": self.feature_names,
                    "importance": self.model.feature_importances_,
                }
            )
            .sort_values(by="importance", ascending=False)
            .head(5)
        )

        example = X_test.iloc[[0]]
        predicted_class = int(self.model.predict(example)[0])
        real_class = int(y_test.iloc[0] if hasattr(y_test, "iloc") else y_test[0])
        self._example_true_class = real_class
        self.sample_input = {name: float(example.iloc[0][name]) for name in self.feature_names}

        self.metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
            "confusion_matrix": json_safe(confusion_matrix(y_test, predictions)),
            "top_features": json_safe(importance_table.to_dict(orient="records")),
            "test_example": {
                "real_class": self.class_names[real_class],
                "predicted_class": self.class_names[predicted_class],
            },
        }
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
            "predicted_category": self.class_names[predicted_number],
        }

    def run_exercise(self) -> dict[str, Any]:
        result = super().run_exercise()
        result["prediction"] = self.predict(self.sample_input)
        return result
