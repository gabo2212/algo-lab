"""ALGORITHME 1 — Régression linéaire

Prédire le prix d'une maison à partir de sa superficie et du nombre de chambres.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from backend.json_safe import json_safe
from backend.ml.base import SupervisedAlgorithm


class RegressionLineaire(SupervisedAlgorithm):
    slug = "regression-lineaire"
    name = "Régression linéaire"
    kind = "regression"
    objective = (
        "Prédire le prix d'une maison à partir de sa superficie et du nombre de chambres."
    )
    principle = (
        "La régression linéaire recherche une relation mathématique entre les "
        "caractéristiques d'entrée et une valeur numérique à prédire. Ici, le prix est la cible y."
    )
    how_to_read = (
        "La MAE représente l'erreur moyenne en dollars. Le coefficient R² se rapproche "
        "de 1 lorsque le modèle explique correctement la variation des prix."
    )
    file_name = "algorithme-01-regression-lineaire.py"
    exercise = {
        "prompt": (
            "Modifiez la nouvelle maison pour utiliser une superficie de 130 m² "
            "et 4 chambres. Notez le nouveau prix prédit."
        ),
        "changed": {"surface": 130, "chambres": 4},
        "input": {"surface": 130, "chambres": 4},
    }

    def __init__(self) -> None:
        super().__init__()
        self.feature_names = ["surface", "chambres"]
        self.target_name = "prix"
        self.sample_input = {"surface": 100, "chambres": 3}
        self.hyperparameters = {}

    def _dataset(self) -> pd.DataFrame:
        # Petit jeu supervisé : chaque ligne a déjà un prix connu.
        return pd.DataFrame(
            {
                "surface": [45, 55, 65, 75, 85, 95, 110, 125, 140, 155, 170, 185],
                "chambres": [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
                "prix": [
                    155000,
                    180000,
                    215000,
                    240000,
                    280000,
                    310000,
                    350000,
                    395000,
                    440000,
                    475000,
                    520000,
                    565000,
                ],
            }
        )

    def train(self, **hyperparameters: Any) -> dict[str, Any]:
        data = self._dataset()
        X = data[self.feature_names]
        y = data[self.target_name]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        self.metrics = {
            "y_true": json_safe(y_test.to_list()),
            "y_pred": json_safe(predictions.round(2).tolist()),
            "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
            "r2": round(float(r2_score(y_test, predictions)), 3),
        }
        self.trained = True
        return {"metrics": self.metrics, "hyperparameters": self.hyperparameters}

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        if not self.trained:
            self.train()
        row = pd.DataFrame(
            [
                {
                    "surface": float(features["surface"]),
                    "chambres": float(features["chambres"]),
                }
            ]
        )
        price = float(self.model.predict(row)[0])
        return {
            "input": {"surface": row.at[0, "surface"], "chambres": row.at[0, "chambres"]},
            "predicted_price": round(price, 2),
            "unit": "$",
        }
