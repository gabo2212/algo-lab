"""ALGORITHME 2 — Régression logistique

Prédire si une personne réussira un examen (classification binaire).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.json_safe import json_safe
from backend.ml.base import SupervisedAlgorithm


class RegressionLogistique(SupervisedAlgorithm):
    slug = "regression-logistique"
    name = "Régression logistique"
    kind = "classification-binaire"
    objective = (
        "Prédire si une personne réussira un examen à partir de son nombre "
        "d'heures d'étude et de son taux de présence."
    )
    principle = (
        "Malgré son nom, la régression logistique est un algorithme de classification. "
        "Elle produit une probabilité, puis choisit une classe comme 0 ou 1."
    )
    how_to_read = (
        "L'exactitude indique la proportion de bonnes prédictions. La probabilité "
        "de réussite permet de voir le niveau de confiance du modèle avant la décision finale 0 ou 1."
    )
    file_name = "algorithme-02-regression-logistique.py"
    exercise = {
        "prompt": (
            "Testez un étudiant qui étudie 3 heures et possède un taux de présence "
            "de 60 %. Comparez sa classe et sa probabilité avec celles de l'exemple."
        ),
        "changed": {"heures_etude": 3, "presence": 60},
        "input": {"heures_etude": 3, "presence": 60},
    }

    def __init__(self) -> None:
        super().__init__()
        self.feature_names = ["heures_etude", "presence"]
        self.target_name = "reussite"
        self.class_names = ["Échec", "Réussite"]
        self.sample_input = {"heures_etude": 5, "presence": 74}
        self.hyperparameters = {}

    def _dataset(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "heures_etude": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11],
                "presence": [45, 50, 55, 58, 62, 65, 68, 72, 75, 78, 82, 85, 88, 92, 95, 98],
                "reussite": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )

    def train(self, **hyperparameters: Any) -> dict[str, Any]:
        custom = self.try_custom_train(**hyperparameters)
        if custom is not None:
            return custom
        data = self._dataset()
        X = data[self.feature_names]
        y = data[self.target_name]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.model = Pipeline(
            [
                ("normalisation", StandardScaler()),
                ("classification", LogisticRegression()),
            ]
        )
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        self.metrics = {
            "y_true": json_safe(y_test.to_list()),
            "y_pred": json_safe(predictions.tolist()),
            "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
            "confusion_matrix": json_safe(confusion_matrix(y_test, predictions)),
        }
        self.trained = True
        return {"metrics": self.metrics, "hyperparameters": self.hyperparameters}
