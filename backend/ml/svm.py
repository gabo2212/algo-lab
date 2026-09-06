"""ALGORITHME 6 — Machine à vecteurs de support (SVM)

Classer une tumeur comme maligne ou bénigne. Exemple pédagogique uniquement.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from backend.json_safe import json_safe
from backend.ml.base import SupervisedAlgorithm


class Svm(SupervisedAlgorithm):
    slug = "svm"
    name = "Machine à vecteurs de support (SVM)"
    kind = "classification-binaire"
    objective = (
        "Classer une tumeur comme maligne ou bénigne à partir de mesures médicales déjà préparées."
    )
    principle = (
        "SVM cherche une frontière qui sépare les classes avec la plus grande marge possible. "
        "La normalisation est importante, car les caractéristiques du jeu de données n'utilisent "
        "pas toutes la même échelle."
    )
    how_to_read = (
        "La précision mesure la fiabilité des prédictions positives, tandis que le rappel "
        "mesure la capacité à retrouver les observations d'une classe. Le score F1 combine "
        "ces deux mesures."
    )
    file_name = "algorithme-06-svm.py"
    exercise = {
        "prompt": (
            "Remplacez kernel=\"rbf\" par kernel=\"linear\". Exécutez le programme et "
            "comparez l'exactitude et le rapport de classification."
        ),
        "changed": {"kernel": "linear"},
        "hyperparameters": {"kernel": "linear"},
    }

    def __init__(self) -> None:
        super().__init__()
        data = load_breast_cancer()
        self._data = data
        self.feature_names = list(data.feature_names)
        self.target_name = "diagnostic"
        self.class_names = list(data.target_names)
        self.sample_input = dict(zip(self.feature_names, data.data[0].tolist()))
        self.hyperparameters = {"kernel": "rbf"}
        self.disclaimer = (
            "Exemple pédagogique uniquement — ce n'est pas un diagnostic médical."
        )

    def info(self) -> dict[str, Any]:
        payload = super().info()
        payload["disclaimer"] = self.disclaimer
        return payload

    def train(self, **hyperparameters: Any) -> dict[str, Any]:
        kernel = str(hyperparameters.get("kernel", "rbf"))
        self.hyperparameters = {"kernel": kernel}

        X = pd.DataFrame(self._data.data, columns=self.feature_names)
        y = self._data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        self.model = Pipeline(
            [
                ("normalisation", StandardScaler()),
                (
                    "svm",
                    CalibratedClassifierCV(
                        SVC(kernel=kernel, random_state=42),
                        ensemble=False,
                    ),
                ),
            ]
        )
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        example = X_test.iloc[[0]]
        predicted_class = int(self.model.predict(example)[0])
        real_class = int(y_test.iloc[0] if hasattr(y_test, "iloc") else y_test[0])
        probabilities = self.model.predict_proba(example)[0]
        self.sample_input = {name: float(example.iloc[0][name]) for name in self.feature_names}

        self.metrics = json_safe(
            {
                "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
                "classification_report": classification_report(
                    y_test,
                    predictions,
                    target_names=self.class_names,
                    output_dict=True,
                ),
                "test_example": {
                    "real_class": self.class_names[real_class],
                    "predicted_class": self.class_names[predicted_class],
                    "probabilities_percent": {
                        name: round(float(prob) * 100, 2)
                        for name, prob in zip(self.class_names, probabilities)
                    },
                },
            }
        )
        self.trained = True
        return {
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "disclaimer": self.disclaimer,
        }

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        if not self.trained:
            self.train()
        row = pd.DataFrame(
            [[float(features[name]) for name in self.feature_names]],
            columns=self.feature_names,
        )
        predicted_number = int(self.model.predict(row)[0])
        probabilities = self.model.predict_proba(row)[0]
        return {
            "disclaimer": self.disclaimer,
            "input": {name: row.at[0, name] for name in self.feature_names},
            "predicted_class": predicted_number,
            "predicted_label": self.class_names[predicted_number],
            "probabilities_percent": {
                name: round(float(prob) * 100, 2)
                for name, prob in zip(self.class_names, probabilities)
            },
        }

    def run_exercise(self) -> dict[str, Any]:
        result = super().run_exercise()
        result["prediction"] = self.predict(self.sample_input)
        return result
