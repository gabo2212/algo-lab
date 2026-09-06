"""Fit any of the six algorithms on an external labelled table."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

from backend.json_safe import json_safe
from backend.ml.base import SupervisedAlgorithm


def build_model(algorithm: SupervisedAlgorithm, n_train: int, **hyperparameters: Any):
    slug = algorithm.slug
    if slug == "regression-lineaire":
        algorithm.hyperparameters = {}
        return LinearRegression()
    if slug == "regression-logistique":
        algorithm.hyperparameters = {}
        return Pipeline(
            [
                ("normalisation", StandardScaler()),
                ("classification", LogisticRegression(max_iter=400)),
            ]
        )
    if slug == "arbre-decision":
        max_depth = int(hyperparameters.get("max_depth", algorithm.hyperparameters.get("max_depth", 3)))
        algorithm.hyperparameters = {"max_depth": max_depth}
        return DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    if slug == "knn":
        requested = int(hyperparameters.get("n_neighbors", algorithm.hyperparameters.get("n_neighbors", 5)))
        n_neighbors = max(1, min(requested, max(1, n_train - 1)))
        algorithm.hyperparameters = {"n_neighbors": n_neighbors}
        return Pipeline(
            [
                ("normalisation", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
            ]
        )
    if slug == "foret-aleatoire":
        n_estimators = int(hyperparameters.get("n_estimators", algorithm.hyperparameters.get("n_estimators", 100)))
        max_depth = int(hyperparameters.get("max_depth", algorithm.hyperparameters.get("max_depth", 5)))
        algorithm.hyperparameters = {"n_estimators": n_estimators, "max_depth": max_depth}
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
    kernel = str(hyperparameters.get("kernel", algorithm.hyperparameters.get("kernel", "rbf")))
    algorithm.hyperparameters = {"kernel": kernel}
    return Pipeline(
        [
            ("normalisation", StandardScaler()),
            (
                "svm",
                CalibratedClassifierCV(SVC(kernel=kernel, random_state=42), ensemble=False),
            ),
        ]
    )


def fit_custom(
    algorithm: SupervisedAlgorithm,
    X: pd.DataFrame,
    y: pd.Series,
    **hyperparameters: Any,
) -> dict[str, Any]:
    stratify = None
    if algorithm.kind != "regression" and y.nunique() > 1 and (y.value_counts() >= 2).all():
        stratify = y
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25 if len(X) >= 16 else 0.2,
        random_state=42,
        stratify=stratify,
    )
    algorithm.model = build_model(algorithm, n_train=len(X_train), **hyperparameters)
    algorithm.model.fit(X_train, y_train)
    predictions = algorithm.model.predict(X_test)
    algorithm.feature_names = list(X.columns)
    algorithm.target_name = str(y.name or "target")
    algorithm.sample_input = {name: float(X.iloc[0][name]) for name in algorithm.feature_names}
    if algorithm.kind == "regression":
        algorithm.class_names = None
        algorithm.metrics = {
            "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
            "r2": round(float(r2_score(y_test, predictions)), 3),
            "y_true": json_safe([round(float(value), 2) for value in list(y_test)[:80]]),
            "y_pred": json_safe([round(float(value), 2) for value in list(predictions)[:80]]),
        }
    else:
        estimator = algorithm.model
        if hasattr(estimator, "named_steps"):
            estimator = list(estimator.named_steps.values())[-1]
        classes = [str(item) for item in getattr(estimator, "classes_", pd.unique(y))]
        algorithm.class_names = classes
        algorithm.metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
            "confusion_matrix": json_safe(confusion_matrix(y_test, predictions)),
        }
        try:
            algorithm.metrics["classification_report"] = classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            )
        except Exception:
            pass
        if algorithm.slug == "arbre-decision":
            algorithm.metrics["rules"] = export_text(
                algorithm.model,
                feature_names=algorithm.feature_names,
            )
        if algorithm.slug == "foret-aleatoire":
            importance = (
                pd.DataFrame(
                    {
                        "caracteristique": algorithm.feature_names,
                        "importance": algorithm.model.feature_importances_,
                    }
                )
                .sort_values(by="importance", ascending=False)
                .head(5)
            )
            algorithm.metrics["top_features"] = json_safe(importance.to_dict(orient="records"))
    algorithm.trained = True
    algorithm.metrics = json_safe(algorithm.metrics)
    return {"metrics": algorithm.metrics, "hyperparameters": algorithm.hyperparameters}


def predict_observation(algorithm: SupervisedAlgorithm, features: dict[str, Any]) -> dict[str, Any]:
    if not algorithm.trained:
        algorithm.train()
    row = pd.DataFrame(
        [[float(features[name]) for name in algorithm.feature_names]],
        columns=algorithm.feature_names,
    )
    raw = algorithm.model.predict(row)[0]
    payload: dict[str, Any] = {
        "input": {name: row.at[0, name] for name in algorithm.feature_names},
    }
    if algorithm.kind == "regression":
        value = round(float(raw), 2)
        payload["predicted_value"] = value
        payload["predicted_price"] = value
        payload["unit"] = "$" if "prix" in algorithm.target_name.lower() or "price" in algorithm.target_name.lower() else ""
        payload["target"] = algorithm.target_name
        estimator = algorithm.model
        if hasattr(estimator, "named_steps"):
            estimator = list(estimator.named_steps.values())[-1]
        coef = getattr(estimator, "coef_", None)
        if coef is not None:
            weights = [float(item) for item in list(coef.ravel())]
            intercept_raw = getattr(estimator, "intercept_", 0)
            intercept = (
                float(intercept_raw.ravel()[0])
                if hasattr(intercept_raw, "ravel")
                else float(intercept_raw or 0)
            )
            payload["intercept"] = round(intercept, 2)
            payload["feature_effects"] = [
                {
                    "name": name,
                    "effect": round(weights[index] * float(row.at[0, name]), 2),
                }
                for index, name in enumerate(algorithm.feature_names)
                if index < len(weights)
            ]
        return payload

    label = str(raw)
    payload["predicted_class"] = json_safe(raw)
    payload["predicted_label"] = label
    payload["predicted_species"] = label
    payload["predicted_category"] = label
    payload["result"] = label
    if hasattr(algorithm.model, "predict_proba"):
        probabilities = algorithm.model.predict_proba(row)[0]
        names = algorithm.class_names or [str(index) for index in range(len(probabilities))]
        payload["probabilities_percent"] = {
            name: round(float(prob) * 100, 2) for name, prob in zip(names, probabilities)
        }
        if len(probabilities) == 2:
            payload["success_probability_percent"] = round(float(probabilities[-1]) * 100, 2)
    if algorithm.slug == "svm":
        payload["disclaimer"] = (
            "Exemple pédagogique uniquement — ce n'est pas un diagnostic médical."
        )
    return payload
