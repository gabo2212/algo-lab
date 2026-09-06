"""Shared interface for the six supervised-learning algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SupervisedAlgorithm(ABC):
    slug: str
    name: str
    kind: str
    objective: str
    principle: str
    how_to_read: str
    exercise: dict[str, Any]
    file_name: str

    def __init__(self) -> None:
        self.model: Any = None
        self.feature_names: list[str] = []
        self.target_name: str = ""
        self.class_names: list[str] | None = None
        self.metrics: dict[str, Any] = {}
        self.sample_input: dict[str, Any] = {}
        self.hyperparameters: dict[str, Any] = {}
        self.trained = False
        self.custom_training: dict[str, Any] | None = None
        self.dataset_source: str = "assignment"

    def info(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "name": self.name,
            "kind": self.kind,
            "objective": self.objective,
            "principle": self.principle,
            "how_to_read": self.how_to_read,
            "file_name": self.file_name,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "class_names": self.class_names,
            "sample_input": self.sample_input,
            "hyperparameters": self.hyperparameters,
            "trained": self.trained,
            "exercise": self.exercise,
            "source": self.dataset_source,
        }

    def try_custom_train(self, **hyperparameters: Any) -> dict[str, Any] | None:
        if self.custom_training is None:
            return None
        from backend.ml.trainer import fit_custom

        return fit_custom(
            self,
            self.custom_training["X"],
            self.custom_training["y"],
            **hyperparameters,
        )

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        from backend.ml.trainer import predict_observation

        return predict_observation(self, features)

    @abstractmethod
    def train(self, **hyperparameters: Any) -> dict[str, Any]:
        """Train on the assignment dataset and store evaluation metrics."""

    def run_exercise(self) -> dict[str, Any]:
        """Run the 'À vous de jouer' variant from the assignment."""
        params = dict(self.exercise.get("hyperparameters") or {})
        train_result = self.train(**params) if params else None
        prediction = None
        exercise_input = self.exercise.get("input")
        if exercise_input is not None and set(exercise_input).issubset(set(self.feature_names)):
            prediction = self.predict(exercise_input)
        return {
            "prompt": self.exercise["prompt"],
            "changed": self.exercise.get("changed"),
            "train": train_result,
            "prediction": prediction,
        }
