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
        }

    @abstractmethod
    def train(self, **hyperparameters: Any) -> dict[str, Any]:
        """Train on the assignment dataset and store evaluation metrics."""

    @abstractmethod
    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Predict one new observation with the same structure as X."""

    def run_exercise(self) -> dict[str, Any]:
        """Run the 'À vous de jouer' variant from the assignment."""
        params = dict(self.exercise.get("hyperparameters") or {})
        train_result = self.train(**params) if params else None
        prediction = None
        if self.exercise.get("input") is not None:
            prediction = self.predict(self.exercise["input"])
        return {
            "prompt": self.exercise["prompt"],
            "changed": self.exercise.get("changed"),
            "train": train_result,
            "prediction": prediction,
        }
