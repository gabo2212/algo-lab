"""API FastAPI pour les six algorithmes d'apprentissage supervisé du devoir."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

from backend.csv_io import predict_csv, sample_csv_response
from backend.kaggle_io import import_dataset, kaggle_status, list_compatible_datasets
from backend.json_safe import json_safe
from backend.ml import ALGORITHMS
from backend.ml.base import SupervisedAlgorithm

COMMON_CYCLE = [
    "Obtenir un jeu de données contenant des exemples et des réponses connues.",
    "Séparer X, les caractéristiques, et y, la cible à prédire.",
    "Créer des ensembles d'entraînement et de test.",
    "Entraîner le modèle avec model.fit(X_train, y_train).",
    "Faire des prédictions avec model.predict(X_test).",
    "Comparer les prédictions aux vraies réponses avec des métriques adaptées.",
    "Tester le modèle sur une nouvelle observation ayant la même structure que X.",
]


class TrainRequest(BaseModel):
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    features: dict[str, Any] | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class KaggleImportRequest(BaseModel):
    dataset: str
    target_column: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


registry: dict[str, SupervisedAlgorithm] = {}


def get_algorithm(slug: str) -> SupervisedAlgorithm:
    algorithm = registry.get(slug)
    if algorithm is None:
        raise HTTPException(status_code=404, detail=f"Algorithme inconnu : {slug}")
    return algorithm


@asynccontextmanager
async def lifespan(_app: FastAPI):
    for slug, algorithm_cls in ALGORITHMS.items():
        instance = algorithm_cls()
        instance.train()
        registry[slug] = instance
    yield
    registry.clear()


app = FastAPI(
    title="Apprentissage supervisé — 6 algorithmes",
    description=(
        "Backend minimal du devoir : entraîner, évaluer et prédire avec "
        "régression linéaire, régression logistique, arbre de décision, KNN, "
        "forêt aléatoire et SVM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/meta")
def meta() -> dict[str, Any]:
    return {
        "title": "Apprentissage supervisé",
        "goal": (
            "Exécuter six programmes, comprendre la différence entre la régression "
            "et la classification, puis interpréter les prédictions et les métriques."
        ),
        "cycle": COMMON_CYCLE,
        "algorithms": [algo.info() for algo in registry.values()],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/algorithms")
def list_algorithms() -> dict[str, Any]:
    return {"algorithms": [algo.info() for algo in registry.values()]}


@app.get("/algorithms/{slug}")
def get_algorithm_detail(slug: str) -> dict[str, Any]:
    algorithm = get_algorithm(slug)
    return json_safe(
        {
            **algorithm.info(),
            "metrics": algorithm.metrics,
        }
    )


@app.post("/algorithms/{slug}/train")
def train_algorithm(slug: str, body: TrainRequest | None = None) -> dict[str, Any]:
    algorithm = get_algorithm(slug)
    hyperparameters = body.hyperparameters if body else {}
    result = algorithm.train(**hyperparameters)
    return json_safe({"slug": slug, **result})


@app.post("/algorithms/{slug}/predict")
def predict_algorithm(slug: str, body: PredictRequest | None = None) -> dict[str, Any]:
    algorithm = get_algorithm(slug)
    payload = body or PredictRequest()
    if payload.hyperparameters:
        algorithm.train(**payload.hyperparameters)
    features = payload.features or algorithm.sample_input
    missing = [name for name in algorithm.feature_names if name not in features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Caractéristiques manquantes : {', '.join(missing)}",
        )
    prediction = algorithm.predict(features)
    return json_safe(
        {
            "slug": slug,
            "hyperparameters": algorithm.hyperparameters,
            "metrics": algorithm.metrics,
            "prediction": prediction,
        }
    )


@app.post("/algorithms/{slug}/exercise")
def run_exercise(slug: str) -> dict[str, Any]:
    algorithm = get_algorithm(slug)
    result = algorithm.run_exercise()
    return json_safe(
        {
            "slug": slug,
            "how_to_read": algorithm.how_to_read,
            **result,
        }
    )


@app.get("/kaggle/status")
def kaggle_status_endpoint() -> dict[str, Any]:
    return kaggle_status()


@app.get("/algorithms/{slug}/kaggle/suggestions")
def kaggle_suggestions(slug: str) -> dict[str, Any]:
    algorithm = get_algorithm(slug)
    return json_safe(list_compatible_datasets(algorithm))


@app.post("/algorithms/{slug}/kaggle")
def import_kaggle_dataset(slug: str, body: KaggleImportRequest) -> dict[str, Any]:
    algorithm = get_algorithm(slug)
    return json_safe(
        import_dataset(
            algorithm,
            body.dataset,
            body.target_column,
            body.hyperparameters,
        )
    )


@app.get("/algorithms/{slug}/sample.csv")
def download_sample_csv(slug: str) -> Response:
    return sample_csv_response(get_algorithm(slug))


@app.post("/algorithms/{slug}/predict-csv")
async def predict_csv_endpoint(
    slug: str,
    file: UploadFile = File(...),
    hyperparameters: str | None = Form(default=None),
) -> dict[str, Any]:
    algorithm = get_algorithm(slug)
    params: dict[str, Any] = {}
    if hyperparameters:
        try:
            loaded = json.loads(hyperparameters)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Hyperparamètres JSON invalides.") from exc
        if isinstance(loaded, dict):
            params = loaded
    raw = await file.read()
    result = predict_csv(
        algorithm,
        raw,
        file.filename or "upload.csv",
        hyperparameters=params,
    )
    return json_safe(
        {
            "slug": slug,
            "hyperparameters": algorithm.hyperparameters,
            "metrics": algorithm.metrics,
            **result,
        }
    )


app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")
