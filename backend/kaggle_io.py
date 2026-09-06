"""Download a Kaggle dataset, check it matches an algorithm, then retrain."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException

from backend.csv_io import normalize_name
from backend.json_safe import json_safe
from backend.ml.base import SupervisedAlgorithm
from backend.ml.trainer import fit_custom

MAX_CSV_BYTES = 20_000_000
MAX_TRAIN_ROWS = 4000
ID_NAME = re.compile(
    r"^(id|idx|index|unnamed.*|name|nom|url|email|date|time|timestamp|filename)$",
    re.I,
)
TARGET_HINTS = (
    "target",
    "label",
    "class",
    "classe",
    "y",
    "species",
    "espece",
    "espèce",
    "prix",
    "price",
    "reussite",
    "outcome",
    "diagnosis",
    "diagnostic",
    "quality",
    "survived",
    "categorie",
    "category",
)

CATALOG = [
    {
        "slugs": ["regression-lineaire"],
        "ref": "camnugent/california-housing-prices",
        "title": "California Housing Prices",
        "target": "median_house_value",
        "why": "Prix médian des logements — cible continue.",
    },
    {
        "slugs": ["regression-lineaire"],
        "ref": "harlfoxem/housesalesprediction",
        "title": "House Sales in King County",
        "target": "price",
        "why": "Prix de vente, caractéristiques numériques du logement.",
    },
    {
        "slugs": ["regression-lineaire"],
        "ref": "quantbruce/real-estate-price-prediction",
        "title": "Real Estate Price Prediction",
        "target": "Y house price of unit area",
        "why": "Prix unitaire immobilier, régression.",
    },
    {
        "slugs": ["regression-lineaire"],
        "ref": "uciml/autompg-dataset",
        "title": "Auto MPG",
        "target": "mpg",
        "why": "Consommation (mpg) à partir de mesures numériques.",
    },
    {
        "slugs": ["regression-logistique", "svm"],
        "ref": "uciml/pima-indians-diabetes-database",
        "title": "Pima Indians Diabetes",
        "target": "Outcome",
        "why": "Cible binaire Outcome, colonnes numériques.",
    },
    {
        "slugs": ["regression-logistique", "svm", "arbre-decision", "knn", "foret-aleatoire"],
        "ref": "uciml/breast-cancer-wisconsin-data",
        "title": "Breast Cancer Wisconsin",
        "target": "diagnosis",
        "why": "Diagnostic M/B et mesures numériques.",
    },
    {
        "slugs": ["regression-logistique", "svm", "arbre-decision", "knn", "foret-aleatoire"],
        "ref": "rashikrahmanpritom/heart-attack-analysis-prediction-dataset",
        "title": "Heart Attack Analysis",
        "target": "output",
        "why": "Risque cardiaque binaire, variables numériques.",
    },
    {
        "slugs": ["regression-logistique", "svm"],
        "ref": "johnsmith88/heart-disease-dataset",
        "title": "Heart Disease Dataset",
        "target": "target",
        "why": "Présence de maladie cardiaque (2 classes).",
    },
    {
        "slugs": ["arbre-decision", "knn", "foret-aleatoire"],
        "ref": "uciml/iris",
        "title": "Iris",
        "target": "Species",
        "why": "3 espèces, 4 mesures numériques.",
    },
    {
        "slugs": ["arbre-decision", "knn", "foret-aleatoire"],
        "ref": "uciml/red-wine-quality-cortez-et-al-2009",
        "title": "Red Wine Quality",
        "target": "quality",
        "why": "Qualité du vin (classes) et chimie numérique.",
    },
    {
        "slugs": ["arbre-decision", "knn", "foret-aleatoire"],
        "ref": "uciml/glass",
        "title": "Glass Identification",
        "target": "Type",
        "why": "Types de verre, composition chimique.",
    },
    {
        "slugs": ["arbre-decision", "knn", "foret-aleatoire"],
        "ref": "iabhishekofficial/mobile-price-classification",
        "title": "Mobile Price Classification",
        "target": "price_range",
        "why": "4 gammes de prix, caractéristiques numériques.",
    },
]

SUGGESTIONS = {
    slug: [{"ref": item["ref"], "why": item["why"]} for item in CATALOG if slug in item["slugs"]]
    for slug in {slug for item in CATALOG for slug in item["slugs"]}
}

DATASET_REF = re.compile(r"([A-Za-z0-9_-]+)/([A-Za-z0-9._-]+)")


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def kaggle_status() -> dict[str, Any]:
    load_local_env()
    cred_file = Path.home() / ".kaggle" / "kaggle.json"
    token_file = Path.home() / ".kaggle" / "access_token"
    token = (os.environ.get("KAGGLE_API_TOKEN") or "").strip()
    configured = bool(
        token
        or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
        or cred_file.exists()
        or token_file.exists()
    )
    return {
        "configured": configured,
        "hint": (
            "Connecté."
            if configured
            else "Créez un token sur kaggle.com/settings/api, puis placez KAGGLE_API_TOKEN dans .env."
        ),
    }


def _catalog_items(slug: str) -> list[dict[str, Any]]:
    items = []
    for entry in CATALOG:
        if slug not in entry["slugs"]:
            continue
        items.append(
            {
                "ref": entry["ref"],
                "title": entry["title"],
                "why": entry["why"],
                "target_column": entry["target"],
                "verified": True,
                "url": f"https://www.kaggle.com/datasets/{entry['ref']}",
            }
        )
    return items


def list_compatible_datasets(algorithm: SupervisedAlgorithm) -> dict[str, Any]:
    status = kaggle_status()
    return {
        "slug": algorithm.slug,
        "kind": algorithm.kind,
        "configured": status["configured"],
        "hint": status["hint"],
        "datasets": _catalog_items(algorithm.slug),
    }


def parse_dataset_ref(raw: str) -> str:
    text = (raw or "").strip()
    match = DATASET_REF.search(text.replace("datasets/", ""))
    if not match:
        raise HTTPException(
            status_code=400,
            detail="Référence Kaggle invalide. Utilisez owner/dataset ou une URL kaggle.com/datasets/...",
        )
    return f"{match.group(1)}/{match.group(2)}"


def _api():
    load_local_env()
    if not kaggle_status()["configured"]:
        raise HTTPException(
            status_code=503,
            detail=kaggle_status()["hint"],
        )
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Le paquet kaggle n'est pas installé. Exécutez pip install kaggle.",
        ) from exc
    api = KaggleApi()
    try:
        api.authenticate()
    except SystemExit as exc:
        raise HTTPException(
            status_code=503,
            detail="Authentification Kaggle refusée. Vérifiez le token API.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail="Authentification Kaggle refusée. Vérifiez le token API.",
        ) from exc
    return api


def download_dataset(ref: str) -> Path:
    api = _api()
    folder = Path(tempfile.mkdtemp(prefix="kaggle-"))
    try:
        api.dataset_download_files(ref, path=str(folder), unzip=True, quiet=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"Téléchargement Kaggle impossible pour {ref}: {exc}",
        ) from exc
    return folder


def find_tabular_file(folder: Path) -> Path:
    candidates = [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".txt"}
    ]
    if not candidates:
        raise HTTPException(
            status_code=422,
            detail=(
                "Aucun tableau CSV/TSV dans ce dataset. "
                "Les algorithmes d'ici n'acceptent pas les images, l'audio ou le texte brut."
            ),
        )
    ranked = sorted(
        candidates,
        key=lambda path: (
            0 if "train" in path.name.lower() or "data" in path.name.lower() else 1,
            -path.stat().st_size,
        ),
    )
    chosen = ranked[0]
    if chosen.stat().st_size > MAX_CSV_BYTES:
        raise HTTPException(status_code=422, detail="Le fichier tabulaire dépasse 20 Mo.")
    return chosen


def read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        frame = pd.read_csv(path, sep=sep)
    except Exception:
        frame = pd.read_csv(path, sep=None, engine="python")
    if frame.empty:
        raise HTTPException(status_code=422, detail="Le fichier Kaggle est vide.")
    frame = frame.loc[:, ~frame.columns.astype(str).str.match(r"^Unnamed")]
    if len(frame) > MAX_TRAIN_ROWS:
        frame = frame.sample(n=MAX_TRAIN_ROWS, random_state=42)
    return frame.reset_index(drop=True)


def numeric_feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    usable: list[str] = []
    for column in frame.columns:
        if column == target or ID_NAME.match(str(column).strip()):
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().mean() >= 0.8:
            usable.append(str(column))
    return usable


def guess_target_column(frame: pd.DataFrame, explicit: str | None) -> str:
    if explicit:
        lookup = {normalize_name(column): str(column) for column in frame.columns}
        column = lookup.get(normalize_name(explicit))
        if column is None:
            raise HTTPException(
                status_code=422,
                detail=f"Colonne cible introuvable : {explicit}. Colonnes : {', '.join(map(str, frame.columns))}",
            )
        return column
    lookup = {normalize_name(column): str(column) for column in frame.columns}
    for hint in TARGET_HINTS:
        column = lookup.get(normalize_name(hint))
        if column:
            return column
    return str(frame.columns[-1])


def target_kind(series: pd.Series) -> str:
    numeric = pd.to_numeric(series, errors="coerce")
    nunique = series.dropna().astype(str).str.strip().nunique()
    if numeric.notna().mean() >= 0.9 and (numeric.nunique(dropna=True) > 12 or pd.api.types.is_float_dtype(numeric)):
        if nunique > 12:
            return "continuous"
    if nunique == 2:
        return "binary"
    if 3 <= nunique <= 20:
        return "multiclass"
    if nunique > 20 and numeric.notna().mean() >= 0.9:
        return "continuous"
    return "unsuitable"


def required_task(kind: str) -> str:
    return {
        "regression": "continuous",
        "classification-binaire": "binary",
        "classification-multiclasse": "multiclass",
        "classification-proximite": "classes",
        "classification-ensemble": "classes",
    }.get(kind, "classes")


def validate_frame(
    frame: pd.DataFrame,
    algorithm: SupervisedAlgorithm,
    target_column: str | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    target = guess_target_column(frame, target_column)
    kind = target_kind(frame[target])
    features = numeric_feature_columns(frame, target)
    n_rows = int(len(frame))
    needed = required_task(algorithm.kind)

    if n_rows < 12:
        reasons.append(f"Seulement {n_rows} lignes. Il en faut au moins 12.")
    if len(features) < 1:
        reasons.append("Aucune caractéristique numérique exploitable (hors id / texte).")
    if kind == "unsuitable":
        reasons.append(
            f"La colonne cible « {target} » n'est ni un nombre continu ni un ensemble raisonnable de classes."
        )

    if algorithm.kind == "regression":
        if kind != "continuous":
            reasons.append(
                f"{algorithm.name} prédit une valeur numérique. La cible « {target} » est {kind}, pas continue."
            )
    elif algorithm.kind == "classification-binaire":
        if kind != "binary":
            reasons.append(
                f"{algorithm.name} exige deux classes. La cible « {target} » est {kind}."
            )
    else:
        if kind not in {"binary", "multiclass"}:
            reasons.append(
                f"{algorithm.name} est une classification. La cible « {target} » est {kind}, pas des classes."
            )
        if kind == "binary" and algorithm.kind == "classification-multiclasse":
            # binary still works for a tree; allow it
            pass

    if algorithm.slug == "knn" and len(features) == 0:
        reasons.append("KNN a besoin de distances : colonnes numériques uniquement.")

    if n_rows and len(features) > n_rows:
        reasons.append("Plus de caractéristiques que de lignes — le dataset ne convient pas.")

    inspection = {
        "rows": n_rows,
        "target": target,
        "target_kind": kind,
        "numeric_features": features,
        "columns": [str(column) for column in frame.columns],
        "required_task": needed,
    }
    if reasons:
        raise HTTPException(
            status_code=422,
            detail={
                "accepted": False,
                "reasons": reasons,
                "inspection": inspection,
            },
        )
    return inspection


def prepare_xy(frame: pd.DataFrame, inspection: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    target = inspection["target"]
    feature_names = inspection["numeric_features"]
    X = frame[feature_names].apply(pd.to_numeric, errors="coerce")
    y = frame[target]
    if target_kind(y) == "continuous":
        y = pd.to_numeric(y, errors="coerce")
    else:
        y = y.astype(str).str.strip()
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    if len(X) < 12:
        raise HTTPException(
            status_code=422,
            detail="Trop de valeurs manquantes après nettoyage (moins de 12 lignes valides).",
        )
    return X, y


def import_dataset(
    algorithm: SupervisedAlgorithm,
    dataset: str,
    target_column: str | None = None,
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ref = parse_dataset_ref(dataset)
    folder = download_dataset(ref)
    table = find_tabular_file(folder)
    frame = read_table(table)
    inspection = validate_frame(frame, algorithm, target_column)
    X, y = prepare_xy(frame, inspection)
    result = fit_custom(algorithm, X, y, **(hyperparameters or {}))
    algorithm.custom_training = {"X": X, "y": y}
    algorithm.dataset_source = f"kaggle:{ref}"
    return json_safe(
        {
            "accepted": True,
            "dataset": ref,
            "file": table.name,
            "inspection": {
                **inspection,
                "rows_used": int(len(X)),
                "features_used": list(X.columns),
            },
            "train": result,
            "feature_names": algorithm.feature_names,
            "sample_input": algorithm.sample_input,
            "target_name": algorithm.target_name,
            "class_names": algorithm.class_names,
            "metrics": algorithm.metrics,
            "source": algorithm.dataset_source,
        }
    )
