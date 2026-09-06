"""Parse uploaded CSVs and map columns onto an algorithm's features."""

from __future__ import annotations

import io
import re
from typing import Any

import pandas as pd
from fastapi import HTTPException
from fastapi.responses import Response

from backend.json_safe import json_safe
from backend.ml.base import SupervisedAlgorithm

MAX_ROWS = 50
MAX_BYTES = 1_000_000
TARGET_ALIASES = {
    "y",
    "target",
    "label",
    "classe",
    "class",
    "espece",
    "espèce",
    "species",
    "prix",
    "reussite",
    "réussite",
    "categorie",
    "catégorie",
    "diagnostic",
}


def normalize_name(name: str) -> str:
    text = str(name).strip().lower().replace("(cm)", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def read_csv_bytes(raw: bytes, filename: str) -> pd.DataFrame:
    if not raw:
        raise HTTPException(status_code=400, detail="Le fichier CSV est vide.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="Le CSV dépasse 1 Mo.")
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"Impossible de lire {filename}: {exc}",
        ) from exc
    if frame.empty:
        raise HTTPException(status_code=400, detail="Le CSV ne contient aucune ligne.")
    if len(frame) > MAX_ROWS:
        frame = frame.head(MAX_ROWS)
    return frame


def map_feature_columns(
    frame: pd.DataFrame, feature_names: list[str]
) -> dict[str, str]:
    lookup = {normalize_name(column): str(column) for column in frame.columns}
    mapping: dict[str, str] = {}
    missing: list[str] = []
    for feature in feature_names:
        column = lookup.get(normalize_name(feature))
        if column is None:
            missing.append(feature)
        else:
            mapping[feature] = column
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                "Colonnes manquantes dans le CSV : "
                + ", ".join(missing)
                + ". Colonnes attendues : "
                + ", ".join(feature_names)
            ),
        )
    return mapping


def find_target_column(frame: pd.DataFrame, target_name: str) -> str | None:
    lookup = {normalize_name(column): str(column) for column in frame.columns}
    candidates = [target_name, *TARGET_ALIASES]
    for candidate in candidates:
        column = lookup.get(normalize_name(candidate))
        if column:
            return column
    return None


def summarize_prediction(prediction: dict[str, Any]) -> str:
    for key in (
        "predicted_price",
        "result",
        "predicted_species",
        "predicted_category",
        "predicted_label",
    ):
        if prediction.get(key) is not None:
            value = prediction[key]
            unit = prediction.get("unit", "")
            return f"{value} {unit}".strip()
    return str(prediction.get("predicted_class", "—"))


def predict_csv(
    algorithm: SupervisedAlgorithm,
    raw: bytes,
    filename: str,
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if hyperparameters:
        algorithm.train(**hyperparameters)
    elif not algorithm.trained:
        algorithm.train()

    frame = read_csv_bytes(raw, filename)
    mapping = map_feature_columns(frame, algorithm.feature_names)
    target_column = find_target_column(frame, algorithm.target_name)

    rows: list[dict[str, Any]] = []
    matches = 0
    labeled = 0
    for index, record in frame.iterrows():
        features = {
            feature: float(record[column])
            for feature, column in mapping.items()
        }
        prediction = algorithm.predict(features)
        actual = None
        if target_column is not None and pd.notna(record[target_column]):
            actual = record[target_column]
            labeled += 1
            predicted = summarize_prediction(prediction)
            if str(actual).strip().lower() == str(predicted).strip().lower():
                matches += 1
            elif str(actual) == str(prediction.get("predicted_class")):
                matches += 1
        rows.append(
            {
                "row": int(index) + 1,
                "input": features,
                "prediction": prediction,
                "summary": summarize_prediction(prediction),
                "actual": None if actual is None else json_safe(actual),
            }
        )

    payload: dict[str, Any] = {
        "filename": filename,
        "row_count": len(rows),
        "columns_used": algorithm.feature_names,
        "truncated": len(frame) == MAX_ROWS,
        "rows": rows,
    }
    if labeled:
        payload["label_matches"] = matches
        payload["labeled_rows"] = labeled
        payload["match_rate"] = round(matches / labeled, 3)
    return json_safe(payload)


def sample_frame(algorithm: SupervisedAlgorithm) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if algorithm.sample_input:
        rows.append(dict(algorithm.sample_input))
    exercise_input = algorithm.exercise.get("input")
    if exercise_input:
        rows.append(dict(exercise_input))
    if len(rows) == 1:
        tweaked = dict(rows[0])
        for key, value in list(tweaked.items())[:2]:
            try:
                tweaked[key] = round(float(value) * 1.1, 3)
            except (TypeError, ValueError):
                continue
        rows.append(tweaked)
    frame = pd.DataFrame(rows)
    return frame[algorithm.feature_names]


def sample_csv_response(algorithm: SupervisedAlgorithm) -> Response:
    buffer = io.StringIO()
    sample_frame(algorithm).to_csv(buffer, index=False)
    filename = f"{algorithm.slug}-exemple.csv"
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
