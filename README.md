# Algo Lab

Discord-style lab for the six supervised-learning algorithms in the assignment: linear regression, logistic regression, decision tree, KNN, random forest, and SVM.

The backend trains each model on the same datasets as the course document, then the UI lets you pick an algorithm, predict a new observation, run the “À vous de jouer” exercise, or upload a CSV.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m backend
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Algorithms

| Channel | Type | Predicts |
|---|---|---|
| `# regression-lineaire` | Regression | House price from surface and rooms |
| `# regression-logistique` | Binary classification | Exam pass/fail |
| `# arbre-decision` | Multiclass | Iris species |
| `# knn` | Neighbors | Iris species |
| `# foret-aleatoire` | Ensemble | Wine class |
| `# svm` | Binary classification | Tumor label (demo only, not medical advice) |

## CSV samples

Use **CSV** in the composer, or download a matching template from the `+` menu.

| File | Algorithm |
|---|---|
| `samples/maisons.csv` | Linear regression |
| `samples/examens.csv` | Logistic regression |
| `samples/iris.csv` | Decision tree / KNN |
| `samples/vins.csv` | Random forest |
| `samples/tumeurs.csv` | SVM |

Column names can use spaces or underscores (`sepal length (cm)` and `sepal_length` both work).

## Layout

```
backend/     FastAPI + scikit-learn models
frontend/     Discord-style chat UI
algorithms/   CLI scripts from the assignment
samples/      Test CSVs
```
