from backend.ml.arbre_decision import ArbreDecision
from backend.ml.foret_aleatoire import ForetAleatoire
from backend.ml.knn import Knn
from backend.ml.regression_lineaire import RegressionLineaire
from backend.ml.regression_logistique import RegressionLogistique
from backend.ml.svm import Svm

ALGORITHMS = {
    "regression-lineaire": RegressionLineaire,
    "regression-logistique": RegressionLogistique,
    "arbre-decision": ArbreDecision,
    "knn": Knn,
    "foret-aleatoire": ForetAleatoire,
    "svm": Svm,
}

__all__ = ["ALGORITHMS"]
