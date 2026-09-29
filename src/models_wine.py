from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE


def crear_modelos_wine():
    """
    Modelos de clasificacion evaluados sobre Red Wine.

    RNA:
    MLPClassifier representa una red neuronal artificial
    de tipo perceptron multicapa (MLP), con arquitectura
    feedforward y entrenamiento mediante backpropagation.
    """

    return {
        "regresion_logistica": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "arbol_decision": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "tree",
                    DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_split=10,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "svc_rbf": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "mlp_classifier": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(
                            32,
                            16,
                        ),
                        activation="relu",
                        solver="lbfgs",
                        alpha=0.001,
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
