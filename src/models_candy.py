from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.config import RANDOM_STATE


def crear_modelos_candy():
    """
    Modelos de regresion evaluados sobre Candy.

    RNA:
    MLPRegressor representa una red neuronal artificial
    de tipo perceptron multicapa (MLP), con arquitectura
    feedforward y entrenamiento mediante backpropagation.
    """

    return {
        "regresion_lineal": Pipeline(
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
                    LinearRegression(),
                ),
            ]
        ),

        "arbol_regresion": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=5,
                        min_samples_split=5,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "svr_rbf": Pipeline(
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
                    SVR(
                        kernel="rbf",
                        C=1.0,
                        epsilon=0.1,
                        gamma="scale",
                    ),
                ),
            ]
        ),

        "mlp_regressor": Pipeline(
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
                    MLPRegressor(
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
