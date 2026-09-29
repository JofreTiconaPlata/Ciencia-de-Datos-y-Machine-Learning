import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
)

from src.config import RANDOM_STATE


# ============================================================
# METRICAS
# ============================================================

def metricas_regresion(y_true, y_pred):
    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    return {
        "MAE": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(
            y_true,
            y_pred,
        ),
        "MedAE": median_absolute_error(
            y_true,
            y_pred,
        ),
    }


def metricas_clasificacion(
    y_true,
    y_pred,
    y_score,
):
    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "F1-score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Balanced Accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "ROC-AUC": roc_auc_score(
            y_true,
            y_score,
        ),
    }


def dataframe_metricas(metricas):
    return pd.DataFrame(
        {
            "metrica": list(
                metricas.keys()
            ),
            "valor": list(
                metricas.values()
            ),
        }
    )


def comparacion_train_test(
    metricas_train,
    metricas_test,
):
    nombres = list(
        metricas_test.keys()
    )

    return pd.DataFrame(
        {
            "metrica": nombres,
            "train": [
                metricas_train[m]
                for m in nombres
            ],
            "test": [
                metricas_test[m]
                for m in nombres
            ],
        }
    )


# ============================================================
# SCORE DE CLASE POSITIVA
# ============================================================

def obtener_score_positivo(
    modelo,
    X,
):
    if hasattr(
        modelo,
        "predict_proba",
    ):
        probabilidades = (
            modelo.predict_proba(X)
        )

        return probabilidades[:, 1]

    if hasattr(
        modelo,
        "decision_function",
    ):
        return modelo.decision_function(
            X
        )

    # Fallback únicamente para modelos sin score continuo.
    return modelo.predict(X)


# ============================================================
# VALIDACION CRUZADA - REGRESION
#
# IMPORTANTE:
# Recibe exclusivamente X_train / y_train.
# El conjunto TEST queda fuera de la seleccion y validacion.
# ============================================================

def validacion_cruzada_regresion(
    modelo,
    X_train,
    y_train,
):
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "MAE": make_scorer(
            mean_absolute_error,
            greater_is_better=False,
        ),
        "MSE": make_scorer(
            mean_squared_error,
            greater_is_better=False,
        ),
        "R2": "r2",
        "MedAE": make_scorer(
            median_absolute_error,
            greater_is_better=False,
        ),
    }

    resultados = cross_validate(
        modelo,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1,
    )

    mae = -resultados["test_MAE"]
    mse = -resultados["test_MSE"]
    rmse = np.sqrt(mse)
    r2 = resultados["test_R2"]
    medae = -resultados["test_MedAE"]

    folds = pd.DataFrame(
        {
            "fold": np.arange(
                1,
                len(mae) + 1,
            ),
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "MedAE": medae,
        }
    )

    resumen = pd.DataFrame(
        {
            "metrica": [
                "MAE",
                "MSE",
                "RMSE",
                "R2",
                "MedAE",
            ],
            "media": [
                mae.mean(),
                mse.mean(),
                rmse.mean(),
                r2.mean(),
                medae.mean(),
            ],
            "desviacion_estandar": [
                mae.std(ddof=1),
                mse.std(ddof=1),
                rmse.std(ddof=1),
                r2.std(ddof=1),
                medae.std(ddof=1),
            ],
        }
    )

    return folds, resumen


# ============================================================
# VALIDACION CRUZADA - CLASIFICACION
#
# StratifiedKFold conserva aproximadamente la proporcion
# de clases en cada particion.
# ============================================================

def validacion_cruzada_clasificacion(
    modelo,
    X_train,
    y_train,
):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "Accuracy": "accuracy",
        "Precision": make_scorer(
            precision_score,
            zero_division=0,
        ),
        "Recall": make_scorer(
            recall_score,
            zero_division=0,
        ),
        "F1-score": make_scorer(
            f1_score,
            zero_division=0,
        ),
        "Balanced Accuracy": (
            "balanced_accuracy"
        ),
        "ROC-AUC": "roc_auc",
    }

    resultados = cross_validate(
        modelo,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1,
    )

    nombres = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Balanced Accuracy",
        "ROC-AUC",
    ]

    valores = {
        nombre: resultados[
            f"test_{nombre}"
        ]
        for nombre in nombres
    }

    folds = pd.DataFrame(
        {
            "fold": np.arange(
                1,
                6,
            ),
            **valores,
        }
    )

    resumen = pd.DataFrame(
        {
            "metrica": nombres,
            "media": [
                valores[m].mean()
                for m in nombres
            ],
            "desviacion_estandar": [
                valores[m].std(ddof=1)
                for m in nombres
            ],
        }
    )

    return folds, resumen
