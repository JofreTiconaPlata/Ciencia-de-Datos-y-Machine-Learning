from __future__ import annotations

import json
import warnings
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from src.candy import CANDY_FEATURES
from src.config import (
    DATA_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE,
)
from src.data_processing import limpiar_y_documentar
from src.evaluation import (
    metricas_clasificacion,
    metricas_regresion,
    obtener_score_positivo,
)
from src.models_candy import crear_modelos_candy
from src.models_wine import crear_modelos_wine
from src.wine import WINE_FEATURES


warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ============================================================
# DATOS
# ============================================================

def preparar_candy():
    df = pd.read_csv(
        DATA_DIR / "candy-data.csv"
    )

    clean = limpiar_y_documentar(
        df,
        nombre="candy",
        variables_iqr=CANDY_FEATURES,
    )

    X = clean[CANDY_FEATURES]
    y = clean["winpercent"]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def preparar_wine():
    df = pd.read_csv(
        DATA_DIR / "winequality-red.csv"
    )

    clean = limpiar_y_documentar(
        df,
        nombre="redwine",
        variables_iqr=WINE_FEATURES,
    )

    clean["quality_class"] = (
        clean["quality"] >= 7
    ).astype(int)

    X = clean[WINE_FEATURES]
    y = clean["quality_class"]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# ============================================================
# CANDY
# ============================================================

def ajustar_candy():

    print()
    print("=" * 64)
    print(" TUNING CANDY")
    print("=" * 64)

    X_train, X_test, y_train, y_test = (
        preparar_candy()
    )

    modelos = crear_modelos_candy()

    grids = {
        "regresion_lineal": None,

        "arbol_regresion": {
            "model__max_depth": [
                3,
                5,
                None,
            ],
            "model__min_samples_leaf": [
                1,
                3,
            ],
        },

        "svr_rbf": {
            "model__C": [
                1.0,
                10.0,
            ],
            "model__epsilon": [
                0.1,
                0.5,
            ],
        },

        "mlp_regressor": {
            "model__hidden_layer_sizes": [
                (16,),
                (32, 16),
            ],
            "model__alpha": [
                0.001,
                0.01,
            ],
        },
    }

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    filas = []
    mejores = {}

    for nombre, modelo in modelos.items():

        print(
            f"Ajustando: {nombre}"
        )

        grid = grids[nombre]

        if grid is None:

            scores = cross_val_score(
                modelo,
                X_train,
                y_train,
                cv=cv,
                scoring=(
                    "neg_root_mean_squared_error"
                ),
                n_jobs=-1,
            )

            rmse_scores = -scores

            modelo_final = clone(
                modelo
            )

            modelo_final.fit(
                X_train,
                y_train,
            )

            media = float(
                rmse_scores.mean()
            )

            std = float(
                rmse_scores.std(
                    ddof=1
                )
            )

            params = {}

        else:

            search = GridSearchCV(
                estimator=modelo,
                param_grid=grid,
                scoring=(
                    "neg_root_mean_squared_error"
                ),
                cv=cv,
                n_jobs=-1,
                refit=True,
                return_train_score=False,
            )

            search.fit(
                X_train,
                y_train,
            )

            modelo_final = (
                search.best_estimator_
            )

            media = float(
                -search.best_score_
            )

            std = float(
                search.cv_results_[
                    "std_test_score"
                ][
                    search.best_index_
                ]
            )

            params = (
                search.best_params_
            )

        mejores[nombre] = (
            modelo_final
        )

        filas.append(
            {
                "modelo": nombre,
                "RMSE_cv_media": media,
                "RMSE_cv_std": std,
                "mejores_parametros": json.dumps(
                    params,
                    ensure_ascii=False,
                ),
            }
        )

    tabla = pd.DataFrame(
        filas
    ).sort_values(
        "RMSE_cv_media"
    )

    tabla.to_csv(
        RESULTS_DIR
        / "candy_tuning.csv",
        index=False,
    )

    mejor_nombre = (
        tabla.iloc[0]["modelo"]
    )

    mejor_modelo = (
        mejores[mejor_nombre]
    )

    # TEST se usa recién después de seleccionar por CV.
    pred_train = mejor_modelo.predict(
        X_train
    )

    pred_test = mejor_modelo.predict(
        X_test
    )

    train_metrics = (
        metricas_regresion(
            y_train,
            pred_train,
        )
    )

    test_metrics = (
        metricas_regresion(
            y_test,
            pred_test,
        )
    )

    mejor_cv = float(
        tabla.iloc[0][
            "RMSE_cv_media"
        ]
    )

    diagnostico = pd.DataFrame(
        [
            {
                "modelo": mejor_nombre,
                "RMSE_train": (
                    train_metrics["RMSE"]
                ),
                "RMSE_cv_media": mejor_cv,
                "RMSE_test": (
                    test_metrics["RMSE"]
                ),
                "gap_cv_train": (
                    mejor_cv
                    - train_metrics["RMSE"]
                ),
                "gap_test_cv": (
                    test_metrics["RMSE"]
                    - mejor_cv
                ),
                "R2_train": (
                    train_metrics["R2"]
                ),
                "R2_test": (
                    test_metrics["R2"]
                ),
            }
        ]
    )

    diagnostico.to_csv(
        RESULTS_DIR
        / "candy_diagnostico_final.csv",
        index=False,
    )

    joblib.dump(
        mejor_modelo,
        MODELS_DIR
        / "candy_final.joblib",
    )

    metadata = {
        "dataset": "Candy",
        "target": "winpercent",
        "modelo": mejor_nombre,
        "criterio_seleccion": (
            "Menor RMSE medio en "
            "validacion cruzada sobre TRAIN"
        ),
        "cv_rmse": mejor_cv,
        "features": CANDY_FEATURES,
        "metricas_test": {
            k: float(v)
            for k, v
            in test_metrics.items()
        },
        "python_random_state": (
            RANDOM_STATE
        ),
    }

    (
        MODELS_DIR
        / "candy_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(tabla.to_string(index=False))

    print()
    print(
        "Modelo final Candy:",
        mejor_nombre,
    )

    print(
        "RMSE CV:",
        f"{mejor_cv:.6f}",
    )

    print(
        "RMSE TEST:",
        f"{test_metrics['RMSE']:.6f}",
    )

    return {
        "modelo": mejor_nombre,
        "pipeline": mejor_modelo,
        "tabla": tabla,
        "test": test_metrics,
    }


# ============================================================
# RED WINE
# ============================================================

def ajustar_wine():

    print()
    print("=" * 64)
    print(" TUNING RED WINE")
    print("=" * 64)

    X_train, X_test, y_train, y_test = (
        preparar_wine()
    )

    modelos = crear_modelos_wine()

    grids = {
        "regresion_logistica": {
            "model__C": [
                0.1,
                1.0,
                10.0,
            ],
            "model__class_weight": [
                None,
                "balanced",
            ],
        },

        "arbol_decision": {
            "tree__max_depth": [
                3,
                5,
                7,
                None,
            ],
            "tree__class_weight": [
                None,
                "balanced",
            ],
        },

        "svc_rbf": {
            "model__C": [
                1.0,
                5.0,
                10.0,
            ],
            "model__class_weight": [
                None,
                "balanced",
            ],
        },

        "mlp_classifier": {
            "model__hidden_layer_sizes": [
                (16,),
                (32, 16),
            ],
            "model__alpha": [
                0.001,
                0.01,
            ],
        },
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    filas = []
    mejores = {}

    for nombre, modelo in modelos.items():

        print(
            f"Ajustando: {nombre}"
        )

        search = GridSearchCV(
            estimator=modelo,
            param_grid=grids[nombre],
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )

        search.fit(
            X_train,
            y_train,
        )

        modelo_final = (
            search.best_estimator_
        )

        media = float(
            search.best_score_
        )

        std = float(
            search.cv_results_[
                "std_test_score"
            ][
                search.best_index_
            ]
        )

        mejores[nombre] = (
            modelo_final
        )

        filas.append(
            {
                "modelo": nombre,
                "F1_cv_media": media,
                "F1_cv_std": std,
                "mejores_parametros": json.dumps(
                    search.best_params_,
                    ensure_ascii=False,
                ),
            }
        )

    tabla = pd.DataFrame(
        filas
    ).sort_values(
        "F1_cv_media",
        ascending=False,
    )

    tabla.to_csv(
        RESULTS_DIR
        / "redwine_tuning.csv",
        index=False,
    )

    mejor_nombre = (
        tabla.iloc[0]["modelo"]
    )

    mejor_modelo = (
        mejores[mejor_nombre]
    )

    # TEST recién después de selección por CV.
    pred_train = mejor_modelo.predict(
        X_train
    )

    pred_test = mejor_modelo.predict(
        X_test
    )

    score_train = (
        obtener_score_positivo(
            mejor_modelo,
            X_train,
        )
    )

    score_test = (
        obtener_score_positivo(
            mejor_modelo,
            X_test,
        )
    )

    train_metrics = (
        metricas_clasificacion(
            y_train,
            pred_train,
            score_train,
        )
    )

    test_metrics = (
        metricas_clasificacion(
            y_test,
            pred_test,
            score_test,
        )
    )

    mejor_cv = float(
        tabla.iloc[0][
            "F1_cv_media"
        ]
    )

    diagnostico = pd.DataFrame(
        [
            {
                "modelo": mejor_nombre,
                "F1_train": (
                    train_metrics[
                        "F1-score"
                    ]
                ),
                "F1_cv_media": mejor_cv,
                "F1_test": (
                    test_metrics[
                        "F1-score"
                    ]
                ),
                "gap_train_cv": (
                    train_metrics[
                        "F1-score"
                    ]
                    - mejor_cv
                ),
                "gap_cv_test": (
                    mejor_cv
                    - test_metrics[
                        "F1-score"
                    ]
                ),
                "Balanced_Accuracy_test": (
                    test_metrics[
                        "Balanced Accuracy"
                    ]
                ),
                "ROC_AUC_test": (
                    test_metrics[
                        "ROC-AUC"
                    ]
                ),
            }
        ]
    )

    diagnostico.to_csv(
        RESULTS_DIR
        / "redwine_diagnostico_final.csv",
        index=False,
    )

    joblib.dump(
        mejor_modelo,
        MODELS_DIR
        / "wine_final.joblib",
    )

    metadata = {
        "dataset": "Red Wine",
        "target": "quality_class",
        "definicion_target": (
            "1 si quality >= 7; "
            "0 si quality < 7"
        ),
        "modelo": mejor_nombre,
        "criterio_seleccion": (
            "Mayor F1 medio en "
            "validacion cruzada "
            "estratificada sobre TRAIN"
        ),
        "cv_f1": mejor_cv,
        "features": WINE_FEATURES,
        "metricas_test": {
            k: float(v)
            for k, v
            in test_metrics.items()
        },
        "random_state": (
            RANDOM_STATE
        ),
    }

    (
        MODELS_DIR
        / "wine_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(tabla.to_string(index=False))

    print()
    print(
        "Modelo final Wine:",
        mejor_nombre,
    )

    print(
        "F1 CV:",
        f"{mejor_cv:.6f}",
    )

    print(
        "F1 TEST:",
        f"{test_metrics['F1-score']:.6f}",
    )

    return {
        "modelo": mejor_nombre,
        "pipeline": mejor_modelo,
        "tabla": tabla,
        "test": test_metrics,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    candy = ajustar_candy()
    wine = ajustar_wine()

    print()
    print("=" * 68)
    print(" TUNING Y PERSISTENCIA COMPLETADOS")
    print("=" * 68)

    print(
        "Candy final:",
        candy["modelo"],
    )

    print(
        "Wine final:",
        wine["modelo"],
    )

    print()
    print(
        "Modelos guardados en models/"
    )

    print(
        "TEST no intervino en "
        "la seleccion de modelos."
    )

    print("=" * 68)


if __name__ == "__main__":
    main()
