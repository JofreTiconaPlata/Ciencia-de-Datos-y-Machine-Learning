import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE,
)
from src.data_processing import limpiar_y_documentar
from src.evaluation import (
    comparacion_train_test,
    dataframe_metricas,
    metricas_regresion,
    validacion_cruzada_regresion,
)
from src.models_candy import crear_modelos_candy


CANDY_FEATURES = [
    "chocolate",
    "fruity",
    "caramel",
    "peanutyalmondy",
    "nougat",
    "crispedricewafer",
    "hard",
    "bar",
    "pluribus",
    "sugarpercent",
    "pricepercent",
]


METRICAS_CANDY = [
    "MAE",
    "MSE",
    "RMSE",
    "R2",
    "MedAE",
]


def ejecutar_candy():

    print(
        "\n=== DATASET CANDY ==="
    )

    df = pd.read_csv(
        DATA_DIR / "candy-data.csv"
    )

    print(
        "Dimensiones originales:",
        df.shape,
    )

    print(
        "Faltantes:",
        int(df.isna().sum().sum()),
    )

    print(
        "Duplicados:",
        int(df.duplicated().sum()),
    )

    clean = limpiar_y_documentar(
        df,
        nombre="candy",
        variables_iqr=CANDY_FEATURES,
    )

    clean.to_csv(
        RESULTS_DIR
        / "candy_clean.csv",
        index=False,
    )

    X = clean[CANDY_FEATURES]
    y = clean["winpercent"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    modelos = crear_modelos_candy()

    filas_comparacion = []
    detalle_cv = []

    modelos_entrenados = {}
    predicciones_test = {}
    metricas_train_modelos = {}
    metricas_test_modelos = {}
    resumenes_cv = {}
    folds_cv = {}

    for nombre, modelo_base in modelos.items():

        print(
            f"\nEvaluando Candy: {nombre}"
        )

        # ====================================================
        # CROSS-VALIDATION EXCLUSIVAMENTE SOBRE TRAIN
        # ====================================================

        cv_folds, cv_resumen = (
            validacion_cruzada_regresion(
                modelo_base,
                X_train,
                y_train,
            )
        )

        cv_folds.insert(
            0,
            "modelo",
            nombre,
        )

        detalle_cv.append(
            cv_folds
        )

        resumenes_cv[nombre] = (
            cv_resumen.copy()
        )

        folds_cv[nombre] = (
            cv_folds.copy()
        )

        # ====================================================
        # ENTRENAMIENTO HOLDOUT
        # ====================================================

        modelo = clone(
            modelo_base
        )

        modelo.fit(
            X_train,
            y_train,
        )

        pred_train = modelo.predict(
            X_train
        )

        pred_test = modelo.predict(
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

        modelos_entrenados[nombre] = (
            modelo
        )

        predicciones_test[nombre] = (
            pred_test
        )

        metricas_train_modelos[nombre] = (
            train_metrics
        )

        metricas_test_modelos[nombre] = (
            test_metrics
        )

        cv_index = (
            cv_resumen
            .set_index("metrica")
        )

        fila = {
            "modelo": nombre,
        }

        for metrica in METRICAS_CANDY:

            fila[
                f"{metrica}_train"
            ] = train_metrics[
                metrica
            ]

            fila[
                f"{metrica}_test"
            ] = test_metrics[
                metrica
            ]

            fila[
                f"{metrica}_cv_media"
            ] = cv_index.loc[
                metrica,
                "media",
            ]

            fila[
                f"{metrica}_cv_std"
            ] = cv_index.loc[
                metrica,
                "desviacion_estandar",
            ]

        filas_comparacion.append(
            fila
        )

    # ========================================================
    # TABLAS COMPARATIVAS
    # ========================================================

    comparacion_modelos = pd.DataFrame(
        filas_comparacion
    )

    comparacion_modelos.to_csv(
        RESULTS_DIR
        / "candy_modelos_comparacion.csv",
        index=False,
    )

    pd.concat(
        detalle_cv,
        ignore_index=True,
    ).to_csv(
        RESULTS_DIR
        / "candy_modelos_cv_detalle.csv",
        index=False,
    )

    # ========================================================
    # SELECCION DEL MODELO
    #
    # Candy:
    # minimizar RMSE medio de validacion cruzada.
    #
    # TEST NO interviene en la seleccion.
    # ========================================================

    indice_mejor = (
        comparacion_modelos[
            "RMSE_cv_media"
        ].idxmin()
    )

    mejor_modelo = (
        comparacion_modelos.loc[
            indice_mejor,
            "modelo",
        ]
    )

    mejor_rmse_cv = (
        comparacion_modelos.loc[
            indice_mejor,
            "RMSE_cv_media",
        ]
    )

    pd.DataFrame(
        {
            "modelo": [
                mejor_modelo
            ],
            "criterio": [
                "Menor RMSE medio en CV sobre TRAIN"
            ],
            "valor_cv": [
                mejor_rmse_cv
            ],
        }
    ).to_csv(
        RESULTS_DIR
        / "candy_modelo_seleccionado.csv",
        index=False,
    )

    # ========================================================
    # CONSERVAR RESULTADOS HISTORICOS DE REGRESION LINEAL
    # ========================================================

    baseline = "regresion_lineal"

    df_train = dataframe_metricas(
        metricas_train_modelos[
            baseline
        ]
    )

    df_test = dataframe_metricas(
        metricas_test_modelos[
            baseline
        ]
    )

    comparacion_train_test(
        metricas_train_modelos[
            baseline
        ],
        metricas_test_modelos[
            baseline
        ],
    ).to_csv(
        RESULTS_DIR
        / "candy_comparacion_train_test.csv",
        index=False,
    )

    df_train.to_csv(
        RESULTS_DIR
        / "candy_metricas_train.csv",
        index=False,
    )

    df_test.to_csv(
        RESULTS_DIR
        / "candy_metricas.csv",
        index=False,
    )

    folds_baseline = (
        folds_cv[baseline]
        .drop(
            columns=["modelo"]
        )
    )

    folds_baseline.to_csv(
        RESULTS_DIR
        / "candy_validacion_cruzada.csv",
        index=False,
    )

    resumenes_cv[
        baseline
    ].to_csv(
        RESULTS_DIR
        / "candy_validacion_cruzada_resumen.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "real": y_test.values,
            "predicho": (
                predicciones_test[
                    baseline
                ]
            ),
        }
    ).to_csv(
        RESULTS_DIR
        / "candy_predicciones.csv",
        index=False,
    )

    # ========================================================
    # GRAFICO BASELINE
    # ========================================================

    pred_lineal = (
        predicciones_test[
            baseline
        ]
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        y_test,
        pred_lineal,
    )

    minimo = min(
        y_test.min(),
        pred_lineal.min(),
    )

    maximo = max(
        y_test.max(),
        pred_lineal.max(),
    )

    plt.plot(
        [minimo, maximo],
        [minimo, maximo],
    )

    plt.xlabel(
        "Winpercent real"
    )

    plt.ylabel(
        "Winpercent predicho"
    )

    plt.title(
        "Candy - Regresión lineal: real vs predicho"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "candy_real_vs_predicho.png",
        dpi=150,
    )

    plt.close()

    # ========================================================
    # CONSOLA
    # ========================================================

    print(
        "\nCandy - Comparación de modelos"
    )

    columnas = [
        "modelo",
        "RMSE_cv_media",
        "RMSE_cv_std",
        "R2_cv_media",
        "RMSE_test",
        "R2_test",
    ]

    print(
        comparacion_modelos[
            columnas
        ].to_string(
            index=False
        )
    )

    print(
        "\nModelo seleccionado por CV:"
    )

    print(
        f"  {mejor_modelo}"
    )

    print(
        f"  RMSE CV = "
        f"{mejor_rmse_cv:.6f}"
    )

    return {
        "modelos": modelos_entrenados,
        "comparacion": comparacion_modelos,
        "modelo_seleccionado": mejor_modelo,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
