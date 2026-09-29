import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

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
    metricas_clasificacion,
    obtener_score_positivo,
    validacion_cruzada_clasificacion,
)
from src.models_wine import crear_modelos_wine


WINE_FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


METRICAS_WINE = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "Balanced Accuracy",
    "ROC-AUC",
]


def ejecutar_wine():

    print(
        "\n=== DATASET RED WINE ==="
    )

    df = pd.read_csv(
        DATA_DIR
        / "winequality-red.csv"
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
        nombre="redwine",
        variables_iqr=WINE_FEATURES,
    )

    clean.to_csv(
        RESULTS_DIR
        / "redwine_clean.csv",
        index=False,
    )

    clean["quality_class"] = (
        clean["quality"] >= 7
    ).astype(int)

    X = clean[WINE_FEATURES]
    y = clean["quality_class"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    modelos = crear_modelos_wine()

    filas_comparacion = []
    detalle_cv = []

    modelos_entrenados = {}
    predicciones_test = {}
    scores_test = {}
    metricas_train_modelos = {}
    metricas_test_modelos = {}
    resumenes_cv = {}
    folds_cv = {}

    for nombre, modelo_base in modelos.items():

        print(
            f"\nEvaluando Red Wine: {nombre}"
        )

        # ====================================================
        # CROSS-VALIDATION EXCLUSIVAMENTE SOBRE TRAIN
        # ====================================================

        cv_folds, cv_resumen = (
            validacion_cruzada_clasificacion(
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

        score_train = (
            obtener_score_positivo(
                modelo,
                X_train,
            )
        )

        score_test = (
            obtener_score_positivo(
                modelo,
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

        modelos_entrenados[nombre] = (
            modelo
        )

        predicciones_test[nombre] = (
            pred_test
        )

        scores_test[nombre] = (
            score_test
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

        for metrica in METRICAS_WINE:

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
    # TABLA COMPARATIVA
    # ========================================================

    comparacion_modelos = pd.DataFrame(
        filas_comparacion
    )

    comparacion_modelos.to_csv(
        RESULTS_DIR
        / "redwine_modelos_comparacion.csv",
        index=False,
    )

    pd.concat(
        detalle_cv,
        ignore_index=True,
    ).to_csv(
        RESULTS_DIR
        / "redwine_modelos_cv_detalle.csv",
        index=False,
    )

    # ========================================================
    # SELECCION DEL MODELO
    #
    # Debido al desbalance de clases no seleccionamos
    # por Accuracy.
    #
    # Criterio:
    # mayor F1-score medio en CV sobre TRAIN.
    #
    # TEST NO participa en la seleccion.
    # ========================================================

    indice_mejor = (
        comparacion_modelos[
            "F1-score_cv_media"
        ].idxmax()
    )

    mejor_modelo = (
        comparacion_modelos.loc[
            indice_mejor,
            "modelo",
        ]
    )

    mejor_f1_cv = (
        comparacion_modelos.loc[
            indice_mejor,
            "F1-score_cv_media",
        ]
    )

    pd.DataFrame(
        {
            "modelo": [
                mejor_modelo
            ],
            "criterio": [
                "Mayor F1-score medio en CV sobre TRAIN"
            ],
            "valor_cv": [
                mejor_f1_cv
            ],
        }
    ).to_csv(
        RESULTS_DIR
        / "redwine_modelo_seleccionado.csv",
        index=False,
    )

    # ========================================================
    # CONSERVAR RESULTADOS HISTORICOS DEL ARBOL
    # ========================================================

    baseline = "arbol_decision"

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
        / "redwine_comparacion_train_test.csv",
        index=False,
    )

    df_train.to_csv(
        RESULTS_DIR
        / "redwine_metricas_train.csv",
        index=False,
    )

    df_test.to_csv(
        RESULTS_DIR
        / "redwine_metricas.csv",
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
        / "redwine_validacion_cruzada.csv",
        index=False,
    )

    resumenes_cv[
        baseline
    ].to_csv(
        RESULTS_DIR
        / "redwine_validacion_cruzada_resumen.csv",
        index=False,
    )

    # ========================================================
    # MATRIZ DE CONFUSION BASELINE
    # ========================================================

    pred_tree = (
        predicciones_test[
            baseline
        ]
    )

    cm = confusion_matrix(
        y_test,
        pred_tree,
    )

    pd.DataFrame(
        cm,
        index=[
            "Real_No_buena",
            "Real_Buena",
        ],
        columns=[
            "Pred_No_buena",
            "Pred_Buena",
        ],
    ).to_csv(
        RESULTS_DIR
        / "redwine_matriz_confusion.csv"
    )

    with open(
        RESULTS_DIR
        / "redwine_reporte_clasificacion.txt",
        "w",
        encoding="utf-8",
    ) as archivo:

        archivo.write(
            classification_report(
                y_test,
                pred_tree,
                target_names=[
                    "No buena",
                    "Buena",
                ],
                zero_division=0,
            )
        )

    # ========================================================
    # MATRIZ GRAFICA
    # ========================================================

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No buena",
            "Buena",
        ],
    )

    display.plot()

    plt.title(
        "Red Wine - Árbol de decisión: matriz de confusión"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "redwine_matriz_confusion.png",
        dpi=150,
    )

    plt.close()

    # ========================================================
    # ARBOL
    # ========================================================

    arbol_pipeline = (
        modelos_entrenados[
            baseline
        ]
    )

    plt.figure(
        figsize=(18, 10)
    )

    plot_tree(
        arbol_pipeline.named_steps[
            "tree"
        ],
        feature_names=X.columns,
        class_names=[
            "No buena",
            "Buena",
        ],
        filled=False,
        rounded=True,
        fontsize=7,
    )

    plt.title(
        "Red Wine - Árbol de decisión"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "redwine_arbol.png",
        dpi=150,
    )

    plt.close()

    # ========================================================
    # CONSOLA
    # ========================================================

    print(
        "\nRed Wine - Comparación de modelos"
    )

    columnas = [
        "modelo",
        "F1-score_cv_media",
        "F1-score_cv_std",
        "Balanced Accuracy_cv_media",
        "ROC-AUC_cv_media",
        "F1-score_test",
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
        f"  F1 CV = "
        f"{mejor_f1_cv:.6f}"
    )

    print(
        "\nMatriz de confusión "
        "del árbol histórico:"
    )

    print(cm)

    return {
        "modelos": modelos_entrenados,
        "comparacion": comparacion_modelos,
        "modelo_seleccionado": mejor_modelo,
        "matriz_confusion_baseline": cm,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
