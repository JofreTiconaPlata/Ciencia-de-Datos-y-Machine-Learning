import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
)

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

    model = Pipeline(
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
    )

    # ========================================================
    # VALIDACION CRUZADA
    # SOLO sobre TRAIN.
    # ========================================================

    cv_folds, cv_resumen = (
        validacion_cruzada_clasificacion(
            model,
            X_train,
            y_train,
        )
    )

    # ========================================================
    # ENTRENAMIENTO DEFINITIVO
    # ========================================================

    model.fit(
        X_train,
        y_train,
    )

    pred_train = model.predict(
        X_train
    )

    pred_test = model.predict(
        X_test
    )

    score_train = (
        obtener_score_positivo(
            model,
            X_train,
        )
    )

    score_test = (
        obtener_score_positivo(
            model,
            X_test,
        )
    )

    metricas_train = (
        metricas_clasificacion(
            y_train,
            pred_train,
            score_train,
        )
    )

    metricas_test = (
        metricas_clasificacion(
            y_test,
            pred_test,
            score_test,
        )
    )

    df_train = dataframe_metricas(
        metricas_train
    )

    df_test = dataframe_metricas(
        metricas_test
    )

    comparacion = (
        comparacion_train_test(
            metricas_train,
            metricas_test,
        )
    )

    # ========================================================
    # EXPORTACION DE METRICAS
    # ========================================================

    df_test.to_csv(
        RESULTS_DIR
        / "redwine_metricas.csv",
        index=False,
    )

    df_train.to_csv(
        RESULTS_DIR
        / "redwine_metricas_train.csv",
        index=False,
    )

    comparacion.to_csv(
        RESULTS_DIR
        / "redwine_comparacion_train_test.csv",
        index=False,
    )

    cv_folds.to_csv(
        RESULTS_DIR
        / "redwine_validacion_cruzada.csv",
        index=False,
    )

    cv_resumen.to_csv(
        RESULTS_DIR
        / "redwine_validacion_cruzada_resumen.csv",
        index=False,
    )

    # ========================================================
    # MATRIZ DE CONFUSION
    # ========================================================

    cm = confusion_matrix(
        y_test,
        pred_test,
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

    # ========================================================
    # REPORTE
    # ========================================================

    with open(
        RESULTS_DIR
        / "redwine_reporte_clasificacion.txt",
        "w",
        encoding="utf-8",
    ) as archivo:

        archivo.write(
            classification_report(
                y_test,
                pred_test,
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
        "Red Wine - Matriz de confusión"
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

    plt.figure(
        figsize=(18, 10)
    )

    plot_tree(
        model.named_steps["tree"],
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
        "\nRed Wine - Árbol de decisión (TEST)"
    )

    print(
        df_test.to_string(
            index=False
        )
    )

    print(
        "\nRed Wine - Validación cruzada "
        "estratificada 5-fold sobre TRAIN"
    )

    print(
        cv_resumen.to_string(
            index=False
        )
    )

    print(
        "\nMatriz de confusión:"
    )

    print(cm)

    return {
        "modelo": model,
        "metricas_train": df_train,
        "metricas_test": df_test,
        "comparacion": comparacion,
        "validacion_folds": cv_folds,
        "validacion_resumen": cv_resumen,
        "matriz_confusion": cm,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predicciones": pred_test,
    }
