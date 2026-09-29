import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

    model = Pipeline(
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
                "regressor",
                LinearRegression(),
            ),
        ]
    )

    # ========================================================
    # VALIDACION CRUZADA
    # SOLO sobre el conjunto de entrenamiento.
    # ========================================================

    cv_folds, cv_resumen = (
        validacion_cruzada_regresion(
            model,
            X_train,
            y_train,
        )
    )

    # ========================================================
    # ENTRENAMIENTO DEFINITIVO DEL HOLDOUT
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

    metricas_train = (
        metricas_regresion(
            y_train,
            pred_train,
        )
    )

    metricas_test = (
        metricas_regresion(
            y_test,
            pred_test,
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
    # EXPORTACION
    # ========================================================

    df_test.to_csv(
        RESULTS_DIR
        / "candy_metricas.csv",
        index=False,
    )

    df_train.to_csv(
        RESULTS_DIR
        / "candy_metricas_train.csv",
        index=False,
    )

    comparacion.to_csv(
        RESULTS_DIR
        / "candy_comparacion_train_test.csv",
        index=False,
    )

    cv_folds.to_csv(
        RESULTS_DIR
        / "candy_validacion_cruzada.csv",
        index=False,
    )

    cv_resumen.to_csv(
        RESULTS_DIR
        / "candy_validacion_cruzada_resumen.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "real": y_test.values,
            "predicho": pred_test,
        }
    ).to_csv(
        RESULTS_DIR
        / "candy_predicciones.csv",
        index=False,
    )

    # ========================================================
    # GRAFICO
    # ========================================================

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        y_test,
        pred_test,
    )

    minimo = min(
        y_test.min(),
        pred_test.min(),
    )

    maximo = max(
        y_test.max(),
        pred_test.max(),
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
        "Candy - Real vs. predicho"
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
        "\nCandy - Regresión lineal (TEST)"
    )

    print(
        df_test.to_string(
            index=False
        )
    )

    print(
        "\nCandy - Validación cruzada "
        "5-fold sobre TRAIN"
    )

    print(
        cv_resumen.to_string(
            index=False
        )
    )

    return {
        "modelo": model,
        "metricas_train": df_train,
        "metricas_test": df_test,
        "comparacion": comparacion,
        "validacion_folds": cv_folds,
        "validacion_resumen": cv_resumen,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predicciones": pred_test,
    }
