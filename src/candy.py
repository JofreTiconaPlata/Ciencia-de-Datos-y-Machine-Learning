import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE,
)
from src.data_processing import resumen_limpieza


def ejecutar_candy():
    print("\n=== DATASET CANDY ===")

    df = pd.read_csv(DATA_DIR / "candy-data.csv")

    print("Dimensiones originales:", df.shape)
    print("Faltantes:", int(df.isna().sum().sum()))
    print("Duplicados:", int(df.duplicated().sum()))

    clean = resumen_limpieza(df, "candy")
    clean.to_csv(
        RESULTS_DIR / "candy_clean.csv",
        index=False,
    )

    X = clean.drop(
        columns=["winpercent", "competitorname"]
    )
    y = clean["winpercent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
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

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)

    metrics = pd.DataFrame(
        {
            "metrica": [
                "MAE",
                "MSE",
                "RMSE",
                "R2",
            ],
            "valor": [
                mae,
                mse,
                rmse,
                r2,
            ],
        }
    )

    metrics.to_csv(
        RESULTS_DIR / "candy_metricas.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "real": y_test.values,
            "predicho": pred,
        }
    ).to_csv(
        RESULTS_DIR / "candy_predicciones.csv",
        index=False,
    )

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, pred)

    minimo = min(y_test.min(), pred.min())
    maximo = max(y_test.max(), pred.max())

    plt.plot(
        [minimo, maximo],
        [minimo, maximo],
    )

    plt.xlabel("Winpercent real")
    plt.ylabel("Winpercent predicho")
    plt.title("Candy - Real vs. predicho")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "candy_real_vs_predicho.png",
        dpi=150,
    )

    plt.close()

    print("\nCandy - Regresión lineal")
    print(metrics.to_string(index=False))

    return {
        "modelo": model,
        "metricas": metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predicciones": pred,
    }
