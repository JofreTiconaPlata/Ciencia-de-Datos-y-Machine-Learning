import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
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
from src.data_processing import resumen_limpieza


def ejecutar_wine():
    print("\n=== DATASET RED WINE ===")

    df = pd.read_csv(
        DATA_DIR / "winequality-red.csv"
    )

    print("Dimensiones originales:", df.shape)
    print("Faltantes:", int(df.isna().sum().sum()))
    print("Duplicados:", int(df.duplicated().sum()))

    clean = resumen_limpieza(df, "redwine")

    clean.to_csv(
        RESULTS_DIR / "redwine_clean.csv",
        index=False,
    )

    clean["quality_class"] = (
        clean["quality"] >= 7
    ).astype(int)

    X = clean.drop(
        columns=["quality", "quality_class"]
    )

    y = clean["quality_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
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

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        pred,
    )

    precision = precision_score(
        y_test,
        pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        pred,
        zero_division=0,
    )

    metrics = pd.DataFrame(
        {
            "metrica": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
            ],
            "valor": [
                accuracy,
                precision,
                recall,
                f1,
            ],
        }
    )

    metrics.to_csv(
        RESULTS_DIR / "redwine_metricas.csv",
        index=False,
    )

    cm = confusion_matrix(
        y_test,
        pred,
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
                pred,
                target_names=[
                    "No buena",
                    "Buena",
                ],
                zero_division=0,
            )
        )

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
        RESULTS_DIR / "redwine_arbol.png",
        dpi=150,
    )

    plt.close()

    print("\nRed Wine - Árbol de decisión")
    print(metrics.to_string(index=False))

    print("\nMatriz de confusión:")
    print(cm)

    return {
        "modelo": model,
        "metricas": metrics,
        "matriz_confusion": cm,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predicciones": pred,
    }
