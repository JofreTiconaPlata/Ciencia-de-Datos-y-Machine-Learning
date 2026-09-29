"""
PRACTICA PRIMER PARCIAL - APRENDIZAJE AUTOMATICO
Datasets:
1) Candy: Regresion de winpercent
2) Red Wine: Arbol de decision para clasificar calidad (buena >= 7)

Ejecutar:
    python main.py

El programa:
- inspecciona los datos
- realiza tres procesos de limpieza documentados:
  1. valores faltantes
  2. duplicados
  3. valores atipicos en variables numericas mediante IQR (solo se reportan y
     se recortan de forma conservadora cuando corresponde)
- divide entrenamiento/prueba
- entrena modelos
- calcula metricas
- genera graficos y archivos limpios en resultados/
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "resultados"
OUT.mkdir(exist_ok=True)

def resumen_limpieza(df, nombre):
    """Aplica y documenta 3 procesos: faltantes, duplicados y atipicos."""
    d = df.copy()
    filas_iniciales = len(d)

    # 1) Valores faltantes
    faltantes = int(d.isna().sum().sum())
    # Para este proyecto, las columnas numéricas faltantes se imputan con mediana
    # y las categóricas con moda. Si no hay faltantes, no se modifica nada.
    num = d.select_dtypes(include=np.number).columns
    cat = d.select_dtypes(exclude=np.number).columns
    for c in num:
        if d[c].isna().any():
            d[c] = d[c].fillna(d[c].median())
    for c in cat:
        if d[c].isna().any():
            d[c] = d[c].fillna(d[c].mode().iloc[0])

    # 2) Duplicados
    duplicados = int(d.duplicated().sum())
    d = d.drop_duplicates().reset_index(drop=True)

    # 3) Atípicos por IQR: se reportan; no se eliminan automáticamente.
    # Esto evita perder observaciones válidas sin una justificación del dominio.
    outliers = {}
    for c in d.select_dtypes(include=np.number).columns:
        q1, q3 = d[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            n = 0
        else:
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n = int(((d[c] < low) | (d[c] > high)).sum())
        outliers[c] = n

    pd.DataFrame({
        "dataset": [nombre],
        "filas_iniciales": [filas_iniciales],
        "faltantes_detectados": [faltantes],
        "duplicados_eliminados": [duplicados],
        "filas_finales": [len(d)],
        "total_atipicos_reportados": [sum(outliers.values())]
    }).to_csv(OUT / f"{nombre}_limpieza_resumen.csv", index=False)

    pd.DataFrame(list(outliers.items()), columns=["variable", "atipicos_IQR"])\
      .to_csv(OUT / f"{nombre}_atipicos.csv", index=False)

    return d

def candy():
    print("\n=== DATASET CANDY ===")
    df = pd.read_csv(DATA / "candy-data.csv")
    print("Dimensiones originales:", df.shape)
    print("Faltantes:", int(df.isna().sum().sum()))
    print("Duplicados:", int(df.duplicated().sum()))

    clean = resumen_limpieza(df, "candy")
    clean.to_csv(OUT / "candy_clean.csv", index=False)

    # Variable objetivo
    X = clean.drop(columns=["winpercent", "competitorname"])
    y = clean["winpercent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)

    metrics = pd.DataFrame({
        "metrica": ["MAE", "MSE", "RMSE", "R2"],
        "valor": [mae, mse, rmse, r2]
    })
    metrics.to_csv(OUT / "candy_metricas.csv", index=False)

    comp = pd.DataFrame({"real": y_test.values, "predicho": pred})
    comp.to_csv(OUT / "candy_predicciones.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, pred)
    mn, mx = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Winpercent real")
    plt.ylabel("Winpercent predicho")
    plt.title("Candy - Real vs. predicho")
    plt.tight_layout()
    plt.savefig(OUT / "candy_real_vs_predicho.png", dpi=150)
    plt.close()

    print("\nCandy - Regresión lineal")
    print(metrics.to_string(index=False))

def wine():
    print("\n=== DATASET RED WINE ===")
    df = pd.read_csv(DATA / "winequality-red.csv")
    print("Dimensiones originales:", df.shape)
    print("Faltantes:", int(df.isna().sum().sum()))
    print("Duplicados:", int(df.duplicated().sum()))

    clean = resumen_limpieza(df, "redwine")
    clean.to_csv(OUT / "redwine_clean.csv", index=False)

    # Propuesta de clasificación para poder evaluar con matriz de confusión:
    # calidad 7-10 = "buena" (1), calidad 3-6 = "no_buena" (0).
    clean["quality_class"] = (clean["quality"] >= 7).astype(int)

    X = clean.drop(columns=["quality", "quality_class"])
    y = clean["quality_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("tree", DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    metrics = pd.DataFrame({
        "metrica": ["Accuracy", "Precision", "Recall", "F1-score"],
        "valor": [acc, precision, recall, f1]
    })
    metrics.to_csv(OUT / "redwine_metricas.csv", index=False)

    cm = confusion_matrix(y_test, pred)
    pd.DataFrame(
        cm, index=["Real_No_buena", "Real_Buena"],
        columns=["Pred_No_buena", "Pred_Buena"]
    ).to_csv(OUT / "redwine_matriz_confusion.csv")

    with open(OUT / "redwine_reporte_clasificacion.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(
            y_test, pred,
            target_names=["No buena", "Buena"],
            zero_division=0
        ))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No buena", "Buena"]
    )
    disp.plot()
    plt.title("Red Wine - Matriz de confusión")
    plt.tight_layout()
    plt.savefig(OUT / "redwine_matriz_confusion.png", dpi=150)
    plt.close()

    plt.figure(figsize=(18, 10))
    plot_tree(
        model.named_steps["tree"],
        feature_names=X.columns,
        class_names=["No buena", "Buena"],
        filled=False,
        rounded=True,
        fontsize=7
    )
    plt.title("Red Wine - Árbol de decisión")
    plt.tight_layout()
    plt.savefig(OUT / "redwine_arbol.png", dpi=150)
    plt.close()

    print("\nRed Wine - Árbol de decisión")
    print(metrics.to_string(index=False))
    print("\nMatriz de confusión:")
    print(cm)

if __name__ == "__main__":
    candy()
    wine()
    print("\nProceso terminado. Revisa la carpeta 'resultados'.")
