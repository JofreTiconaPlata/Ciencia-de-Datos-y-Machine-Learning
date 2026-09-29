from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

RESULTS = ROOT / "resultados"

OUTPUT = (
    RESULTS
    / "graficos_finales"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CANDY - TUNING
# ============================================================

candy = pd.read_csv(
    RESULTS
    / "candy_tuning.csv"
).sort_values(
    "RMSE_cv_media"
)

plt.figure(
    figsize=(9, 5)
)

plt.bar(
    candy["modelo"],
    candy["RMSE_cv_media"],
)

plt.errorbar(
    candy["modelo"],
    candy["RMSE_cv_media"],
    yerr=candy["RMSE_cv_std"],
    fmt="none",
    capsize=5,
)

plt.ylabel(
    "RMSE medio (CV)"
)

plt.xlabel(
    "Modelo"
)

plt.title(
    "Candy - Comparación de modelos mediante validación cruzada"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    OUTPUT
    / "candy_rmse_cv.png",
    dpi=180,
)

plt.close()


# ============================================================
# CANDY - TRAIN / CV / TEST
# ============================================================

candy_diag = pd.read_csv(
    RESULTS
    / "candy_diagnostico_final.csv"
)

row = candy_diag.iloc[0]

plt.figure(
    figsize=(7, 5)
)

plt.bar(
    [
        "TRAIN",
        "CV",
        "TEST",
    ],
    [
        row["RMSE_train"],
        row["RMSE_cv_media"],
        row["RMSE_test"],
    ],
)

plt.ylabel(
    "RMSE"
)

plt.title(
    f"Candy - Generalización del modelo final ({row['modelo']})"
)

plt.tight_layout()

plt.savefig(
    OUTPUT
    / "candy_train_cv_test.png",
    dpi=180,
)

plt.close()


# ============================================================
# RED WINE - F1
# ============================================================

wine = pd.read_csv(
    RESULTS
    / "redwine_tuning.csv"
).sort_values(
    "F1_cv_media",
    ascending=False,
)

plt.figure(
    figsize=(9, 5)
)

plt.bar(
    wine["modelo"],
    wine["F1_cv_media"],
)

plt.errorbar(
    wine["modelo"],
    wine["F1_cv_media"],
    yerr=wine["F1_cv_std"],
    fmt="none",
    capsize=5,
)

plt.ylabel(
    "F1-score medio (CV)"
)

plt.xlabel(
    "Modelo"
)

plt.title(
    "Red Wine - Comparación de modelos mediante validación cruzada"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    OUTPUT
    / "wine_f1_cv.png",
    dpi=180,
)

plt.close()


# ============================================================
# RED WINE - TRAIN / CV / TEST
# ============================================================

wine_diag = pd.read_csv(
    RESULTS
    / "redwine_diagnostico_final.csv"
)

row = wine_diag.iloc[0]

plt.figure(
    figsize=(7, 5)
)

plt.bar(
    [
        "TRAIN",
        "CV",
        "TEST",
    ],
    [
        row["F1_train"],
        row["F1_cv_media"],
        row["F1_test"],
    ],
)

plt.ylabel(
    "F1-score"
)

plt.ylim(
    0,
    1,
)

plt.title(
    f"Red Wine - Generalización del modelo final ({row['modelo']})"
)

plt.tight_layout()

plt.savefig(
    OUTPUT
    / "wine_train_cv_test.png",
    dpi=180,
)

plt.close()


# ============================================================
# RED WINE - METRICAS FINALES
# ============================================================

metricas = {
    "F1": row["F1_test"],
    "Balanced Accuracy": (
        row["Balanced_Accuracy_test"]
    ),
    "ROC-AUC": (
        row["ROC_AUC_test"]
    ),
}

plt.figure(
    figsize=(7, 5)
)

plt.bar(
    metricas.keys(),
    metricas.values(),
)

plt.ylim(
    0,
    1,
)

plt.ylabel(
    "Valor"
)

plt.title(
    "Red Wine - Métricas del modelo final en TEST"
)

plt.tight_layout()

plt.savefig(
    OUTPUT
    / "wine_metricas_finales.png",
    dpi=180,
)

plt.close()


print(
    "Visualizaciones finales generadas:"
)

for file in sorted(
    OUTPUT.glob("*.png")
):
    print(
        " -",
        file.name,
    )
