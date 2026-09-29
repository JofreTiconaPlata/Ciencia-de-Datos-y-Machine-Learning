from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from src.candy import CANDY_FEATURES
from src.wine import WINE_FEATURES


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


CANDY_BINARY = {
    "chocolate",
    "fruity",
    "caramel",
    "peanutyalmondy",
    "nougat",
    "crispedricewafer",
    "hard",
    "bar",
    "pluribus",
}


def pedir_float(nombre):
    while True:
        try:
            return float(
                input(
                    f"{nombre}: "
                ).strip()
            )
        except ValueError:
            print(
                "Valor inválido. "
                "Introduce un número."
            )


def pedir_binario(nombre):
    while True:
        valor = input(
            f"{nombre} (0/1): "
        ).strip()

        if valor in {
            "0",
            "1",
        }:
            return int(valor)

        print(
            "Introduce solamente 0 o 1."
        )


def predecir_candy():

    modelo = joblib.load(
        MODELS_DIR
        / "candy_final.joblib"
    )

    valores = {}

    print()
    print(
        "=== PREDICCION CANDY ==="
    )

    for feature in CANDY_FEATURES:

        if feature in CANDY_BINARY:
            valores[feature] = (
                pedir_binario(
                    feature
                )
            )
        else:
            valores[feature] = (
                pedir_float(
                    feature
                )
            )

    X = pd.DataFrame(
        [valores],
        columns=CANDY_FEATURES,
    )

    pred = float(
        modelo.predict(X)[0]
    )

    print()
    print(
        f"Winpercent estimado: "
        f"{pred:.2f}"
    )


def predecir_wine():

    modelo = joblib.load(
        MODELS_DIR
        / "wine_final.joblib"
    )

    valores = {}

    print()
    print(
        "=== PREDICCION RED WINE ==="
    )

    for feature in WINE_FEATURES:
        valores[feature] = (
            pedir_float(
                feature
            )
        )

    X = pd.DataFrame(
        [valores],
        columns=WINE_FEATURES,
    )

    pred = int(
        modelo.predict(X)[0]
    )

    etiqueta = (
        "Buena"
        if pred == 1
        else "No buena"
    )

    print()
    print(
        f"Clasificación: {etiqueta}"
    )

    if hasattr(
        modelo,
        "predict_proba",
    ):
        prob = float(
            modelo.predict_proba(
                X
            )[0, 1]
        )

        print(
            "Probabilidad estimada "
            f"de clase Buena: "
            f"{prob:.2%}"
        )

    elif hasattr(
        modelo,
        "decision_function",
    ):
        score = float(
            modelo.decision_function(
                X
            )[0]
        )

        print(
            "Score de decisión "
            f"para clase Buena: "
            f"{score:.4f}"
        )


def main():

    print()
    print(
        "================================"
    )

    print(
        " PREDICCION MACHINE LEARNING"
    )

    print(
        "================================"
    )

    print(
        "1. Candy"
    )

    print(
        "2. Red Wine"
    )

    opcion = input(
        "Selecciona 1 o 2: "
    ).strip()

    if opcion == "1":
        predecir_candy()

    elif opcion == "2":
        predecir_wine()

    else:
        print(
            "Opción inválida."
        )


if __name__ == "__main__":
    main()
