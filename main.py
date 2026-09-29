"""
Proyecto de Aprendizaje Automático.

Casos de estudio:
1. Candy Power Ranking:
   regresión de winpercent.

2. Wine Quality - Red Wine:
   clasificación binaria de calidad.

Ejecutar:
    python main.py
"""

from src.candy import ejecutar_candy
from src.wine import ejecutar_wine


def main():
    ejecutar_candy()
    ejecutar_wine()

    print(
        "\nProceso terminado. "
        "Revisa la carpeta 'resultados'."
    )


if __name__ == "__main__":
    main()
