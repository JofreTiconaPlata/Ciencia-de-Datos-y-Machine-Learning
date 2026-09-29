import pandas as pd

from src.config import RESULTS_DIR


def limpiar_y_documentar(
    df: pd.DataFrame,
    nombre: str,
    variables_iqr: list[str],
) -> pd.DataFrame:
    """
    Realiza la limpieza estructural y documenta el dataset.

    Operaciones:
    1. Cuenta valores faltantes, pero NO los imputa aquí.
       La imputación se realiza posteriormente dentro del Pipeline
       utilizando únicamente información del conjunto de entrenamiento.
    2. Elimina filas duplicadas.
    3. Detecta valores atípicos mediante IQR únicamente en las
       variables predictoras indicadas.

    Los valores atípicos se reportan, pero no se eliminan.
    """

    data = df.copy()

    filas_iniciales = len(data)

    # 1. Valores faltantes
    faltantes = int(data.isna().sum().sum())

    # 2. Registros duplicados
    duplicados = int(data.duplicated().sum())

    data = (
        data
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # 3. Detección IQR únicamente sobre predictores numéricos
    atipicos = {}

    for columna in variables_iqr:
        if columna not in data.columns:
            raise ValueError(
                f"La variable '{columna}' no existe "
                f"en el dataset '{nombre}'."
            )

        if not pd.api.types.is_numeric_dtype(data[columna]):
            continue

        q1 = data[columna].quantile(0.25)
        q3 = data[columna].quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            cantidad = 0
        else:
            limite_inferior = q1 - 1.5 * iqr
            limite_superior = q3 + 1.5 * iqr

            cantidad = int(
                (
                    (data[columna] < limite_inferior)
                    | (data[columna] > limite_superior)
                ).sum()
            )

        atipicos[columna] = cantidad

    resumen = pd.DataFrame(
        {
            "dataset": [nombre],
            "filas_iniciales": [filas_iniciales],
            "faltantes_detectados": [faltantes],
            "duplicados_eliminados": [duplicados],
            "filas_finales": [len(data)],
            "total_atipicos_reportados": [
                sum(atipicos.values())
            ],
        }
    )

    resumen.to_csv(
        RESULTS_DIR / f"{nombre}_limpieza_resumen.csv",
        index=False,
    )

    pd.DataFrame(
        list(atipicos.items()),
        columns=[
            "variable",
            "atipicos_IQR",
        ],
    ).to_csv(
        RESULTS_DIR / f"{nombre}_atipicos.csv",
        index=False,
    )

    return data
