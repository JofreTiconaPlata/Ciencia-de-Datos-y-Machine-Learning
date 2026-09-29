import numpy as np
import pandas as pd

from src.config import RESULTS_DIR


def resumen_limpieza(df: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """
    Aplica y documenta los tres procesos utilizados en los avances originales:

    1. Tratamiento de valores faltantes.
    2. Eliminación de registros duplicados.
    3. Detección de valores atípicos mediante IQR.

    Los valores atípicos se reportan, pero no se eliminan automáticamente.
    """
    data = df.copy()
    filas_iniciales = len(data)

    # 1. Valores faltantes
    faltantes = int(data.isna().sum().sum())

    columnas_numericas = data.select_dtypes(include=np.number).columns
    columnas_categoricas = data.select_dtypes(exclude=np.number).columns

    for columna in columnas_numericas:
        if data[columna].isna().any():
            data[columna] = data[columna].fillna(data[columna].median())

    for columna in columnas_categoricas:
        if data[columna].isna().any():
            data[columna] = data[columna].fillna(
                data[columna].mode().iloc[0]
            )

    # 2. Duplicados
    duplicados = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)

    # 3. Valores atípicos mediante IQR
    atipicos = {}

    for columna in data.select_dtypes(include=np.number).columns:
        q1, q3 = data[columna].quantile([0.25, 0.75])
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

    pd.DataFrame(
        {
            "dataset": [nombre],
            "filas_iniciales": [filas_iniciales],
            "faltantes_detectados": [faltantes],
            "duplicados_eliminados": [duplicados],
            "filas_finales": [len(data)],
            "total_atipicos_reportados": [sum(atipicos.values())],
        }
    ).to_csv(
        RESULTS_DIR / f"{nombre}_limpieza_resumen.csv",
        index=False,
    )

    pd.DataFrame(
        list(atipicos.items()),
        columns=["variable", "atipicos_IQR"],
    ).to_csv(
        RESULTS_DIR / f"{nombre}_atipicos.csv",
        index=False,
    )

    return data
