from pathlib import Path
import json
import sys

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# RUTAS
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


DATA = ROOT / "data"

RESULTS = ROOT / "resultados"

MODELS = ROOT / "models"

PLOTS = (
    RESULTS
    / "graficos_finales"
)


# ============================================================
# CONFIGURACION
# ============================================================

st.set_page_config(
    page_title=(
        "Candy & Red Wine - Machine Learning"
    ),
    page_icon="📊",
    layout="wide",
)


# ============================================================
# CARGA
# ============================================================

@st.cache_resource
def cargar_modelos():

    candy_model = joblib.load(
        MODELS
        / "candy_final.joblib"
    )

    wine_model = joblib.load(
        MODELS
        / "wine_final.joblib"
    )

    return (
        candy_model,
        wine_model,
    )


@st.cache_data
def cargar_json(path):

    with open(
        path,
        encoding="utf-8",
    ) as f:
        return json.load(f)


@st.cache_data
def cargar_datos():

    candy = pd.read_csv(
        DATA
        / "candy-data.csv"
    )

    wine = pd.read_csv(
        DATA
        / "winequality-red.csv"
    )

    return candy, wine


candy_model, wine_model = (
    cargar_modelos()
)

candy_meta = cargar_json(
    MODELS
    / "candy_metadata.json"
)

wine_meta = cargar_json(
    MODELS
    / "wine_metadata.json"
)

candy_df, wine_df = (
    cargar_datos()
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Machine Learning"
)

pagina = st.sidebar.radio(
    "Sección",
    [
        "Inicio",
        "Candy",
        "Red Wine",
        "Redes neuronales",
        "Metodología",
    ],
)


# ============================================================
# INICIO
# ============================================================

if pagina == "Inicio":

    st.title(
        "Proyecto de Ciencia de Datos y Machine Learning"
    )

    st.write(
        """
        Aplicación interactiva para los datasets
        **Candy Power Ranking** y **Red Wine Quality**.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Candy"
        )

        st.metric(
            "Modelo final",
            candy_meta["modelo"],
        )

        st.metric(
            "RMSE CV",
            f"{candy_meta['cv_rmse']:.4f}",
        )

        st.metric(
            "RMSE TEST",
            (
                f"{candy_meta['metricas_test']['RMSE']:.4f}"
            ),
        )

    with col2:

        st.subheader(
            "Red Wine"
        )

        st.metric(
            "Modelo final",
            wine_meta["modelo"],
        )

        st.metric(
            "F1 CV",
            f"{wine_meta['cv_f1']:.4f}",
        )

        st.metric(
            "F1 TEST",
            (
                f"{wine_meta['metricas_test']['F1-score']:.4f}"
            ),
        )

        st.metric(
            "ROC-AUC TEST",
            (
                f"{wine_meta['metricas_test']['ROC-AUC']:.4f}"
            ),
        )

    st.info(
        """
        Los modelos se seleccionaron utilizando
        validación cruzada sobre TRAIN.
        El conjunto TEST permaneció fuera del proceso
        de selección y ajuste.
        """
    )


# ============================================================
# CANDY
# ============================================================

elif pagina == "Candy":

    st.title(
        "Candy Power Ranking"
    )

    tabs = st.tabs(
        [
            "Datos",
            "Modelos",
            "Predicción",
        ]
    )

    with tabs[0]:

        st.subheader(
            "Dataset"
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        col1.metric(
            "Filas",
            candy_df.shape[0],
        )

        col2.metric(
            "Columnas",
            candy_df.shape[1],
        )

        col3.metric(
            "Predictores",
            len(
                candy_meta[
                    "features"
                ]
            ),
        )

        st.dataframe(
            candy_df,
            use_container_width=True,
        )

    with tabs[1]:

        st.subheader(
            "Comparación de modelos"
        )

        tabla = pd.read_csv(
            RESULTS
            / "candy_tuning.csv"
        )

        st.dataframe(
            tabla,
            use_container_width=True,
        )

        st.image(
            str(
                PLOTS
                / "candy_rmse_cv.png"
            )
        )

        st.image(
            str(
                PLOTS
                / "candy_train_cv_test.png"
            )
        )

        st.write(
            "Modelo final:",
            candy_meta["modelo"],
        )

        st.write(
            "Criterio:",
            candy_meta[
                "criterio_seleccion"
            ],
        )

    with tabs[2]:

        st.subheader(
            "Predicción con datos introducidos por el usuario"
        )

        values = {}

        binary = {
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

        for feature in candy_meta[
            "features"
        ]:

            if feature in binary:

                values[feature] = int(
                    st.checkbox(
                        feature,
                        value=False,
                    )
                )

            else:

                serie = candy_df[
                    feature
                ]

                values[feature] = (
                    st.number_input(
                        feature,
                        min_value=float(
                            serie.min()
                        ),
                        max_value=float(
                            serie.max()
                        ),
                        value=float(
                            serie.median()
                        ),
                        step=0.01,
                    )
                )

        if st.button(
            "Predecir winpercent",
            type="primary",
        ):

            entrada = pd.DataFrame(
                [values],
                columns=candy_meta[
                    "features"
                ],
            )

            pred = float(
                candy_model.predict(
                    entrada
                )[0]
            )

            st.success(
                f"Winpercent estimado: {pred:.2f}"
            )

            st.caption(
                """
                El resultado representa la estimación
                del porcentaje de preferencia del dulce
                según las variables proporcionadas.
                """
            )


# ============================================================
# RED WINE
# ============================================================

elif pagina == "Red Wine":

    st.title(
        "Red Wine Quality"
    )

    tabs = st.tabs(
        [
            "Datos",
            "Modelos",
            "Predicción",
        ]
    )

    with tabs[0]:

        st.subheader(
            "Dataset"
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        col1.metric(
            "Filas originales",
            wine_df.shape[0],
        )

        col2.metric(
            "Columnas",
            wine_df.shape[1],
        )

        col3.metric(
            "Predictores",
            len(
                wine_meta[
                    "features"
                ]
            ),
        )

        st.dataframe(
            wine_df,
            use_container_width=True,
        )

        quality_class = (
            wine_df["quality"] >= 7
        ).map(
            {
                True: "Buena",
                False: "No buena",
            }
        )

        st.subheader(
            "Distribución de clases"
        )

        st.bar_chart(
            quality_class
            .value_counts()
        )

    with tabs[1]:

        st.subheader(
            "Comparación de modelos"
        )

        tabla = pd.read_csv(
            RESULTS
            / "redwine_tuning.csv"
        )

        st.dataframe(
            tabla,
            use_container_width=True,
        )

        st.image(
            str(
                PLOTS
                / "wine_f1_cv.png"
            )
        )

        st.image(
            str(
                PLOTS
                / "wine_train_cv_test.png"
            )
        )

        st.image(
            str(
                PLOTS
                / "wine_metricas_finales.png"
            )
        )

        st.write(
            "Modelo final:",
            wine_meta["modelo"],
        )

        st.write(
            "Criterio:",
            wine_meta[
                "criterio_seleccion"
            ],
        )

    with tabs[2]:

        st.subheader(
            "Predicción con datos introducidos por el usuario"
        )

        values = {}

        for feature in wine_meta[
            "features"
        ]:

            serie = wine_df[
                feature
            ]

            rango = float(
                serie.max()
                - serie.min()
            )

            step = max(
                rango / 100,
                0.001,
            )

            values[feature] = (
                st.number_input(
                    feature,
                    min_value=float(
                        serie.min()
                    ),
                    max_value=float(
                        serie.max()
                    ),
                    value=float(
                        serie.median()
                    ),
                    step=step,
                    format="%.4f",
                )
            )

        if st.button(
            "Clasificar vino",
            type="primary",
        ):

            entrada = pd.DataFrame(
                [values],
                columns=wine_meta[
                    "features"
                ],
            )

            pred = int(
                wine_model.predict(
                    entrada
                )[0]
            )

            etiqueta = (
                "Buena"
                if pred == 1
                else "No buena"
            )

            if pred == 1:
                st.success(
                    "Clasificación: Buena"
                )
            else:
                st.warning(
                    "Clasificación: No buena"
                )

            if hasattr(
                wine_model,
                "predict_proba",
            ):

                prob = float(
                    wine_model
                    .predict_proba(
                        entrada
                    )[0, 1]
                )

                st.metric(
                    "Probabilidad estimada de Buena",
                    f"{prob:.2%}",
                )

            elif hasattr(
                wine_model,
                "decision_function",
            ):

                score = float(
                    wine_model
                    .decision_function(
                        entrada
                    )[0]
                )

                st.metric(
                    "Score de decisión",
                    f"{score:.4f}",
                )

                st.caption(
                    """
                    Un score positivo favorece la clase
                    Buena y uno negativo favorece
                    No buena. No debe interpretarse
                    directamente como probabilidad.
                    """
                )


# ============================================================
# RNA
# ============================================================

elif pagina == "Redes neuronales":

    st.title(
        "Redes Neuronales Artificiales"
    )

    st.write(
        """
        En ambos casos se implementó un
        **Perceptrón Multicapa (MLP)**.

        El flujo de información es **feedforward** y
        sus parámetros se ajustan durante el
        entrenamiento mediante **backpropagation**.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Candy - MLPRegressor"
        )

        candy_compare = pd.read_csv(
            RESULTS
            / "candy_modelos_comparacion.csv"
        )

        row = candy_compare[
            candy_compare["modelo"]
            == "mlp_regressor"
        ]

        st.dataframe(
            row,
            use_container_width=True,
        )

        st.write(
            """
            Se utiliza como modelo neuronal
            para el problema de regresión.
            """
        )

    with col2:

        st.subheader(
            "Red Wine - MLPClassifier"
        )

        wine_compare = pd.read_csv(
            RESULTS
            / "redwine_modelos_comparacion.csv"
        )

        row = wine_compare[
            wine_compare["modelo"]
            == "mlp_classifier"
        ]

        st.dataframe(
            row,
            use_container_width=True,
        )

        st.write(
            """
            Se utiliza como modelo neuronal
            para el problema de clasificación.
            """
        )


# ============================================================
# METODOLOGIA
# ============================================================

else:

    st.title(
        "Metodología"
    )

    st.markdown(
        """
        ### Flujo experimental

        1. Carga de los datasets.
        2. Revisión de faltantes y duplicados.
        3. Detección IQR únicamente sobre predictores.
        4. División independiente TRAIN / TEST.
        5. Preprocesamiento dentro de Pipeline.
        6. Validación cruzada de 5 folds exclusivamente en TRAIN.
        7. Comparación de cuatro modelos por dataset.
        8. Ajuste de hiperparámetros únicamente con TRAIN.
        9. Selección usando CV.
        10. Evaluación final una sola vez sobre TEST.

        ### Candy

        Modelos:

        - Regresión lineal
        - Árbol de regresión
        - SVR con kernel RBF
        - MLPRegressor

        Criterio principal de selección:

        **menor RMSE medio de validación cruzada.**

        ### Red Wine

        Modelos:

        - Regresión logística
        - Árbol de decisión
        - SVC con kernel RBF
        - MLPClassifier

        Debido al desbalance de clases, el modelo no se
        selecciona exclusivamente por Accuracy.

        Criterio principal:

        **mayor F1-score medio de validación cruzada.**
        """
    )
