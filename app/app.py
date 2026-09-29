from pathlib import Path
import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


# ============================================================
# RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATA = ROOT / "data"
RESULTS = ROOT / "resultados"
MODELS = ROOT / "models"


RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# CONFIGURACION
# ============================================================

st.set_page_config(
    page_title="Proyecto Machine Learning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CARGA CACHEADA
# ============================================================

@st.cache_resource
def cargar_modelos():
    return (
        joblib.load(
            MODELS / "candy_final.joblib"
        ),
        joblib.load(
            MODELS / "wine_final.joblib"
        ),
    )


@st.cache_data
def cargar_json(path):
    with open(
        path,
        encoding="utf-8",
    ) as archivo:
        return json.load(archivo)


@st.cache_data
def cargar_datasets():
    candy = pd.read_csv(
        DATA / "candy-data.csv"
    )

    wine = pd.read_csv(
        DATA / "winequality-red.csv"
    )

    return candy, wine


@st.cache_data
def preparar_holdout_candy(
    features,
):
    df = pd.read_csv(
        DATA / "candy-data.csv"
    )

    df = (
        df.drop_duplicates()
        .reset_index(drop=True)
    )

    X = df[features]
    y = df["winpercent"]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


@st.cache_data
def preparar_holdout_wine(
    features,
):
    df = pd.read_csv(
        DATA / "winequality-red.csv"
    )

    df = (
        df.drop_duplicates()
        .reset_index(drop=True)
    )

    df["quality_class"] = (
        df["quality"] >= 7
    ).astype(int)

    X = df[features]
    y = df["quality_class"]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


candy_model, wine_model = (
    cargar_modelos()
)

candy_meta = cargar_json(
    MODELS / "candy_metadata.json"
)

wine_meta = cargar_json(
    MODELS / "wine_metadata.json"
)

candy_df, wine_df = (
    cargar_datasets()
)


# ============================================================
# UTILIDADES VISUALES
# ============================================================

def encabezado_paso(
    numero,
    titulo,
    descripcion,
):
    st.markdown(
        f"""
        ### {numero}. {titulo}

        {descripcion}
        """
    )


def grafico_modelos_candy():

    df = pd.read_csv(
        RESULTS / "candy_tuning.csv"
    ).sort_values(
        "RMSE_cv_media"
    )

    fig, ax = plt.subplots(
        figsize=(9, 4.8)
    )

    ax.bar(
        df["modelo"],
        df["RMSE_cv_media"],
    )

    ax.errorbar(
        df["modelo"],
        df["RMSE_cv_media"],
        yerr=df["RMSE_cv_std"],
        fmt="none",
        capsize=6,
    )

    ax.set_ylabel(
        "RMSE medio en CV"
    )

    ax.set_title(
        "Candy: comparación de modelos"
    )

    ax.tick_params(
        axis="x",
        rotation=15,
    )

    fig.tight_layout()

    return fig


def grafico_modelos_wine():

    df = pd.read_csv(
        RESULTS / "redwine_tuning.csv"
    ).sort_values(
        "F1_cv_media",
        ascending=False,
    )

    fig, ax = plt.subplots(
        figsize=(9, 4.8)
    )

    ax.bar(
        df["modelo"],
        df["F1_cv_media"],
    )

    ax.errorbar(
        df["modelo"],
        df["F1_cv_media"],
        yerr=df["F1_cv_std"],
        fmt="none",
        capsize=6,
    )

    ax.set_ylim(
        0,
        max(
            0.65,
            df["F1_cv_media"].max()
            + 0.1,
        ),
    )

    ax.set_ylabel(
        "F1-score medio en CV"
    )

    ax.set_title(
        "Red Wine: comparación de modelos"
    )

    ax.tick_params(
        axis="x",
        rotation=15,
    )

    fig.tight_layout()

    return fig


def grafico_candy_real_vs_predicho():

    _, X_test, _, y_test = (
        preparar_holdout_candy(
            candy_meta["features"]
        )
    )

    pred = candy_model.predict(
        X_test
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 5)
    )

    ax.scatter(
        y_test,
        pred,
        alpha=0.8,
    )

    minimo = min(
        y_test.min(),
        pred.min(),
    )

    maximo = max(
        y_test.max(),
        pred.max(),
    )

    ax.plot(
        [minimo, maximo],
        [minimo, maximo],
        linestyle="--",
    )

    ax.set_xlabel(
        "Winpercent real"
    )

    ax.set_ylabel(
        "Winpercent predicho"
    )

    ax.set_title(
        "Candy: real vs predicho"
    )

    fig.tight_layout()

    return fig


def grafico_residuos_candy():

    _, X_test, _, y_test = (
        preparar_holdout_candy(
            candy_meta["features"]
        )
    )

    pred = candy_model.predict(
        X_test
    )

    residuos = (
        y_test.to_numpy()
        - pred
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 5)
    )

    ax.scatter(
        pred,
        residuos,
        alpha=0.8,
    )

    ax.axhline(
        0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Predicción"
    )

    ax.set_ylabel(
        "Residuo (real - predicho)"
    )

    ax.set_title(
        "Candy: análisis de residuos"
    )

    fig.tight_layout()

    return fig


def grafico_confusion_wine():

    _, X_test, _, y_test = (
        preparar_holdout_wine(
            wine_meta["features"]
        )
    )

    pred = wine_model.predict(
        X_test
    )

    cm = confusion_matrix(
        y_test,
        pred,
    )

    fig, ax = plt.subplots(
        figsize=(5.5, 5)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No buena",
            "Buena",
        ],
    )

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        "SVC final: matriz de confusión"
    )

    fig.tight_layout()

    return fig, cm


def grafico_roc_wine():

    _, X_test, _, y_test = (
        preparar_holdout_wine(
            wine_meta["features"]
        )
    )

    if hasattr(
        wine_model,
        "predict_proba",
    ):
        scores = (
            wine_model.predict_proba(
                X_test
            )[:, 1]
        )

    else:
        scores = (
            wine_model.decision_function(
                X_test
            )
        )

    fpr, tpr, _ = roc_curve(
        y_test,
        scores,
    )

    auc = roc_auc_score(
        y_test,
        scores,
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.plot(
        fpr,
        tpr,
        label=f"SVC (AUC = {auc:.3f})",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Azar",
    )

    ax.set_xlabel(
        "Tasa de falsos positivos"
    )

    ax.set_ylabel(
        "Tasa de verdaderos positivos"
    )

    ax.set_title(
        "Red Wine: curva ROC del modelo final"
    )

    ax.legend()

    fig.tight_layout()

    return fig, auc


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📊 Proyecto ML"
)

st.sidebar.caption(
    "Candy Power Ranking + Red Wine Quality"
)

pagina = st.sidebar.radio(
    "Navegación",
    [
        "🏠 Inicio",
        "🧭 Proceso ML",
        "🍬 Candy",
        "🍷 Red Wine",
        "🧠 Redes neuronales",
    ],
)


# ============================================================
# INICIO
# ============================================================

if pagina == "🏠 Inicio":

    st.title(
        "Proyecto de Ciencia de Datos y Machine Learning"
    )

    st.write(
        """
        Dos problemas supervisados analizados mediante
        múltiples modelos, validación cruzada, ajuste de
        hiperparámetros y evaluación sobre un conjunto
        TEST independiente.
        """
    )

    st.markdown(
        "## Resultado final"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🍬 Candy"
        )

        st.metric(
            "Modelo final",
            candy_meta["modelo"]
            .replace("_", " ")
            .title(),
        )

        a, b = st.columns(2)

        a.metric(
            "RMSE CV",
            f"{candy_meta['cv_rmse']:.4f}",
        )

        b.metric(
            "RMSE TEST",
            f"{candy_meta['metricas_test']['RMSE']:.4f}",
        )

        st.pyplot(
            grafico_modelos_candy()
        )

    with col2:

        st.subheader(
            "🍷 Red Wine"
        )

        st.metric(
            "Modelo final",
            wine_meta["modelo"]
            .replace("_", " ")
            .upper(),
        )

        a, b, c = st.columns(3)

        a.metric(
            "F1 CV",
            f"{wine_meta['cv_f1']:.4f}",
        )

        b.metric(
            "F1 TEST",
            f"{wine_meta['metricas_test']['F1-score']:.4f}",
        )

        c.metric(
            "ROC-AUC",
            f"{wine_meta['metricas_test']['ROC-AUC']:.4f}",
        )

        st.pyplot(
            grafico_modelos_wine()
        )

    st.info(
        """
        TEST no fue utilizado para elegir ni ajustar los
        modelos. La selección se realizó mediante validación
        cruzada exclusivamente dentro de TRAIN.
        """
    )


# ============================================================
# PROCESO ML
# ============================================================

elif pagina == "🧭 Proceso ML":

    st.title(
        "Proceso completo de Machine Learning"
    )

    st.caption(
        "Recorrido didáctico desde los datos hasta la predicción."
    )

    pasos = st.tabs(
        [
            "1️⃣ Datos",
            "2️⃣ Limpieza",
            "3️⃣ División",
            "4️⃣ Modelos",
            "5️⃣ Validación",
            "6️⃣ Tuning",
            "7️⃣ TEST",
            "8️⃣ Predicción",
        ]
    )

    with pasos[0]:

        encabezado_paso(
            1,
            "Datos",
            """
            Se parte de dos datasets independientes:
            Candy Power Ranking y Red Wine Quality.
            """,
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Candy",
            f"{len(candy_df)} filas",
        )

        c2.metric(
            "Red Wine",
            f"{len(wine_df)} filas",
        )

        st.write(
            "Candy:"
        )

        st.dataframe(
            candy_df.head(10),
            use_container_width=True,
        )

        st.write(
            "Red Wine:"
        )

        st.dataframe(
            wine_df.head(10),
            use_container_width=True,
        )

    with pasos[1]:

        encabezado_paso(
            2,
            "Limpieza y preprocesamiento",
            """
            Se revisan valores faltantes, duplicados y
            valores atípicos mediante IQR. Los valores
            atípicos se reportan y no se eliminan
            automáticamente.
            """,
        )

        candy_clean = pd.read_csv(
            RESULTS
            / "candy_limpieza_resumen.csv"
        )

        wine_clean = pd.read_csv(
            RESULTS
            / "redwine_limpieza_resumen.csv"
        )

        st.subheader(
            "Candy"
        )

        st.dataframe(
            candy_clean,
            use_container_width=True,
        )

        st.subheader(
            "Red Wine"
        )

        st.dataframe(
            wine_clean,
            use_container_width=True,
        )

        st.success(
            """
            La imputación no se aprende antes de dividir
            los datos: forma parte del Pipeline del modelo,
            evitando fuga de información.
            """
        )

    with pasos[2]:

        encabezado_paso(
            3,
            "TRAIN y TEST",
            """
            Se reserva el 20 % para TEST y el 80 % para
            entrenamiento. TEST permanece independiente
            hasta la evaluación final.
            """,
        )

        c1, c2 = st.columns(2)

        with c1:

            st.subheader(
                "Candy"
            )

            st.metric(
                "TRAIN",
                "68 muestras",
            )

            st.metric(
                "TEST",
                "17 muestras",
            )

        with c2:

            st.subheader(
                "Red Wine"
            )

            st.metric(
                "TRAIN",
                "1087 muestras",
            )

            st.metric(
                "TEST",
                "272 muestras",
            )

    with pasos[3]:

        encabezado_paso(
            4,
            "Modelos evaluados",
            """
            Se comparan cuatro algoritmos por problema,
            incluyendo una Red Neuronal Artificial.
            """,
        )

        c1, c2 = st.columns(2)

        with c1:

            st.subheader(
                "Candy"
            )

            st.markdown(
                """
                - Regresión lineal
                - Árbol de regresión
                - SVR RBF
                - MLPRegressor
                """
            )

        with c2:

            st.subheader(
                "Red Wine"
            )

            st.markdown(
                """
                - Regresión logística
                - Árbol de decisión
                - SVC RBF
                - MLPClassifier
                """
            )

    with pasos[4]:

        encabezado_paso(
            5,
            "Validación cruzada",
            """
            Cada candidato se evalúa mediante 5 folds
            exclusivamente sobre TRAIN.
            """,
        )

        c1, c2 = st.columns(2)

        with c1:
            st.pyplot(
                grafico_modelos_candy()
            )

        with c2:
            st.pyplot(
                grafico_modelos_wine()
            )

    with pasos[5]:

        encabezado_paso(
            6,
            "Ajuste de hiperparámetros",
            """
            Grid Search explora configuraciones dentro
            de TRAIN. TEST sigue sin participar.
            """,
        )

        st.subheader(
            "Candy"
        )

        st.dataframe(
            pd.read_csv(
                RESULTS
                / "candy_tuning.csv"
            ),
            use_container_width=True,
        )

        st.subheader(
            "Red Wine"
        )

        st.dataframe(
            pd.read_csv(
                RESULTS
                / "redwine_tuning.csv"
            ),
            use_container_width=True,
        )

    with pasos[6]:

        encabezado_paso(
            7,
            "Evaluación independiente",
            """
            Una vez elegido el modelo mediante CV, se
            mide su comportamiento sobre TEST.
            """,
        )

        c1, c2 = st.columns(2)

        with c1:

            st.subheader(
                "Candy"
            )

            st.metric(
                "Modelo",
                candy_meta["modelo"],
            )

            st.metric(
                "RMSE TEST",
                f"{candy_meta['metricas_test']['RMSE']:.4f}",
            )

            st.metric(
                "R² TEST",
                f"{candy_meta['metricas_test']['R2']:.4f}",
            )

        with c2:

            st.subheader(
                "Red Wine"
            )

            st.metric(
                "Modelo",
                wine_meta["modelo"],
            )

            st.metric(
                "F1 TEST",
                f"{wine_meta['metricas_test']['F1-score']:.4f}",
            )

            st.metric(
                "Balanced Accuracy",
                f"{wine_meta['metricas_test']['Balanced Accuracy']:.4f}",
            )

            st.metric(
                "ROC-AUC",
                f"{wine_meta['metricas_test']['ROC-AUC']:.4f}",
            )

    with pasos[7]:

        encabezado_paso(
            8,
            "Predicción",
            """
            Los modelos finales se guardan con joblib y
            pueden utilizarse para datos introducidos
            directamente por el usuario.
            """,
        )

        st.success(
            """
            Ve a las secciones **Candy** o **Red Wine**
            y abre la pestaña **Predicción interactiva**.
            """
        )


# ============================================================
# CANDY
# ============================================================

elif pagina == "🍬 Candy":

    st.title(
        "🍬 Candy Power Ranking"
    )

    tabs = st.tabs(
        [
            "📁 Dataset",
            "📊 Modelos",
            "🎯 Resultado final",
            "🧪 Predicción interactiva",
        ]
    )

    with tabs[0]:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Filas",
            candy_df.shape[0],
        )

        c2.metric(
            "Columnas",
            candy_df.shape[1],
        )

        c3.metric(
            "Predictores",
            len(
                candy_meta["features"]
            ),
        )

        st.dataframe(
            candy_df,
            use_container_width=True,
        )

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.hist(
            candy_df["winpercent"],
            bins=12,
        )

        ax.set_xlabel(
            "Winpercent"
        )

        ax.set_ylabel(
            "Frecuencia"
        )

        ax.set_title(
            "Distribución de winpercent"
        )

        fig.tight_layout()

        st.pyplot(fig)

    with tabs[1]:

        st.pyplot(
            grafico_modelos_candy()
        )

        st.dataframe(
            pd.read_csv(
                RESULTS
                / "candy_tuning.csv"
            ),
            use_container_width=True,
        )

        st.info(
            """
            Para Candy se selecciona el modelo con
            menor RMSE medio de validación cruzada.
            """
        )

    with tabs[2]:

        st.subheader(
            "Modelo final"
        )

        st.metric(
            "Modelo",
            candy_meta["modelo"],
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "RMSE CV",
            f"{candy_meta['cv_rmse']:.4f}",
        )

        c2.metric(
            "RMSE TEST",
            f"{candy_meta['metricas_test']['RMSE']:.4f}",
        )

        c3.metric(
            "R² TEST",
            f"{candy_meta['metricas_test']['R2']:.4f}",
        )

        a, b = st.columns(2)

        with a:
            st.pyplot(
                grafico_candy_real_vs_predicho()
            )

        with b:
            st.pyplot(
                grafico_residuos_candy()
            )

        st.subheader(
            "Diagnóstico TRAIN / CV / TEST"
        )

        st.dataframe(
            pd.read_csv(
                RESULTS
                / "candy_diagnostico_final.csv"
            ),
            use_container_width=True,
        )

    with tabs[3]:

        st.subheader(
            "Introduce las características del dulce"
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

        c1, c2 = st.columns(2)

        for i, feature in enumerate(
            candy_meta["features"]
        ):

            columna = (
                c1
                if i % 2 == 0
                else c2
            )

            with columna:

                if feature in binary:

                    values[feature] = int(
                        st.checkbox(
                            feature,
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
            "🍬 Predecir Candy",
            type="primary",
            use_container_width=True,
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

            st.progress(
                min(
                    max(
                        pred / 100,
                        0.0,
                    ),
                    1.0,
                )
            )

            st.write(
                """
                **Interpretación:** es la estimación del
                porcentaje de preferencia del producto
                según las características introducidas.
                """
            )


# ============================================================
# RED WINE
# ============================================================

elif pagina == "🍷 Red Wine":

    st.title(
        "🍷 Red Wine Quality"
    )

    tabs = st.tabs(
        [
            "📁 Dataset",
            "📊 Modelos",
            "🎯 Resultado final",
            "🧪 Predicción interactiva",
        ]
    )

    with tabs[0]:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Filas originales",
            wine_df.shape[0],
        )

        c2.metric(
            "Columnas",
            wine_df.shape[1],
        )

        c3.metric(
            "Predictores",
            len(
                wine_meta["features"]
            ),
        )

        st.dataframe(
            wine_df,
            use_container_width=True,
        )

        a, b = st.columns(2)

        with a:

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )

            wine_df[
                "quality"
            ].value_counts().sort_index().plot(
                kind="bar",
                ax=ax,
            )

            ax.set_xlabel(
                "Quality"
            )

            ax.set_ylabel(
                "Cantidad"
            )

            ax.set_title(
                "Distribución de quality"
            )

            fig.tight_layout()

            st.pyplot(fig)

        with b:

            clases = (
                wine_df["quality"] >= 7
            ).map(
                {
                    True: "Buena",
                    False: "No buena",
                }
            )

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )

            clases.value_counts().plot(
                kind="bar",
                ax=ax,
            )

            ax.set_ylabel(
                "Cantidad"
            )

            ax.set_title(
                "Desbalance de clases"
            )

            fig.tight_layout()

            st.pyplot(fig)

    with tabs[1]:

        st.pyplot(
            grafico_modelos_wine()
        )

        st.dataframe(
            pd.read_csv(
                RESULTS
                / "redwine_tuning.csv"
            ),
            use_container_width=True,
        )

        st.warning(
            """
            Debido al desbalance entre vinos Buena y
            No buena, Accuracy por sí sola no es
            suficiente. El criterio principal utilizado
            es F1-score.
            """
        )

    with tabs[2]:

        st.subheader(
            "Modelo final: SVC RBF"
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "F1 CV",
            f"{wine_meta['cv_f1']:.4f}",
        )

        b.metric(
            "F1 TEST",
            f"{wine_meta['metricas_test']['F1-score']:.4f}",
        )

        c.metric(
            "Balanced Accuracy",
            f"{wine_meta['metricas_test']['Balanced Accuracy']:.4f}",
        )

        d.metric(
            "ROC-AUC",
            f"{wine_meta['metricas_test']['ROC-AUC']:.4f}",
        )

        col1, col2 = st.columns(2)

        with col1:

            fig, cm = (
                grafico_confusion_wine()
            )

            st.pyplot(fig)

            st.caption(
                f"""
                TN={cm[0,0]},
                FP={cm[0,1]},
                FN={cm[1,0]},
                TP={cm[1,1]}
                """
            )

        with col2:

            fig, auc = (
                grafico_roc_wine()
            )

            st.pyplot(fig)

            st.caption(
                f"ROC-AUC TEST = {auc:.4f}"
            )

        st.subheader(
            "Diagnóstico TRAIN / CV / TEST"
        )

        st.dataframe(
            pd.read_csv(
                RESULTS
                / "redwine_diagnostico_final.csv"
            ),
            use_container_width=True,
        )

    with tabs[3]:

        st.subheader(
            "Introduce las propiedades fisicoquímicas"
        )

        values = {}

        c1, c2 = st.columns(2)

        for i, feature in enumerate(
            wine_meta["features"]
        ):

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

            columna = (
                c1
                if i % 2 == 0
                else c2
            )

            with columna:

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
            "🍷 Clasificar vino",
            type="primary",
            use_container_width=True,
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

            if hasattr(
                wine_model,
                "decision_function",
            ):

                score = float(
                    wine_model
                    .decision_function(
                        entrada
                    )[0]
                )

            else:

                score = None

            if pred == 1:

                st.success(
                    "Clasificación predicha: BUENA"
                )

            else:

                st.warning(
                    "Clasificación predicha: NO BUENA"
                )

            if score is not None:

                st.metric(
                    "Score de decisión SVC",
                    f"{score:.4f}",
                )

                st.caption(
                    """
                    Un valor positivo favorece la clase
                    Buena; un valor negativo favorece
                    No buena. No es una probabilidad.
                    """
                )


# ============================================================
# REDES NEURONALES
# ============================================================

else:

    st.title(
        "🧠 Redes Neuronales Artificiales"
    )

    st.markdown(
        """
        ## Perceptrón Multicapa

        En ambos problemas se implementó una
        **Red Neuronal Artificial de tipo Perceptrón
        Multicapa (MLP)**.

        La arquitectura es **feedforward**:

        **Entrada → capas ocultas → salida**

        Durante el entrenamiento, el error se propaga
        hacia atrás mediante **backpropagation** para
        ajustar los pesos de la red.
        """
    )

    st.markdown(
        """
        ```text
        Variables de entrada
                │
                ▼
        ┌───────────────────┐
        │   Capa oculta 1   │
        │      ReLU         │
        └────────┬──────────┘
                 │
                 ▼
        ┌───────────────────┐
        │   Capa oculta 2   │
        │      ReLU         │
        └────────┬──────────┘
                 │
                 ▼
             Salida
        ```
        """
    )

    c1, c2 = st.columns(2)

    with c1:

        st.subheader(
            "Candy: MLPRegressor"
        )

        candy_models = pd.read_csv(
            RESULTS
            / "candy_modelos_comparacion.csv"
        )

        st.dataframe(
            candy_models[
                candy_models["modelo"]
                == "mlp_regressor"
            ],
            use_container_width=True,
        )

        st.write(
            """
            La RNA resuelve un problema de
            **regresión**, produciendo un valor
            continuo de `winpercent`.
            """
        )

    with c2:

        st.subheader(
            "Red Wine: MLPClassifier"
        )

        wine_models = pd.read_csv(
            RESULTS
            / "redwine_modelos_comparacion.csv"
        )

        st.dataframe(
            wine_models[
                wine_models["modelo"]
                == "mlp_classifier"
            ],
            use_container_width=True,
        )

        st.write(
            """
            La RNA resuelve un problema de
            **clasificación binaria**:
            Buena / No buena.
            """
        )

    st.info(
        """
        La inclusión de RNA no implica que deba ser el
        modelo final. Todos los algoritmos se comparan
        bajo el mismo protocolo y la selección se basa
        en validación cruzada.
        """
    )
