# Ciencia de Datos y Machine Learning
## Candy Power Ranking + Red Wine Quality

Proyecto académico de **Ciencia de Datos y Machine Learning — Universidad Mayor de San Simón (UMSS)**.

El proyecto implementa un flujo completo de Machine Learning sobre dos datasets:

- **Candy Power Ranking** → problema de **regresión** para estimar `winpercent`.
- **Red Wine Quality** → problema de **clasificación binaria** para predecir si un vino es **Buena** (`quality >= 7`) o **No buena** (`quality < 7`).

Incluye preparación y limpieza de datos, prevención de fuga de información mediante `Pipeline`, separación TRAIN / TEST, validación cruzada de 5 folds, 8 modelos, Redes Neuronales Artificiales MLP, múltiples métricas, ajuste de hiperparámetros, análisis TRAIN / CV / TEST, persistencia con `joblib`, predicción desde terminal, aplicación web interactiva con Streamlit y visualizaciones.

---

# Índice

1. [Instalación y ejecución desde cero](#1-instalación-y-ejecución-desde-cero)
2. [Requisitos](#2-requisitos)
3. [Clonar el repositorio](#3-clonar-el-repositorio)
4. [Crear y activar el entorno virtual](#4-crear-y-activar-el-entorno-virtual)
5. [Instalar dependencias](#5-instalar-dependencias)
6. [Verificar la instalación](#6-verificar-la-instalación)
7. [Ejecutar la aplicación Streamlit](#7-ejecutar-la-aplicación-streamlit)
8. [Ejecutar predicciones desde terminal](#8-ejecutar-predicciones-desde-terminal)
9. [Reproducir experimentos](#9-reproducir-experimentos)
10. [Repetir tuning y regenerar modelos](#10-repetir-tuning-y-regenerar-modelos)
11. [Regenerar gráficos](#11-regenerar-gráficos)
12. [Flujo rápido para una máquina nueva](#12-flujo-rápido-para-una-máquina-nueva)
13. [Solución de problemas](#13-solución-de-problemas)
14. [Descripción del proyecto](#14-descripción-del-proyecto)
15. [Metodología](#15-metodología)
16. [Modelos implementados](#16-modelos-implementados)
17. [Redes Neuronales Artificiales](#17-redes-neuronales-artificiales)
18. [Métricas](#18-métricas)
19. [Resultados finales](#19-resultados-finales)
20. [Aplicación interactiva](#20-aplicación-interactiva)
21. [Estructura del proyecto](#21-estructura-del-proyecto)
22. [Reproducibilidad](#22-reproducibilidad)
23. [Limitaciones](#23-limitaciones)
24. [Autores](#24-autores)

---

# 1. Instalación y ejecución desde cero

Esta es la sección principal para ejecutar el proyecto en una computadora nueva.

```text
Clonar repositorio
        ↓
Crear .venv
        ↓
Activar .venv
        ↓
Actualizar pip
        ↓
Instalar requirements.txt
        ↓
Ejecutar check_environment.py
        ↓
ENTORNO LISTO
        ↓
Ejecutar Streamlit
```

# 2. Requisitos

Se necesita:

- Git
- Python
- pip
- conexión a Internet durante la instalación de dependencias
- navegador web para utilizar Streamlit

Entorno utilizado durante el desarrollo y validación:

```text
Windows 11
Python 3.14.7
```

Comprobar Python:

```bash
python --version
```

Comprobar Git:

```bash
git --version
```

# 3. Clonar el repositorio

```bash
git clone https://github.com/JofreTiconaPlata/Ciencia-de-Datos-y-Machine-Learning.git
cd Ciencia-de-Datos-y-Machine-Learning
```

Mientras el desarrollo final permanezca en la rama `feat/proyecto-final`:

```bash
git switch feat/proyecto-final
```

Cuando esta rama sea fusionada a `main`, ese paso dejará de ser necesario.

# 4. Crear y activar el entorno virtual

Se recomienda utilizar siempre un entorno virtual `.venv`.

## Windows — Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
```

## Windows — PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cuando el entorno esté activo, la terminal debería mostrar:

```text
(.venv)
```

En Git Bash, comprobar el intérprete:

```bash
which python
```

Debe apuntar aproximadamente a:

```text
.../.venv/Scripts/python
```

# 5. Instalar dependencias

Con `.venv` activo:

## 5.1 Actualizar pip

```bash
python -m pip install --upgrade pip
```

## 5.2 Instalación recomendada

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` contiene las dependencias directas necesarias para ejecutar el proyecto, incluyendo NumPy, pandas, Matplotlib, scikit-learn, joblib y Streamlit.

## 5.3 Reproducción exacta del entorno de desarrollo

También existe:

```text
requirements-lock.txt
```

Para intentar reproducir exactamente todas las versiones instaladas durante el desarrollo:

```bash
python -m pip install -r requirements-lock.txt
```

Normalmente debe utilizarse **solo uno** de los dos métodos. Para uso normal se recomienda `requirements.txt`.

# 6. Verificar la instalación

Antes de ejecutar la aplicación:

```bash
python scripts/check_environment.py
```

El resultado correcto debe finalizar con:

```text
ENTORNO LISTO
```

El verificador comprueba:

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn
- joblib
- Streamlit
- datasets
- modelos `.joblib`
- metadatos
- escritura en `resultados/`
- carga real de los modelos
- predicción de prueba de Candy
- predicción de prueba de Red Wine

# 7. Ejecutar la aplicación Streamlit

Con `.venv` activo:

```bash
python -m streamlit run app/app.py
```

Streamlit mostrará normalmente:

```text
Local URL: http://localhost:8501
```

Abrir en el navegador:

```text
http://localhost:8501
```

En la primera ejecución Streamlit puede solicitar opcionalmente un correo electrónico. Puede dejarse vacío y presionar `Enter`.

Para detener Streamlit:

```text
Ctrl + C
```

# 8. Ejecutar predicciones desde terminal

También existe una interfaz CLI:

```bash
python scripts/predict_cli.py
```

Permite seleccionar:

```text
1. Candy
2. Red Wine
```

y luego introducir manualmente las variables solicitadas.

# 9. Reproducir experimentos

Para ejecutar los experimentos principales:

```bash
python main.py
```

Este proceso carga los datasets, prepara los datos, separa TRAIN / TEST, evalúa modelos, ejecuta validación cruzada, genera métricas y guarda resultados en `resultados/`.

# 10. Repetir tuning y regenerar modelos

Para repetir la búsqueda de hiperparámetros:

```bash
python scripts/tune_and_save.py
```

Este script:

1. carga los datasets;
2. prepara TRAIN y TEST;
3. ejecuta validación cruzada exclusivamente sobre TRAIN;
4. realiza Grid Search;
5. selecciona el modelo utilizando CV;
6. evalúa después sobre TEST;
7. genera diagnóstico TRAIN / CV / TEST;
8. persiste los modelos finales.

Modelos finales:

```text
models/candy_final.joblib
models/wine_final.joblib
```

Metadatos:

```text
models/candy_metadata.json
models/wine_metadata.json
```

# 11. Regenerar gráficos

```bash
python scripts/generate_final_plots.py
```

Los gráficos se almacenan en:

```text
resultados/graficos_finales/
```

# 12. Flujo rápido para una máquina nueva

## Git Bash

```bash
git clone https://github.com/JofreTiconaPlata/Ciencia-de-Datos-y-Machine-Learning.git
cd Ciencia-de-Datos-y-Machine-Learning
git switch feat/proyecto-final
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
python -m streamlit run app/app.py
```

## PowerShell

```powershell
git clone https://github.com/JofreTiconaPlata/Ciencia-de-Datos-y-Machine-Learning.git
cd Ciencia-de-Datos-y-Machine-Learning
git switch feat/proyecto-final
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
python -m streamlit run app/app.py
```

Si `check_environment.py` termina en `ENTORNO LISTO`, la instalación fue correcta.

# 13. Solución de problemas

## `python` no se reconoce

```bash
python --version
```

Si no funciona, instalar Python y habilitarlo en `PATH`.

## `ModuleNotFoundError`

Comprobar que `.venv` esté activo:

```bash
which python
```

Después:

```bash
python -m pip install -r requirements.txt
```

## PowerShell bloquea `Activate.ps1`

Puede utilizarse Git Bash:

```bash
source .venv/Scripts/activate
```

## Streamlit no abre el navegador

```bash
python -m streamlit run app/app.py
```

y abrir manualmente:

```text
http://localhost:8501
```

## Puerto 8501 ocupado

```bash
python -m streamlit run app/app.py --server.port 8502
```

Abrir:

```text
http://localhost:8502
```

## Error al cargar pandas / NumPy / scikit-learn en Windows

Si aparece:

```text
Una directiva de Control de aplicaciones bloqueó este archivo
```

el problema corresponde a una política de seguridad de Windows que bloquea extensiones nativas de Python. No es un error del código del proyecto. En equipos institucionales o administrados se recomienda consultar al administrador antes de modificar políticas de seguridad.

## Los modelos `.joblib` no cargan

```bash
python scripts/check_environment.py
```

Si continúa el problema:

```bash
python -m pip install -r requirements.txt
```

# 14. Descripción del proyecto

## Candy Power Ranking

Objetivo: predecir `winpercent`.

Tipo: **Regresión**.

Datos:

```text
85 productos
13 columnas originales
11 variables predictoras
```

Se excluye `competitorname` de los predictores.

## Red Wine Quality

Objetivo: clasificación binaria.

```text
quality >= 7  -> Buena
quality < 7   -> No buena
```

Datos originales:

```text
1599 filas
12 columnas
```

Después de eliminar duplicados:

```text
1359 filas
```

Se utilizan 11 variables fisicoquímicas como predictores.

# 15. Metodología

```text
Dataset
   |
   v
Inspección inicial
   |
   +-- Valores faltantes
   +-- Duplicados
   +-- IQR sobre predictores
   |
   v
Separación TRAIN / TEST
   |
   +----------------------------+
   |                            |
   v                            v
TRAIN                          TEST
   |                            |
   v                            |
Pipeline                        |
   |                            |
   +-- Imputación               |
   +-- Escalado                 |
   +-- Modelo                   |
   |                            |
   v                            |
Validación cruzada 5-fold       |
   |                            |
   v                            |
Tuning / Grid Search            |
   |                            |
   v                            |
Selección por CV                |
   |                            |
   +-------------+--------------+
                 |
                 v
       Evaluación final TEST
                 |
                 v
        Persistencia joblib
                 |
                 v
          Predicción usuario
```

La imputación se realiza dentro de los `Pipeline`, evitando aprender información desde TEST. TEST no participa en la selección del modelo ni en el ajuste de hiperparámetros.

# 16. Modelos implementados

Se implementan **8 modelos**.

## Candy — Regresión

1. Regresión Lineal
2. Árbol de Regresión
3. SVR con kernel RBF
4. `MLPRegressor`

## Red Wine — Clasificación

1. Regresión Logística
2. Árbol de Decisión
3. SVC con kernel RBF
4. `MLPClassifier`

# 17. Redes Neuronales Artificiales

```text
Candy    -> MLPRegressor
Red Wine -> MLPClassifier
```

Ambos corresponden a Redes Neuronales Artificiales de tipo **Perceptrón Multicapa (MLP)**, con arquitectura **feedforward** y entrenamiento mediante **backpropagation**.

La RNA se evalúa bajo el mismo protocolo que los demás algoritmos y no se fuerza su selección como modelo final.

# 18. Métricas

## Candy

```text
MAE
MSE
RMSE
R²
MedAE
```

Criterio principal de selección:

```text
menor RMSE medio en validación cruzada
```

## Red Wine

```text
Accuracy
Precision
Recall
F1-score
Balanced Accuracy
ROC-AUC
```

Debido al desbalance entre clases, Accuracy no se utiliza de forma aislada.

Criterio principal:

```text
mayor F1-score medio en validación cruzada estratificada
```

# 19. Resultados finales

## Candy

Modelo final después del tuning:

```text
Árbol de regresión
```

Configuración seleccionada:

```text
max_depth = 3
min_samples_leaf = 1
```

Resultados:

```text
RMSE CV medio : 11.7910
RMSE TEST     : 13.3053
R² TEST       : 0.1170
```

La Regresión Lineal obtuvo un RMSE CV muy cercano: `11.8154`. Por ello, la diferencia entre ambos modelos debe interpretarse con cautela debido al reducido tamaño del dataset.

## Red Wine

Modelo final:

```text
SVC RBF
```

Configuración:

```text
C = 1.0
class_weight = balanced
```

Resultados:

```text
F1 CV medio       : 0.5146
F1 TEST           : 0.5225
Balanced Accuracy : 0.7961
ROC-AUC TEST      : 0.8838
```

# 20. Aplicación interactiva

La aplicación Streamlit dispone de:

```text
🏠 Inicio
🧭 Proceso ML
🍬 Candy
🍷 Red Wine
🧠 Redes neuronales
```

Incluye:

- resumen general;
- visualización de datasets;
- limpieza;
- TRAIN / TEST;
- validación cruzada;
- tuning;
- comparación de modelos;
- métricas;
- distribución de `winpercent`;
- distribución de `quality`;
- desbalance de clases;
- real vs. predicho;
- residuos;
- matriz de confusión;
- curva ROC;
- explicación de RNA;
- predicciones con datos introducidos por el usuario.

La aplicación utiliza los modelos `.joblib` existentes y no reentrena todo cada vez que se inicia.

# 21. Estructura del proyecto

```text
.
├── app/
│   └── app.py
├── data/
│   ├── candy-data.csv
│   └── winequality-red.csv
├── models/
│   ├── candy_final.joblib
│   ├── candy_metadata.json
│   ├── wine_final.joblib
│   └── wine_metadata.json
├── resultados/
│   ├── graficos_finales/
│   ├── candy_tuning.csv
│   ├── candy_diagnostico_final.csv
│   ├── redwine_tuning.csv
│   ├── redwine_diagnostico_final.csv
│   └── ...
├── scripts/
│   ├── check_environment.py
│   ├── generate_final_plots.py
│   ├── predict_cli.py
│   └── tune_and_save.py
├── src/
│   ├── __init__.py
│   ├── candy.py
│   ├── config.py
│   ├── data_processing.py
│   ├── evaluation.py
│   ├── models_candy.py
│   ├── models_wine.py
│   └── wine.py
├── main.py
├── requirements.txt
├── requirements-lock.txt
├── .python-version
├── .gitignore
├── .gitattributes
└── README.md
```

# 22. Reproducibilidad

Configuración general:

```text
random_state = 42
test_size = 0.20
```

Candy:

```text
KFold
n_splits = 5
shuffle = True
random_state = 42
```

Red Wine:

```text
StratifiedKFold
n_splits = 5
shuffle = True
random_state = 42
```

Además:

- los modelos se encuentran persistidos;
- los datasets forman parte del repositorio;
- las dependencias están registradas;
- los resultados se exportan a CSV;
- los gráficos pueden regenerarse;
- existe un verificador automático del entorno.

# 23. Limitaciones

## Candy

El dataset contiene solamente 85 productos. Con la división 80/20:

```text
TRAIN = 68
TEST  = 17
```

Las diferencias pequeñas entre modelos deben interpretarse con precaución.

El modelo final presenta aproximadamente:

```text
R² TEST = 0.117
```

por lo que una parte considerable de la variabilidad de `winpercent` no es explicada por las variables disponibles.

## Red Wine

Existe un importante desbalance de clases. Por ello se utilizan F1-score, Balanced Accuracy y ROC-AUC además de Accuracy.

También debe considerarse que convertir `quality` en `Buena / No buena` reduce información originalmente ordinal.

# 24. Autores

- Jose Aima Lupa
- Jofre Ticona Plata
- Wilber Lancea Mamani
- Ethan Gerl Serelis
- Ramiro Eduardo Linarez Cosio
- Diego Cesar Cerezo Felipez

**Asignatura:** Ciencia de Datos y Machine Learning  
**Universidad:** Universidad Mayor de San Simón — UMSS
