# Práctica ML - Candy y Red Wine

## Qué se hizo
- Candy: regresión lineal para predecir `winpercent`.
- Red Wine: árbol de decisión para clasificar calidad como buena (>=7) o no buena (<7), propuesta metodológica para poder generar matriz de confusión.

## Tres procesos de limpieza
1. Valores faltantes: detección y tratamiento si aparecen.
2. Duplicados: detección y eliminación.
3. Valores atípicos: detección mediante IQR y reporte; no se eliminan automáticamente porque su eliminación requiere justificación del dominio.

## Ejecución
1. Instalar Python.
2. Abrir esta carpeta en VS Code.
3. En la terminal:
   `pip install -r requirements.txt`
4. Ejecutar:
   `python main.py`

Los resultados se guardan en `resultados/`.

## Nota académica
La práctica entregada indica limpieza, separación entrenamiento/prueba, implementación de modelos, pruebas y validación con al menos 4 parámetros y matriz de confusión. La elección concreta de modelos y el umbral de calidad de Red Wine son una propuesta de implementación y deben ajustarse si la docente asignó otra área o criterio.
