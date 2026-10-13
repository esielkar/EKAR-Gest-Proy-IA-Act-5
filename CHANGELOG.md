
# Registro de Cambios (Changelog)

Todas las modificaciones notables correspondientes al desarrollo de la Actividad 5 serán documentadas en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere al versionado semántico [Semantic Versioning](https://semver.org/lang/es/).

## [1.0.0] - 2026-10-12

### Añadido
- **Estructura del Proyecto:** Creación de la arquitectura de directorios obligatoria (`datos/datos_ini`, `datos/datos_limp`, `fuentes/`).
- **Versionamiento de Datos:**
  - Incorporación del dataset original Iris (`iris.csv`) en la ruta de datos crudos.
  - Implementación del script `fuentes/datos_prep.py` para la validación de valores nulos y estandarización de variables mediante `StandardScaler`, generando el archivo versionado `iris_procesado.csv`.
- **Modelado y Optimización:**
  - Desarrollo del script `fuentes/train.py` para el entrenamiento concurrente de dos algoritmos de clasificación de distinta naturaleza matemática: **Regresión Logística** y **Random Forest Classifier**.
  - Configuración de estrategias de búsqueda de hiperparámetros mediante `GridSearchCV` acopladas a validación cruzada de 5 particiones (`cv=5`).
- **Trazabilidad y MLOps (MLflow):**
  - Integración completa con **MLflow Tracking** para el registro automatizado de hiperparámetros, métricas estandarizadas de evaluación (*Accuracy, Precision, Recall, F1-Score*) y serialización de artefactos binarios de los modelos.
- **Documentación:**
  - Redacción del manual de reproducción en el archivo `README.md` y formalización del control de cambios en `CHANGELOG.md`.