# Actividad 5: Entrenamiento, Optimización y Trazabilidad con MLflow

## Descripción General
Este repositorio contiene el desarrollo de un proyecto aplicado de Machine Learning correspondiente a la **Actividad 5** del Master de IA del Tecmilenio. El objetivo principal es implementar un flujo de trabajo reproducible y estandarizado para la clasificación multiclase utilizando el clásico **Iris Dataset**, integrando prácticas de versionamiento de datos, validación cruzada, ajuste de hiperparámetros (*Grid Search*) y registro de experimentos mediante **MLflow Tracking**.

## Estructura del Repositorio
```text
EKAR-GEST-PROY-IA-ACT-5/
│── datos/
│   ├── datos_ini/
│   │   └── iris.csv                 # Dataset original sin procesar (datos crudos)
│   └── datos_limp/
│       └── iris_procesado.csv       # Dataset estandarizado y versionado
│── fuentes/
│   ├── datos_prep.py                # Script de automatización de limpieza y normalización
│   └── train.py                     # Script de entrenamiento, GridSearch y logging en MLflow
│── requirements.txt                 # Dependencias del entorno de desarrollo
│── README.md                        # Guía de reproducción y arquitectura
└── CHANGELOG.md                     # Registro histórico de versiones y cambios
```

### Requisitos del Sistema y Dependencias
El proyecto está optimizado para ejecutarse en entornos locales basados en Python 3.8+. Las librerías requeridas se encuentran especificadas en el archivo requirements.txt:

- mlflow >= 2.8.0
- scikit-learn >= 1.2.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

### Guía de Reproducción Paso a Paso
1. Clonar o descargar el repositorio y abrir la carpeta raíz en tu entorno de desarrollo (por ejemplo, VS Code).
2. Crear y activar un entorno virtual de Python:
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Instalar las dependencias del proyecto:
```bash
pip install -r requirements.txt
```
4. Ejecutar el script de preparación y versionamiento de datos:
```bash
python fuentes/datos_prep.py
```
Esto generará los directorios datos_ini y datos_limp con sus respectivos archivos CSV.

5. Ejecutar el entrenamiento, validación cruzada y registro de experimentos:
```bash
python fuentes/train.py
```
Esto entrenará los modelos de Regresión Logística y Random Forest utilizando GridSearchCV y almacenará los artefactos y métricas en la carpeta local mlruns/.

6. Visualizar el panel de MLflow:
```bash
mlflow ui
```
Abre en tu navegador web la URL local generada (usualmente http://127.0.0.1:5000) para inspeccionar métricas, parámetros y modelos comparativos.

## Decisiones Técnicas y Arquitectura
- Separación de Linaje de Datos: Se implementó una división estricta entre los datos originales (datos_ini) y los datos transformados (datos_limp), asegurando la auditabilidad ante cambios en las fases de preprocesamiento.

- Estandarización: Se aplicó StandardScaler para normalizar las características continuas, optimizando la convergencia de los modelos lineales.

- Validación Cruzada: Se configuró un esquema de 5 particiones (cv=5) combinado con búsqueda en rejilla para prevenir el sobreajuste.