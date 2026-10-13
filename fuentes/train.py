import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def entrenar_y_registrar():
    mlflow.set_experiment("Actividad5_Iris_Local")
    
    data_path = "datos/datos_limp/iris_procesado.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró {data_path}. Ejecuta primero datos_prep.py")
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Modelo 1: Regresión Logística
    with mlflow.start_run(run_name="Regresion_Logistica"):
        param_grid_lr = {'C': [0.01, 0.1, 1.0, 10.0], 'solver': ['lbfgs'], 'max_iter': [200]}
        grid_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, scoring='accuracy')
        grid_lr.fit(X_train, y_train)
        
        best_lr = grid_lr.best_estimator_
        preds_lr = best_lr.predict(X_test)
        
        mlflow.log_params(grid_lr.best_params_)
        mlflow.log_metric("accuracy", accuracy_score(y_test, preds_lr))
        mlflow.log_metric("precision", precision_score(y_test, preds_lr, average='weighted'))
        mlflow.log_metric("recall", recall_score(y_test, preds_lr, average='weighted'))
        mlflow.log_metric("f1_score", f1_score(y_test, preds_lr, average='weighted'))
        mlflow.sklearn.log_model(best_lr, "modelo_regresion_logistica")
        print("[MLflow] Regresión Logística registrada con éxito.")

    # Modelo 2: Random Forest
    with mlflow.start_run(run_name="Random_Forest"):
        param_grid_rf = {'n_estimators': [10, 50, 100], 'max_depth': [None, 3, 5], 'random_state': [42]}
        grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='accuracy')
        grid_rf.fit(X_train, y_train)
        
        best_rf = grid_rf.best_estimator_
        preds_rf = best_rf.predict(X_test)
        
        mlflow.log_params(grid_rf.best_params_)
        mlflow.log_metric("accuracy", accuracy_score(y_test, preds_rf))
        mlflow.log_metric("precision", precision_score(y_test, preds_rf, average='weighted'))
        mlflow.log_metric("recall", recall_score(y_test, preds_rf, average='weighted'))
        mlflow.log_metric("f1_score", f1_score(y_test, preds_rf, average='weighted'))
        mlflow.sklearn.log_model(best_rf, "modelo_random_forest")
        print("[MLflow] Random Forest registrado con éxito.")

if __name__ == "__main__":
    entrenar_y_registrar()