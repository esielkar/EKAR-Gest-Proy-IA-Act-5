import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def preparar_datos():
    os.makedirs("datos/datos_ini", exist_ok=True)
    os.makedirs("datos/datos_limp", exist_ok=True)
    
    # Cargar dataset original
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Guardar en datos_ini
    path_ini = "datos/datos_ini/iris.csv"
    df.to_csv(path_ini, index=False)
    print(f"[OK] Datos originales guardados en: {path_ini}")
    
    # Limpieza y estandarización
    print(f"Valores nulos detectados: {df.isnull().sum().sum()}")
    scaler = StandardScaler()
    feature_cols = iris.feature_names
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Guardar en datos_limp
    path_limp = "datos/datos_limp/iris_procesado.csv"
    df.to_csv(path_limp, index=False)
    print(f"[OK] Datos limpios guardados en: {path_limp}")

if __name__ == "__main__":
    preparar_datos()