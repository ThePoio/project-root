import pandas as pd
from sklearn.model_selection import train_test_split
import os

def cargar_y_limpiar_datos(config):
    # Definir rutas
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta = os.path.join(base, 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Cargar CSV
    df = pd.read_csv(ruta)

    # 1. Limpiar TotalCharges: convertir a numérico y rellenar vacíos
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # 2. Eliminar columna innecesaria
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # 3. Solo convertir el target (Churn) a binario
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Separar X e y
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Dividir datos (manteniendo la aleatoriedad según configuración)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.get('test_size', 0.2),
        random_state=config.get('random_state', 42)
    )

    # Guardar resultados
    guardar_datos_procesados(base, X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test

def guardar_datos_procesados(base_path, X_train, X_test, y_train, y_test):
    ruta_salida = os.path.join(base_path, 'data', 'processed')
    os.makedirs(ruta_salida, exist_ok=True)
    
    # Guardar sin transformar categorías
    X_train.to_csv(os.path.join(ruta_salida, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(ruta_salida, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(ruta_salida, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(ruta_salida, 'y_test.csv'), index=False)
    
    print(f" Archivos generados correctamente en: {ruta_salida}")

if __name__ == "__main__":
    config = {'test_size': 0.2, 'random_state': 42}
    cargar_y_limpiar_datos(config)
