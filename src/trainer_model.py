import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC # Importamos Support Vector Classifier

def train_and_save_models():
    # 1. DEFINICIÓN DE RUTAS
    # Identifica la ruta del proyecto y define dónde leer/guardar archivos
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_path = os.path.join(base, 'data', 'processed')
    models_path = os.path.join(base, 'models')
    
    # Crea la carpeta 'models/' automáticamente si no existe para evitar errores
    os.makedirs(models_path, exist_ok=True)
    
    # 2. CARGA DE DATOS
    # Lee los archivos CSV limpios (X_train, y_train) del Data Engineer
    X_train = pd.read_csv(os.path.join(processed_data_path, 'X_train.csv'))
    # .values.ravel() convierte la tabla en un vector simple para el modelo
    y_train = pd.read_csv(os.path.join(processed_data_path, 'y_train.csv')).values.ravel()

    # 3. TRANSFORMACIÓN DE DATOS (Dummies)
    # Convierte texto (ej. "DSL", "Male") a columnas numéricas (0 y 1)
    X_train = pd.get_dummies(X_train)
    print(f"Datos preparados. Columnas procesadas: {X_train.shape[1]}")

    # 4. ENTRENAMIENTO DE MODELOS
    # Definimos el diccionario con los 3 algoritmos: Regresión Logística, Random Forest y SVM
    modelos = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm": SVC(probability=True) # probability=True permite calcular la probabilidad del Churn
    }

    # Entrena cada modelo y guárdalo
    for nombre, modelo in modelos.items():
        print(f"Entrenando {nombre}...")
        modelo.fit(X_train, y_train)
        
        # 5. GUARDADO (SERIALIZACIÓN)
        # Joblib "congela" el modelo entrenado en un archivo .pkl para usarlo después
        ruta_modelo = os.path.join(models_path, f"{nombre}.pkl")
        joblib.dump(modelo, ruta_modelo)
        print(f" {nombre} guardado en: {ruta_modelo}")

if __name__ == "__main__":
    # Ejecuta el entrenamiento principal
    train_and_save_models()
    print("--- Pipeline de entrenamiento finalizado con éxito ---")