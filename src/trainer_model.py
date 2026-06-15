import pandas as pd
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
    X_test = pd.read_csv(os.path.join(processed_data_path, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(processed_data_path, 'y_test.csv')).values.ravel()

    # 3. TRANSFORMACIÓN DE DATOS
    # Codifica variables categóricas y escala numéricas dentro del propio modelo
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X_train.select_dtypes(exclude=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ]
    )

    print(f"Datos preparados. Columnas numéricas: {len(numeric_features)} | categóricas: {len(categorical_features)}")

    # 4. ENTRENAMIENTO DE MODELOS
    # Definimos el diccionario con los 3 algoritmos: Regresión Logística, Random Forest y SVM
    modelos = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm": SVC()
    }

    # Entrena cada modelo y guárdalo
    for nombre, modelo in modelos.items():
        print(f"Entrenando {nombre}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', modelo),
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(
            f" {nombre} -> accuracy: {accuracy:.4f} | recall: {recall:.4f} | f1: {f1:.4f}"
        )
        
        # 5. GUARDADO (SERIALIZACIÓN)
        # Joblib "congela" el pipeline completo para usarlo después con datos crudos
        ruta_modelo = os.path.join(models_path, f"{nombre}.pkl")
        joblib.dump(pipeline, ruta_modelo)
        print(f" {nombre} guardado en: {ruta_modelo}")

if __name__ == "__main__":
    # Ejecuta el entrenamiento principal
    train_and_save_models()
    print("--- Pipeline de entrenamiento finalizado con éxito ---")