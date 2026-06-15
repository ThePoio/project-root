"""Rol 4 - QA & Production Engineer: inferencia para nuevos clientes.

Carga un modelo entrenado (pipeline completo de sklearn guardado por
``src/trainer_model.py``) y predice si un cliente de telecomunicaciones
abandonara el servicio (churn).

Los pipelines guardados aceptan un DataFrame con las columnas Telco *crudas*
(texto sin codificar): el preprocesamiento ocurre dentro del propio pipeline.

Uso:
    python -m src.predict
"""

import os

import joblib
import pandas as pd

# Carpeta donde el ML Engineer guarda los modelos entrenados (.pkl).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Orden de preferencia al autodetectar el modelo a usar.
PREFERRED_MODELS = ("random_forest", "logistic_regression", "svm")


def find_default_model(models_dir=MODELS_DIR):
    """Devuelve la ruta del primer modelo disponible segun PREFERRED_MODELS.

    Si no hay carpeta de modelos o ningun .pkl, retorna ``None`` para que el
    llamador muestre un mensaje claro.
    """
    if not os.path.isdir(models_dir):
        return None

    # Primero respetamos el orden de preferencia.
    for name in PREFERRED_MODELS:
        candidate = os.path.join(models_dir, f"{name}.pkl")
        if os.path.exists(candidate):
            return candidate

    # Si no, tomamos cualquier otro .pkl que exista.
    for fname in sorted(os.listdir(models_dir)):
        if fname.endswith(".pkl"):
            return os.path.join(models_dir, fname)

    return None


def load_model(model_path=None):
    """Carga un pipeline entrenado desde disco con manejo de errores claro.

    Si ``model_path`` es ``None`` se autodetecta el modelo. Lanza
    ``FileNotFoundError`` con un mensaje accionable cuando no hay modelo, en
    lugar de un traceback criptico.
    """
    if model_path is None:
        model_path = find_default_model()

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(
            "No se encontro ningun modelo entrenado en 'models/'.\n"
            "Ejecuta primero el pipeline de entrenamiento con:\n"
            "    python -m src.main"
        )

    return joblib.load(model_path)


def build_sample_client():
    """Construye un cliente de ejemplo con las 19 columnas Telco crudas.

    Sirve como demostracion de inferencia y como dato de prueba. Las columnas
    coinciden con las que entran al pipeline (sin ``customerID`` ni ``Churn``).
    """
    cliente = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5,
        "TotalCharges": 1074.0,
    }
    return pd.DataFrame([cliente])


def predict(client_df, model_path=None):
    """Predice churn para uno o varios clientes.

    Args:
        client_df: DataFrame con las columnas Telco crudas.
        model_path: ruta a un .pkl concreto; si es ``None`` se autodetecta.

    Returns:
        dict con la clase predicha (0/1), su etiqueta legible y, si el modelo
        lo soporta, la probabilidad de churn.
    """
    model = load_model(model_path)

    pred = int(model.predict(client_df)[0])
    label = "Churn" if pred == 1 else "No churn"

    resultado = {"prediccion": pred, "etiqueta": label, "probabilidad_churn": None}

    # SVC por defecto no expone predict_proba; lo manejamos opcionalmente.
    if hasattr(model, "predict_proba"):
        try:
            resultado["probabilidad_churn"] = float(model.predict_proba(client_df)[0][1])
        except Exception:
            # Algunos estimadores no tienen probabilidades calibradas; no es fatal.
            resultado["probabilidad_churn"] = None

    return resultado


def main():
    """Demostracion: predice churn para el cliente de ejemplo."""
    try:
        model_path = find_default_model()
        cliente = build_sample_client()
        resultado = predict(cliente, model_path)
    except FileNotFoundError as error:
        print(f"[ERROR] {error}")
        return

    nombre_modelo = os.path.basename(model_path) if model_path else "desconocido"
    print(f"Modelo usado: {nombre_modelo}")
    print(f"Prediccion: {resultado['etiqueta']} (clase={resultado['prediccion']})")
    if resultado["probabilidad_churn"] is not None:
        print(f"Probabilidad de churn: {resultado['probabilidad_churn']:.2%}")


if __name__ == "__main__":
    main()
