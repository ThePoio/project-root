import os
import yaml
# Importaciones del Data Engineer
from src.data_loader import cargar_y_limpiar_datos
# Importaciones del ML Engineer
from src.trainer_model import train_and_save_models

def load_params(yaml_path="config/params.yaml"):
    """Carga de forma segura el archivo de configuración."""
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)

def main():
    # ---------------------------------------------------------
    # 0. Cargar Parámetros
    # ---------------------------------------------------------
    print("🤖 [MLOps] Cargando configuración externa...")
    params = load_params()
    
    # ---------------------------------------------------------
    # 1. Pipeline del Data Engineer
    # ---------------------------------------------------------
    print("📊 [Data] Ejecutando función maestra de datos...")
    # La función maestra carga, limpia, divide y guarda los datos
    cargar_y_limpiar_datos(params["data_loader"])
    
    # ---------------------------------------------------------
    # 2. Pipeline del ML Engineer (Iteración de Modelos)
    # ---------------------------------------------------------
    print("🧠 [ML] Iniciando el entrenamiento de modelos...")
    train_and_save_models()

    print("✅ [MLOps] ¡Pipeline completo ejecutado con éxito sin errores!")

if __name__ == "__main__":
    main()