"""Rol 4 - QA & Production Engineer: tests basicos del pipeline.

Estos tests son robustos ante artefactos faltantes:
- El CSV crudo NO se sube a git (cada alumno lo descarga), asi que el test del
  data_loader se salta si el dataset no esta presente.
- Los modelos .pkl se generan al correr ``python -m src.main``; los tests que
  dependen de ellos se saltan si aun no existen.

Asi la suite pasa tanto en una maquina recien clonada como en una con los
artefactos ya generados.
"""

import os

import pandas as pd
import pytest

from src import predict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(
    BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def test_build_sample_client_estructura():
    """El cliente de ejemplo tiene exactamente 1 fila y columnas crudas Telco."""
    cliente = predict.build_sample_client()

    assert isinstance(cliente, pd.DataFrame)
    assert len(cliente) == 1
    # No debe incluir el target ni el id (esas columnas se eliminan antes).
    assert "Churn" not in cliente.columns
    assert "customerID" not in cliente.columns
    # Algunas columnas clave que el pipeline espera.
    for col in ("gender", "tenure", "Contract", "MonthlyCharges", "TotalCharges"):
        assert col in cliente.columns


def test_load_model_inexistente_da_error_claro():
    """Cargar un modelo inexistente lanza FileNotFoundError controlado."""
    ruta_falsa = os.path.join(BASE_DIR, "models", "no_existe_xyz.pkl")
    with pytest.raises(FileNotFoundError):
        predict.load_model(ruta_falsa)


def test_load_data_no_vacio():
    """``cargar_y_limpiar_datos`` no retorna DataFrames vacios y es consistente."""
    if not os.path.exists(RAW_CSV):
        pytest.skip(
            "Dataset crudo no presente (descargalo en data/raw/). "
            "Test omitido."
        )
    try:
        from src.data_loader import cargar_y_limpiar_datos
    except ImportError:
        pytest.skip(
            "cargar_y_limpiar_datos aun no integrado en esta rama "
            "(data_loader es un stub)."
        )

    config = {"test_size": 0.2, "random_state": 42}
    X_train, X_test, y_train, y_test = cargar_y_limpiar_datos(config)

    assert not X_train.empty, "X_train no debe estar vacio"
    assert not X_test.empty, "X_test no debe estar vacio"
    # X e y deben tener el mismo numero de filas en cada split.
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_predict_smoke():
    """Smoke test: la prediccion de ejemplo devuelve una clase 0 o 1.

    Se salta si no hay ningun modelo entrenado en ``models/``.
    """
    model_path = predict.find_default_model()
    if model_path is None:
        pytest.skip(
            "No hay modelos entrenados en models/. "
            "Corre 'python -m src.main' primero."
        )

    cliente = predict.build_sample_client()
    resultado = predict.predict(cliente, model_path)

    assert resultado["prediccion"] in (0, 1)
    assert resultado["etiqueta"] in ("Churn", "No churn")
