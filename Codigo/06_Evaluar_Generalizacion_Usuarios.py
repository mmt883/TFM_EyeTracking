"""Evalua modelos en participantes que no aparecen durante el entrenamiento.

Se limita el analisis a las imagenes marcadas Train en image_types.csv y se
separan participantes completos. Mide generalizacion a usuarios no vistos sobre
estimulos conocidos, como experimento complementario al Test oficial por imagen.
"""

from datetime import datetime
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from autogluon.tabular import TabularPredictor

from Funciones_Comunes import (
    LOGS_DIR,
    METADATA_PATH,
    PROJECT_DIR,
    leer_metadatos,
)


modulo02 = import_module("02_Exploracion_AutoGluon")
FEATURE_COLUMNS = modulo02.FEATURE_COLUMNS
extraer_ejemplos = modulo02.extraer_ejemplos

RESULTS_DIR = PROJECT_DIR / "Resultados" / "Resultados_Generalizacion_Usuarios"


# ==========================================
# CONFIGURACION
# ==========================================
# Cambia estos atributos para filtrar/configurar sin argumentos de terminal.
CATEGORY_FILTER = None  # None o categoria exacta: desktop, mobile, poster, web.
USER_FILTER = None  # Debe ser None para generalizar; un id como "kh000" impide separar grupos.
CUTOFF_SECONDS = 3.0  # Segundos iniciales usados como features; > 0 y < fin del registro.
TEST_SIZE = 0.20  # Proporcion de participantes Test; numero estrictamente entre 0 y 1.
RANDOM_SEED = 42  # Entero; cambia la asignacion reproducible de usuarios.
TIME_LIMIT_SECONDS = 120  # Segundos de entrenamiento por target; entero positivo.
USE_OFFICIAL_TRAIN_IMAGES_ONLY = False  # Booleano: restringe estimulos a filas Train de metadata.

TARGET_COLUMNS = {
    # Targets futuros evaluados en cada ejecucion.
    "future_fixation_count",
    "future_mean_duration",
    "future_scanpath_length",
}
TARGETS_TO_PREDICT = tuple(sorted(TARGET_COLUMNS))


# ==========================================
# FASE 1: DIVISION POR PARTICIPANTE
# ==========================================
def dividir_por_participante(data, test_size, random_seed):
    """Separa usuarios completos y comprueba que no hay usuarios compartidos."""
    unique_participants = data["participant"].dropna().unique()
    if len(unique_participants) < 2:
        raise ValueError(
            "Se necesitan al menos dos participantes; elimina USER_FILTER o amplia el dataset."
        )

    # GroupShuffleSplit asigna todas las filas de un participante a un solo split.
    # Asi se evita que el modelo vea en entrenamiento otra interfaz del mismo usuario.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_seed,
    )
    train_indices, test_indices = next(
        splitter.split(data, groups=data["participant"])
    )
    #iloc sirve para seleccionar filas por indices; copy() evita advertencias de pandas.
    train_data = data.iloc[train_indices].copy()
    test_data = data.iloc[test_indices].copy()

    # Esta asercion hace explicita la condicion principal del experimento.
    train_users = set(train_data["participant"])
    test_users = set(test_data["participant"])
    if train_users.intersection(test_users):
        raise AssertionError("Se detectaron participantes compartidos entre Train y Test.")
    return train_data, test_data


# ==========================================
# FASE 2: ENTRENAMIENTO Y COMPARACION
# ==========================================
def entrenar_y_evaluar(train_data, test_data, target, time_limit, run_dir):
    """Entrena en unas personas y evalua en otras con las mismas interfaces."""
    # Reducimos los DataFrames a entradas y etiqueta: ni usuario, ni imagen, ni
    # split pueden actuar como pistas directas para el predictor.
    model_train = train_data[FEATURE_COLUMNS + [target]].copy()
    model_test = test_data[FEATURE_COLUMNS + [target]].copy()
    model_train[target] = pd.to_numeric(model_train[target], errors="coerce")
    model_test[target] = pd.to_numeric(model_test[target], errors="coerce")
    model_train = model_train.dropna(subset=[target])
    model_test = model_test.dropna(subset=[target])

    if model_train.empty or model_test.empty:
        raise ValueError("La division por usuario no dejo filas utilizables en ambos grupos.")
    if model_train[target].nunique() < 2:
        raise ValueError(f"El target '{target}' es constante en los participantes Train.")

    # El predictor ve solo caracteristicas y target. No ve participant ni ids.
    predictor = TabularPredictor(
        label=target,
        problem_type="regression",
        path=str(run_dir / "modelo"),
        verbosity=0,
    ).fit(
        model_train,
        time_limit=time_limit,
        presets="medium_quality",
    )

    # Comparamos tambien con un baseline que predice la media de personas Train.
    metrics = predictor.evaluate(model_test, display=False)
    predictions = predictor.predict(model_test[FEATURE_COLUMNS])
    # El baseline siempre predice la media observada en usuarios Train.
    baseline_value = float(model_train[target].mean())
    baseline_rmse = float(
        np.sqrt(np.mean(np.square(model_test[target].to_numpy() - baseline_value)))
    )
    # AutoGluon invierte el signo de metricas de error para maximizarlas.
    model_rmse = float(-metrics["root_mean_squared_error"])
    results = {
        "target": target,
        **metrics,
        "model_rmse_positive": model_rmse,
        "baseline_mean_rmse": baseline_rmse,
        "rmse_improvement_percent": (
            100.0 * (baseline_rmse - model_rmse) / baseline_rmse
            if baseline_rmse > 0
            else np.nan
        ),
        "train_participants": train_data["participant"].nunique(),
        "test_participants": test_data["participant"].nunique(),
    }

    # Los identificadores se agregan despues de predecir para poder auditar errores.
    predictions_table = test_data.loc[model_test.index, [
        "participant",
        "image_name",
        target,
    ]].copy()
    predictions_table["prediction"] = predictions.to_numpy()
    predictions_table.to_csv(
        run_dir / "predicciones_usuarios_no_vistos.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([results]).to_csv(
        run_dir / "metricas_usuarios_no_vistos.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return results


# ==========================================
# FASE 3: EJECUCION
# ==========================================
def ejecutar():
    """Prepara la tabla, separa usuarios y guarda la evaluacion complementaria."""
    if not 0 < TEST_SIZE < 1:
        raise ValueError("TEST_SIZE debe estar entre 0 y 1.")
    # La metadata usa las mismas columnas y el mismo separador de los demas scripts.
    metadata = leer_metadatos(
        path=METADATA_PATH,
        category_filter=CATEGORY_FILTER,
    )
    if USE_OFFICIAL_TRAIN_IMAGES_ONLY:
        metadata = metadata[metadata["_split"] == "train"].copy()
    if metadata.empty:
        raise ValueError("No hay imagenes Train para los filtros seleccionados.")

    # USER_FILTER debe permanecer None para que se puedan crear dos grupos.
    data = extraer_ejemplos(
        logs_dir=LOGS_DIR,
        metadata=metadata,
        cutoff_seconds=CUTOFF_SECONDS,
        user_filter=USER_FILTER,
    )
    if data.empty:
        raise ValueError("No se generaron ejemplos con los filtros actuales.")

    train_data, test_data = dividir_por_participante(
        data=data,
        test_size=TEST_SIZE,
        random_seed=RANDOM_SEED,
    )

    # Cada ejecucion se guarda en una carpeta nueva para conservar resultados.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    category_name = (CATEGORY_FILTER or "todas").casefold()
    run_dir = RESULTS_DIR / f"{timestamp}_{category_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Exportamos la asignacion de usuarios para hacer reproducible la evaluacion.
    # Se exporta la lista de grupos para poder repetir y auditar el holdout.
    participant_split = pd.concat(
        [
            pd.DataFrame({"participant": train_data["participant"].unique(), "user_split": "train"}),
            pd.DataFrame({"participant": test_data["participant"].unique(), "user_split": "test"}),
        ],
        ignore_index=True,
    )
    participant_split.to_csv(
        run_dir / "participantes_por_split.csv",
        index=False,
        encoding="utf-8-sig",
    )
    data.to_csv(run_dir / "dataset_completo.csv", index=False, encoding="utf-8-sig")

    print(f"Filas Train/Test: {len(train_data)} / {len(test_data)}")
    print(f"Participantes Train: {train_data['participant'].nunique()}")
    print(f"Participantes Test: {test_data['participant'].nunique()}")
    print(f"Resultados: {run_dir}")

    results_by_target = []
    for target in TARGETS_TO_PREDICT:
        print(f"Target: {target}")
        target_dir = run_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)
        results = entrenar_y_evaluar(
            train_data=train_data,
            test_data=test_data,
            target=target,
            time_limit=TIME_LIMIT_SECONDS,
            run_dir=target_dir,
        )
        results_by_target.append(results)
        print(f"Metricas para participantes no vistos: {results}")

    summary_path = run_dir / "resumen_metricas_usuarios_no_vistos.csv"
    pd.DataFrame(results_by_target).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Resumen por target: {summary_path}")


# Al importar este modulo solo se definen funciones; la ejecucion comienza aqui.
if __name__ == "__main__":
    ejecutar()