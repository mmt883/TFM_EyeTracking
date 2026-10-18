"""Predice proxies espaciales y temporales de atencion visual.

Las caracteristicas se extraen de los primeros segundos y los proxies objetivo
del intervalo siguiente. Son indicadores derivados de la mirada, no medidas
clinicas ni medidas directas de carga cognitiva o usabilidad.
"""

from datetime import datetime
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

from Funciones_Comunes import (
    LOGS_DIR,
    METADATA_PATH,
    PROJECT_DIR,
    obtener_columna_tiempo,
    normalizar_nombre_imagen,
    columna_numerica,
    participante_desde_nombre_archivo,
    leer_metadatos,
    resumir_fijaciones,
)

RESULTS_DIR = PROJECT_DIR / "Resultados" / "Resultados_Proxies_Visuales"
FEATURE_COLUMNS = import_module("02_Exploracion_AutoGluon").FEATURE_COLUMNS


# ==========================================
# CONFIGURACION
# ==========================================
# Edita estos atributos para configurar una ejecucion desde el propio archivo.
CATEGORY_FILTER = None  # None o categoria exacta: desktop, mobile, poster, web.
USER_FILTER = None  # None o id como "kh000"; filtrar puede dejar vacio Train o Test.
FEATURE_CUTOFF_SECONDS = 3.0  # Segundos iniciales para features; numero >= 0.
TARGET_END_SECONDS = 5.0  # Fin del tramo objetivo: debe ser > corte y <= fin del log.
ENTROPY_GRID_SIZE = 8  # Entero >= 2; rejilla cuadrada de este numero de celdas por eje.
MIN_TARGET_FIXATIONS = 2  # Entero >= 1; minimo de fijaciones futuras por ejemplo.
TIME_LIMIT_SECONDS = 120  # Limite por target en segundos; entero positivo.
RUN_TRAINING = True  # Booleano: False solo exporta tabla; True entrena AutoGluon.

TARGET_COLUMNS = {
    # Targets calculados en la ventana futura y entrenados en cada ejecucion.
    "future_mean_duration": "Duracion media de fijacion en la ventana futura",
    "future_spatial_dispersion": "Dispersion espacial Raiz cuadrada en la ventana futura",
    "future_attention_entropy": "Entropia espacial normalizada en la ventana futura",
    "future_grid_coverage": "Fraccion de celdas visitadas en la ventana futura",
}
TARGETS_TO_PREDICT = tuple(TARGET_COLUMNS)


# ==========================================
# FASE 1: PROXIES DE ATENCION ESPACIAL
# ==========================================
def calcular_proxies_espaciales(fixations, grid_size):
    """Calcula dispersion, entropia y cobertura sobre coordenadas normalizadas.

    La entropia se calcula sobre una rejilla uniforme. Se normaliza por la
    entropia maxima para que el resultado quede entre 0 y 1.
    """
    x = columna_numerica(frame=fixations, name="FPOGX")
    y = columna_numerica(frame=fixations, name="FPOGY")
    coordinates = pd.DataFrame({"x": x, "y": y}).dropna()
    if coordinates.empty:
        return {
            "future_spatial_dispersion": np.nan,
            "future_attention_entropy": np.nan,
            "future_grid_coverage": np.nan,
        }

    # El tracker expresa x/y como fraccion de pantalla. Limitamos excursiones fuera del rango para asjustarlas a las celdas del borde.
    positions = coordinates.to_numpy(dtype=float)
    positions = np.clip(positions, 0.0, 1.0)

    # Raiz cuadrada mide la distancia media al centroide de las fijaciones de la ventana;
    # una CIFRA MAYOR indica que la MIRADA se reparte por un AREA mas EXTENSA.
    center = positions.mean(axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    dispersion = float(np.sqrt(np.mean(np.square(distances))))

    # Histogram2d cuenta fijaciones por celda; la entropia mide cuan uniforme
    # esta distribuida la atencion por la rejilla.
    # Se cuenta cuantas fijaciones caen en cada celda de la rejilla 2D.
    counts, _, _ = np.histogram2d(
        x=positions[:, 0],
        y=positions[:, 1],
        bins=grid_size,
        range=((0.0, 1.0), (0.0, 1.0)),
    )
    # Entropia de Shannon: cero si todas las fijaciones caen en una sola celda,
    # y mayor cuando la atencion se reparte entre mas celdas.
    probabilities = counts[counts > 0] / counts.sum() #Se calcula la probabilidad de fijacion en cada celda dividiendo el numero de fijaciones en esa celda por el total de fijaciones.
    entropy = float(-np.sum(probabilities * np.log2(probabilities))) #Formula de entropia de Shannon: -sum(p * log2(p)) para todas las celdas con fijaciones. 
    maximum_entropy = float(np.log2(grid_size**2))
    normalized_entropy = entropy / maximum_entropy if maximum_entropy > 0 else 0.0

    # La cobertura es la proporcion de celdas de la rejilla que recibieron
    # al menos una fijacion dentro de esta ventana.
    # Cobertura ignora el numero de visitas repetidas: solo cuenta celdas ocupadas.
    coverage = float(np.count_nonzero(counts) / (grid_size**2))
    return {
        "future_spatial_dispersion": dispersion,
        "future_attention_entropy": normalized_entropy,
        "future_grid_coverage": coverage,
    }


# ==========================================
# FASE 2: CONSTRUCCION DE TABLA
# ==========================================
def extraer_datos_proxies(logs_dir, metadata, cutoff_seconds, end_seconds, user_filter=None):
    """Crea una fila por participante-interfaz con entradas y proxies futuros."""
    if end_seconds <= cutoff_seconds:
        raise ValueError("TARGET_END_SECONDS debe ser mayor que FEATURE_CUTOFF_SECONDS.")

    records = []
    user_filter = user_filter.strip().casefold() if user_filter else None

    # Recorremos los logs en orden estable y obtenemos el identificador de
    # participante del nombre del archivo, como en los otros programas.
    for log_path in sorted(Path(logs_dir).glob("*.csv")):
        participant = participante_desde_nombre_archivo(log_path)
        if participant is None or (user_filter and participant != user_filter):
            continue

        try:
            log = pd.read_csv(log_path, low_memory=False)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
            print(f"Aviso: no se pudo leer {log_path.name}: {error}")
            continue

        # Seleccionamos TIME(timestamp).
        time_column = obtener_columna_tiempo(log.columns)
        if "MEDIA_NAME" not in log.columns or time_column is None:
            print(f"Aviso: {log_path.name} no tiene MEDIA_NAME o TIME valido.")
            continue

        log = log.copy()
        log["_image_key"] = log["MEDIA_NAME"].map(normalizar_nombre_imagen)
        log["_elapsed_time"] = pd.to_numeric(log[time_column], errors="coerce")

        # FPOGV=1 es la bandera de coordenada de mirada valida.
        if "FPOGV" in log.columns:
            valid = pd.to_numeric(log["FPOGV"], errors="coerce").eq(1)
            log = log[valid].copy()
        log = log.dropna(subset=["_elapsed_time"])

        # Cada grupo corresponde a una interfaz de un participante.
        for image_key, group in log.groupby("_image_key", sort=False):
            if image_key not in metadata.index:
                continue
            image_metadata = metadata.loc[image_key]
            group = group.sort_values("_elapsed_time")

            # Entradas y objetivo ocupan intervalos disjuntos. Solo se usa el
            # tramo futuro hasta TARGET_END_SECONDS para mantener consistencia.
            early = group[group["_elapsed_time"] <= cutoff_seconds]
            future = group[
                (group["_elapsed_time"] > cutoff_seconds)
                & (group["_elapsed_time"] <= end_seconds)
            ]
            if early.empty or len(future) < MIN_TARGET_FIXATIONS:
                continue

            # resumir_fijaciones comparte las caracteristicas usadas por el
            # resto del proyecto. Sus targets originales no se usan aqui.
            example = resumir_fijaciones(
                early=early,
                future=future,
                category=image_metadata["Category"],
            )
            example.update(
                calcular_proxies_espaciales(
                    fixations=future,
                    grid_size=ENTROPY_GRID_SIZE,
                )
            )
            example.update(
                {
                    "image_name": image_metadata["Image Name"],
                    "participant": participant,
                    "official_split": image_metadata["_split"],
                }
            )
            records.append(example)

    return pd.DataFrame(records)


# ==========================================
# FASE 3: ENTRENAMIENTO Y EVALUACION
# ==========================================
def entrenar_y_evaluar(data, target, time_limit, run_dir):
    """Entrena un proxy con Train oficial y calcula metricas en Test oficial."""
    if target not in TARGET_COLUMNS:
        raise ValueError(f"TARGET_TO_PREDICT debe ser uno de: {list(TARGET_COLUMNS)}")

    # official_split se usa solo para preparar los conjuntos; no forma parte
    # de FEATURE_COLUMNS y por tanto nunca se entrega como predictor.
    model_data = data[FEATURE_COLUMNS + [target, "official_split"]].copy()
    model_data[target] = pd.to_numeric(model_data[target], errors="coerce")
    model_data = model_data.dropna(subset=[target])
    train_data = model_data[model_data["official_split"] == "train"]
    test_data = model_data[model_data["official_split"] == "test"]
    if train_data.empty or test_data.empty:
        raise ValueError(
            f"Se requieren ejemplos Train y Test; encontrados {len(train_data)} y {len(test_data)}."
        )
    if train_data[target].nunique() < 2:
        raise ValueError(
            f"El target '{target}' es constante en Train para los filtros seleccionados."
        )

    from autogluon.tabular import TabularPredictor

    # Solo entran al modelo las caracteristicas tempranas y la etiqueta elegida.
    predictor = TabularPredictor(
        label=target,
        problem_type="regression",
        path=str(run_dir / "modelo"),
        verbosity=0,
    ).fit(
        train_data[FEATURE_COLUMNS + [target]],
        time_limit=time_limit,
        presets="medium_quality",
    )

    # Test se reserva para la evaluacion. Los ids se agregan luego para auditar
    # predicciones, pero nunca se incluyen en FEATURE_COLUMNS.
    test_features = test_data[FEATURE_COLUMNS]
    # evaluate compara predicciones con etiquetas reales; las metricas de
    # regresion incluyen errores y R2 para resumir ajuste y capacidad explicativa.
    metrics = predictor.evaluate(
        test_data[FEATURE_COLUMNS + [target]],
        display=False,
    )
    predictions = predictor.predict(test_features)
    prediction_table = data.loc[test_data.index, ["image_name", "participant", target]].copy()
    prediction_table["prediction"] = predictions.to_numpy()
    prediction_table.to_csv(
        run_dir / "predicciones_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([metrics]).to_csv(
        run_dir / "metricas_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Metricas Test para {target}: {metrics}")
    return metrics


# ==========================================
# FASE 4: EJECUCION
# ==========================================
def ejecutar():
    """Aplica los filtros configurados, exporta la tabla y entrena el modelo."""
    if FEATURE_CUTOFF_SECONDS < 0 or TARGET_END_SECONDS <= FEATURE_CUTOFF_SECONDS:
        raise ValueError("Revisa FEATURE_CUTOFF_SECONDS y TARGET_END_SECONDS.")
    if ENTROPY_GRID_SIZE < 2:
        raise ValueError("ENTROPY_GRID_SIZE debe ser al menos 2.")
    # image_types.csv conserva el mismo separador, columnas y filtro por categoria.
    metadata = leer_metadatos(
        path=METADATA_PATH,
        category_filter=CATEGORY_FILTER,
    )
    if metadata.empty:
        raise ValueError("No hay imagenes en image_types.csv para CATEGORY_FILTER.")

    data = extraer_datos_proxies(
        logs_dir=LOGS_DIR,
        metadata=metadata,
        cutoff_seconds=FEATURE_CUTOFF_SECONDS,
        end_seconds=TARGET_END_SECONDS,
        user_filter=USER_FILTER,
    )
    if data.empty:
        raise ValueError("No se generaron ejemplos validos con los filtros y ventanas actuales.")

    # La marca temporal evita que una ejecucion sustituya resultados anteriores.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    category_name = (CATEGORY_FILTER or "todas").casefold()
    run_dir = RESULTS_DIR / f"{timestamp}_{category_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        run_dir / "caracteristicas_y_proxies.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Ejemplos participante-interfaz: {len(data)}")
    print(f"Splits oficiales: {data['official_split'].value_counts().to_dict()}")
    print(f"Resultados: {run_dir}")

    if RUN_TRAINING:
        results_by_target = []
        for target in TARGETS_TO_PREDICT:
            print(f"Proxy: {TARGET_COLUMNS[target]}")
            target_dir = run_dir / target
            target_dir.mkdir(parents=True, exist_ok=True)
            metrics = entrenar_y_evaluar(
                data=data,
                target=target,
                time_limit=TIME_LIMIT_SECONDS,
                run_dir=target_dir,
            )
            results_by_target.append({"target": target, **metrics})

        summary_path = run_dir / "resumen_metricas_test.csv"
        pd.DataFrame(results_by_target).to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Resumen por target: {summary_path}")


# Al importar este modulo solo se definen funciones; la ejecucion comienza aqui.
if __name__ == "__main__":
    ejecutar()