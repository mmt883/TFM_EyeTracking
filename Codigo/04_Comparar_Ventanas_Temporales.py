"""Compara la informacion de los primeros 1, 3 y 5 segundos de mirada.

Las tres variantes predicen el mismo target: fijaciones observadas despues del
segundo 5. De este modo, cambian las caracteristicas disponibles, no la
definicion de la etiqueta ni los ejemplos de evaluacion.
"""

from datetime import datetime
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor

from Funciones_Comunes import (
    LOGS_DIR,
    METADATA_PATH,
    PROJECT_DIR,
    leer_metadatos,
    obtener_columna_tiempo,
    normalizar_nombre_imagen,
    participante_desde_nombre_archivo,
    resumir_fijaciones,
)

RESULTS_DIR = PROJECT_DIR / "Resultados" / "Resultados_Comparacion_Temporal"

FEATURE_COLUMNS = import_module("02_Exploracion_AutoGluon").FEATURE_COLUMNS


# ==========================================
# CONFIGURACION
# ==========================================
# Cambia estos atributos para elegir filtros, target y tipo de ejecucion.
# Los nombres coinciden con los usados en los otros scripts del directorio.
CATEGORY_FILTER = None  # None o: desktop, mobile, poster, web.
USER_FILTER = None  # None o un id como "kh000"; puede dejar sin filas Test.
TIME_LIMIT_SECONDS = 120  # Segundos por modelo y target; entero positivo.
RUN_TRAINING = True  # Booleano: False solo exporta datos; True entrena las tres ventanas.

# Cada prefijo acumula la mirada desde el comienzo del estimulo hasta su corte.
# Los tres usan el mismo target despues del segundo 5 para que solo cambien las
# entradas disponibles y la comparacion no mezcle distintas tareas predictivas.
WINDOW_SECONDS = (1, 3, 5)  # Cortes acumulativos en segundos; cambiar requiere datos tras cada corte.
TARGET_START_SECONDS = 5.0  # Inicio comun del target; debe ser >= al mayor WINDOW_SECONDS.
# Los nombres permiten validar el target y escribir una descripcion legible.
TARGET_LABELS = {
    # Todos se miden despues de TARGET_START_SECONDS.
    "future_fixation_count": "Numero de fijaciones despues del segundo 5",
    "future_mean_duration": "Duracion media de fijaciones despues del segundo 5",
    "future_scanpath_length": "Longitud del scanpath despues del segundo 5",
}
TARGETS_TO_COMPARE = tuple(TARGET_LABELS)


# ==========================================
# FASE 1: CREACION DEL CONJUNTO COMPARABLE
# ==========================================
def extraer_datos_comparacion(logs_dir, metadata, user_filter=None):
    """Extrae las tres ventanas manteniendo iguales los ejemplos y el target.

    Args:
        logs_dir: carpeta de CSV de fijaciones.
        metadata: tabla de image_types.csv, leida con leer_metadatos().
        user_filter: participante opcional, por ejemplo "kh000".

    Returns:
        DataFrame con una fila por participante, interfaz y ventana temporal.
    """
    records = []
    user_filter = user_filter.strip().casefold() if user_filter else None

    # Se extrae el participante del nombre del archivo.
    for log_path in sorted(Path(logs_dir).glob("*.csv")):
        participant = participante_desde_nombre_archivo(log_path)
        if participant is None or (user_filter and participant != user_filter):
            continue

        try:
            log = pd.read_csv(log_path, low_memory=False)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
            print(f"Aviso: no se pudo leer {log_path.name}: {error}")
            continue

        # TIME contiene segundos transcurridos. 
        time_column = obtener_columna_tiempo(log.columns)
        if "MEDIA_NAME" not in log.columns or time_column is None:
            print(f"Aviso: {log_path.name} no tiene MEDIA_NAME o TIME valido.")
            continue

        log = log.copy()
        log["_image_key"] = log["MEDIA_NAME"].map(normalizar_nombre_imagen)
        log["_elapsed_time"] = pd.to_numeric(log[time_column], errors="coerce")

        # Conservamos solo puntos de mirada validos, igual que en el pipeline
        # de exploracion. Los tiempos ausentes no se pueden asignar a una ventana.
        if "FPOGV" in log.columns:
            valid = pd.to_numeric(log["FPOGV"], errors="coerce").eq(1)
            log = log[valid].copy()
        log = log.dropna(subset=["_elapsed_time"])

        # Cada archivo contiene las interfaces de un bloque; se procesa cada
        # interfaz por separado para respetar la unidad participante-interfaz.
        for image_key, group in log.groupby("_image_key", sort=False):
            if image_key not in metadata.index:
                continue

            image_metadata = metadata.loc[image_key]
            group = group.sort_values("_elapsed_time")

            # El target se define una sola vez y es comun a las tres ventanas.
            # Si no hay ninguna fijacion despues de 5 s, esta observacion no
            # sirve para comparar un target de comportamiento posterior.
            target_window = group[group["_elapsed_time"] > TARGET_START_SECONDS]
            if target_window.empty:
                continue

            # Se exige al menos una fijacion en cada prefijo. Asi las tres
            # versiones se entrenan y evalua sobre exactamente los mismos casos.
            prefixes = {
                seconds: group[group["_elapsed_time"] <= seconds]
                for seconds in WINDOW_SECONDS
            }
            if any(prefix.empty for prefix in prefixes.values()):
                continue

            for seconds, prefix in prefixes.items():
                # resumir_fijaciones calcula predictores desde prefix y target
                # solo desde target_window. Las filas entre el corte y 5 s no
                # intervienen en ninguna de las dos partes para cortes menores.
                example = resumir_fijaciones(
                    early=prefix,
                    future=target_window,
                    category=image_metadata["Category"],
                )
                example.update(
                    {
                        "window_seconds": seconds,
                        "image_name": image_metadata["Image Name"],
                        "participant": participant,
                        "official_split": image_metadata["_split"],
                    }
                )
                records.append(example)

    return pd.DataFrame(records)


# ==========================================
# FASE 2: ENTRENAMIENTO Y COMPARACION
# ==========================================
def comparar_ventanas(data, target, time_limit, run_dir):
    """Ajusta un predictor por ventana y compara resultados en el mismo Test."""
    if target not in TARGET_LABELS:
        raise ValueError(f"Target no reconocido: {target}")

    # Cada diccionario de esta lista se convertira en una fila de resultados.
    comparison_rows = []
    # El target no es una caracteristica y los identificadores tampoco se
    # entregan al modelo. Solo se usa el split para separar train y test.
    columns_for_model = FEATURE_COLUMNS + [target, "official_split"]

    for seconds in WINDOW_SECONDS:
        # Aislamos una de las tres versiones y preparamos sus columnas de modelo.
        window_data = data[data["window_seconds"] == seconds].copy()
        model_data = window_data[columns_for_model].copy()
        model_data[target] = pd.to_numeric(model_data[target], errors="coerce")
        model_data = model_data.dropna(subset=[target])

        train_data = model_data[model_data["official_split"] == "train"]
        test_data = model_data[model_data["official_split"] == "test"]
        if train_data.empty or test_data.empty:
            raise ValueError(
                f"La ventana de {seconds} s no tiene ambos splits: "
                f"{len(train_data)} Train y {len(test_data)} Test."
            )
        if train_data[target].nunique() < 2:
            raise ValueError(
                f"El target '{target}' es constante en Train para la ventana de "
                f"{seconds} s y los filtros seleccionados. Prueba otro target o "
                "amplia la categoria/usuarios."
            )

        # Cada modelo se guarda en su propia carpeta. El split de validacion
        # interno de AutoGluon se obtiene solo de Train, nunca de Test.
        predictor = TabularPredictor(
            label=target,
            problem_type="regression",
            path=str(run_dir / f"modelo_{seconds}s"),
            verbosity=0,
        ).fit(
            train_data[FEATURE_COLUMNS + [target]],
            time_limit=time_limit,
            presets="medium_quality",
        )

        # Mantenemos tanto una referencia sencilla como las metricas del modelo.
        # La referencia siempre predice la media del conjunto Train.
        test_features = test_data[FEATURE_COLUMNS]
        test_labels = test_data[target]
        predictions = predictor.predict(test_features)
        metrics = predictor.evaluate(
            test_data[FEATURE_COLUMNS + [target]],
            display=False,
        )
        # Baseline constante: predice la media del target en Train para todos
        # los casos Test. No aprende de los targets reservados para evaluacion.
        baseline_value = float(train_data[target].mean())
        baseline_rmse = float(np.sqrt(np.mean((test_labels - baseline_value) ** 2)))

        # AutoGluon devuelve puntuaciones de error con signo negativo porque
        # su API maximiza metricas; se invierten para reportar RMSE/MAE usuales.
        model_rmse = float(-metrics["root_mean_squared_error"])
        model_mae = float(-metrics["mean_absolute_error"])
        improvement = (
            100.0 * (baseline_rmse - model_rmse) / baseline_rmse
            if baseline_rmse > 0
            else np.nan
        )

        # Predicciones e identificadores se exportan para inspeccion manual;
        # dichos identificadores nunca se usaron como entradas del predictor.
        predictions_table = data.loc[test_data.index, [
            "image_name",
            "participant",
            target,
        ]].copy()
        predictions_table["prediction"] = predictions.to_numpy()
        predictions_table.to_csv(
            run_dir / f"predicciones_test_{seconds}s.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Mejora positiva significa que el modelo reduce el RMSE del baseline.
        comparison_rows.append(
            {
                "window_seconds": seconds,
                "target": target,
                "train_rows": len(train_data),
                "test_rows": len(test_data),
                "model_rmse": model_rmse,
                "baseline_rmse": baseline_rmse,
                "rmse_improvement_percent": improvement,
                "model_mae": model_mae,
                "r2": metrics.get("r2", np.nan),
            }
        )

    # Ordenamos cronologicamente las ventanas para facilitar la lectura del CSV.
    comparison = pd.DataFrame(comparison_rows).sort_values("window_seconds")
    comparison.to_csv(
        run_dir / "comparacion_ventanas.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return comparison


# ==========================================
# FASE 3: EJECUCION
# ==========================================
def ejecutar():
    """Lee metadata y logs, guarda datos y, opcionalmente, compara modelos."""
    metadata = leer_metadatos(
        path=METADATA_PATH,
        category_filter=CATEGORY_FILTER,
    )
    if metadata.empty:
        raise ValueError("No hay imagenes en image_types.csv para ese filtro.")

    # La misma funcion y columnas de metadata se usan en Exploracion_AutoGluon.py.
    data = extraer_datos_comparacion(
        logs_dir=LOGS_DIR,
        metadata=metadata,
        user_filter=USER_FILTER,
    )
    if data.empty:
        raise ValueError(
            "No hay ejemplos con fijaciones en los tres prefijos y despues de 5 s."
        )

    # Cada ventana debe contener los mismos pares participante-interfaz,
    # etiquetas y split. equals() confirma que el contraste se hace con los
    # mismos casos, no con subconjuntos distintos por falta de fijaciones.
    for target in TARGETS_TO_COMPARE:
        example_keys = ["image_name", "participant", "official_split", target]
        reference = data[data["window_seconds"] == WINDOW_SECONDS[0]][example_keys]
        for seconds in WINDOW_SECONDS[1:]:
            candidate = data[data["window_seconds"] == seconds][example_keys]
            if not reference.reset_index(drop=True).equals(candidate.reset_index(drop=True)):
                raise AssertionError(
                    f"Las ventanas no comparten los mismos ejemplos para '{target}'."
                )

    # Salidas separadas por ejecucion y filtros, para no sobrescribir resultados.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    category_name = (CATEGORY_FILTER or "todas").casefold()
    run_dir = RESULTS_DIR / f"{timestamp}_{category_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        run_dir / "caracteristicas_y_target_por_ventana.csv",
        index=False,
        encoding="utf-8-sig",
    )

    first_window = data[data["window_seconds"] == WINDOW_SECONDS[0]]
    print(f"Casos comparables: {len(first_window)} participante-interfaz")
    print(f"Split oficial: {first_window['official_split'].value_counts().to_dict()}")
    print(f"Datos exportados en: {run_dir}")

    if RUN_TRAINING:
        comparisons_by_target = []
        for target in TARGETS_TO_COMPARE:
            print(f"Target comun: {TARGET_LABELS[target]}")
            target_dir = run_dir / target
            target_dir.mkdir(parents=True, exist_ok=True)
            comparison = comparar_ventanas(
                data=data,
                target=target,
                time_limit=TIME_LIMIT_SECONDS,
                run_dir=target_dir,
            )
            comparisons_by_target.append(comparison)
            print("Comparacion de ventanas (menor RMSE es mejor):")
            print(comparison.to_string(index=False))

        summary_path = run_dir / "resumen_comparacion_ventanas.csv"
        pd.concat(comparisons_by_target, ignore_index=True).to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Resumen por target: {summary_path}")


# Importar este modulo define funciones; ejecutarlo inicia el pipeline.
if __name__ == "__main__":
    ejecutar()