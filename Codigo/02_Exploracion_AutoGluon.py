"""Pipeline de exploracion eye-tracking -> caracteristicas -> AutoGluon.

Las caracteristicas se calculan con fijaciones hasta el segundo de corte y
los targets con fijaciones posteriores. El split oficial de image_types.csv
se conserva por imagen para evitar que una interfaz aparezca en Train y Test.

El archivo puede ejecutarse tal cual o configurarse modificando los atributos
de CONFIGURACION. Sus funciones tambien se reutilizan desde otros scripts.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor

from Funciones_Comunes import (
    METADATA_COLUMNS,   # Columnas de image_types.csv que se usan en este pipeline.
    LOGS_DIR,      # Carpeta con los logs de fijaciones.
    METADATA_PATH,  # Ruta a image_types.csv.
    PROJECT_DIR,    # Carpeta raiz del proyecto; se crean subcarpetas de resultados.
    obtener_columna_tiempo, # Devuelve el nombre de la columna TIME o TIMETICK en un DataFrame.
    normalizar_nombre_imagen,   # Normaliza MEDIA_NAME para que coincida con image_types.csv.
    columna_numerica,   # Convierte una columna a numerica y devuelve un Series con NaN en errores.
    participante_desde_nombre_archivo,  # Extrae el id de participante desde el nombre de un archivo de log.
    leer_metadatos, # Lee image_types.csv y devuelve un DataFrame indexado por nombre de imagen.
    resumir_fijaciones, # Calcula caracteristicas y targets a partir de dos ventanas temporales.
)

RESULTS_DIR = PROJECT_DIR / "Resultados" / "Resultados_AutoGluon"

# ==========================================
# CONFIGURACION
# ==========================================
# Cambia estos valores para configurar la ejecucion desde este archivo.
CATEGORY_FILTER = None  # None o una categoria exacta de metadata: desktop, mobile, poster, web.
USER_FILTER = None  # None o id de participante, por ejemplo "kh000"; filtrar puede dejar un solo split.
CUTOFF_SECONDS = 3.0  # Numero en segundos, > 0 y < duracion del log.
TIME_LIMIT_SECONDS = 120  # Entero en segundos por target para AutoGluon.
FEATURES_ONLY = False  # Booleano: True exporta sin entrenar; False ejecuta AutoGluon.

# Columnas que recibira AutoGluon como entradas. No incluimos image_name ni
# participant: son identificadores, no caracteristicas de comportamiento.
FEATURE_COLUMNS = [
    "category",
    "early_fixation_count",
    "early_duration_mean",
    "early_duration_std",
    "early_duration_total",
    "early_x_mean",
    "early_y_mean",
    "early_x_std",
    "early_y_std",
    "early_spatial_dispersion",
    "early_step_distance_mean",
    "early_scanpath_length",
]

TARGET_COLUMNS = {
    # Targets futuros que se entrenan en una misma ejecucion.
    "future_fixation_count": "Número de fijaciones posteriores al corte",
    "future_mean_duration": "Duración media de fijaciones posteriores al corte",
    "future_scanpath_length": "Longitud del scanpath posterior al corte",
}
TARGETS_TO_PREDICT = tuple(TARGET_COLUMNS)

# ==========================================
# FASE 2: LECTURA Y PROCESAMIENTO DE LOGS
# ==========================================
def extraer_ejemplos(logs_dir, metadata, cutoff_seconds, user_filter=None):
    """Crea una fila por participante e interfaz usando dos ventanas temporales.

    Args:
        logs_dir: carpeta con los archivos de fijaciones.
        metadata: metadatos cargados por leer_metadatos().
        cutoff_seconds: limite; <= limite son entradas y > limite son targets.
        user_filter: identificador de participante opcional (por ejemplo kh000).

    Returns:
        DataFrame con caracteristicas, targets e identificadores de auditoria.
    """
    examples = []
    user_filter = user_filter.strip().casefold() if user_filter else None

    # sorted() hace reproducible el orden de procesamiento de los archivos.
    for log_path in sorted(Path(logs_dir).glob("*.csv")):
        participant = participante_desde_nombre_archivo(log_path)

        # Se pueden omitir participantes antes de abrir archivos para reducir
        # trabajo cuando se solicita un usuario concreto.
        if participant is None or (user_filter and participant != user_filter):
            continue

        try:
            # low_memory=False evita inferencias de tipo inconsistentes entre
            # bloques en estos CSV relativamente anchos.
            log = pd.read_csv(log_path, low_memory=False)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
            print(f"Aviso: no se pudo leer {log_path.name}: {error}")
            continue

        # TIME representa el tiempo transcurrido; TIMETICK es el reloj de alta
        # precision y no debe confundirse con segundos.
        time_column = obtener_columna_tiempo(log.columns)
        if "MEDIA_NAME" not in log.columns or time_column is None:
            print(f"Aviso: {log_path.name} no contiene MEDIA_NAME o una columna TIME válida.")
            continue

        # MEDIA_NAME permite asociar cada fijacion con image_types.csv.
        log = log.copy()
        log["_image_key"] = log["MEDIA_NAME"].map(normalizar_nombre_imagen)
        log["_elapsed_time"] = pd.to_numeric(log[time_column], errors="coerce")

        # FPOGV=1 identifica datos validos de punto de mirada. Si no existe la
        # columna, se mantienen las filas y el pipeline puede seguir trabajando.
        if "FPOGV" in log.columns:
            valid = pd.to_numeric(log["FPOGV"], errors="coerce").eq(1)
            log = log[valid].copy()
        log = log.dropna(subset=["_elapsed_time"]) # Eliminamos filas sin tiempo transcurrido; no se pueden usar para caracteristicas ni targets.

        # Un archivo contiene varios estimulos. groupby crea un grupo de filas
        # por imagen para poder dividir su secuencia de forma independiente.
        for image_key, group in log.groupby("_image_key", sort=False):
            if image_key not in metadata.index:
                continue
            image_metadata = metadata.loc[image_key] # Recuperamos categoria y split oficial de image_types.csv.

            # Ordenamos por tiempo antes de formar las ventanas. La fijacion
            # exactamente en el corte pertenece a las entradas, no al target.
            group = group.sort_values("_elapsed_time")
            early = group[group["_elapsed_time"] <= cutoff_seconds]
            future = group[group["_elapsed_time"] > cutoff_seconds]

            # No hay ejemplo supervisado si falta cualquiera de las dos partes.
            if early.empty or future.empty:
                continue

            # category procede de los metadatos; el split no se calcula ni se
            # muestrea aqui: se conserva la etiqueta oficial por imagen.
            example = resumir_fijaciones(
                early=early,
                future=future,
                category=image_metadata["Category"],
            )
            example.update(
                {
                    "image_name": image_metadata["Image Name"],
                    "participant": participant,
                    "official_split": image_metadata["_split"],
                }
            )
            examples.append(example)

    return pd.DataFrame(examples)


# ==========================================
# FASE 3: ENTRENAMIENTO Y EVALUACION
# ==========================================
def entrenar_y_evaluar(data, target, time_limit, run_dir):
    """Entrena solo con Train oficial y evalua una vez sobre Test oficial.

    Las columnas de identificacion y la particion se excluyen de FEATURE_COLUMNS;
    AutoGluon recibe unicamente entradas y la columna objetivo seleccionada.
    """
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Target no reconocido: {target}")

    # Se construye una tabla solo con las entradas, el target y el split. La
    # conversion numerica detecta targets invalidos antes de llamar al modelo.
    model_data = data[FEATURE_COLUMNS + [target, "official_split"]].copy()
    model_data[target] = pd.to_numeric(model_data[target], errors="coerce")
    model_data = model_data.dropna(subset=[target])
    # El split esta definido por imagen en image_types.csv. Todas las filas de
    # participantes de una imagen heredan la misma particion oficial.
    train_data = model_data[model_data["official_split"] == "train"]
    test_data = model_data[model_data["official_split"] == "test"]
    if train_data.empty or test_data.empty:
        raise ValueError(
            f"Se necesitan ejemplos Train y Test oficiales; encontrados: "
            f"{len(train_data)} Train y {len(test_data)} Test."
        )

    # fit solo ve train_data. AutoGluon crea su validacion interna a partir de
    # ese Train; el conjunto Test queda reservado para la evaluacion final.
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

    # evaluate necesita el target real; predict recibe solo caracteristicas.
    test_features = test_data[FEATURE_COLUMNS]
    metrics = predictor.evaluate(test_data[FEATURE_COLUMNS + [target]], display=False)
    predictions = predictor.predict(test_features)
    # Guardamos identificadores para poder inspeccionar errores despues, pero
    # nunca se entregan al predictor como entradas.
    # Conservamos verdad y prediccion junto a IDs de auditoria, que no se usan
    # para entrenar ni inferir.
    prediction_table = data.loc[test_data.index, ["image_name", "participant", target]].copy()
    prediction_table["prediction"] = predictions.to_numpy()
    prediction_table.to_csv(run_dir / "predicciones_test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(run_dir / "metricas_test.csv", index=False, encoding="utf-8-sig")
    predictor.leaderboard(test_data[FEATURE_COLUMNS + [target]], silent=True).to_csv(
        run_dir / "leaderboard_test.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Evaluación oficial Test: {metrics}")
    return metrics


# ==========================================
# FASE 4: EJECUCION
# ==========================================
def ejecutar():
    # Los filtros y valores de entrenamiento se editan arriba, en CONFIGURACION.
    if CUTOFF_SECONDS <= 0:
        raise ValueError("CUTOFF_SECONDS debe ser mayor que cero.")
    # image_types.csv es la fuente de categoria y particion Train/Test. Se lee
    # antes que los logs para descartar desde el inicio imagenes no seleccionadas.
    metadata = leer_metadatos(
        path=METADATA_PATH,
        category_filter=CATEGORY_FILTER,
    )
    if metadata.empty:
        raise ValueError("No hay imágenes en image_types.csv para el filtro solicitado.")

    # Un ejemplo requiere al menos una fijacion a cada lado del corte temporal.
    data = extraer_ejemplos(
        logs_dir=LOGS_DIR,
        metadata=metadata,
        cutoff_seconds=CUTOFF_SECONDS,
        user_filter=USER_FILTER,
    )
    if data.empty:
        raise ValueError("No se generaron ejemplos; revisa los filtros, los logs y el corte temporal.")

    # Una carpeta distinta por ejecucion evita sobreescribir resultados previos.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filter_name = (CATEGORY_FILTER or "todas").casefold()
    run_dir = RESULTS_DIR / f"{timestamp}_{filter_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    features_path = run_dir / "caracteristicas_y_targets.csv"
    # Se exportan juntas las entradas, targets y columnas de auditoria. El
    # entrenamiento posterior selecciona explicitamente FEATURE_COLUMNS.
    data.to_csv(features_path, index=False, encoding="utf-8-sig")

    print(f"Ejemplos participante-interfaz: {len(data)}")
    print(f"Por split oficial: {data['official_split'].value_counts().to_dict()}")
    print(f"Características y targets: {features_path}")

    # Este modo facilita inspeccionar el CSV antes de entrenar.
    if not FEATURES_ONLY:
        results_by_target = []
        for target in TARGETS_TO_PREDICT:
            print(f"Target: {TARGET_COLUMNS[target]}")
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


# Al importar este modulo solo se definen funciones; ejecutar() se ejecuta cuando
# se lanza el archivo directamente desde Python o desde la terminal.
if __name__ == "__main__":
    ejecutar()