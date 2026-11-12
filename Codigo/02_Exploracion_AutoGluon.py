"""Pipeline de exploracion eye-tracking -> caracteristicas -> AutoGluon.

Las caracteristicas se calculan con fijaciones hasta el segundo de corte y
los targets con fijaciones posteriores. El split oficial de image_types.csv
se conserva por imagen para evitar que una interfaz aparezca en Train y Test.
Las regiones de texto e imagen son proxies heurísticos obtenidos de la captura,
no etiquetas semánticas ni resultados de OCR.
Al final se realiza, por target, una segunda evaluacion con las cinco features
mejor posicionadas por SHAP calculado sobre Train; se guardan por separado.

El archivo puede ejecutarse tal cual o configurarse modificando los atributos
de CONFIGURACION. Sus funciones tambien se reutilizan desde otros scripts.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor

# Detectores comunes para asociar fijaciones con las celdas AOI heuristicas.
from Funciones_AOI import clasificar_fijacion, detectar_aois_imagen
# Lectura, normalizacion y resumen de logs compartidos por los pipelines.
from Funciones_Comunes import (
    IMAGES_DIR,
    METADATA_COLUMNS,
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
# Metricas comunes, comparacion por target y explicaciones SHAP.
from Funciones_Modelado import (
    evaluar_modelos_autogluon,
    guardar_explicacion_shap,
    metricas_clasificacion_por_terciles,
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

# Columnas que recibira AutoGluon como entradas. Las medidas temporales,
# espaciales y el recuento describen solo fijaciones hasta CUTOFF_SECONDS.
# No incluimos image_name ni participant: son identificadores, no features.
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
    # Los targets describen fijaciones posteriores al corte. Se conserva el
    # recuento de fijaciones, que ya formaba parte de los objetivos anteriores.
    "future_fixation_count": "Número de fijaciones posteriores al corte",
    "future_mean_duration": "Duración media de fijaciones posteriores al corte",
    "future_scanpath_length": "Longitud del scanpath posterior al corte",
    "future_spatial_dispersion": "Dispersión espacial de las fijaciones posteriores",
    "future_ttff_aoi_seconds": "Tiempo hasta la primera fijación en una AOI posterior al corte",
    "future_time_on_text_proxy_seconds": "Tiempo proxy sobre texto en la ventana futura",
    "future_time_on_images_proxy_seconds": "Tiempo proxy sobre imágenes en la ventana futura",
}
TARGETS_TO_PREDICT = tuple(TARGET_COLUMNS)

# ==========================================
# EXTRACCIÓN DE EJEMPLOS
# ==========================================
def resumir_targets_visuales_futuros(
    future,
    cutoff_seconds,
    aois,
    image_width=None,
    image_height=None,
):
    """Calcula TTFF, tiempos por AOI y dispersion para la ventana futura.

    TTFF se mide desde el corte hasta la primera fijacion sobre una region
    text-like o visual-like. Las AOI son proxies visuales, no etiquetas OCR.

    Args:
        future: Filas de fijacion posteriores al corte temporal.
        cutoff_seconds: Instante de corte expresado en las unidades de TIME.
        aois: Regiones detectadas para la captura, o None si no esta disponible.
        image_width: Ancho original usado para proyectar coordenadas.
        image_height: Alto original usado para proyectar coordenadas.

    Returns:
        Diccionario de targets AOI para la ventana futura. Los tiempos de
        permanencia quedan como NaN si faltan AOI o duraciones validas; TTFF
        queda como NaN si no se encuentra ninguna fijacion clasificable.
    """
    duration = columna_numerica(frame=future, name="FPOGD")
    duration = duration.where(duration >= 0)
    if aois is None or not duration.notna().any():
        # Sin imagen/AOI o duraciones no se inventa un tiempo de permanencia.
        time_on_text = np.nan
        time_on_images = np.nan
    else:
        time_on_text = 0.0
        time_on_images = 0.0

    fixation_times = []
    if aois is not None:
        for _, fixation in future.sort_values("_elapsed_time").iterrows():
            x = pd.to_numeric(fixation.get("FPOGX"), errors="coerce")
            y = pd.to_numeric(fixation.get("FPOGY"), errors="coerce")
            if pd.isna(x) or pd.isna(y):
                continue

            aoi = clasificar_fijacion(
                x=x,
                y=y,
                image_width=image_width,
                image_height=image_height,
                aois=aois,
            )
            if aoi is None or aoi["aoi_type"] not in {"text_like", "visual_like"}:
                continue

            fixation_duration = pd.to_numeric(
                fixation.get("FPOGD"),
                errors="coerce",
            )
            if pd.notna(fixation_duration) and fixation_duration >= 0:
                if aoi["aoi_type"] == "text_like" and pd.notna(time_on_text):
                    time_on_text += float(fixation_duration)
                elif aoi["aoi_type"] == "visual_like" and pd.notna(time_on_images):
                    time_on_images += float(fixation_duration)

            # FPOGS registra el inicio de la fijacion; TIME es el respaldo si falta.
            fixation_start = pd.to_numeric(fixation.get("FPOGS"), errors="coerce")
            if pd.isna(fixation_start):
                fixation_start = pd.to_numeric(
                    fixation.get("_elapsed_time"),
                    errors="coerce",
                )
            if pd.notna(fixation_start):
                fixation_times.append(float(fixation_start))

    coordinates = pd.DataFrame(
        {
            "x": columna_numerica(frame=future, name="FPOGX"),
            "y": columna_numerica(frame=future, name="FPOGY"),
        }
    ).dropna()
    x_std = coordinates["x"].std()
    y_std = coordinates["y"].std()
    x_std = 0.0 if pd.isna(x_std) else float(x_std)
    y_std = 0.0 if pd.isna(y_std) else float(y_std)
    spatial_dispersion = (
        float(np.hypot(x_std, y_std)) if not coordinates.empty else np.nan
    )

    return {
        "future_spatial_dispersion": spatial_dispersion,
        "future_ttff_aoi_seconds": (
            max(0.0, min(fixation_times) - cutoff_seconds)
            if fixation_times
            else np.nan
        ),
        "future_time_on_text_proxy_seconds": time_on_text,
        "future_time_on_images_proxy_seconds": time_on_images,
    }


def extraer_ejemplos(
    logs_dir,
    metadata,
    cutoff_seconds,
    user_filter=None,
    images_dir=IMAGES_DIR,
    include_aoi_targets=True,
):
    """Crea una fila por participante e interfaz usando dos ventanas temporales.

    Args:
        logs_dir: Carpeta con los archivos de fijaciones.
        metadata: Metadatos de ``leer_metadatos``, indexados por ``_image_key``.
        cutoff_seconds: Limite temporal; el instante exacto pertenece a las
            entradas y solo valores posteriores forman los targets.
        user_filter: Identificador opcional de participante, como ``kh000``.
        images_dir: Carpeta de capturas para detectar las AOI heuristicas.
        include_aoi_targets: Si es False, omite la lectura de capturas y no
            agrega los targets temporales basados en AOI.

    Returns:
        DataFrame con una fila por imagen-participante, caracteristicas de la
        ventana temprana, targets de la ventana futura y columnas de auditoria
        ``image_name``, ``participant`` y ``official_split``. Se omiten logs
        ilegibles, no validos o sin ejemplos en ambos lados del corte.
    """
    examples = []
    image_aoi_cache = {}
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

            if include_aoi_targets:
                # Cada captura se analiza una vez y sus AOI se reutilizan para
                # todos los participantes que hayan observado la misma imagen.
                image_name = image_metadata["Image Name"]
                if image_name not in image_aoi_cache:
                    image_path = Path(images_dir) / image_name
                    try:
                        image_aoi_cache[image_name] = detectar_aois_imagen(image_path)
                    except OSError as error:
                        print(f"Aviso: no se pudieron detectar AOI en {image_path}: {error}")
                        image_aoi_cache[image_name] = None

                image_aoi_data = image_aoi_cache[image_name]
                if image_aoi_data is None:
                    aois = None
                    image_width = None
                    image_height = None
                else:
                    aois, image_width, image_height = image_aoi_data

            # category procede de los metadatos; el split no se calcula ni se
            # muestrea aqui: se conserva la etiqueta oficial por imagen.
            example = resumir_fijaciones(
                early=early,
                future=future,
                category=image_metadata["Category"],
            )
            if include_aoi_targets:
                example.update(
                    resumir_targets_visuales_futuros(
                        future=future,
                        cutoff_seconds=cutoff_seconds,
                        aois=aois,
                        image_width=image_width,
                        image_height=image_height,
                    )
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
# ENTRENAMIENTO Y EVALUACIÓN
# ==========================================
def entrenar_y_evaluar(
    data,
    target,
    time_limit,
    run_dir,
    feature_columns=None,
    generate_shap=True,
):
    """Entrena solo con Train oficial y evalua una vez sobre Test oficial.

    Las columnas de identificacion y la particion no se entregan al modelo.
    Si se especifica ``feature_columns``, se usan esas entradas en vez de la
    lista completa. ``generate_shap=False`` permite evaluar una ablacion sin
    volver a generar explicaciones.

    Args:
        data: Tabla de ejemplos con features, target y split oficial.
        target: Clave de target definida en ``TARGET_COLUMNS``.
        time_limit: Presupuesto AutoGluon en segundos para el ajuste.
        run_dir: Carpeta destino del modelo y sus resultados.
        feature_columns: Features a usar en esta ejecucion; None usa la lista
            global ``FEATURE_COLUMNS``.
        generate_shap: Si es True, exporta explicaciones del Test y seleccion
            de variables calculada sobre Train.

    Returns:
        Diccionario de metricas de la evaluacion oficial sobre Test.

    Raises:
        ValueError: Si el target o las features no estan configurados, si no
            hay ejemplos en ambos splits o si alguna particion esta vacia.
    """
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Target no reconocido: {target}")
    selected_features = (
        FEATURE_COLUMNS if feature_columns is None else list(feature_columns)
    )
    if not selected_features:
        raise ValueError("Se necesita al menos una característica para entrenar.")
    if len(selected_features) != len(set(selected_features)):
        raise ValueError("La lista de características contiene duplicados.")
    unknown_features = set(selected_features).difference(FEATURE_COLUMNS)
    if unknown_features:
        raise ValueError(
            f"Características no reconocidas: {sorted(unknown_features)}"
        )

    # Se construye una tabla solo con las entradas, el target y el split. La
    # conversion numerica detecta targets invalidos antes de llamar al modelo.
    model_data = data[selected_features + [target, "official_split"]].copy()
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
        train_data[selected_features + [target]],
        time_limit=time_limit,
        presets="medium_quality",
    )

    # evaluate necesita el target real; predict recibe solo caracteristicas.
    train_features = train_data[selected_features]
    test_features = test_data[selected_features]
    metrics = predictor.evaluate(
        test_data[selected_features + [target]],
        display=False,
    )
    predictions = predictor.predict(test_features)
    metrics.update(
        metricas_clasificacion_por_terciles(
            train_target=train_data[target],
            actual_target=test_data[target],
            predictions=predictions,
        )
    )
    # Guardamos identificadores para poder inspeccionar errores despues, pero
    # nunca se entregan al predictor como entradas.
    # Conservamos verdad y prediccion junto a IDs de auditoria, que no se usan
    # para entrenar ni inferir.
    prediction_table = data.loc[test_data.index, ["image_name", "participant", target]].copy()
    prediction_table["prediction"] = predictions.to_numpy()
    prediction_table.to_csv(run_dir / "predicciones_test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(run_dir / "metricas_test.csv", index=False, encoding="utf-8-sig")
    predictor.leaderboard(
        test_data[selected_features + [target]],
        silent=True,
    ).to_csv(
        run_dir / "leaderboard_test.csv", index=False, encoding="utf-8-sig"
    )
    comparison = evaluar_modelos_autogluon(
        predictor=predictor,
        train_target=train_data[target],
        test_features=test_features,
        test_target=test_data[target],
        context={"target": target},
    )
    comparison.to_csv(
        run_dir / "comparacion_modelos.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if generate_shap:
        guardar_explicacion_shap(
            predictor=predictor,
            train_features=train_features,
            test_features=test_features,
            output_dir=run_dir / "shap",
            context={"target": target},
        )

        # La seleccion se basa exclusivamente en SHAP de Train para no usar el
        # conjunto Test al decidir que atributos incluir en la ablacion.
        guardar_explicacion_shap(
            predictor=predictor,
            train_features=train_features,
            test_features=train_features,
            output_dir=run_dir / "shap_seleccion_train",
            context={"target": target},
        )
    print(f"Evaluación oficial Test: {metrics}")
    return metrics


# ==========================================
# EJECUCIÓN DEL PIPELINE
# ==========================================
def ejecutar():
    """Ejecuta la extraccion de ventanas y el entrenamiento configurado.

    Primero exporta el dataset completo para permitir inspeccion. Si
    ``FEATURES_ONLY`` es False, ajusta un modelo por target con Train oficial,
    evalua Test y guarda tambien una segunda comparacion basada en las cinco
    features SHAP principales seleccionadas sin consultar Test.
    """
    # Los filtros y valores de entrenamiento se editan arriba, en CONFIGURACION.
    if CUTOFF_SECONDS <= 0:
        raise ValueError("CUTOFF_SECONDS debe ser mayor que cero.")
    # FASE 1: PREPARACIÓN DE LOS DATOS
    # image_types.csv es la fuente de categoria y particion Train/Test. Se lee
    # antes que los logs para descartar desde el inicio imagenes no seleccionadas.
    metadata = leer_metadatos(
        path=METADATA_PATH,
        category_filter=CATEGORY_FILTER,
    )
    if metadata.empty:
        raise ValueError("No hay imágenes en image_types.csv para el filtro solicitado.")

    # FASE 2: EXTRACCIÓN DE CARACTERÍSTICAS
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
        # FASE 3: ENTRENAMIENTO DE MODELOS
        results_by_target = []
        model_comparisons = []
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
            model_comparisons.append(
                pd.read_csv(target_dir / "comparacion_modelos.csv")
            )

        # FASE 4: EVALUACIÓN DE LOS RESULTADOS OBTENIDOS
        summary_path = run_dir / "resumen_metricas_test.csv"
        pd.DataFrame(results_by_target).to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Resumen por target: {summary_path}")
        pd.concat(model_comparisons, ignore_index=True).to_csv(
            run_dir / "comparacion_modelos.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # FASE 5: SELECCIÓN SHAP Y NUEVA EVALUACIÓN CON CINCO FEATURES
        # El ranking se calcula con SHAP sobre Train; Test solo se utiliza para
        # medir el rendimiento final y nunca para elegir las cinco entradas.
        top5_results_by_target = []
        top5_model_comparisons = []
        for target in TARGETS_TO_PREDICT:
            target_dir = run_dir / target
            train_importances = pd.read_csv(
                target_dir
                / "shap_seleccion_train"
                / "shap_importancia_atributos.csv"
            )
            if len(train_importances) < 5:
                raise ValueError(
                    f"El target {target} no tiene cinco características "
                    "disponibles para la comparación SHAP."
                )

            top_features = train_importances["feature"].head(5).tolist()
            top5_dir = target_dir / "top_5_features_shap"
            top5_dir.mkdir(parents=True, exist_ok=True)
            train_importances.head(5).to_csv(
                top5_dir / "features_seleccionadas_shap.csv",
                index=False,
                encoding="utf-8-sig",
            )
            top5_metrics = entrenar_y_evaluar(
                data=data,
                target=target,
                time_limit=TIME_LIMIT_SECONDS,
                run_dir=top5_dir,
                feature_columns=top_features,
                generate_shap=False,
            )
            pd.DataFrame([{"target": target, **top5_metrics}]).to_csv(
                top5_dir / "resumen_metricas_test.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print(
                f"Evaluación Test con las 5 características SHAP de Train "
                f"({target}): {top5_metrics}"
            )
            top5_results_by_target.append(
                pd.read_csv(top5_dir / "resumen_metricas_test.csv")
            )
            top5_model_comparisons.append(
                pd.read_csv(top5_dir / "comparacion_modelos.csv")
            )

        top5_summary_path = run_dir / "resumen_metricas_test_top_5_shap.csv"
        pd.concat(top5_results_by_target, ignore_index=True).to_csv(
            top5_summary_path,
            index=False,
            encoding="utf-8-sig",
        )
        pd.concat(top5_model_comparisons, ignore_index=True).to_csv(
            run_dir / "comparacion_modelos_top_5_shap.csv",
            index=False,
            encoding="utf-8-sig",
        )
        # Los CSV y el modelo reducido quedan bajo su carpeta propia; las
        # metricas y comparaciones anteriores no se reemplazan.
        print(f"Resumen de los 5 atributos SHAP: {top5_summary_path}")


# Al importar este modulo solo se definen funciones; ejecutar() se ejecuta cuando
# se lanza el archivo directamente desde Python o desde la terminal.
if __name__ == "__main__":
    ejecutar()