"""Input 1: calcula medidas de mirada y targets usando la sesion completa.

Flujo general:
1. Lee los logs de cada participante y los metadatos de cada interfaz.
2. Resume las fijaciones validas de cada pareja participante-interfaz.
3. Calcula medidas temporales/espaciales y, si hay captura, medidas AOI.
4. Conserva el split oficial de cada imagen al exportar y entrenar modelos.

Las AOI ``text_like`` y ``visual_like`` son aproximaciones por pixeles y
rejillas; no proceden de OCR ni son etiquetas semanticas verificadas.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from Funciones_AOI import (
    detectar_aois_imagen,
)
from Funciones_Comunes import (
    IMAGES_DIR,
    LOGS_DIR,
    METADATA_PATH,
    PROJECT_DIR,
    leer_metadatos,
    normalizar_nombre_imagen,
    obtener_columna_tiempo,
    participante_desde_nombre_archivo,
)
from Funciones_Entradas import (
    extraer_datos_fijaciones_completas as extraer_datos_compartidos,
    resumir_sesion_completa,
)
from Funciones_Modelado import (
    PROXY_TARGET_COLUMNS,
    agregar_proxies_observados_y_predichos,
    calcular_indicadores_proxy,
    entrenar_predecir_targets,
)


RESULTS_DIR = PROJECT_DIR / "Resultados" / "Resultados_Input_Fijaciones_Completas"

# ==========================================
# CONFIGURACION
# ==========================================
# Estos valores controlan filtros de categoria, granularidad espacial y
# entrenamiento. Todas las sesiones de los participantes reconocibles se usan.
CATEGORY_FILTER = None  # None o categoria oficial: desktop, mobile, poster, web.
# Esta rejilla clasifica la captura en AOI heuristicas para cada sesion.
AOI_GRID_COLUMNS = 16  # Numero de columnas AOI a lo ancho de la imagen.
AOI_GRID_ROWS = 16  # Numero de filas AOI a lo alto de la imagen.
# Esta segunda rejilla se aplica a las coordenadas de mirada normalizadas.
ATTENTION_GRID_SIZE = 8  # Numero de celdas por eje para entropia/cobertura.
# AutoGluon recibe este presupuesto de tiempo para cada target, no para toda
# la ejecucion. RUN_TRAINING permite generar primero el CSV sin ajustar modelos.
TIME_LIMIT_SECONDS = 900  # Limite por target de regresion AutoGluon.
RUN_TRAINING = True  # False exporta los targets medidos sin entrenar modelos.
# El modo SHAP top 3 selecciona variables con Train y repite la evaluacion;
# sus resultados se guardan en una subcarpeta y no sustituyen al modelo base.
RUN_SHAP_TOP3 = True  # True repite cada modelo usando sus tres features SHAP principales.

# Entradas del modelo: resumenes del registro completo de fijaciones.
# Todas las columnas siguientes describen la sesion entera (no solo sus
# primeros segundos). El codigo 12 las usa como etiquetas, no como entradas.
FEATURE_COLUMNS = [
    "category",
    "feature_observation_duration_seconds",
    "feature_fixation_duration_mean", #Duracion media de las fijaciones
    "feature_fixation_duration_std", #Desviacion estandar de la duracion de las fijaciones
    "feature_fixation_duration_total", #Duracion total de las fijaciones
    "feature_fixation_count",
    "feature_x_mean", 
    "feature_y_mean",
    "feature_x_std",
    "feature_y_std",
    "feature_spatial_dispersion",
    "feature_inter_fixation_distance_mean",
]

# Para algunos targets se excluyen predictores que son la misma medida o una
# transformacion muy cercana del objetivo. Estas listas reducen la fuga
# directa de informacion, aunque no pretenden eliminar toda correlacion entre
# variables ni cambiar la definicion del target.
FEATURE_COLUMNS_BY_TARGET = {
    # La media, desviacion y suma de duracion comparten la medicion con esta
    # etiqueta y podrian permitir reconstruirla casi directamente.
    "target_fixation_duration_mean": [
        column
        for column in FEATURE_COLUMNS
        if column
        not in {
            "feature_fixation_duration_mean",
            "feature_fixation_duration_std",
            "feature_fixation_duration_total",
        }
    ],
    # Evita usar el centro y la dispersion global de la mirada como atajos
    # para el tiempo hasta la primera fijacion.
    "target_ttff_seconds": [
        column
        for column in FEATURE_COLUMNS
        if column not in {"feature_x_mean", "feature_y_mean", "feature_x_std", "feature_y_std"}
    ],
    # Los tiempos de permanencia dependen de la duracion y ubicacion de las
    # fijaciones; se excluyen tambien las medidas espaciales derivadas.
    "target_time_on_text_seconds": [
        column
        for column in FEATURE_COLUMNS
        if column
        not in {
            "feature_fixation_duration_mean",
            "feature_fixation_duration_total",
            "feature_fixation_count",
            "feature_x_mean",
            "feature_y_mean",
            "feature_x_std",
            "feature_y_std",
            "feature_spatial_dispersion",
            "feature_inter_fixation_distance_mean",
        }
    ],
    # Se aplica la misma exclusion que al tiempo en texto por tratarse de la
    # otra clase de permanencia sobre AOI.
    "target_time_on_images_seconds": [
        column
        for column in FEATURE_COLUMNS
        if column
        not in {
            "feature_fixation_duration_mean",
            "feature_fixation_duration_total",
            "feature_fixation_count",
            "feature_x_mean",
            "feature_y_mean",
            "feature_x_std",
            "feature_y_std",
            "feature_spatial_dispersion",
            "feature_inter_fixation_distance_mean",
        }
    ],
    # Se mantiene la duracion como posible predictor, pero se quitan las
    # medidas de posicion y distancia que describen el recorrido objetivo.
    "target_scanpath_length": [
        column
        for column in FEATURE_COLUMNS
        if column
        not in {
            "feature_fixation_count",
            "feature_x_mean",
            "feature_y_mean",
            "feature_x_std",
            "feature_y_std",
            "feature_spatial_dispersion",
            "feature_inter_fixation_distance_mean",
        }
    ],
    # El propio numero de fijaciones no puede servir simultaneamente de input
    # y etiqueta; las otras medidas quedan disponibles.
    "target_fixation_count": [
        column
        for column in FEATURE_COLUMNS
        if column != "feature_fixation_count"
    ],
    # Evita variables de posicion y recorrido que podrian aproximar la
    # distribucion espacial usada en la entropia.
    "target_attention_entropy": [
        column
        for column in FEATURE_COLUMNS
        if column
        not in {
            "feature_fixation_count",
            "feature_x_mean",
            "feature_y_mean",
            "feature_x_std",
            "feature_y_std",
            "feature_spatial_dispersion",
            "feature_inter_fixation_distance_mean",
        }
    ],
    # Elimina las coordenadas-resumen y la propia dispersion, pero conserva
    # otras caracteristicas temporales de la sesion.
    "target_spatial_dispersion": [
        column
        for column in FEATURE_COLUMNS
        if column
        not in {
            "feature_x_mean",
            "feature_y_mean",
            "feature_x_std",
            "feature_y_std",
            "feature_spatial_dispersion",
        }
    ],
    # Para cobertura, se excluyen medidas de posicion y distancia directamente
    # relacionadas con la ocupacion de celdas.
    "target_grid_coverage": [
        column
        for column in FEATURE_COLUMNS
        if column
        not in {
            "feature_fixation_count",
            "feature_x_mean",
            "feature_y_mean",
            "feature_x_std",
            "feature_y_std",
            "feature_spatial_dispersion",
            "feature_inter_fixation_distance_mean",
        }
    ],
    # Recuento de fijaciones y coordenadas-resumen podrian revelar cuantas
    # regiones AOI se visitan, por eso no se usan para este objetivo.
    "target_visited_elements": [
        column
        for column in FEATURE_COLUMNS
        if column
        not in {
            "feature_fixation_count",
            "feature_x_mean",
            "feature_y_mean",
            "feature_x_std",
            "feature_y_std",
            "feature_spatial_dispersion",
            "feature_inter_fixation_distance_mean",
        }
    ],
    # Como en los otros objetivos AOI, se retiran recuentos y resumenes
    # espaciales que describen de forma muy cercana la secuencia de visitas.
    "target_transitions": [
        column
        for column in FEATURE_COLUMNS
        if column
        not in {
            "feature_fixation_count",
            "feature_x_mean",
            "feature_y_mean",
            "feature_x_std",
            "feature_y_std",
            "feature_spatial_dispersion",
            "feature_inter_fixation_distance_mean",
        }
    ],
}


def extraer_datos_fijaciones_completas(
    logs_dir,
    metadata,
    images_dir=IMAGES_DIR,
):
    """Construye una fila por participante e interfaz a partir de los logs.

    Se emparejan los logs con ``metadata`` usando el nombre normalizado de la
    imagen. Los identificadores y el split se separan de las columnas predictoras al entrenar. 
    La deteccion AOI se almacena en cache por imagen porque sus resultados no dependen del
    participante; el cache solo vive durante esta llamada.

    Args:
        logs_dir: Carpeta con los CSV de fijaciones por participante.
        metadata: DataFrame de ``leer_metadatos``, indexado por ``_image_key``.
        images_dir: Carpeta de capturas usadas para detectar AOI.

    Returns:
        DataFrame con una fila por combinacion imagen-participante, las medidas
        calculadas y los campos ``image_name``, ``participant``, ``category``
        y ``official_split``. Los ficheros ilegibles o sin columnas requeridas
        se anuncian y se omiten; los errores no relacionados con lectura o
        deteccion de imagen no se silencian.
    """
    records = []
    aoi_cache = {}

    # Orden fijo para que la extraccion y los CSV sean reproducibles.
    for log_path in sorted(Path(logs_dir).glob("*.csv")):
        participant = participante_desde_nombre_archivo(log_path)
        if participant is None:
            continue
        try:
            log = pd.read_csv(log_path, low_memory=False)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
            print(f"Aviso: no se pudo leer {log_path.name}: {error}")
            continue

        time_column = obtener_columna_tiempo(log.columns)
        if "MEDIA_NAME" not in log.columns or time_column is None:
            print(f"Aviso: {log_path.name} no contiene MEDIA_NAME o TIME.")
            continue

        # La clave normalizada permite cruzar el nombre del log con el CSV de
        # metadatos. _elapsed_time proporciona un orden numerico comun para
        # ordenar fijaciones y calcular la duracion observada.
        log = log.copy()
        log["_image_key"] = log["MEDIA_NAME"].map(normalizar_nombre_imagen)
        log["_elapsed_time"] = pd.to_numeric(log[time_column], errors="coerce")
        log = log.dropna(subset=["_elapsed_time"])
        # Cuando existe FPOGV, solo el valor 1 identifica una fijacion valida.
        # En logs antiguos sin esa bandera se conserva el comportamiento
        # compatible de considerar todas las filas con tiempo como fijaciones.
        if "FPOGV" in log.columns:
            valid = pd.to_numeric(log["FPOGV"], errors="coerce").eq(1)
            valid_fixations = log[valid].copy()
        else:
            valid_fixations = log.copy()

        # recorded_rows sirve para medir el intervalo total de observacion;
        # valid_fixations se reserva para estadisticos de fijaciones. Separar
        # ambos evita que muestras no fijadas desaparezcan de la duracion.
        for image_key, recorded_rows in log.groupby("_image_key", sort=False):
            if image_key not in metadata.index:
                continue
            image_metadata = metadata.loc[image_key]
            image_name = image_metadata["Image Name"]

            # La deteccion AOI por captura se reutiliza entre participantes.
            # Si falta el fichero o no se puede leer, las metricas temporales
            # y espaciales siguen siendo calculables, pero las AOI quedan NaN.
            if image_name not in aoi_cache:
                image_path = Path(images_dir) / image_name
                try:
                    aoi_cache[image_name] = detectar_aois_imagen(
                        image_path=image_path,
                        grid_columns=AOI_GRID_COLUMNS,
                        grid_rows=AOI_GRID_ROWS,
                    )
                except OSError as error:
                    print(f"Aviso: no se pudieron detectar AOI en {image_path}: {error}")
                    aoi_cache[image_name] = None

            image_fixations = valid_fixations[
                valid_fixations["_image_key"] == image_key
            ].sort_values("_elapsed_time")
            # No se crea una fila vacia: sin fijaciones validas no hay una
            # sesion ocular que resumir, aunque el log contenga otras muestras.
            if image_fixations.empty:
                continue

            image_aoi_data = aoi_cache[image_name]
            if image_aoi_data is None:
                aois = image_width = image_height = None
            else:
                aois, image_width, image_height = image_aoi_data

            record = resumir_sesion_completa(
                fixations=image_fixations,
                recorded_rows=recorded_rows,
                time_column=time_column,
                aois=aois,
                image_width=image_width,
                image_height=image_height,
            )
            record.update(
                {
                    "image_name": image_name,
                    "participant": participant,
                    "category": image_metadata["Category"],
                    "official_split": image_metadata["_split"],
                }
            )
            records.append(record)

    return pd.DataFrame(records)


def ejecutar():
    """Ejecuta extraccion, exportacion, proxies y modelos si estan habilitados.

    La referencia para normalizar los proxies se calcula solo con Train.
    ``RUN_TRAINING=False`` deja disponible el CSV de medidas sin ajustar modelos.
    Cada ejecucion usa una carpeta con fecha y filtro de categoria para
    conservar los resultados anteriores.
    """
    if TIME_LIMIT_SECONDS <= 0:
        raise ValueError("TIME_LIMIT_SECONDS debe ser mayor que cero.")
    # Fase 1: cargar metadatos; su split determina la separacion Train/Test.
    metadata = leer_metadatos(
        path=METADATA_PATH,
        category_filter=CATEGORY_FILTER,
    )
    if metadata.empty:
        raise ValueError("No hay imagenes en image_types.csv para los filtros.")

    # Fase 2: extraer una fila por usuario e interfaz mediante el modulo
    # compartido. Se pasan los ajustes locales de rejilla para conservar el
    # comportamiento configurable de este script sin hacerlo dependiente de
    # ningun otro pipeline numerado.
    data = extraer_datos_compartidos(
        logs_dir=LOGS_DIR,
        metadata=metadata,
        images_dir=IMAGES_DIR,
        aoi_grid_columns=AOI_GRID_COLUMNS,
        aoi_grid_rows=AOI_GRID_ROWS,
        attention_grid_size=ATTENTION_GRID_SIZE,
    )
    if data.empty:
        raise ValueError("No se generaron sesiones con fijaciones validas.")

    # category es descriptivo y procede del metadata, no del log de fijaciones.
    category_by_image = metadata.set_index("Image Name")["Category"].to_dict()
    data["category"] = data["image_name"].map(category_by_image)
    if data["category"].isna().any():
        raise ValueError("No se pudo asociar la categoria a todas las imagenes.")
    # Solo Train fija los rangos min/max de los proxies. La misma referencia
    # se aplica a filas de Train y Test para evitar que la distribucion del
    # conjunto de evaluacion influya en la escala de los indicadores.
    train_reference = data[
        data["official_split"] == "train"
    ][list(PROXY_TARGET_COLUMNS)]
    data = calcular_indicadores_proxy(
        data,
        normalization_reference=train_reference,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    category_name = (CATEGORY_FILTER or "todas").casefold()
    run_dir = RESULTS_DIR / f"{timestamp}_{category_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "targets_y_proxies_input_completo.csv"
    data.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Sesiones usuario-interfaz: {len(data)}")
    print(f"Targets con TTFF AOI calculable: {data['target_ttff_seconds'].notna().sum()}")
    print(f"Targets y proxies: {output_path}")
    print(
        "Aviso: las AOI text-like/visual-like son aproximaciones basadas en "
        "pixeles, no clasificaciones semanticas."
    )

    # Fase 3: opcionalmente ajustar y evaluar cada target con el split oficial.
    # Entrenar_predecir_targets recibe las columnas objetivo completas y
    # aplica FEATURE_COLUMNS_BY_TARGET para excluir predictores por objetivo.
    if not RUN_TRAINING:
        return

    predictions, metrics = entrenar_predecir_targets(
        data=data,
        feature_columns=FEATURE_COLUMNS,
        target_columns=PROXY_TARGET_COLUMNS,
        output_dir=run_dir,
        time_limit=TIME_LIMIT_SECONDS,
        feature_columns_by_target=FEATURE_COLUMNS_BY_TARGET,
        run_shap_top3=RUN_SHAP_TOP3,
    )

    predictions = agregar_proxies_observados_y_predichos(
        predictions=predictions,
        train_reference=train_reference,
    )
    predictions.to_csv(
        run_dir / "predicciones_y_proxies_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Metricas de targets: {run_dir / 'metricas_test.csv'}")
    print(f"Comparacion de modelos: {run_dir / 'comparacion_modelos.csv'}")
    print(f"Predicciones y proxies: {run_dir / 'predicciones_y_proxies_test.csv'}")
    if RUN_SHAP_TOP3:
        print(
            "Resultados de modelos con 3 features SHAP: "
            f"{run_dir / 'top_3_features_shap'}"
        )
    print(metrics[["target", "rmse", "mae", "r2"]].to_string(index=False))


if __name__ == "__main__":
    ejecutar()
