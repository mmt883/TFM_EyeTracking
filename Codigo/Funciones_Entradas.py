"""Extraccion compartida de caracteristicas y targets de mirada.

Los pipelines de analisis pueden obtener aqui medidas de sesion completa o
ventanas temporales, sin importar ni ejecutar entre si otros scripts.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from Funciones_AOI import clasificar_fijacion, detectar_aois_imagen
from Funciones_Comunes import (
    IMAGES_DIR,
    columna_numerica,
    normalizar_nombre_imagen,
    obtener_columna_tiempo,
    participante_desde_nombre_archivo,
    resumir_fijaciones,
)


AOI_GRID_COLUMNS = 16
AOI_GRID_ROWS = 16
ATTENTION_GRID_SIZE = 8

EARLY_FEATURE_COLUMNS = [
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


def calcular_entropia_y_cobertura(coordinates, grid_size):
    """Calcula entropia normalizada y cobertura de una rejilla de coordenadas.

    Las coordenadas se limitan al rango normalizado [0, 1] y se cuentan en una
    rejilla ``grid_size`` por ``grid_size``. La entropia de Shannon se divide
    por su maximo teorico (todas las celdas ocupadas por igual), por lo que
    queda entre 0 y 1. La cobertura es la fraccion de celdas con al menos una
    fijacion.

    Args:
        coordinates: DataFrame con columnas ``x`` e ``y`` normalizadas.
        grid_size: Numero de celdas en cada eje de la rejilla.

    Returns:
        Tupla ``(entropia_normalizada, cobertura)``. Si no hay coordenadas,
        ambas medidas son NaN porque no hay observaciones para estimarlas.
    """
    if coordinates.empty:
        return np.nan, np.nan

    # Las coordenadas fuera de pantalla se limitan a los bordes antes de
    # asignarlas a celdas; histogram2d incluye el extremo superior en la ultima.
    positions = np.clip(coordinates[["x", "y"]].to_numpy(dtype=float), 0.0, 1.0)
    counts, _, _ = np.histogram2d(
        positions[:, 0],
        positions[:, 1],
        bins=grid_size,
        range=((0.0, 1.0), (0.0, 1.0)),
    )
    occupied = counts[counts > 0]
    probabilities = occupied / occupied.sum()
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    # El maximo supone que la mirada se distribuye uniformemente por toda la
    # rejilla; la division hace comparables rejillas del mismo tamano.
    maximum_entropy = float(np.log2(grid_size**2))
    normalized_entropy = entropy / maximum_entropy if maximum_entropy > 0 else 0.0
    coverage = float(len(occupied) / (grid_size**2))
    return normalized_entropy, coverage


def resumir_sesion_completa(
    fixations,
    recorded_rows,
    time_column,
    aois,
    image_width,
    image_height,
    attention_grid_size=ATTENTION_GRID_SIZE,
):
    """Resume una sesion de mirada en entradas y targets numericos.

    ``fixations`` contiene solo las filas de fijacion validas; ``recorded_rows``
    conserva todas las filas temporales del estimulo para estimar la duracion
    observada. Las coordenadas de mirada estan normalizadas y las duraciones
    mantienen las unidades del log (segundos en los registros del proyecto).
    Si no se pudieron detectar AOI, los targets que dependen de ellas quedan
    como NaN en vez de interpretarse como cero.

    Args:
        fixations: Fijaciones validas de una imagen, ordenables por tiempo.
        recorded_rows: Todas las muestras temporales asociadas a esa imagen.
        time_column: Columna TIME que contiene el tiempo de cada muestra.
        aois: Regiones detectadas en la captura, o None si no se pudo abrirla.
        image_width: Ancho original de la captura en pixeles.
        image_height: Alto original de la captura en pixeles.

    Returns:
        Diccionario con predictores de la sesion completa y los once targets
        consumidos por ``Funciones_Modelado``. Para estadisticos espaciales
        sin coordenadas validas devuelve NaN; una duracion inexistente tambien
        queda como NaN, mientras que una duracion individual valida puede
        tener desviacion estandar cero.
    """
    # Las metricas de duracion ignoran valores negativos y datos no numericos.
    durations = columna_numerica(fixations, "FPOGD")
    durations = durations.where(durations >= 0).dropna()

    # La dispersion y el scanpath se calculan sobre coordenadas validas en
    # orden temporal; una sesion sin coordenadas no aporta estas medidas.
    coordinates = pd.DataFrame(
        {
            "x": columna_numerica(fixations, "FPOGX"),
            "y": columna_numerica(fixations, "FPOGY"),
        }
    ).dropna()
    if not coordinates.empty:
        coordinates["x"] = coordinates["x"].clip(0.0, 1.0)
        coordinates["y"] = coordinates["y"].clip(0.0, 1.0)
        x_std = float(coordinates["x"].std()) if len(coordinates) > 1 else 0.0
        y_std = float(coordinates["y"].std()) if len(coordinates) > 1 else 0.0
        # La dispersion es la distancia cuadratica media respecto al centro
        # de mirada; se expresa en unidades de coordenadas normalizadas.
        center = coordinates[["x", "y"]].mean().to_numpy(dtype=float)
        distances_from_center = np.linalg.norm(
            coordinates[["x", "y"]].to_numpy(dtype=float) - center,
            axis=1,
        )
        spatial_dispersion = float(
            np.sqrt(np.mean(np.square(distances_from_center)))
        )
        steps = np.hypot(
            coordinates["x"].diff(),
            coordinates["y"].diff(),
        ).dropna()
        scanpath_length = float(steps.sum())
        mean_step_distance = float(steps.mean()) if not steps.empty else 0.0
        x_mean = float(coordinates["x"].mean())
        y_mean = float(coordinates["y"].mean())
    else:
        x_mean = y_mean = x_std = y_std = np.nan
        spatial_dispersion = scanpath_length = mean_step_distance = np.nan

    # La duracion observada se obtiene de todas las filas del estimulo, no
    # solo de las fijaciones validas usadas en las otras metricas.
    elapsed = pd.to_numeric(recorded_rows[time_column], errors="coerce").dropna()
    observation_duration = (
        float(elapsed.max() - elapsed.min()) if len(elapsed) > 1 else 0.0
    )
    mean_duration = float(durations.mean()) if not durations.empty else np.nan
    duration_std = float(durations.std()) if len(durations) > 1 else (
        0.0 if len(durations) == 1 else np.nan
    )
    duration_total = float(durations.sum()) if not durations.empty else np.nan
    entropy, coverage = calcular_entropia_y_cobertura(
        coordinates=coordinates,
        grid_size=attention_grid_size,
    )

    # Acumula tiempo y secuencia de AOI para obtener permanencia, TTFF,
    # elementos visitados y cambios entre regiones consecutivas.
    text_dwell = 0.0
    image_dwell = 0.0
    visited_elements = set()
    visited_sequence = []
    fixation_times = []
    has_classifiable_duration = False

    if aois is not None:
        fixation_start_column = "_elapsed_time"
        if "FPOGS" in fixations.columns:
            gaze_starts = pd.to_numeric(fixations["FPOGS"], errors="coerce").dropna()
            if not gaze_starts.empty:
                fixation_start_column = "FPOGS"
        starts = pd.to_numeric(
            fixations[fixation_start_column],
            errors="coerce",
        ).dropna()
        trial_start = float(starts.min()) if not starts.empty else np.nan

        for _, fixation in fixations.sort_values("_elapsed_time").iterrows():
            x = pd.to_numeric(fixation.get("FPOGX"), errors="coerce")
            y = pd.to_numeric(fixation.get("FPOGY"), errors="coerce")
            if pd.isna(x) or pd.isna(y):
                continue
            aoi = clasificar_fijacion(
                x=float(x),
                y=float(y),
                image_width=image_width,
                image_height=image_height,
                aois=aois,
            )
            if aoi is None or aoi["aoi_type"] not in {"text_like", "visual_like"}:
                continue

            visited_elements.add(aoi["aoi_id"])
            visited_sequence.append(aoi["aoi_id"])
            duration = pd.to_numeric(fixation.get("FPOGD"), errors="coerce")
            if pd.notna(duration) and duration >= 0:
                has_classifiable_duration = True
                if aoi["aoi_type"] == "text_like":
                    text_dwell += float(duration)
                else:
                    image_dwell += float(duration)

            # TTFF usa el inicio de la primera fijacion respecto al inicio del
            # trial; si falta FPOGS, se recurre al tiempo de la fila del log.
            start = pd.to_numeric(
                fixation.get(fixation_start_column),
                errors="coerce",
            )
            if pd.isna(start):
                start = pd.to_numeric(fixation.get("_elapsed_time"), errors="coerce")
            if pd.notna(start) and pd.notna(trial_start):
                fixation_times.append(max(0.0, float(start) - trial_start))

    # Una transicion cuenta cada cambio consecutivo de AOI clasificable;
    # repetir una fijacion dentro de la misma celda no suma una transicion.
    transitions = sum(
        previous != current
        for previous, current in zip(visited_sequence, visited_sequence[1:])
    )
    if aois is None:
        ttff = text_dwell = image_dwell = np.nan
        visited_count = transitions = np.nan
    else:
        ttff = min(fixation_times) if fixation_times else np.nan
        if not has_classifiable_duration:
            text_dwell = image_dwell = np.nan
        visited_count = len(visited_elements)

    return {
        # Las features y algunos targets comparten definicion porque este
        # script es la linea base de input completo; el mapa de exclusiones de
        # arriba evita que un target se prediga desde su duplicado directo.
        "feature_observation_duration_seconds": observation_duration,
        "feature_fixation_duration_mean": mean_duration,
        "feature_fixation_duration_std": duration_std,
        "feature_fixation_duration_total": duration_total,
        "feature_fixation_count": int(len(fixations)),
        "feature_x_mean": x_mean,
        "feature_y_mean": y_mean,
        "feature_x_std": x_std,
        "feature_y_std": y_std,
        "feature_spatial_dispersion": spatial_dispersion,
        "feature_inter_fixation_distance_mean": mean_step_distance,
        # Targets principales y auxiliares. La normalizacion de estos valores
        # para construir proxies se realiza despues usando exclusivamente Train.
        "target_fixation_duration_mean": mean_duration,
        "target_ttff_seconds": ttff,
        "target_time_on_text_seconds": text_dwell,
        "target_time_on_images_seconds": image_dwell,
        "target_scanpath_length": scanpath_length,
        "target_spatial_dispersion": spatial_dispersion,
        "target_fixation_count": int(len(fixations)),
        "target_attention_entropy": entropy,
        "target_grid_coverage": coverage,
        "target_visited_elements": visited_count,
        "target_transitions": transitions,
    }


def extraer_datos_fijaciones_completas(
    logs_dir,
    metadata,
    images_dir=IMAGES_DIR,
    user_filter=None,
    aoi_grid_columns=AOI_GRID_COLUMNS,
    aoi_grid_rows=AOI_GRID_ROWS,
    attention_grid_size=ATTENTION_GRID_SIZE,
):
    """Extrae una fila de medidas completas por sesion participante-imagen."""
    records = []
    aoi_cache = {}
    user_filter = user_filter.strip().casefold() if user_filter else None

    for log_path in sorted(Path(logs_dir).glob("*.csv")):
        participant = participante_desde_nombre_archivo(log_path)
        if participant is None or (user_filter and participant != user_filter):
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

        log = log.copy()
        log["_image_key"] = log["MEDIA_NAME"].map(normalizar_nombre_imagen)
        log["_elapsed_time"] = pd.to_numeric(log[time_column], errors="coerce")
        log = log.dropna(subset=["_elapsed_time"])
        if "FPOGV" in log.columns:
            valid = pd.to_numeric(log["FPOGV"], errors="coerce").eq(1)
            valid_fixations = log[valid].copy()
        else:
            valid_fixations = log.copy()

        for image_key, recorded_rows in log.groupby("_image_key", sort=False):
            if image_key not in metadata.index:
                continue
            image_metadata = metadata.loc[image_key]
            image_name = image_metadata["Image Name"]

            if image_name not in aoi_cache:
                image_path = Path(images_dir) / image_name
                try:
                    aoi_cache[image_name] = detectar_aois_imagen(
                        image_path=image_path,
                        grid_columns=aoi_grid_columns,
                        grid_rows=aoi_grid_rows,
                    )
                except OSError as error:
                    print(f"Aviso: no se pudieron detectar AOI en {image_path}: {error}")
                    aoi_cache[image_name] = None

            image_fixations = valid_fixations[
                valid_fixations["_image_key"] == image_key
            ].sort_values("_elapsed_time")
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
                attention_grid_size=attention_grid_size,
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


def resumir_targets_visuales_futuros(
    future,
    cutoff_seconds,
    aois,
    image_width=None,
    image_height=None,
):
    """Calcula TTFF, permanencia AOI y dispersion de la ventana futura."""
    duration = columna_numerica(frame=future, name="FPOGD")
    duration = duration.where(duration >= 0)
    if aois is None or not duration.notna().any():
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
    """Extrae caracteristicas tempranas y, opcionalmente, targets futuros."""
    examples = []
    image_aoi_cache = {}
    user_filter = user_filter.strip().casefold() if user_filter else None

    for log_path in sorted(Path(logs_dir).glob("*.csv")):
        participant = participante_desde_nombre_archivo(log_path)
        if participant is None or (user_filter and participant != user_filter):
            continue
        try:
            log = pd.read_csv(log_path, low_memory=False)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
            print(f"Aviso: no se pudo leer {log_path.name}: {error}")
            continue

        time_column = obtener_columna_tiempo(log.columns)
        if "MEDIA_NAME" not in log.columns or time_column is None:
            print(f"Aviso: {log_path.name} no contiene MEDIA_NAME o una columna TIME válida.")
            continue

        log = log.copy()
        log["_image_key"] = log["MEDIA_NAME"].map(normalizar_nombre_imagen)
        log["_elapsed_time"] = pd.to_numeric(log[time_column], errors="coerce")
        if "FPOGV" in log.columns:
            valid = pd.to_numeric(log["FPOGV"], errors="coerce").eq(1)
            log = log[valid].copy()
        log = log.dropna(subset=["_elapsed_time"])

        for image_key, group in log.groupby("_image_key", sort=False):
            if image_key not in metadata.index:
                continue
            image_metadata = metadata.loc[image_key]
            group = group.sort_values("_elapsed_time")
            early = group[group["_elapsed_time"] <= cutoff_seconds]
            future = group[group["_elapsed_time"] > cutoff_seconds]
            if early.empty or future.empty:
                continue

            if include_aoi_targets:
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
                    aois = image_width = image_height = None
                else:
                    aois, image_width, image_height = image_aoi_data

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
