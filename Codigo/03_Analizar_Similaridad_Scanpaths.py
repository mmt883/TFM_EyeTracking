"""Analiza si los scanpaths se parecen dentro y entre participantes.

La similitud intra-persona compara interfaces distintas vistas por la misma
persona. La similitud entre personas compara participantes distintos mirando
la misma interfaz, controlando asi que el estimulo sea el mismo.
"""

import itertools
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from Funciones_Comunes import (
    LOGS_DIR,
    METADATA_PATH,
    PROJECT_DIR,
    leer_metadatos,
    obtener_columna_tiempo,
    normalizar_nombre_imagen,
    participante_desde_nombre_archivo,
)

RESULTS_DIR = PROJECT_DIR / "Resultados" / "Resultados_Similaridad_Scanpaths"

# ==========================================
# CONFIGURACION
# ==========================================
# Cambia estos atributos para filtrar el analisis sin argumentos de terminal.
CATEGORY_FILTER = None  # None o desktop, mobile, poster, web.
USER_FILTER = None  # None o id como "kh000"; con un usuario no hay pares inter-persona.
SPLIT_FILTER = "all"  # Opciones: "all", "train" o "test" del split oficial.
MAX_PAIRS_PER_GROUP = 1000  # Entero >= 1; limita combinaciones por persona/imagen.
RANDOM_SEED = 42  # Entero; controla el muestreo reproducible de pares.

# ==========================================
# FASE 1: CARGA DE SCANPATHS
# ==========================================
def cargar_scanpaths(logs_dir, metadata, user_filter=None, split_filter="all"):
    """Carga secuencias ordenadas de coordenadas con filtros del dataset.

    Args:
        logs_dir: carpeta de logs de eye-tracking.
        metadata: tabla creada por leer_metadatos(), con split oficial.
        user_filter: participante opcional, por ejemplo "kh000".
        split_filter: "all", "train" o "test" segun image_types.csv.

    Returns:
        Lista de diccionarios, uno por participante e interfaz con datos validos.
    """
    scanpaths_by_key = {}
    user_filter = user_filter.strip().casefold() if user_filter else None

    # La particion y la categoria se filtran desde image_types.csv.
    if split_filter != "all":
        metadata = metadata[metadata["_split"] == split_filter].copy()

    for log_path in sorted(Path(logs_dir).glob("*.csv")):
        participant = participante_desde_nombre_archivo(log_path)
        if participant is None or (user_filter and participant != user_filter):
            continue

        try:
            log = pd.read_csv(log_path, low_memory=False)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
            print(f"Aviso: no se pudo leer {log_path.name}: {error}")
            continue

        # TIME(...) ordena los puntos en el tiempo.
        time_column = obtener_columna_tiempo(log.columns)
        if "MEDIA_NAME" not in log.columns or time_column is None:
            print(f"Aviso: {log_path.name} no tiene MEDIA_NAME o TIME valido.")
            continue

        # Las columnas internas facilitan agrupar y ordenar sin modificar los
        # nombres originales del CSV ni las coordenadas normalizadas.
        log = log.copy()
        log["_image_key"] = log["MEDIA_NAME"].map(normalizar_nombre_imagen)
        log["_elapsed_time"] = pd.to_numeric(log[time_column], errors="coerce")
        log["_x"] = pd.to_numeric(log.get("FPOGX"), errors="coerce")
        log["_y"] = pd.to_numeric(log.get("FPOGY"), errors="coerce")

        # FPOGV=1 indica una coordenada de mirada valida. Se exige x, y y tiempo
        # para que la secuencia resultante describa puntos comparables.
        if "FPOGV" in log.columns:
            valid = pd.to_numeric(log["FPOGV"], errors="coerce").eq(1)
            log = log[valid].copy() # Se eligen solo fijaciones con coordenadas validas.
        log = log.dropna(subset=["_elapsed_time", "_x", "_y"])

        # Una secuencia corresponde a un participante mirando una interfaz.
        # Se agrupa por imagen y se descartan las que no esten en image_types.csv.
        for image_key, group in log.groupby("_image_key", sort=False):
            if image_key not in metadata.index:
                continue
            image_metadata = metadata.loc[image_key]
            group = group.sort_values("_elapsed_time")
            coordinates = group[["_x", "_y"]].to_numpy(dtype=float)

            # Una secuencia de un solo punto no permite comparar un patron de
            # recorrido, asi que se omite.
            if len(coordinates) < 2:
                continue

            key = (participant, image_key)
            # image_key es la clave normalizada para agrupar/unir; image_name
            # conserva el nombre oficial legible (incluidas mayusculas y espacios).
            scanpaths_by_key[key] = {
                "participant": participant,
                "image_key": image_key,
                "image_name": image_metadata["Image Name"],
                "category": image_metadata["Category"],
                "official_split": image_metadata["_split"],
                "coordinates": coordinates,
            }

    return list(scanpaths_by_key.values())


# ==========================================
# FASE 2: DISTANCIA ENTRE TRAYECTORIAS
# ==========================================

# Distancia DTW (Dynamic Time Warping) entre dos secuencias de coordenadas.
def distancia_dtw(first, second):
    """Calcula el coste DTW medio entre dos secuencias de coordenadas.

    El alineamiento dinamico permite comparar rutas con distinta cantidad de
    fijaciones. La distancia euclidea se calcula en coordenadas normalizadas.
    """
    row_count = len(first)
    column_count = len(second)

    # cost guarda el menor coste acumulado para cada prefijo de ambas rutas.
    # path_length permite devolver el coste medio del camino, no una suma que
    # creciera automaticamente con el numero de fijaciones alineadas.
    cost = np.full((row_count + 1, column_count + 1), np.inf)
    path_length = np.zeros((row_count + 1, column_count + 1), dtype=int)
    # La celda (0, 0) es el inicio del alineamiento; otros bordes permanecen
    # infinitos para impedir saltarse puntos iniciales de una secuencia.
    cost[0, 0] = 0.0

    for row in range(1, row_count + 1):
        for column in range(1, column_count + 1):
            # El siguiente punto puede alinearse avanzando por primera ruta,
            # por segunda ruta o por ambas. Escogemos el predecesor mas barato.
            previous = (
                cost[row - 1, column],
                cost[row, column - 1],
                cost[row - 1, column - 1],
            )
            best_previous = int(np.argmin(previous))
            if best_previous == 0:
                previous_row, previous_column = row - 1, column
            elif best_previous == 1:
                previous_row, previous_column = row, column - 1
            else:
                previous_row, previous_column = row - 1, column - 1

            # Se agrega la distancia euclidea local entre dos fijaciones
            # normalizadas; DTW acumula la distancia de todo el alineamiento.
            point_distance = float(np.linalg.norm(first[row - 1] - second[column - 1]))
            cost[row, column] = cost[previous_row, previous_column] + point_distance
            path_length[row, column] = path_length[previous_row, previous_column] + 1

    return float(cost[row_count, column_count] / path_length[row_count, column_count]) # Normaliza el coste DTW por la longitud del camino.


def similitud_desde_distancia(distance):
    """Normaliza la distancia a similitud: 1 es igual y 0 es la maxima distancia."""
    # En coordenadas [0, 1], sqrt(2) es la mayor distancia euclidea entre puntos.
    maximum_distance = float(np.sqrt(2.0))
    return float(np.clip(1.0 - distance / maximum_distance, 0.0, 1.0))


def seleccionar_pares(items, maximum_pairs, random_generator):
    """Forma pares y los limita de manera reproducible si el grupo es grande."""
    # combinations evita pares repetidos e incluye cada par una sola vez.
    pairs = list(itertools.combinations(items, 2))
    if len(pairs) > maximum_pairs:
        # El muestreo acota el coste cuadratico de comparar todos los pares.
        pairs = random_generator.sample(pairs, maximum_pairs)
    return pairs


# ==========================================
# FASE 3: COMPARACIONES Y RESUMENES
# ==========================================
def comparar_scanpaths(scanpaths, maximum_pairs_per_group=1000, seed=42):
    """Compara dentro de participante y entre participantes por interfaz."""
    random_generator = random.Random(seed)
    records = []
    by_participant = defaultdict(list)
    by_image = defaultdict(list)

    # Creamos grupos distintos para las dos preguntas del estudio.
    for scanpath in scanpaths:
        by_participant[scanpath["participant"]].append(scanpath)
        by_image[scanpath["image_key"]].append(scanpath)

    # Comparacion intra-persona: misma persona, distintas interfaces. Mide
    # consistencia espacial general, pero puede verse afectada por el diseño.
    for participant, participant_paths in sorted(by_participant.items()):
        pairs = seleccionar_pares(
            items=participant_paths,
            maximum_pairs=maximum_pairs_per_group,
            random_generator=random_generator,
        )
        for first, second in pairs:
            distance = distancia_dtw(
                first=first["coordinates"],
                second=second["coordinates"],
            )
            records.append(
                {
                    "comparison": "same_person_different_interfaces",
                    "group_id": participant,
                    "participant_a": participant,
                    "participant_b": participant,
                    "image_a": first["image_name"],
                    "image_b": second["image_name"],
                    "category_a": first["category"],
                    "category_b": second["category"],
                    "split_a": first["official_split"],
                    "split_b": second["official_split"],
                    "dtw_distance": distance,
                    "similarity": similitud_desde_distancia(distance=distance),
                }
            )

    # Comparacion entre personas: misma interfaz, diferentes participantes.
    # Al fijar el estimulo, se reduce el efecto de comparar diseños distintos.
    for image_key, image_paths in sorted(by_image.items()):
        unique_participants = {path["participant"] for path in image_paths}
        if len(unique_participants) < 2:
            continue
        # Aunque la clave suele contener un registro por participante e imagen,
        # se comprueba el participante para excluir autocorrelaciones por seguridad.
        pairs = seleccionar_pares(
            items=image_paths,
            maximum_pairs=maximum_pairs_per_group,
            random_generator=random_generator,
        )
        for first, second in pairs:
            if first["participant"] == second["participant"]:
                continue
            distance = distancia_dtw(
                first=first["coordinates"],
                second=second["coordinates"],
            )
            records.append(
                {
                    "comparison": "different_people_same_interface",
                    "group_id": first["image_name"],
                    "participant_a": first["participant"],
                    "participant_b": second["participant"],
                    "image_a": first["image_name"],
                    "image_b": second["image_name"],
                    "category_a": first["category"],
                    "category_b": second["category"],
                    "split_a": first["official_split"],
                    "split_b": second["official_split"],
                    "dtw_distance": distance,
                    "similarity": similitud_desde_distancia(distance=distance),
                }
            )

    return pd.DataFrame(records)


def resumir_comparaciones(pairs):
    """Crea resumen global y por participante/interfaz."""
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Una distancia DTW menor indica scanpaths mas cercanos; la similitud se
    # normaliza respecto a la distancia maxima entre coordenadas en [0, 1].
    # Resumen global compara las dos definiciones de similitud; el resumen por
    # grupo permite detectar participantes o interfaces especialmente distintos.
    overall = (
        pairs.groupby("comparison")
        .agg(
            pair_count=("similarity", "size"),
            mean_similarity=("similarity", "mean"),
            median_similarity=("similarity", "median"),
            mean_dtw_distance=("dtw_distance", "mean"),
            median_dtw_distance=("dtw_distance", "median"),
        )
        .reset_index()
    )
    by_group = (
        pairs.groupby(["comparison", "group_id"])
        .agg(
            pair_count=("similarity", "size"),
            mean_similarity=("similarity", "mean"),
            median_similarity=("similarity", "median"),
            median_dtw_distance=("dtw_distance", "median"),
        )
        .reset_index()
    )
    return overall, by_group


# ==========================================
# FASE 4: EJECUCION
# ==========================================
def ejecutar():
    """Carga los datos filtrados, calcula DTW y guarda tablas de resultados."""
    if MAX_PAIRS_PER_GROUP < 1:
        raise ValueError("MAX_PAIRS_PER_GROUP debe ser al menos 1.")
    if SPLIT_FILTER not in {"all", "train", "test"}:
        raise ValueError('SPLIT_FILTER debe ser "all", "train" o "test".')

    metadata = leer_metadatos(
        path=METADATA_PATH,
        category_filter=CATEGORY_FILTER,
    )
    if metadata.empty:
        raise ValueError("No hay imagenes en image_types.csv para ese filtro.")

    scanpaths = cargar_scanpaths(
        logs_dir=LOGS_DIR,
        metadata=metadata,
        user_filter=USER_FILTER,
        split_filter=SPLIT_FILTER,
    )
    if not scanpaths:
        raise ValueError("No se encontraron scanpaths validos para esos filtros.")

    pairs = comparar_scanpaths(
        scanpaths=scanpaths,
        maximum_pairs_per_group=MAX_PAIRS_PER_GROUP,
        seed=RANDOM_SEED,
    )
    overall, by_group = resumir_comparaciones(pairs)

    # Cada ejecucion y combinacion de filtros tiene su propia carpeta.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    category_name = (CATEGORY_FILTER or "todas").casefold()
    run_dir = RESULTS_DIR / f"{timestamp}_{category_name}_{SPLIT_FILTER}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Este indice registra que secuencias entraron en el analisis sin guardar
    # de nuevo todos los puntos de mirada dentro del CSV de comparaciones.
    pd.DataFrame(
        [
            {
                "participant": item["participant"],
                "image_name": item["image_name"],
                "category": item["category"],
                "official_split": item["official_split"],
                "fixation_count": len(item["coordinates"]),
            }
            for item in scanpaths
        ]
    ).to_csv(run_dir / "scanpaths_incluidos.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(run_dir / "pares_similares.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(run_dir / "resumen_global.csv", index=False, encoding="utf-8-sig")
    by_group.to_csv(run_dir / "resumen_por_grupo.csv", index=False, encoding="utf-8-sig")

    print(f"Scanpaths incluidos: {len(scanpaths)}")
    print(f"Participantes: {len({item['participant'] for item in scanpaths})}")
    print(f"Comparaciones calculadas: {len(pairs)}")
    if USER_FILTER:
        print("Aviso: USER_FILTER limita el analisis a una persona; no hay comparacion inter-persona.")
    if overall.empty:
        print("No hay pares suficientes para calcular similitudes con estos filtros.")
    else:
        print("Resumen global (menor distancia / mayor similitud indica rutas mas parecidas):")
        print(overall.to_string(index=False))
    print(f"Resultados guardados en: {run_dir}")


# Al importar este modulo solo se definen funciones; la ejecucion comienza aqui.
if __name__ == "__main__":
    ejecutar()