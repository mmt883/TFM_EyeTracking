"""Input 3: predice medidas de mirada usando captura y descriptores visuales.

Flujo general:
1. Calcula descriptores visuales y conserva la ruta de cada captura.
2. Une esos descriptores a los targets observados de la sesion completa,
   extraidos por ``Funciones_Entradas``.
3. Entrena un MultiModalPredictor con imagen y descriptores, y evalua Test.

Los descriptores se derivan de pixeles y AOI heuristicas. La entrada no
contiene fijaciones ni metadatos de participantes.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from Funciones_AOI import detectar_aois_imagen
from Funciones_Comunes import (
    IMAGES_DIR,
    LOGS_DIR,
    METADATA_PATH,
    PROJECT_DIR,
    leer_metadatos,
)
from Funciones_Entradas import extraer_datos_fijaciones_completas
from Funciones_Modelado import (
    PROXY_TARGET_COLUMNS,
    agregar_proxies_observados_y_predichos,
    entrenar_predecir_targets_multimodal,
)


RESULTS_DIR = PROJECT_DIR / "Resultados" / "Resultados_Input_Captura_Visual"

# ==========================================
# CONFIGURACION
# ==========================================
# Filtro opcional para restringir las imagenes de la ejecucion.
CATEGORY_FILTER = None  # None o categoria oficial: desktop, mobile, poster, web.
# Las copias de analisis se reducen a este tamanho maximo para acelerar los
# calculos. Las dimensiones originales se guardan como caracteristicas aparte.
VISUAL_ANALYSIS_MAX_SIZE = 256  # Resolucion maxima para los descriptores de pixeles.
# El limite de tiempo se aplica a cada regresion. La opcion de no entrenar
# permite inspeccionar primero el CSV construido.
TIME_LIMIT_SECONDS = 900  # Limite por target de regresion AutoGluon.
RUN_TRAINING = True  # False exporta caracteristicas y targets sin ajustar modelos.
# SHAP top 3 no se aplica a MultiModalPredictor; True solo emite un aviso.
RUN_SHAP_TOP3 = False


def extraer_caracteristicas_imagen(image_path):
    """Calcula descriptores visuales y heuristicas de saliencia para la captura.

    Se conservan las dimensiones originales; los descriptores de pixeles se
    calculan sobre una copia RGB reducida. Las fracciones AOI describen tipos
    heurísticos de regiones, no proporciones semanticas verificadas.

    Args:
        image_path: Ruta de una captura que puede abrir Pillow.

    Returns:
        Diccionario de escalares numericos con descriptores de color, contraste,
        densidad y distribucion espacial, tamano de bloques y saliencia.

    Notes:
        La imagen se convierte a RGB para que todas las capturas tengan tres
        canales comparables. Los valores de canal quedan en [0, 1]; las
        entropias se basan en histogramas y no describen objetos ni texto.
    """
    with Image.open(image_path) as source_image:
        # Se guarda el tamanho original antes de reducir la imagen para el analisis.
        image_width, image_height = source_image.size
        rgb_image = source_image.convert("RGB")
        rgb_image.thumbnail(
            (VISUAL_ANALYSIS_MAX_SIZE, VISUAL_ANALYSIS_MAX_SIZE)
        )
        rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0

    # Luminancia ponderada que aproxima el brillo percibido de cada pixel RGB.
    luminance = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )
    # Diferencia entre el canal mas alto y el mas bajo: es una aproximacion
    # sencilla a la saturacion, no una conversion al espacio HSV.
    saturation = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    # Diferencias entre pixeles vecinos aproximan los bordes horizontales y
    # verticales; edge_density es la fraccion que supera el umbral establecido.
    horizontal_edges = np.abs(np.diff(luminance, axis=1))
    vertical_edges = np.abs(np.diff(luminance, axis=0))
    edge_values = np.concatenate(
        [horizontal_edges.ravel(), vertical_edges.ravel()]
    )
    edge_density = (
        float(np.mean(edge_values >= 0.16)) if edge_values.size else 0.0
    )

    # Cada canal se divide en 16 intervalos para estimar la variedad de sus
    # intensidades. La entropia final promedia los tres canales por igual.
    channel_entropies = []
    # Recoge los colores RGB en 8 niveles por canal (0-7) para contar la diversidad 8*8*8 = 512 colores posibles. 
    # Se usa floor para que 1.0 quede en el nivel 7.
    quantized_colors = np.floor(rgb * 8).clip(0, 7).astype(np.uint8)

    # Entropia de color: -sum(p * log2(p)) para cada canal, promediada.
    for channel in range(3):
        histogram, _ = np.histogram(rgb[:, :, channel], bins=16, range=(0, 1))
        probabilities = histogram[histogram > 0] / histogram.sum()
        channel_entropies.append(
            float(-np.sum(probabilities * np.log2(probabilities)))
        )

    # La rejilla 4x4 resume la distribucion espacial del brillo y de la saliencia.
    cell_luminance_means = []
    cell_saliency_means = []
    padded_luminance = np.pad(luminance, 1, mode="edge") # 
    local_luminance = sum(
        padded_luminance[row:row + luminance.shape[0],
                         column:column + luminance.shape[1]]
        for row in range(3)
        for column in range(3)
    ) / 9.0
    local_contrast = np.abs(luminance - local_luminance)
    global_color = rgb.mean(axis=(0, 1))
    color_distance = np.linalg.norm(rgb - global_color, axis=2) / np.sqrt(3.0)
    saliency = local_contrast + color_distance
    saliency_values = saliency.ravel()
    salient_pixel_count = max(1, int(np.ceil(saliency_values.size * 0.1)))
    top_saliency = np.partition(
        saliency_values,
        saliency_values.size - salient_pixel_count,
    )[-salient_pixel_count:]
    total_saliency = float(saliency_values.sum())
    saliency_concentration = (
        float(top_saliency.sum() / total_saliency) if total_saliency else 0.0
    )

    saliency_rows = np.array_split(saliency, 4, axis=0)
    for row_index, row in enumerate(np.array_split(luminance, 4, axis=0)):
        for column_index, cell in enumerate(np.array_split(row, 4, axis=1)):
            if cell.size:
                cell_luminance_means.append(float(cell.mean()))
                saliency_cell = np.array_split(
                    saliency_rows[row_index], 4, axis=1
                )[column_index]
                if saliency_cell.size:
                    cell_saliency_means.append(float(saliency_cell.mean()))

    # Se usa el detector compartido con un tamanho de analisis equivalente.
    # Todas las celdas detectadas cuentan en el denominador, incluidas las
    # clasificadas como "other".
    aois, _, _ = detectar_aois_imagen(
        image_path=image_path,
        analysis_max_size=VISUAL_ANALYSIS_MAX_SIZE,
    )
    aoi_counts = pd.Series(
        [aoi["aoi_type"] for aoi in aois],
        dtype="string",
    ).value_counts()
    aoi_count = max(len(aois), 1)
    aoi_positions = []
    for aoi in aois:
        grid_row, grid_column = map(int, aoi["aoi_id"].split("_")[1:])
        aoi_positions.append((grid_row, grid_column, aoi["aoi_type"]))

    max_grid_row = max((position[0] for position in aoi_positions), default=-1)
    max_grid_column = max((position[1] for position in aoi_positions), default=-1)
    quadrant_counts = {
        quadrant: {"total": 0, "text_like": 0, "visual_like": 0}
        for quadrant in ("top_left", "top_right", "bottom_left", "bottom_right")
    }
    for grid_row, grid_column, aoi_type in aoi_positions:
        vertical = "top" if grid_row < (max_grid_row + 1) / 2 else "bottom"
        horizontal = (
            "left" if grid_column < (max_grid_column + 1) / 2 else "right"
        )
        quadrant = f"{vertical}_{horizontal}"
        quadrant_counts[quadrant]["total"] += 1
        if aoi_type in {"text_like", "visual_like"}:
            quadrant_counts[quadrant][aoi_type] += 1

    aoi_spatial_features = {}
    for quadrant, counts in quadrant_counts.items():
        for aoi_type in ("text_like", "visual_like"):
            denominator = max(counts["total"], 1)
            aoi_spatial_features[
                f"visual_{aoi_type}_aoi_{quadrant}_fraction"
            ] = float(counts[aoi_type] / denominator)

    # Componentes contiguos de celdas clasificadas como texto/visual son una
    # estimacion gruesa del area ocupada por bloques, no objetos segmentados.
    aoi_grid = {
        (grid_row, grid_column): aoi_type
        for grid_row, grid_column, aoi_type in aoi_positions
    }
    unvisited = {
        position
        for position, aoi_type in aoi_grid.items()
        if aoi_type in {"text_like", "visual_like"}
    }
    component_area_fractions = []
    while unvisited:
        start = unvisited.pop()
        component_type = aoi_grid[start]
        component = [start]
        stack = [start]
        while stack:
            current_row, current_column = stack.pop()
            for neighbor in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    neighbor in unvisited
                    and aoi_grid[neighbor] == component_type
                ):
                    unvisited.remove(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)
        component_area_fractions.append(len(component) / aoi_count)

    features = {
        "visual_image_width": int(image_width),
        "visual_image_height": int(image_height),
        "visual_aspect_ratio": float(image_width / image_height),
        "visual_mean_red": float(rgb[:, :, 0].mean()),
        "visual_mean_green": float(rgb[:, :, 1].mean()),
        "visual_mean_blue": float(rgb[:, :, 2].mean()),
        "visual_luminance_mean": float(luminance.mean()),
        "visual_luminance_std": float(luminance.std()),
        "visual_saturation_mean": float(saturation.mean()),
        "visual_saturation_std": float(saturation.std()),
        "visual_edge_density": edge_density,
        "visual_color_entropy_mean": float(np.mean(channel_entropies)),
        "visual_quantized_color_count": int(
            len(np.unique(quantized_colors.reshape(-1, 3), axis=0))
        ),
        "visual_grid_luminance_std": float(np.std(cell_luminance_means)),
        "visual_text_like_aoi_fraction": float(
            aoi_counts.get("text_like", 0) / aoi_count
        ),
        "visual_image_like_aoi_fraction": float(
            aoi_counts.get("visual_like", 0) / aoi_count
        ),
        "visual_other_aoi_fraction": float(aoi_counts.get("other", 0) / aoi_count),
        "visual_aoi_component_area_mean_fraction": float(
            np.mean(component_area_fractions) if component_area_fractions else 0.0
        ),
        "visual_aoi_component_area_max_fraction": float(
            np.max(component_area_fractions) if component_area_fractions else 0.0
        ),
        "visual_saliency_mean": float(saliency.mean()),
        "visual_saliency_std": float(saliency.std()),
        "visual_saliency_max": float(saliency.max()),
        "visual_saliency_top_decile_concentration": saliency_concentration,
        "visual_saliency_grid_std": float(np.std(cell_saliency_means)),
    }
    features.update(aoi_spatial_features)
    return features


def extraer_caracteristicas_imagenes(image_names, images_dir):
    """Calcula una fila por imagen unica y devuelve su nombre como clave.

    Ordenar y deduplicar los nombres evita analizar varias veces la misma
    captura cuando diferentes participantes observaron esa interfaz.

    Args:
        image_names: Nombres de captura referenciados por los targets.
        images_dir: Carpeta donde se encuentran los archivos de imagen.

    Returns:
        DataFrame con una fila por nombre de imagen, el campo ``image_name``
        y todas las caracteristicas devueltas por
        ``extraer_caracteristicas_imagen``.

    Raises:
        FileNotFoundError: Si una captura requerida no existe; no se omite
            silenciosamente porque dejaria sesiones sin su entrada visual.
    """
    records = []
    for image_name in sorted(set(image_names), key=str.casefold):
        image_path = Path(images_dir) / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"No se encontro la captura: {image_path}")
        record = extraer_caracteristicas_imagen(image_path)
        record["image_name"] = image_name
        record["image_path"] = str(image_path.resolve())
        records.append(record)
    return pd.DataFrame(records)


def ejecutar():
    """Extrae descriptores, crea el dataset y entrena si esta habilitado.

    Los descriptores de imagen se unen a cada sesion participante-interfaz.
    La categoria y el split proceden de los metadatos/targets, y no forman
    parte de las entradas visuales del modelo.

    El dataset exportado mantiene una fila por sesion: una captura puede
    repetirse para varios participantes, de modo que la union es muchos-a-uno.
    El modelo recibe la captura por ``image_path`` y solo descriptores visuales;
    los targets de mirada se reservan para etiquetas y evaluacion.
    """
    if VISUAL_ANALYSIS_MAX_SIZE < 16:
        raise ValueError("VISUAL_ANALYSIS_MAX_SIZE debe ser al menos 16 pixeles.")
    if TIME_LIMIT_SECONDS <= 0:
        raise ValueError("TIME_LIMIT_SECONDS debe ser mayor que cero.")

    # Fase 1: seleccionar imagenes y cargar su particion oficial.
    metadata = leer_metadatos(
        path=METADATA_PATH,
        category_filter=CATEGORY_FILTER,
    )
    if metadata.empty:
        raise ValueError("No hay imagenes en image_types.csv para los filtros.")

    # Fase 2: extraer los targets de mirada medidos sobre la sesion completa.
    # El modulo compartido evita depender del script 11 y conserva la misma
    # definicion de las medidas objetivo.
    target_data = extraer_datos_fijaciones_completas(
        logs_dir=LOGS_DIR,
        metadata=metadata,
        images_dir=IMAGES_DIR,
    )
    if target_data.empty:
        raise ValueError("No se pudieron calcular los targets observados.")

    # Fase 3: calcular una sola vez las caracteristicas por imagen y asociarlas
    # a todas las sesiones que vieron esa captura. Esto evita repetir el
    # procesamiento de pixeles por participante.
    image_features = extraer_caracteristicas_imagenes(
        image_names=target_data["image_name"],
        images_dir=IMAGES_DIR,
    )
    feature_columns = ["image_path"] + [
        column
        for column in image_features.columns
        if column not in {"image_name", "image_path"}
    ]
    # La clave image_name permite una union muchos-a-uno. validate comprueba
    # que la tabla visual tenga una fila por captura; participante e split
    # identifican la sesion, pero se excluyen de los predictores.
    identifiers = ["image_name", "participant", "official_split"]
    model_data = target_data[
        identifiers + list(PROXY_TARGET_COLUMNS)
    ].merge(
        image_features,
        on="image_name",
        how="left",
        validate="many_to_one",
    )
    if model_data[feature_columns].isna().any().any():
        raise ValueError("Faltan caracteristicas visuales despues de unir las capturas.")

    # Guardar cada ejecucion aparte evita perder datasets y metricas anteriores.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    category_name = (CATEGORY_FILTER or "todas").casefold()
    run_dir = RESULTS_DIR / f"{timestamp}_{category_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_data.to_csv(
        run_dir / "dataset_captura_visual_y_targets.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Sesiones con captura y targets: {len(model_data)}")
    print(f"Descriptores de imagen: {len(feature_columns)}")
    print(f"Dataset: {run_dir / 'dataset_captura_visual_y_targets.csv'}")
    print(
        "Aviso: los descriptores AOI son proxies visuales; no representan "
        "etiquetas semanticas verificadas."
    )

    # Se puede inspeccionar el dataset sin entrenar ni crear predicciones.
    if not RUN_TRAINING:
        return

    # Fase 4: la captura y sus descriptores son las unicas entradas; las
    # medidas observadas son etiquetas y referencias de evaluacion.
    predictions, metrics = entrenar_predecir_targets_multimodal(
        data=model_data,
        feature_columns=feature_columns,
        target_columns=PROXY_TARGET_COLUMNS,
        output_dir=run_dir,
        time_limit=TIME_LIMIT_SECONDS,
        run_shap_top3=RUN_SHAP_TOP3,
    )
    # Normalizar los proxies con la referencia Train para evitar usar estadisticas
    # del conjunto Test en el calculo de los indicadores.
    train_reference = model_data[
        model_data["official_split"] == "train"
    ][list(PROXY_TARGET_COLUMNS)]
    predictions = agregar_proxies_observados_y_predichos(
        predictions,
        train_reference=train_reference,
    )
    predictions.to_csv(
        run_dir / "predicciones_y_proxies_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Metricas de targets: {run_dir / 'metricas_test.csv'}")
    print(f"Predicciones y proxies: {run_dir / 'predicciones_y_proxies_test.csv'}")
    print(metrics[["target", "rmse", "mae", "r2"]].to_string(index=False))


if __name__ == "__main__":
    ejecutar()
