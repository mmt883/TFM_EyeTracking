"""Utilidades compartidas para detectar AOI visuales heuristicas en imagenes."""

from pathlib import Path

import numpy as np
from PIL import Image


def clasificar_celda_visual(
    luminance,
    rgb,
    pixel_edge_difference_threshold=0.16,
    text_like_edge_density_min=0.045,
    text_like_gray_variation_min=0.055,
    text_like_saturation_max=0.22,
    visual_like_saturation_min=0.12,
    visual_like_gray_variation_min=0.16,
    visual_like_edge_density_min=0.04,
):
    """Clasifica una celda por bordes, contraste y saturacion de color.

    Es una aproximacion basada en pixeles: no reconoce palabras, objetos ni
    limites semanticos. Una celda se marca ``text_like`` si combina suficiente
    variacion de gris y bordes con saturacion baja; solo despues se comprueba
    si presenta color o textura suficientes para ``visual_like``. Lo restante
    queda como ``other``. El orden de esas reglas da prioridad a la primera
    categoria y evita que una celda de aspecto textual se marque como visual.

    Args:
        luminance: Matriz 2D de brillo normalizado en [0, 1].
        rgb: Matriz RGB de la misma celda, normalizada en [0, 1].
        pixel_edge_difference_threshold: Diferencia minima de luminancia
            entre vecinos para contar un borde.
        text_like_edge_density_min: Densidad minima de bordes para aspecto
            textual.
        text_like_gray_variation_min: Desviacion de luminancia minima para
            aspecto textual.
        text_like_saturation_max: Saturacion media maxima para aspecto textual.
        visual_like_saturation_min: Saturacion media que basta para marcar
            aspecto visual.
        visual_like_gray_variation_min: Contraste minimo alternativo para
            aspecto visual.
        visual_like_edge_density_min: Densidad de bordes que, junto al
            contraste minimo, permite marcar aspecto visual.

    Returns:
        Una de las etiquetas ``text_like``, ``visual_like`` o ``other``.
    """
    # La variacion de luminancia aproxima el contraste; la saturacion separa
    # zonas poco coloreadas (posible texto) de zonas mas coloridas.
    gray_variation = float(luminance.std())
    saturation = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    mean_saturation = float(saturation.mean())

    # La densidad es la fraccion de pares vecinos cuyo salto de luminancia
    # supera el umbral. Las matrices de una sola fila/columna no tienen
    # diferencias en ese eje y aportan densidad cero para el mismo.
    horizontal_edges = np.abs(np.diff(luminance, axis=1))
    vertical_edges = np.abs(np.diff(luminance, axis=0))
    horizontal_density = (
        float(np.mean(horizontal_edges >= pixel_edge_difference_threshold))
        if horizontal_edges.size
        else 0.0
    )
    vertical_density = (
        float(np.mean(vertical_edges >= pixel_edge_difference_threshold))
        if vertical_edges.size
        else 0.0
    )
    edge_density = (horizontal_density + vertical_density) / 2.0

    # Aplicar primero la regla de texto hace que prevalezca en celdas que
    # cumplen tambien una condicion visual, por ejemplo por sus bordes.
    if (
        gray_variation >= text_like_gray_variation_min
        and edge_density >= text_like_edge_density_min
        and mean_saturation < text_like_saturation_max
    ):
        return "text_like"
    if mean_saturation >= visual_like_saturation_min or (
        gray_variation >= visual_like_gray_variation_min
        and edge_density >= visual_like_edge_density_min
    ):
        return "visual_like"
    return "other"


def detectar_aois_imagen(
    image_path,
    analysis_max_size=256,
    grid_columns=16,
    grid_rows=12,
    pixel_edge_difference_threshold=0.16,
    text_like_edge_density_min=0.045,
    text_like_gray_variation_min=0.055,
    text_like_saturation_max=0.22,
    visual_like_saturation_min=0.12,
    visual_like_gray_variation_min=0.16,
    visual_like_edge_density_min=0.04,
):
    """Divide una captura en AOI de rejilla y clasifica sus patrones visuales.

    Devuelve cajas en pixeles de la imagen original para compararlas con las
    coordenadas de mirada normalizadas del log. La imagen se reduce solo para
    medir los rasgos visuales; la rejilla se proyecta despues sobre las
    dimensiones originales. Cada celda no vacia produce un AOI rectangular y
    sus umbrales se pasan a ``clasificar_celda_visual``.

    Args:
        image_path: Ruta de la captura que se analizara.
        analysis_max_size: Lado maximo de la miniatura usada para calcular
            pixeles y bordes; no cambia las cajas devueltas.
        grid_columns: Numero de columnas de la rejilla AOI.
        grid_rows: Numero de filas de la rejilla AOI.
        pixel_edge_difference_threshold: Salto de luminancia para contar un
            borde entre pixeles vecinos.
        text_like_edge_density_min: Densidad minima de bordes para la regla de
            aspecto textual.
        text_like_gray_variation_min: Contraste minimo para esa regla.
        text_like_saturation_max: Saturacion maxima para aspecto textual.
        visual_like_saturation_min: Saturacion suficiente para aspecto visual.
        visual_like_gray_variation_min: Contraste minimo alternativo visual.
        visual_like_edge_density_min: Densidad de bordes requerida junto al
            contraste alternativo visual.

    Returns:
        Tupla ``(aois, ancho_original, alto_original)``. Cada AOI es un
        diccionario con identificador de fila/columna, tipo heuristico, caja
        en pixeles y campos de compatibilidad ``label`` y ``score``.

    Raises:
        OSError: Si Pillow no puede abrir o leer el archivo de imagen.
    """
    with Image.open(Path(image_path)) as source_image:
        original_width, original_height = source_image.size
        image = source_image.convert("RGB")
        # La miniatura limita el coste de clasificacion sin cambiar el tamano
        # de las cajas, que se proyectan despues sobre la captura original.
        # Pillow conserva la relacion de aspecto y solo reduce, nunca amplia.
        image.thumbnail((analysis_max_size, analysis_max_size))
        rgb = np.asarray(image, dtype=np.float32) / 255.0

    # Luminancia perceptual aproximada calculada a partir de los canales RGB.
    luminance = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )
    resized_height, resized_width = luminance.shape
    aois = []

    # Los limites fraccionarios cubren toda la miniatura aunque sus dimensiones
    # no sean divisibles exactamente por la rejilla; las cajas de salida se
    # calculan directamente sobre el tamano original para evitar acumular
    # errores de escala entre miniatura y captura.
    for row in range(grid_rows):
        y1_small = round(row * resized_height / grid_rows)
        y2_small = round((row + 1) * resized_height / grid_rows)
        for column in range(grid_columns):
            x1_small = round(column * resized_width / grid_columns)
            x2_small = round((column + 1) * resized_width / grid_columns)
            cell_luminance = luminance[y1_small:y2_small, x1_small:x2_small]
            cell_rgb = rgb[y1_small:y2_small, x1_small:x2_small]
            if cell_luminance.size == 0:
                continue

            aoi_type = clasificar_celda_visual(
                cell_luminance,
                cell_rgb,
                pixel_edge_difference_threshold=pixel_edge_difference_threshold,
                text_like_edge_density_min=text_like_edge_density_min,
                text_like_gray_variation_min=text_like_gray_variation_min,
                text_like_saturation_max=text_like_saturation_max,
                visual_like_saturation_min=visual_like_saturation_min,
                visual_like_gray_variation_min=visual_like_gray_variation_min,
                visual_like_edge_density_min=visual_like_edge_density_min,
            )
            box = (
                round(column * original_width / grid_columns),
                round(row * original_height / grid_rows),
                round((column + 1) * original_width / grid_columns),
                round((row + 1) * original_height / grid_rows),
            )
            aois.append(
                {
                    "aoi_id": f"grid_{row}_{column}",
                    "aoi_type": aoi_type,
                    "label": aoi_type,
                    "box": box,
                    "score": np.nan,
                }
            )

    return aois, original_width, original_height


def clasificar_fijacion(x, y, image_width, image_height, aois):
    """Devuelve la AOI que contiene una fijacion, priorizando texto.

    FPOGX/FPOGY se expresan entre 0 y 1, por lo que primero se proyectan al
    tamano original de la captura. Los limites de las cajas son inclusivos,
    asi que un punto de borde puede pertenecer a mas de una AOI. En ese caso
    la seleccion compara primero el tipo (``text_like`` antes que otros) y
    despues el area; no se define un desempate adicional si ambos criterios
    coinciden. Si no hay caja contenedora se devuelve None.

    Args:
        x: Coordenada horizontal normalizada de la fijacion.
        y: Coordenada vertical normalizada de la fijacion.
        image_width: Ancho original de la imagen en pixeles.
        image_height: Alto original de la imagen en pixeles.
        aois: AOI devueltas por ``detectar_aois_imagen`` o por un detector
            compatible que proporcione ``box`` y ``aoi_type``.

    Returns:
        Diccionario de la AOI seleccionada, o None cuando la lista esta vacia
        o ninguna region contiene la coordenada.
    """
    # Las coordenadas fuera del viewport se limitan al borde mas cercano para
    # usar la misma convencion normalizada que en la extraccion de targets.
    pixel_x = float(np.clip(x, 0.0, 1.0) * image_width)
    pixel_y = float(np.clip(y, 0.0, 1.0) * image_height)
    candidates = []
    for aoi in aois:
        x1, y1, x2, y2 = aoi["box"]
        if x1 <= pixel_x <= x2 and y1 <= pixel_y <= y2:
            area = max(x2 - x1, 0) * max(y2 - y1, 0)
            # La prioridad de texto y luego el area permite resolver solapes
            # si en el futuro se usan AOI manuales de distinto tamano.
            type_priority = 0 if aoi["aoi_type"] == "text_like" else 1
            candidates.append((type_priority, area, aoi))

    if not candidates:
        return None
    return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2] #Devuelve la AOI con menor prioridad y menor area
