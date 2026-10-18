"""Funciones compartidas para leer metadatos y procesar registros de mirada."""

import re
from pathlib import Path

import numpy as np
import pandas as pd


METADATA_COLUMNS = ["Image Name", "Category", "Train/Test"]

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "Data" / "UEyes_dataset"
LOGS_DIR = DATASET_DIR / "eyetracker_logs"
METADATA_PATH = DATASET_DIR / "image_types.csv"
IMAGES_DIR = DATASET_DIR / "images"


def normalizar_nombre_imagen(value):
    """Normaliza un nombre de imagen para compararlo con metadatos.

    Elimina las carpetas previas al nombre de archivo, quita espacios externos
    y convierte el nombre a minúsculas sin perder la extensión.

    Args:
        value: Nombre o ruta de la imagen, habitualmente procedente del log.

    Returns:
        Nombre de archivo normalizado para usar como clave de comparación.
    """
    return Path(str(value).strip()).name.casefold()


def leer_metadatos(path, category_filter=None):
    """Lee, valida y normaliza los metadatos oficiales del dataset.

    Solo conserva las filas con una partición Train/Test reconocida y mantiene
    una fila por imagen. Si se especifica una categoría, la aplica sin
    distinguir mayúsculas y minúsculas.

    Args:
        path: Ruta al CSV oficial, separado por punto y coma.
        category_filter: Categoría opcional para filtrar, o ``None``.

    Returns:
        DataFrame indexado por ``_image_key`` con los metadatos y el split.

    Raises:
        ValueError: Si faltan columnas oficiales requeridas en el CSV.
    """
    metadata = pd.read_csv(path, sep=";")
    missing = [column for column in METADATA_COLUMNS if column not in metadata.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas en metadata: {missing}. "
            f"Columnas encontradas: {metadata.columns.tolist()}"
        )

    metadata = metadata[METADATA_COLUMNS].copy()
    metadata["Image Name"] = metadata["Image Name"].astype(str).str.strip()
    metadata["Category"] = metadata["Category"].astype(str).str.strip()
    metadata["Train/Test"] = metadata["Train/Test"].astype(str).str.strip()
    metadata["_image_key"] = metadata["Image Name"].map(normalizar_nombre_imagen)
    metadata["_split"] = metadata["Train/Test"].str.casefold()
    metadata = metadata[metadata["_split"].isin({"train", "test"})].copy()
    metadata = metadata.drop_duplicates("_image_key", keep="first")

    if category_filter:
        metadata = metadata[
            metadata["Category"].str.casefold() == category_filter.strip().casefold()
        ].copy()
    return metadata.set_index("_image_key", drop=False)


def participante_desde_nombre_archivo(path):
    """Extrae el identificador del participante del nombre de un log.

    Args:
        path: Ruta o nombre de archivo que puede incluir un ID ``kh`` seguido
            de dígitos, como ``00_kh000_fixations.csv``.

    Returns:
        Identificador en minúsculas, o ``None`` si el nombre no contiene uno.
    """
    match = re.search(r"(?:^|[_-])(kh\d+)(?:[_-]|$)", Path(path).name, re.IGNORECASE)
    return match.group(1).casefold() if match else None


def obtener_columna_tiempo(columns):
    """Localiza la columna TIME del log y descarta el contador TIMETICK.

    Args:
        columns: Nombres de columnas disponibles en el archivo de datos.

    Returns:
        El nombre original de la primera columna que empieza por ``TIME``
        excepto ``TIMETICK``, o ``None`` si no existe.
    """
    for column in columns:
        normalized = str(column).strip().upper()
        if normalized.startswith("TIME") and not normalized.startswith("TIMETICK"):
            return column
    return None


def columna_numerica(frame, name):
    """Devuelve una columna numérica y convierte datos inválidos en NaN.

    Si la columna solicitada no existe, crea una serie de NaN con el mismo
    índice que el DataFrame para mantener la alineación de las filas.

    Args:
        frame: DataFrame que contiene las mediciones.
        name: Nombre de la columna que se quiere convertir.

    Returns:
        Serie numérica alineada con el índice de ``frame``.
    """
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def resumir_fijaciones(early, future, category):
    """Calcula características de mirada temprana y targets de la ventana futura.

    Las fijaciones de ``early`` se resumen como entradas del modelo. Las
    fijaciones de ``future`` se usan únicamente para crear etiquetas, evitando
    mezclar ambas ventanas.

    Args:
        early: Filas de fijaciones pertenecientes a la ventana de entrada.
        future: Filas posteriores pertenecientes a la ventana objetivo.
        category: Categoría de la interfaz asociada a estas fijaciones.

    Returns:
        Diccionario de características tempranas y targets futuros, incluidos
        el recuento y la longitud de scanpath, calculada sumando ``SACCADE_MAG``
        en cada ventana para mantener el mismo criterio que la prueba inicial.
    """
    early_x = columna_numerica(frame=early, name="FPOGX")
    early_y = columna_numerica(frame=early, name="FPOGY")
    early_duration = columna_numerica(frame=early, name="FPOGD").dropna()

    early_coordinates = pd.DataFrame({"x": early_x, "y": early_y}).dropna()
    early_steps = np.hypot(
        early_coordinates["x"].diff(), early_coordinates["y"].diff()
    ).dropna()
    early_saccade_magnitude = columna_numerica(
        frame=early, name="SACCADE_MAG"
    ).dropna()

    early_x_std = early_coordinates["x"].std()
    early_y_std = early_coordinates["y"].std()
    early_x_std = 0.0 if pd.isna(early_x_std) else float(early_x_std)
    early_y_std = 0.0 if pd.isna(early_y_std) else float(early_y_std)

    future_duration = columna_numerica(frame=future, name="FPOGD").dropna()
    future_saccade_magnitude = columna_numerica(
        frame=future, name="SACCADE_MAG"
    ).dropna()

    return {
        "category": category,
        "early_fixation_count": int(len(early)),
        "early_duration_mean": float(early_duration.mean()) if not early_duration.empty else 0.0,
        "early_duration_std": float(early_duration.std()) if len(early_duration) > 1 else 0.0,
        "early_duration_total": float(early_duration.sum()),
        "early_x_mean": float(early_coordinates["x"].mean()) if not early_coordinates.empty else 0.0,
        "early_y_mean": float(early_coordinates["y"].mean()) if not early_coordinates.empty else 0.0,
        "early_x_std": early_x_std,
        "early_y_std": early_y_std,
        "early_spatial_dispersion": float(np.hypot(early_x_std, early_y_std)),
        "early_step_distance_mean": float(early_steps.mean()) if not early_steps.empty else 0.0,
        "early_scanpath_length": float(early_saccade_magnitude.sum()),
        "future_fixation_count": int(len(future)),
        "future_mean_duration": float(future_duration.mean()) if not future_duration.empty else np.nan,
        "future_scanpath_length": float(future_saccade_magnitude.sum()),
    }
