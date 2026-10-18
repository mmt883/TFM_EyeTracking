"""Primera prueba de extraccion de caracteristicas y AutoGluon para UEyes.

Selecciona imagenes segun categoria y split oficial, agrega los logs por imagen
y entrena una regresion sencilla como ejemplo inicial de extremo a extremo.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

from Funciones_Comunes import (
    LOGS_DIR,
    METADATA_PATH,
    PROJECT_DIR,
)

RESULTS_DIR = PROJECT_DIR / "Resultados" / "Resultados_PruebaInicial"

# ==========================================
# CONFIGURACIÓN
# ==========================================
# Máximo de interfaces que se seleccionan por categoría y por split.
N_TRAIN_POR_CATEGORIA = 20  # Entero >= 0: maximo de imagenes Train por categoria.
N_TEST_POR_CATEGORIA = 8  # Entero >= 0: maximo de imagenes Test por categoria.
TARGETS_TO_PREDICT = (
    "target_scanpath_length",
    "target_dispersion",
    "saccade_mag_media",
    "num_fijaciones",
)
TIME_LIMIT_SECONDS = 1200  # Segundos de AutoGluon; debe ser un entero positivo.


# ==========================================
# FASE 1: SELECCIÓN ESTRATIFICADA
# ==========================================
def seleccionar_subconjunto(
    ruta_metadata,
    n_train_por_categoria=N_TRAIN_POR_CATEGORIA,
    n_test_por_categoria=N_TEST_POR_CATEGORIA,
):
    """
    Lee image_types.csv y selecciona interfaces para Train y Test por categoría.

    Args:
        ruta_metadata (str): ruta del CSV de metadatos.
        n_train_por_categoria (int): máximo de interfaces Train por categoría.
        n_test_por_categoria (int): máximo de interfaces Test por categoría.

    Returns:
        dict: pares con formato {"nombre_interfaz": "Train" o "Test"}.
        Devuelve un diccionario vacío si falta el archivo o no hay selección.
    """
    print(f"Cargando metadata desde {ruta_metadata}...")
    try:
        df_meta = pd.read_csv(ruta_metadata, sep=';')
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo de metadatos en {ruta_metadata}")
        return {}

    # DataFrame es una tabla de filas y columnas. Estas son las columnas
    # requeridas para identificar cada interfaz, categoría y tipo de split.
    columnas_esperadas = ["Image Name", "Category", "Train/Test"]
    columnas_faltantes = [col for col in columnas_esperadas if col not in df_meta.columns]
    if columnas_faltantes:
        print(f"ERROR: Faltan columnas en metadata: {columnas_faltantes}")
        print(df_meta.columns.tolist())
        return {}

    # Usamos copy() para modificar una copia independiente del DataFrame original.
    df_meta = df_meta[columnas_esperadas].copy()

    # Normalizamos los valores de texto.
    df_meta["Image Name"] = df_meta["Image Name"].astype(str).str.strip()
    df_meta["Category"] = df_meta["Category"].astype(str).str.strip()
    df_meta["Train/Test"] = df_meta["Train/Test"].astype(str).str.strip()

    # Lista donde se guardará un DataFrame con la muestra de cada grupo.
    seleccionadas = []

    # groupby agrupa las filas por los valores de las columnas indicadas.
    # La clave de cada grupo es una tupla (Category, Train/Test), por ejemplo:
    # ("desktop", "Train"). El valor asociado es otro DataFrame con las filas
    # que coinciden con esa categoría y ese split.
    
    # En cada iteración, (_, split_type) es la tupla de agrupación y grupo es
    # el DataFrame de filas correspondiente. "_" recibe Category, que aquí no
    # se necesita; split_type recibe el valor de Train/Test.
    for (_, split_type), grupo in df_meta.groupby(["Category", "Train/Test"]):
        if str(split_type).lower() == "train":
            n_muestras = n_train_por_categoria
        elif str(split_type).lower() == "test":
            n_muestras = n_test_por_categoria
        else:
            # Ignoramos grupos cuyo valor no sea Train ni Test.
            continue

        # Si hay menos filas disponibles que las solicitadas, se usan todas.
        n_real = min(n_muestras, len(grupo))
        if n_real > 0:
            # sample selecciona filas aleatoriamente. random_state fija la semilla
            # y hace que el resultado sea reproducible entre ejecuciones.
            muestra = grupo.sample(n=n_real, random_state=42)
            seleccionadas.append(muestra)

    if not seleccionadas:
        print("ADVERTENCIA: No se pudo realizar la selección estratificada.")
        return {}

    # concat une los grupos seleccionados en una sola tabla. ignore_index=True
    # genera un índice consecutivo nuevo para el DataFrame resultante.
    df_seleccion = pd.concat(seleccionadas, ignore_index=True)

    # zip empareja los nombres de interfaz con Train/Test y dict crea el formato:
    # {"nombre_interfaz": "Train"}.
    return dict(zip(df_seleccion["Image Name"], df_seleccion["Train/Test"]))

# ==========================================
# FASE 2: EXTRACCIÓN DE CARACTERÍSTICAS
# ==========================================
def procesar_archivos(ruta_datos, filtro_interfaces):
    """Extrae características de las interfaces seleccionadas en los logs.

    Args:
        ruta_datos (str): carpeta donde se encuentran los CSV de los logs.
        filtro_interfaces (dict): diccionario con formato
            {"nombre_interfaz": "Train" o "Test"}.

    Returns:
        pandas.DataFrame: una fila por interfaz y una columna por característica;
        si no se procesa ninguna interfaz, devuelve un DataFrame vacío.
    """
    # glob busca rutas que coinciden con el patrón; *.csv selecciona los CSV.
    # sorted establece un orden estable al procesar los archivos.
    archivos = sorted(Path(ruta_datos).glob("*.csv"))

    # Lista de diccionarios; cada diccionario contendrá las características de
    # una interfaz y después se convertirá en una fila del DataFrame final.
    resultados = []

    print(f"Buscando archivos de log en {ruta_datos}...")

    for archivo in archivos:
        try:
            # Carga el CSV como una tabla de pandas.
            df = pd.read_csv(archivo)

            # MEDIA_NAME identifica la interfaz a la que corresponde cada fila.
            if "MEDIA_NAME" not in df.columns:
                continue

            # Trabajamos sobre una copia y normalizamos los nombres de interfaz.
            df = df.copy()
            df["MEDIA_NAME"] = df["MEDIA_NAME"].astype(str).str.strip()

            # FPOGV marca si la muestra de mirada es válida. Solo filtramos si
            # esa columna existe en el archivo.
            if "FPOGV" in df.columns:
                df = df[df["FPOGV"] == 1].copy()

            # groupby("MEDIA_NAME") produce grupos por interfaz:
            # media_name es el nombre de la interfaz y grupo es el DataFrame
            # con todas las filas válidas que pertenecen a esa interfaz.
            for media_name, grupo in df.groupby("MEDIA_NAME"):
                # Se omiten interfaces no seleccionadas y con pocas observaciones.
                if media_name not in filtro_interfaces or len(grupo) < 3:
                    continue

                # Series vacías numéricas: se usan si faltan columnas de pupila
                # o no hay mediciones válidas. Una Serie es una columna de pandas.
                pupila_izq_valida = pd.Series(dtype=float)
                pupila_der_valida = pd.Series(dtype=float)

                # LPMMV y RPMMV indican la validez de las mediciones de pupila
                # izquierda (LPMM) y derecha (RPMM), respectivamente.
                if {"LPMM", "LPMMV"}.issubset(grupo.columns):
                    # loc conserva las medidas cuyo indicador de validez es 1.
                    pupila_izq_valida = grupo.loc[grupo["LPMMV"] == 1, "LPMM"]

                if {"RPMM", "RPMMV"}.issubset(grupo.columns):
                    pupila_der_valida = grupo.loc[grupo["RPMMV"] == 1, "RPMM"]

                # Las claves de este diccionario se convertirán en columnas del
                # DataFrame. Los valores son agregados calculados para la interfaz.
                features = {
                    # Identificación y split oficial de la interfaz.
                    "media_name": media_name,
                    "split_oficial": filtro_interfaces[media_name],

                    # len cuenta filas. mean calcula la media y std la desviación
                    # estándar de cada Serie numérica.
                    "num_fijaciones": len(grupo),
                    "duracion_fijacion_media": grupo["FPOGD"].mean() if "FPOGD" in grupo.columns else 0,
                    "duracion_fijacion_std": grupo["FPOGD"].std() if "FPOGD" in grupo.columns else 0,
                    "x_media": grupo["FPOGX"].mean() if "FPOGX" in grupo.columns else 0,
                    "y_media": grupo["FPOGY"].mean() if "FPOGY" in grupo.columns else 0,
                    "saccade_mag_media": grupo["SACCADE_MAG"].mean() if "SACCADE_MAG" in grupo.columns else 0,
                    "saccade_dir_std": grupo["SACCADE_DIR"].std() if "SACCADE_DIR" in grupo.columns else 0,
                    # Se calcula cada media solo con las medidas válidas de ese ojo.
                    "diametro_pupila_izq_media": pupila_izq_valida.mean() if not pupila_izq_valida.empty else 0,
                    "diametro_pupila_der_media": pupila_der_valida.mean() if not pupila_der_valida.empty else 0,
                    # Suma de las magnitudes de sacada como longitud agregada.
                    "target_scanpath_length": grupo["SACCADE_MAG"].sum() if "SACCADE_MAG" in grupo.columns else 0,
                    # La dispersión se aproxima sumando las desviaciones estándar
                    # de las coordenadas X e Y, si ambas columnas están presentes.
                    "target_dispersion": (grupo["FPOGX"].std() + grupo["FPOGY"].std()) if {"FPOGX", "FPOGY"}.issubset(grupo.columns) else 0,
                }
                # Guardamos una fila de características por cada interfaz.
                resultados.append(features)

        except Exception as e:
            # Un archivo con errores no impide que continúe el procesamiento
            # del resto de CSV.
            print(f"Error silenciado en archivo {archivo}: {e}")

    # Si no se procesó ninguna interfaz, devolvemos un DataFrame vacío.
    if not resultados:
        return pd.DataFrame()

    # pandas construye una tabla desde la lista de diccionarios: cada clave es
    # una columna y cada diccionario representa una fila.
    df_resultado = pd.DataFrame(resultados)

    # fillna(0) reemplaza valores ausentes (por ejemplo, una desviación estándar
    # no calculable) por cero. La tabla se devuelve y se exporta desde el bloque
    # principal para evitar volcar todas sus filas en la consola.
    df_resultado = df_resultado.fillna(0)
    return df_resultado

# ==========================================
# FASE 3: AUTOML
# ==========================================
# Este bloque solo se ejecuta al iniciar este archivo directamente, no cuando
# se importa desde otro módulo de Python.
if __name__ == "__main__":
    print("Iniciando pipeline de prueba de AutoGluon...")

    # Todas las salidas CSV de esta ejecución comparten un identificador temporal.
    id_ejecucion = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / id_ejecucion
    run_dir.mkdir(parents=True, exist_ok=True)

    # Obtiene un diccionario {nombre_interfaz: split} a partir de los metadatos.
    interfaces_seleccionadas = seleccionar_subconjunto(
        ruta_metadata=METADATA_PATH
    )
    print(f"Interfaces seleccionadas para la prueba: {len(interfaces_seleccionadas)}")

    if len(interfaces_seleccionadas) > 0:
        print("Interfaces seleccionadas:")
        # for interfaz in interfaces_seleccionadas:
        #     # interfaz es una clave del diccionario; el valor asociado indica
        #     # si pertenece al conjunto Train o Test.
        #     print(f"  - {interfaz}, {interfaces_seleccionadas[interfaz]}")
        print("Extrayendo características de los archivos de log...")
        df_tabular = procesar_archivos(
            ruta_datos=LOGS_DIR,
            filtro_interfaces=interfaces_seleccionadas,
        )

        if not df_tabular.empty:
            # Los CSV y el modelo quedan agrupados en la carpeta de esta ejecucion.
            ruta_csv_caracteristicas = run_dir / "caracteristicas_eye_tracking.csv"
            df_tabular.to_csv(
                ruta_csv_caracteristicas,
                index=False,
                encoding="utf-8-sig",
            )
            print(f"Características guardadas en: {ruta_csv_caracteristicas}")

            # Comparamos el split sin distinguir mayúsculas/minúsculas.
            # drop elimina columnas identificativas que no se usan como features.
            train_data = df_tabular[
                df_tabular["split_oficial"].astype(str).str.lower() == "train"
            ].drop(columns=["media_name", "split_oficial"])
            test_data = df_tabular[
                df_tabular["split_oficial"].astype(str).str.lower() == "test"
            ].drop(columns=["media_name", "split_oficial"])

            print(f"\nDatos tabulares generados: {len(train_data)} filas Train | {len(test_data)} filas Test")

            # Comprobamos que haya ejemplos en ambos conjuntos antes de entrenar.
            if train_data.empty or test_data.empty:
                print("No hay suficientes filas para entrenar o evaluar. Revisa la selección del split.")
            else:
                resultados_por_target = []

                for target in TARGETS_TO_PREDICT:
                    if target not in train_data.columns:
                        raise ValueError(f"No se ha generado la columna target '{target}'.")
                    if train_data[target].nunique() < 2:
                        raise ValueError(
                            f"El target '{target}' es constante en Train; "
                            "no se puede entrenar una regresión."
                        )

                    print(f"\n--- ENTRENANDO MODELO (TARGET = {target}) ---")

                    # No se usan los demás resultados agregados como predictores.
                    # Así, target_scanpath_length no se reconstruye directamente
                    # combinando saccade_mag_media y num_fijaciones.
                    columnas_no_predictoras = [
                        columna
                        for columna in TARGETS_TO_PREDICT
                        if columna != target
                    ]
                    train_m1 = train_data.drop(columns=columnas_no_predictoras)
                    test_m1 = test_data.drop(columns=columnas_no_predictoras)

                    target_dir = run_dir / target
                    target_dir.mkdir(parents=True, exist_ok=True)
                    model_path = target_dir / "modelo"

                    predictor = TabularPredictor(
                        label=target,
                        problem_type="regression",
                        path=str(model_path),
                        verbosity=0,
                    ).fit(
                        train_data=train_m1,
                        time_limit=TIME_LIMIT_SECONDS,
                        presets="medium_quality",
                    )

                    resultados = predictor.evaluate(
                        data=test_m1,
                        display=False,
                    )
                    fila_resultados = {"target": target, **resultados}
                    resultados_por_target.append(fila_resultados)

                    ruta_csv_evaluacion = target_dir / "evaluacion.csv"
                    pd.DataFrame([fila_resultados]).to_csv(
                        ruta_csv_evaluacion,
                        index=False,
                        encoding="utf-8-sig",
                    )
                    print(f"Métricas guardadas en: {ruta_csv_evaluacion}")

                ruta_csv_resumen = run_dir / "resumen_evaluaciones.csv"
                pd.DataFrame(resultados_por_target).to_csv(
                    ruta_csv_resumen,
                    index=False,
                    encoding="utf-8-sig",
                )
                print(f"Resumen de evaluaciones guardado en: {ruta_csv_resumen}")
        else:
            print("No se generaron datos tabulares. Revisa los nombres de archivo y las columnas.")
