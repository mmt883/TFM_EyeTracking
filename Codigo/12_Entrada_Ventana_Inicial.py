"""Input 2: predice medidas de la sesion completa a partir de la mirada inicial.

El script combina dos fuentes calculadas por ``Funciones_Entradas``: las
caracteristicas de una ventana inicial de mirada y los targets observados
sobre toda la sesion.
La union se hace por imagen, participante y split. ``category`` no se usa como
entrada para que la prueba dependa solo de la mirada inicial. El split oficial
por interfaz se conserva para separar entrenamiento y evaluacion.

Esta ablacion responde a si los primeros segundos de fijaciones contienen
informacion predictiva sobre el comportamiento que se observa despues y/o en
la sesion completa. No calcula los targets a partir de esa ventana: los toma
del extractor completo y los mantiene separados de las features de entrada.
"""

from datetime import datetime

from Funciones_Comunes import (
    LOGS_DIR,
    METADATA_PATH,
    PROJECT_DIR,
    leer_metadatos,
)
from Funciones_Entradas import (
    EARLY_FEATURE_COLUMNS,
    extraer_datos_fijaciones_completas,
    extraer_ejemplos,
)
from Funciones_Modelado import (
    PROXY_TARGET_COLUMNS,
    agregar_proxies_observados_y_predichos,
    entrenar_predecir_targets,
)


RESULTS_DIR = PROJECT_DIR / "Resultados" / "Resultados_Input_Ventana_Inicial"

# ==========================================
# CONFIGURACION
# ==========================================
# Filtro opcional para limitar imagenes.
CATEGORY_FILTER = None  # None o categoria oficial: desktop, mobile, poster, web.
# Solo las fijaciones hasta este segundo generan las entradas del modelo. El
# extractor compartido divide la sesion usando este limite temporal.
FEATURE_CUTOFF_SECONDS = 3.0  # Mirada inicial utilizada como input.
# El limite se aplica a cada target en AutoGluon, no a toda la ejecucion.
# RUN_TRAINING=False permite verificar el dataset antes de iniciar los modelos.
TIME_LIMIT_SECONDS = 900  # Limite por target de regresion AutoGluon.
RUN_TRAINING = True  # False exporta los datos de prueba sin ajustar modelos.
# Si se activa, Funciones_Modelado escoge las tres variables mas importantes
# con SHAP sobre Train y evalua una segunda variante reducida.
RUN_SHAP_TOP3 = True  # True repite cada modelo usando sus tres features SHAP principales.


def ejecutar():
    """Construye los datos de entrada y ejecuta la prediccion opcional.

    Primero valida la configuracion y carga los metadatos. Luego obtiene las
    etiquetas completas y las entradas tempranas, las une y guarda el dataset.
    Si el entrenamiento esta activo, evalua cada target en el Test oficial y
    exporta predicciones y proxies observados/predichos. La extraccion de
    entradas y targets se delega en ``Funciones_Entradas`` para que este
    pipeline no dependa de la implementacion interna de otros scripts.

    El resultado contiene una fila por imagen-participante con un conjunto de
    predictores formado solo por las medidas tempranas y los targets medidos en
    la sesion completa. Los identificadores y el split se conservan para
    asociar filas y evaluar, pero el helper de modelado no los trata como
    predictores.
    """
    if FEATURE_CUTOFF_SECONDS <= 0:
        raise ValueError("FEATURE_CUTOFF_SECONDS debe ser mayor que cero.")
    if TIME_LIMIT_SECONDS <= 0:
        raise ValueError("TIME_LIMIT_SECONDS debe ser mayor que cero.")

    # Fase 1: los metadatos filtran interfaces y proporcionan el split oficial.
    # El campo Train/Test de cada imagen es la unica fuente de particion del
    # experimento; no se vuelve a sortear el split a nivel de observacion.
    metadata = leer_metadatos(
        path=METADATA_PATH,
        category_filter=CATEGORY_FILTER,
    )
    if metadata.empty:
        raise ValueError("No hay imagenes en image_types.csv para los filtros.")

    # Los targets resumen toda la sesion; no se usan como predictores. Esto es
    # deliberado: representan el resultado que se quiere anticipar a partir de
    # una observacion limitada a los primeros segundos.
    full_targets = extraer_datos_fijaciones_completas(
        logs_dir=LOGS_DIR,
        metadata=metadata,
    )
    if full_targets.empty:
        raise ValueError("No se pudieron calcular los targets completos.")

    # include_aoi_targets=False evita calcular etiquetas futuras de AOI; solo
    # interesa usar las medidas tempranas como predictores.
    early_features = extraer_ejemplos(
        logs_dir=LOGS_DIR,
        metadata=metadata,
        cutoff_seconds=FEATURE_CUTOFF_SECONDS,
        include_aoi_targets=False,
    )
    if early_features.empty:
        raise ValueError("No se generaron features para la ventana inicial.")

    # category procede de metadatos y no de la observacion ocular; excluirla
    # evita que esta ablacion use informacion ajena a la ventana inicial.
    feature_columns = [
        column for column in EARLY_FEATURE_COLUMNS if column != "category"
    ]
    
    # La union uno-a-uno verifica que cada sesion tenga exactamente una fila
    # de entrada y otra de targets; validate evita ocultar duplicados. Los
    # identificadores alinean las fuentes, pero no se incluyen en feature_columns.
    identifiers = ["image_name", "participant", "official_split"]
    model_data = early_features[identifiers + feature_columns].merge(
        full_targets[identifiers + list(PROXY_TARGET_COLUMNS)],
        on=identifiers,
        how="inner",
        validate="one_to_one",
    )
    if model_data.empty:
        raise ValueError("No coinciden las sesiones entre features y targets completos.")

    # Cada ejecucion tiene su carpeta para no sobrescribir resultados previos.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    category_name = (CATEGORY_FILTER or "todas").casefold()
    run_dir = RESULTS_DIR / f"{timestamp}_{category_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_data.to_csv(
        run_dir / "dataset_ventana_inicial_y_targets.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Sesiones con input inicial y target completo: {len(model_data)}")
    print(f"Features de entrada: {len(feature_columns)}")
    print(f"Dataset: {run_dir / 'dataset_ventana_inicial_y_targets.csv'}")

    # En modo exploratorio se conserva el dataset, pero no se entrenan modelos.
    # El retorno temprano es intencional: no crea predicciones ni metricas.
    if not RUN_TRAINING:
        return

    # Fase 3: ajuste en Train y evaluacion sobre Test, usando unicamente las
    # features de los primeros segundos especificadas arriba. La funcion
    # compartida mantiene las imagenes de cada particion separadas.
    predictions, metrics = entrenar_predecir_targets(
        data=model_data,
        feature_columns=feature_columns,
        target_columns=PROXY_TARGET_COLUMNS,
        output_dir=run_dir,
        time_limit=TIME_LIMIT_SECONDS,
        run_shap_top3=RUN_SHAP_TOP3,
    )
    # Los proxies se normalizan con rangos calculados en Train, nunca en Test.
    # La misma escala convierte tanto los targets observados como las
    # predicciones, haciendo comparables sus indicadores compuestos.
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
    if RUN_SHAP_TOP3:
        print(
            "Resultados de modelos con 3 features SHAP: "
            f"{run_dir / 'top_3_features_shap'}"
        )
    print(metrics[["target", "rmse", "mae", "r2"]].to_string(index=False))


if __name__ == "__main__":
    ejecutar()
