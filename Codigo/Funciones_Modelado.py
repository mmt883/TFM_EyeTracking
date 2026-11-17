"""Utilidades compartidas para comparar y explicar modelos AutoGluon."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import shutil
import tempfile
import warnings


# Limites deliberados para que las explicaciones por permutacion no crezcan
# sin control con el numero de sesiones y de variables.
SHAP_MAX_BACKGROUND_ROWS = 50
SHAP_MAX_EXPLANATION_ROWS = 10
SHAP_MAX_DISPLAY_FEATURES = 20

# Contrato comun de targets consumido por los codigos 11, 12 y 13. El orden
# tambien determina el orden de ajuste y exportacion de resultados.
PROXY_TARGET_COLUMNS = (
    "target_ttff_seconds",
    "target_time_on_text_seconds",
    "target_time_on_images_seconds",
    "target_scanpath_length",
    "target_fixation_count",
    "target_attention_entropy",
    "target_spatial_dispersion",
    "target_grid_coverage",
    "target_visited_elements",
    "target_transitions",
    "target_fixation_duration_mean",
)


def calcular_indicadores_proxy(data, normalization_reference=None):
    """Calcula los tres indicadores de las diapositivas a partir de targets.

    Cada magnitud se normaliza al rango observado en ``normalization_reference``
    (o en ``data`` si no se proporciona). Asi los resultados quedan entre 0 y 1
    y pueden calcularse de la misma forma para observaciones y predicciones.
    La referencia debe representar el conjunto de entrenamiento al evaluar un
    modelo, para que las filas de Test no definan su propia escala.

    Args:
        data: Tabla con una columna para cada target de
            ``PROXY_TARGET_COLUMNS``. Se devuelve una copia ampliada.
        normalization_reference: Tabla de referencia con esos mismos targets.
            Si es None, se usa ``data``; esta opcion conviene para describir
            datos observados, no para comparar Train y Test en una evaluacion.

    Returns:
        Copia de ``data`` con targets normalizados y tres indicadores proxy:
        eficiencia exploratoria, distribucion de atencion y estrategia
        exploratoria, ademas de la etiqueta categorica de esta ultima.

    Raises:
        ValueError: Si falta un target en los datos o en la referencia.
    """
    result = data.copy()
    reference = data if normalization_reference is None else normalization_reference
    normalized_columns = {}

    for column in PROXY_TARGET_COLUMNS:
        if column not in result.columns:
            raise ValueError(f"Falta el target requerido para indicadores proxy: {column}")
        if column not in reference.columns:
            raise ValueError(f"Falta {column} en la referencia de normalizacion.")

        # Los valores no convertibles pasan a NaN. Los limites se calculan
        # unicamente con valores finitos de la referencia seleccionada.
        values = pd.to_numeric(result[column], errors="coerce")
        reference_values = pd.to_numeric(reference[column], errors="coerce")
        reference_values = reference_values[np.isfinite(reference_values)]
        if reference_values.empty:
            normalized = pd.Series(np.nan, index=result.index, dtype=float)
        else:
            minimum = float(reference_values.min())
            maximum = float(reference_values.max())
            if maximum == minimum:
                normalized = pd.Series(
                    np.where(values.notna(), 0.0, np.nan),
                    index=result.index,
                    dtype=float,
                )
            else:
                # Recortar a [0, 1] evita que valores de Test fuera del rango Train
                # generen una escala incompatible con los pesos de los proxies.
                normalized = ((values - minimum) / (maximum - minimum)).clip(0, 1)

        normalized_column = f"{column}_norm"
        result[normalized_column] = normalized
        normalized_columns[column] = normalized

    # Para los tres componentes de eficiencia, valores menores (p. ej. TTFF,
    # longitud o numero de fijaciones) reciben una puntuacion normalizada mayor.
    efficiency_components = pd.concat(
        [
            1.0 - normalized_columns["target_ttff_seconds"],
            1.0 - normalized_columns["target_scanpath_length"],
            1.0 - normalized_columns["target_fixation_count"],
        ],
        axis=1,
    )
    # La entropia y dispersion bajas invierten su escala; la cobertura se deja
    # en sentido directo. Los pesos suman 1 para conservar el rango del proxy.
    attention_components = pd.concat(
        [
            1.0 - normalized_columns["target_attention_entropy"],
            1.0 - normalized_columns["target_spatial_dispersion"],
            normalized_columns["target_grid_coverage"],
        ],
        axis=1,
    )
    # La estrategia combina regiones visitadas, pocas transiciones y duracion
    # media de fijacion. Son definiciones operativas, no etiquetas psicologicas.
    strategy_components = pd.concat(
        [
            normalized_columns["target_visited_elements"],
            1.0 - normalized_columns["target_transitions"],
            normalized_columns["target_fixation_duration_mean"],
        ],
        axis=1,
    )

    result["proxy_exploration_efficiency"] = (
        0.4 * efficiency_components.iloc[:, 0]
        + 0.3 * efficiency_components.iloc[:, 1]
        + 0.3 * efficiency_components.iloc[:, 2]
    ).where(efficiency_components.notna().all(axis=1))
    result["proxy_attention_distribution"] = (
        0.4 * attention_components.iloc[:, 0]
        + 0.3 * attention_components.iloc[:, 1]
        + 0.3 * attention_components.iloc[:, 2]
    ).where(attention_components.notna().all(axis=1))
    result["proxy_exploration_strategy"] = (
        0.4 * strategy_components.iloc[:, 0]
        + 0.3 * strategy_components.iloc[:, 1]
        + 0.3 * strategy_components.iloc[:, 2]
    ).where(strategy_components.notna().all(axis=1))

    # Los cortes forman cuatro categorias ordenadas. Los NaN no satisfacen
    # ningun corte y reciben la etiqueta explicita de no calculable.
    strategy_score = result["proxy_exploration_strategy"]
    result["proxy_exploration_strategy_label"] = np.select(
        [
            strategy_score >= 0.75,
            strategy_score >= 0.50,
            strategy_score >= 0.25,
            strategy_score.notna(),
        ],
        [
            "Exploracion enfocada",
            "Exploracion sistematica",
            "Exploracion superficial",
            "Exploracion erratica",
        ],
        default="No calculable",
    )
    return result


def entrenar_predecir_targets(
    data,
    feature_columns,
    target_columns,
    output_dir,
    time_limit=30,
    feature_columns_by_target=None,
    run_shap_top3=False,
):
    """Ajusta un modelo de regresion por target y exporta su evaluacion.

    Si ``run_shap_top3`` es True, selecciona las cinco variables con mayor
    SHAP calculado sobre Train y repite el entrenamiento/evaluacion en una
    carpeta separada, sin reemplazar los resultados del modelo completo. La
    particion se hereda de ``official_split`` y se valida por imagen, de modo
    que participantes que observaron la misma interfaz no quedan repartidos
    entre Train y Test.

    Args:
        data: Tabla con features, targets, identificadores y ``official_split``.
        feature_columns: Features comunes a todos los targets.
        target_columns: Targets continuos para entrenar; se ajusta un modelo
            independiente para cada uno.
        output_dir: Carpeta raiz de las predicciones, metricas y modelos.
        time_limit: Presupuesto AutoGluon en segundos para cada target.
        feature_columns_by_target: Mapa opcional que reemplaza el conjunto de
            features comun para objetivos especificos.
        run_shap_top3: Si True, selecciona tres features con SHAP de Train y
            vuelve a evaluar el target en un directorio separado.

    Returns:
        Tupla ``(predictions_table, metrics)``. La primera tabla contiene
        identificadores, targets observados del Test y columnas
        ``prediction_<target>``; la segunda contiene una fila de metricas por
        target.

    Raises:
        ValueError: Si faltan columnas/features, no hay filas Train/Test para
            un target, el target es constante en Train o AutoGluon no puede
            recibir un conjunto de features valido.
        AssertionError: Si una misma imagen aparece en ambos splits.
    """
    from autogluon.tabular import TabularPredictor

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    required_columns = set(feature_columns) | set(target_columns) | {
        "official_split",
        "image_name",
        "participant",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas necesarias para entrenar: {sorted(missing_columns)}")

    # La particion oficial esta definida por interfaz. Comprobar la imagen,
    # no solo la fila, evita fuga cuando hay varias personas por interfaz.
    train_images = set(data.loc[data["official_split"] == "train", "image_name"])
    test_images = set(data.loc[data["official_split"] == "test", "image_name"])
    if train_images.intersection(test_images):
        raise AssertionError("Hay imagenes compartidas entre Train y Test.")

    # La tabla conjunta solo contiene filas Test: es el artefacto comun para
    # comparar targets reales con predicciones sin exportar filas Train.
    predictions_table = data.loc[
        data["official_split"] == "test",
        ["image_name", "participant", "official_split"] + list(target_columns),
    ].copy()
    metric_rows = []
    model_comparisons = []
    top3_metric_frames = []
    top3_skipped_targets = []

    for target in target_columns:
        target_dir = output_dir / target
        target_features = (
            feature_columns_by_target[target]
            if feature_columns_by_target is not None
            else feature_columns
        )
        missing_features = set(target_features).difference(data.columns)
        if missing_features:
            raise ValueError(
                f"Faltan features para {target}: {sorted(missing_features)}"
            )
        if not target_features:
            raise ValueError(f"No hay features configuradas para el target {target}.")

        # Una etiqueta ausente/no numerica invalida esa fila para este target,
        # pero no elimina la sesion de los modelos de otros targets.
        values = pd.to_numeric(data[target], errors="coerce")
        valid_target = values.notna() & np.isfinite(values)
        model_data = data.loc[
            valid_target,
            list(target_features) + [target, "official_split"],
        ].copy()
        model_data[target] = values.loc[model_data.index]
        train_data = model_data[model_data["official_split"] == "train"]
        test_data = model_data[model_data["official_split"] == "test"]
        if train_data.empty or test_data.empty:
            raise ValueError(
                f"El target {target} necesita ejemplos Train y Test; "
                f"encontrados {len(train_data)} y {len(test_data)}."
            )
        if train_data[target].nunique() < 2:
            raise ValueError(f"El target {target} es constante en Train.")

        target_dir.mkdir(parents=True, exist_ok=True)
        # El ajuste recibe exclusivamente Train y las features del target;
        # Test se mantiene fuera hasta la prediccion y evaluacion posteriores.
        with tempfile.TemporaryDirectory() as temp_model_dir:
            predictor = TabularPredictor(
                label=target,
                problem_type="regression",
                path=temp_model_dir,
                verbosity=0,
            ).fit(
                train_data[list(target_features) + [target]],
                time_limit=time_limit,
                presets="medium_quality",
            )
            test_features = test_data[list(target_features)]
            predicted = np.asarray(
                predictor.predict(test_features),
                dtype=float,
            )
            actual = test_data[target].to_numpy(dtype=float)
            predictions_table.loc[test_data.index, f"prediction_{target}"] = predicted
            # Se guardan tanto las metricas nativas de AutoGluon como las calculadas
            # explicitamente para mantener un formato estable entre ejecuciones.
            metrics = predictor.evaluate(
                test_data[list(target_features) + [target]],
                display=False,
            )
            metrics.update(
                metricas_clasificacion_por_terciles(
                    train_target=train_data[target],
                    actual_target=test_data[target],
                    predictions=predicted,
                )
            )
            metrics_row = {
                "target": target,
                "train_rows": len(train_data),
                "test_rows": len(test_data),
                "feature_count": len(target_features),
                "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                "mae": float(mean_absolute_error(actual, predicted)),
                "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
                **metrics,
            }
            target_predictions = data.loc[
                test_data.index,
                ["image_name", "participant", target],
            ].copy()
            target_predictions["prediction"] = predicted
            target_predictions.to_csv(
                target_dir / "predicciones_test.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame([metrics_row]).to_csv(
                target_dir / "metricas_test.csv",
                index=False,
                encoding="utf-8-sig",
            )
            # El leaderboard resume el rendimiento interno de los modelos
            # AutoGluon sobre este conjunto de evaluacion reservado.
            predictor.leaderboard(
                test_data[list(target_features) + [target]],
                silent=True,
            ).to_csv(
                target_dir / "leaderboard_test.csv",
                index=False,
                encoding="utf-8-sig",
            )
            comparison = evaluar_modelos_autogluon(
                predictor=predictor,
                train_target=train_data[target],
                test_features=test_features,
                test_target=test_data[target],
                context={"target": target},
            )
            comparison.to_csv(
                target_dir / "comparacion_modelos.csv",
                index=False,
                encoding="utf-8-sig",
            )
            guardar_explicacion_shap(
                predictor=predictor,
                train_features=train_data[list(target_features)],
                test_features=test_features,
                output_dir=target_dir / "shap",
                context={"target": target},
            )
            if run_shap_top3:
                # La importancia se calcula sobre Train; Test queda reservado para
                # evaluar el modelo reducido, sin intervenir en la seleccion.
                train_importances = guardar_explicacion_shap(
                    predictor=predictor,
                    train_features=train_data[list(target_features)],
                    test_features=train_data[list(target_features)],
                    output_dir=target_dir / "shap_seleccion_train",
                    context={"target": target},
                )

                predictor.delete_models(models_to_keep=[], dry_run=False)

        if run_shap_top3:
            if len(train_importances) < 3:
                reason = (
                    f"Solo hay {len(train_importances)} features disponibles "
                    "tras aplicar las exclusiones configuradas para este target."
                )
                print(f"Aviso: se omite SHAP top 3 para {target}. {reason}")
                top3_skipped_targets.append(
                    {
                        "target": target,
                        "available_features": len(train_importances),
                        "reason": reason,
                    }
                )
                continue

            # El ranking se obtuvo en Train; este conjunto reducido se evalua
            # despues con el mismo Test sin usarlo para escoger variables.
            top_features = train_importances["feature"].head(3).tolist()
            top3_root = output_dir / "top_3_features_shap"
            top3_target_dir = top3_root / target
            top3_target_dir.mkdir(parents=True, exist_ok=True)
            train_importances.head(3).to_csv(
                top3_target_dir / "features_seleccionadas_shap.csv",
                index=False,
                encoding="utf-8-sig",
            )
            _, top3_metrics = entrenar_predecir_targets(
                data=data,
                feature_columns=top_features,
                target_columns=[target],
                output_dir=top3_root,
                time_limit=time_limit,
            )
            top3_metric_frames.append(top3_metrics)

        metric_rows.append(metrics_row)
        model_comparisons.append(comparison)

    predictions_table.to_csv(
        output_dir / "predicciones_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metricas_test.csv", index=False, encoding="utf-8-sig")
    pd.concat(model_comparisons, ignore_index=True).to_csv(
        output_dir / "comparacion_modelos.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if run_shap_top3:
        if top3_metric_frames:
            top3_summary = pd.concat(top3_metric_frames, ignore_index=True)
        else:
            top3_summary = pd.DataFrame(
                columns=["target", "train_rows", "test_rows", "feature_count"]
            )
        top3_summary.to_csv(
            output_dir / "resumen_metricas_test_top_3_shap.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if top3_skipped_targets:
            pd.DataFrame(top3_skipped_targets).to_csv(
                output_dir / "targets_omitidos_top_3_shap.csv",
                index=False,
                encoding="utf-8-sig",
            )
    return predictions_table, metrics


def entrenar_predecir_targets_multimodal(
    data,
    feature_columns,
    target_columns,
    output_dir,
    time_limit=30,
    run_shap_top3=False,
):
    """Entrena un MultiModalPredictor por target con imagen y features visuales.

    El helper conserva la particion oficial por imagen. Solo admite como
    entradas ``image_path`` y columnas ``visual_*`` para impedir que los
    identificadores, las fijaciones o los targets se filtren al modelo.
    """
    from autogluon.multimodal import MultiModalPredictor

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = data.copy().reset_index(drop=True)
    required_columns = set(feature_columns) | set(target_columns) | {
        "official_split",
        "image_name",
        "participant",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Faltan columnas necesarias para entrenamiento multimodal: "
            f"{sorted(missing_columns)}"
        )
    if "image_path" not in feature_columns:
        raise ValueError("feature_columns debe incluir image_path.")
    invalid_features = {
        feature
        for feature in feature_columns
        if feature != "image_path" and not feature.startswith("visual_")
    }
    if invalid_features:
        raise ValueError(
            "El modelo multimodal solo admite image_path y features visuales: "
            f"{sorted(invalid_features)}"
        )
    if set(feature_columns).intersection(target_columns):
        raise ValueError("Un target no puede formar parte de feature_columns.")
    if data["official_split"].isna().any() or set(
        data["official_split"].unique()
    ) != {"train", "test"}:
        raise ValueError("official_split debe contener filas Train y Test solamente.")

    image_paths = data["image_path"].dropna().astype(str).unique()
    missing_images = [
        image_path
        for image_path in image_paths
        if not Path(image_path).is_absolute() or not Path(image_path).is_file()
    ]
    if data["image_path"].isna().any() or missing_images:
        invalid_path = missing_images[0] if missing_images else "<ruta vacia>"
        raise FileNotFoundError(
            f"image_path debe ser una ruta absoluta a un archivo existente: "
            f"{invalid_path}"
        )

    train_images = set(data.loc[data["official_split"] == "train", "image_name"])
    test_images = set(data.loc[data["official_split"] == "test", "image_name"])
    if train_images.intersection(test_images):
        raise AssertionError("Hay imagenes compartidas entre Train y Test.")

    prediction_columns = [
        "image_name",
        "participant",
        "official_split",
        *target_columns,
    ]
    predictions_table = data.loc[
        data["official_split"] == "test", prediction_columns
    ].copy()
    metric_rows = []
    comparison_rows = []

    if run_shap_top3:
        warnings.warn(
            "RUN_SHAP_TOP3 se omite para MultiModalPredictor; la seleccion SHAP "
            "del helper tabular no es compatible con image_path.",
            RuntimeWarning,
            stacklevel=2,
        )

    for target in target_columns:
        values = pd.to_numeric(data[target], errors="coerce")
        valid_target = values.notna() & np.isfinite(values)
        model_data = data.loc[
            valid_target,
            [*feature_columns, target, "official_split"],
        ].copy()
        model_data[target] = values.loc[model_data.index]
        train_data = model_data[model_data["official_split"] == "train"]
        test_data = model_data[model_data["official_split"] == "test"]
        if train_data.empty or test_data.empty:
            raise ValueError(
                f"El target {target} necesita ejemplos Train y Test; "
                f"encontrados {len(train_data)} y {len(test_data)}."
            )
        if train_data[target].nunique() < 2:
            raise ValueError(f"El target {target} es constante en Train.")

        target_dir = output_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)
        predictor = MultiModalPredictor(
            label=target,
            problem_type="regression",
            path=str(target_dir / "modelo_multimodal"),
            verbosity=0,
        )
        predictor.fit(
            train_data=train_data[[*feature_columns, target]],
            time_limit=time_limit,
            presets="medium_quality",
            column_types={
                feature: "image_path" if feature == "image_path" else "numerical"
                for feature in feature_columns
            },
        )
        predicted = np.asarray(
            predictor.predict(test_data[list(feature_columns)]),
            dtype=float,
        ).reshape(-1)
        actual = test_data[target].to_numpy(dtype=float)
        if predicted.size != actual.size or not np.isfinite(predicted).all():
            raise ValueError(
                f"MultiModalPredictor produjo predicciones invalidas para {target}."
            )

        predictions_table.loc[test_data.index, f"prediction_{target}"] = predicted
        metrics_row = {
            "target": target,
            "model": "MultiModalPredictor",
            "train_rows": len(train_data),
            "test_rows": len(test_data),
            "feature_count": len(feature_columns),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "mae": float(mean_absolute_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
            **metricas_clasificacion_por_terciles(
                train_target=train_data[target],
                actual_target=actual,
                predictions=predicted,
            ),
        }
        metric_rows.append(metrics_row)
        comparison_rows.append(metrics_row.copy())

        target_predictions = data.loc[
            test_data.index,
            ["image_name", "participant", target],
        ].copy()
        target_predictions["prediction"] = predicted
        target_predictions.to_csv(
            target_dir / "predicciones_test.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame([metrics_row]).to_csv(
            target_dir / "metricas_test.csv",
            index=False,
            encoding="utf-8-sig",
        )

    metrics = pd.DataFrame(metric_rows)
    predictions_table = predictions_table.reset_index(drop=True)
    predictions_table.to_csv(
        output_dir / "predicciones_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        output_dir / "metricas_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "comparacion_modelos.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return predictions_table, metrics


def agregar_proxies_observados_y_predichos(predictions, train_reference):
    """Anade proxies comparables para targets reales y predicciones de Test.

    Tanto la fila observada como la predicha se normalizan contra los rangos
    del mismo conjunto Train. Para el segundo calculo, cada columna objetivo
    se sustituye por ``prediction_<target>`` antes de derivar los indicadores.

    Args:
        predictions: Salida de ``entrenar_predecir_targets`` para filas Test.
        train_reference: Targets observados de Train que fijan los rangos
            min/max de normalizacion.

    Returns:
        Copia con los proxies observados prefijados por ``actual_`` y los
        calculados con las predicciones prefijados por ``predicted_``.

    Raises:
        ValueError: Si falta alguna columna de prediccion necesaria.
    """
    result = calcular_indicadores_proxy(
        predictions,
        normalization_reference=train_reference,
    )
    for column in (
        "proxy_exploration_efficiency",
        "proxy_attention_distribution",
        "proxy_exploration_strategy",
        "proxy_exploration_strategy_label",
    ):
        result[f"actual_{column}"] = result[column]

    predicted_targets = predictions.copy()
    for target in PROXY_TARGET_COLUMNS:
        prediction_column = f"prediction_{target}"
        if prediction_column not in predicted_targets:
            raise ValueError(f"Falta la prediccion requerida: {prediction_column}")
        predicted_targets[target] = predicted_targets[prediction_column]
    predicted_scores = calcular_indicadores_proxy(
        predicted_targets,
        normalization_reference=train_reference,
    )
    for column in (
        "proxy_exploration_efficiency",
        "proxy_attention_distribution",
        "proxy_exploration_strategy",
        "proxy_exploration_strategy_label",
    ):
        result[f"predicted_{column}"] = predicted_scores[column]
    return result


def metricas_clasificacion_por_terciles(train_target, actual_target, predictions):
    """Calcula accuracy y F1 tras discretizar target y prediccion en terciles.

    Los puntos de corte se obtienen solo de los valores finitos de Train y se
    aplican sin reajuste a valores observados y predichos de Test. Si hay
    muchos empates, ``np.unique`` puede producir menos de dos cortes y, por
    tanto, menos de tres clases efectivas.

    Args:
        train_target: Valores objetivo de entrenamiento para calcular cortes.
        actual_target: Valores reales del conjunto de prueba.
        predictions: Predicciones continuas correspondientes a Test.

    Returns:
        Diccionario con accuracy, F1 ponderado, numero de clases y cortes
        serializados para registrar como se obtuvo la clasificacion.

    Raises:
        ValueError: Si Train no contiene valores validos o si las entradas de
            Test tienen longitudes distintas, estan vacias o no son finitas.
    """
    train_values = pd.to_numeric(
        pd.Series(train_target),
        errors="coerce",
    ).to_numpy(dtype=float)
    train_values = train_values[np.isfinite(train_values)]
    actual = pd.to_numeric(
        pd.Series(actual_target),
        errors="coerce",
    ).to_numpy(dtype=float)
    predicted = pd.to_numeric(
        pd.Series(predictions),
        errors="coerce",
    ).to_numpy(dtype=float)

    if train_values.size == 0:
        raise ValueError("No hay valores válidos del target Train para calcular terciles.")
    if actual.size == 0 or actual.size != predicted.size:
        raise ValueError("Los targets y predicciones de Test deben tener igual longitud no nula.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Accuracy y F1 requieren targets y predicciones finitos.")

    cutoffs = np.unique(
        np.quantile(train_values, [1 / 3, 2 / 3])
    )
    actual_classes = np.searchsorted(cutoffs, actual, side="right")
    predicted_classes = np.searchsorted(cutoffs, predicted, side="right")
    class_labels = np.arange(len(cutoffs) + 1)
    return {
        "accuracy": float(accuracy_score(actual_classes, predicted_classes)),
        "f1_weighted": float(
            f1_score(
                actual_classes,
                predicted_classes,
                labels=class_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_class_count": len(class_labels),
        "classification_tertile_cutoffs": "|".join(
            f"{cutoff:.12g}" for cutoff in cutoffs
        ),
    }


def evaluar_modelos_autogluon(
    predictor,
    train_target,
    test_features,
    test_target,
    context=None,
):
    """Compara todos los modelos AutoGluon sobre el mismo Test reservado.

    Para cada modelo del leaderboard genera predicciones y calcula RMSE, MAE,
    R2 y metricas de terciles. La tabla final se ordena de menor a mayor RMSE;
    ``context`` permite adjuntar identificadores como el nombre del target.

    Args:
        predictor: Predictor ya ajustado con AutoGluon.
        train_target: Valores Train necesarios para calcular los cortes de
            terciles sin consultar el Test.
        test_features: Features del conjunto de evaluacion.
        test_target: Valores observados del target en Test.
        context: Campos constantes opcionales que se copian a cada fila.

    Returns:
        DataFrame con una fila por modelo y su posicion ordenada por RMSE.

    Raises:
        ValueError: Si el leaderboard esta vacio o un modelo devuelve un
            numero de predicciones distinto al de filas de Test.
    """
    leaderboard = predictor.leaderboard(silent=True)
    if leaderboard.empty or "model" not in leaderboard:
        raise ValueError("AutoGluon no devolvió modelos para comparar.")

    actual = np.asarray(test_target, dtype=float)
    rows = []
    for model_name in leaderboard["model"]:
        predictions = np.asarray(
            predictor.predict(test_features, model=model_name),
            dtype=float,
        )
        if predictions.shape[0] != actual.shape[0]:
            raise ValueError(
                f"El modelo {model_name} generó {len(predictions)} predicciones "
                f"para {len(actual)} filas de Test."
            )

        row = {
            "model": model_name,
            "is_predictor_best_model": model_name == predictor.model_best,
            "test_rows": len(actual),
            "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
            "mae": float(mean_absolute_error(actual, predictions)),
            "r2": float(r2_score(actual, predictions)) if len(actual) > 1 else np.nan,
            **metricas_clasificacion_por_terciles(
                train_target=train_target,
                actual_target=actual,
                predictions=predictions,
            ),
        }
        if context:
            row.update(context)
        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values(
        ["rmse", "mae"],
        ascending=True,
        kind="mergesort",
    )
    comparison.insert(
        0,
        "test_rank_by_rmse",
        np.arange(1, len(comparison) + 1),
    )
    return comparison.reset_index(drop=True)


def guardar_explicacion_shap(
    predictor,
    train_features,
    test_features,
    output_dir,
    context=None,
):
    """Guarda importancias y graficos SHAP para el predictor seleccionado.

    Para acotar el coste se toman muestras deterministas de Train y Test. Las
    variables categoricas se codifican numericamente para el explainer y se
    decodifican antes de llamar a AutoGluon, de forma que el modelo recibe los
    tipos de valores originales. La interpretacion de variables categoricas
    sigue siendo aproximada: SHAP permuta sus codigos numericos.

    Args:
        predictor: Predictor AutoGluon ya entrenado.
        train_features: Features de entrenamiento para el fondo/background.
        test_features: Features de evaluacion de las que se explican ejemplos.
        output_dir: Carpeta para CSV de importancia y graficos PNG.
        context: Metadatos opcionales que se agregan a la tabla de importancia.

    Returns:
        DataFrame ordenado por media de valor absoluto SHAP, con importancia
        absoluta media y contribucion SHAP media para cada feature.

    Raises:
        RuntimeError: Si no estan instalados SHAP o Matplotlib.
        ValueError: Si no hay features o alguna de las tablas esta vacia.
    """
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError as error:
        raise RuntimeError(
            "Para generar explicaciones SHAP instala las dependencias indicadas "
            "en Codigo/requirements-shap.txt."
        ) from error

    feature_names = list(test_features.columns)
    if not feature_names:
        raise ValueError("No hay atributos para explicar con SHAP.")
    if train_features.empty or test_features.empty:
        raise ValueError("SHAP necesita filas de entrenamiento y de Test.")

    # random_state mantiene las explicaciones reproducibles y los limites
    # globales fijan el coste aproximado de la explicacion por permutacion.
    background = train_features.sample(
        n=min(SHAP_MAX_BACKGROUND_ROWS, len(train_features)),
        random_state=42,
    )
    examples = test_features.sample(
        n=min(SHAP_MAX_EXPLANATION_ROWS, len(test_features)),
        random_state=42,
    )
    encoded_background = pd.DataFrame(index=background.index)
    encoded_examples = pd.DataFrame(index=examples.index)
    categorical_decoders = {}
    # SHAP trabaja con matrices numericas. Para categorias se crea una tabla
    # de correspondencia estable que permite reconstruir el valor antes de
    # consultar el predictor; las columnas numericas solo se convierten.
    for feature in feature_names:
        if pd.api.types.is_numeric_dtype(train_features[feature]):
            encoded_background[feature] = pd.to_numeric(
                background[feature],
                errors="coerce",
            ).to_numpy(dtype=float)
            encoded_examples[feature] = pd.to_numeric(
                examples[feature],
                errors="coerce",
            ).to_numpy(dtype=float)
            continue

        category_values = pd.concat(
            [train_features[feature], test_features[feature]],
            ignore_index=True,
        ).dropna()
        category_lookup = {}
        for value in category_values:
            category_lookup.setdefault(str(value), value)
        category_codes = pd.Series(
            {
                category: float(code)
                for code, category in enumerate(category_lookup)
            }
        )
        categorical_decoders[feature] = {
            code: value
            for code, value in enumerate(category_lookup.values())
        }
        for encoded_frame, source_frame in (
            (encoded_background, background),
            (encoded_examples, examples),
        ):
            encoded_frame[feature] = source_frame[feature].astype("string").map(
                category_codes
            ).to_numpy(dtype=float)

    def predecir(filas):
        """Adapta la matriz numerica de SHAP a los tipos que espera el modelo."""
        frame = pd.DataFrame(filas, columns=feature_names)
        for feature, decoder in categorical_decoders.items():
            frame[feature] = frame[feature].map(
                lambda code: decoder.get(int(round(code)), np.nan)
                if pd.notna(code)
                else np.nan
            )
        return np.asarray(predictor.predict(frame), dtype=float)

    # El explainer de permutacion estima la contribucion de cada feature
    # comparando predicciones con combinaciones de valores alteradas.
    explainer = shap.Explainer(
        predecir,
        encoded_background.to_numpy(dtype=float),
        algorithm="permutation",
        feature_names=feature_names,
    )
    shap_values = explainer(
        encoded_examples.to_numpy(dtype=float),
        max_evals=2 * len(feature_names) + 1,
    )
    values = np.asarray(shap_values.values, dtype=float)
    base_values = np.asarray(shap_values.base_values, dtype=float)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # El valor absoluto refleja magnitud media de influencia; la media con
    # signo conserva si la contribucion desplaza generalmente la prediccion.
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": np.mean(np.abs(values), axis=0),
            "mean_shap": np.mean(values, axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    if context:
        for column, value in context.items():
            importances.insert(0, column, value)
    importances.to_csv(
        output_dir / "shap_importancia_atributos.csv",
        index=False,
        encoding="utf-8-sig",
    )

    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=encoded_examples.to_numpy(dtype=float),
        feature_names=feature_names,
    )
    shap.plots.bar(
        explanation,
        max_display=min(SHAP_MAX_DISPLAY_FEATURES, len(feature_names)),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importancia_global.png", dpi=160, bbox_inches="tight")
    plt.close()

    shap.plots.beeswarm(
        explanation,
        max_display=min(SHAP_MAX_DISPLAY_FEATURES, len(feature_names)),
        color="coolwarm",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_resumen.png", dpi=160, bbox_inches="tight")
    plt.close()
    return importances
