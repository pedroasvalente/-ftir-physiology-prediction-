import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from tqdm import tqdm

from ftir_pred.config import RANDOM_SEED, RESULTS_DIR, R2_THRESHOLD, TRAINING_DATA_PATH, init_mlflow
from ftir_pred.data.config import REGRESSION_TARGETS, get_target_group
from ftir_pred.data.loader import filter_samples, get_ftir_columns, load_csv
from ftir_pred.data.splits import make_cv_splitter, split_by_person
from ftir_pred.models.baselines import BASELINE_MODELS
from ftir_pred.models.configs import MODEL_CONFIGS
from ftir_pred.models.evaluation import evaluate
from ftir_pred.preprocessing.pipeline import (
    get_pls_vip,
    make_pipeline,
    prefix_params,
)


def _run_search(pipe, X_train, y_train, groups_train, params, search_type: str):
    cv = make_cv_splitter(n_splits=5)
    search_kwargs = dict(
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    if search_type == "grid":
        search = GridSearchCV(pipe, params, cv=cv, **search_kwargs)
    else:
        search = BayesSearchCV(pipe, params, cv=cv, n_iter=50, n_points=10, **search_kwargs)
    search.fit(X_train, y_train, groups=groups_train)
    return search


def _run_baselines(X_train, X_test, y_train, y_test, target, sample_type, timepoints):
    rows = []
    for key, (display_name, model) in BASELINE_MODELS.items():
        try:
            m = model.__class__(**model.get_params())
            if hasattr(m, "random_state"):
                m.set_params(random_state=RANDOM_SEED)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            if y_pred.ndim > 1:
                y_pred = y_pred.ravel()
            metrics = evaluate(y_test, y_pred)
        except Exception as exc:
            logger.warning(f"Baseline {display_name} failed for {target}/{sample_type}: {exc}")
            continue
        rows.append({
            "target": target,
            "sample_type": sample_type,
            "timepoints": _tp_str(timepoints),
            "model": display_name,
            "search": "none",
            "scale": True,
            "apply_pls": False,
            "n_components": None,
            "is_baseline": True,
            "target_group": get_target_group(target),
            **metrics,
        })
    return rows


def _nc_str(nc) -> str:
    """Normalise n_components to a string for checkpoint keying (handles None and NaN)."""
    if nc is None:
        return "None"
    try:
        if np.isnan(float(nc)):
            return "None"
        return str(int(float(nc)))
    except (TypeError, ValueError):
        return str(nc)


def _tp_str(tp) -> str:
    """Normalise a timepoints value for checkpoint keying.

    pandas treats the string 'None' as NaN when reading CSVs, so we need to
    convert NaN back to the canonical "None" string used during training.
    """
    if tp is None:
        return "None"
    try:
        if pd.isna(tp):
            return "None"
    except (TypeError, ValueError):
        pass
    return str(tp)


def run_experiment(config_path: str) -> None:
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = json.load(f)

    init_mlflow()

    targets = cfg.get("targets_to_predict", REGRESSION_TARGETS)
    sample_types = cfg.get("sample_types", ["SERUM"])
    timepoints_list = cfg.get("timepoints", [None])
    model_names = cfg.get("model_types_to_train", ["random_forest"])
    search_types = cfg.get("searchs_hipermetrics", ["grid"])
    scale_options = cfg.get("scale", [True])
    pls_options = cfg.get("apply_pls", [True])
    n_components_list = cfg.get("n_components", [10])
    experiment_name = cfg.get("experiment_name", "FTIR Physiology Prediction")
    run_slug = cfg.get("run_name", config_path.stem)

    df = load_csv(cfg.get("data_path") or str(TRAINING_DATA_PATH))
    wavenumbers = np.array([float(c) for c in get_ftir_columns(df).tolist()])
    water_mask = (wavenumbers < 1850) | (wavenumbers > 2500)
    valid_wavenumbers = wavenumbers[water_mask]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS_DIR / run_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Checkpoint loading ---
    checkpoint_csv = out_dir / "results_summary.csv"
    pred_path = out_dir / "predictions_data.json"
    imp_path = out_dir / "wavenumber_importance.json"

    if checkpoint_csv.exists():
        existing_df = pd.read_csv(checkpoint_csv)
        all_results: list[dict] = existing_df.to_dict("records")
        done_combos: set = set()
        done_split_baselines: set = set()
        for row in all_results:
            tp = _tp_str(row.get("timepoints"))
            if row.get("is_baseline"):
                done_split_baselines.add((row["target"], row["sample_type"], tp))
            else:
                done_combos.add((
                    row["target"], row["sample_type"], tp,
                    row["model"], row["search"],
                    bool(row["scale"]), bool(row["apply_pls"]),
                    _nc_str(row.get("n_components")),
                ))
        logger.info(
            f"Checkpoint found — resuming. "
            f"{len(done_combos)} ML combos + {len(done_split_baselines)} baseline sets already done."
        )
    else:
        all_results = []
        done_combos = set()
        done_split_baselines = set()

    predictions_store: dict = json.loads(pred_path.read_text()) if pred_path.exists() else {}
    importances_store: dict = json.loads(imp_path.read_text()) if imp_path.exists() else {}

    def _save_checkpoint() -> None:
        pd.DataFrame(all_results).to_csv(checkpoint_csv, index=False)
        with open(pred_path, "w") as fh:
            json.dump(predictions_store, fh)
        with open(imp_path, "w") as fh:
            json.dump(importances_store, fh)

    # --- Build combo list ---
    combo_iter = [
        (target, sample_type, timepoints, model_name, scale, apply_pls, n_comp, search_type)
        for target in targets
        for sample_type in sample_types
        for timepoints in timepoints_list
        for model_name in model_names
        for scale in scale_options
        for apply_pls in pls_options
        for n_comp in (n_components_list if apply_pls else [None])
        for search_type in search_types
        if not (not scale and not apply_pls)
    ]

    seen: set = set()
    unique_combos = []
    for c in combo_iter:
        key = (c[0], c[1], _tp_str(c[2]), c[3], c[4], c[5], _nc_str(c[6]), c[7])
        if key not in seen:
            seen.add(key)
            unique_combos.append(c)

    split_cache: dict = {}

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_slug) as main_run:
        mlflow.log_artifact(str(config_path))
        mlflow.set_tag("run_slug", run_slug)

        with tqdm(unique_combos, desc="Training") as bar:
            for (target, sample_type, timepoints, model_name, scale, apply_pls, n_comp, search_type) in bar:
                bar.set_postfix(target=target, matrix=sample_type, model=model_name, search=search_type)

                split_key = (target, sample_type, _tp_str(timepoints))

                # Populate split cache if needed
                if split_key not in split_cache:
                    result = filter_samples(df, sample_type, target, timepoints)
                    if result is None:
                        split_cache[split_key] = None
                    else:
                        X, y, groups = result
                        X_arr = X.values.astype(float)
                        y_arr = y.values.astype(float)
                        g_arr = groups.values

                        X_train, X_test, y_train, y_test, groups_train = split_by_person(
                            X_arr, y_arr, g_arr, test_size=0.2
                        )

                        if len(y_test) < 5:
                            split_cache[split_key] = None
                        else:
                            if split_key in done_split_baselines:
                                split_cache[split_key] = (X_train, X_test, y_train, y_test, groups_train, [])
                            else:
                                baselines = _run_baselines(
                                    X_train, X_test, y_train, y_test, target, sample_type, timepoints
                                )
                                split_cache[split_key] = (X_train, X_test, y_train, y_test, groups_train, baselines)
                                all_results.extend(baselines)
                                done_split_baselines.add(split_key)
                                _save_checkpoint()

                cached = split_cache[split_key]
                if cached is None:
                    continue

                X_train, X_test, y_train, y_test, groups_train, _ = cached

                model_cfg = MODEL_CONFIGS[model_name]

                # Skip if already done in a previous run
                combo_done_key = (
                    target, sample_type, _tp_str(timepoints),
                    model_cfg.display_name, search_type,
                    scale, apply_pls,
                    _nc_str(n_comp),
                )
                if combo_done_key in done_combos:
                    continue

                try:
                    pipe = make_pipeline(model_cfg.get_model(), scale=scale, apply_pls=apply_pls,
                                         n_components=n_comp if apply_pls else 10)
                except ValueError as exc:
                    logger.warning(str(exc))
                    continue

                params = prefix_params(
                    model_cfg.grid_params if search_type == "grid" else model_cfg.bayes_params
                )

                with mlflow.start_run(run_name=f"{target}_{sample_type}_{model_name}_{search_type}",
                                      nested=True, parent_run_id=main_run.info.run_id):
                    try:
                        search = _run_search(pipe, X_train, y_train, groups_train, params, search_type)
                    except Exception as exc:
                        logger.error(f"Search failed for {target}/{sample_type}/{model_name}/{search_type}: {exc}")
                        continue

                    y_pred = search.predict(X_test)
                    if hasattr(y_pred, "ravel"):
                        y_pred = y_pred.ravel()

                    metrics = evaluate(y_test, y_pred)
                    best_pipe = search.best_estimator_

                    vip = get_pls_vip(best_pipe)
                    wn_imp_valid = vip[water_mask] if vip is not None else np.zeros(len(valid_wavenumbers))

                    mlflow.set_tag("target", target)
                    mlflow.set_tag("sample_type", sample_type)
                    mlflow.set_tag("model", model_cfg.display_name)
                    mlflow.set_tag("search", search_type)
                    mlflow.log_param("scale", scale)
                    mlflow.log_param("apply_pls", apply_pls)
                    mlflow.log_param("n_components", n_comp)
                    mlflow.log_param("n_train", len(y_train))
                    mlflow.log_param("n_test", len(y_test))
                    mlflow.log_param("timepoints", str(timepoints))
                    for k, v in metrics.items():
                        if isinstance(v, float):
                            mlflow.log_metric(k, v)
                    try:
                        for k, v in search.best_params_.items():
                            mlflow.log_param(k.replace("model__", ""), v)
                    except Exception:
                        pass

                    store_key = f"{target}|{sample_type}|{model_cfg.display_name}|{search_type}"
                    if metrics["r2"] >= R2_THRESHOLD:
                        existing_r2 = predictions_store.get(store_key, {}).get("r2", -np.inf)
                        if metrics["r2"] >= existing_r2:
                            predictions_store[store_key] = {
                                "y_test": y_test.tolist(),
                                "y_pred": y_pred.tolist(),
                                "target": target,
                                "sample_type": sample_type,
                                "model": model_cfg.display_name,
                                "search": search_type,
                                "r2": metrics["r2"],
                            }
                            if vip is not None:
                                importances_store[store_key] = {
                                    "wavenumbers": valid_wavenumbers.tolist(),
                                    "importances": wn_imp_valid.tolist(),
                                    "target": target,
                                    "sample_type": sample_type,
                                    "model": model_cfg.display_name,
                                    "search": search_type,
                                    "r2": metrics["r2"],
                                }

                    all_results.append({
                        "target": target,
                        "target_group": get_target_group(target),
                        "sample_type": sample_type,
                        "timepoints": _tp_str(timepoints),
                        "model": model_cfg.display_name,
                        "search": search_type,
                        "scale": scale,
                        "apply_pls": apply_pls,
                        "n_components": n_comp,
                        "is_baseline": False,
                        **metrics,
                    })
                    done_combos.add(combo_done_key)
                    _save_checkpoint()

        logger.info(
            f"Run complete. {len(all_results)} total rows "
            f"({len(predictions_store)} predictions, {len(importances_store)} importances) → {out_dir}"
        )
