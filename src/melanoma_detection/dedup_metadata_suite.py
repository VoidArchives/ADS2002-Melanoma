"""Remaining clean metadata comparisons, with frozen patient memberships.

The historical notebooks are protocol references only. This module never opens
reserved-test data, rewrites existing results, or fits an all-development model.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits
import xgboost
from xgboost import XGBClassifier
import yaml

from .dedup_metadata import NN_SPECIFICATION, _verify_predictions, verify_saved_receipt
from .development_dataset import CANONICAL_CSV, load_development_dataset, stable_sha256
from .metadata_evaluation import decision_metrics, probability_metrics, validate_probabilities
from .metadata_model import SPECIFICATION as SPLINE_SPECIFICATION, file_sha256, numeric_age
from .metadata_validation import validate_metadata_frame

FEATURES = ["sex", "age_approx", "anatom_site_general_challenge"]
KEYS = ["image_name", "patient_id", "fold", "target"]
OUTPUT_PREFIX = "results/dedup_v1/metadata_suite"
ABLATIONS = {
    "sex": ["sex"], "age": ["age_approx"],
    "site": ["anatom_site_general_challenge"],
    "sex_age": ["sex", "age_approx"],
    "sex_site": ["sex", "anatom_site_general_challenge"],
    "age_site": ["age_approx", "anatom_site_general_challenge"],
}
GRIDS = {
    "spline": {"classifier__C": [0.1, 1.0, 10.0],
               "preprocessor__smooth_age__spline__n_knots": [3, 4, 5],
               "preprocessor__smooth_age__spline__degree": [2, 3]},
    "random_forest": {"classifier__max_depth": [4, 8, None],
                      "classifier__min_samples_leaf": [5, 15],
                      "classifier__max_features": ["sqrt", 1.0]},
    "xgboost": {"classifier__max_depth": [2, 3],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__min_child_weight": [1, 5]},
}
SOURCE_FILES = (
    "src/melanoma_detection/dedup_metadata_suite.py",
    "scripts/run_dedup_metadata_suite.py",
    "src/melanoma_detection/development_dataset.py",
    "src/melanoma_detection/dedup_metadata.py",
    "src/melanoma_detection/metadata_model.py",
    "src/melanoma_detection/metadata_evaluation.py",
    "src/melanoma_detection/metadata_validation.py",
    "notebooks/Subset/04_dummy_logistic_baseline.ipynb",
    "notebooks/Subset/05_metadata_models.ipynb",
    "notebooks/Subset/07_metadata_tuning.ipynb",
)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path, frame):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _path(root, value, prefix):
    value = Path(value)
    candidate = (root / value).resolve() if not value.is_absolute() else value.resolve()
    if not candidate.is_relative_to((root / prefix).resolve()):
        raise ValueError(f"Artifact path must remain within {prefix}.")
    return candidate


def _versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "joblib": joblib.__version__}


def _relative(root, path):
    return Path(path).resolve().relative_to(root).as_posix()


def categorical_frame(frame):
    """Match notebook 04/05/07 age-as-string encoding without changing source."""
    prepared = validate_metadata_frame(frame).copy()
    for column in FEATURES:
        prepared[column] = prepared[column].astype("string").fillna("Missing").astype(str)
    return prepared


def categorical_pipeline(classifier, features=FEATURES):
    return Pipeline([
        ("preprocessor", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(features)),
        ], remainder="drop")),
        ("classifier", classifier),
    ])


def spline_pipeline(features=FEATURES):
    """The exact smooth-age protocol and parameter names from notebook 07."""
    transformers = []
    if "age_approx" in features:
        transformers.append(("smooth_age", Pipeline([
            ("to_numeric", FunctionTransformer(numeric_age, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(n_knots=4, degree=3, knots="quantile", include_bias=False)),
            ("scaler", StandardScaler()),
        ]), ["age_approx"]))
    categories = [column for column in ("sex", "anatom_site_general_challenge") if column in features]
    if categories:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categories))
    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers, remainder="drop")),
        ("classifier", LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=42)),
    ])


def model_setups(threads=4):
    return {
        "spline": spline_pipeline(),
        "random_forest": categorical_pipeline(RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=threads,
        )),
        "xgboost": categorical_pipeline(XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=.05, min_child_weight=1,
            objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=threads,
        )),
    }


def reconstruct_inner_memberships(original, retained_ids):
    """Run the historical splitter on original rows, then apply retention only.

    Original file row order is required. Reconstructed memberships are expressly
    not represented as a recovered historical execution receipt.
    """
    original = validate_metadata_frame(original).reset_index(drop=True)
    if not set(retained_ids).issubset(set(original.image_name)):
        raise ValueError("Retained IDs must be a subset of original development IDs.")
    original_parts, retained_parts = [], []
    for outer_fold in range(5):
        pool = original.loc[original.fold.ne(outer_fold)].reset_index(drop=True)
        assigned = np.full(len(pool), -1, dtype=int)
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42 + outer_fold)
        for inner_fold, (_, validation) in enumerate(splitter.split(pool[FEATURES], pool.target, pool.patient_id)):
            assigned[validation] = inner_fold
        membership = pool[KEYS].copy()
        membership["outer_fold"] = outer_fold
        membership["inner_fold"] = assigned
        if (assigned < 0).any() or not membership.groupby("patient_id").inner_fold.nunique().eq(1).all():
            raise ValueError("Reconstruction failed to assign patients to exactly one inner fold.")
        original_parts.append(membership)
        retained_parts.append(membership.loc[membership.image_name.isin(retained_ids)].copy())
    return pd.concat(original_parts, ignore_index=True), pd.concat(retained_parts, ignore_index=True)


def inner_splits_from_memberships(frame, outer_fold, memberships):
    pool = frame.loc[frame.fold.ne(outer_fold)].reset_index(drop=True)
    rows = memberships.loc[memberships.outer_fold.eq(outer_fold)].copy()
    if not rows.image_name.is_unique or set(rows.image_name) != set(pool.image_name):
        raise ValueError("Frozen metadata inner membership coverage differs from the fitting pool.")
    ordered = rows.set_index("image_name").loc[pool.image_name].reset_index()
    for column in ("patient_id", "fold", "target"):
        if not np.array_equal(pool[column].to_numpy(), ordered[column].to_numpy()):
            raise ValueError("Frozen metadata inner membership records changed.")
    if set(ordered.inner_fold) != {0, 1, 2} or not ordered.groupby("patient_id").inner_fold.nunique().eq(1).all():
        raise ValueError("Inner folds must keep each patient in one of three groups.")
    splits, audits = [], []
    outer_patients = set(frame.loc[frame.fold.eq(outer_fold), "patient_id"])
    for inner_fold in range(3):
        validation = np.flatnonzero(ordered.inner_fold.eq(inner_fold).to_numpy())
        fitting = np.flatnonzero(ordered.inner_fold.ne(inner_fold).to_numpy())
        fitting_patients = set(pool.iloc[fitting].patient_id)
        validation_patients = set(pool.iloc[validation].patient_id)
        if not fitting_patients.isdisjoint(validation_patients) or not (fitting_patients | validation_patients).isdisjoint(outer_patients):
            raise ValueError("An inner fitting boundary contains overlapping or outer patients.")
        if any(pool.iloc[positions].target.nunique() != 2 for positions in (fitting, validation)):
            raise ValueError("Every inner fitting and validation partition requires both classes.")
        splits.append((fitting, validation))
        audits.append({"outer_fold": outer_fold, "inner_fold": inner_fold,
                       "fit_images": len(fitting), "validation_images": len(validation),
                       "fit_patients": len(fitting_patients), "validation_patients": len(validation_patients),
                       "fit_positive_images": int(pool.iloc[fitting].target.sum()),
                       "validation_positive_images": int(pool.iloc[validation].target.sum()),
                       "patient_overlap": 0, "outer_patient_overlap": 0})
    return splits, pd.DataFrame(audits)


def verify_reuse(root, frame, identity, receipt_path):
    path = _path(root, receipt_path, "results/dedup_v1/metadata")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("dataset_identity") != identity:
        raise ValueError("Reused metadata receipt has a different development dataset identity.")
    if receipt.get("specification", {}).get("neural_network") != NN_SPECIFICATION or receipt.get("specification", {}).get("spline") != SPLINE_SPECIFICATION:
        raise ValueError("Reused metadata comparator protocol differs from the prescribed protocol.")
    verify_saved_receipt(root, receipt)
    for relative, digest in receipt["source_sha256"].items():
        source = (root / relative).resolve()
        if not source.is_relative_to(root) or file_sha256(source) != digest:
            raise ValueError(f"Reused comparator source changed: {relative}")
    predictions = pd.read_csv(_path(root, receipt["prediction_path"], "results/dedup_v1/metadata"), float_precision="round_trip")
    calibration = pd.read_csv(_path(root, receipt["calibration_prediction_path"], "results/dedup_v1/metadata"), float_precision="round_trip")
    _verify_predictions(frame, predictions)
    _verify_predictions(frame, calibration)
    predictions = predictions.set_index("image_name").loc[frame.image_name].reset_index()
    calibration = calibration.set_index("image_name").loc[frame.image_name].reset_index()
    if not np.array_equal(predictions.p_spline.to_numpy(), calibration.p_spline_raw.to_numpy()):
        raise ValueError("Reused spline and calibration records disagree on raw scores.")
    return receipt, predictions, calibration


def run_outer_fold(frame, outer_fold, memberships, reused, *, threads=4, grids=None, setups=None):
    """Fit only outer-training patients; outer targets are output/evaluation only."""
    frame = categorical_frame(frame)
    grids = GRIDS if grids is None else grids
    setups = model_setups(threads) if setups is None else setups
    fitting = frame.loc[frame.fold.ne(outer_fold)].reset_index(drop=True)
    heldout = frame.loc[frame.fold.eq(outer_fold)].reset_index(drop=True)
    if not set(fitting.patient_id).isdisjoint(heldout.patient_id):
        raise ValueError("Outer fitting and validation patients overlap.")
    inner_splits, audit = inner_splits_from_memberships(frame, outer_fold, memberships)
    score = heldout[KEYS].copy()
    reused = reused.set_index("image_name").loc[heldout.image_name]
    for column in ("p_spline", "p_nn_seed_42", "p_nn_seed_142", "p_nn_seed_242", "p_nn_mean"):
        score[column] = reused[column].to_numpy()
    base_estimators = {
        "dummy": DummyClassifier(strategy="prior"),
        "categorical_logistic": categorical_pipeline(LogisticRegression(
            C=1.0, solver="liblinear", max_iter=1000, random_state=42,
        )),
        "random_forest": setups["random_forest"], "xgboost": setups["xgboost"],
    }
    models, fit_log, selected, candidates = {}, [], [], []
    for name, estimator in base_estimators.items():
        started = time.monotonic()
        model = clone(estimator).fit(fitting[FEATURES], fitting.target)
        score[f"p_{name}"] = model.predict_proba(heldout[FEATURES])[:, list(model.classes_).index(1)]
        models[name] = model
        fit_log.append({"outer_fold": outer_fold, "stage": "baseline", "model": name,
                        "fits": 1, "elapsed_seconds": time.monotonic() - started})
    for name, features in ABLATIONS.items():
        started = time.monotonic()
        model = spline_pipeline(features).fit(fitting[features], fitting.target)
        score[f"p_ablation_{name}"] = model.predict_proba(heldout[features])[:, 1]
        models[f"ablation_{name}"] = model
        fit_log.append({"outer_fold": outer_fold, "stage": "ablation", "model": name,
                        "fits": 1, "elapsed_seconds": time.monotonic() - started})
    for name, estimator in setups.items():
        started = time.monotonic()
        search = GridSearchCV(clone(estimator), grids[name], scoring="average_precision",
                              cv=inner_splits, refit=True, n_jobs=1, error_score="raise")
        search.fit(fitting[FEATURES], fitting.target)
        score[f"p_{name}_tuned"] = search.predict_proba(heldout[FEATURES])[:, 1]
        models[f"{name}_tuned"] = search.best_estimator_
        selected.append({"outer_fold": outer_fold, "model": name,
                         "inner_best_average_precision": float(search.best_score_),
                         "best_candidate_index": int(search.best_index_),
                         "best_parameters": json.dumps(search.best_params_, sort_keys=True)})
        results = search.cv_results_
        for index, parameters in enumerate(results["params"]):
            row = {"outer_fold": outer_fold, "model": name, "candidate_index": index,
                   "parameters": json.dumps(parameters, sort_keys=True),
                   "mean_inner_average_precision": float(results["mean_test_score"][index]),
                   "std_inner_average_precision": float(results["std_test_score"][index]),
                   "mean_fit_seconds": float(results["mean_fit_time"][index]),
                   "selected": index == search.best_index_}
            row.update({f"inner_{fold}_average_precision": float(results[f"split{fold}_test_score"][index]) for fold in range(3)})
            candidates.append(row)
        fit_log.append({"outer_fold": outer_fold, "stage": "nested", "model": name,
                        "fits": len(list(ParameterGrid(grids[name]))) * 3 + 1,
                        "elapsed_seconds": time.monotonic() - started})
    _verify_predictions(frame, score, fold=outer_fold)
    return {"predictions": score, "selected_parameters": pd.DataFrame(selected),
            "inner_candidates": pd.DataFrame(candidates), "inner_audit": audit,
            "fit_log": pd.DataFrame(fit_log),
            "models": {"estimators": models, "fit_image_ids": fitting.image_name.tolist(),
                       "fit_patient_ids": sorted(fitting.patient_id.unique()), "outer_fold": outer_fold,
                       "features": FEATURES, "selection_objective": "unweighted mean inner average precision"}}


def metrics_table(frame, predictions):
    rows = []
    for column in predictions.filter(like="p_"):
        folds = []
        for fold in range(5):
            mask = predictions.fold.eq(fold)
            result = probability_metrics(predictions.loc[mask, "target"], predictions.loc[mask, column])
            folds.append(result)
            rows.append({"model": column.removeprefix("p_"), "scope": "fold", "fold": fold, **result})
        rows.append({"model": column.removeprefix("p_"), "scope": "mean_fold", "fold": None,
                     **{key: float(np.mean([result[key] for result in folds])) for key in folds[0]}})
        rows.append({"model": column.removeprefix("p_"), "scope": "pooled", "fold": None,
                     **probability_metrics(predictions.target, predictions[column])})
    return pd.DataFrame(rows)


def verify_suite_receipt(root, receipt, *, compatibility_sha256=None):
    if receipt.get("status") != "complete":
        raise ValueError("Complete clean metadata suite receipt required.")
    if compatibility_sha256 is not None and receipt.get("compatibility_sha256") != compatibility_sha256:
        raise ValueError("Dataset, source, config, reuse or runtime changed; select a new run identity.")
    for relative, digest in receipt["output_sha256"].items():
        path = _path(root, relative, OUTPUT_PREFIX)
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Clean metadata suite output is missing or changed: {relative}")


def run_suite(project_root, config_path="configs/metadata_dedup_suite.yaml", *, progress=print):
    root = Path(project_root).resolve()
    config_path = (root / config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not config.get("dataset_manifest"):
        raise ValueError("An explicit frozen development manifest is required.")
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in run_id):
        raise ValueError("A simple non-empty run_id is required.")
    threads = config.get("threads", 4)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer.")
    frame, identity = load_development_dataset(root, config["dataset_manifest"])
    # The loader sorts IDs. Recover original notebook CSV ordering and age text
    # only from the already verified canonical development file.
    original = validate_metadata_frame(pd.read_csv(root / CANONICAL_CSV, dtype={"age_approx": "string"}))
    raw_by_id = original.set_index("image_name")
    frame["age_approx"] = raw_by_id.loc[frame.image_name, "age_approx"].to_numpy()
    frame = categorical_frame(frame)
    reuse_path = config.get("reuse_receipt", "results/dedup_v1/metadata/comparator_receipt.json")
    reuse_receipt, reused, calibration = verify_reuse(root, frame, identity, reuse_path)
    compatibility = {"dataset_identity": identity, "config_sha256": file_sha256(config_path),
                     "source_sha256": {relative: file_sha256(root / relative) for relative in SOURCE_FILES},
                     "versions": _versions(), "features": FEATURES, "grids": GRIDS,
                     "reused_receipt_sha256": file_sha256(_path(root, reuse_path, "results/dedup_v1/metadata")),
                     "selection": "unweighted mean inner AP; first candidate in sklearn ParameterGrid order on ties",
                     "inner_protocol": "reconstructed from original CSV row order; seed 42 + outer_fold; retained mask afterward",
                     "fit_resource_setting": {"estimator_threads": threads, "grid_search_jobs": 1}}
    compatibility_sha256 = stable_sha256(compatibility)
    output = _path(root, f"{OUTPUT_PREFIX}/{run_id}", OUTPUT_PREFIX)
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        verify_suite_receipt(root, receipt, compatibility_sha256=compatibility_sha256)
        if progress:
            progress("Verified complete clean metadata suite; reused outputs without fitting.")
        return receipt
    session_path = output / "session.json"
    if session_path.exists():
        if json.loads(session_path.read_text())["compatibility_sha256"] != compatibility_sha256:
            raise ValueError("Partial clean metadata run has incompatible provenance; choose a new run identity.")
    elif any(output.iterdir()):
        raise ValueError("Existing output directory lacks a session receipt; choose a new run identity.")
    else:
        _write_json(session_path, {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "compatibility_sha256": compatibility_sha256, **compatibility,
                                  "test_set_loaded": False, "final_model_fitted": False})
    started = time.monotonic()
    membership_path = output / "memberships_receipt.json"
    if membership_path.exists():
        membership_receipt = json.loads(membership_path.read_text())
        if membership_receipt["compatibility_sha256"] != compatibility_sha256:
            raise ValueError("Frozen metadata tuning reconstruction identity changed.")
        for relative, digest in membership_receipt["output_sha256"].items():
            if file_sha256(_path(root, relative, OUTPUT_PREFIX)) != digest:
                raise ValueError("Frozen metadata tuning membership artifact changed.")
        memberships = pd.read_csv(output / "inner_memberships_retained.csv")
    else:
        original_memberships, memberships = reconstruct_inner_memberships(original, set(frame.image_name))
        _write_csv(output / "inner_memberships_original.csv", original_memberships)
        _write_csv(output / "inner_memberships_retained.csv", memberships)
        patients = original_memberships[["outer_fold", "patient_id", "fold", "inner_fold"]].drop_duplicates()
        _write_csv(output / "inner_patient_memberships.csv", patients)
        files = ("inner_memberships_original.csv", "inner_memberships_retained.csv", "inner_patient_memberships.csv")
        _write_json(membership_path, {"compatibility_sha256": compatibility_sha256,
                    "status": "reconstructed_and_frozen", "historical_membership_receipt_available": False,
                    "reconstruction_source": CANONICAL_CSV, "versions": _versions(),
                    "original_row_order_used": True, "split_on_retained_rows": False,
                    "seed_rule": "42 + outer_fold", "output_sha256": {
                        _relative(root, output / name): file_sha256(output / name) for name in files}})
    parts, fold_receipts, invocation_fits = [], [], 0
    with threadpool_limits(limits=threads):
        for fold in range(5):
            folder = output / f"fold_{fold}"
            folder.mkdir(exist_ok=True)
            completion = folder / "receipt.json"
            if completion.exists():
                receipt = json.loads(completion.read_text())
                verify_suite_receipt(root, receipt, compatibility_sha256=compatibility_sha256)
                if progress:
                    progress(f"Verified clean metadata suite fold {fold}; no refitting.")
            else:
                if progress:
                    progress(f"Fitting clean metadata suite fold {fold}; 139 CPU fits.")
                fold_started = time.monotonic()
                result = run_outer_fold(frame, fold, memberships, reused, threads=threads)
                file_names = []
                for name in ("predictions", "selected_parameters", "inner_candidates", "inner_audit", "fit_log"):
                    name_csv = f"{name}.csv"
                    _write_csv(folder / name_csv, result[name])
                    file_names.append(name_csv)
                model_path = folder / "models.joblib"
                temporary = folder / "models.joblib.tmp"
                joblib.dump(result["models"], temporary)
                os.replace(temporary, model_path)
                file_names.append("models.joblib")
                fits = int(result["fit_log"].fits.sum())
                invocation_fits += fits
                receipt = {"status": "complete", "outer_fold": fold,
                           "compatibility_sha256": compatibility_sha256,
                           "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
                           "executed_fits": fits, "elapsed_seconds": time.monotonic() - fold_started,
                           "fit_patient_ids": result["models"]["fit_patient_ids"],
                           "output_sha256": {_relative(root, folder / name): file_sha256(folder / name) for name in file_names}}
                _write_json(completion, receipt)
            part = pd.read_csv(folder / "predictions.csv", float_precision="round_trip")
            _verify_predictions(frame, part, fold=fold)
            parts.append(part)
            fold_receipts.append(receipt)
    combined = pd.concat(parts, ignore_index=True).set_index("image_name").loc[frame.image_name].reset_index()
    _verify_predictions(frame, combined)
    combined["p_spline_calibrated"] = calibration.p_spline_calibrated.to_numpy()
    prediction_folder = output / "predictions"
    prediction_folder.mkdir(exist_ok=True)
    prediction_paths = {}
    for column in combined.filter(like="p_"):
        name = column.removeprefix("p_")
        name = {"spline": "spline_raw", "nn_mean": "metadata_nn_mean"}.get(name, name)
        model_predictions = combined[KEYS].copy()
        model_predictions["score"] = combined[column]
        path = prediction_folder / f"{name}.csv"
        _write_csv(path, model_predictions)
        prediction_paths[name] = _relative(root, path)
    # Explicit all-feature ablation alias; reuse exact fixed-spline scores.
    prediction_paths["ablation_all_features"] = prediction_paths["spline_raw"]
    _write_csv(output / "combined_oof.csv", combined)
    _write_csv(output / "baseline_oof.csv", combined[KEYS + ["p_dummy", "p_categorical_logistic", "p_spline", "p_random_forest", "p_xgboost"]])
    _write_csv(output / "ablation_oof.csv", combined[KEYS + [f"p_ablation_{name}" for name in ABLATIONS] + ["p_spline"]])
    _write_csv(output / "nested_oof.csv", combined[KEYS + [f"p_{name}{suffix}" for name in GRIDS for suffix in ("", "_tuned")]])
    _write_csv(output / "metrics.csv", metrics_table(frame, combined))
    for name in ("selected_parameters", "inner_candidates", "inner_audit", "fit_log"):
        _write_csv(output / f"{name}.csv", pd.concat([pd.read_csv(output / f"fold_{fold}" / f"{name}.csv") for fold in range(5)], ignore_index=True))
    decisions = []
    for column in combined.filter(like="p_"):
        decisions.append({"model": column.removeprefix("p_"), "scope": "pooled",
                          "cutoff": .5, "purpose": "fixed descriptive scenario; no OOF cutoff optimisation",
                          **decision_metrics(combined.target, combined[column].ge(.5).astype(int))})
    _write_csv(output / "fixed_cutoff_metrics.csv", pd.DataFrame(decisions))
    summary = {"schema_version": 1, "status": "complete", "run_id": run_id,
               "variant": "dedup_v1", "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
               "dataset_identity": identity, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "rows": len(frame), "patients": int(frame.patient_id.nunique()),
               "malignant_images": int(frame.target.sum()),
               "malignant_patients": int(frame.loc[frame.target.eq(1), "patient_id"].nunique()),
               "fit_counts": {"additional_baselines": 20, "partial_feature_ablations": 30,
                              "nested_inner": 630, "tuned_outer_refits": 15},
               "executed_fits": sum(item["executed_fits"] for item in fold_receipts),
               "executed_fits_this_invocation": invocation_fits, "reused_fits": sum(reuse_receipt["fit_counts"].values()),
               "avoided_repeat_fits": {"full_feature_ablation": 5, "untuned_nested_comparators": 15},
               "completed_fold_seconds": sum(item["elapsed_seconds"] for item in fold_receipts),
               "elapsed_seconds_this_invocation": time.monotonic() - started,
               "prediction_paths": prediction_paths, "prediction_columns": KEYS + ["score"],
               "metrics_path": _relative(root, output / "metrics.csv"),
               "selected_parameters_path": _relative(root, output / "selected_parameters.csv"),
               "memberships_receipt_path": _relative(root, membership_path),
               "reuse_receipt_path": _relative(root, _path(root, reuse_path, "results/dedup_v1/metadata")),
               "test_set_loaded": False, "final_model_fitted": False,
               "primary_metric": "unweighted mean outer-fold average precision",
               "interpretation": "All comparisons beyond predeclared B0/spline/NN are secondary; tuning selections use inner training patients only."}
    if summary["executed_fits"] != 695:
        raise ValueError("Clean metadata suite fit count differs from the locked 695-fit protocol.")
    _write_json(output / "summary.json", summary)
    files = [path for path in output.rglob("*") if path.is_file() and path != receipt_path]
    receipt = {**summary, "compatibility_sha256": compatibility_sha256,
               "source_sha256": compatibility["source_sha256"], "config_sha256": compatibility["config_sha256"],
               "versions": compatibility["versions"],
               "output_sha256": {_relative(root, path): file_sha256(path) for path in files}}
    _write_json(receipt_path, receipt)
    verify_suite_receipt(root, receipt, compatibility_sha256=compatibility_sha256)
    return receipt
