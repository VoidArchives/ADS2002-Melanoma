"""Bounded, resumable metadata NN selection on original and clean development.

All selection happens inside the saved metadata patient memberships. This
module never resolves a test path, reads reserved-test labels, or fits a final
all-development model. Existing modelling implementations remain unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
import json
import os
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import sklearn
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits
import torch
import xgboost
import yaml

from .dedup_metadata import (
    NN_SPECIFICATION, _verify_predictions, fit_network, neural_preprocessor,
    predict_network, verify_saved_receipt,
)
from .dedup_metadata_suite import (
    GRIDS, categorical_frame, inner_splits_from_memberships, metrics_table,
    model_setups, verify_suite_receipt,
)
from .development_dataset import CANONICAL_CSV, load_development_dataset, stable_sha256
from .metadata_evaluation import validate_probabilities
from .metadata_model import FEATURES, file_sha256
from .metadata_source_archive import verified_source_path
from .metadata_validation import validate_metadata_frame


KEYS = ["image_name", "patient_id", "fold", "target"]
BASE_SEEDS = [42, 142, 242]
DEFAULT_RUN_ID = "nn_grid36_v1"
CONDITION_PREFIXES = {
    "original": "results/metadata_nn_tuning/v1/original",
    "dedup_v1": "results/dedup_v1/metadata_nn_tuning/v1",
}
FIXED_COLUMNS = [
    "p_spline", "p_spline_tuned", "p_random_forest", "p_random_forest_tuned",
    "p_xgboost", "p_xgboost_tuned", "p_nn_mean",
    *[f"p_nn_seed_{seed}" for seed in BASE_SEEDS],
]
OOF_COLUMNS = KEYS + FIXED_COLUMNS + [
    "p_nn_tuned", *[f"p_nn_tuned_seed_{seed}" for seed in BASE_SEEDS],
]
SOURCE_FILES = (
    "src/melanoma_detection/metadata_nn_tuning.py",
    "scripts/run_metadata_nn_tuning.py",
    "src/melanoma_detection/dedup_metadata.py",
    "src/melanoma_detection/dedup_metadata_suite.py",
    "src/melanoma_detection/development_dataset.py",
    "src/melanoma_detection/metadata_model.py",
    "src/melanoma_detection/metadata_evaluation.py",
    "src/melanoma_detection/metadata_validation.py",
    "src/melanoma_detection/metadata_source_archive.py",
)


def recipes():
    """Return all 36 declared recipes, with the historical recipe first."""
    baseline = (32, 0.001, 0.2)
    combinations = [baseline] + [
        values for values in product([8, 16, 32, 64], [0.0003, 0.001, 0.003], [0.0, 0.2, 0.4])
        if values != baseline
    ]
    return [{"recipe_id": f"r{index:02d}", "recipe_order": index,
             "hidden_units": width, "learning_rate": rate, "dropout": dropout}
            for index, (width, rate, dropout) in enumerate(combinations)]


def ensemble_average_precision(y, seed_probabilities):
    """Score the equally weighted ensemble, never the mean of seed AP scores."""
    matrix = np.asarray(seed_probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("Three seed probability columns are required.")
    for column in matrix.T:
        validate_probabilities(y, column)
    mean = matrix.mean(axis=1)
    target, mean = validate_probabilities(y, mean)
    if np.unique(target).size != 2:
        raise ValueError("Candidate validation requires both classes.")
    return mean, float(average_precision_score(target, mean))


def select_recipe(candidates):
    """Choose by mean inner ensemble AP; exact ties follow declared order."""
    required = {"recipe_id", "recipe_order", "mean_inner_average_precision"}
    if candidates.empty or not required.issubset(candidates.columns):
        raise ValueError("Candidate AP scores and declared recipe ordering are required.")
    if candidates.recipe_id.duplicated().any() or candidates.recipe_order.duplicated().any():
        raise ValueError("Candidate recipe identities must be unique.")
    scores = candidates.mean_inner_average_precision.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
        raise ValueError("All declared candidates require finite AP scores.")
    ordered = candidates.sort_values(
        ["mean_inner_average_precision", "recipe_order"], ascending=[False, True], kind="stable",
    )
    return ordered.iloc[0].to_dict()


def _numeric_frame(frame):
    prepared = frame.copy()
    # Validation rejects malformed ages first; only sanctioned missing tokens
    # become NaN here, without learning anything from evaluation records.
    prepared["age_approx"] = pd.to_numeric(prepared.age_approx, errors="coerce")
    for column in ("sex", "anatom_site_general_challenge"):
        prepared[column] = prepared[column].fillna("Missing").astype(str)
    return prepared


def _validate_partition(parts, excluded_patients=()):
    fit, stop, refit, score = (parts[name] for name in ("epoch_fit", "epoch_stop", "refit", "score"))
    groups = [set(rows.patient_id) for rows in (fit, stop, score)]
    if any(not a.isdisjoint(b) for index, a in enumerate(groups) for b in groups[index + 1:]):
        raise ValueError("Fitting, stopping and scoring patients must be disjoint.")
    if set(refit.image_name) != set(fit.image_name) | set(stop.image_name):
        raise ValueError("Refitting must cover the full fitting-plus-stopping pool.")
    if not set(refit.patient_id).isdisjoint(score.patient_id):
        raise ValueError("A scoring patient entered refitting.")
    if set(refit.patient_id) & set(excluded_patients):
        raise ValueError("An outer patient entered inner training or stopping.")
    if any(rows.empty or rows.target.nunique() != 2 for rows in (fit, stop, refit, score)):
        raise ValueError("Every fitting, stopping and scoring partition needs both classes.")
    if any(not rows.image_name.is_unique for rows in (fit, stop, refit, score)):
        raise ValueError("Partition image IDs must be unique.")
    return parts


def inner_partition(frame, memberships, outer_fold, inner_fold, seed_index):
    """Use only the two remaining metadata inner groups for stopping/refit."""
    if outer_fold not in range(5) or inner_fold not in range(3) or seed_index not in range(3):
        raise ValueError("Invalid outer fold, inner fold or prescribed seed index.")
    inner_splits_from_memberships(frame, outer_fold, memberships)
    pool = frame.loc[frame.fold.ne(outer_fold)].reset_index(drop=True)
    assigned = memberships.loc[memberships.outer_fold.eq(outer_fold)].set_index("image_name")
    assignment = assigned.loc[pool.image_name, "inner_fold"].to_numpy()
    remaining = [(inner_fold + 1) % 3, (inner_fold + 2) % 3]
    stop_group = remaining[seed_index % 2]
    fit_group = remaining[1 - seed_index % 2]
    parts = {"epoch_fit": pool.loc[assignment == fit_group].copy(),
             "epoch_stop": pool.loc[assignment == stop_group].copy(),
             "refit": pool.loc[assignment != inner_fold].copy(),
             "score": pool.loc[assignment == inner_fold].copy(),
             "stop_group": stop_group}
    return _validate_partition(parts, frame.loc[frame.fold.eq(outer_fold), "patient_id"])


def outer_partition(frame, outer_fold, seed_index):
    """Keep the historical three stopping-fold rules for outer refitting."""
    if outer_fold not in range(5) or seed_index not in range(3):
        raise ValueError("Invalid outer fold or prescribed seed index.")
    stop_fold = (outer_fold + 1 + seed_index) % 5
    parts = {"epoch_fit": frame.loc[~frame.fold.isin([outer_fold, stop_fold])].copy(),
             "epoch_stop": frame.loc[frame.fold.eq(stop_fold)].copy(),
             "refit": frame.loc[frame.fold.ne(outer_fold)].copy(),
             "score": frame.loc[frame.fold.eq(outer_fold)].copy(),
             "stop_group": stop_fold}
    return _validate_partition(parts)


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _csv(path, frame):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _joblib(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def _relative(root, path):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _contained(root, value, allowed):
    root = Path(root).resolve()
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    allowed = Path(allowed)
    allowed = allowed.resolve() if allowed.is_absolute() else (root / allowed).resolve()
    if not allowed.is_relative_to(root) or not path.is_relative_to(allowed):
        raise ValueError("Artifact path must stay within its declared run directory.")
    return path


def verify_receipt(root, receipt, *, compatibility_sha256=None, output_root=None):
    """Verify a completed run or phase without importing edited notebooks."""
    root = Path(root).resolve()
    if receipt.get("status") != "complete":
        raise ValueError("A complete metadata tuning receipt is required.")
    if compatibility_sha256 is not None and receipt.get("compatibility_sha256") != compatibility_sha256:
        raise ValueError("Dataset, source, config, runtime or trial changed; use a new run identity.")
    declared = receipt.get("output_root")
    if not isinstance(declared, str):
        raise ValueError("Receipt must declare its output root.")
    condition = receipt.get("condition")
    if condition not in CONDITION_PREFIXES:
        raise ValueError("Receipt condition must be original or dedup_v1.")
    run_root = _contained(root, declared, CONDITION_PREFIXES[condition])
    if output_root is not None and run_root != Path(output_root).resolve():
        raise ValueError("Receipt output root differs from the requested run.")
    hashes = receipt.get("output_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Receipt requires output fingerprints.")
    for relative, expected in hashes.items():
        path = _contained(root, relative, run_root)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Completed tuning artifact is missing or changed: {relative}")
    return receipt


def condition_directory(project_root, condition, run_id=DEFAULT_RUN_ID):
    """Resolve only the two versioned development output namespaces."""
    if condition not in CONDITION_PREFIXES or not isinstance(run_id, str) or not run_id or len(run_id) > 64 or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in run_id
    ):
        raise ValueError("A known condition and simple run identity are required.")
    root = Path(project_root).resolve()
    return _contained(root, f"{CONDITION_PREFIXES[condition]}/{run_id}", CONDITION_PREFIXES[condition])


def verify_prediction_coverage(frame, predictions, *, fold=None):
    """Require exactly one valid score record for every supplied development ID."""
    return _verify_predictions(frame, predictions, fold=fold)


def verify_tuning_receipt(root, receipt, *, condition=None, compatibility_sha256=None):
    if condition is not None and receipt.get("condition") != condition:
        raise ValueError("Tuning receipt belongs to another development condition.")
    return verify_receipt(root, receipt, compatibility_sha256=compatibility_sha256)


def load_tuning_outputs(project_root, condition, run_id=None):
    """Load completed, verified outputs; absence is an explicit pending stage."""
    root = Path(project_root).resolve()
    if run_id is None:
        if condition not in CONDITION_PREFIXES:
            raise ValueError("Unknown development condition.")
        pointer_path = root / CONDITION_PREFIXES[condition] / "latest.json"
        if pointer_path.exists():
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            run_id = pointer["run_id"]
        else:
            run_id = DEFAULT_RUN_ID
    directory = condition_directory(root, condition, run_id)
    receipt_path = directory / "receipt.json"
    if not receipt_path.is_file():
        return {"status": "pending", "directory": directory, "receipt": None, "oof": None}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verify_tuning_receipt(root, receipt, condition=condition)
    if receipt.get("run_id") != run_id or (root / receipt["output_root"]).resolve() != directory:
        raise ValueError("Receipt does not identify the requested condition/run.")
    manifest = receipt["dataset_identity"].get("dataset_manifest") if condition == "dedup_v1" else None
    frame, identity = load_development_dataset(root, manifest)
    if receipt["dataset_identity"] != identity:
        raise ValueError("Completed NN tuning dataset identity changed.")
    oof = pd.read_csv(directory / "oof.csv", float_precision="round_trip")
    verify_prediction_coverage(frame, oof)
    return {"status": "complete", "directory": directory, "receipt": receipt, "oof": oof}


def _check_stop(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Metadata tuning paused between complete fitting phases.")


def _phase(root, folder, compatibility_sha256, context, compute, stop_event=None):
    """Persist a completed phase before starting its dependent refit or score."""
    if folder is None:
        _check_stop(stop_event)
        return compute(), False
    root, folder = Path(root).resolve(), Path(folder).resolve()
    output_root = root / context["output_root"]
    _contained(root, folder, output_root)
    completion = folder / "receipt.json"
    if completion.exists():
        receipt = json.loads(completion.read_text(encoding="utf-8"))
        verify_receipt(root, receipt, compatibility_sha256=compatibility_sha256, output_root=output_root)
        if receipt.get("context") != context:
            raise ValueError("Saved phase context changed.")
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        for name in receipt.get("csv_names", []):
            result[name] = pd.read_csv(folder / f"{name}.csv", float_precision="round_trip")
        # Model artifacts are saved for audit/resume but scoring was already
        # completed in this phase; loading their pickle is unnecessary.
        return result, True
    _check_stop(stop_event)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = compute()
    serializable, csv_names, files = {}, [], []
    for name, value in result.items():
        if isinstance(value, pd.DataFrame):
            _csv(folder / f"{name}.csv", value)
            csv_names.append(name)
            files.append(folder / f"{name}.csv")
        elif name == "model":
            _joblib(folder / "model.joblib", value)
            files.append(folder / "model.joblib")
        else:
            serializable[name] = value
    serializable["elapsed_seconds"] = time.monotonic() - started
    _json(folder / "result.json", serializable)
    files.append(folder / "result.json")
    _json(completion, {"schema_version": 1, "status": "complete",
          "condition": context["condition"], "output_root": context["output_root"],
          "compatibility_sha256": compatibility_sha256, "context": context,
          "csv_names": csv_names, "output_sha256": {_relative(root, path): file_sha256(path) for path in files}})
    result["elapsed_seconds"] = serializable["elapsed_seconds"]
    return result, False


def fit_nn_pair(partitions, *, base_seed, outer_fold, spec=None, context=None,
                project_root=None, output_dir=None, compatibility_sha256=None, stop_event=None):
    """Select duration, freshly refit and score; optionally checkpoint phases.

    ``output_dir=None`` supports small synthetic checks without writing any run.
    Predictions are a DataFrame with KEYS and a single ``score`` column. The
    selected-epoch record is returned as ``epoch`` and loss rows as ``history``.
    """
    _validate_partition(partitions)
    spec = deepcopy(NN_SPECIFICATION if spec is None else spec)
    torch.set_num_threads(spec["threads"])
    context = {} if context is None else dict(context)
    actual_seed = int(base_seed) + int(outer_fold)
    record = {**context, "outer_fold": int(outer_fold), "base_seed": int(base_seed),
              "training_seed": actual_seed, "stop_group": int(partitions["stop_group"])}
    if output_dir is not None and (project_root is None or compatibility_sha256 is None):
        raise ValueError("Checkpointing needs project_root and compatibility_sha256.")
    folder = None if output_dir is None else Path(output_dir)
    parts = {name: _numeric_frame(partitions[name]) for name in ("epoch_fit", "epoch_stop", "refit", "score")}
    identity = stable_sha256({"context": record, "spec": spec,
                             "image_ids": {name: rows.image_name.tolist() for name, rows in parts.items()}})
    phase_compatibility = stable_sha256({"run": compatibility_sha256, "trial": identity})

    def stopping_fit():
        prep = neural_preprocessor().fit(parts["epoch_fit"][FEATURES])
        _, selected, loss, history = fit_network(
            prep.transform(parts["epoch_fit"][FEATURES]), parts["epoch_fit"].target, actual_seed,
            x_stop=prep.transform(parts["epoch_stop"][FEATURES]), y_stop=parts["epoch_stop"].target, spec=spec,
        )
        return {"selected_epochs": int(selected), "stop_log_loss": float(loss),
                "attempted_stopping_epochs": len(history), "history": pd.DataFrame(history)}

    stopping, reused_stopping = _phase(
        project_root, None if folder is None else folder / "stopping", phase_compatibility,
        {**record, "phase": "stopping"}, stopping_fit, stop_event,
    )

    def refit_fit():
        prep = neural_preprocessor().fit(parts["refit"][FEATURES])
        encoded = prep.transform(parts["refit"][FEATURES])
        model, _, _, history = fit_network(encoded, parts["refit"].target, actual_seed,
                                           epochs=stopping["selected_epochs"], spec=spec)
        probability = predict_network(model, prep.transform(parts["score"][FEATURES]))
        validate_probabilities(parts["score"].target, probability)
        predictions = parts["score"][KEYS].copy()
        predictions["score"] = probability
        return {"predictions": predictions, "history": pd.DataFrame(history),
                "encoded_input_width": int(encoded.shape[1]),
                "model": {"preprocessor": prep, "state_dict": model.state_dict(), "spec": spec,
                          "training_seed": actual_seed, "selected_epochs": stopping["selected_epochs"],
                          "fit_image_ids": parts["refit"].image_name.tolist(),
                          "fit_patient_ids": sorted(parts["refit"].patient_id.unique())}}

    refitting, reused_refit = _phase(
        project_root, None if folder is None else folder / "refit", phase_compatibility,
        {**record, "phase": "refit", "selected_epochs": stopping["selected_epochs"]}, refit_fit, stop_event,
    )
    epoch = {**record, "selected_epochs": stopping["selected_epochs"],
             "stop_log_loss": stopping["stop_log_loss"],
             "attempted_stopping_epochs": stopping["attempted_stopping_epochs"],
             "encoded_input_width": refitting["encoded_input_width"]}
    for name, rows in parts.items():
        epoch[f"{name}_images"] = len(rows)
        epoch[f"{name}_patients"] = int(rows.patient_id.nunique())
    histories, logs = [], []
    for phase, result, reused in (("stopping", stopping, reused_stopping), ("refit", refitting, reused_refit)):
        history = result["history"].copy()
        for name, value in {**record, "phase": phase}.items():
            history[name] = value
        histories.append(history)
        logs.append({**record, "phase": phase, "fits": 1, "reused_fits": 0,
                     "executed_this_invocation": 0 if reused else 1,
                     "elapsed_seconds": result.get("elapsed_seconds", 0.0)})
    return {"predictions": refitting["predictions"], "epoch": epoch,
            "history": pd.concat(histories, ignore_index=True), "fit_log": logs}


def _versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "torch": torch.__version__,
            "xgboost": xgboost.__version__, "joblib": joblib.__version__}


def _read_config(root, config_path):
    path = (root / config_path).resolve()
    if not path.is_relative_to(root / "configs"):
        raise ValueError("A workspace configuration in configs/ is required.")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"run_id", "conditions", "dataset_manifest", "metadata_suite_receipt",
                "clean_comparator_receipt", "threads", "hidden_units", "learning_rates", "dropouts", "weight_decay"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("The bounded metadata tuning configuration has unexpected or missing fields.")
    run_id = config["run_id"]
    if not isinstance(run_id, str) or not run_id or len(run_id) > 64 or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in run_id
    ):
        raise ValueError("A simple, nonempty run_id of at most 64 characters is required.")
    if (config["conditions"] != ["original", "dedup_v1"] or config["threads"] != 4
        or config["hidden_units"] != [8, 16, 32, 64]
        or config["learning_rates"] != [0.0003, 0.001, 0.003]
        or config["dropouts"] != [0.0, 0.2, 0.4] or config["weight_decay"] != 0.001):
        raise ValueError("The approved 36 recipes, both conditions and four CPU threads are fixed.")
    return path, config


def _load_inputs(root, config):
    original, original_identity = load_development_dataset(root)
    clean, clean_identity = load_development_dataset(root, config["dataset_manifest"])
    if (len(original), len(clean)) != (6864, 6706):
        raise ValueError("The approved original and deduplicated development cohorts changed.")
    suite_path = _contained(root, config["metadata_suite_receipt"], "results/dedup_v1/metadata_suite")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    verify_suite_receipt(root, suite)
    if suite["dataset_identity"] != clean_identity:
        raise ValueError("Clean suite and development manifest identities disagree.")
    # Notebook sources are editable reports. Only stable training dependency
    # sources are compared here; every reused output is still verified above.
    for relative, digest in suite["source_sha256"].items():
        verified_source_path(root, relative, digest)
    source_directory = suite_path.parent
    original_memberships = pd.read_csv(source_directory / "inner_memberships_original.csv")
    clean_memberships = pd.read_csv(source_directory / "inner_memberships_retained.csv")
    expected = original_memberships.loc[original_memberships.image_name.isin(clean.image_name)].copy()
    order = ["outer_fold", "image_name"]
    try:
        assert_frame_equal(expected.sort_values(order).reset_index(drop=True),
                           clean_memberships.sort_values(order).reset_index(drop=True), check_dtype=False)
    except AssertionError as error:
        raise ValueError("Clean metadata memberships must be the original memberships filtered by retained IDs.") from error
    for frame, membership in ((original, original_memberships), (clean, clean_memberships)):
        for outer in range(5):
            inner_splits_from_memberships(frame, outer, membership)
    comparator_path = _contained(root, config["clean_comparator_receipt"], "results/dedup_v1/metadata")
    comparator = json.loads(comparator_path.read_text(encoding="utf-8"))
    verify_saved_receipt(root, comparator)
    if comparator.get("dataset_identity") != clean_identity or comparator["specification"]["neural_network"] != NN_SPECIFICATION:
        raise ValueError("Clean fixed-NN comparator identity or protocol changed.")
    for relative, digest in comparator["source_sha256"].items():
        verified_source_path(root, relative, digest)
    reused = pd.read_csv(source_directory / "combined_oof.csv", float_precision="round_trip")
    fixed_nn = pd.read_csv(_contained(root, comparator["prediction_path"], "results/dedup_v1/metadata"), float_precision="round_trip")
    _verify_predictions(clean, reused)
    _verify_predictions(clean, fixed_nn)
    reused = reused.set_index("image_name").loc[clean.image_name].reset_index()
    fixed_nn = fixed_nn.set_index("image_name").loc[clean.image_name].reset_index()
    neural_columns = ["p_nn_mean", *[f"p_nn_seed_{seed}" for seed in BASE_SEEDS]]
    if not np.array_equal(reused[neural_columns].to_numpy(), fixed_nn[neural_columns].to_numpy()):
        raise ValueError("Clean suite fixed neural predictions disagree with the verified comparator.")
    raw = validate_metadata_frame(pd.read_csv(root / CANONICAL_CSV, dtype={"age_approx": "string"}))
    frames = {"original": original, "dedup_v1": clean}
    categorical = {}
    for condition, frame in frames.items():
        categorical[condition] = frame.copy()
        categorical[condition]["age_approx"] = raw.set_index("image_name").loc[frame.image_name, "age_approx"].to_numpy()
        categorical[condition] = categorical_frame(categorical[condition])
    reuse_identity = {"metadata_suite_receipt_sha256": file_sha256(suite_path),
                      "clean_comparator_receipt_sha256": file_sha256(comparator_path),
                      "metadata_original_memberships_sha256": file_sha256(source_directory / "inner_memberships_original.csv"),
                      "metadata_clean_memberships_sha256": file_sha256(source_directory / "inner_memberships_retained.csv")}
    return {"frames": frames, "identities": {"original": original_identity, "dedup_v1": clean_identity},
            "memberships": {"original": original_memberships, "dedup_v1": clean_memberships},
            "categorical": categorical, "clean_fixed": reused[KEYS + FIXED_COLUMNS],
            "reuse_identity": reuse_identity, "suite": suite, "comparator": comparator}


@contextmanager
def _run_lock(output):
    """A crashed process releases the OS lock; stale lock files are harmless."""
    path = Path(output) / "run.lock"
    handle = path.open("a+b")
    try:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        handle.close()


def _role_rows(frame, parts, context):
    fit, stop, score, refit = (set(parts[name].patient_id) for name in ("epoch_fit", "epoch_stop", "score", "refit"))
    rows = []
    for (patient, fold), group in frame.groupby(["patient_id", "fold"], sort=False):
        role = ("epoch_fitting" if patient in fit else "epoch_stopping" if patient in stop
                else "validation" if patient in score else "outer_excluded")
        rows.append({**context, "patient_id": patient, "fold": int(fold), "role": role,
                     "images": len(group), "refit_included": patient in refit})
    return rows


def _reused_fixed_diagnostics(root, output, frame, comparator):
    """Carry verified clean fixed-NN diagnostics into the new report namespace."""
    source = (root / comparator["prediction_path"]).parent
    epochs = pd.read_csv(source / "epochs.csv")
    history = pd.read_csv(source / "history.csv")
    epoch_rows = []
    for old in epochs.to_dict("records"):
        outer, seed = int(old["outer_fold"]), int(old["base_seed"])
        parts = outer_partition(frame, outer, BASE_SEEDS.index(seed))
        context = {"condition": "dedup_v1", "outer_fold": outer,
                   "output_root": _relative(root, output), **recipes()[0],
                   "stage": "nn_outer_fixed", "inner_fold": -1,
                   "provenance": "reused_clean_fixed"}
        record = {**old, **context, "stop_group": old["stop_fold"],
                  "training_seed": seed + outer}
        for name in ("epoch_fit", "epoch_stop", "refit", "score"):
            record[f"{name}_images"] = len(parts[name])
            record[f"{name}_patients"] = int(parts[name].patient_id.nunique())
        epoch_rows.append(record)
    history["condition"] = "dedup_v1"
    history["output_root"] = _relative(root, output)
    history["stage"] = "nn_outer_fixed"
    history["inner_fold"] = -1
    history["training_seed"] = history.base_seed + history.outer_fold
    history["stop_group"] = (history.outer_fold + 1 + history.base_seed.map(BASE_SEEDS.index)) % 5
    history["phase"] = history.phase.replace({"outer_refit": "refit"})
    history["provenance"] = "reused_clean_fixed"
    for name, value in recipes()[0].items():
        history[name] = value
    return epoch_rows, history


def _classical_outer(root, output, frame, memberships, outer, compatibility, stop_event):
    """Checkpoint every fixed, inner-candidate and selected classical fit."""
    fitting = frame.loc[frame.fold.ne(outer)].reset_index(drop=True)
    heldout = frame.loc[frame.fold.eq(outer)].reset_index(drop=True)
    splits, audit = inner_splits_from_memberships(frame, outer, memberships)
    setups = model_setups(4)
    predictions = heldout[KEYS].copy()
    candidates, selected, logs = [], [], []
    condition = "original"
    base_context = {"condition": condition, "outer_fold": outer, "output_root": _relative(root, output)}

    def fitted_phase(estimator, train, validation, *, save_model=True):
        model = clone(estimator).fit(train[FEATURES], train.target)
        scores = model.predict_proba(validation[FEATURES])[:, 1]
        validate_probabilities(validation.target, scores)
        predicted = validation[KEYS].copy()
        predicted["score"] = scores
        result = {"predictions": predicted,
                  "average_precision": float(average_precision_score(validation.target, scores))}
        if save_model:
            result["model"] = model
        return result

    for name, estimator in setups.items():
        context = {**base_context, "model": name, "phase": "classical_fixed", "inner_fold": -1}
        result, reused = _phase(root, output / "classical" / f"outer_{outer}" / name / "fixed", compatibility,
                                context, lambda est=estimator: fitted_phase(est, fitting, heldout), stop_event)
        predictions[f"p_{name}"] = result["predictions"].score.to_numpy()
        logs.append({**context, "fits": 1, "reused_fits": 0, "executed_this_invocation": int(not reused),
                     "elapsed_seconds": result["elapsed_seconds"]})
        rows = []
        for index, parameters in enumerate(ParameterGrid(GRIDS[name])):
            scores = []
            for inner, (fit_ids, val_ids) in enumerate(splits):
                context = {**base_context, "model": name, "phase": "classical_inner", "inner_fold": inner,
                           "candidate_index": index, "parameters": parameters}
                candidate = clone(estimator).set_params(**parameters)
                result, reused = _phase(
                    root, output / "classical" / f"outer_{outer}" / name / f"candidate_{index:02d}" / f"inner_{inner}",
                    compatibility, context,
                    lambda est=candidate, a=fit_ids, b=val_ids: fitted_phase(est, fitting.iloc[a], fitting.iloc[b], save_model=False), stop_event,
                )
                scores.append(result["average_precision"])
                logs.append({**context, "parameters": json.dumps(parameters, sort_keys=True), "fits": 1, "reused_fits": 0,
                             "executed_this_invocation": int(not reused), "elapsed_seconds": result["elapsed_seconds"]})
            row = {"condition": condition, "outer_fold": outer, "model": name, "candidate_index": index,
                   "parameters": json.dumps(parameters, sort_keys=True),
                   "mean_inner_average_precision": float(np.mean(scores)),
                   "std_inner_average_precision": float(np.std(scores))}
            row.update({f"inner_{index}_average_precision": value for index, value in enumerate(scores)})
            rows.append(row)
        winner = sorted(rows, key=lambda row: (-row["mean_inner_average_precision"], row["candidate_index"]))[0]
        for row in rows:
            row["selected"] = row["candidate_index"] == winner["candidate_index"]
        candidates.extend(rows)
        selected.append(winner)
        context = {**base_context, "model": name, "phase": "classical_tuned", "inner_fold": -1,
                   "candidate_index": winner["candidate_index"], "parameters": json.loads(winner["parameters"])}
        candidate = clone(estimator).set_params(**context["parameters"])
        result, reused = _phase(root, output / "classical" / f"outer_{outer}" / name / "selected", compatibility,
                                context, lambda est=candidate: fitted_phase(est, fitting, heldout), stop_event)
        predictions[f"p_{name}_tuned"] = result["predictions"].score.to_numpy()
        logs.append({**context, "parameters": winner["parameters"], "fits": 1, "reused_fits": 0,
                     "executed_this_invocation": int(not reused), "elapsed_seconds": result["elapsed_seconds"]})
    return predictions, candidates, selected, logs, audit


def _protocol():
    return {"recipes": recipes(), "neural_training": NN_SPECIFICATION,
            "classical_grids": GRIDS, "base_seeds": BASE_SEEDS,
            "candidate_selection": "unweighted mean of three inner ensemble average-precision scores",
            "tie_rule": "first in declared recipe order; historical baseline first",
            "inner_stopping": "remaining=[(v+1)%3,(v+2)%3]; stop=remaining[s%2]; fit=remaining[1-s%2]",
            "outer_stopping": "(outer_fold+1+seed_index)%5",
            "training_seed": "base_seed+outer_fold", "ensemble": "equal mean of all three seed probabilities",
            "membership_source": "saved metadata-suite original memberships; clean is retained-ID filtering",
            "primary_metric": "unweighted mean outer-fold average precision",
            "test_set_loaded": False, "final_model_fitted": False}


def _run_condition(root, config_path, config, inputs, condition, progress, stop_event):
    frame = inputs["frames"][condition]
    membership = inputs["memberships"][condition]
    output = _contained(root, f"{CONDITION_PREFIXES[condition]}/{config['run_id']}", CONDITION_PREFIXES[condition])
    source_hashes = {relative: file_sha256(root / relative) for relative in SOURCE_FILES}
    compatibility = {"condition": condition, "dataset_identity": inputs["identities"][condition],
                     "reuse_identity": inputs["reuse_identity"], "protocol": _protocol(),
                     "source_sha256": source_hashes, "config_sha256": file_sha256(config_path), "versions": _versions()}
    digest = stable_sha256(compatibility)
    output.mkdir(parents=True, exist_ok=True)
    with _run_lock(output):
        completion = output / "receipt.json"
        if completion.exists():
            receipt = json.loads(completion.read_text(encoding="utf-8"))
            verify_receipt(root, receipt, compatibility_sha256=digest, output_root=output)
            if progress:
                progress(f"Verified completed {condition} metadata tuning; no fits repeated.")
            return receipt
        session = output / "session.json"
        if session.exists():
            if json.loads(session.read_text(encoding="utf-8"))["compatibility_sha256"] != digest:
                raise ValueError("Partial tuning inputs changed; choose a new run identity.")
        elif any(path.name != "run.lock" for path in output.iterdir()):
            raise ValueError("Existing tuning directory has no compatible session receipt.")
        else:
            _json(session, {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                            "run_id": config["run_id"], "output_root": _relative(root, output),
                            "compatibility_sha256": digest, **compatibility})
        _csv(output / "inner_memberships.csv", membership)
        oof_parts, candidates, chosen, epoch_rows, histories, roles, fit_log = [], [], [], [], [], [], []
        if condition == "dedup_v1":
            fixed_epochs, fixed_history = _reused_fixed_diagnostics(root, output, frame, inputs["comparator"])
            epoch_rows.extend(fixed_epochs)
            histories.append(fixed_history)
        classical_candidates, classical_selected, classical_audits = [], [], []
        started = time.monotonic()
        for outer in range(5):
            if progress:
                progress(f"{condition}: outer fold {outer + 1}/5; evaluating all 36 NN recipes on inner patients.")
            base_context = {"condition": condition, "outer_fold": outer, "output_root": _relative(root, output)}
            inner_parts = {(inner, s): inner_partition(frame, membership, outer, inner, s)
                           for inner in range(3) for s in range(3)}
            for (inner, s), parts in inner_parts.items():
                roles.extend(_role_rows(frame, parts, {**base_context, "stage": "nn_inner", "inner_fold": inner,
                                                      "base_seed": BASE_SEEDS[s], "stop_group": parts["stop_group"]}))
            outer_candidates = []
            for recipe in recipes():
                spec = {**NN_SPECIFICATION, **{key: recipe[key] for key in ("hidden_units", "learning_rate", "dropout")}}
                inner_scores = []
                for inner in range(3):
                    probabilities = []
                    for s, seed in enumerate(BASE_SEEDS):
                        context = {**base_context, **recipe, "stage": "nn_inner", "inner_fold": inner}
                        result = fit_nn_pair(inner_parts[(inner, s)], base_seed=seed, outer_fold=outer,
                            spec=spec, context=context, project_root=root,
                            output_dir=output / "trials" / f"outer_{outer}" / recipe["recipe_id"] / f"inner_{inner}" / f"seed_{seed}",
                            compatibility_sha256=digest, stop_event=stop_event)
                        probabilities.append(result["predictions"].score.to_numpy())
                        epoch_rows.append(result["epoch"])
                        histories.append(result["history"])
                        fit_log.extend(result["fit_log"])
                    _, ap = ensemble_average_precision(inner_parts[(inner, 0)]["score"].target, np.column_stack(probabilities))
                    inner_scores.append(ap)
                row = {"condition": condition, "outer_fold": outer, **recipe,
                       "mean_inner_average_precision": float(np.mean(inner_scores)),
                       "std_inner_average_precision": float(np.std(inner_scores))}
                row.update({f"inner_{index}_average_precision": score for index, score in enumerate(inner_scores)})
                outer_candidates.append(row)
                if progress:
                    progress(f"{condition} outer {outer}: {recipe['recipe_id']} width={recipe['hidden_units']} "
                             f"lr={recipe['learning_rate']} dropout={recipe['dropout']} inner ensemble AP={row['mean_inner_average_precision']:.6f}.")
            winner = select_recipe(pd.DataFrame(outer_candidates))
            ordered_scores = sorted((row["mean_inner_average_precision"] for row in outer_candidates), reverse=True)
            winner["selection_margin"] = ordered_scores[0] - ordered_scores[1]
            for row in outer_candidates:
                row["selected"] = row["recipe_id"] == winner["recipe_id"]
            candidates.extend(outer_candidates)
            chosen.append(winner)
            _csv(output / "candidates.csv", pd.DataFrame(candidates))
            _csv(output / "selected_parameters.csv", pd.DataFrame(chosen))
            selected_spec = {**NN_SPECIFICATION, **{key: winner[key] for key in ("hidden_units", "learning_rate", "dropout")}}
            scored = frame.loc[frame.fold.eq(outer), KEYS].copy()
            for s, seed in enumerate(BASE_SEEDS):
                parts = outer_partition(frame, outer, s)
                context = {**base_context, **{key: winner[key] for key in recipes()[0]},
                           "stage": "nn_outer_tuned", "inner_fold": -1}
                roles.extend(_role_rows(frame, parts, {**base_context, "stage": "nn_outer_tuned", "inner_fold": -1,
                                                      "base_seed": seed, "stop_group": parts["stop_group"]}))
                result = fit_nn_pair(parts, base_seed=seed, outer_fold=outer, spec=selected_spec,
                    context=context, project_root=root,
                    output_dir=output / "trials" / f"outer_{outer}" / "selected" / f"seed_{seed}",
                    compatibility_sha256=digest, stop_event=stop_event)
                scored[f"p_nn_tuned_seed_{seed}"] = result["predictions"].score.to_numpy()
                epoch_rows.append(result["epoch"])
                histories.append(result["history"])
                fit_log.extend(result["fit_log"])
            scored["p_nn_tuned"] = scored[[f"p_nn_tuned_seed_{seed}" for seed in BASE_SEEDS]].mean(axis=1)
            if condition == "original":
                for s, seed in enumerate(BASE_SEEDS):
                    parts = outer_partition(frame, outer, s)
                    context = {**base_context, **recipes()[0], "stage": "nn_outer_fixed", "inner_fold": -1}
                    roles.extend(_role_rows(frame, parts, {**base_context, "stage": "nn_outer_fixed", "inner_fold": -1,
                                                          "base_seed": seed, "stop_group": parts["stop_group"]}))
                    result = fit_nn_pair(parts, base_seed=seed, outer_fold=outer,
                        context=context, project_root=root,
                        output_dir=output / "trials" / f"outer_{outer}" / "fixed_nn" / f"seed_{seed}",
                        compatibility_sha256=digest, stop_event=stop_event)
                    scored[f"p_nn_seed_{seed}"] = result["predictions"].score.to_numpy()
                    epoch_rows.append(result["epoch"])
                    histories.append(result["history"])
                    fit_log.extend(result["fit_log"])
                scored["p_nn_mean"] = scored[[f"p_nn_seed_{seed}" for seed in BASE_SEEDS]].mean(axis=1)
                conventional, cc, cs, logs, audit = _classical_outer(
                    root, output, inputs["categorical"][condition], membership, outer, digest, stop_event)
                for column in conventional.filter(like="p_"):
                    scored[column] = conventional[column].to_numpy()
                classical_candidates.extend(cc)
                classical_selected.extend(cs)
                classical_audits.append(audit)
                fit_log.extend(logs)
            else:
                for s, seed in enumerate(BASE_SEEDS):
                    parts = outer_partition(frame, outer, s)
                    roles.extend(_role_rows(frame, parts, {**base_context, "stage": "nn_outer_fixed", "inner_fold": -1,
                                                          "base_seed": seed, "stop_group": parts["stop_group"],
                                                          "provenance": "reused_clean_fixed"}))
                fixed = inputs["clean_fixed"].set_index("image_name").loc[scored.image_name]
                for column in FIXED_COLUMNS:
                    scored[column] = fixed[column].to_numpy()
                fit_log.append({**base_context, "phase": "reused_fixed_nn", "model": "nn_fixed", "inner_fold": -1,
                                "fits": 0, "reused_fits": 6, "executed_this_invocation": 0, "elapsed_seconds": 0.0})
                fit_log.append({**base_context, "phase": "reused_classical", "model": "three_families", "inner_fold": -1,
                                "fits": 0, "reused_fits": 132, "executed_this_invocation": 0, "elapsed_seconds": 0.0})
            scored = scored[OOF_COLUMNS]
            _verify_predictions(frame, scored, fold=outer)
            oof_parts.append(scored)
            _csv(output / "epochs.csv", pd.DataFrame(epoch_rows))
            _csv(output / "histories.csv", pd.concat(histories, ignore_index=True))
            _csv(output / "memberships.csv", pd.DataFrame(roles))
            _csv(output / "fit_log.csv", pd.DataFrame(fit_log))
            _csv(output / "oof_partial.csv", pd.concat(oof_parts, ignore_index=True))
        oof = pd.concat(oof_parts, ignore_index=True).set_index("image_name").loc[frame.image_name].reset_index()
        _verify_predictions(frame, oof)
        _csv(output / "oof.csv", oof[OOF_COLUMNS])
        _csv(output / "metrics.csv", metrics_table(frame, oof))
        if condition == "original":
            _csv(output / "classical_candidates.csv", pd.DataFrame(classical_candidates))
            _csv(output / "classical_selected_parameters.csv", pd.DataFrame(classical_selected))
            _csv(output / "classical_inner_audit.csv", pd.concat(classical_audits, ignore_index=True))
        else:
            source = root / config["metadata_suite_receipt"]
            classical_candidates = pd.read_csv(source.parent / "inner_candidates.csv")
            classical_selected = pd.read_csv(source.parent / "selected_parameters.csv").rename(columns={
                "best_parameters": "parameters", "inner_best_average_precision": "mean_inner_average_precision",
                "best_candidate_index": "candidate_index",
            })
            classical_audit = pd.read_csv(source.parent / "inner_audit.csv")
            for table in (classical_candidates, classical_selected, classical_audit):
                table["condition"] = condition
                table["provenance"] = "reused_clean_classical"
            _csv(output / "classical_candidates.csv", classical_candidates)
            _csv(output / "classical_selected_parameters.csv", classical_selected)
            _csv(output / "classical_inner_audit.csv", classical_audit)
        total_fits = sum(row["fits"] for row in fit_log)
        expected_fits = 3960 if condition == "original" else 3270
        if total_fits != expected_fits:
            raise ValueError("Completed fit count differs from the approved bounded experiment.")
        receipt = {"schema_version": 1, "status": "complete", "condition": condition,
                   "run_id": config["run_id"], "output_root": _relative(root, output),
                   "created_at_utc": datetime.now(timezone.utc).isoformat(), "compatibility_sha256": digest,
                   **compatibility, "rows": len(frame), "patients": int(frame.patient_id.nunique()),
                   "positive_images": int(frame.target.sum()), "positive_patients": int(frame.loc[frame.target.eq(1)].patient_id.nunique()),
                   "executed_fits": total_fits, "reused_fits": sum(row["reused_fits"] for row in fit_log),
                   "executed_fits_this_invocation": sum(row["executed_this_invocation"] for row in fit_log),
                   "elapsed_seconds_this_invocation": time.monotonic() - started,
                   "neural_tuning_fits": 3270, "fixed_neural_fits": 30 if condition == "original" else 0,
                   "classical_fits": 660 if condition == "original" else 0,
                   "test_set_loaded": False, "final_model_fitted": False,
                   "oof_path": _relative(root, output / "oof.csv"), "prediction_columns": OOF_COLUMNS,
                   "interpretation": "Nested development evidence after previous exploration; all candidates were frozen before fitting."}
        files = [path for path in output.rglob("*") if path.is_file()
                 and path != completion and path.name != "run.lock" and ".tmp" not in path.name
                 and "report" not in path.relative_to(output).parts]
        receipt["output_sha256"] = {_relative(root, path): file_sha256(path) for path in files}
        _json(completion, receipt)
        verify_receipt(root, receipt, compatibility_sha256=digest, output_root=output)
        _json(output.parent / "latest.json", {"schema_version": 1, "condition": condition,
              "run_id": config["run_id"], "receipt_path": _relative(root, completion),
              "receipt_sha256": file_sha256(completion)})
        if progress:
            progress(f"Complete {condition}: {len(frame)} OOF records; {total_fits} executed fits, "
                     f"{receipt['reused_fits']} existing fits reused; reserved test remained unopened.")
        return receipt


def run_tuning(project_root, config_path="configs/metadata_nn_tuning.yaml", *, condition=None,
               progress=print, stop_event=None):
    """Execute or resume the frozen search; never choose from outer outcomes."""
    root = Path(project_root).resolve()
    path, config = _read_config(root, config_path)
    if condition is not None and condition not in config["conditions"]:
        raise ValueError("Condition must be original or dedup_v1.")
    inputs = _load_inputs(root, config)
    torch.set_num_threads(4)
    conditions = config["conditions"] if condition is None else [condition]
    with threadpool_limits(limits=4):
        receipts = {name: _run_condition(root, path, config, inputs, name, progress, stop_event) for name in conditions}
    return receipts if condition is None else receipts[condition]
