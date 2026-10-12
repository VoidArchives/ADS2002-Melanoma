"""Explicit-metric metadata tuning using verified, immutable v1 predictions.

V2 changes candidate selection and reporting, not the training/preprocessing
protocol. Only absent clean classical candidates and unmatched selected outer
procedures are fitted. Reserved-test and full-development fitting are absent.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits
import torch
import yaml

from . import metadata_nn_tuning as v1
from .dedup_metadata import NN_SPECIFICATION, fit_network, neural_preprocessor, predict_network
from .dedup_metadata_suite import (FEATURES, GRIDS, KEYS, categorical_frame, inner_splits_from_memberships,
                                   metrics_table, model_setups, verify_suite_receipt)
from .dedup_metadata import verify_saved_receipt
from .development_dataset import CANONICAL_CSV, load_development_dataset, stable_sha256
from .metadata_evaluation import validate_probabilities
from .metadata_model import file_sha256
from .metadata_validation import validate_metadata_frame
from .metadata_roc_archive import ARCHIVE, verified_source_path

EXPERIMENT_VERSION = 2
DEFAULT_RUN_ID = "roc_auc_grid36_v1"
BASE_SEEDS = v1.BASE_SEEDS
OOF_COLUMNS = v1.OOF_COLUMNS
CONDITION_PREFIXES = {
    "original": "results/metadata_nn_tuning/v2/original",
    "dedup_v1": "results/dedup_v1/metadata_nn_tuning/v2",
}
SOURCE_FILES = (*v1.SOURCE_FILES, "src/melanoma_detection/metadata_tuning_v2.py",
                "scripts/run_metadata_tuning_v2.py", "src/melanoma_detection/metadata_roc_archive.py",
                "src/melanoma_detection/historical_reporting_source.py")
METRICS = ("roc_auc", "average_precision")
GRID_FIELDS = ("recipe_id", "recipe_order", "hidden_units", "learning_rate", "dropout")
recipes = v1.recipes
inner_partition = v1.inner_partition
outer_partition = v1.outer_partition
verify_prediction_coverage = v1.verify_prediction_coverage


def _contained(root, value, prefix):
    return v1._contained(root, value, prefix)


def condition_directory(root, condition, run_id=DEFAULT_RUN_ID, *, experiment_version=2):
    if type(experiment_version) is not int or experiment_version != 2 or condition not in CONDITION_PREFIXES:
        raise ValueError("An explicit v2 original or dedup_v1 condition is required.")
    if not isinstance(run_id, str) or not 1 <= len(run_id) <= 64 or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in run_id
    ):
        raise ValueError("A simple run identity of at most 64 characters is required.")
    root = Path(root).resolve()
    return _contained(root, f"{CONDITION_PREFIXES[condition]}/{run_id}", CONDITION_PREFIXES[condition])


def ensemble_score(target, seed_probabilities, metric="roc_auc"):
    if metric not in METRICS:
        raise ValueError("The selection metric must be roc_auc or average_precision.")
    matrix = np.asarray(seed_probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("A three-seed probability matrix is required.")
    for probability in matrix.T:
        validate_probabilities(target, probability)
    mean = matrix.mean(axis=1)
    target, mean = validate_probabilities(target, mean)
    if np.unique(target).size != 2:
        raise ValueError("Candidate assessment requires both classes.")
    scorer = roc_auc_score if metric == "roc_auc" else average_precision_score
    return mean, float(scorer(target, mean))


def select_candidate(table, selection_metric="roc_auc", *, order_column="recipe_order"):
    if selection_metric not in METRICS:
        raise ValueError("Unknown candidate selection metric.")
    score = f"mean_inner_{selection_metric}"
    if table.empty or not {score, order_column}.issubset(table):
        raise ValueError("Candidate scores and a declared tie order are required.")
    if table[order_column].duplicated().any():
        raise ValueError("Candidate order must be unique.")
    values = table[score].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Every candidate requires a finite score in [0, 1].")
    winner = table.sort_values([score, order_column], ascending=[False, True], kind="stable").iloc[0].to_dict()
    ordered = sorted(values, reverse=True)
    winner.update(selection_metric=selection_metric, selection_score=float(winner[score]),
                  selection_margin=float(ordered[0] - ordered[1]) if len(ordered) > 1 else 0.0)
    return winner


def _exact(actual, expected, message):
    try:
        assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                           check_dtype=False, check_exact=True)
    except AssertionError as error:
        raise ValueError(message) from error


def verify_receipt(root, receipt, *, compatibility_sha256=None, output_root=None,
                   experiment_version=2, expected_selection_metric=None):
    root = Path(root).resolve()
    if (receipt.get("status") != "complete" or type(receipt.get("experiment_version")) is not int
        or receipt.get("experiment_version") != experiment_version or type(experiment_version) is not int or experiment_version != 2):
        raise ValueError("A complete explicitly versioned v2 receipt is required.")
    metric = receipt.get("selection_metric")
    if metric not in METRICS or (expected_selection_metric is not None and metric != expected_selection_metric):
        raise ValueError("The receipt selection metric differs from the requested experiment.")
    if compatibility_sha256 is not None and receipt.get("compatibility_sha256") != compatibility_sha256:
        raise ValueError("V2 source, input, parent, runtime or protocol changed; use a new run.")
    condition = receipt.get("condition")
    if condition not in CONDITION_PREFIXES:
        raise ValueError("Unknown v2 condition.")
    directory = _contained(root, receipt.get("output_root", ""), CONDITION_PREFIXES[condition])
    if output_root is not None and directory != Path(output_root).resolve():
        raise ValueError("V2 receipt output root differs from the requested run.")
    hashes = receipt.get("output_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("V2 receipt lacks output fingerprints.")
    for relative, digest in hashes.items():
        source = _contained(root, relative, directory)
        if not source.is_file() or file_sha256(source) != digest:
            raise ValueError(f"V2 output is missing or changed: {relative}")
    for relative, digest in receipt.get("external_source_sha256", {}).items():
        source = _contained(root, relative, root / "results")
        if not source.is_file() or file_sha256(source) != digest:
            raise ValueError(f"Verified external evidence changed: {relative}")
    for relative, digest in receipt.get("parent_receipt_sha256", {}).items():
        path = _contained(root, relative, root / "results")
        if file_sha256(path) != digest:
            raise ValueError("The immutable AP parent receipt changed.")
        parent = json.loads(path.read_text(encoding="utf-8"))
        v1.verify_receipt(root, parent, output_root=path.parent)
    return receipt


def load_tuning_outputs(root, condition, *, experiment_version=2, run_id=DEFAULT_RUN_ID,
                        expected_selection_metric="roc_auc"):
    if expected_selection_metric not in METRICS:
        raise ValueError("The expected selection metric must be explicit.")
    root = Path(root).resolve()
    directory = condition_directory(root, condition, run_id, experiment_version=experiment_version)
    completion = directory / "receipt.json"
    if not completion.exists():
        return {"status": "pending", "directory": directory, "receipt": None, "oof": None}
    receipt = json.loads(completion.read_text(encoding="utf-8"))
    verify_receipt(root, receipt, output_root=directory, experiment_version=experiment_version,
                   expected_selection_metric=expected_selection_metric)
    if receipt.get("condition") != condition or receipt.get("run_id") != run_id:
        raise ValueError("V2 receipt does not identify the requested condition/run.")
    if receipt.get("test_set_loaded") is not False or receipt.get("final_model_fitted") is not False:
        raise ValueError("Only development-only, no-final-fit evidence may be loaded.")
    manifest = receipt["dataset_identity"].get("dataset_manifest") if condition == "dedup_v1" else None
    frame, identity = load_development_dataset(root, manifest)
    if identity != receipt["dataset_identity"]:
        raise ValueError("V2 development identity changed.")
    oof = pd.read_csv(directory / "oof.csv", float_precision="round_trip")
    v1.verify_prediction_coverage(frame, oof)
    return {"status": "complete", "directory": directory, "receipt": receipt, "oof": oof}


def _phase(root, folder, compatibility, context, compute, stop_event=None):
    """Seal complete v2 phases; reject incompatible resumes before computing."""
    if folder is None:
        v1._check_stop(stop_event)
        return compute(), False
    root, folder = Path(root).resolve(), Path(folder).resolve()
    output = _contained(root, context["output_root"], CONDITION_PREFIXES[context["condition"]])
    _contained(root, folder, output)
    completion = folder / "receipt.json"
    if completion.exists():
        receipt = json.loads(completion.read_text(encoding="utf-8"))
        verify_receipt(root, receipt, compatibility_sha256=compatibility, output_root=output,
                       expected_selection_metric=context["selection_metric"])
        if receipt.get("context") != context:
            raise ValueError("Saved v2 phase context changed.")
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        for name in receipt.get("csv_names", []):
            result[name] = pd.read_csv(folder / f"{name}.csv", float_precision="round_trip")
        if context.get("model") == "nn" and "predictions" in result:
            result["predictions"]["score"] = result["predictions"].score.astype(np.float32)
        return result, True
    v1._check_stop(stop_event)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result, serializable, csv_names, files = compute(), {}, [], []
    for name, value in result.items():
        if isinstance(value, pd.DataFrame):
            v1._csv(folder / f"{name}.csv", value)
            csv_names.append(name)
            files.append(folder / f"{name}.csv")
        elif name == "model":
            v1._joblib(folder / "model.joblib", value)
            files.append(folder / "model.joblib")
        else:
            serializable[name] = value
    serializable["elapsed_seconds"] = time.monotonic() - started
    v1._json(folder / "result.json", serializable)
    files.append(folder / "result.json")
    v1._json(completion, {"schema_version": 2, "experiment_version": 2, "status": "complete",
             "condition": context["condition"], "selection_metric": context["selection_metric"],
             "output_root": context["output_root"], "compatibility_sha256": compatibility,
             "context": context, "csv_names": csv_names,
             "output_sha256": {v1._relative(root, path): file_sha256(path) for path in files}})
    result["elapsed_seconds"] = serializable["elapsed_seconds"]
    return result, False


def _log(context, phase, *, external=False, resumed=False, source=None, scoring=False, elapsed=0.0):
    return {**context, "phase": phase, "provenance": "external_reused" if external else "resumed" if resumed else "executed",
            "fits": int(not external and not scoring), "executed_fits": int(not external and not scoring),
            "external_reused_fits": int(external and not scoring), "reused_fits": int(external and not scoring),
            "resumed_fits": int(resumed and not scoring),
            "executed_this_invocation": int(not external and not resumed and not scoring),
            "rescoring_operations": int(scoring), "elapsed_seconds": elapsed,
            "source_receipt": source or ""}


def fit_nn_pair(partitions, *, base_seed, outer_fold, spec=None, context=None,
                project_root=None, output_dir=None, compatibility_sha256=None, stop_event=None):
    """BCE stopping and fresh train-only refit, with independent v2 phase seals."""
    v1._validate_partition(partitions)
    spec = deepcopy(NN_SPECIFICATION if spec is None else spec)
    torch.set_num_threads(spec["threads"])
    context = {} if context is None else dict(context)
    record = {**context, "model": "nn", "outer_fold": int(outer_fold), "base_seed": int(base_seed),
              "training_seed": int(base_seed) + int(outer_fold), "stop_group": int(partitions["stop_group"])}
    if output_dir is not None and (project_root is None or compatibility_sha256 is None):
        raise ValueError("Checkpointed v2 fitting requires a project root and frozen digest.")
    parts = {name: v1._numeric_frame(partitions[name]) for name in ("epoch_fit", "epoch_stop", "refit", "score")}
    identity = stable_sha256({"context": record, "spec": spec,
                             "image_ids": {name: part.image_name.tolist() for name, part in parts.items()}})
    digest = stable_sha256({"run": compatibility_sha256, "trial": identity})
    folder = None if output_dir is None else Path(output_dir)

    def stopping_fit():
        prep = neural_preprocessor().fit(parts["epoch_fit"][FEATURES])
        _, selected, loss, history = fit_network(prep.transform(parts["epoch_fit"][FEATURES]), parts["epoch_fit"].target,
            record["training_seed"], x_stop=prep.transform(parts["epoch_stop"][FEATURES]),
            y_stop=parts["epoch_stop"].target, spec=spec)
        return {"selected_epochs": int(selected), "stop_log_loss": float(loss),
                "attempted_stopping_epochs": len(history), "history": pd.DataFrame(history)}

    stopping, reused_stop = _phase(project_root, None if folder is None else folder / "stopping", digest,
        {**record, "phase": "stopping"}, stopping_fit, stop_event)

    def refit_fit():
        prep = neural_preprocessor().fit(parts["refit"][FEATURES])
        encoded = prep.transform(parts["refit"][FEATURES])
        model, _, _, history = fit_network(encoded, parts["refit"].target, record["training_seed"],
            epochs=stopping["selected_epochs"], spec=spec)
        probability = predict_network(model, prep.transform(parts["score"][FEATURES]))
        validate_probabilities(parts["score"].target, probability)
        predictions = parts["score"][KEYS].copy()
        predictions["score"] = probability
        return {"predictions": predictions, "history": pd.DataFrame(history), "encoded_input_width": int(encoded.shape[1]),
                "model": {"preprocessor": prep, "state_dict": model.state_dict(), "spec": spec,
                          "training_seed": record["training_seed"], "selected_epochs": stopping["selected_epochs"],
                          "fit_image_ids": parts["refit"].image_name.tolist(),
                          "fit_patient_ids": sorted(parts["refit"].patient_id.unique())}}

    refit, reused_refit = _phase(project_root, None if folder is None else folder / "refit", digest,
        {**record, "phase": "refit", "selected_epochs": stopping["selected_epochs"]}, refit_fit, stop_event)
    epoch = {**record, "selected_epochs": stopping["selected_epochs"], "stop_log_loss": stopping["stop_log_loss"],
             "attempted_stopping_epochs": stopping["attempted_stopping_epochs"], "encoded_input_width": refit["encoded_input_width"],
             "provenance": "resumed" if reused_stop and reused_refit else "executed"}
    for name, part in parts.items():
        epoch[f"{name}_images"], epoch[f"{name}_patients"] = len(part), int(part.patient_id.nunique())
    histories, logs = [], []
    for phase, result, resumed in (("stopping", stopping, reused_stop), ("refit", refit, reused_refit)):
        history = result["history"].copy()
        for name, value in {**record, "phase": phase, "provenance": "resumed" if resumed else "executed"}.items():
            history[name] = value
        histories.append(history)
        logs.append(_log(record, phase, resumed=resumed, elapsed=result.get("elapsed_seconds", 0)))
    return {"predictions": refit["predictions"], "epoch": epoch,
            "history": pd.concat(histories, ignore_index=True), "fit_log": logs}


def _read_config(root, config_path):
    path = _contained(root, config_path, root)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"experiment_version", "run_id", "selection_metric", "conditions", "ap_parent_run_ids", "dataset_manifest",
                "metadata_suite_receipt", "clean_comparator_receipt", "threads", "outer_reuse", "fixed_baselines", "missing_candidate_policy"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Unknown or missing v2 configuration fields.")
    condition_directory(root, "original", config["run_id"], experiment_version=config["experiment_version"])
    if (config["selection_metric"] not in METRICS or config["conditions"] != ["original", "dedup_v1"] or config["threads"] != 4
        or config["outer_reuse"] != "matching_ap_selected" or config["fixed_baselines"] != "verified_ap_oof"
        or config["missing_candidate_policy"] != "clean_classical_only" or set(config["ap_parent_run_ids"]) != set(CONDITION_PREFIXES)):
        raise ValueError("V2 requires the prescribed grids, both conditions, strict reuse and four CPU threads.")
    for condition, run_id in config["ap_parent_run_ids"].items():
        v1.condition_directory(root, condition, run_id)
    return path, config


def _verify_parent_sources(root, receipt):
    if receipt.get("protocol", {}).get("neural_training") != NN_SPECIFICATION or receipt["protocol"].get("recipes") != v1.recipes():
        raise ValueError("AP parent NN training protocol differs from the fixed v2 protocol.")
    if receipt["protocol"].get("classical_grids") != json.loads(json.dumps(GRIDS)):
        raise ValueError("AP parent classical grid differs from the v2 grid.")
    if receipt["protocol"].get("candidate_selection") != "unweighted mean of three inner ensemble average-precision scores":
        raise ValueError("Expected an explicitly AP-selected parent experiment.")
    if receipt["versions"] != v1._versions():
        raise ValueError("AP parent runtime differs from the current fitting runtime.")
    if file_sha256(root / "configs/metadata_nn_tuning.yaml") != receipt["config_sha256"]:
        raise ValueError("The immutable AP configuration changed.")
    for relative, digest in receipt["source_sha256"].items():
        if file_sha256(root / relative) != digest:
            raise ValueError(f"Frozen AP fitting dependency changed: {relative}")


def _load_inputs(root, config):
    preservation = json.loads((root / ARCHIVE / "preservation_receipt.json").read_text(encoding="utf-8"))
    if preservation.get("status") != "complete":
        raise ValueError("Complete pre-ROC preservation is required before reading parents or fitting.")
    original, original_identity = load_development_dataset(root)
    clean, clean_identity = load_development_dataset(root, config["dataset_manifest"])
    if (len(original), len(clean)) != (6864, 6706):
        raise ValueError("The approved development populations changed.")
    suite_path = _contained(root, config["metadata_suite_receipt"], "results/dedup_v1/metadata_suite")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    verify_suite_receipt(root, suite)
    if suite["dataset_identity"] != clean_identity:
        raise ValueError("Metadata suite and clean development identity differ.")
    for relative, digest in suite["source_sha256"].items():
        verified_source_path(root, relative, digest, parent_receipt=config["metadata_suite_receipt"])
    memberships = {"original": pd.read_csv(suite_path.parent / "inner_memberships_original.csv"),
                   "dedup_v1": pd.read_csv(suite_path.parent / "inner_memberships_retained.csv")}
    keys = ["outer_fold", "image_name"]
    retained = memberships["original"].loc[memberships["original"].image_name.isin(clean.image_name)]
    _exact(retained.sort_values(keys), memberships["dedup_v1"].sort_values(keys),
           "Clean memberships must be literal retained-ID filtering of saved original metadata memberships.")
    for frame, membership in ((original, memberships["original"]), (clean, memberships["dedup_v1"])):
        for outer in range(5):
            inner_splits_from_memberships(frame, outer, membership)
    comparator_path = _contained(root, config["clean_comparator_receipt"], "results/dedup_v1/metadata")
    comparator = json.loads(comparator_path.read_text(encoding="utf-8"))
    verify_saved_receipt(root, comparator)
    if comparator.get("dataset_identity") != clean_identity or comparator["specification"]["neural_network"] != NN_SPECIFICATION:
        raise ValueError("Clean fixed NN identity/protocol differs.")
    for relative, digest in comparator["source_sha256"].items():
        verified_source_path(root, relative, digest, parent_receipt=config["clean_comparator_receipt"])
    raw = validate_metadata_frame(pd.read_csv(root / CANONICAL_CSV, dtype={"age_approx": "string"}))
    frames, categorical = {"original": original, "dedup_v1": clean}, {}
    for condition, frame in frames.items():
        copied = frame.copy()
        copied["age_approx"] = raw.set_index("image_name").loc[frame.image_name, "age_approx"].to_numpy()
        categorical[condition] = categorical_frame(copied)
    inputs = {"frames": frames, "identities": {"original": original_identity, "dedup_v1": clean_identity},
              "memberships": memberships, "categorical": categorical, "suite": suite, "comparator": comparator,
              "reuse_identity": {"metadata_suite_receipt_sha256": file_sha256(suite_path),
                                 "clean_comparator_receipt_sha256": file_sha256(comparator_path),
                                 "metadata_original_memberships_sha256": file_sha256(suite_path.parent / "inner_memberships_original.csv"),
                                 "metadata_clean_memberships_sha256": file_sha256(suite_path.parent / "inner_memberships_retained.csv")}}
    parents, parent_hashes = {}, {}
    for condition in CONDITION_PREFIXES:
        parent = v1.load_tuning_outputs(root, condition, run_id=config["ap_parent_run_ids"][condition])
        if parent["status"] != "complete":
            raise ValueError("Both complete AP parent experiments are required before v2 fitting.")
        parent_relative = v1._relative(root, parent["directory"] / "receipt.json")
        if preservation["protected_files"].get(parent_relative) != file_sha256(parent["directory"] / "receipt.json"):
            raise ValueError("The AP parent receipt is changed or lacks its pre-ROC preservation binding.")
        _verify_parent_sources(root, parent["receipt"])
        if parent["receipt"]["dataset_identity"] != inputs["identities"][condition]:
            raise ValueError("AP parent and v2 development identities differ.")
        if parent["receipt"]["reuse_identity"] != inputs["reuse_identity"]:
            raise ValueError("AP parent legacy suite/comparator/membership bindings changed.")
        membership = pd.read_csv(parent["directory"] / "inner_memberships.csv")
        columns = ["outer_fold", "image_name"]
        _exact(membership.sort_values(columns), inputs["memberships"][condition].sort_values(columns),
               "AP parent inner membership differs from the saved metadata-suite groups.")
        for filename in ("selected_parameters", "classical_selected_parameters", "epochs", "histories", "memberships"):
            parent[filename] = pd.read_csv(parent["directory"] / f"{filename}.csv", float_precision="round_trip", low_memory=False)
        parents[condition] = parent
        parent_hashes[v1._relative(root, parent["directory"] / "receipt.json")] = file_sha256(parent["directory"] / "receipt.json")
    inputs.update(parents=parents, parent_receipt_sha256=parent_hashes)
    return inputs


def outer_nn_reuse_route(winner, ap_selected):
    if all(winner[field] == ap_selected[field] for field in GRID_FIELDS):
        return "selected"
    if all(winner[field] == v1.recipes()[0][field] for field in GRID_FIELDS):
        return "fixed_nn"
    return None


def _reused_outer_nn(root, parent, winner, outer, seed, seed_index, parts, condition):
    old_selected = parent["selected_parameters"].loc[parent["selected_parameters"].outer_fold.eq(outer)].iloc[0]
    route = outer_nn_reuse_route(winner, old_selected)
    if route is None:
        raise ValueError("The selected NN has no exact matching outer source.")
    source_stage = "nn_outer_tuned" if route == "selected" else "nn_outer_fixed"
    expected = {"condition": condition, "outer_fold": outer, **{field: winner[field] for field in GRID_FIELDS},
                "stage": source_stage, "inner_fold": -1, "base_seed": seed,
                "training_seed": seed + outer, "stop_group": parts["stop_group"], "phase": "refit"}
    if condition == "original" or route == "selected":
        folder = parent["directory"] / "trials" / f"outer_{outer}" / route / f"seed_{seed}"
        prediction, reference = _cached_prediction(root, parent, folder / "refit/predictions.csv", parts["score"], nn=True,
                                                   context_fields=expected)
        phase_sources = []
        for phase in ("stopping", "refit"):
            source = folder / phase / "receipt.json"
            recorded = json.loads(source.read_text(encoding="utf-8"))
            v1.verify_receipt(root, recorded, output_root=parent["directory"])
            fields = {**expected, "phase": phase}
            if any(recorded["context"].get(key) != value for key, value in fields.items()):
                raise ValueError("Matching NN stopping/refit context changed.")
            phase_sources.append(v1._relative(root, source))
    else:
        # The clean fixed NN is already bound through the exact v1 source,
        # comparator receipt, NN specification and complete per-seed diagnostics.
        source = parent["directory"] / "receipt.json"
        rows = parent["oof"].set_index("image_name").loc[parts["score"].image_name]
        prediction = parts["score"][KEYS].copy()
        prediction["score"] = rows[f"p_nn_seed_{seed}"].to_numpy().astype(np.float32)
        reference = {"source_prediction": v1._relative(root, parent["directory"] / "oof.csv"),
                     "source_prediction_sha256": file_sha256(parent["directory"] / "oof.csv"),
                     "source_receipt": v1._relative(root, source), "source_receipt_sha256": file_sha256(source),
                     "provenance": "external_reused", "source_stage": source_stage, "source_probability_column": f"p_nn_seed_{seed}"}
        phase_sources = [reference["source_receipt"]] * 2
    epoch = parent["epochs"].loc[parent["epochs"].stage.eq(source_stage) & parent["epochs"].outer_fold.eq(outer)
                                 & parent["epochs"].base_seed.eq(seed)].copy()
    history = parent["histories"].loc[parent["histories"].stage.eq(source_stage) & parent["histories"].outer_fold.eq(outer)
                                      & parent["histories"].base_seed.eq(seed)].copy()
    if len(epoch) != 1 or set(history.phase) != {"stopping", "refit"}:
        raise ValueError("Matching NN outer epoch/history coverage is incomplete.")
    recorded = epoch.iloc[0]
    if (int(recorded.training_seed) != seed + outer or int(recorded.stop_group) != (outer + 1 + seed_index) % 5
        or int(recorded.refit_images) != len(parts["refit"]) or int(recorded.score_images) != len(parts["score"])):
        raise ValueError("Matching NN outer training membership/seed diagnostics changed.")
    return prediction, reference, phase_sources, epoch, history


def _cached_prediction(root, parent, path, expected, *, context_fields=None, nn=False):
    path = _contained(root, path, parent["directory"])
    completion = path.parent / "receipt.json"
    receipt = json.loads(completion.read_text(encoding="utf-8"))
    v1.verify_receipt(root, receipt, output_root=parent["directory"])
    if context_fields is not None and any(receipt["context"].get(key) != value for key, value in context_fields.items()):
        raise ValueError("Cached AP phase context differs from the requested candidate.")
    relative = v1._relative(root, path)
    if relative not in parent["receipt"]["output_sha256"] or file_sha256(path) != parent["receipt"]["output_sha256"][relative]:
        raise ValueError("Cached candidate predictions are not bound to their complete AP parent.")
    predictions = pd.read_csv(path, float_precision="round_trip")
    _exact(predictions[KEYS].sort_values("image_name"), expected[KEYS].sort_values("image_name"),
           "Cached candidate identity differs from its exact held-out metadata membership.")
    validate_probabilities(predictions.target, predictions.score)
    predictions = predictions.set_index("image_name").loc[expected.image_name].reset_index()
    if nn:
        predictions["score"] = predictions.score.astype(np.float32)
    reference = {"source_prediction": relative, "source_prediction_sha256": file_sha256(path),
                 "source_receipt": v1._relative(root, completion), "source_receipt_sha256": file_sha256(completion),
                 "provenance": "external_reused"}
    return predictions, reference


def _score_row(context, scores, metric):
    row = {**context, "selection_metric": metric}
    for name in METRICS:
        values = [result[name] for result in scores]
        row.update({f"inner_{index}_{name}": value for index, value in enumerate(values)})
        row[f"mean_inner_{name}"], row[f"std_inner_{name}"] = float(np.mean(values)), float(np.std(values))
    row["selection_score"] = row[f"mean_inner_{metric}"]
    return row


def _classical_candidates(root, output, frame, membership, outer, parent, condition, digest, metric, stop_event):
    fitting = frame.loc[frame.fold.ne(outer)].reset_index(drop=True)
    splits, audit = inner_splits_from_memberships(frame, outer, membership)
    candidates, references, logs = [], [], []
    setups = model_setups(4)
    base = {"condition": condition, "output_root": v1._relative(root, output), "selection_metric": metric, "outer_fold": outer}
    for name, estimator in setups.items():
        for index, parameters in enumerate(ParameterGrid(GRIDS[name])):
            context = {**base, "model": name, "candidate_index": index, "parameters": parameters, "stage": "classical_inner"}
            scores = []
            for inner, (fit_ids, val_ids) in enumerate(splits):
                ctx = {**context, "inner_fold": inner}
                expected = fitting.iloc[val_ids]
                if condition == "original":
                    path = parent["directory"] / "classical" / f"outer_{outer}" / name / f"candidate_{index:02d}" / f"inner_{inner}" / "predictions.csv"
                    prediction, reference = _cached_prediction(root, parent, path, expected, context_fields={
                        "condition": condition, "outer_fold": outer, "model": name, "candidate_index": index,
                        "inner_fold": inner, "parameters": parameters, "phase": "classical_inner"})
                    logs.append(_log({**ctx, "parameters": json.dumps(parameters, sort_keys=True)}, "classical_inner", external=True,
                                     source=reference["source_receipt"]))
                else:
                    folder = output / "classical" / f"outer_{outer}" / name / f"candidate_{index:02d}" / f"inner_{inner}"
                    def compute(est=estimator, params=parameters, train=fitting.iloc[fit_ids], heldout=expected):
                        model = clone(est).set_params(**params).fit(train[FEATURES], train.target)
                        probability = model.predict_proba(heldout[FEATURES])[:, list(model.classes_).index(1)]
                        validate_probabilities(heldout.target, probability)
                        prediction = heldout[KEYS].copy()
                        prediction["score"] = probability
                        return {"predictions": prediction}
                    result, resumed = _phase(root, folder, digest, {**ctx, "phase": "classical_inner"}, compute, stop_event)
                    prediction = result["predictions"]
                    reference = {"source_prediction": v1._relative(root, folder / "predictions.csv"),
                                 "source_prediction_sha256": file_sha256(folder / "predictions.csv"),
                                 "source_receipt": v1._relative(root, folder / "receipt.json"),
                                 "source_receipt_sha256": file_sha256(folder / "receipt.json"),
                                 "provenance": "resumed" if resumed else "executed"}
                    logs.append(_log({**ctx, "parameters": json.dumps(parameters, sort_keys=True)}, "classical_inner",
                                     resumed=resumed, source=reference["source_receipt"], elapsed=result.get("elapsed_seconds", 0)))
                references.append({**ctx, "parameters": json.dumps(parameters, sort_keys=True), **reference})
                scores.append({"roc_auc": float(roc_auc_score(prediction.target, prediction.score)),
                               "average_precision": float(average_precision_score(prediction.target, prediction.score))})
                logs.append(_log({**ctx, "parameters": json.dumps(parameters, sort_keys=True)}, "candidate_scoring",
                                 scoring=True, source=reference["source_receipt"]))
            candidates.append(_score_row({**context, "parameters": json.dumps(parameters, sort_keys=True)}, scores, metric))
    selected = []
    for name in setups:
        table = pd.DataFrame([row for row in candidates if row["model"] == name])
        winner = select_candidate(table, metric, order_column="candidate_index")
        selected.append(winner)
        for row in candidates:
            if row["model"] == name:
                row["selected"] = row["candidate_index"] == winner["candidate_index"]
    return candidates, selected, references, logs, audit


def _protocol(metric):
    protocol = deepcopy(v1._protocol())
    protocol.update(experiment_version=2, selection_metric=metric,
                    candidate_selection=f"unweighted mean of three inner {metric} scores; NN probabilities averaged before scoring",
                    primary_metric=f"unweighted mean outer-fold {metric}", secondary_metric="average_precision" if metric == "roc_auc" else "roc_auc",
                    fixed_baselines="verified v1 AP parent OOF",
                    outer_reuse="exact matching AP-selected procedures; fixed NN only with identical protocol and per-seed diagnostics")
    return protocol


def _compatibility(root, config_path, config, inputs, condition):
    value = {"experiment_version": 2, "condition": condition, "selection_metric": config["selection_metric"],
             "dataset_identity": inputs["identities"][condition], "protocol": _protocol(config["selection_metric"]),
             "source_sha256": {relative: file_sha256(root / relative) for relative in SOURCE_FILES},
             "config_sha256": file_sha256(config_path), "versions": v1._versions(),
             "parent_receipt_sha256": inputs["parent_receipt_sha256"], "reuse_identity": inputs["reuse_identity"]}
    return value, stable_sha256(value)


def plan_run(root, config_path="configs/metadata_nn_tuning_roc_auc.yaml"):
    """Verify parents and inputs without creating artifacts or fitting models."""
    root = Path(root).resolve()
    path, config = _read_config(root, config_path)
    inputs = _load_inputs(root, config)
    cache = verify_candidate_cache(root, inputs, config["selection_metric"])
    return {"status": "verified_plan", "experiment_version": 2, "selection_metric": config["selection_metric"],
            "run_id": config["run_id"], "conditions": {condition: {"rows": len(inputs["frames"][condition]),
                "nn_cached_fit_phases": 3240, "nn_cached_prediction_files": 1620,
                "classical_cached_prediction_files": 630 if condition == "original" else 0,
                "missing_classical_inner_fits": 0 if condition == "original" else 630,
                "maximum_new_fit_calls": 45 if condition == "original" else 675}
                for condition in CONDITION_PREFIXES}, "maximum_new_fit_calls": 720,
            "parent_receipt_sha256": inputs["parent_receipt_sha256"], "cache_checks": cache,
            "test_set_loaded": False, "final_model_fitted": False}


def verify_candidate_cache(root, inputs, metric="roc_auc"):
    """Cold-check every available candidate and every matching outer NN source."""
    results = {}
    for condition in CONDITION_PREFIXES:
        frame, membership, parent = inputs["frames"][condition], inputs["memberships"][condition], inputs["parents"][condition]
        nn_selected, classical_selected, prediction_files = [], [], 0
        for outer in range(5):
            partitions = {(inner, seed_index): v1.inner_partition(frame, membership, outer, inner, seed_index)
                          for inner in range(3) for seed_index in range(3)}
            candidates = []
            for recipe in v1.recipes():
                scores = []
                for inner in range(3):
                    vectors = []
                    for seed_index, seed in enumerate(BASE_SEEDS):
                        parts = partitions[(inner, seed_index)]
                        path = parent["directory"] / "trials" / f"outer_{outer}" / recipe["recipe_id"] / f"inner_{inner}" / f"seed_{seed}" / "refit/predictions.csv"
                        prediction, _ = _cached_prediction(root, parent, path, parts["score"], nn=True, context_fields={
                            "condition": condition, "outer_fold": outer, **recipe, "stage": "nn_inner", "inner_fold": inner,
                            "base_seed": seed, "training_seed": seed + outer, "stop_group": parts["stop_group"], "phase": "refit"})
                        vectors.append(prediction.score.to_numpy())
                        prediction_files += 1
                    scores.append({name: ensemble_score(parts["score"].target, np.column_stack(vectors), name)[1] for name in METRICS})
                candidates.append(_score_row({"outer_fold": outer, **recipe}, scores, metric))
            chosen = select_candidate(pd.DataFrame(candidates), metric)
            old = parent["selected_parameters"].loc[parent["selected_parameters"].outer_fold.eq(outer)].iloc[0]
            route = outer_nn_reuse_route(chosen, old)
            if route:
                for seed_index, seed in enumerate(BASE_SEEDS):
                    _reused_outer_nn(root, parent, chosen, outer, seed, seed_index, v1.outer_partition(frame, outer, seed_index), condition)
            nn_selected.append({"outer_fold": outer, "recipe_id": chosen["recipe_id"], "selection_score": chosen["selection_score"],
                                "outer_reuse": route or "fresh"})
            if condition == "original":
                fitting = inputs["categorical"][condition].loc[frame.fold.ne(outer)].reset_index(drop=True)
                splits, _ = inner_splits_from_memberships(frame, outer, membership)
                for name in GRIDS:
                    rows = []
                    for index, parameters in enumerate(ParameterGrid(GRIDS[name])):
                        scores = []
                        for inner, (_, val_ids) in enumerate(splits):
                            prediction, _ = _cached_prediction(root, parent,
                                parent["directory"] / "classical" / f"outer_{outer}" / name / f"candidate_{index:02d}" / f"inner_{inner}/predictions.csv",
                                fitting.iloc[val_ids], context_fields={"model": name, "outer_fold": outer, "inner_fold": inner,
                                                                     "candidate_index": index, "parameters": parameters, "phase": "classical_inner"})
                            scores.append({"roc_auc": float(roc_auc_score(prediction.target,prediction.score)),
                                           "average_precision": float(average_precision_score(prediction.target,prediction.score))})
                            prediction_files += 1
                        rows.append(_score_row({"candidate_index": index, "parameters": json.dumps(parameters,sort_keys=True)},scores,metric))
                    winner=select_candidate(pd.DataFrame(rows),metric,order_column="candidate_index")
                    old=parent["classical_selected_parameters"].loc[parent["classical_selected_parameters"].outer_fold.eq(outer)
                                                                      & parent["classical_selected_parameters"].model.eq(name)].iloc[0]
                    matches=json.loads(winner["parameters"])==json.loads(old.parameters)
                    if matches:
                        _cached_prediction(root,parent,parent["directory"] / "classical" / f"outer_{outer}" / name / "selected/predictions.csv",
                            inputs["categorical"][condition].loc[frame.fold.eq(outer)],context_fields={"model":name,"outer_fold":outer,
                            "candidate_index":winner["candidate_index"],"parameters":json.loads(winner["parameters"]),"phase":"classical_tuned"})
                    classical_selected.append({"outer_fold":outer,"model":name,"candidate_index":winner["candidate_index"],"outer_reuse":"selected" if matches else "fresh"})
        results[condition]={"prediction_files_checked":prediction_files,"nn_ensembles_checked":540,
                            "nn_selected":nn_selected,"classical_selected":classical_selected,
                            "clean_classical_missing_expected":630 if condition=="dedup_v1" else 0}
    return results


def _run_condition(root, config_path, config, inputs, condition, progress, stop_event):
    metric = config["selection_metric"]
    frame, membership, parent = inputs["frames"][condition], inputs["memberships"][condition], inputs["parents"][condition]
    output = condition_directory(root, condition, config["run_id"])
    compatibility, digest = _compatibility(root, config_path, config, inputs, condition)
    output.mkdir(parents=True, exist_ok=True)
    with v1._run_lock(output):
        completion = output / "receipt.json"
        if completion.exists():
            receipt = json.loads(completion.read_text(encoding="utf-8"))
            verify_receipt(root, receipt, compatibility_sha256=digest, output_root=output, expected_selection_metric=metric)
            return receipt
        session = output / "session.json"
        if session.exists():
            if json.loads(session.read_text(encoding="utf-8"))["compatibility_sha256"] != digest:
                raise ValueError("Partial v2 inputs changed; select a new run identity.")
        elif any(path.name != "run.lock" for path in output.iterdir()):
            raise ValueError("V2 output directory has no compatible session receipt.")
        else:
            v1._json(session, {"run_id": config["run_id"], "output_root": v1._relative(root, output),
                              "compatibility_sha256": digest, "created_at_utc": datetime.now(timezone.utc).isoformat(), **compatibility})
        v1._csv(output / "inner_memberships.csv", membership)
        candidates, selected, classical_candidates, classical_selected, references, logs, oof_parts, audits, roles = [], [], [], [], [], [], [], [], []
        inherited_roles = parent["memberships"].loc[parent["memberships"].stage.isin(["nn_inner", "nn_outer_fixed"])].copy()
        inherited_roles["provenance"], inherited_roles["selection_metric"] = "external_reused", metric
        inherited_roles["source_receipt"] = v1._relative(root, parent["directory"] / "receipt.json")
        inherited_roles["output_root"] = v1._relative(root, output)
        roles.extend(inherited_roles.to_dict("records"))
        epochs = parent["epochs"].loc[parent["epochs"].stage.isin(["nn_inner", "nn_outer_fixed"])].copy()
        histories = parent["histories"].loc[parent["histories"].stage.isin(["nn_inner", "nn_outer_fixed"])].copy()
        for table in (epochs, histories):
            table["provenance"], table["selection_metric"] = "external_reused", metric
            table["source_receipt"] = v1._relative(root, parent["directory"] / "receipt.json")
        epoch_rows, history_parts = epochs.to_dict("records"), [histories]
        for outer in range(5):
            if progress:
                progress(f"{condition} v2 outer {outer + 1}/5: rescoring verified candidates using {metric}.")
            base = {"condition": condition, "outer_fold": outer, "selection_metric": metric, "output_root": v1._relative(root, output)}
            partitions = {(inner, seed_index): v1.inner_partition(frame, membership, outer, inner, seed_index)
                          for inner in range(3) for seed_index in range(3)}
            outer_rows = []
            for recipe in v1.recipes():
                inner_scores = []
                for inner in range(3):
                    parts = partitions[(inner, 0)]
                    probabilities = []
                    for seed_index, seed in enumerate(BASE_SEEDS):
                        seed_parts = partitions[(inner, seed_index)]
                        ctx = {**base, **recipe, "model": "nn", "stage": "nn_inner", "inner_fold": inner, "base_seed": seed,
                               "training_seed": seed + outer, "stop_group": seed_parts["stop_group"]}
                        folder = parent["directory"] / "trials" / f"outer_{outer}" / recipe["recipe_id"] / f"inner_{inner}" / f"seed_{seed}"
                        prediction, reference = _cached_prediction(root, parent, folder / "refit/predictions.csv", parts["score"], nn=True,
                            context_fields={"condition": condition, "outer_fold": outer, **recipe, "stage": "nn_inner",
                                            "inner_fold": inner, "base_seed": seed, "training_seed": seed + outer,
                                            "stop_group": seed_parts["stop_group"], "phase": "refit"})
                        probabilities.append(prediction.score.to_numpy())
                        references.append({**ctx, **reference})
                        for phase in ("stopping", "refit"):
                            receipt_path = folder / phase / "receipt.json"
                            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                            v1.verify_receipt(root, receipt, output_root=parent["directory"])
                            logs.append(_log(ctx, phase, external=True, source=v1._relative(root, receipt_path)))
                    matrix = np.column_stack(probabilities)
                    inner_scores.append({name: ensemble_score(parts["score"].target, matrix, name)[1] for name in METRICS})
                    logs.append(_log({**base, **recipe, "model": "nn", "stage": "nn_inner", "inner_fold": inner},
                                     "candidate_scoring", scoring=True))
                outer_rows.append(_score_row({**base, **recipe, "model": "nn"}, inner_scores, metric))
            winner = select_candidate(pd.DataFrame(outer_rows), metric)
            selected.append(winner)
            for row in outer_rows:
                row["selected"] = row["recipe_id"] == winner["recipe_id"]
            candidates.extend(outer_rows)
            scored = frame.loc[frame.fold.eq(outer), KEYS].copy()
            fixed = parent["oof"].set_index("image_name").loc[scored.image_name]
            for column in v1.FIXED_COLUMNS:
                scored[column] = fixed[column].to_numpy()
            logs.append({**_log({**base, "stage": "fixed_baselines", "model": "four_families"}, "baseline_oof_reuse",
                               external=True, source=v1._relative(root, parent["directory"] / "receipt.json")),
                         "external_reused_fits": 9, "reused_fits": 9})
            ap_winner = parent["selected_parameters"].loc[parent["selected_parameters"].outer_fold.eq(outer)].iloc[0]
            nn_route = outer_nn_reuse_route(winner, ap_winner)
            for seed_index, seed in enumerate(BASE_SEEDS):
                parts = v1.outer_partition(frame, outer, seed_index)
                ctx = {**base, **{field: winner[field] for field in GRID_FIELDS}, "model": "nn", "stage": "nn_outer_tuned", "inner_fold": -1}
                roles.extend(v1._role_rows(frame, parts, {**ctx, "base_seed": seed, "stop_group": parts["stop_group"]}))
                if nn_route:
                    prediction, reference, phase_sources, ep, hi = _reused_outer_nn(root, parent, winner, outer, seed, seed_index, parts, condition)
                    references.append({**ctx, "base_seed": seed, **reference})
                    for phase, source in zip(("stopping", "refit"), phase_sources):
                        logs.append(_log({**ctx, "base_seed": seed, "training_seed": seed + outer, "stop_group": parts["stop_group"]},
                                         phase, external=True, source=source))
                    for table in (ep, hi):
                        table["source_stage"] = table.stage
                        table["stage"], table["output_root"] = "nn_outer_tuned", base["output_root"]
                        table["provenance"], table["selection_metric"], table["source_receipt"] = "external_reused", metric, reference["source_receipt"]
                    epoch_rows.extend(ep.to_dict("records"))
                    history_parts.append(hi)
                else:
                    spec = {**NN_SPECIFICATION, **{field: winner[field] for field in ("hidden_units", "learning_rate", "dropout")}}
                    result = fit_nn_pair(parts, base_seed=seed, outer_fold=outer, spec=spec, context=ctx, project_root=root,
                        output_dir=output / "trials" / f"outer_{outer}" / "selected" / f"seed_{seed}", compatibility_sha256=digest, stop_event=stop_event)
                    prediction = result["predictions"]
                    epoch_rows.append(result["epoch"])
                    history_parts.append(result["history"])
                    logs.extend(result["fit_log"])
                    folder = output / "trials" / f"outer_{outer}" / "selected" / f"seed_{seed}" / "refit"
                    references.append({**ctx, "base_seed": seed,
                        "source_prediction": v1._relative(root, folder / "predictions.csv"),
                        "source_prediction_sha256": file_sha256(folder / "predictions.csv"),
                        "source_receipt": v1._relative(root, folder / "receipt.json"),
                        "source_receipt_sha256": file_sha256(folder / "receipt.json"),
                        "provenance": result["epoch"]["provenance"]})
                scored[f"p_nn_tuned_seed_{seed}"] = prediction.score.to_numpy()
            scored["p_nn_tuned"] = scored[[f"p_nn_tuned_seed_{seed}" for seed in BASE_SEEDS]].mean(axis=1)
            cc, cs, refs, fit_logs, audit = _classical_candidates(root, output, inputs["categorical"][condition], membership,
                outer, parent, condition, digest, metric, stop_event)
            classical_candidates.extend(cc)
            classical_selected.extend(cs)
            references.extend(refs)
            logs.extend(fit_logs)
            audits.append(audit)
            fitting = inputs["categorical"][condition].loc[frame.fold.ne(outer)].reset_index(drop=True)
            heldout = inputs["categorical"][condition].loc[frame.fold.eq(outer)].reset_index(drop=True)
            for chosen in cs:
                name, parameters = chosen["model"], json.loads(chosen["parameters"])
                old = parent["classical_selected_parameters"].loc[parent["classical_selected_parameters"].outer_fold.eq(outer) & parent["classical_selected_parameters"].model.eq(name)].iloc[0]
                ctx = {**base, "model": name, "stage": "classical_outer_tuned", "inner_fold": -1,
                       "candidate_index": chosen["candidate_index"], "parameters": parameters}
                if parameters == json.loads(old.parameters):
                    if condition == "original":
                        prediction, reference = _cached_prediction(root, parent,
                            parent["directory"] / "classical" / f"outer_{outer}" / name / "selected/predictions.csv", heldout,
                            context_fields={"condition": condition, "outer_fold": outer, "model": name,
                                            "candidate_index": int(chosen["candidate_index"]), "parameters": parameters, "phase": "classical_tuned"})
                    else:
                        prediction = heldout[KEYS].copy()
                        prediction["score"] = fixed[f"p_{name}_tuned"].to_numpy()
                        source = parent["directory"] / "receipt.json"
                        reference = {"source_prediction": v1._relative(root, parent["directory"] / "oof.csv"),
                                     "source_prediction_sha256": file_sha256(parent["directory"] / "oof.csv"),
                                     "source_receipt": v1._relative(root, source), "source_receipt_sha256": file_sha256(source),
                                     "provenance": "external_reused", "legacy_suite_receipt": config["metadata_suite_receipt"],
                                     "source_probability_column": f"p_{name}_tuned"}
                    references.append({**ctx, "parameters": json.dumps(parameters, sort_keys=True), **reference})
                    logs.append(_log({**ctx, "parameters": json.dumps(parameters, sort_keys=True)}, "classical_tuned", external=True,
                                     source=reference["source_receipt"]))
                else:
                    folder = output / "classical" / f"outer_{outer}" / name / "selected"
                    def compute(model_name=name, params=parameters):
                        model = clone(model_setups(4)[model_name]).set_params(**params).fit(fitting[FEATURES], fitting.target)
                        probability = model.predict_proba(heldout[FEATURES])[:, list(model.classes_).index(1)]
                        prediction = heldout[KEYS].copy()
                        prediction["score"] = probability
                        validate_probabilities(heldout.target, probability)
                        return {"predictions": prediction, "model": model}
                    result, resumed = _phase(root, folder, digest, {**ctx, "phase": "classical_tuned"}, compute, stop_event)
                    prediction = result["predictions"]
                    logs.append(_log({**ctx, "parameters": json.dumps(parameters, sort_keys=True)}, "classical_tuned",
                                     resumed=resumed, elapsed=result.get("elapsed_seconds", 0)))
                    references.append({**ctx, "parameters": json.dumps(parameters, sort_keys=True),
                        "source_prediction": v1._relative(root, folder / "predictions.csv"),
                        "source_prediction_sha256": file_sha256(folder / "predictions.csv"),
                        "source_receipt": v1._relative(root, folder / "receipt.json"),
                        "source_receipt_sha256": file_sha256(folder / "receipt.json"),
                        "provenance": "resumed" if resumed else "executed"})
                scored[f"p_{name}_tuned"] = prediction.score.to_numpy()
            v1.verify_prediction_coverage(frame, scored, fold=outer)
            oof_parts.append(scored[OOF_COLUMNS])
            for name, rows in (("candidates", candidates), ("selected_parameters", selected),
                               ("classical_candidates", classical_candidates), ("classical_selected_parameters", classical_selected),
                               ("inner_prediction_references", references), ("fit_log", logs), ("epochs", epoch_rows), ("memberships", roles)):
                v1._csv(output / f"{name}.csv", pd.DataFrame(rows))
            v1._csv(output / "histories.csv", pd.concat(history_parts, ignore_index=True))
            v1._csv(output / "classical_inner_audit.csv", pd.concat(audits, ignore_index=True))
            v1._csv(output / "oof_partial.csv", pd.concat(oof_parts, ignore_index=True))
        oof = pd.concat(oof_parts, ignore_index=True).set_index("image_name").loc[frame.image_name].reset_index()
        v1.verify_prediction_coverage(frame, oof)
        v1._csv(output / "oof.csv", oof)
        v1._csv(output / "metrics.csv", metrics_table(frame, oof))
        counters = {field: int(sum(row[field] for row in logs)) for field in
                    ("executed_fits", "external_reused_fits", "resumed_fits", "executed_this_invocation", "rescoring_operations")}
        maximum = 45 if condition == "original" else 675
        if counters["executed_fits"] > maximum:
            raise ValueError("V2 core fitting count exceeded its approved bound.")
        if counters["executed_fits"] + counters["external_reused_fits"] != 3960 or counters["rescoring_operations"] != 1170:
            raise ValueError("V2 completed fit/scoring ledger does not cover the complete four-family protocol.")
        for relative, frozen in compatibility["source_sha256"].items():
            if file_sha256(root / relative) != frozen:
                raise ValueError("A frozen v2 fitting dependency changed while this run was active.")
        if file_sha256(config_path) != compatibility["config_sha256"]:
            raise ValueError("The frozen v2 configuration changed while this run was active.")
        files = [path for path in output.rglob("*") if path.is_file() and "report" not in path.relative_to(output).parts
                 and path.name not in ("receipt.json", "run.lock") and not path.name.endswith(".tmp")]
        # Include child phase receipts, but not the not-yet-written final receipt.
        files.extend(path for path in output.rglob("receipt.json") if path != completion and "report" not in path.relative_to(output).parts)
        receipt = {"schema_version": 2, "experiment_version": 2, "status": "complete", "condition": condition,
                   "run_id": config["run_id"], "selection_metric": metric, "output_root": v1._relative(root, output),
                   "compatibility_sha256": digest, **compatibility, **counters,
                   "reused_fits": counters["external_reused_fits"], "maximum_new_fit_calls": maximum,
                   "rows": len(oof), "patients": int(oof.patient_id.nunique()), "positive_images": int(oof.target.sum()),
                   "positive_patients": int(oof.loc[oof.target.eq(1), "patient_id"].nunique()),
                   "output_sha256": {v1._relative(root, path): file_sha256(path) for path in files},
                   "external_source_sha256": inputs["parent_receipt_sha256"], "test_set_loaded": False, "final_model_fitted": False,
                   "interpretation": "ROC/AP objective change after previous development inspection; conditional development evidence, not a new independent test."}
        v1._json(completion, receipt)
        verify_receipt(root, receipt, output_root=output, expected_selection_metric=metric)
        if progress:
            progress(f"Completed {condition} v2: {counters['executed_fits']} new fit phases; {counters['external_reused_fits']} external phases reused.")
        return receipt


def run_tuning(root, config_path="configs/metadata_nn_tuning_roc_auc.yaml", *, condition=None,
               progress=print, stop_event=None, resume=True):
    root = Path(root).resolve()
    path, config = _read_config(root, config_path)
    inputs = _load_inputs(root, config)
    selected_conditions = config["conditions"] if condition is None else [condition]
    if any(name not in CONDITION_PREFIXES for name in selected_conditions):
        raise ValueError("Unknown v2 condition.")
    if not resume and any((condition_directory(root, name, config["run_id"]) / "session.json").exists() for name in selected_conditions):
        raise ValueError("Existing v2 phases require explicit resume or a new run identity.")
    with threadpool_limits(limits=4):
        results = {name: _run_condition(root, path, config, inputs, name, progress, stop_event) for name in selected_conditions}
    return results if condition is None else results[condition]
