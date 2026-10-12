"""Fixed clean metadata replication with training-only fitting and sealed folds.

This entrypoint reads only the frozen development variant. It never accesses
reserved-test records, modifies historical outputs, or creates a final model.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch import nn
import yaml

from melanoma_detection.metadata_evaluation import (
    decision_metrics, probability_metrics, select_recall_threshold,
    validate_probabilities,
)
from melanoma_detection.metadata_model import (
    FEATURES, SPECIFICATION as SPLINE_SPECIFICATION, build_spline_logistic,
    file_sha256, fit_recalibrator, grouped_oof, predict_recalibrated,
)
from melanoma_detection.metadata_validation import validate_metadata_frame


NN_SPECIFICATION = {
    "base_seeds": [42, 142, 242], "hidden_units": 32, "dropout": 0.20,
    "max_epochs": 100, "min_epochs": 10, "patience": 12,
    "minimum_improvement": 1e-5, "batch_size": 128,
    "learning_rate": 0.001, "weight_decay": 0.001,
    "loss": "unweighted BCEWithLogitsLoss", "device": "CPU", "threads": 4,
    "training_seed_rule": "base_seed + outer_fold",
    "stop_fold_rule": "(outer_fold + 1 + run_index) % 5",
    "ensemble": "equal mean of all three prescribed seeds",
    "epoch_selection": "minimum stopping-fold BCE within patient-disjoint outer training pool",
    "refit": "new preprocessor, network, optimiser and scheduler-free fixed epoch fit",
    "features": FEATURES,
}
SOURCE_FILES = (
    "src/melanoma_detection/dedup_metadata.py",
    "src/melanoma_detection/development_dataset.py",
    "src/melanoma_detection/metadata_model.py",
    "src/melanoma_detection/metadata_validation.py",
    "src/melanoma_detection/metadata_evaluation.py",
    "scripts/run_dedup_metadata.py",
    "notebooks/Subset/06_metadata_neural_network.ipynb",
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _csv(path, frame):
    temporary = Path(path).with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _joblib(path, value):
    temporary = Path(path).with_suffix(".joblib.tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def _versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "torch": torch.__version__, "joblib": joblib.__version__}


def _relative(root, path):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _within(root, relative, expected_prefix):
    path = (Path(root) / relative).resolve()
    try:
        path.relative_to((Path(root) / expected_prefix).resolve())
    except ValueError as error:
        raise ValueError(f"Path must remain under {expected_prefix}.") from error
    return path


def _prepare(frame):
    frame = validate_metadata_frame(frame).reset_index(drop=True)
    frame["age_approx"] = pd.to_numeric(frame["age_approx"], errors="raise")
    for column in ("sex", "anatom_site_general_challenge"):
        frame[column] = frame[column].fillna("Missing").astype(str)
    return frame


def neural_preprocessor():
    """The exact notebook 06 numerical and categorical preprocessing."""
    return ColumnTransformer([
        ("age", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), ["age_approx"]),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["sex", "anatom_site_general_challenge"]),
    ])


class MetadataNet(nn.Module):
    def __init__(self, input_width, spec=None):
        super().__init__()
        spec = NN_SPECIFICATION if spec is None else spec
        self.layers = nn.Sequential(
            nn.Linear(input_width, spec["hidden_units"]), nn.ReLU(),
            nn.Dropout(spec["dropout"]), nn.Linear(spec["hidden_units"], 1),
        )

    def forward(self, features):
        return self.layers(features).squeeze(1)


def fit_network(x_fit, y_fit, seed, *, epochs=None, x_stop=None, y_stop=None, spec=None):
    """Reinitialise every fit; stopping data never include outer patients."""
    spec = NN_SPECIFICATION if spec is None else spec
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    features = torch.tensor(np.asarray(x_fit, dtype=np.float32))
    labels = torch.tensor(np.asarray(y_fit, dtype=np.float32))
    if not torch.isfinite(features).all():
        raise ValueError("Encoded fitting features must be finite.")
    model = MetadataNet(features.shape[1], spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    stop_features = None if x_stop is None else torch.tensor(np.asarray(x_stop, dtype=np.float32))
    stop_labels = None if y_stop is None else torch.tensor(np.asarray(y_stop, dtype=np.float32))
    if (stop_features is None) != (stop_labels is None):
        raise ValueError("Stopping features and labels must be supplied together.")
    if epochs is None and stop_features is None:
        raise ValueError("Epoch selection requires stopping data.")
    if stop_features is not None and not torch.isfinite(stop_features).all():
        raise ValueError("Encoded stopping features must be finite.")
    best_loss, best_epoch, stale = float("inf"), 1, 0
    history = []
    for epoch in range(1, (spec["max_epochs"] if epochs is None else epochs) + 1):
        model.train()
        accumulated = 0.0
        for indices in torch.randperm(len(features)).split(spec["batch_size"]):
            optimizer.zero_grad()
            loss = criterion(model(features[indices]), labels[indices])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite neural-network fitting loss.")
            loss.backward()
            optimizer.step()
            accumulated += float(loss.detach()) * len(indices)
        record = {"epoch": epoch, "fit_bce": accumulated / len(features), "stop_bce": None}
        if stop_features is not None:
            model.eval()
            with torch.no_grad():
                stop_loss = float(criterion(model(stop_features), stop_labels))
            if not np.isfinite(stop_loss):
                raise ValueError("Non-finite stopping loss.")
            record["stop_bce"] = stop_loss
            if stop_loss < best_loss - spec["minimum_improvement"]:
                best_loss, best_epoch, stale = stop_loss, epoch, 0
            else:
                stale += 1
        history.append(record)
        if stop_features is not None and epoch >= spec["min_epochs"] and stale >= spec["patience"]:
            break
    return model, best_epoch, (None if stop_features is None else best_loss), history


def predict_network(model, features):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(np.asarray(features, dtype=np.float32)))).numpy()


def run_outer_fold(frame, outer_fold, *, spline_calibration=True, target_recall=0.8, nn_spec=None, progress=None):
    """One complete training-only procedure; carry outer labels only in output."""
    spec = deepcopy(NN_SPECIFICATION if nn_spec is None else nn_spec)
    frame = _prepare(frame)
    torch.set_num_threads(spec["threads"])
    score_mask = frame.fold.eq(outer_fold)
    outer_mask = ~score_mask
    fitting, heldout = frame.loc[outer_mask], frame.loc[score_mask]
    score_patients = set(heldout.patient_id)
    if not set(fitting.patient_id).isdisjoint(score_patients):
        raise ValueError("Outer fitting and scoring patients overlap.")
    baseline = build_spline_logistic().fit(fitting[FEATURES], fitting.target)
    predictions = heldout[["image_name", "patient_id", "target", "fold"]].copy()
    predictions["p_spline"] = baseline.predict_proba(heldout[FEATURES])[:, 1]
    epochs, histories, membership_rows, bundles = [], [], [], {}
    for run_index, base_seed in enumerate(spec["base_seeds"]):
        stop_fold = (outer_fold + 1 + run_index) % 5
        fit_mask = ~frame.fold.isin([outer_fold, stop_fold])
        stop_mask = frame.fold.eq(stop_fold)
        epoch_fit, stopping = frame.loc[fit_mask], frame.loc[stop_mask]
        fit_patients, stop_patients = set(epoch_fit.patient_id), set(stopping.patient_id)
        if not fit_patients.isdisjoint(stop_patients) or not (fit_patients | stop_patients).isdisjoint(score_patients):
            raise ValueError("Epoch selection contains overlapping or outer patients.")
        if any(rows.target.nunique() != 2 for rows in (epoch_fit, stopping, fitting)):
            raise ValueError("Every fitting and stopping partition requires both classes.")
        patients = frame.groupby(["patient_id", "fold"]).size().reset_index(name="retained_images")
        for row in patients.to_dict("records"):
            role = ("outer_validation" if row["fold"] == outer_fold else
                    "epoch_stopping" if row["fold"] == stop_fold else "epoch_fitting")
            membership_rows.append({**row, "outer_fold": outer_fold, "base_seed": base_seed,
                                    "role": role, "outer_refit": role != "outer_validation"})
        actual_seed = base_seed + outer_fold
        stop_prep = neural_preprocessor().fit(epoch_fit[FEATURES])
        _, selected, stop_loss, stop_history = fit_network(
            stop_prep.transform(epoch_fit[FEATURES]), epoch_fit.target, actual_seed,
            x_stop=stop_prep.transform(stopping[FEATURES]), y_stop=stopping.target, spec=spec,
        )
        outer_prep = neural_preprocessor().fit(fitting[FEATURES])
        encoded = outer_prep.transform(fitting[FEATURES])
        model, _, _, fit_history = fit_network(encoded, fitting.target, actual_seed, epochs=selected, spec=spec)
        predictions[f"p_nn_seed_{base_seed}"] = predict_network(model, outer_prep.transform(heldout[FEATURES]))
        epochs.append({"base_seed": base_seed, "outer_fold": outer_fold, "training_seed": actual_seed,
                       "stop_fold": stop_fold, "selected_epochs": selected, "stop_log_loss": stop_loss,
                       "attempted_stopping_epochs": len(stop_history), "outer_train_patients": fitting.patient_id.nunique(),
                       "score_patients": heldout.patient_id.nunique(), "epoch_fit_patients": len(fit_patients),
                       "epoch_stop_patients": len(stop_patients), "encoded_input_width": encoded.shape[1]})
        histories.extend({**row, "base_seed": base_seed, "outer_fold": outer_fold, "phase": phase}
                         for phase, history in (("stopping", stop_history), ("outer_refit", fit_history)) for row in history)
        bundles[base_seed] = {"preprocessor": outer_prep, "input_width": encoded.shape[1],
                              "state_dict": model.state_dict(), "seed": actual_seed,
                              "selected_epochs": selected}
        if progress is not None:
            progress(f"Clean metadata fold {outer_fold}, seed {base_seed}: stopping fold {stop_fold}, selected {selected} epochs.")
    predictions["p_nn_mean"] = predictions[[f"p_nn_seed_{seed}" for seed in spec["base_seeds"]]].mean(axis=1)
    # Match the historical NN table's exact column order for comparator loading.
    predictions = predictions[["image_name", "patient_id", "target", "fold",
                               *[f"p_nn_seed_{seed}" for seed in spec["base_seeds"]], "p_nn_mean", "p_spline"]]
    calibration, training_oof, audit, calibrator = None, None, pd.DataFrame(), None
    if spline_calibration:
        inner_probabilities, audit = grouped_oof(fitting.reset_index(drop=True))
        calibrator = fit_recalibrator(fitting.target.to_numpy(), inner_probabilities)
        raw_threshold = select_recall_threshold(fitting.target, inner_probabilities, target_recall)
        calibrated_inner = predict_recalibrated(calibrator, inner_probabilities)
        calibrated_threshold = select_recall_threshold(fitting.target, calibrated_inner, target_recall)
        training_oof = fitting[["image_name", "patient_id", "target", "fold"]].copy()
        training_oof["outer_fold"] = outer_fold
        training_oof["p_spline_raw"] = inner_probabilities
        training_oof["p_spline_calibrated"] = calibrated_inner
        calibration = heldout[["image_name", "patient_id", "target", "fold"]].copy()
        calibration["p_spline_raw"] = predictions.p_spline.to_numpy()
        calibration["p_spline_calibrated"] = predict_recalibrated(calibrator, predictions.p_spline.to_numpy())
        calibration["raw_threshold"] = raw_threshold
        calibration["calibrated_threshold"] = calibrated_threshold
        calibration["decision_raw"] = (calibration.p_spline_raw >= raw_threshold).astype(int)
        calibration["decision_calibrated"] = (calibration.p_spline_calibrated >= calibrated_threshold).astype(int)
        audit.insert(0, "outer_fold", outer_fold)
        audit["outer_validation_patients"] = heldout.patient_id.nunique()
        audit["calibrator_slope"] = float(calibrator.coef_[0, 0])
        audit["calibrator_intercept"] = float(calibrator.intercept_[0])
    for column in predictions.filter(like="p_"):
        validate_probabilities(predictions.target, predictions[column])
    return {"predictions": predictions.reset_index(drop=True), "calibration": None if calibration is None else calibration.reset_index(drop=True),
            "training_oof": None if training_oof is None else training_oof.reset_index(drop=True),
            "epochs": pd.DataFrame(epochs), "history": pd.DataFrame(histories), "inner_audit": audit,
            "selection_memberships": pd.DataFrame(membership_rows),
            "models": {"spline": baseline, "networks": bundles, "recalibrator": calibrator,
                       "fit_image_ids": fitting.image_name.tolist(), "fit_patient_ids": sorted(fitting.patient_id.unique()),
                       "outer_fold": outer_fold, "nn_specification": spec}}


def _verify_predictions(frame, predictions, *, fold=None):
    expected = frame if fold is None else frame.loc[frame.fold.eq(fold)]
    keys = ["image_name", "patient_id", "target", "fold"]
    if not predictions.image_name.is_unique or len(predictions) != len(expected):
        raise ValueError("Prediction coverage is not exactly once per retained image.")
    left = expected[keys].sort_values("image_name").reset_index(drop=True)
    right = predictions[keys].sort_values("image_name").reset_index(drop=True)
    for table in (left, right):
        table["target"] = table["target"].astype("int64")
        table["fold"] = table["fold"].astype("int64")
    if not left.equals(right):
        raise ValueError("Predictions do not match the retained image/patient/label/fold records.")
    for column in predictions.filter(like="p_"):
        validate_probabilities(predictions.target, predictions[column])


def verify_saved_receipt(root, receipt, *, compatibility_sha256=None):
    """Verify immutable spec and every hashed output before resume or comparison."""
    if receipt.get("status") != "complete":
        raise ValueError("A complete receipt is required.")
    if compatibility_sha256 is not None and receipt.get("compatibility_sha256") != compatibility_sha256:
        raise ValueError("Dataset, source, runtime or protocol changed; use a new run directory.")
    for relative, expected in receipt["output_sha256"].items():
        path = _within(root, relative, "results/dedup_v1/metadata")
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Saved output is missing or changed: {relative}")


def _report_tables(frame, predictions, calibration):
    metrics, subgroups = [], []
    scores = {column: predictions[column].to_numpy() for column in predictions.filter(like="p_")}
    if calibration is not None:
        scores["p_spline_calibrated"] = calibration.p_spline_calibrated.to_numpy()
    for name, probability in scores.items():
        metrics.append({"model": name, "scope": "pooled", "fold": None, **probability_metrics(frame.target, probability)})
        fold_rows = []
        for fold in range(5):
            mask = frame.fold.eq(fold).to_numpy()
            row = {"model": name, "scope": "fold", "fold": fold, **probability_metrics(frame.loc[mask, "target"], probability[mask])}
            metrics.append(row)
            fold_rows.append(row)
        metrics.append({"model": name, "scope": "mean_fold", "fold": None,
                        **{key: float(np.mean([row[key] for row in fold_rows])) for key in probability_metrics(frame.target, probability)}})
        if name not in ("p_spline", "p_nn_mean", "p_spline_calibrated"):
            continue
        grouping = {"sex": frame.sex, "site": frame.anatom_site_general_challenge,
                    "site_known": frame.anatom_site_general_challenge.isin(["Missing", "Unknown"]).map({True: "unknown", False: "known"}),
                    "age_band": pd.cut(frame.age_approx, [-1, 39, 59, 120], labels=["0-39", "40-59", "60-120"]).astype("string").fillna("Missing")}
        for category, groups in grouping.items():
            for group in sorted(groups.unique()):
                mask = groups.eq(group).to_numpy(dtype=bool)
                rows = frame.loc[mask]
                subgroups.append({"model": name, "category": category, "group": str(group),
                                  "images": len(rows), "malignant_images": int(rows.target.sum()),
                                  "patients": rows.patient_id.nunique(), "malignant_patients": rows.loc[rows.target.eq(1), "patient_id"].nunique(),
                                  **probability_metrics(rows.target, probability[mask])})
    decisions = []
    if calibration is not None:
        for probability, decision in (("p_spline_raw", "decision_raw"), ("p_spline_calibrated", "decision_calibrated")):
            decisions.append({"model": probability, "scope": "pooled", "fold": None,
                              **decision_metrics(calibration.target, calibration[decision])})
            for fold, rows in calibration.groupby("fold"):
                decisions.append({"model": probability, "scope": "fold", "fold": fold,
                                  **decision_metrics(rows.target, rows[decision])})
    return pd.DataFrame(metrics), pd.DataFrame(subgroups), pd.DataFrame(decisions)


def run_metadata(project_root, config_path="configs/metadata_dedup.yaml"):
    """Resume sealed folds or run the exact fixed clean CPU comparison."""
    from melanoma_detection.development_dataset import load_development_dataset

    root = Path(project_root).resolve()
    config_path = (root / config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if set(config) != {"dataset_manifest", "output_dir", "spline_calibration", "target_recall"}:
        raise ValueError("Unknown or missing clean metadata configuration keys.")
    if not isinstance(config["spline_calibration"], bool) or config["target_recall"] != 0.8:
        raise ValueError("The replication uses fixed calibration choice and recall 0.8.")
    output = _within(root, config["output_dir"], "results/dedup_v1/metadata")
    frame, identity = load_development_dataset(root, config["dataset_manifest"])
    frame = _prepare(frame)
    if identity["variant"] != "dedup_v1" or (len(frame), int(frame.target.sum()), frame.patient_id.nunique()) != (6706, 127, 422):
        raise ValueError("The frozen 6,706-image dedup_v1 development dataset is required.")
    specification = {"neural_network": NN_SPECIFICATION, "spline": SPLINE_SPECIFICATION,
                     "spline_calibration": config["spline_calibration"], "target_recall": config["target_recall"],
                     "patient_membership_source": {
                         "path": "data/subset/processed/train_folds.csv",
                         "sha256": identity["canonical_source_sha256"],
                         "neural_stopping": "Original outer-fold assignments; stop_fold=(outer_fold+1+run_index)%5; retained mask only",
                         "spline_inner_calibration": "Four remaining original outer folds; retained mask only",
                         "splitters_rerun": False}}
    compatibility = {"dataset_identity": identity, "specification": specification, "versions": _versions(),
                     "config_sha256": file_sha256(config_path),
                     "source_sha256": {path: file_sha256(root / path) for path in SOURCE_FILES}}
    compatibility_sha256 = _digest(compatibility)
    completion_path = output / "comparator_receipt.json"
    if completion_path.exists():
        receipt = json.loads(completion_path.read_text(encoding="utf-8"))
        verify_saved_receipt(root, receipt, compatibility_sha256=compatibility_sha256)
        _verify_predictions(frame, pd.read_csv(root / receipt["prediction_path"]))
        print("Verified existing complete clean metadata results; no fitting repeated.", flush=True)
        return receipt
    output.mkdir(parents=True, exist_ok=True)
    session_path = output / "session.json"
    if session_path.exists():
        session = json.loads(session_path.read_text(encoding="utf-8"))
        if session["compatibility_sha256"] != compatibility_sha256:
            raise ValueError("Existing clean metadata session differs; preserve it and choose a new output directory.")
    else:
        session = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "compatibility_sha256": compatibility_sha256, **compatibility}
        _json(session_path, session)
    started = time.monotonic()
    pieces, calibrated_pieces, logs, histories, audits = [], [], [], [], []
    fold_receipts = []
    for fold in range(5):
        folder = output / f"fold_{fold}"
        folder.mkdir(exist_ok=True)
        fold_receipt_path = folder / "receipt.json"
        if fold_receipt_path.exists():
            receipt = json.loads(fold_receipt_path.read_text(encoding="utf-8"))
            verify_saved_receipt(root, receipt, compatibility_sha256=compatibility_sha256)
            predictions = pd.read_csv(folder / "predictions.csv")
            _verify_predictions(frame, predictions, fold=fold)
            calibration = pd.read_csv(folder / "calibration.csv") if config["spline_calibration"] else None
            if calibration is not None:
                _verify_predictions(frame, calibration, fold=fold)
            print(f"Verified clean metadata outer fold {fold}; reused complete outputs.", flush=True)
        else:
            print(f"Fitting clean metadata outer fold {fold}: fixed spline and three stopping/refit NN pairs.", flush=True)
            fold_start = time.monotonic()
            result = run_outer_fold(frame, fold, spline_calibration=config["spline_calibration"], target_recall=config["target_recall"],
                                    progress=lambda message: print(message, flush=True))
            predictions, calibration = result["predictions"], result["calibration"]
            _verify_predictions(frame, predictions, fold=fold)
            _csv(folder / "predictions.csv", predictions)
            if calibration is not None:
                _verify_predictions(frame, calibration, fold=fold)
                _csv(folder / "calibration.csv", calibration)
                _csv(folder / "inner_audit.csv", result["inner_audit"])
                _csv(folder / "training_oof.csv", result["training_oof"])
            _csv(folder / "epochs.csv", result["epochs"])
            _csv(folder / "history.csv", result["history"])
            _csv(folder / "selection_memberships.csv", result["selection_memberships"])
            _joblib(folder / "models.joblib", {**result["models"], "dataset_identity": identity,
                                               "compatibility_sha256": compatibility_sha256})
            files = ["predictions.csv", "epochs.csv", "history.csv", "selection_memberships.csv", "models.joblib"]
            if calibration is not None:
                files.extend(["calibration.csv", "inner_audit.csv", "training_oof.csv"])
            receipt = {"status": "complete", "outer_fold": fold, "rows": len(predictions),
                       "compatibility_sha256": compatibility_sha256,
                       "elapsed_seconds": time.monotonic() - fold_start,
                       "output_sha256": {_relative(root, folder / name): file_sha256(folder / name) for name in files}}
            _json(fold_receipt_path, receipt)
            print(f"Completed clean metadata fold {fold} in {receipt['elapsed_seconds']:.1f}s.", flush=True)
        pieces.append(predictions)
        if calibration is not None:
            calibrated_pieces.append(calibration)
            audits.append(pd.read_csv(folder / "inner_audit.csv"))
        logs.append(pd.read_csv(folder / "epochs.csv"))
        histories.append(pd.read_csv(folder / "history.csv"))
        fold_receipts.append(receipt)
    order = {name: index for index, name in enumerate(frame.image_name)}
    predictions = pd.concat(pieces).sort_values("image_name", key=lambda names: names.map(order)).reset_index(drop=True)
    _verify_predictions(frame, predictions)
    calibration = None if not calibrated_pieces else pd.concat(calibrated_pieces).sort_values("image_name", key=lambda names: names.map(order)).reset_index(drop=True)
    epochs, history = pd.concat(logs, ignore_index=True), pd.concat(histories, ignore_index=True)
    _csv(output / "metadata_nn_oof.csv", predictions)
    _csv(output / "epochs.csv", epochs)
    _csv(output / "history.csv", history)
    output_names = ["metadata_nn_oof.csv", "epochs.csv", "history.csv", "session.json"]
    if calibration is not None:
        _csv(output / "metadata_calibrated_oof.csv", calibration)
        _csv(output / "spline_inner_audit.csv", pd.concat(audits, ignore_index=True))
        output_names.extend(["metadata_calibrated_oof.csv", "spline_inner_audit.csv"])
    metrics, subgroups, decisions = _report_tables(frame, predictions, calibration)
    for name, table in (("metrics.csv", metrics), ("probability_subgroups.csv", subgroups), ("cutoff_metrics.csv", decisions)):
        _csv(output / name, table)
        output_names.append(name)
    individual = metrics.loc[metrics.scope.eq("pooled") & metrics.model.isin([f"p_nn_seed_{seed}" for seed in NN_SPECIFICATION["base_seeds"]])]
    stability = pd.DataFrame([{"metric": metric, "runs": len(individual),
                               "mean": individual[metric].mean(), "sample_std": individual[metric].std(ddof=1),
                               "minimum": individual[metric].min(), "maximum": individual[metric].max(),
                               "range": individual[metric].max() - individual[metric].min()}
                              for metric in ("average_precision", "roc_auc", "brier", "log_loss")])
    _csv(output / "nn_stability.csv", stability)
    output_names.append("nn_stability.csv")
    run_config = {
        "schema_version": 1, "status": "complete", "variant": "dedup_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "dataset_identity": identity,
        "input_path": "data/subset/processed/train_folds.csv", "input_sha256": identity["canonical_source_sha256"],
        "dataset_manifest": identity["dataset_manifest"], "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
        "retained_ids_sha256": identity["retained_ids_sha256"], "specification": specification,
        "metadata_features": FEATURES, "base_seeds": NN_SPECIFICATION["base_seeds"],
        "source_sha256": compatibility["source_sha256"], "versions": compatibility["versions"],
        "code_sha256": compatibility["source_sha256"]["src/melanoma_detection/dedup_metadata.py"],
        "prediction_path": _relative(root, output / "metadata_nn_oof.csv"),
        "prediction_sha256": file_sha256(output / "metadata_nn_oof.csv"), "prediction_columns": list(predictions.columns),
        "calibration_prediction_path": None if calibration is None else _relative(root, output / "metadata_calibrated_oof.csv"),
        "calibration_prediction_sha256": None if calibration is None else file_sha256(output / "metadata_calibrated_oof.csv"),
        "rows": len(frame), "patients": int(frame.patient_id.nunique()), "positive_images": int(frame.target.sum()),
        "outer_folds": [0, 1, 2, 3, 4], "device": "CPU", "torch_threads": torch.get_num_threads(),
        "fit_counts": {"outer_spline": 5, "neural_stopping": 15, "neural_outer_refit": 15,
                       "inner_spline": 20 if calibration is not None else 0, "spline_recalibrator": 5 if calibration is not None else 0},
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "completed_fold_seconds": sum(receipt["elapsed_seconds"] for receipt in fold_receipts),
        "compatibility_sha256": compatibility_sha256,
        "interpretation": "Clean development OOF replication; final reserved test not evaluated; no final model fitted.",
    }
    _json(output / "run_config.json", run_config)
    output_names.append("run_config.json")
    hashes = {_relative(root, output / name): file_sha256(output / name) for name in output_names}
    for fold in range(5):
        folder = output / f"fold_{fold}"
        hashes[_relative(root, folder / "receipt.json")] = file_sha256(folder / "receipt.json")
        hashes.update(fold_receipts[fold]["output_sha256"])
    receipt = {**run_config, "output_sha256": hashes}
    _json(completion_path, receipt)
    verify_saved_receipt(root, receipt, compatibility_sha256=compatibility_sha256)
    return receipt
