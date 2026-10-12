"""Five-fold original baseline reconstruction and verified clean baseline reads."""
from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .dedup_metadata_suite import categorical_frame, categorical_pipeline
from .development_dataset import load_development_dataset, stable_sha256
from .metadata_model import FEATURES, file_sha256
from .metadata_evaluation import probability_metrics, validate_probabilities
from .metadata_nn_tuning import KEYS, verify_prediction_coverage
from .metadata_roc_archive import original_snapshot

DIRECTORY = "results/metadata_nn_tuning/v2/baseline_diagnostics/original_baseline_v1"
SPECIFICATION = {"features": list(FEATURES), "encoding": "age, sex and site as strings; train-only one-hot",
                 "C": 1.0, "solver": "liblinear", "max_iter": 1000, "random_state": 42,
                 "dummy": "outer-training malignant prevalence", "outer_folds": 5}


def _write_json(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def baseline_predictions(frame, outer):
    """Fit exactly one prescribed logistic model, excluding the outer patients."""
    prepared = categorical_frame(frame)
    fitting = prepared.loc[prepared.fold.ne(outer)]
    heldout = prepared.loc[prepared.fold.eq(outer)]
    if set(fitting.patient_id) & set(heldout.patient_id):
        raise ValueError("Baseline patients cross the outer partition.")
    if set(fitting.target) != {0, 1} or set(heldout.target) != {0, 1}:
        raise ValueError("Baseline folds require both classes.")
    classifier = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=42)
    model = categorical_pipeline(classifier).fit(fitting[FEATURES], fitting.target)
    predictions = heldout[KEYS].copy()
    predictions["p_dummy"] = float(fitting.target.mean())
    predictions["p_categorical_logistic"] = model.predict_proba(heldout[FEATURES])[:, 1]
    for column in ["p_dummy", "p_categorical_logistic"]:
        validate_probabilities(predictions.target, predictions[column])
    audit = {"outer_fold": outer, "fitting_images": len(fitting), "heldout_images": len(heldout),
             "fitting_patients": fitting.patient_id.nunique(), "heldout_patients": heldout.patient_id.nunique(),
             "patient_overlap": 0, "preprocessing_fitted_on": "outer-training records only", "fits": 1}
    roles = prepared[KEYS].copy()
    roles["outer_fold"] = outer
    roles["role"] = np.where(prepared.fold.eq(outer), "heldout", "fitting")
    return predictions, model, audit, roles


def _verify(root, receipt):
    if receipt.get("status") != "complete" or receipt.get("condition") != "original":
        raise ValueError("A completed original baseline receipt is required.")
    directory = (root / DIRECTORY).resolve()
    for relative, digest in receipt["source_sha256"].items():
        if file_sha256(root / relative) != digest:
            raise ValueError("Baseline fitting source changed.")
    for relative, digest in receipt["output_sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(directory) or file_sha256(path) != digest:
            raise ValueError("Baseline artifact changed or escaped its output root.")


def run_baselines(project_root):
    root = Path(project_root).resolve()
    frame, identity = load_development_dataset(root)
    snapshot = original_snapshot(root, "notebooks/Subset/04_dummy_logistic_baseline.ipynb")
    sources = {"src/melanoma_detection/metadata_roc_baselines.py": file_sha256(__file__),
               "src/melanoma_detection/dedup_metadata_suite.py": file_sha256(root / "src/melanoma_detection/dedup_metadata_suite.py")}
    compatibility = {"dataset_identity": identity, "specification": SPECIFICATION, "source_sha256": sources,
                     "historical_protocol_sha256": file_sha256(snapshot)}
    digest = stable_sha256(compatibility)
    directory = root / DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    session_path = directory / "session.json"
    if session_path.exists():
        if json.loads(session_path.read_text())["compatibility_sha256"] != digest:
            raise ValueError("Baseline inputs or implementation changed; use a new diagnostic identity.")
    else:
        _write_json(session_path, {**compatibility, "compatibility_sha256": digest})
    if (directory / "receipt.json").exists():
        receipt = json.loads((directory / "receipt.json").read_text())
        _verify(root, receipt)
        return receipt
    parts, audits, role_parts, executed = [], [], [], 0
    with threadpool_limits(limits=4):
        for outer in range(5):
            folder = directory / f"fold_{outer}"
            folder.mkdir(exist_ok=True)
            checkpoint = folder / "receipt.json"
            if checkpoint.exists():
                record = json.loads(checkpoint.read_text())
                if record["compatibility_sha256"] != digest or record["outer_fold"] != outer:
                    raise ValueError("Baseline fold checkpoint is incompatible.")
                for name, value in record["output_sha256"].items():
                    if file_sha256(folder / name) != value:
                        raise ValueError("Baseline checkpoint changed.")
                part = pd.read_csv(folder / "predictions.csv", float_precision="round_trip")
                roles = pd.read_csv(folder / "roles.csv")
                audit = record["audit"]
            else:
                part, model, audit, roles = baseline_predictions(frame, outer)
                part.to_csv(folder / "predictions.csv", index=False)
                roles.to_csv(folder / "roles.csv", index=False)
                joblib.dump(model, folder / "model.joblib")
                audit = {key: int(value) if isinstance(value, np.integer) else value for key, value in audit.items()}
                _write_json(checkpoint, {"status": "complete", "compatibility_sha256": digest,
                    "outer_fold": outer, "audit": audit,
                    "output_sha256": {name: file_sha256(folder / name) for name in ["predictions.csv", "roles.csv", "model.joblib"]}})
                executed += 1
            verify_prediction_coverage(frame, part, fold=outer)
            parts.append(part)
            audits.append(audit)
            role_parts.append(roles)
    predictions = pd.concat(parts, ignore_index=True).set_index("image_name").loc[frame.image_name].reset_index()
    verify_prediction_coverage(frame, predictions)
    predictions.to_csv(directory / "oof.csv", index=False)
    pd.DataFrame(audits).to_csv(directory / "fold_audit.csv", index=False)
    pd.concat(role_parts, ignore_index=True).to_csv(directory / "roles.csv", index=False)
    rows = []
    for model, column in [("dummy", "p_dummy"), ("categorical_logistic", "p_categorical_logistic")]:
        fold_metrics = []
        for outer, subset in predictions.groupby("fold", sort=True):
            values = probability_metrics(subset.target, subset[column])
            fold_metrics.append(values)
            rows.append({"model": model, "scope": "fold", "fold": outer, **values})
        rows.append({"model": model, "scope": "mean_fold", "fold": None, **pd.DataFrame(fold_metrics).mean().to_dict()})
        rows.append({"model": model, "scope": "pooled", "fold": None, **probability_metrics(predictions.target, predictions[column])})
    pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
    files = [path for path in directory.rglob('*') if path.is_file() and path.name != 'receipt.json']
    files += [directory / f"fold_{i}/receipt.json" for i in range(5)]
    receipt = {"status": "complete", "condition": "original", "dataset_identity": identity,
               "compatibility_sha256": digest, **compatibility, "executed_fits": 5,
               "executed_fits_this_invocation": executed, "reused_fits": 0, "dummy_fits": 0,
               "rows": len(frame), "patients": int(frame.patient_id.nunique()), "reserved_test_loaded": False,
               "final_model_fitted": False, "output_sha256": {p.relative_to(root).as_posix(): file_sha256(p) for p in files}}
    _write_json(directory / "receipt.json", receipt)
    _verify(root, receipt)
    return receipt


def load_baselines(project_root, condition):
    root = Path(project_root).resolve()
    if condition == "original":
        directory = root / DIRECTORY
        if not (directory / "receipt.json").exists():
            return {"status": "pending", "directory": directory, "receipt": None, "oof": None}
        receipt = json.loads((directory / "receipt.json").read_text())
        _verify(root, receipt)
        frame, identity = load_development_dataset(root)
        if receipt["dataset_identity"] != identity:
            raise ValueError("Baseline development identity changed.")
        predictions = pd.read_csv(directory / "oof.csv", float_precision="round_trip")
    elif condition == "dedup_v1":
        from .historical_metadata_artifacts import verify_historical_suite
        from .metadata_roc_archive import SUITE_RECEIPT
        directory = root / Path(SUITE_RECEIPT).parent
        receipt = json.loads((root / SUITE_RECEIPT).read_text())
        verify_historical_suite(root, receipt)
        frame, identity = load_development_dataset(root, "data/subset/processed/dedup_v1/dataset_manifest.json")
        if receipt["dataset_identity"] != identity:
            raise ValueError("Clean baseline development identity changed.")
        predictions = frame[KEYS].copy()
        for label, column in [("dummy", "p_dummy"), ("categorical_logistic", "p_categorical_logistic")]:
            saved = pd.read_csv(directory / "predictions" / f"{label}.csv", float_precision="round_trip")
            verify_prediction_coverage(frame, saved)
            predictions[column] = saved.set_index("image_name").loc[frame.image_name, "score"].to_numpy()
    else:
        raise ValueError("Expected original or dedup_v1 baselines.")
    verify_prediction_coverage(frame, predictions)
    return {"status": "complete", "directory": directory, "receipt": receipt, "oof": predictions}
