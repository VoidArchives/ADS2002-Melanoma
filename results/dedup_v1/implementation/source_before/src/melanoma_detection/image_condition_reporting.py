"""Fitting-only image QA and saved controlled-condition reporting.

These functions never train, approve their own quality gate, tune thresholds,
fit calibration or load excluded/test patients. Predictions must be from fresh
matched twelve-epoch fits before a condition can enter a comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

from .image_evaluation import assert_matching_predictions, curve_tables, metric_table, sha256
from .image_model import canonical_hash, capture_rng_state, make_transform, restore_rng_state, seed_everything
from .image_reporting import (plot_confusion_matrices, plot_learning_curves,
    plot_raw_reliability, plot_roc_precision_recall, plot_training_losses, save_figure)
from .metadata_evaluation import probability_metrics


QUALITY_CATEGORIES = ("clipping", "boundary", "texture", "colour", "reflection")
CONDITION_LABELS = {"A": "A weak / unchanged", "B": "B stronger / unchanged",
                    "C": "C weak / CLAHE"}
PREVIEW_SEED = 4200
HORIZON = 12


def _frame_ids(frame):
    frame = frame.copy()
    if "image_id" not in frame:
        frame["image_id"] = frame["image_name"]
    required = {"image_id", "patient_id", "target", "image_path"}
    if not required.issubset(frame):
        raise ValueError("Quality preview requires fitting image IDs, patients, labels and paths.")
    if frame.image_id.isna().any() or frame.image_id.duplicated().any():
        raise ValueError("Quality preview requires unique nonmissing fitting image IDs.")
    return frame


def fitting_membership_hash(fitting):
    frame = _frame_ids(fitting)
    return canonical_hash(frame[["image_id", "patient_id", "target"]]
                          .sort_values("image_id").to_dict("records"))


def select_quality_examples(fitting, selected_ids=None, *, per_class=12, seed=2026):
    """Freeze balanced distinct fitting patients before any performance inspection.

    Default sampling spans observed image brightness. Framing and hair coverage
    are subsequently inspected by the reviewer; automatic selection does not
    certify those properties or any transformation's safety.
    """
    frame = _frame_ids(fitting)
    if not frame.target.isin([0, 1]).all():
        raise ValueError("Quality sample labels must be binary.")
    if selected_ids is not None:
        selected_ids = list(selected_ids)
        if len(selected_ids) != 2 * per_class or len(set(selected_ids)) != len(selected_ids):
            raise ValueError("Quality IDs must contain exactly 24 unique images.")
        if not set(selected_ids).issubset(set(frame.image_id)):
            raise ValueError("Quality previews must contain fitting images only.")
        chosen = frame.set_index("image_id").loc[selected_ids].reset_index()
    else:
        positive_patients = set(frame.loc[frame.target.eq(1), "patient_id"])
        groups = []
        for target in (1, 0):
            eligible = frame.loc[frame.target.eq(target)]
            if target == 0:
                eligible = eligible.loc[~eligible.patient_id.isin(positive_patients)]
            eligible = eligible.sort_values("image_id").sample(frac=1, random_state=seed + target)
            eligible = eligible.drop_duplicates("patient_id").head(max(48, per_class * 4)).copy()
            if len(eligible) < per_class:
                raise ValueError("Insufficient distinct fitting patients for balanced quality inspection.")
            brightness = []
            for row in eligible.itertuples():
                with Image.open(row.image_path) as image:
                    brightness.append(float(np.asarray(image.convert("RGB").resize((64, 64))).mean()))
            eligible["qa_brightness"] = brightness
            eligible = eligible.sort_values(["qa_brightness", "image_id"])
            indices = np.rint(np.linspace(0, len(eligible) - 1, per_class)).astype(int)
            groups.append(eligible.iloc[indices])
        chosen = pd.concat(groups, ignore_index=True)
    if chosen.patient_id.nunique() != 2 * per_class:
        raise ValueError("Quality sample must use 24 distinct fitting patients.")
    if not all(int(chosen.target.eq(value).sum()) == per_class for value in (0, 1)):
        raise ValueError("Quality sample must contain twelve positive and twelve negative images.")
    return chosen.reset_index(drop=True)


def save_quality_contact_sheet(chosen, output_dir):
    """Save the selected fitting-only originals for sample coverage inspection."""
    chosen = _frame_ids(chosen)
    output_dir = Path(output_dir)
    paths = []
    for page_start in range(0, len(chosen), 12):
        subset = chosen.iloc[page_start:page_start + 12]
        fig, axes = plt.subplots(3, 4, figsize=(12, 9), squeeze=False, constrained_layout=True)
        for ax, row in zip(axes.flat, subset.itertuples()):
            with Image.open(row.image_path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(f"{row.image_id} | target {row.target}\npatient {row.patient_id}", fontsize=9)
            ax.axis("off")
        for ax in list(axes.flat)[len(subset):]:
            ax.axis("off")
        fig.suptitle("Fitting-only quality sample: inspect framing, hair and lighting", fontsize=13)
        paths.append(save_figure(fig, output_dir / f"quality_originals_{page_start // 12 + 1:02d}.png", dpi=130))
        plt.close(fig)
    return paths


def _display_tensor(tensor):
    mean = np.asarray([.485, .456, .406])
    std = np.asarray([.229, .224, .225])
    return np.clip(tensor.detach().cpu().permute(1, 2, 0).numpy() * std + mean, 0, 1)


def _seed(index, variant):
    return PREVIEW_SEED + 4 * index + variant


def save_sampled_geometry(chosen, output_path, *, image_size=224):
    """Replay the declared RNG draw order without altering transformation output."""
    from .image_transforms import ReflectAffine
    state = capture_rng_state()
    rows = []
    try:
        image = Image.new("RGB", (image_size, image_size))
        for index, row in enumerate(_frame_ids(chosen).itertuples()):
            for variant in range(4):
                for condition_id in ("A", "B"):
                    seed = _seed(index, variant)
                    seed_everything(seed)
                    horizontal = bool(torch.rand(1).item() < .5)
                    vertical = bool(torch.rand(1).item() < .5)
                    if condition_id == "A":
                        angle = float(torch.empty(1).uniform_(-20, 20).item())
                        translate, scale = [0, 0], 1.0
                    else:
                        angle, translate, scale = ReflectAffine.parameters(image)
                    rows.append({"image_id": str(row.image_id), "condition_id": condition_id,
                        "variant": variant, "seed": seed, "horizontal_flip": horizontal,
                        "vertical_flip": vertical, "rotation_degrees": angle,
                        "translation_pixels": translate, "scale": scale})
    finally:
        restore_rng_state(state)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"draw_order": "flips, affine, then stronger colour jitter",
        "translation_reference_pixels": image_size, "samples": rows}, indent=2), encoding="utf-8")
    return path


def _base_receipt(fitting, chosen):
    return {"version": 1, "fitting_membership_sha256": fitting_membership_hash(fitting),
        "selected_examples": [{"image_id": str(row.image_id), "patient_id": str(row.patient_id),
            "target": int(row.target), "source_sha256": sha256(row.image_path)}
            for row in chosen.itertuples()],
        "seed_definition": {"base_seed": PREVIEW_SEED, "variants_per_image": 4,
            "formula": "base_seed + 4 * image_index + variant_index"},
        "severity_definition": {"0": "no observed defect", "1": "minor background-only artifact",
            "2": "useful lesion information lost or implausible lesion appearance", "3": "major lesion damage"},
        "quality_gate": "Every reviewed lesion category must have severity below 2.",
        "sample_coverage": {"framing": None, "hair": None, "lighting": None},
        "preview_sha256": {}, "conditions": {}, "reviewed_by": None, "reviewed_at_utc": None}


def build_augmentation_previews(fitting, specs, output_dir, *, selected_ids=None):
    """Create matched 24 x (original + four weak + four stronger) pages.

    The written receipt is pending. Only a separate explicit visual review may
    fill ratings and approve it. The process restores all captured RNG states.
    """
    chosen = select_quality_examples(fitting, selected_ids)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = _base_receipt(fitting, chosen)
    state = capture_rng_state()
    try:
        transforms = {key: make_transform(specs[key], training=True) for key in ("A", "B")}
        original_transform = make_transform({**specs["A"], "preprocessing": "none"}, training=False)
        for condition_id in ("A", "B"):
            receipt["conditions"][condition_id] = {"verdict": "pending_review", "reviewed_image_ids": [],
                "spec": specs[condition_id], "reviews": [{"image_id": str(row.image_id),
                    "variant": variant, "seed": _seed(index, variant),
                    **{name: None for name in QUALITY_CATEGORIES}, "notes": ""}
                    for index, row in enumerate(chosen.itertuples()) for variant in range(4)]}
        for page_start in range(0, len(chosen), 2):
            subset = chosen.iloc[page_start:page_start + 2]
            fig, axes = plt.subplots(2, 9, figsize=(20, 5.4), squeeze=False, constrained_layout=True)
            for row_index, row in enumerate(subset.itertuples()):
                global_index = page_start + row_index
                with Image.open(row.image_path) as source:
                    original = source.convert("RGB")
                    axes[row_index, 0].imshow(_display_tensor(original_transform(original)))
                    axes[row_index, 0].set_title(f"Original | target {row.target}\n{row.image_id}", fontsize=8)
                    for condition_index, condition_id in enumerate(("A", "B")):
                        for variant in range(4):
                            seed = _seed(global_index, variant)
                            seed_everything(seed)
                            ax = axes[row_index, 1 + condition_index * 4 + variant]
                            ax.imshow(_display_tensor(transforms[condition_id](original)))
                            ax.set_title(f"{condition_id} variant {variant + 1}\nseed {seed}", fontsize=8)
                for ax in axes[row_index]:
                    ax.axis("off")
            fig.suptitle("Fitting-only QA: originals / existing weak / predefined stronger", fontsize=13)
            path = save_figure(fig, output_dir / f"augmentation_matched_{page_start // 2 + 1:02d}.png", dpi=140)
            receipt["preview_sha256"][str(path.resolve())] = sha256(path)
            plt.close(fig)
    finally:
        restore_rng_state(state)
    receipt["review_sha256"] = canonical_hash(receipt)
    (output_dir.parent / "quality_review.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    chosen[["image_id", "patient_id", "target"]].to_csv(output_dir.parent / "quality_sample.csv", index=False)
    save_sampled_geometry(chosen, output_dir.parent / "augmentation_sampled_geometry.json")
    return {"selected": chosen, "receipt": receipt,
            "preview_paths": [Path(path) for path in receipt["preview_sha256"]]}


def build_clahe_previews(fitting, chosen, spec, output_dir):
    """Create fitting-only original/corrected pairs and an unapproved C receipt."""
    chosen = select_quality_examples(fitting, list(_frame_ids(chosen).image_id))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = _base_receipt(fitting, chosen)
    receipt["seed_definition"] = {"variants_per_image": 1, "stochastic": False}
    receipt["conditions"]["C"] = {"verdict": "pending_review", "reviewed_image_ids": [], "spec": spec,
        "reviews": [{"image_id": str(row.image_id), "variant": 0, "seed": None,
            **{name: None for name in QUALITY_CATEGORIES}, "notes": ""} for row in chosen.itertuples()]}
    state = capture_rng_state()
    try:
        unchanged = make_transform({**spec, "preprocessing": "none"}, training=False)
        correction = make_transform(spec, training=False)
        for page_start in range(0, len(chosen), 4):
            fig, axes = plt.subplots(4, 2, figsize=(8, 12), squeeze=False, constrained_layout=True)
            for row_index, row in enumerate(chosen.iloc[page_start:page_start + 4].itertuples()):
                with Image.open(row.image_path) as source:
                    image = source.convert("RGB")
                    axes[row_index, 0].imshow(_display_tensor(unchanged(image)))
                    axes[row_index, 1].imshow(_display_tensor(correction(image)))
                axes[row_index, 0].set_title(f"Original | target {row.target} | {row.image_id}", fontsize=9)
                axes[row_index, 1].set_title("CLAHE: lightness only | clip 2 | 8×8 tiles", fontsize=9)
                for ax in axes[row_index]:
                    ax.axis("off")
            fig.suptitle("Fitting-only deterministic correction QA", fontsize=13)
            path = save_figure(fig, output_dir / f"clahe_pairs_{page_start // 4 + 1:02d}.png", dpi=130)
            receipt["preview_sha256"][str(path.resolve())] = sha256(path)
            plt.close(fig)
    finally:
        restore_rng_state(state)
    receipt["review_sha256"] = canonical_hash(receipt)
    (output_dir.parent / "quality_review.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return {"selected": chosen, "receipt": receipt,
            "preview_paths": [Path(path) for path in receipt["preview_sha256"]]}


def validate_quality_review(receipt, condition_id, *, fitting=None, verify_files=True):
    """Validate a human/agent visual review; never infer safety from sample pixels."""
    receipt = json.loads(Path(receipt).read_text()) if isinstance(receipt, (str, Path)) else receipt
    payload = {key: value for key, value in receipt.items() if key != "review_sha256"}
    if receipt.get("review_sha256") != canonical_hash(payload):
        raise ValueError("Quality receipt hash is invalid.")
    if fitting is not None and receipt["fitting_membership_sha256"] != fitting_membership_hash(fitting):
        raise ValueError("Quality preview belongs to different fitting patients.")
    selected = receipt.get("selected_examples", [])
    if len(selected) != 24 or len({row["patient_id"] for row in selected}) != 24:
        raise ValueError("Quality review requires 24 distinct fitting patients.")
    if sorted(row["target"] for row in selected) != [0] * 12 + [1] * 12:
        raise ValueError("Quality review requires twelve images per class.")
    condition = receipt.get("conditions", {}).get(condition_id, {})
    if condition.get("verdict") not in {"passed", "quality_blocked"}:
        raise ValueError("Quality gate requires explicit completed visual review.")
    if not receipt.get("reviewed_by") or not receipt.get("reviewed_at_utc"):
        raise ValueError("Quality gate requires reviewer identity and time.")
    if any(not receipt.get("sample_coverage", {}).get(key) for key in ("framing", "hair", "lighting")):
        raise ValueError("Reviewer must document framing, hair and lighting sample coverage.")
    expected_variants = 1 if condition_id == "C" else 4
    expected = {(row["image_id"], variant) for row in selected for variant in range(expected_variants)}
    reviews = condition.get("reviews", [])
    if len(reviews) != len(expected) or {(row["image_id"], row["variant"]) for row in reviews} != expected:
        raise ValueError("Quality review lacks exact image/variant coverage.")
    if set(condition.get("reviewed_image_ids", [])) != {row["image_id"] for row in selected}:
        raise ValueError("Quality review lacks inspected fitting IDs.")
    values = [row.get(category) for row in reviews for category in QUALITY_CATEGORIES]
    if any(isinstance(value, bool) or not isinstance(value, int) or value not in range(4) for value in values):
        raise ValueError("Every quality category must have an explicit severity 0–3.")
    failed = any(value >= 2 for value in values)
    if (condition["verdict"] == "passed") == failed:
        raise ValueError("Quality verdict disagrees with the declared lesion-safety gate.")
    if verify_files:
        if not receipt.get("preview_sha256"):
            raise ValueError("Quality review must bind saved preview pages.")
        for path, expected_hash in receipt["preview_sha256"].items():
            if not Path(path).is_file() or sha256(path) != expected_hash:
                raise ValueError("Quality preview file changed after review.")
        if fitting is not None:
            paths = _frame_ids(fitting).set_index("image_id").image_path
            for row in selected:
                if row["image_id"] not in paths or sha256(paths[row["image_id"]]) != row["source_sha256"]:
                    raise ValueError("Fitting quality source image changed after review.")
    return {"verdict": condition["verdict"], "review_sha256": receipt["review_sha256"],
            "n_images": 24, "n_variants": len(reviews)}


def _resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def load_condition_outputs(project_root, summary):
    """Read committed conditions; an unequal horizon never yields an AP delta."""
    predictions, epoch12, histories, records = {}, {}, [], []
    conditions = summary.get("conditions", {})
    conditions = list(conditions.values()) if isinstance(conditions, dict) else conditions
    for record in conditions:
        condition_id = record["condition_id"]
        identity = record.get("pipeline_id", CONDITION_LABELS.get(condition_id, condition_id))
        records.append({**record, "pipeline_id": identity})
        if record.get("history"):
            histories.append(pd.DataFrame(record["history"]).assign(pipeline_id=identity,
                condition_id=condition_id, selected_epoch=record.get("best_epoch"),
                run_status=record.get("status"), loss_weight="sqrt_ratio"))
        if record.get("status") != "complete":
            continue
        if record.get("completed_epochs") != HORIZON:
            raise ValueError("A complete controlled condition must contain twelve epochs.")
        if not record.get("predictions_file") or not record.get("epoch12_predictions_file"):
            raise ValueError("Complete condition lacks best and epoch-12 predictions.")
        for destination, path_key, metric_key in ((predictions, "predictions_file", "best_metrics"),
                                                 (epoch12, "epoch12_predictions_file", "epoch12_metrics")):
            path = _resolve(project_root, record[path_key])
            frame = pd.read_csv(path)
            frame["pipeline_id"] = identity
            frame["condition_id"] = condition_id
            metrics = probability_metrics(frame.target, frame.base_probability)
            if record.get(metric_key) and any(not np.isclose(value, record[metric_key][name], rtol=0, atol=1e-12)
                                             for name, value in metrics.items()):
                raise ValueError(f"Saved condition metric mismatch: {identity}/{metric_key}")
            destination[identity] = frame
    assert_matching_predictions(predictions)
    assert_matching_predictions(epoch12)
    return {"summary": summary, "records": records, "predictions": predictions,
            "epoch12_predictions": epoch12,
            "histories": pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()}


def condition_comparisons(bundle, *, margin=.005):
    frames = bundle["predictions"]
    ids = {str(frame.condition_id.iloc[0]): identity for identity, frame in frames.items()}
    rows = []
    for candidate in ("B", "C"):
        complete = "A" in ids and candidate in ids
        row = {"comparison": f"{candidate} - A", "status": "complete" if complete else "incomplete",
               "ap_delta": np.nan, "screening_margin": margin, "promising_b0_hypothesis": False,
               "interpretation": "Unequal or unavailable twelve-epoch results cannot rank conditions."}
        if complete:
            assert_matching_predictions({key: frames[ids[key]] for key in ("A", candidate)})
            scores = {key: probability_metrics(frames[ids[key]].target, frames[ids[key]].base_probability)
                      for key in ("A", candidate)}
            delta = scores[candidate]["average_precision"] - scores["A"]["average_precision"]
            row.update(ap_delta=delta, promising_b0_hypothesis=bool(delta >= margin),
                interpretation=("Promising exploratory B0 hypothesis; no statistical proof or architecture transfer."
                    if delta >= margin else "Retain weak/unchanged development reference; no global candidate pruning."))
        rows.append(row)
    return pd.DataFrame(rows)


def render_condition_report(bundle, run_dir):
    """Save genuine curves/tables and return figure objects for notebook display."""
    run_dir = Path(run_dir)
    figures_dir, tables_dir = run_dir / "figures", run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    frames, final = bundle["predictions"], bundle["epoch12_predictions"]
    history = bundle["histories"]
    epochs = {row["pipeline_id"]: row.get("best_epoch") for row in bundle["records"]}
    outputs, figure_paths, resources = [], [], pd.DataFrame()
    def add(fig, name):
        figure_paths.append(save_figure(fig, figures_dir / name))
        outputs.append(fig)
    if not history.empty:
        prevalence = next(iter(frames.values())).target.mean() if frames else np.nan
        add(plot_learning_curves(history, prevalence,
            "Exploratory B0 conditions: twelve-epoch cap; committed epochs only"), "learning_curves.png")
        add(plot_training_losses(history,
            "Matched fitting objective: square-root positive-weighted BCE"), "training_loss.png")
        history.to_csv(tables_dir / "committed_learning_history.csv", index=False)
        resources = history.groupby("pipeline_id").agg(
            measured_training_minutes=("epoch_seconds", lambda values: values.sum() / 60),
            peak_allocated_GiB=("peak_memory_bytes", lambda values: values.max() / 2**30),
            committed_epochs=("epoch", "max")).reset_index()
        resources.to_csv(tables_dir / "measured_time_and_memory.csv", index=False)
    metric_rows = metric_table(frames, threshold=.5, epochs=epochs) if frames else pd.DataFrame()
    final_rows = metric_table(final, threshold=.5, epochs={key: HORIZON for key in final}) if final else pd.DataFrame()
    metric_rows.to_csv(tables_dir / "best_checkpoint_metrics_raw_threshold_0_5.csv", index=False)
    final_rows.to_csv(tables_dir / "epoch12_metrics_raw_threshold_0_5.csv", index=False)
    comparison = condition_comparisons(bundle)
    comparison.to_csv(tables_dir / "exploratory_ap_deltas.csv", index=False)
    if frames:
        add(plot_roc_precision_recall(frames, epochs,
            "Exploratory B0 highest-AP checkpoints on matching inner patients"), "best_roc_precision_recall.png")
        add(plot_confusion_matrices(frames, epochs, threshold=.5,
            title="Exploratory classification: declared raw probability ≥ 0.5"), "best_confusion_threshold_0_5.png")
        reliability, bins = plot_raw_reliability(frames)
        add(reliability, "best_raw_reliability.png")
        bins.to_csv(tables_dir / "raw_reliability_support.csv", index=False)
        for identity, frame in frames.items():
            safe_identity = identity.replace("/", "__").replace(" ", "_")
            for name, table in curve_tables(frame.target, frame.base_probability).items():
                if isinstance(table, pd.DataFrame):
                    table.to_csv(tables_dir / f"{safe_identity}_{name}.csv", index=False)
    if final:
        add(plot_roc_precision_recall(final, {key: HORIZON for key in final},
            "Exploratory B0 epoch-12 checkpoints on matching inner patients"), "epoch12_roc_precision_recall.png")
    return {"figures": outputs, "figure_paths": figure_paths, "metrics": metric_rows,
            "resources": resources,
            "epoch12_metrics": final_rows, "comparisons": comparison}
