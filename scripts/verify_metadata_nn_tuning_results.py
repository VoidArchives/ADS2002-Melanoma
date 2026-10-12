"""Compatibility CLI for current ROC-first v2 result verification.

Historical AP helpers remain importable by the independent v2 verifier. Running
this command verifies v2 evidence and writes only its new v2 verification receipt.
The pre-ROC script is archived; existing AP verification receipts remain protected.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.metadata_model import file_sha256
from melanoma_detection.metadata_nn_tuning import BASE_SEEDS, load_tuning_outputs, recipes, select_recipe

KEYS = ["image_name", "patient_id", "fold", "target"]
GRID_FIELDS = ["recipe_id", "recipe_order", "hidden_units", "learning_rate", "dropout"]
SCORE_FIELDS = ["mean_inner_average_precision", "std_inner_average_precision"] + [
    f"inner_{inner}_average_precision" for inner in range(3)
]


def exact_frame(actual, expected, message):
    """Compare values exactly while allowing CSV integer/float dtype inference."""
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                      check_dtype=False, check_exact=True)
    except AssertionError as error:
        raise ValueError(message) from error


def verify_grid_and_selection(candidates, selected, condition):
    required = {"condition", "outer_fold", *GRID_FIELDS, *SCORE_FIELDS}
    if not (required | {"selected"}).issubset(candidates) or not (required | {"selection_margin"}).issubset(selected):
        raise ValueError("Candidate/selection records lack their frozen grid or score fields.")
    if len(candidates) != 180 or len(selected) != 5:
        raise ValueError("Expected 36 candidates per outer fold and five selected recipes.")
    for table in (candidates, selected):
        if not table.condition.eq(condition).all() or set(table.outer_fold) != set(range(5)):
            raise ValueError("Candidate/selection condition or outer fold identities differ.")
    if not selected.groupby("outer_fold").size().eq(1).all():
        raise ValueError("Exactly one selected recipe per outer fold is required.")
    if not pd.api.types.is_bool_dtype(candidates.selected):
        raise ValueError("Candidate selected flags must be explicit booleans.")
    grid = pd.DataFrame(recipes())[GRID_FIELDS]
    for outer in range(5):
        rows = candidates.loc[candidates.outer_fold.eq(outer)].sort_values("recipe_order")
        exact_frame(rows[GRID_FIELDS], grid, "Candidate parameters/order differ from the frozen 36-recipe grid.")
        winner = select_recipe(rows)
        if not np.array_equal(rows.selected.to_numpy(), rows.recipe_id.eq(winner["recipe_id"]).to_numpy()):
            raise ValueError("Candidate selected flags do not identify the AP/order winner.")
        fields = ["condition", "outer_fold", *GRID_FIELDS, *SCORE_FIELDS]
        declared = selected.loc[selected.outer_fold.eq(outer)]
        exact_frame(declared[fields], pd.DataFrame([winner])[fields],
                    "Selected recipe fields differ from the complete AP/order winner record.")
        ordered_scores = sorted(rows.mean_inner_average_precision, reverse=True)
        if not np.isclose(declared.selection_margin.iloc[0], ordered_scores[0] - ordered_scores[1], rtol=0, atol=1e-14):
            raise ValueError("Selected recipe margin differs from its two leading candidates.")


def verified_inner_groups(memberships, oof):
    required = {*KEYS, "outer_fold", "inner_fold"}
    if not required.issubset(memberships) or memberships[list(required)].isna().any().any():
        raise ValueError("Saved inner memberships lack complete identities.")
    if set(memberships.outer_fold) != set(range(5)) or memberships.duplicated(["outer_fold", "image_name"]).any():
        raise ValueError("Saved inner memberships require unique fitting records in all outer folds.")
    groups = {}
    for outer in range(5):
        rows = memberships.loc[memberships.outer_fold.eq(outer)]
        expected = oof.loc[oof.fold.ne(outer), KEYS].sort_values("image_name")
        exact_frame(rows[KEYS].sort_values("image_name"), expected,
                    "Saved inner membership coverage differs from the outer fitting population.")
        if set(rows.inner_fold) != {0, 1, 2} or not rows.groupby("patient_id").inner_fold.nunique().eq(1).all():
            raise ValueError("Saved inner memberships must preserve three patient-isolated groups.")
        for inner in range(3):
            group = rows.loc[rows.inner_fold.eq(inner), KEYS].sort_values("image_name").reset_index(drop=True)
            if set(group.target) != {0, 1}:
                raise ValueError("Every saved inner assessment group requires both classes.")
            groups[(outer, inner)] = group
    return groups


def verify_fit_log(log, receipt, selected, condition):
    required = {"condition", "outer_fold", "stage", "phase", "fits", "reused_fits", "executed_this_invocation",
                "inner_fold", "base_seed", *GRID_FIELDS}
    if not required.issubset(log) or not log.condition.eq(condition).all() or set(log.outer_fold) != set(range(5)):
        raise ValueError("Fit log lacks the required condition/fold/protocol fields.")
    counts = log[["fits", "reused_fits", "executed_this_invocation"]].to_numpy(float)
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Fit log counters must be non-negative integers.")
    if (log.executed_this_invocation > log.fits).any():
        raise ValueError("Invocation fit counts cannot exceed recorded completed fits.")
    totals = {name: int(log[name].sum()) for name in ("fits", "reused_fits", "executed_this_invocation")}
    expected_total = (3960, 0) if condition == "original" else (3270, 690)
    if (totals["fits"], totals["reused_fits"]) != expected_total:
        raise ValueError("Independent fit-log sums differ from the frozen experiment.")
    if (receipt["executed_fits"], receipt["reused_fits"], receipt["executed_fits_this_invocation"]) != tuple(totals.values()):
        raise ValueError("Completion receipt fit counters disagree with the independent fit log.")
    nn = log.loc[log.stage.astype("string").str.startswith("nn_", na=False)]
    stages = ["nn_inner", "nn_outer_tuned"] + (["nn_outer_fixed"] if condition == "original" else [])
    if set(nn.stage) != set(stages) or not nn.fits.eq(1).all() or not nn.reused_fits.eq(0).all():
        raise ValueError("NN fit log has unexpected stages or phase counters.")
    stage_totals = {}
    for stage in stages:
        rows = nn.loc[nn.stage.eq(stage)]
        fields = ["outer_fold", "recipe_id", "inner_fold", "base_seed", "phase"]
        expected = []
        for outer in range(5):
            grid = recipes() if stage == "nn_inner" else [
                selected.loc[selected.outer_fold.eq(outer)].iloc[0] if stage == "nn_outer_tuned" else recipes()[0]
            ]
            for recipe in grid:
                for inner in (range(3) if stage == "nn_inner" else [-1]):
                    for seed in BASE_SEEDS:
                        for phase in ("stopping", "refit"):
                            expected.append({"outer_fold": outer, **{field: recipe[field] for field in GRID_FIELDS},
                                             "inner_fold": inner, "base_seed": seed, "phase": phase})
        columns = [*fields, *[field for field in GRID_FIELDS if field != "recipe_id"]]
        exact_frame(rows[columns].sort_values(fields), pd.DataFrame(expected)[columns].sort_values(fields),
                    "NN fit phases, seeds, inner groups or recipe parameters differ from the complete frozen protocol.")
        stage_totals[stage] = {phase: int(rows.loc[rows.phase.eq(phase), "fits"].sum()) for phase in ("stopping", "refit")}
    remaining = log.loc[~log.index.isin(nn.index)]
    expected_phases = ({"classical_fixed": 3, "classical_inner": 126, "classical_tuned": 3}
                       if condition == "original" else {"reused_fixed_nn": 6, "reused_classical": 132})
    counter = "fits" if condition == "original" else "reused_fits"
    other_counter = "reused_fits" if condition == "original" else "fits"
    if set(remaining.phase) != set(expected_phases) or not remaining[other_counter].eq(0).all():
        raise ValueError("Classical/reused fit phases or counters differ from the frozen protocol.")
    for outer in range(5):
        for phase, expected in expected_phases.items():
            rows = remaining.loc[remaining.outer_fold.eq(outer) & remaining.phase.eq(phase)]
            expected_rows = expected if condition == "original" else 1
            if len(rows) != expected_rows or int(rows[counter].sum()) != expected:
                raise ValueError("Classical/reused fit totals differ within an outer fold.")
    return {"totals": totals, "nn_stage_phases": stage_totals,
            "classical_or_reused_phases": {phase: int(remaining.loc[remaining.phase.eq(phase), counter].sum())
                                          for phase in expected_phases}}


def verify(condition):
    bundle = load_tuning_outputs(ROOT, condition)
    if bundle["status"] != "complete":
        raise RuntimeError(f"{condition} is pending; complete results are required.")
    directory, oof, receipt = bundle["directory"], bundle["oof"], bundle["receipt"]
    candidates = pd.read_csv(directory / "candidates.csv", float_precision="round_trip")
    selected = pd.read_csv(directory / "selected_parameters.csv", float_precision="round_trip")
    verify_grid_and_selection(candidates, selected, condition)
    inner_groups = verified_inner_groups(pd.read_csv(directory / "inner_memberships.csv"), oof)
    fit_counts = verify_fit_log(pd.read_csv(directory / "fit_log.csv", float_precision="round_trip"), receipt, selected, condition)
    recomputed = []
    for outer in range(5):
        rows = candidates.loc[candidates.outer_fold.eq(outer)].sort_values("recipe_order")
        outer_patients = set(oof.loc[oof.fold.eq(outer), "patient_id"])
        for _, recipe in rows.iterrows():
            inner_scores = []
            for inner in range(3):
                predictions = []
                for seed in BASE_SEEDS:
                    path = directory / "trials" / f"outer_{outer}" / recipe.recipe_id / f"inner_{inner}" / f"seed_{seed}" / "refit/predictions.csv"
                    frame = pd.read_csv(path, float_precision="round_trip")
                    if set(frame.patient_id) & outer_patients:
                        raise ValueError("Outer patients appear in an inner assessment.")
                    if not frame.image_name.is_unique or set(frame.target.unique()) != {0, 1}:
                        raise ValueError("Inner predictions require unique images and both classes.")
                    exact_frame(frame[KEYS].sort_values("image_name"), inner_groups[(outer, inner)],
                                "Inner predictions differ from their exact saved assessment membership group.")
                    if predictions:
                        pd.testing.assert_frame_equal(frame.drop(columns="score"), predictions[0].drop(columns="score"))
                    predictions.append(frame)
                # Restore the saved Torch float32 outputs before float64
                # ensemble scoring, exactly as the original fitting phase did.
                matrix = np.column_stack([frame.score.to_numpy().astype(np.float32).astype(float) for frame in predictions])
                if not np.isfinite(matrix).all() or ((matrix < 0) | (matrix > 1)).any():
                    raise ValueError("Inner probabilities are invalid.")
                ap = average_precision_score(predictions[0].target, matrix.mean(axis=1))
                if not np.isclose(ap, recipe[f"inner_{inner}_average_precision"], rtol=0, atol=1e-14):
                    raise ValueError("Saved inner AP does not match the equally averaged seed predictions.")
                inner_scores.append(ap)
            mean = float(np.mean(inner_scores))
            if not np.isclose(mean, recipe.mean_inner_average_precision, rtol=0, atol=1e-14):
                raise ValueError("Candidate mean AP does not average its three inner fold scores equally.")
            if not np.isclose(np.std(inner_scores), recipe.std_inner_average_precision, rtol=0, atol=1e-14):
                raise ValueError("Candidate AP standard deviation differs from its three inner fold scores.")
            recomputed.append({"outer_fold": outer, "recipe_id": recipe.recipe_id,
                               "mean_inner_average_precision": mean})
    for prefix, mean in [("p_nn_tuned_seed_", "p_nn_tuned"), ("p_nn_seed_", "p_nn_mean")]:
        expected = oof[[prefix + str(seed) for seed in BASE_SEEDS]].mean(axis=1)
        if not np.allclose(expected, oof[mean], rtol=1e-7, atol=1e-8):
            raise ValueError("Outer ensemble disagrees with the equal three-seed average.")
    expected = {"original": (6864, 3960, 0), "dedup_v1": (6706, 3270, 690)}[condition]
    if (len(oof), receipt["executed_fits"], receipt["reused_fits"]) != expected:
        raise ValueError("Cohort or executed/reused fit counts differ from the frozen experiment.")
    source = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    for relative, digest in source["source_sha256"].items():
        if file_sha256(ROOT / relative) != digest:
            raise ValueError("Frozen fitting sources changed during execution.")
    output = directory / "report" / "independent_verification"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recomputed).to_csv(output / "recomputed_candidate_scores.csv", index=False)
    record = {"status": "complete", "condition": condition, "candidate_ensembles_checked": 540,
              "candidates_checked": 180, "outer_predictions": len(oof), "outer_folds": 5,
              "patients": int(oof.patient_id.nunique()), "executed_fits": expected[1], "reused_fits": expected[2],
              "fit_log_checks": fit_counts, "inner_prediction_memberships_checked": 1620,
              "ensemble_scoring": "AP after equal probability averaging, with unweighted inner-fold mean",
              "selection_and_training_source_checks": "passed", "reserved_test_loaded": False,
              "recomputed_scores_sha256": file_sha256(output / "recomputed_candidate_scores.csv")}
    (output / "receipt.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["both", "original", "dedup_v1"], default="both")
    args = parser.parse_args(argv)
    # Import only after parsing and only when invoked, avoiding a helper-import cycle.
    from verify_metadata_tuning_v2_results import verify as verify_current
    for condition in (["original", "dedup_v1"] if args.condition == "both" else [args.condition]):
        print(json.dumps(verify_current(condition), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
