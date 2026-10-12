"""Reconstruct v2 choices and OOF metrics from sealed predictions, without fitting."""
from pathlib import Path
import argparse
import json
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import ParameterGrid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.metadata_model import file_sha256
from melanoma_detection.metadata_tuning_v2 import load_tuning_outputs
from melanoma_detection.dedup_metadata_suite import GRIDS
from verify_metadata_nn_tuning_results import verified_inner_groups, exact_frame

KEYS = ["image_name", "patient_id", "fold", "target"]
METRICS = ["roc_auc", "average_precision", "log_loss", "brier"]


def score(y, p):
    p = np.asarray(p, float)
    if set(y) != {0, 1} or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Complete finite probabilities and both classes are required")
    return dict(roc_auc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)),
                log_loss=float(log_loss(y, p, labels=[0, 1])), brier=float(brier_score_loss(y, p)))


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def same_native_probabilities(left, right):
    # CSV may contain short float32 decimals while a mixed-dtype concat writes their exact promotion.
    return np.array_equal(left, right) or np.array_equal(np.asarray(left, np.float32), np.asarray(right, np.float32))


def expected_recipes():
    current = (32, .001, .2)
    tuples = [current] + [(h, lr, d) for h in (8, 16, 32, 64) for lr in (.0003, .001, .003)
                          for d in (0., .2, .4) if (h, lr, d) != current]
    return pd.DataFrame([dict(recipe_id=f"r{i:02d}", recipe_order=i, hidden_units=h,
                              learning_rate=lr, dropout=d) for i, (h, lr, d) in enumerate(tuples)])


def winner_check(rows, declared, order, identity):
    ordered = rows.sort_values(["mean_inner_roc_auc", order], ascending=[False, True], kind="stable")
    winner = ordered.iloc[0]
    if len(declared) != 1 or declared.iloc[0][identity] != winner[identity]:
        raise ValueError("Selection differs from the independently reconstructed ROC/tie-order winner")
    if not np.array_equal(rows.selected.to_numpy(), rows[identity].eq(winner[identity]).to_numpy()):
        raise ValueError("Selected flags differ from the ROC winner")
    fields = (["recipe_id", "recipe_order", "hidden_units", "learning_rate", "dropout"] if identity == "recipe_id"
              else ["candidate_index", "parameters", "model"])
    if any(declared.iloc[0][field] != winner[field] for field in fields):
        raise ValueError("Selected parameters differ from the winning complete candidate record")
    for field in ["selection_score", "mean_inner_roc_auc", "mean_inner_average_precision"]:
        value = winner.mean_inner_roc_auc if field == "selection_score" else winner[field]
        if not np.isclose(declared.iloc[0][field], value, atol=1e-14, rtol=0):
            raise ValueError("Selected candidate score changed")
    margin = ordered.mean_inner_roc_auc.iloc[0] - ordered.mean_inner_roc_auc.iloc[1]
    if not np.isclose(declared.selection_margin.iloc[0], margin, atol=1e-14, rtol=0):
        raise ValueError("Selection margin differs from the leading two ROC scores")


def prediction(root, ref, expected):
    path = (root / ref.source_prediction).resolve()
    receipt_path = (root / ref.source_receipt).resolve()
    if not path.is_relative_to(root / "results") or not receipt_path.is_relative_to(root / "results"):
        raise ValueError("Prediction reference escaped the results directory")
    if file_sha256(path) != ref.source_prediction_sha256 or file_sha256(receipt_path) != ref.source_receipt_sha256:
        raise ValueError("Referenced prediction/receipt bytes changed")
    values = read(path)
    column = ref.get("source_probability_column")
    aggregate = isinstance(column, str) and bool(column)
    if not aggregate and (len(values) != len(expected) or set(values.image_name) != set(expected.image_name) or values.image_name.duplicated().any()):
        raise ValueError("Child prediction file has missing, repeated or extraneous validation rows")
    parent = json.loads(receipt_path.read_text(encoding="utf-8"))
    if aggregate:
        parent_dir = root / parent["output_root"]
        if path != parent_dir / "oof.csv" or parent.get("condition") != ref.condition:
            raise ValueError("Aggregate reuse must reference the explicit condition's sealed parent OOF")
        if ref.model == "nn":
            fixed = ref.get("source_stage") == "nn_outer_fixed"
            selected = read(parent_dir / "selected_parameters.csv")
            expected_recipe = expected_recipes().iloc[0] if fixed else selected.loc[selected.outer_fold.eq(ref.outer_fold)].iloc[0]
            if any(ref[field] != expected_recipe[field] for field in ["recipe_id", "recipe_order", "hidden_units", "learning_rate", "dropout"]):
                raise ValueError("Aggregate NN reuse does not match the selected or fixed source procedure")
            suffix = f"p_nn_seed_{int(ref.base_seed)}" if fixed else f"p_nn_tuned_seed_{int(ref.base_seed)}"
        else:
            selected = read(parent_dir / "classical_selected_parameters.csv")
            row = selected.loc[selected.outer_fold.eq(ref.outer_fold) & selected.model.eq(ref.model)].iloc[0]
            if json.loads(row.parameters) != json.loads(ref.parameters):
                raise ValueError("Aggregate conventional reuse has different AP/ROC parameters")
            suffix = f"p_{ref.model}_tuned"
        if column != suffix or ref.stage not in ["nn_outer_tuned", "classical_outer_tuned"]:
            raise ValueError("Aggregate reuse has an incorrect source probability column or stage")
    else:
        context = parent.get("context", {})
        phase = "refit" if ref.model == "nn" else "classical_inner" if ref.stage == "classical_inner" else "classical_tuned"
        fields = ["condition", "outer_fold", "model"] if ref.model != "nn" else ["condition", "outer_fold", "stage", "inner_fold", "base_seed", "recipe_id", "recipe_order", "hidden_units", "learning_rate", "dropout"]
        if context.get("phase") != phase or any(context.get(field) != ref[field] for field in fields):
            raise ValueError("Source phase context does not identify the requested procedure")
        if ref.model == "nn":
            stop = (int(ref.outer_fold) + 1 + [42, 142, 242].index(int(ref.base_seed))) % 5 if ref.stage == "nn_outer_tuned" else int(ref.stop_group)
            if context.get("training_seed") != ref.base_seed + ref.outer_fold or context.get("stop_group") != stop:
                raise ValueError("NN outer seed or stopping/refit procedure changed")
        elif context.get("candidate_index") != ref.candidate_index or context.get("parameters") != json.loads(ref.parameters):
            raise ValueError("Conventional source parameters/order differ")
    values = values.set_index("image_name").loc[expected.image_name].reset_index()
    exact_frame(values[KEYS], expected[KEYS], "Prediction keys differ from excluded inner/outer patients")
    column = column if isinstance(column, str) and column else "score"
    return values[column].to_numpy(float)


def check_ledger(log, condition, refs, chosen, selected):
    counters = ["executed_fits", "external_reused_fits", "resumed_fits", "executed_this_invocation", "rescoring_operations"]
    values = log[counters].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Fit/scoring counters must be nonnegative integers")
    if not log.condition.eq(condition).all() or not log.selection_metric.eq("roc_auc").all():
        raise ValueError("Fit ledger condition or metric differs")
    for stage, expected, fields in [
        ("nn_inner", [(o, f"r{r:02d}", i, s, p) for o in range(5) for r in range(36) for i in range(3) for s in [42,142,242] for p in ["stopping","refit"]], ["outer_fold","recipe_id","inner_fold","base_seed","phase"]),
        ("classical_inner", [(o,m,c,i,"classical_inner") for o in range(5) for m,g in GRIDS.items() for c in range(len(ParameterGrid(g))) for i in range(3)], ["outer_fold","model","candidate_index","inner_fold","phase"]),
        ("nn_outer_tuned", [(o,s,p) for o in range(5) for s in [42,142,242] for p in ["stopping","refit"]], ["outer_fold","base_seed","phase"]),
        ("classical_outer_tuned", [(o,m,"classical_tuned") for o in range(5) for m in GRIDS], ["outer_fold","model","phase"]),
    ]:
        rows = log.loc[log.stage.eq(stage) & log.rescoring_operations.eq(0)]
        actual = [tuple(row) for row in rows[fields].to_numpy()]
        if len(actual) != len(expected) or set(actual) != set(expected):
            raise ValueError(f"Missing/repeated/unexpected fit ledger context: {stage}")
        if stage == "nn_inner" or (stage == "classical_inner" and condition == "original"):
            if not rows.external_reused_fits.eq(1).all() or not rows.executed_fits.eq(0).all():
                raise ValueError("Expected historical inner fit reuse was not explicitly recorded")
        elif stage == "classical_inner" and not rows.executed_fits.eq(1).all():
            raise ValueError("Clean conventional inner fits were substituted rather than reconstructed")
    baseline = log.loc[log.stage.eq("fixed_baselines")]
    if len(baseline) != 5 or set(baseline.outer_fold) != set(range(5)) or not baseline.external_reused_fits.eq(9).all():
        raise ValueError("Fixed baseline reuse buckets are incomplete")
    scoring = log.loc[log.rescoring_operations.eq(1)]
    if len(scoring.loc[scoring.stage.eq("nn_inner")]) != 540 or len(scoring.loc[scoring.stage.eq("classical_inner")]) != 630:
        raise ValueError("Candidate rescoring coverage differs")
    if scoring[counters[:-1]].to_numpy().any():
        raise ValueError("Scoring operations were counted as fits")
    fits = log.loc[log.rescoring_operations.eq(0)]
    if not (fits.executed_fits + fits.external_reused_fits).eq(np.where(fits.stage.eq("fixed_baselines"), 9, 1)).all():
        raise ValueError("Fit phases were duplicated between external and native counters")
    if (fits.resumed_fits > fits.executed_fits).any() or not fits.executed_this_invocation.eq(fits.executed_fits - fits.resumed_fits).all():
        raise ValueError("Resume phases were counted as new invocations")
    if not fits.loc[fits.external_reused_fits.gt(0), "provenance"].eq("external_reused").all():
        raise ValueError("External fit counts lack explicit cross-run provenance")
    for _, row in fits.loc[~fits.stage.eq("fixed_baselines")].iterrows():
        rr = refs.loc[refs.stage.eq(row.stage) & refs.outer_fold.eq(row.outer_fold) & refs.model.eq(row.model)]
        if row.model == "nn":
            rr = rr.loc[rr.base_seed.eq(row.base_seed)]
            if row.stage == "nn_inner":
                rr = rr.loc[rr.recipe_id.eq(row.recipe_id) & rr.inner_fold.eq(row.inner_fold)]
            fields = ["recipe_id", "recipe_order", "hidden_units", "learning_rate", "dropout"]
        else:
            rr = rr.loc[rr.candidate_index.eq(row.candidate_index)]
            if row.stage == "classical_inner":
                rr = rr.loc[rr.inner_fold.eq(row.inner_fold)]
            fields = ["candidate_index", "parameters"]
        if len(rr) != 1 or any(row[field] != rr.iloc[0][field] for field in fields):
            raise ValueError("Fit ledger parameters do not match its unique prediction reference")
        ref = rr.iloc[0]
        path = ROOT / ref.source_receipt
        if row.model == "nn" and row.phase == "stopping" and path.parent.name == "refit":
            path = path.parent.parent / "stopping" / "receipt.json"
        if row.external_reused_fits:
            if (ROOT / row.source_receipt).resolve() != path.resolve():
                raise ValueError("External ledger phase points to a different stopping/refit source")
        source = json.loads(path.read_text(encoding="utf-8"))
        if path.name == "receipt.json" and source.get("context"):
            context = source["context"]
            for field in ["condition", "outer_fold", *fields, *( ["stage", "base_seed", "inner_fold"] if row.model == "nn" else ["model"] )]:
                value = json.loads(row.parameters) if field == "parameters" else row[field]
                if context.get(field) != value:
                    raise ValueError("Fit phase receipt context differs from ledger")
            if context.get("phase") != row.phase:
                raise ValueError("Stopping/refit phase receipt was substituted")


def check_roles(roles, oof, memberships):
    keys = ["stage", "outer_fold", "inner_fold", "base_seed"]
    if roles.duplicated(keys + ["patient_id"]).any():
        raise ValueError("Role records repeat a patient within a procedure")
    inner = roles.stage.eq("nn_inner")
    if not roles.loc[inner & roles.fold.eq(roles.outer_fold), "role"].eq("outer_excluded").all():
        raise ValueError("Held-out outer patients entered preprocessing/duration/fitting")
    if not roles.loc[~inner & roles.fold.eq(roles.outer_fold), "role"].eq("validation").all():
        raise ValueError("Outer NN validation roles are incorrect")
    if roles.loc[roles.fold.eq(roles.outer_fold), "refit_included"].any():
        raise ValueError("Held-out patients entered a fresh refit")
    if roles.loc[inner & roles.role.isin(["inner_validation", "validation"]), "refit_included"].any():
        raise ValueError("Inner validation entered a fresh refit")
    for _, rows in roles.groupby(keys, dropna=False):
        fitting = set(rows.loc[rows.role.eq("epoch_fitting"), "patient_id"])
        stopping = set(rows.loc[rows.role.eq("epoch_stopping"), "patient_id"])
        excluded = set(rows.loc[rows.role.isin(["outer_excluded", "inner_validation", "outer_validation", "validation"]), "patient_id"])
        if fitting & stopping or fitting & excluded or stopping & excluded:
            raise ValueError("Patient roles are not isolated")
    patients = oof.groupby("patient_id").agg(fold=("fold", "first"), images=("image_name", "size"))
    if set(roles.stage) != {"nn_inner", "nn_outer_fixed", "nn_outer_tuned"} or len(roles) != 75 * len(patients):
        raise ValueError("Role ledger lacks the complete 75 patient contexts")
    for stage in ["nn_inner", "nn_outer_fixed", "nn_outer_tuned"]:
        for outer in range(5):
            inner_map = memberships.loc[memberships.outer_fold.eq(outer)].groupby("patient_id").inner_fold.first()
            for inner_fold in (range(3) if stage == "nn_inner" else [-1]):
                for seed_index, seed in enumerate([42, 142, 242]):
                    rows = roles.loc[roles.stage.eq(stage) & roles.outer_fold.eq(outer) & roles.inner_fold.eq(inner_fold) & roles.base_seed.eq(seed)].set_index("patient_id")
                    if set(rows.index) != set(patients.index):
                        raise ValueError("Patient context coverage changed")
                    for patient, expected in patients.iterrows():
                        if stage == "nn_inner":
                            stop = [(inner_fold + 1) % 3, (inner_fold + 2) % 3][1 if seed_index == 1 else 0]
                            role = ("outer_excluded" if expected.fold == outer else "validation" if inner_map[patient] == inner_fold
                                    else "epoch_stopping" if inner_map[patient] == stop else "epoch_fitting")
                        else:
                            stop = (outer + 1 + seed_index) % 5
                            role = "validation" if expected.fold == outer else "epoch_stopping" if expected.fold == stop else "epoch_fitting"
                        actual = rows.loc[patient]
                        if actual.role != role or actual.stop_group != stop or actual.images != expected.images or actual.refit_included != (role in ["epoch_fitting", "epoch_stopping"]):
                            raise ValueError("Patient role/stopping group/refit membership differs from frozen rules")


def verify(condition):
    bundle = load_tuning_outputs(ROOT, condition, experiment_version=2, run_id="roc_auc_grid36_v1",
                                 expected_selection_metric="roc_auc")
    if bundle["status"] != "complete":
        raise RuntimeError(f"{condition} remains pending")
    out, oof, receipt = bundle["directory"], bundle["oof"], bundle["receipt"]
    nn, chosen = read(out / "candidates.csv"), read(out / "selected_parameters.csv")
    classical, selected = read(out / "classical_candidates.csv"), read(out / "classical_selected_parameters.csv")
    refs = read(out / "inner_prediction_references.csv")
    groups = verified_inner_groups(read(out / "inner_memberships.csv"), oof)
    if (len(nn), len(chosen), len(classical), len(selected)) != (180, 5, 210, 15):
        raise ValueError("Incomplete frozen candidate/selection coverage")
    for table in (nn, chosen, classical, selected, refs):
        if not table.condition.eq(condition).all() or not table.selection_metric.eq("roc_auc").all():
            raise ValueError("ROC readers were supplied another experiment's evidence")
    reconstructed = 0
    for outer in range(5):
        rows = nn.loc[nn.outer_fold.eq(outer)].sort_values("recipe_order")
        grid = expected_recipes()
        exact_frame(rows[list(grid)], grid, "Frozen NN grid/order changed")
        winner_check(rows, chosen.loc[chosen.outer_fold.eq(outer)], "recipe_order", "recipe_id")
        for _, row in pd.concat([rows, classical.loc[classical.outer_fold.eq(outer)]], ignore_index=True).iterrows():
            results = []
            for inner in range(3):
                group = groups[(outer, inner)]
                rr = refs.loc[refs.outer_fold.eq(outer) & refs.inner_fold.eq(inner) & refs.model.eq(row.model)]
                if row.model == "nn":
                    rr = rr.loc[rr.recipe_id.eq(row.recipe_id)].sort_values("base_seed")
                    if rr.base_seed.tolist() != [42., 142., 242.]:
                        raise ValueError("NN candidate requires all three seeds")
                    # Restore the native float32 NN outputs, then promote before ensemble scoring.
                    p = np.column_stack([prediction(ROOT, ref, group).astype(np.float32) for _, ref in rr.iterrows()]).astype(float).mean(axis=1)
                else:
                    rr = rr.loc[rr.candidate_index.eq(row.candidate_index)]
                    if len(rr) != 1:
                        raise ValueError("Classical candidate lacks exactly one inner prediction")
                    p = prediction(ROOT, rr.iloc[0], group)
                results.append(score(group.target, p))
                reconstructed += 1
            for metric in ["roc_auc", "average_precision"]:
                values = np.array([result[metric] for result in results])
                fields = [*[f"inner_{i}_{metric}" for i in range(3)], f"mean_inner_{metric}", f"std_inner_{metric}"]
                if not np.allclose(row[fields].to_numpy(float), [*values, values.mean(), values.std()], atol=1e-14, rtol=0):
                    raise ValueError("Candidate scores are not reconstructed from saved probabilities")
        for model, params in GRIDS.items():
            rows = classical.loc[classical.outer_fold.eq(outer) & classical.model.eq(model)].sort_values("candidate_index")
            if [json.loads(value) for value in rows.parameters] != list(ParameterGrid(params)):
                raise ValueError("Conventional grid/order changed")
            winner_check(rows, selected.loc[selected.outer_fold.eq(outer) & selected.model.eq(model)], "candidate_index", "candidate_index")
            expected = oof.loc[oof.fold.eq(outer), KEYS].reset_index(drop=True)
            rr = refs.loc[refs.outer_fold.eq(outer) & refs.model.eq(model) & refs.stage.eq("classical_outer_tuned")]
            winner = selected.loc[selected.outer_fold.eq(outer) & selected.model.eq(model)].iloc[0]
            if len(rr) != 1 or rr.iloc[0].candidate_index != winner.candidate_index or json.loads(rr.iloc[0].parameters) != json.loads(winner.parameters):
                raise ValueError("Classical outer reference does not use the reconstructed ROC winner")
            if not same_native_probabilities(prediction(ROOT, rr.iloc[0], expected), oof.loc[oof.fold.eq(outer), f"p_{model}_tuned"]):
                raise ValueError("Selected classical outer predictions do not reproduce OOF")
        expected = oof.loc[oof.fold.eq(outer), KEYS].reset_index(drop=True)
        for seed in [42, 142, 242]:
            rr = refs.loc[refs.outer_fold.eq(outer) & refs.stage.eq("nn_outer_tuned") & refs.base_seed.eq(seed)]
            winner = chosen.loc[chosen.outer_fold.eq(outer)].iloc[0]
            if len(rr) != 1 or any(rr.iloc[0][field] != winner[field] for field in ["recipe_id", "recipe_order", "hidden_units", "learning_rate", "dropout"]):
                raise ValueError("NN outer reference does not use the reconstructed ROC winner")
            if not same_native_probabilities(prediction(ROOT, rr.iloc[0], expected), oof.loc[oof.fold.eq(outer), f"p_nn_tuned_seed_{seed}"]):
                raise ValueError("Selected NN outer predictions do not reproduce OOF")
    for kind in ["tuned", "mean"]:
        columns = [f"p_nn_tuned_seed_{seed}" if kind == "tuned" else f"p_nn_seed_{seed}" for seed in [42, 142, 242]]
        if not np.allclose(oof[columns].to_numpy(np.float32).mean(axis=1), oof[f"p_nn_{kind}"], rtol=0, atol=3e-8):
            raise ValueError("Outer NN ensemble was not averaged before scoring")
    metrics = read(out / "metrics.csv")
    metric_oof = oof.copy()
    # Reconstruct the producer's pre-CSV dtypes without importing its metric helper.
    # Every selected NN phase returns native float32; fixed columns were read from parent CSV as float64.
    for column in ["p_nn_tuned", *[f"p_nn_tuned_seed_{seed}" for seed in [42, 142, 242]]]:
        metric_oof[column] = metric_oof[column].astype(np.float32)
    xgb_refs = refs.loc[refs.stage.eq("classical_outer_tuned") & refs.model.eq("xgboost")]
    if not xgb_refs.provenance.eq("external_reused").any():
        metric_oof["p_xgboost_tuned"] = metric_oof.p_xgboost_tuned.astype(np.float32)
    for column in oof.filter(like="p_"):
        fold_scores = [score(metric_oof.loc[oof.fold.eq(f), "target"], metric_oof.loc[oof.fold.eq(f), column]) for f in range(5)]
        for scope, result in [("mean_fold", {m: np.mean([r[m] for r in fold_scores]) for m in METRICS}),
                              ("pooled", score(metric_oof.target, metric_oof[column]))]:
            actual = metrics.loc[metrics.model.eq(column[2:]) & metrics.scope.eq(scope), METRICS]
            if len(actual) != 1 or not np.allclose(actual.iloc[0], [result[m] for m in METRICS], atol=1e-14, rtol=0):
                raise ValueError("OOF metric summaries are incorrect")
        for fold, result in enumerate(fold_scores):
            actual = metrics.loc[metrics.model.eq(column[2:]) & metrics.scope.eq("fold") & metrics.fold.eq(fold), METRICS]
            if len(actual) != 1 or not np.allclose(actual.iloc[0], [result[m] for m in METRICS], atol=1e-14, rtol=0):
                raise ValueError("Stored individual outer-fold metrics are incorrect")
    check_roles(read(out / "memberships.csv"), oof, read(out / "inner_memberships.csv"))
    log = read(out / "fit_log.csv")
    check_ledger(log, condition, refs, chosen, selected)
    counters = ["executed_fits", "external_reused_fits", "resumed_fits", "executed_this_invocation", "rescoring_operations"]
    counts = {name: int(log[name].sum()) for name in counters}
    if any(counts[name] != receipt[name] for name in counters) or reconstructed != counts["rescoring_operations"]:
        raise ValueError("Fit/rescoring ledger differs from completion receipt")
    if counts["executed_fits"] + counts["external_reused_fits"] != 3960 or counts["executed_fits"] > (45 if condition == "original" else 675):
        raise ValueError("Unexpected fit counts or approved maximum exceeded")
    native = log.loc[log.executed_fits.gt(0)]
    if native.duplicated(["stage", "outer_fold", "model", "candidate_index", "inner_fold", "base_seed", "phase"]).any():
        raise ValueError("Native fit phases were counted twice")
    result = dict(status="complete", experiment_version=2, selection_metric="roc_auc", condition=condition,
                  verified_utc=datetime.now(timezone.utc).isoformat(), verification_source_sha256=file_sha256(Path(__file__)),
                  independently_reconstructed_inner_scores=reconstructed, counts=counts,
                  oof_rows=len(oof), patients=int(oof.patient_id.nunique()), outer_folds=5,
                  test_set_loaded=False, final_model_fitted=False,
                  completion_receipt_sha256=file_sha256(out / "receipt.json"))
    report = out / "report" / "independent_verification.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["both", "original", "dedup_v1"], default="both")
    args = parser.parse_args()
    for condition in (["original", "dedup_v1"] if args.condition == "both" else [args.condition]):
        print(json.dumps(verify(condition), indent=2), flush=True)
