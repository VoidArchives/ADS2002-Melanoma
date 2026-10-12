"""V2 metric isolation, exact cache identity, train-only fitting and resume."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection import metadata_tuning_v2 as m
from melanoma_detection.metadata_model import file_sha256


def toy_frame():
    rows = []
    for fold in range(5):
        for patient in range(12):
            for image in range(2):
                rows.append({"image_name": f"f{fold}p{patient}i{image}", "patient_id": f"p{fold}_{patient}",
                    "fold": fold, "target": int(patient < 3 and image == 0),
                    "age_approx": 20 + patient + fold, "sex": "male" if patient % 2 else "female",
                    "anatom_site_general_challenge": "torso"})
    return pd.DataFrame(rows)


def memberships(frame):
    parts = []
    for outer in range(5):
        part = frame.loc[frame.fold.ne(outer), m.KEYS].copy()
        part["outer_fold"] = outer
        part["inner_fold"] = part.patient_id.str.split("_").str[-1].astype(int) % 3
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


class V2ProtocolTests(unittest.TestCase):
    def setUp(self):
        cache = ROOT / ".pytest_cache"
        cache.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="metadata-v2-", dir=cache)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(ROOT.resolve()))

    def tearDown(self):
        self.temporary.cleanup()

    def context(self, phase="refit"):
        output = m.condition_directory(self.root, "original", "synthetic")
        return {"condition": "original", "selection_metric": "roc_auc", "model": "nn",
                "output_root": output.relative_to(self.root).as_posix(), "phase": phase}

    def test_ensemble_scores_probabilities_before_both_metrics(self):
        y = [0, 0, 1, 1, 0, 1]
        probabilities = np.array([[.8,.1,.2],[.2,.4,.9],[.4,.9,.1],[.3,.6,.8],[.1,.8,.7],[.9,.3,.5]])
        mean = probabilities.mean(axis=1)
        for metric, scorer in (("roc_auc", roc_auc_score), ("average_precision", average_precision_score)):
            returned, score = m.ensemble_score(y, probabilities, metric)
            np.testing.assert_array_equal(returned, mean)
            self.assertEqual(score, scorer(y, mean))
        self.assertNotAlmostEqual(m.ensemble_score(y, probabilities, "average_precision")[1],
                                 np.mean([average_precision_score(y, column) for column in probabilities.T]))
        with self.assertRaises(ValueError):
            m.ensemble_score(y, probabilities[:, :2])

    def test_explicit_metric_selection_and_baseline_exact_ties(self):
        table = pd.DataFrame({"recipe_id": ["r01", "r00"], "recipe_order": [1, 0],
                              "mean_inner_roc_auc": [.8, .8], "mean_inner_average_precision": [.7, .6]})
        self.assertEqual(m.select_candidate(table, "roc_auc")["recipe_id"], "r00")
        self.assertEqual(m.select_candidate(table, "average_precision")["recipe_id"], "r01")
        table.loc[0, "mean_inner_roc_auc"] += 1e-8
        self.assertEqual(m.select_candidate(table, "roc_auc")["recipe_id"], "r01")
        with self.assertRaises(ValueError):
            m.select_candidate(table, "accuracy")

    def test_outer_nn_reuse_requires_exact_selected_or_baseline_recipe(self):
        baseline, other = m.recipes()[0], m.recipes()[1]
        self.assertEqual(m.outer_nn_reuse_route(other, other), "selected")
        self.assertEqual(m.outer_nn_reuse_route(baseline, other), "fixed_nn")
        changed = {**baseline, "dropout": .4}
        self.assertIsNone(m.outer_nn_reuse_route(changed, other))

    def test_declared_grid_and_generic_protocol_are_unchanged(self):
        self.assertEqual(len(m.recipes()),36)
        self.assertEqual(m.recipes()[0],m.v1.recipes()[0])
        for metric in m.METRICS:
            protocol=m._protocol(metric)
            self.assertEqual(protocol['neural_training'],m.NN_SPECIFICATION)
            self.assertEqual(protocol['classical_grids'],m.GRIDS)
            self.assertIn(metric,protocol['primary_metric'])
            self.assertIn('stopping-fold BCE',protocol['neural_training']['epoch_selection'])

    def test_v2_namespaces_and_run_identity_cannot_escape(self):
        self.assertIn("/v2/original/", m.condition_directory(self.root, "original").as_posix())
        self.assertIn("/v2/", m.condition_directory(self.root, "dedup_v1").as_posix())
        for run_id in ("../old", "a" * 65, "", "run/extra"):
            with self.assertRaises(ValueError):
                m.condition_directory(self.root, "original", run_id)
        with self.assertRaises(ValueError):
            m.condition_directory(self.root, "original", experiment_version=1)

    def test_inner_and_outer_partitions_keep_patient_boundaries(self):
        frame = toy_frame()
        assigned = memberships(frame)
        for outer in range(5):
            outer_patients = set(frame.loc[frame.fold.eq(outer), "patient_id"])
            for inner in range(3):
                for seed_index in range(3):
                    parts = m.v1.inner_partition(frame, assigned, outer, inner, seed_index)
                    fit, stop, score = [set(parts[name].patient_id) for name in ("epoch_fit", "epoch_stop", "score")]
                    self.assertFalse((fit | stop | score) & outer_patients)
                    self.assertTrue(fit.isdisjoint(stop) and fit.isdisjoint(score) and stop.isdisjoint(score))
                    self.assertEqual(set(parts["refit"].patient_id), fit | stop)
            for seed_index in range(3):
                parts = m.v1.outer_partition(frame, outer, seed_index)
                self.assertEqual(parts["stop_group"], (outer + 1 + seed_index) % 5)
                self.assertTrue(set(parts["refit"].patient_id).isdisjoint(outer_patients))

    def test_native_phase_resume_rejects_drift_before_computation(self):
        context = self.context()
        output = self.root / context["output_root"]
        folder = output / "phases/one"
        calls = []
        def compute():
            calls.append(1)
            return {"predictions": pd.DataFrame({"score": np.array([.2,.7], dtype=np.float32)})}
        _, reused = m._phase(self.root, folder, "fixed", context, compute)
        self.assertFalse(reused)
        result, reused = m._phase(self.root, folder, "fixed", context, compute)
        self.assertTrue(reused)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["predictions"].score.dtype, np.dtype("float32"))
        for digest, changed in (("changed-source", context), ("fixed", {**context, "phase": "other"}),
                                ("fixed", {**context, "selection_metric": "average_precision"})):
            with self.assertRaises(ValueError):
                m._phase(self.root, folder, digest, changed, compute)
        self.assertEqual(len(calls), 1)

    def test_phase_hash_tampering_and_v1_paths_are_rejected(self):
        context = self.context()
        folder = self.root / context["output_root"] / "phase"
        m._phase(self.root, folder, "fixed", context, lambda: {"value": 1})
        (folder / "result.json").write_text('{"value":2}', encoding="utf-8")
        with self.assertRaises(ValueError):
            m._phase(self.root, folder, "fixed", context, lambda: self.fail("must not compute"))
        context["output_root"] = "results/metadata_nn_tuning/v1/original/old"
        with self.assertRaises(ValueError):
            m._phase(self.root, self.root / context["output_root"] / "phase", "fixed", context, lambda: {})

    def cached_parent(self):
        frame = toy_frame().iloc[:6].copy()
        directory = self.root / "results/metadata_nn_tuning/v1/original/ap"
        phase = directory / "trials/phase"
        phase.mkdir(parents=True)
        prediction = frame[m.KEYS].assign(score=np.linspace(.1, .8, len(frame)).astype(np.float32))
        prediction.to_csv(phase / "predictions.csv", index=False)
        relative = (phase / "predictions.csv").relative_to(self.root).as_posix()
        receipt = {"status": "complete", "condition": "original", "output_root": directory.relative_to(self.root).as_posix(),
                   "context": {"recipe_id": "r00"}, "output_sha256": {relative: file_sha256(phase / "predictions.csv")}}
        (phase / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        parent = {"directory": directory, "receipt": {"output_sha256": {relative: file_sha256(phase / "predictions.csv")}}}
        return parent, phase / "predictions.csv", frame

    def test_ap_cache_requires_exact_prediction_and_context_identity(self):
        parent, path, expected = self.cached_parent()
        result, reference = m._cached_prediction(self.root, parent, path, expected,
            context_fields={"recipe_id": "r00"}, nn=True)
        self.assertEqual(result.score.dtype, np.dtype("float32"))
        self.assertEqual(reference["provenance"], "external_reused")
        changed = expected.copy()
        changed.loc[changed.index[0], "patient_id"] = "wrong"
        for frame, context in ((changed, {"recipe_id": "r00"}), (expected, {"recipe_id": "r01"})):
            with self.assertRaises(ValueError):
                m._cached_prediction(self.root, parent, path, frame, context_fields=context)

    def test_missing_candidate_cache_never_silently_starts_training(self):
        parent, path, expected = self.cached_parent()
        with patch.object(m, "fit_network", side_effect=AssertionError("No fallback NN training")):
            with self.assertRaises(FileNotFoundError):
                m._cached_prediction(self.root, parent, path.parent / "missing/predictions.csv", expected)

    def test_counters_separate_external_resume_and_rescoring(self):
        fresh = m._log({}, "refit")
        resumed = m._log({}, "refit", resumed=True)
        external = m._log({}, "refit", external=True)
        scoring = m._log({}, "candidate_scoring", scoring=True)
        self.assertEqual((fresh["executed_fits"], fresh["executed_this_invocation"]), (1,1))
        self.assertEqual((resumed["executed_fits"], resumed["resumed_fits"], resumed["executed_this_invocation"]), (1,1,0))
        self.assertEqual((external["executed_fits"], external["external_reused_fits"]), (0,1))
        self.assertEqual((scoring["executed_fits"], scoring["rescoring_operations"]), (0,1))

    def test_train_only_preprocessing_and_outer_label_invariance(self):
        frame = toy_frame()
        parts = m.v1.outer_partition(frame, 0, 0)
        fitted_indices, calls = [], []
        real_preprocessor = m.neural_preprocessor
        class PrepSpy:
            def __init__(self): self.base = real_preprocessor()
            def fit(self, features):
                fitted_indices.append(list(features.index))
                self.base.fit(features)
                return self
            def transform(self, features): return self.base.transform(features)
        class Model:
            def state_dict(self): return {"stub": np.array([1])}
        def fake_fit(x, y, seed, **kwargs):
            calls.append((list(y), kwargs.get("epochs"), list(kwargs.get("y_stop", []))))
            return Model(), 1, .2, [{"epoch":1,"fit_bce":.3,"stop_bce":.2}]
        with patch.object(m,"neural_preprocessor",PrepSpy), patch.object(m,"fit_network",side_effect=fake_fit), \
             patch.object(m,"predict_network",side_effect=lambda model,x:np.full(len(x),.25,dtype=np.float32)):
            first=m.fit_nn_pair(parts,base_seed=42,outer_fold=0)
            changed=deepcopy(parts)
            changed["score"]["target"]=1-changed["score"].target
            second=m.fit_nn_pair(changed,base_seed=42,outer_fold=0)
        self.assertEqual(fitted_indices,[list(parts["epoch_fit"].index),list(parts["refit"].index)]*2)
        self.assertEqual(calls[:2],calls[2:])
        self.assertEqual(first["epoch"]["selected_epochs"],second["epoch"]["selected_epochs"])
        np.testing.assert_array_equal(first["predictions"].score,second["predictions"].score)

    def test_loader_rejects_wrong_metric_before_dataset_reads(self):
        directory=m.condition_directory(self.root,"original","synthetic")
        directory.mkdir(parents=True)
        output=directory/'oof.csv'
        output.write_text('verified\n',encoding='utf-8')
        receipt={"status":"complete","experiment_version":2,"condition":"original","selection_metric":"average_precision",
                 "output_root":directory.relative_to(self.root).as_posix(),
                 "output_sha256":{output.relative_to(self.root).as_posix():file_sha256(output)}}
        (directory/'receipt.json').write_text(json.dumps(receipt),encoding='utf-8')
        with patch.object(m,'load_development_dataset',side_effect=AssertionError('No dataset read')):
            with self.assertRaises(ValueError):
                m.load_tuning_outputs(self.root,'original',experiment_version=2,run_id='synthetic',expected_selection_metric='roc_auc')


if __name__ == "__main__":
    unittest.main()
