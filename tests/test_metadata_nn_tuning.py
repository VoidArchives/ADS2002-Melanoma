"""Small synthetic checks of the approved nested metadata-NN procedure.

These tests never run the production grid. Short helper fits and mocked phase
calls verify isolation and resume behaviour on artificial development rows.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melanoma_detection import metadata_nn_tuning as tuning
from melanoma_detection.dedup_metadata import NN_SPECIFICATION
from melanoma_detection.metadata_model import file_sha256


def development_fixture():
    """Every outer/inner/epoch partition has patients from both classes."""
    rows = []
    for fold in range(5):
        for patient in range(6):
            for image in range(2):
                rows.append({"image_name": f"image_{fold}_{patient}_{image}",
                    "patient_id": f"patient_{fold}_{patient}", "fold": fold,
                    "target": patient % 2, "age_approx": 20 + patient * 5 + fold,
                    "sex": "female" if patient % 2 else "male",
                    "anatom_site_general_challenge": "torso" if patient < 3 else "head/neck",
                    "diagnosis": "outcome proxy excluded from features"})
    frame = pd.DataFrame(rows)
    memberships = []
    for outer in range(5):
        pool = frame.loc[frame.fold.ne(outer), ["image_name", "patient_id", "fold", "target"]].copy()
        pool["outer_fold"] = outer
        pool["inner_fold"] = pool.patient_id.map(lambda value: int(value.rsplit("_", 1)[1]) % 3)
        memberships.append(pool)
    return frame, pd.concat(memberships, ignore_index=True)


def short_spec():
    spec = deepcopy(NN_SPECIFICATION)
    spec.update(max_epochs=2, min_epochs=1, patience=1, hidden_units=8)
    return spec


def temporary_run_inputs(root):
    """Provide artificial source/config fingerprints and development inputs."""
    frame, memberships = development_fixture()
    for relative in tuning.SOURCE_FILES:
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("synthetic source fingerprint\n", encoding="utf-8")
    config = yaml.safe_load((ROOT / "configs/metadata_nn_tuning.yaml").read_text(encoding="utf-8"))
    config["run_id"] = "synthetic_run"
    config_path = root / "configs/metadata_nn_tuning.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    inputs = {"frames": {"original": frame}, "memberships": {"original": memberships},
        "categorical": {"original": frame}, "identities": {"original": {"source": "synthetic-development"}},
        "reuse_identity": {"metadata_original_memberships_sha256": "frozen-memberships"}}
    return config_path, config, inputs


class MetadataNNTuningProtocolTests(unittest.TestCase):
    def test_grid_has_36_distinct_recipes_with_baseline_then_ascending_remainder(self):
        recipes = tuning.recipes()
        triples = [(row["hidden_units"], row["learning_rate"], row["dropout"]) for row in recipes]
        expected = {(width, rate, dropout) for width in [8, 16, 32, 64]
                    for rate in [.0003, .001, .003] for dropout in [0., .2, .4]}
        self.assertEqual(len(recipes), 36)
        self.assertEqual(set(triples), expected)
        self.assertEqual(triples[0], (32, .001, .2))
        self.assertEqual(triples[1:], sorted(expected - {(32, .001, .2)}))
        self.assertEqual(len({row["recipe_id"] for row in recipes}), 36)
        self.assertEqual([row["recipe_order"] for row in recipes], list(range(36)))

    def test_exact_ties_keep_baseline_but_any_real_gain_wins(self):
        candidates = pd.DataFrame(tuning.recipes()).iloc[:4].copy()
        candidates["mean_inner_average_precision"] = [.2, .2, .1, .2]
        selected = tuning.select_recipe(candidates.sample(frac=1, random_state=8))
        self.assertEqual(selected["recipe_order"], 0)
        candidates.loc[candidates.recipe_order.eq(3), "mean_inner_average_precision"] = .2 + 1e-8
        selected = tuning.select_recipe(candidates)
        self.assertEqual(selected["recipe_order"], 3)

    def test_three_seed_probabilities_are_averaged_before_average_precision(self):
        target = np.array([1, 0, 1, 0])
        probabilities = np.array([[.9, .1, .1], [.8, .2, .9], [.2, .8, .1], [.1, .9, .2]])
        expected_mean = probabilities.mean(axis=1)
        mean_probability, score = tuning.ensemble_average_precision(target, probabilities)
        np.testing.assert_allclose(mean_probability, expected_mean, rtol=0, atol=0)
        self.assertAlmostEqual(score, average_precision_score(target, expected_mean))
        mean_seed_ap = np.mean([average_precision_score(target, probabilities[:, seed]) for seed in range(3)])
        self.assertNotAlmostEqual(score, mean_seed_ap)
        with self.assertRaises(ValueError):
            tuning.ensemble_average_precision(target, probabilities[:, :2])

    def test_inner_stop_epoch_fit_refit_and_score_match_prescribed_patients(self):
        frame, memberships = development_fixture()
        for outer in range(5):
            outer_patients = set(frame.loc[frame.fold.eq(outer), "patient_id"])
            for inner in range(3):
                remaining = [(inner + 1) % 3, (inner + 2) % 3]
                for seed_index in range(3):
                    parts = tuning.inner_partition(frame, memberships, outer, inner, seed_index)
                    selected = memberships.loc[memberships.outer_fold.eq(outer)]
                    expected_stop = set(selected.loc[selected.inner_fold.eq(remaining[seed_index % 2]), "patient_id"])
                    expected_fit = set(selected.loc[selected.inner_fold.eq(remaining[1 - seed_index % 2]), "patient_id"])
                    expected_score = set(selected.loc[selected.inner_fold.eq(inner), "patient_id"])
                    self.assertEqual(set(parts["epoch_fit"].patient_id), expected_fit)
                    self.assertEqual(set(parts["epoch_stop"].patient_id), expected_stop)
                    self.assertEqual(set(parts["refit"].patient_id), expected_fit | expected_stop)
                    self.assertEqual(set(parts["score"].patient_id), expected_score)
                    self.assertTrue(expected_fit.isdisjoint(expected_stop))
                    self.assertTrue((expected_fit | expected_stop).isdisjoint(expected_score | outer_patients))
                    self.assertTrue(expected_score.isdisjoint(outer_patients))

    def test_outer_stopping_uses_only_remaining_original_groups(self):
        frame, _ = development_fixture()
        for outer in range(5):
            for seed_index in range(3):
                parts = tuning.outer_partition(frame, outer, seed_index)
                stop_fold = (outer + 1 + seed_index) % 5
                self.assertEqual(set(parts["epoch_stop"].fold), {stop_fold})
                self.assertEqual(set(parts["epoch_fit"].fold), set(range(5)) - {outer, stop_fold})
                self.assertEqual(set(parts["refit"].fold), set(range(5)) - {outer})
                self.assertEqual(set(parts["score"].fold), {outer})
                self.assertTrue(set(parts["refit"].patient_id).isdisjoint(parts["score"].patient_id))

    def test_inner_membership_changed_labels_or_cross_patient_assignment_is_rejected(self):
        frame, memberships = development_fixture()
        bad = memberships.copy()
        bad.loc[bad.index[0], "target"] = 1 - bad.loc[bad.index[0], "target"]
        with self.assertRaises(ValueError):
            tuning.inner_partition(frame, bad, 0, 0, 0)
        bad = memberships.copy()
        bad.loc[bad.index[0], "inner_fold"] = (bad.loc[bad.index[0], "inner_fold"] + 1) % 3
        with self.assertRaises(ValueError):
            tuning.inner_partition(frame, bad, 0, 0, 0)

    def test_input_loader_requires_literal_metadata_membership_filter_and_never_opens_test_inputs(self):
        small, _ = development_fixture()
        original = pd.concat([small] * 115, ignore_index=True).iloc[:6864].copy()
        original["image_name"] = [f"synthetic_{index:05d}" for index in range(len(original))]
        clean = original.iloc[:6706].copy()
        tables = []
        for outer in range(5):
            pool = original.loc[original.fold.ne(outer), tuning.KEYS].copy()
            pool["outer_fold"] = outer
            pool["inner_fold"] = pool.patient_id.map(lambda value: int(value.rsplit("_", 1)[1]) % 3)
            tables.append(pool)
        original_memberships = pd.concat(tables, ignore_index=True)
        retained_memberships = original_memberships.loc[original_memberships.image_name.isin(clean.image_name)].copy()
        identities = {"original": {"variant": "original"}, "dedup_v1": {"variant": "dedup_v1"}}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            _, config, _ = temporary_run_inputs(root)
            suite_path = root / config["metadata_suite_receipt"]
            suite_path.parent.mkdir(parents=True)
            suite_path.write_text(json.dumps({"dataset_identity": identities["dedup_v1"],
                                              "source_sha256": {}}), encoding="utf-8")
            original_memberships.to_csv(suite_path.parent / "inner_memberships_original.csv", index=False)
            clean_path = suite_path.parent / "inner_memberships_retained.csv"
            retained_memberships.to_csv(clean_path, index=False)
            predictions = clean[tuning.KEYS].copy()
            for column in tuning.FIXED_COLUMNS:
                predictions[column] = .5
            predictions.to_csv(suite_path.parent / "combined_oof.csv", index=False)
            comparator_path = root / config["clean_comparator_receipt"]
            comparator_path.parent.mkdir(parents=True)
            fixed_path = comparator_path.parent / "fixed_nn.csv"
            predictions.to_csv(fixed_path, index=False)
            comparator_path.write_text(json.dumps({"dataset_identity": identities["dedup_v1"],
                "specification": {"neural_network": NN_SPECIFICATION}, "source_sha256": {},
                "prediction_path": fixed_path.relative_to(root).as_posix()}), encoding="utf-8")
            canonical = root / tuning.CANONICAL_CSV
            canonical.parent.mkdir(parents=True)
            original.to_csv(canonical, index=False)
            reads = []
            actual_read = pd.read_csv

            def tracked_read(path, *args, **kwargs):
                relative = Path(path).relative_to(root).as_posix()
                self.assertNotIn("test", relative.lower())
                self.assertNotIn("image_inner", relative.lower())
                reads.append(relative)
                return actual_read(path, *args, **kwargs)

            def development_only(project_root, manifest=None):
                self.assertEqual(project_root, root)
                self.assertIn(manifest, [None, config["dataset_manifest"]])
                condition = "original" if manifest is None else "dedup_v1"
                return (original if condition == "original" else clean).copy(), identities[condition]

            with patch.object(tuning, "load_development_dataset", side_effect=development_only), \
                 patch.object(tuning, "verify_suite_receipt"), patch.object(tuning, "verify_saved_receipt"), \
                 patch.object(tuning, "fit_network", side_effect=AssertionError("input verification started fitting")), \
                 patch.object(pd, "read_csv", side_effect=tracked_read):
                inputs = tuning._load_inputs(root, config)
                self.assertEqual(len(inputs["frames"]["original"]), 6864)
                self.assertEqual(len(inputs["frames"]["dedup_v1"]), 6706)
                pd.testing.assert_frame_equal(inputs["memberships"]["dedup_v1"].reset_index(drop=True),
                    retained_memberships.reset_index(drop=True), check_dtype=False)
                # This is still a valid patient partition, but it is a different
                # metadata experiment and cannot replace the saved assignments.
                changed = retained_memberships.copy()
                changed["inner_fold"] = (changed.inner_fold + 1) % 3
                changed.to_csv(clean_path, index=False)
                with self.assertRaisesRegex(ValueError, "filtered by retained IDs"):
                    tuning._load_inputs(root, config)
            self.assertIn(tuning.CANONICAL_CSV, reads)

    def test_preprocessors_are_fresh_and_fit_only_epoch_fit_then_refit(self):
        frame, memberships = development_fixture()
        parts = tuning.inner_partition(frame, memberships, 0, 0, 0)
        created = []
        factory = tuning.neural_preprocessor

        def recording_factory():
            processor = factory()
            actual_fit = processor.fit
            record = {"processor": processor}

            def recording_fit(features, *args, **kwargs):
                record["features"] = features.copy()
                return actual_fit(features, *args, **kwargs)

            processor.fit = recording_fit
            created.append(record)
            return processor

        with patch.object(tuning, "neural_preprocessor", side_effect=recording_factory):
            tuning.fit_nn_pair(parts, base_seed=42, outer_fold=0, spec=short_spec())
        self.assertEqual(len(created), 2)
        self.assertIsNot(created[0]["processor"], created[1]["processor"])
        for record, expected in zip(created, [parts["epoch_fit"], parts["refit"]]):
            self.assertEqual(set(record["features"].index), set(expected.index))
            self.assertEqual(set(record["features"].columns),
                {"age_approx", "sex", "anatom_site_general_challenge"})
            self.assertAlmostEqual(record["processor"].named_transformers_["age"].named_steps["scale"].mean_[0],
                                   expected.age_approx.mean())

    def test_outer_labels_cannot_change_epoch_or_probabilities(self):
        frame, _ = development_fixture()
        original_parts = tuning.outer_partition(frame, 0, 0)
        original = tuning.fit_nn_pair(original_parts, base_seed=42, outer_fold=0, spec=short_spec())
        changed = frame.copy()
        changed.loc[changed.fold.eq(0), "target"] = 1 - changed.loc[changed.fold.eq(0), "target"]
        changed_parts = tuning.outer_partition(changed, 0, 0)
        altered = tuning.fit_nn_pair(changed_parts, base_seed=42, outer_fold=0, spec=short_spec())
        np.testing.assert_array_equal(original["predictions"].score, altered["predictions"].score)
        self.assertEqual(original["epoch"], altered["epoch"])

    def test_outer_labels_cannot_change_inner_candidate_scores_or_selected_setting(self):
        frame, memberships = development_fixture()
        changed = frame.copy()
        changed.loc[changed.fold.eq(0), "target"] = 1 - changed.loc[changed.fold.eq(0), "target"]

        def select_on_inner_patients(data):
            records = []
            for recipe in tuning.recipes()[:2]:
                spec = short_spec()
                spec.update({key: recipe[key] for key in ("hidden_units", "learning_rate", "dropout")})
                scores = []
                for inner in range(3):
                    predictions = []
                    for index, seed in enumerate(tuning.BASE_SEEDS):
                        parts = tuning.inner_partition(data, memberships, 0, inner, index)
                        fitted = tuning.fit_nn_pair(parts, base_seed=seed, outer_fold=0, spec=spec)
                        predictions.append(fitted["predictions"].score.to_numpy())
                    _, ap = tuning.ensemble_average_precision(parts["score"].target,
                                                               np.column_stack(predictions))
                    scores.append(ap)
                records.append({**recipe, "mean_inner_average_precision": float(np.mean(scores))})
            table = pd.DataFrame(records)
            return table, tuning.select_recipe(table)

        original_scores, original_setting = select_on_inner_patients(frame)
        changed_scores, changed_setting = select_on_inner_patients(changed)
        pd.testing.assert_frame_equal(original_scores, changed_scores, check_exact=True)
        self.assertEqual(original_setting, changed_setting)

    def test_prediction_coverage_requires_each_id_once_with_matching_patient_fold_and_target(self):
        frame, _ = development_fixture()
        predictions = frame[tuning.KEYS].copy()
        predictions["p_nn_tuned"] = .5
        tuning.verify_prediction_coverage(frame, predictions)
        tuning.verify_prediction_coverage(frame, predictions.loc[predictions.fold.eq(2)], fold=2)
        malformed = {"missing": predictions.iloc[:-1],
            "duplicate": pd.concat([predictions.iloc[:-1], predictions.iloc[[0]]], ignore_index=True)}
        for key, value in [("image_name", "foreign_image"), ("patient_id", "foreign_patient"),
                           ("fold", 4), ("target", 1)]:
            changed = predictions.copy()
            changed.loc[changed.index[0], key] = value
            malformed[key] = changed
        invalid_probability = predictions.copy()
        invalid_probability.loc[invalid_probability.index[0], "p_nn_tuned"] = np.nan
        malformed["probability"] = invalid_probability
        for name, changed in malformed.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                tuning.verify_prediction_coverage(frame, changed)

    def test_stopping_checkpoint_resumes_only_refit_then_reuses_both_phases(self):
        frame, memberships = development_fixture()
        parts = tuning.inner_partition(frame, memberships, 0, 0, 0)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            output = root / "results/metadata_nn_tuning/v1/original/synthetic/trial"
            context = {"condition": "original", "output_root": "results/metadata_nn_tuning/v1/original/synthetic",
                "stage": "nn_inner", "inner_fold": 0, "recipe_id": "r00"}
            arguments = {"base_seed": 42, "outer_fold": 0, "spec": short_spec(), "context": context,
                "project_root": root, "output_dir": output, "compatibility_sha256": "test-a"}
            stop_event = Event()
            actual_fit = tuning.fit_network

            def stop_after_fit(*args, **kwargs):
                result = actual_fit(*args, **kwargs)
                stop_event.set()
                return result

            with patch.object(tuning, "fit_network", side_effect=stop_after_fit) as fit:
                with self.assertRaises(InterruptedError):
                    tuning.fit_nn_pair(parts, **arguments, stop_event=stop_event)
                self.assertEqual(fit.call_count, 1)
            self.assertTrue((output / "stopping/receipt.json").is_file())
            self.assertFalse((output / "refit/receipt.json").exists())
            saved = json.loads((output / "stopping/result.json").read_text(encoding="utf-8"))
            stop_event.clear()
            with patch.object(tuning, "fit_network", wraps=actual_fit) as fit:
                resumed = tuning.fit_nn_pair(parts, **arguments, stop_event=stop_event)
                self.assertEqual(fit.call_count, 1)
                self.assertEqual(fit.call_args.kwargs["epochs"], saved["selected_epochs"])
                self.assertNotIn("x_stop", fit.call_args.kwargs)
            self.assertEqual([row["executed_this_invocation"] for row in resumed["fit_log"]], [0, 1])
            with patch.object(tuning, "fit_network", side_effect=AssertionError("a completed phase was fitted again")):
                complete = tuning.fit_nn_pair(parts, **arguments)
                pd.testing.assert_frame_equal(resumed["predictions"].reset_index(drop=True),
                                              complete["predictions"].reset_index(drop=True),
                                              check_dtype=False, rtol=1e-7, atol=1e-8)
                self.assertEqual([row["executed_this_invocation"] for row in complete["fit_log"]], [0, 0])
                with self.assertRaises(ValueError):
                    tuning.fit_nn_pair(parts, **{**arguments, "compatibility_sha256": "changed-run-inputs"})

    def test_partial_run_rejects_changed_source_configuration_and_metadata_membership_fingerprints(self):
        for changed in ["source", "config", "memberships"]:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                root = Path(directory)
                config_path, config, inputs = temporary_run_inputs(root)
                stop_event = Event()
                stop_event.set()
                with patch.object(tuning, "fit_network", side_effect=AssertionError("cancelled run started fitting")):
                    with self.assertRaises(InterruptedError):
                        tuning._run_condition(root, config_path, config, inputs, "original", None, stop_event)
                    output = tuning.condition_directory(root, "original", config["run_id"])
                    self.assertTrue((output / "session.json").is_file())
                    self.assertFalse((output / "receipt.json").exists())
                    if changed == "source":
                        (root / tuning.SOURCE_FILES[0]).write_text("changed source\n", encoding="utf-8")
                    elif changed == "config":
                        with config_path.open("a", encoding="utf-8") as handle:
                            handle.write("# changed frozen configuration bytes\n")
                    else:
                        inputs["reuse_identity"]["metadata_original_memberships_sha256"] = "changed-memberships"
                    with self.assertRaisesRegex(ValueError, "inputs changed"):
                        tuning._run_condition(root, config_path, config, inputs, "original", None, stop_event)

    def test_public_runner_rejects_unapproved_grid_and_test_scope_before_loading_inputs(self):
        for field, value in [("threads", 2), ("dropouts", [0., .2]), ("hidden_units", [32]),
                             ("conditions", ["original", "test"]), ("test_path", "reserved_test.csv")]:
            with self.subTest(field=field), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                root = Path(directory)
                config_path, config, _ = temporary_run_inputs(root)
                config[field] = value
                config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with patch.object(tuning, "_load_inputs", side_effect=AssertionError("invalid config reached data")):
                    with self.assertRaises(ValueError):
                        tuning.run_tuning(root, config_path)

    def test_only_new_condition_namespaces_accept_simple_run_ids(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            for condition, relative in [("original", "results/metadata_nn_tuning/v1/original/check"),
                                        ("dedup_v1", "results/dedup_v1/metadata_nn_tuning/v1/check")]:
                self.assertEqual(tuning.condition_directory(root, condition, "check"), root / relative)
            for run_id in ["../metadata_d1", "nested/run", "..", "", "a" * 65]:
                with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                    tuning.condition_directory(root, "original", run_id)
            with self.assertRaises(ValueError):
                tuning.condition_directory(root, "test", "check")

    def test_receipt_rejects_identity_hash_change_output_change_and_historical_namespace(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            output_root = root / "results/metadata_nn_tuning/v1/original"
            output_root.mkdir(parents=True)
            output = output_root / "trial_predictions.csv"
            output.write_text("unchanged", encoding="utf-8")
            receipt = {"status": "complete", "compatibility_sha256": "frozen",
                "condition": "original", "output_root": output_root.relative_to(root).as_posix(),
                "output_sha256": {output.relative_to(root).as_posix(): file_sha256(output)}}
            tuning.verify_receipt(root, receipt, compatibility_sha256="frozen", output_root=output_root)
            for changed_identity in ["changed_config", "changed_source", "changed_memberships"]:
                with self.subTest(changed_identity=changed_identity), self.assertRaises(ValueError):
                    tuning.verify_receipt(root, receipt, compatibility_sha256=changed_identity, output_root=output_root)
            output.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                tuning.verify_receipt(root, receipt, output_root=output_root)
            historic = root / "results/metadata_d1/predictions.csv"
            historic.parent.mkdir(parents=True)
            historic.write_text("preserved", encoding="utf-8")
            bad = {**receipt, "output_sha256": {historic.relative_to(root).as_posix(): file_sha256(historic)}}
            with self.assertRaises(ValueError):
                tuning.verify_receipt(root, bad, output_root=output_root)


if __name__ == "__main__":
    unittest.main()
