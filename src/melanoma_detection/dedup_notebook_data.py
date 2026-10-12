"""Original-cell templates for the development-only clean data notebooks.

Original text and plotting cells are the presentation authority. Only dataset
access, saved findings and explicitly labelled scope changes are adapted.
Executing these reports never creates folds or rewrites data files.
"""
from pathlib import Path
from textwrap import dedent

from .dedup_notebook_templates import original_cells, template_sections


def section(heading, prose, code):
    return heading, prose, dedent(code).strip()


def _source(name, number):
    return original_cells(name)[number - 1].source


def _code(value):
    return dedent(value).strip()


def _findings():
    """Derive reader-facing counts from frozen development records only."""
    from .development_dataset import load_development_dataset
    root = Path(__file__).resolve().parents[2]
    frame, _ = load_development_dataset(
        root, dataset_manifest="data/subset/processed/dedup_v1/dataset_manifest.json")
    patients = frame.groupby("patient_id").target.max()
    images = frame.groupby("patient_id").size()
    folds = frame.groupby("fold").target.agg(images="size", positives="sum", rate="mean")
    return {
        "images": len(frame), "patients": len(patients), "positives": int(frame.target.sum()),
        "positive_patients": int(patients.sum()), "image_rate": frame.target.mean(),
        "patient_rate": patients.mean(), "median": images.median(), "maximum": images.max(),
        "missing_site": int(frame.anatom_site_general_challenge.eq("Unknown").sum()),
        "fold_min": int(folds.images.min()), "fold_max": int(folds.images.max()),
        "positive_min": int(folds.positives.min()), "positive_max": int(folds.positives.max()),
        "rate_min": folds.rate.min(), "rate_max": folds.rate.max(),
    }


def _cleaning_spec(stats):
    original = "01_data_cleaning.ipynb"
    codes = {
        3: _code('''
            from melanoma_detection.metadata_validation import validate_binary_target
            from melanoma_detection.development_dataset import sha256_file
            PROJECT_ROOT = ROOT
            DATA_DIR = ROOT / "data/subset"
            TRAIN_PATH = DATA_DIR / "train_labels.csv"
            TRAIN_CLEAN_PATH = ROOT / "data/subset/processed/dedup_v1/train_folds.csv"
            UNKNOWN_LABEL = "Unknown"
            MODEL_FEATURES = ["sex", "age_approx", "anatom_site_general_challenge"]
            LEAKAGE_COLUMNS = ["diagnosis", "benign_malignant", "target"]
            ledger = context["ledger"]
            original_ids = set(ledger.image_name)
            retained_ids = set(data.image_name)
            print(f"Data directory: {DATA_DIR}")
            print(f"Frozen development output: {TRAIN_CLEAN_PATH}")
        '''),
        5: _code('''
            # Inspect IDs first, then parse only audited development rows.
            source_ids = pd.read_csv(TRAIN_PATH, usecols=["image_name"])["image_name"].tolist()
            original_source = pd.read_csv(TRAIN_PATH, dtype={"target": "string"},
                skiprows=lambda number: number > 0 and source_ids[number - 1] not in original_ids)
            assert set(original_source.image_name) == original_ids
            train_raw = original_source.loc[original_source.image_name.isin(retained_ids)].copy()
            train_raw = train_raw.sort_values("image_name").reset_index(drop=True)
            print(f"Original development rows: {len(original_source):,}")
            print(f"Retained training rows: {len(train_raw):,}; columns: {train_raw.shape[1]}")
            print("Reserved-test data and labels: not loaded")
        '''),
        7: _source(original, 7).replace(
            'print("\\nTest missingness:")\nprint(missing_profile(test_raw).to_string())',
            'print("\\nOriginal development missingness (before retention):")\n'
            'print(missing_profile(original_source).to_string())'),
        9: _source(original, 9).replace("test_clean = clean_metadata(test_raw)",
            '# Verify the frozen export rather than saving a new cleaning variant.\n'
            'frozen_clean = data.drop(columns="fold").copy()\n'
            'for column in [*MODEL_FEATURES, "image_name", "patient_id", "target"]:\n'
            '    assert np.array_equal(train_clean[column].astype(str).to_numpy(),\n'
            '                          frozen_clean[column].astype(str).to_numpy())'),
        11: _source(original, 11).replace('validate_common(test_clean, test_raw, "test")',
            '# Reserved-test verification belongs to the later final-test stage.'),
        13: _source(original, 13).replace('    ("test", test_raw, test_clean),',
            '    ("original development", original_source, clean_metadata(original_source)),')
            .replace('("train", train_raw, train_clean)', '("retained development", train_raw, train_clean)'),
        15: _code('''
            # Verify the pre-existing manifest export; do not rewrite it.
            saved_clean = pd.read_csv(TRAIN_CLEAN_PATH, dtype={"age_approx": "string"})
            assert saved_clean.image_name.tolist() == data.image_name.tolist()
            assert np.array_equal(saved_clean.target.to_numpy(), data.target.to_numpy())
            assert saved_clean.patient_id.tolist() == data.patient_id.tolist()
            assert np.array_equal(saved_clean.fold.to_numpy(), data.fold.to_numpy())
            for relative, digest in context["manifest"]["audited_input_sha256"].items():
                assert sha256_file(ROOT / relative) == digest, "Audited source changed: " + relative
            print(f"Verified existing frozen output: {TRAIN_CLEAN_PATH}")
            print(f"Manifest SHA-256: {dataset_identity['dataset_manifest_sha256']}")
            print("Source metadata and frozen data outputs were not rewritten.")
        '''),
    }
    updates = {
        1: _source(original, 1).replace("Subset data", "deduplicated Subset data")
            .replace("without removing rows or guessing missing details.",
                "without guessing missing details. Metadata cleaning preserves rows; a separate, frozen exact-image-copy audit then retains one image per identical-pixel group.")
            .replace("It reads the source CSV files and saves cleaned copies for [02_data_exploration.ipynb](02_data_exploration.ipynb).",
                "It reads only development source records and verifies the existing `dedup_v1` export for [02_clean_data_eda.ipynb](02_clean_data_eda.ipynb). It does not rewrite source files or access reserved-test records."),
        2: "## Setup\n\nThe setup uses the project root and names the development source and frozen output files in one place. The shared label check applies the original cleaning rule. The manifest and retention ledger define the population; this notebook checks their existing outputs rather than creating a new variant. The reserved-test source is not resolved or opened.",
        4: _source(original, 4).replace("cleaning should not remove rows.",
            "metadata cleaning should not remove rows. Retention is an explicitly separate image-content decision: the original development source has 6,864 rows, of which 6,706 are retained for this report."),
        6: _source(original, 6) + "\n\nThe first table describes retained development records. The second is the original development comparison before exact-copy retention; neither table includes the reserved test.",
        10: _source(original, 10) + "\n\nThe cleaned retained values must also match the frozen manifest export. Repeating the cleaning function in memory checks consistency; it does not replace the established cleaning artifact or alter a record.",
        12: _source(original, 12).replace("one field in one file", "one field in one development population")
            + "\n\nCompare original development with retained development to distinguish formatting changes from exact-copy removal. Retention removes one record with an unknown site, reducing that count from 196 to 195; it does not infer the missing location.",
        14: "## Verify the saved cleaned copies\n\nThe original cleaning stage saved copies in `data/subset/processed`. The deduplicated branch uses the separate frozen export in `data/subset/processed/dedup_v1/train_folds.csv`, with its manifest and retention ledger. The original files remain available so we can trace every retained or excluded record to its source.\n\n**Scope adaptation:** this cell verifies the already saved output and its source hashes. Executing a documentation notebook does not write cleaned data, change retention decisions or modify the original branch.",
        16: _source(original, 16).replace(
            "The checks also do not inspect image contents, identify visually similar images, or establish how well a future model will work.",
            "The metadata checks cannot inspect image contents or establish how well a future model will work. The separately documented audit above confirms exact decoded-pixel copies; it does not automatically remove visually similar but distinct photographs.")
            + "\n\nExact-copy retention reduces repeated-observation weight but preserves every patient. It does not make multiple photographs from a patient independent. The reserved test remains outside this development report, including its identifiers and labels.",
        17: ("## Summary\n\nThe frozen clean run retains "
            f"{stats['images']:,} images from {stats['patients']:,} patients, with {stats['positives']} malignant images and {stats['positive_patients']} patients who have a malignant image. The original development population had 6,864 images and 128 malignant images; the deterministic audit excludes 158 exact copies, including one malignant image. Every patient remains.\n\n"
            f"The same metadata rule represents missing model fields as `Unknown`; {stats['missing_site']} retained sites are unknown, compared with 196 before retention. The record checks and frozen export agreement pass. Source files and historical outputs remain available, and no reserved-test records are loaded.\n\n"
            "Next, [02_clean_data_eda.ipynb](02_clean_data_eda.ipynb) examines the retained training records and explains why images from one patient must stay together during evaluation. The audit details above show why the 158 exclusions are exact copies and how their provenance is preserved."),
    }
    sections = template_sections(original, code_replacements=codes, markdown_updates=updates,
        after_cells={15: _cleaning_audit_sections()},
        context_note="Metadata-format cleaning is unchanged. Exact-copy retention, frozen-output verification and development-only source access are labelled adaptations of the original preparation stage.")
    return ("01_data_cleaning.ipynb", "01 · Clean development metadata and retain exact copies once",
            "Verify the original cleaning rules on the frozen development population.", sections)


def _cleaning_audit_sections():
    return [
        section("Confirm exact copies and retain one deterministic representative",
            "**Added deduplication audit.** A file SHA-256 identifies copies with identical bytes. The full-resolution decoded RGB fingerprint also detects identical pixels stored in different files. The audit compares all decoded pixels to confirm a candidate group; perceptual resemblance alone is insufficient. For each group, the lexicographically smallest image identifier is retained. Labels, model scores and performance never choose the representative.\n\nTwo different lesions can share age, sex and site, so identical metadata is not a reason to exclude them. The conflict check below verifies that confirmed groups agree on patient, target, fold and metadata. If a group had conflicting records, it would need investigation rather than an automatic merge. All original JPEGs remain on disk.", '''
                grouped = ledger.groupby("retained_image_name")
                duplicates = grouped.filter(lambda rows: len(rows) > 1)
                excluded = ledger.loc[ledger.decision.eq("excluded")]
                group_checks = []
                original_clean = pd.read_csv(ROOT / "data/subset/processed/train_clean.csv")
                for representative, rows in duplicates.groupby("retained_image_name"):
                    assert representative == min(rows.image_name)
                    assert rows.pixel_sha256.nunique() == 1
                    assert rows.patient_id.nunique() == rows.target.nunique() == rows.fold.nunique() == 1
                    metadata_rows = original_clean.set_index("image_name").loc[rows.image_name, MODEL_FEATURES]
                    assert metadata_rows.nunique(dropna=False).eq(1).all()
                    assert rows.decision.eq("retained").sum() == 1
                    group_checks.append({"representative": representative, "copies": len(rows),
                        "patient": rows.patient_id.iloc[0], "label": rows.target.iloc[0],
                        "file_hashes": rows.source_sha256.nunique()})
                display(pd.DataFrame(group_checks).head(12))
                display(pd.DataFrame([{"confirmed_groups": len(group_checks), "excluded": len(excluded),
                    "excluded_benign": int(excluded.target.eq(0).sum()),
                    "excluded_malignant": int(excluded.target.eq(1).sum())}]))
                assert len(group_checks) == len(excluded) == 158
            '''),
        section("Inspect an identical pair and a distinct review pair",
            "The contact sheet provides a worked example of the retention rule. The top row is the first exact-copy group by image identifier. One is retained and one excluded because their full decoded RGB pixels are equal. The bottom row is the pair manually merged in the reference image notebook: the photographs differ, so both remain in this clean branch.\n\nThe displayed size can hide small differences. The exact decision therefore rests on full-resolution pixel equality, not a judgement from these thumbnails. Retaining distinct views preserves their observations while the patient folds protect evaluation separation.", '''
                from PIL import Image as PILImage
                exact = duplicates.sort_values("image_name").retained_image_name.iloc[0]
                exact_ids = sorted(duplicates.loc[duplicates.retained_image_name.eq(exact), "image_name"])
                distinct_ids = ["ISIC_2887033", "ISIC_5194874"]
                assert set(distinct_ids).issubset(retained_ids)
                figure, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
                full_pixels = {}
                for axis, image_id in zip(axes.flat, exact_ids + distinct_ids):
                    with PILImage.open(DATA_DIR / "train" / (image_id + ".jpg")) as opened:
                        pixels = np.asarray(opened.convert("RGB"))
                    full_pixels[image_id] = pixels
                    axis.imshow(pixels)
                    row = ledger.set_index("image_name").loc[image_id]
                    axis.set_title(image_id + " · " + row.decision)
                    axis.axis("off")
                figure.suptitle("Confirmed pixel copies (top); distinct photographs retained (bottom)")
                finish_figure(figure, "exact_and_distinct_examples")
                assert np.array_equal(full_pixels[exact_ids[0]], full_pixels[exact_ids[1]])
                assert not np.array_equal(full_pixels[distinct_ids[0]], full_pixels[distinct_ids[1]])
            '''),
        section("Read the manifest, partition and verification receipt",
            "The retention ledger has one decision for every original development image. Retained and excluded IDs must be disjoint and together cover all 6,864 source IDs. The manifest records artifact hashes, the retention rule, pixel-fingerprint version and preserved patient memberships. Changing one of these files should fail verification rather than silently create a new population.\n\nThe cleaning receipt records the previous complete source and decoded-pixel checks. Reading a receipt is different from repeating the expensive image audit: the report verifies its dataset identity and shows its completion flags. Before-and-after fold counts make the effect of removing copies visible without rebuilding patient groups.", '''
                retained = ledger.loc[ledger.decision.eq("retained")]
                assert set(retained.image_name).isdisjoint(set(excluded.image_name))
                assert set(retained.image_name) | set(excluded.image_name) == original_ids
                assert retained_ids == set(retained.image_name)
                assert len(data) == 6706 and int(data.target.sum()) == 127
                assert data.patient_id.nunique() == 422
                assert data.loc[data.target.eq(1), "patient_id"].nunique() == 88
                display(pd.DataFrame([{"manifest_sha256": dataset_identity["dataset_manifest_sha256"],
                    "retention_rule": context["manifest"]["retention_rule"],
                    "fingerprint_version": context["manifest"]["fingerprint_version"]}]))
                display(context["fold_counts"])
                receipt = read_json(ROOT / DIRECTORY / "cleaning/receipt.json")
                assert receipt["dataset_identity"] == dataset_identity
                display(pd.DataFrame([{key: receipt[key] for key in ["status", "sources_verified",
                    "pixels_verified", "all_outer_and_preserved_inner_assignments_verified"]}]))
                display(ledger[["width", "height"]].value_counts().rename("decoded_images").reset_index())
            '''),
        section("Preserve unusual metadata and document missing-site support",
            "Unusual values are documented rather than guessed. A patient can have several recorded age buckets, possibly reflecting different visits or source errors; dates are unavailable, so replacing those ages with one inferred value would add an unsupported cleaning rule. In the retained records, 204 patients have several ages and 41 have an age span of at least 15 years.\n\nUnknown site also has uneven patient, fold and label support. The table reports those counts so a later subgroup rate is read alongside its small malignant-patient denominator. The examples are deterministic inspection aids; they are not exclusions or new model features.", '''
                profiles = pd.read_csv(ROOT / "results/dedup_v1/audit/patient_profile_retained.csv")
                display(profiles.loc[profiles.distinct_ages.gt(1)].sort_values(
                    ["age_span", "patient_id"], ascending=[False, True]).head(12))
                display(pd.DataFrame([{"patients_with_multiple_ages": int(profiles.distinct_ages.gt(1).sum()),
                    "patients_age_span_at_least_15": int(profiles.age_span.ge(15).sum()),
                    "unknown_site_patients": int(profiles.missing_site.gt(0).sum())}]))
                show_saved_table("results/dedup_v1/audit/missing_site_by_fold_label_retained.csv")
            '''),
        section("Review image quality and the historical rotation without extra exclusions",
            "**Added audit context; experiment scope unchanged.** Near-white area, low adjacent-pixel variation and dark borders identify review candidates only. They do not prove an image is unusable. This branch adds no quality or similarity exclusions beyond confirmed exact copies.\n\nThe transform sheet retains the historical weak rotation: nearest-neighbour interpolation, black fill and no expansion can create black corners or clip a lesion. This review documents that limitation while preserving the preparation recipe for replication. Revising interpolation, cropping or quality policy would be a separately declared experiment rather than a quiet change to the deduplicated population.", '''
                audit = read_json(ROOT / "results/dedup_v1/audit/audit_summary.json")
                display(pd.DataFrame([audit["weak_rotation_review"]]))
                for filename, title in [("near_white_fraction.png", "Washed-out review candidates"),
                    ("gradient_proxy.png", "Low-detail review candidates"),
                    ("near_black_border_fraction.png", "Border-heavy review candidates"),
                    ("weak_rotation_review.png", "Historical weak rotation")]:
                    display(Markdown("### " + title))
                    display(Image(filename=str(ROOT / "results/dedup_v1/audit/figures" / filename)))
            '''),
    ]


def _eda_spec(stats):
    original = "02_data_exploration.ipynb"
    codes = {number: _source(original, number) for number in [5, 7, 9, 11, 13, 15, 17, 19, 21]}
    codes[3] = _code('''
        from melanoma_detection.metadata_validation import validate_binary_target
        from matplotlib.ticker import PercentFormatter
        PROJECT_ROOT = ROOT
        DATA_DIR = ROOT / "data/subset"
        TRAIN_CLEAN_PATH = ROOT / "data/subset/processed/dedup_v1/train_folds.csv"
        EXPECTED_COLUMNS = {"image_name", "patient_id", "sex", "age_approx",
            "anatom_site_general_challenge", "diagnosis", "benign_malignant", "target"}
        train = data.drop(columns="fold").copy()
        train["age_approx"] = train["age_approx"].astype("string")
        missing_columns = EXPECTED_COLUMNS - set(train.columns)
        if missing_columns:
            raise ValueError(f"Missing expected columns: {sorted(missing_columns)}")
        train["target"] = validate_binary_target(train["target"])
        print(f"Loaded frozen development export: {TRAIN_CLEAN_PATH}")
        print(f"Shape: {train.shape[0]:,} rows x {train.shape[1]} columns")
    ''')
    # Plot bodies, constants, labels, dimensions and composition are literal originals.
    codes[13] = codes[13].replace("plt.show()", 'finish_figure(fig, "class_balance_images_patients")')
    codes[15] = codes[15].replace("plt.show()", 'finish_figure(fig, "images_per_patient")')
    codes[19] = codes[19].replace("    plt.show()",
        '    finish_figure(fig, {"sex": "sex_image_patient_share_rates",\n'
        '        "age_approx": "age_image_patient_share_rates",\n'
        '        "anatom_site_general_challenge": "site_image_patient_share_rates"}[feature])')
    updates = {
        1: _source(original, 1).replace("Subset data", "deduplicated Subset data")
            .replace("patient groups created in [03_patient_splitting.ipynb](03_patient_splitting.ipynb)",
                "unchanged patient groups verified in [03_patient_fold_verification.ipynb](03_patient_fold_verification.ipynb)"),
        2: _source(original, 2) + "\n\nThe frozen `dedup_v1` manifest defines this report's 6,706 retained records. Age is displayed as the original five-year bucket label; the extra `fold` column is omitted from this descriptive schema because it is a membership key rather than a predictor.",
        4: _source(original, 4) + "\n\n**Deduplication adaptation:** the 158 excluded exact copies remain in the original image directory, so they appear among JPEG files absent from the retained CSV. Their presence is expected. The manifest, rather than all files found on disk, defines the retained dataset.",
        8: _source(original, 8) + "\n\nA unique image name and zero identical metadata rows do not themselves rule out copies with different identifiers. Exact-pixel duplicates were separately audited and excluded through the frozen retention ledger in stage 01.",
        12: _source(original, 12).replace("1.86%", f"{stats['image_rate']:.2%}")
            + "\n\nThe patient rate remains unchanged because no patient or last malignant observation is removed. Exact-copy removal changes image weighting and the image denominator, so the retained image rate must be calculated again.",
        16: _source(original, 16) + "\n\nDeduplication does not collapse a patient's distinct photographs or force them into one age or site category. The same original within-category counting rule remains, making the clean figures directly comparable with the historical figures.",
        22: _source(original, 22).replace("or the absence of visually similar images",
            "or the absence of visually similar but distinct images")
            + "\n\nThe exact-copy audit removes repeated identical pixels only. Related photographs and sampling differences remain, so the later patient-grouped validation and patient-bootstrap intervals are still necessary.",
        23: ("## Summary\n\nThe saved clean run describes "
            f"{stats['images']:,} retained images from {stats['patients']} patients. Malignancy accounts for {stats['image_rate']:.2%} of images; {stats['patient_rate']:.2%} of patients have at least one malignant image. Patients contribute a median of {stats['median']:.0f} images and a maximum of {stats['maximum']:.0f}. The retained record checks find no missing cells, duplicate image names or target inconsistencies.\n\n"
            "Compared with the original 6,864-image collection, 158 confirmed exact copies are excluded while all 422 patients remain. The blue/gold category panels, orange/grey class-balance figure and patient histogram use the original colours, dimensions, labels and counting rules with clean counts.\n\n"
            "Age, sex and site remain candidate predictors; diagnosis and the benign/malignant description must stay out of model inputs. Repeated distinct images and the different counting units support grouping by patient. Next, [03_patient_fold_verification.ipynb](03_patient_fold_verification.ipynb) verifies the shared historical evaluation folds after retention, rather than creating replacement groups."),
    }
    extras = [section("Document retained category support",
        "**Added deduplication context.** The original category figures show unadjusted rates. For a sparse group such as unknown site, read malignant-image and malignant-patient counts before making a subgroup claim. Removing exact copies does not resolve missing information, differing collection conditions or patients with several recorded ages. The saved audit table below documents retained support by fold and label; it does not choose predictors or exclude observations.", '''
            show_saved_table("results/dedup_v1/audit/missing_site_by_fold_label_retained.csv")
            profiles = pd.read_csv(ROOT / "results/dedup_v1/audit/patient_profile_retained.csv")
            display(pd.DataFrame([{"patients_multiple_ages": int(profiles.distinct_ages.gt(1).sum()),
                "patients_age_span_at_least_15": int(profiles.age_span.ge(15).sum())}]))
        ''')]
    sections = template_sections(original, code_replacements=codes, markdown_updates=updates,
        after_cells={21: extras}, context_note="The original EDA cell order and all five plot compositions are retained. Every table and figure is recomputed from the frozen exact-copy-retained development records.")
    return ("02_clean_data_eda.ipynb", "02 · Explore images and patients in the retained development data",
            "Use the original counting units, figure templates and descriptive caveats.", sections)


def _fold_spec(stats):
    original = "03_patient_splitting.ipynb"
    codes = {
        4: _code('''
            from matplotlib.ticker import PercentFormatter
            PROJECT_ROOT = ROOT
            PROCESSED_DIR = ROOT / "data/subset/processed/dedup_v1"
            TRAIN_FOLDS_PATH = PROCESSED_DIR / "train_folds.csv"
            IMAGE_RATE_FIGURE_PATH = FIGURE_DIR / "patient_split_image_malignancy_rate_dedup_v1.png"
            N_SPLITS = 5
            RANDOM_STATE = 42  # Original partition provenance, not a new split.
            print(f"Project root: {PROJECT_ROOT}")
            print(f"Frozen fold input: {TRAIN_FOLDS_PATH}")
        '''),
        6: _source(original, 6).replace(
            'train = pd.read_csv(TRAIN_CLEAN_PATH, dtype={"age_approx": "string"})',
            'train = data.drop(columns="fold").copy()\n'
            'train["age_approx"] = train["age_approx"].astype("string")'),
        8: _code('''
            train_folds = data.copy()
            train_folds["age_approx"] = train_folds["age_approx"].astype("string")
            original_folds = pd.read_csv(ROOT / context["manifest"]["original_source"]["path"])
            before = original_folds.set_index("image_name")
            assert set(original_folds.patient_id) == set(train_folds.patient_id)
            assert np.array_equal(before.loc[train_folds.image_name, "patient_id"].to_numpy(),
                                  train_folds.patient_id.to_numpy())
            assert np.array_equal(before.loc[train_folds.image_name, "fold"].to_numpy(),
                                  train_folds.fold.to_numpy())
            print(f"Preserved folds: {sorted(train_folds['fold'].unique().tolist())}")
            print(f"Fold dtype: {train_folds['fold'].dtype}")
            print("Changed patient memberships: 0; splitter called: False")
        '''),
        10: _source(original, 10),
        12: _source(original, 12).replace(
            'figure.savefig(IMAGE_RATE_FIGURE_PATH, dpi=320, bbox_inches="tight")\nplt.show()',
            'figure.savefig(IMAGE_RATE_FIGURE_PATH, dpi=320, bbox_inches="tight")\n'
            'figure.savefig(IMAGE_RATE_FIGURE_PATH.with_suffix(".pdf"), bbox_inches="tight")\n'
            'plt.show()\nplt.close(figure)'),
        14: _source(original, 14),
        16: _code('''
            # The original test-ID reader is intentionally deferred.
            display(pd.DataFrame([{"check": "Reserved-test patient and exact-image separation",
                "current_status": "Deferred to authorised final-test preparation",
                "reserved_test_identifiers_loaded": False, "reserved_test_labels_loaded": False}]))
            print("No current train–test overlap result is inferred from this notebook.")
        '''),
        18: _code('''
            expected_columns = [*train.columns, "fold"]
            saved_folds = pd.read_csv(TRAIN_FOLDS_PATH, dtype={"age_approx": "string"})
            assert list(saved_folds.columns) == expected_columns
            pd.testing.assert_frame_equal(saved_folds.drop(columns="fold"), train, check_dtype=False)
            assert np.array_equal(saved_folds["fold"].to_numpy(), train_folds["fold"].to_numpy())
            assert saved_folds["image_name"].equals(train["image_name"])
            assert np.array_equal(saved_folds["target"].to_numpy(), train["target"].to_numpy())
            print(f"Verified existing shared fold file: {TRAIN_FOLDS_PATH}")
            print(f"Saved shape: {saved_folds.shape[0]:,} rows x {saved_folds.shape[1]} columns")
            print("No folds were assigned or files rewritten during report execution.")
        '''),
    }
    updates = {
        1: _source(original, 1).replace("Create patient-grouped folds: Subset data", "Verify patient-grouped folds: deduplicated Subset data")
            .replace("This notebook divides the training records into five repeatable groups, keeping each patient's images together and aiming for similar malignant-image percentages.",
                "This notebook verifies the original five repeatable groups after exact-copy retention, keeping each patient's images together. The historical splitter aimed for similar malignant-image percentages; the clean branch preserves those groups rather than rebalancing them.")
            .replace("[02_data_exploration.ipynb](02_data_exploration.ipynb)", "[02_clean_data_eda.ipynb](02_clean_data_eda.ipynb)"),
        2: _source(original, 2).replace("`StratifiedGroupKFold` is the splitting tool:",
            "`StratifiedGroupKFold` was the original splitting tool:")
            + "\n\n**Deduplication adaptation:** removing 158 exact copies changes image counts within groups but does not change any patient's assignment. Rerunning the splitter after removal could move patients, mixing the effect of data cleaning with the effect of a new partition. This report therefore reads and checks existing memberships only.",
        3: _source(original, 3).replace("Use the cleaned subset training file, five folds and seed 42.",
            "Use the frozen deduplicated training file and the original five folds. Seed 42 records the historical assignment; it is not used to generate a new split.")
            .replace("names the saved table and figure", "names the existing saved table and separate clean figure"),
        7: "## Verify the five preserved folds\n\nThe next cell reads the existing `fold` value for every retained training row, labelled 0 through 4. Those numbers identify groups; they do not indicate risk or a ranking. All retained image rows and labels remain in place.\n\n**Scope adaptation:** the original cell assigned folds. This cell instead checks that image-to-patient and image-to-fold mappings match the historical source and that all original patients remain. No splitting function is called, and fold balance is not used to revise memberships.",
        9: _source(original, 9).replace("because we balance image-level targets", "because the original assignment balanced image-level targets")
            + "\n\nClean held-out support ranges from 24 to 26 malignant images after one malignant copy is removed. The before-and-after table added below makes this change explicit; it is not evidence of predictive performance.",
        11: _source(original, 11).replace("all training images", "all retained training images")
            .replace("the split keeps the malignant-image share similar", "the preserved split has a similar malignant-image share")
            + "\n\nThe figure retains the original one-panel composition, blue fill, dark blue outlines, percentage labels and dashed grey baseline. A separate supplementary patient-support figure follows instead of changing the original panel's meaning.",
        13: _source(original, 13).replace("unchanged", "unchanged relative to the retained input")
            + "\n\nFor each held-out outer fold, the additional check below compares fitting and held-out patient sets explicitly. Filtering identical copies cannot make a patient cross a fold boundary because each retained row preserves its original membership.",
        15: "### Check separation from the reserved test\n\nThe original notebook read only reserved-test patient identifiers and required zero overlap. That identifier-only check protects separation between people without using outcomes to choose folds or a model.\n\n**Scope adaptation and deferred check:** this clean documentation run has a development-only access boundary, so even the reserved-test identifier reader is deferred. The table below records that status rather than claiming a newly verified overlap count. Before final scoring, the authorised test-preparation stage should verify patient/image-ID separation and exact-image copies across development and test, then apply the predeclared test-cleaning policy without selecting changes from outcomes.",
        17: "## Verify one shared fold file\n\nThe original workflow saved training rows and assignments in `data/subset/processed/train_folds.csv`. The clean workflow uses the separately frozen `data/subset/processed/dedup_v1/train_folds.csv`, with all retained rows and their unchanged assignments. The checks below confirm that the saved table preserves the retained image order, targets and fold values.\n\nUsing this same frozen file in later notebooks makes comparisons easier to interpret: a score difference then cannot be explained by each model receiving a different outer split. **Scope adaptation:** executing this report verifies an existing file; it does not save replacement assignments or regenerate folds.",
        19: _source(original, 19).replace("25 or 26 malignant images", "24 to 26 malignant images")
            + "\n\nExact-copy removal retains all patients and makes no claim that patient identifiers are clinically correct. The current notebook verifies only development memberships; reserved-test separation must be verified at final-test preparation, and no test accuracy result is available here.",
        20: ("## Summary\n\nAll "
            f"{stats['images']:,} retained images from {stats['patients']} patients keep one of the original five folds. Each fold has {stats['fold_min']:,}–{stats['fold_max']:,} images and {stats['positive_min']}–{stats['positive_max']} malignant images, giving malignant shares of {stats['rate_min']:.2%}–{stats['rate_max']:.2%}. No development patient spans folds, no row is unassigned, and no retained patient changes their historical outer membership.\n\n"
            "The preserved image inner memberships are checked within each outer-training pool; metadata tuning and stopping memberships are documented separately. Reserved-test identifiers and labels remain unopened, so the clean train–test separation check is explicitly deferred.\n\n"
            "Keep this shared split for later comparisons. Start with [04_metadata_baselines.ipynb](04_metadata_baselines.ipynb), then continue to the broader [05_metadata_models.ipynb](05_metadata_models.ipynb). Freeze the final procedure before the later single reserved-test evaluation."),
    }
    sections = template_sections(original, code_replacements=codes, markdown_updates=updates,
        after_cells={12: _fold_support_sections(), 14: _inner_membership_sections()},
        context_note="Original fold-table and single-panel plot templates are preserved. Creating assignments and reading reserved-test IDs are explicitly replaced by verification of frozen development memberships and a deferred-test status.")
    return ("03_patient_fold_verification.ipynb", "03 · Verify the preserved patient folds",
            "Verify the original outer groups and training-only inner memberships.", sections)


def _fold_support_sections():
    return [section("Compare before-and-after image and patient support",
        "**Added retention accounting.** The table aligns original and retained counts for each unchanged outer fold. All patients remain while image counts fall according to the copy ledger. Similar image prevalence does not require identical patient-level prevalence; a patient is positive here if any of their images is malignant.\n\nThe supplementary patient chart uses the same blue bars, dark outlines, percentage labels and grey dashed overall reference as the original image chart. It is a separate figure because its denominator is patients, not images. Neither table nor plot chooses or revises the folds.", '''
            display(context["fold_counts"])
            patient_rates = fold_summary.patient_positivity_rate
            overall_patient_rate = train_folds.groupby("patient_id").target.max().mean()
            figure, axis = plt.subplots(figsize=(9, 5.4))
            bars = axis.bar(patient_rates.index.astype(str), patient_rates.values,
                color="#2563EB", edgecolor="#1E3A8A", linewidth=0.8, width=0.68)
            percentage_labels = axis.bar_label(bars,
                labels=[f"{rate:.2%}" for rate in patient_rates], padding=4, fontsize=10)
            axis.axhline(overall_patient_rate, color="#475569", linestyle="--", linewidth=1.3,
                label=f"Overall patient positivity rate: {overall_patient_rate:.2%}")
            axis.set_title("Patient positivity rate by patient-grouped fold", fontsize=15, pad=13)
            axis.set_xlabel("Outer fold")
            axis.set_ylabel("Patients with malignant image / all patients")
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.set_ylim(0, patient_rates.max() * 1.25)
            axis.grid(axis="y", alpha=0.20)
            axis.legend(frameon=False, loc="lower right")
            figure.tight_layout()
            # Give only labels crossed by the reference line a white backing.
            figure.canvas.draw()
            reference_y = axis.transData.transform((0, overall_patient_rate))[1]
            renderer = figure.canvas.get_renderer()
            for annotation in percentage_labels:
                bounds = annotation.get_window_extent(renderer)
                if bounds.y0 <= reference_y <= bounds.y1:
                    annotation.set_bbox({"facecolor": "white", "edgecolor": "none",
                                         "boxstyle": "square,pad=0.08"})
            finish_figure(figure, "patient_fold_balance")
        ''')]


def _inner_membership_sections():
    return [
        section("Verify fitting and held-out development patient separation",
            "For outer fold k, fitting uses patients in the other four groups and evaluation uses patients in k. Each retained image obtains one held-out prediction. Patient ID is only a grouping key, never a model input. Both patient and image intersections must be empty, and both partitions must contain the two target classes. These development-only assertions are independent of a balanced-looking figure.", '''
                checks = []
                for fold in range(5):
                    fitting = data.loc[data.fold.ne(fold)]
                    held = data.loc[data.fold.eq(fold)]
                    assert set(fitting.patient_id).isdisjoint(set(held.patient_id))
                    assert set(fitting.image_name).isdisjoint(set(held.image_name))
                    assert set(fitting.target.unique()) == set(held.target.unique()) == {0, 1}
                    checks.append({"outer_fold": fold, "fitting_images": len(fitting),
                        "held_images": len(held), "fitting_patients": fitting.patient_id.nunique(),
                        "held_patients": held.patient_id.nunique(), "patient_overlap": 0})
                display(pd.DataFrame(checks))
            '''),
        section("Inspect the preserved image inner memberships",
            "**Added nested-evaluation documentation.** B0 recipe and duration choices use three inner folds inside each outer-training pool. Load their frozen memberships rather than rerunning the splitter after removal. Inner validation helps choose candidate settings; the held-out outer fold evaluates the complete selected procedure. It must not choose the recipe, stopping duration, calibration or cutoff.\n\nThe support table and supplementary blue heatmap count malignant patients in each inner validation group. Small counts explain why candidate rankings can vary. A high support cell is not a model result. These are preserved image-selection memberships, not a promise that every model's training-only internal partition is identical.", '''
                checks = []
                for outer in range(5):
                    members = load_preserved_inner_memberships(ROOT, dataset_identity, outer)
                    assert members.groupby("patient_id").inner_fold.nunique().eq(1).all()
                    assert set(members.image_name) == set(data.loc[data.fold.ne(outer), "image_name"])
                    for inner, held in members.groupby("inner_fold"):
                        fitting = members.loc[members.inner_fold.ne(inner)]
                        assert set(fitting.patient_id).isdisjoint(set(held.patient_id))
                        checks.append({"outer_fold": outer, "inner_fold": int(inner),
                            "held_images": len(held), "held_patients": held.patient_id.nunique(),
                            "held_malignant_images": int(held.target.sum()),
                            "held_malignant_patients": held.loc[held.target.eq(1), "patient_id"].nunique(),
                            "patient_overlap": 0})
                display(pd.DataFrame(checks))
                support = pd.DataFrame(checks).pivot(index="outer_fold", columns="inner_fold",
                    values="held_malignant_patients")
                figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
                support_image = axis.imshow(support, cmap="Blues", aspect="auto")
                for row in range(5):
                    for column in range(3):
                        value = support.iloc[row, column]
                        text_colour = "white" if support_image.norm(value) >= 0.5 else "#172033"
                        axis.text(column, row, str(value), ha="center", va="center", color=text_colour)
                axis.set(title="Malignant-patient support in preserved image inner validation",
                    xlabel="Inner fold", ylabel="Outer-training pool", xticks=range(3), yticks=range(5))
                finish_figure(figure, "inner_malignant_patient_support")
            '''),
        section("Explain metadata tuning and stopping membership provenance",
            "The historical metadata tuning partition uses seed 42 + outer fold, whereas image selection uses the saved three-way memberships described above. Because a historical metadata membership receipt was not saved, the clean runner reconstructed those memberships from the original row order and then filtered retained IDs. This provenance is recorded explicitly rather than claiming a previously saved manifest existed.\n\nThe metadata NN stopping fit uses a designated original outer group within the training pool. These internal groups can differ across modelling procedures while all comparators still share the same outer-held-out records. The receipt, inner audit and saved epochs below document those boundaries; this report neither reconstructs the split again nor starts fitting.", '''
                membership_receipt = read_json(ROOT / METADATA_SUITE / "memberships_receipt.json")
                display(pd.DataFrame([membership_receipt]))
                show_saved_table(METADATA_SUITE + "/inner_audit.csv", 15)
                show_saved_table("results/dedup_v1/metadata/epochs.csv", 15)
                display(Markdown("**Verified contract:** one retained development population, unchanged outer patient groups, training-only inner and stopping memberships, and zero fitting/validation patient overlap. No splitter or reserved-test reader was called."))
            '''),
    ]


def data_specs():
    from .dedup_notebooks import SETUP
    stats = _findings()
    specifications = [_cleaning_spec(stats), _eda_spec(stats), _fold_spec(stats)]
    result = []
    for filename, title, purpose, sections in specifications:
        cells = []
        for kind, prose, code in sections:
            if kind in {"__markdown__", "__code__"}:
                cells.append((kind, prose, code))
            else:
                cells.append(("__markdown__", f"## {kind}\n\n{prose}", ""))
                if code.strip():
                    cells.append(("__code__", "", code))
        first_code = next(index for index, cell in enumerate(cells) if cell[0] == "__code__")
        kind, prose, code = cells[first_code]
        cells[first_code] = (kind, prose, f"NOTEBOOK_STAGE = {Path(filename).stem!r}\n" + SETUP + "\n" + code)
        result.append((filename, title, purpose, cells))
    return result
